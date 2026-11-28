"""A same-chart phrase teacher and an audio-only prior for layout variants."""
from dataclasses import dataclass, replace
import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F

from .intent_policy import POLICY

AUX_WEIGHT = .05
VARIANTS = 4
WIDTH = 32
SONG_SECTIONS = 4
LOOKAHEAD_BEATS = 4.
TEACHER_FRACTION = .5


@dataclass(frozen=True)
class PhraseTeacher:
    audio: torch.Tensor
    baseline: torch.Tensor
    support: torch.Tensor
    steps: torch.Tensor
    descriptors: torch.Tensor
    probabilities: torch.Tensor


@dataclass(frozen=True)
class IntentPlan:
    frames: torch.Tensor
    beats: torch.Tensor
    counts: torch.Tensor
    variant: torch.Tensor
    selected: torch.Tensor
    selection_evidence: torch.Tensor
    probabilities: torch.Tensor
    strength: float
    available: torch.Tensor
    diagnostics: dict | None
    teacher: PhraseTeacher | None = None
    teacher_used: torch.Tensor | None = None
    execution_max: torch.Tensor | None = None

    def record_execution(self, residual, strength):
        if self.execution_max is not None:
            with torch.no_grad():
                observed = torch.stack((strength.float().amax(), residual.float().abs().amax()))
                self.execution_max.copy_(torch.maximum(self.execution_max, observed))

    def execution_diagnostics(self):
        if self.execution_max is None:
            return {}
        strength, shift = self.execution_max.detach().cpu().tolist()
        return dict(max_executed_strength=strength, max_logit_shift=shift)


def _layout_token(token):
    return token.startswith(('N:', 'P:', 'S:')) or token in ('[END_NOTE]', '[END_EVENT]')


def _stable_draw(probabilities, song_keys, seed):
    rows = []
    for probability, key in zip(probabilities.detach().float().cpu(), song_keys, strict=True):
        digest = hashlib.blake2b(f'{seed}:{key}'.encode(), digest_size=8).digest()
        rng = torch.Generator(device='cpu').manual_seed(int.from_bytes(digest, 'little'))
        rows.append(torch.multinomial(probability, 1, generator=rng))
    return torch.cat(rows).to(probabilities.device)


class PhraseIntent(nn.Module):
    def __init__(self, codec, model_width):
        super().__init__()
        from .intent_data import AUDIO_DIM, FIELDS, MAX_PHRASES, STEPS, TARGET_DIM

        self.song_projection = nn.Sequential(nn.LayerNorm(model_width), nn.Linear(model_width, WIDTH), nn.SiLU())
        self.prior = nn.Sequential(nn.Linear(WIDTH * SONG_SECTIONS + 1, WIDTH), nn.SiLU(), nn.Linear(WIDTH, VARIANTS))
        self.current_future = nn.Sequential(nn.Linear(WIDTH * 2, WIDTH), nn.SiLU())
        self.codebook = nn.Parameter(torch.empty(VARIANTS, WIDTH))
        nn.init.normal_(self.codebook, std=.25)
        self.phrase_audio = nn.Sequential(nn.Linear(AUDIO_DIM, WIDTH), nn.SiLU(), nn.Linear(WIDTH, WIDTH))
        self.phrase_position = nn.Parameter(torch.empty(STEPS, WIDTH))
        nn.init.normal_(self.phrase_position, std=.1)
        self.audio_only = nn.Linear(WIDTH, TARGET_DIM)
        nn.init.zeros_(self.audio_only.weight)
        nn.init.zeros_(self.audio_only.bias)
        self.support_step = nn.Sequential(nn.Linear(WIDTH + TARGET_DIM, WIDTH), nn.SiLU())
        self.support_phrase = nn.Sequential(nn.Linear(STEPS * WIDTH, WIDTH), nn.SiLU(), nn.Linear(WIDTH, WIDTH))
        self.reconstruct = nn.Linear(WIDTH, TARGET_DIM, bias=False)
        nn.init.normal_(self.reconstruct.weight, std=.02)
        self.applicability = nn.Linear(WIDTH, 1)
        nn.init.constant_(self.applicability.bias, .25)
        self.layout_query = nn.Linear(model_width, WIDTH)
        self.layout_veto = nn.Linear(WIDTH, 1)
        nn.init.zeros_(self.layout_veto.weight)
        nn.init.constant_(self.layout_veto.bias, -3.)
        self.execute = nn.Linear(WIDTH, model_width, bias=False)
        nn.init.zeros_(self.execute.weight)

        candidates = [index for index, token in enumerate(codec.tokens) if _layout_token(token)]
        # Timing proposals begin after a target/event boundary and have their
        # own tokens. This hook starts only after the event's exact phase.
        layout_inputs = [
            (token != '[END_EVENT]' and _layout_token(token)) or token.startswith(('F:', 'D:', 'V:'))
            or token == '[END_DURATION]' for token in codec.tokens
        ]
        self.layout_input_ids = frozenset(i for i,active in enumerate(layout_inputs) if active)
        self.register_buffer('layout_inputs', torch.tensor(layout_inputs), persistent=False)
        self.register_buffer('component_ids', torch.tensor(candidates, dtype=torch.long), persistent=False)
        # Slot zero belongs to the held-out query. A malformed support mask
        # cannot promote that row into the teacher or the decoder path.
        self.register_buffer('support_slots', torch.arange(MAX_PHRASES).ne(0), persistent=False)
        self.register_buffer('layout_fields', torch.tensor([
            name in ('chord', 'head_cardinality', 'mixed_rim_touch')
            or name.startswith(('first_', 'last_', 'chord_separation_', 'rim_step_'))
            or (name.startswith('slide_') and not name.endswith(
                ('_occupancy', '_release', '_seconds', '_waiting', '_trace_start')))
            for name in FIELDS
        ]), persistent=False)

    def _codes(self):
        # A uniform mixture has exactly the no-code representation. This
        # keeps a shared default correction out of the categorical slots.
        return self.codebook - self.codebook.mean(0)

    def plan(self, song, batch, *, strength, training):
        # The requested level is the only control that enters the audio prior.
        # Radar, style, annotations and decoder state cannot set its switches.
        # Freeze this read into the existing encoder so its auxiliary gradients
        # stay inside the experiment and do not change the base timing path.
        frames = self.song_projection(song.detach())
        mask = batch['song_mask']
        counts = mask.sum(-1).clamp_min(1)
        index = torch.arange(frames.size(1), device=frames.device)
        indices = (index[None] * SONG_SECTIONS // counts[:, None]).clamp_max(SONG_SECTIONS - 1)
        membership = F.one_hot(indices, SONG_SECTIONS).transpose(1, 2) * mask[:, None]
        section_weight = membership.to(frames.dtype)
        pooled = section_weight @ frames / section_weight.sum(-1, keepdim=True).clamp_min(1)
        level = batch['condition'][:, :1].float()
        probabilities = self.prior(torch.cat((pooled.flatten(1), level), dim=-1)).float().softmax(-1)
        teacher, teacher_used = None, None
        selection = probabilities.detach()
        if training:
            # Native token CE learns to execute what the evidence supports.
            # It cannot reinforce an early categorical guess.
            if 'intent_phrase_targets' in batch:
                teacher = self._teacher(batch)
                fraction = float(batch.get('intent_teacher_fraction', TEACHER_FRACTION))
                if not math.isfinite(fraction) or not 0. <= fraction <= 1.:
                    raise ValueError('Phrase intent teacher fraction must be between 0 and 1')
                teacher_used = teacher.support.any(-1) & (torch.rand_like(probabilities[:, 0]) < fraction)
                selection = torch.where(teacher_used[:, None], teacher.probabilities.detach(), selection)
            selected = torch.multinomial(selection, 1).squeeze(-1)
        else:
            seed = int(batch.get('intent_seed', 0))
            keys = (batch.get('intent_song_keys') or batch.get('song_cache_key')
                    or batch.get('song_id') or ('',) * frames.size(0))
            selected = _stable_draw(selection, keys, seed)
        codes = self._codes()
        if training:
            codes = codes.detach()
        variant = codes.index_select(0, selected)
        # The same sampled code has the same effect for the same evidence in
        # training and generation. Uniform or weaker-than-uniform support
        # means exact abstention. Several codes can all have positive support.
        selected_probability = selection.gather(1, selected[:, None]).squeeze(1)
        uniform = 1. / VARIANTS
        selection_evidence = ((selected_probability - uniform) / (1. - uniform)).clamp(0., 1.)
        available = mask.any(-1) & selection_evidence.gt(0.)
        song_beats = batch['song_beats']
        plan = IntentPlan(frames=frames, beats=song_beats.masked_fill(~mask, torch.inf), counts=counts,
                          variant=variant, selected=selected, selection_evidence=selection_evidence,
                          probabilities=probabilities, strength=strength, available=available,
                          diagnostics=None, teacher=teacher, teacher_used=teacher_used)
        if not training:
            with torch.no_grad():
                local = self._pair(self._context(plan, song_beats), variant[:, None, :])
                local_strength = self._strength(self._applicability(local), False) * mask * selection_evidence[:, None]
                strongest = local_strength.amax(-1)
                if frames.size(0) == 1:
                    effective = float(strongest[0]) * strength
                    distribution = probabilities[0].cpu().tolist()
                    diagnostics = dict(variant=int(selected[0]), probabilities=distribution, confidence=max(distribution),
                                       strength=effective, selection_evidence=float(selection_evidence[0]), seed=seed)
                    # The sampled frames summarize suitability for reporting.
                    # Only the actual decoder query can veto a positive
                    # selection between those frames.
                    if not available[0]:
                        strength = 0.
                else:
                    diagnostics = None
            execution = frames.new_zeros(2, dtype=torch.float32) if batch.get('intent_track_execution') else None
            return replace(plan, strength=strength, diagnostics=diagnostics, execution_max=execution)
        return plan

    def _audio_at(self, plan, beats):
        right = torch.searchsorted(plan.beats.contiguous(), beats.contiguous())
        last = plan.counts[:, None] - 1
        lower = torch.minimum((right - 1).clamp_min(0), last)
        upper = torch.minimum(right, last)
        lower_beat = plan.beats.gather(1, lower)
        upper_beat = plan.beats.gather(1, upper)
        fraction = torch.where(lower == upper, 0., (beats - lower_beat) /
                               (upper_beat - lower_beat).clamp_min(1e-6)).clamp(0., 1.)
        rows = torch.arange(beats.size(0), device=beats.device)[:, None]
        return torch.lerp(plan.frames[rows, lower], plan.frames[rows, upper], fraction[..., None].to(plan.frames.dtype))

    def _context(self, plan, beats):
        now = self._audio_at(plan, beats)
        ahead = self._audio_at(plan, beats + LOOKAHEAD_BEATS)
        return self.current_future(torch.cat((now, ahead - now), dim=-1))

    def _pair(self, context, variant):
        return (1. + context.tanh()) * variant

    def _applicability(self, paired):
        return self.applicability(paired).squeeze(-1).float()

    def _strength(self, logits, training):
        probability = logits.sigmoid()
        # Use the same forward abstention in training and generation. Keep a
        # smooth gradient into the gate when its effective strength is zero.
        strength = (3. * probability - 1.2).clamp(0., 1.)
        return strength.detach() + (probability - probability.detach()) if training else strength

    def residual(self, hidden, inputs, beats, token_weight, plan):
        local = self._pair(self._context(plan, beats), plan.variant[:, None, :])
        query = 1. + self.layout_query(hidden.detach()).tanh()
        execution = local * query
        # The decoder's causal state can reduce a recommendation. It cannot
        # revive a variant that the current and upcoming audio do not support.
        veto = .5 * self.layout_veto(execution).squeeze(-1).float().sigmoid()
        strength = self._strength(self._applicability(local) - veto, self.training)
        strength = (strength * self.layout_inputs[inputs] * plan.available[:, None]
                    * plan.selection_evidence[:, None] * plan.strength)
        delta = F.linear(self.execute(execution), token_weight.detach())
        return delta.tanh() * strength[..., None].to(delta.dtype), strength

    def _posterior(self, encoded_audio, descriptors, support_steps, support):
        support_audio = torch.where(support_steps[..., None], encoded_audio, 0.)
        observed = torch.cat((support_audio, descriptors), dim=-1)
        neutral = torch.cat((support_audio, torch.zeros_like(descriptors)), dim=-1)
        steps = self.support_step(torch.stack((observed, neutral), dim=0)) * support_steps[None, ..., None]
        observed_phrase, neutral_phrase = self.support_phrase(steps.flatten(-2)).unbind(0)
        phrases = observed_phrase - neutral_phrase
        pooled = (phrases * support[..., None]).sum(1) / support.sum(1, keepdim=True).clamp_min(1)
        # A fixed floor preserves magnitude when the support contributes
        # very little. Normalizing every observation would amplify noise.
        pooled = pooled.float()
        pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp_min(.25)
        logits = 2. * F.linear(pooled, F.normalize(self._codes().float(), dim=-1))
        return logits.float().softmax(-1)

    def _teacher(self, batch):
        audio = self.phrase_audio(batch['intent_phrase_audio'].float()) + self.phrase_position
        baseline = self.audio_only(audio).float()
        # Forward routing may depend only on explicitly selected support; the
        # query mask and any authored query values belong to the loss below.
        support_mask = batch['intent_support_mask'].bool() & self.support_slots
        eligible = support_mask[..., None] & batch['intent_phrase_valid'].bool()
        weights = torch.where(eligible[..., None], batch['intent_phrase_weights'].float(), 0.)
        steps = weights.gt(0.).any(-1)
        support = steps.any(-1)
        known = weights.gt(0.)
        targets = torch.where(known, batch['intent_phrase_targets'].float(), 0.)
        explained = torch.where(known, baseline.detach(), 0.)
        descriptors = (targets - explained) * weights.clamp(0., 1.)
        probabilities = self._posterior(audio, descriptors, steps, support)
        return PhraseTeacher(audio, baseline, support, steps, descriptors, probabilities)

    @staticmethod
    def _entropy(probability):
        return -(probability * probability.clamp_min(1e-8).log()).sum(-1)

    def auxiliary(self, plan, batch, sample_mass, mass):
        metrics = {name: mass.new_zeros(()) for name in (
            'intent_loss', 'intent_prior_loss', 'intent_reconstruction_loss',
            'intent_audio_only_loss', 'intent_prior_reconstruction_loss', 'intent_prior_top1',
            'intent_wrong_code_loss', 'intent_shuffled_support_loss',
            'intent_posterior_loss', 'intent_query_evidence', 'intent_teacher_confidence', 'intent_selection_evidence',
            'intent_gate_loss', 'intent_gate_target_rate', 'intent_teacher_coverage', 'intent_teacher_route_coverage',
            'intent_future_support_coverage', 'intent_teacher_entropy', 'intent_prior_entropy',
            *(f'intent_teacher_variant_{index}' for index in range(VARIANTS)),
        )}
        metrics['intent_prior_entropy'] = (self._entropy(plan.probabilities) * sample_mass).sum() / mass
        metrics['intent_selection_evidence'] = (plan.selection_evidence * sample_mass).sum() / mass
        if 'intent_phrase_targets' not in batch:
            return metrics['intent_loss'], metrics

        target = batch['intent_phrase_targets'].float()
        valid = batch['intent_phrase_valid'].bool()
        weights = batch['intent_phrase_weights'].float() * valid[..., None]
        covered = weights.sum((-2, -1)) > 0
        query = batch['intent_query_mask'].bool() & covered & ~self.support_slots
        teacher = plan.teacher if plan.teacher is not None else self._teacher(batch)
        eligible = teacher.support.any(-1) & query.any(-1)
        row_mass = sample_mass * eligible

        encoded_audio, baseline, posterior = teacher.audio, teacher.baseline, teacher.probabilities
        teacher_index = posterior.argmax(-1)

        query_weights = weights * query[..., None, None]
        query_mass = query_weights.sum((1, 2, 3)).clamp_min(1e-8)
        bounds = batch['intent_phrase_bounds'].float()
        starts = bounds[..., :1] + .5 * torch.arange(target.size(-2), device=target.device)
        beats = .5 * (starts + torch.minimum(starts + .5, bounds[..., 1:]))
        local = self._context(plan, beats.flatten(1)).reshape(*beats.shape, WIDTH)
        paired = self._pair(local[..., None, :], self._codes())
        shift = self.reconstruct(paired * (1. + encoded_audio.tanh()[..., None, :])).float().tanh()
        structured = baseline.detach()[..., None, :] + shift - shift.mean(-2, keepdim=True)

        base_error = F.smooth_l1_loss(baseline, target, reduction='none')
        variant_error = F.smooth_l1_loss(structured, target[..., None, :].expand_as(structured), reduction='none')
        baseline_cost = (base_error * query_weights).sum((1, 2, 3)) / query_mass
        variant_cost = (variant_error * query_weights[..., None, :]).sum((1, 2, 4)) / query_mass[:, None]
        with torch.no_grad():
            # Fit all experts from independent held-out prediction error,
            # without using either student as its own target. Small, noisy
            # differences can train the experts while keeping both switches
            # uncertain. Only an improvement over the no-code predictor
            # supplies evidence for preferring a configuration.
            gain = (baseline_cost[:, None] - variant_cost) / (baseline_cost[:, None] + .02)
            fit = (gain / .04).softmax(-1)
            evidence = ((gain.amax(-1, keepdim=True) - .015) / .15).clamp(0., 1.)
            responsibility = torch.lerp(torch.full_like(fit, 1. / VARIANTS), fit, evidence)
        expert_cost = (variant_cost * fit).sum(-1)
        selected_cost = (variant_cost * posterior.detach()).sum(-1)
        posterior_cost = -(responsibility * posterior.clamp_min(1e-8).log()).sum(-1)
        prior_cost = -(responsibility * plan.probabilities.clamp_min(1e-8).log()).sum(-1)

        # Train abstention from held-out layout improvement only. Pulse and
        # sustain timing still teach the code, but cannot justify using it at
        # a native arrangement decision. The token objective also supplies a
        # gradient to both gates through the actual component logit residual.
        gate_weights = query_weights * self.layout_fields
        benefit = ((base_error.detach()[..., None, :] - variant_error.detach())
                   * gate_weights[..., None, :]).sum(-1)
        gate_target = (benefit > 0.).float()
        gate_error = F.binary_cross_entropy_with_logits(self._applicability(paired), gate_target, reduction='none')
        gate_mass = gate_weights.sum((1, 2, 3)).clamp_min(1e-8)
        chosen_weight = gate_weights.sum(-1)[..., None] * responsibility[:, None, None, :]
        gate_cost = (gate_error * chosen_weight).sum((1, 2, 3)) / gate_mass
        entropy = self._entropy(posterior)

        def mean(values):
            return (values * row_mass).sum() / mass

        loss = mean(baseline_cost + expert_cost + .25 * (posterior_cost + prior_cost) + .05 * gate_cost)
        with torch.no_grad():
            reversed_support = self._posterior(encoded_audio, teacher.descriptors.flip(-2), teacher.steps, teacher.support)
            shuffled_cost = (variant_cost * reversed_support).sum(-1)
            wrong_cost = (variant_cost * (1 - posterior)).sum(-1) / (VARIANTS - 1)
            metrics.update(intent_loss=loss, intent_prior_loss=mean(prior_cost),
                           intent_reconstruction_loss=mean(selected_cost), intent_audio_only_loss=mean(baseline_cost),
                           intent_prior_reconstruction_loss=mean((variant_cost * plan.probabilities).sum(-1)),
                           intent_prior_top1=mean(plan.probabilities.argmax(-1).eq(teacher_index)),
                           intent_posterior_loss=mean(posterior_cost), intent_query_evidence=mean(evidence.squeeze(-1)),
                           intent_teacher_confidence=mean(((posterior.amax(-1) * VARIANTS - 1) / (VARIANTS - 1))),
                           intent_wrong_code_loss=mean(wrong_cost), intent_shuffled_support_loss=mean(shuffled_cost),
                           intent_gate_loss=mean(gate_cost), intent_teacher_entropy=mean(entropy),
                           intent_gate_target_rate=mean((gate_target * chosen_weight).sum((1, 2, 3)) / gate_mass),
                           intent_teacher_coverage=row_mass.sum() / mass)
            if plan.teacher_used is not None:
                metrics['intent_teacher_route_coverage'] = (plan.teacher_used * sample_mass).sum() / mass
            if 'intent_future_support_mask' in batch:
                future = (batch['intent_future_support_mask'] & teacher.support).any(-1)
                metrics['intent_future_support_coverage'] = mean(future)
            metrics.update({f'intent_teacher_variant_{index}': mean(posterior[:, index])
                            for index in range(VARIANTS)})
        return loss, metrics
