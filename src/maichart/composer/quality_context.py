"""Small learned configuration context, with no future-chart inputs.

Audio/symbol anchors are committed events. Phrase and next-gap labels are read
only by ``quality_context_loss``; decoding receives the model's predictions.
Legacy anchors use [batch, candidates, 8] attention; song-paired history uses
[batch, candidates, history] scalar scores without repeating hidden vectors.
One zero-initialized output projection preserves the baseline action context
when this branch is first introduced.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .acoustics import CONTEXT_DIM
from .stability import bounded_history_memory

QUALITY_CONTEXT_POLICY = 'cooperative_v1'
QUALITY_MAX_ANCHORS = 8
QUALITY_LIFECYCLE_DIM = 88
QUALITY_PHRASE_CHANNELS = (
    'event_rate', 'heads_per_event', 'chord_fraction', 'hold_fraction',
    'slide_fraction', 'touch_fraction', 'button_spread', 'ending_intensity',
)
# These are upper bounds, not rounding boundaries. The final class is >4 beats.
QUALITY_GAP_UPPER_BEATS = (.125, .25, .5, 1., 2., 4.)
QUALITY_GAP_REPRESENTATIVE_BEATS = (.125, .25, .5, 1., 2., 4., 8.)
QUALITY_GAP_CLASSES = len(QUALITY_GAP_REPRESENTATIVE_BEATS)
QUALITY_ACTION_PREDICTION_GRAD_SCALE = .1
QUALITY_PAIR_TIME_DIM = 6
SONG_RELATIVE_TIME_DIM = 10
SONG_RELATIVE_TIME_PERIODS = (2., 4., 1., 3.)


def song_relative_time_scores(projected_query, offsets, ages):
    """Ten relative-time features without a [B,C,H,10] feature tensor.

    Periodic features factor through sin(x+y)/cos(x+y); only the two
    nonperiodic distance terms need a candidate-by-history tensor. This is
    the same score as stacking log/decay and four sin/cos pairs.
    """
    query=projected_query.float()
    delta=offsets.float()[...,None]+ages.float()[:,None]
    positive=delta.clamp_min(0.)
    scores=query[...,0,None]*torch.log1p(positive)+query[...,1,None]*torch.exp(-positive/4.)
    frequency=offsets.new_tensor([2.*math.pi/p for p in SONG_RELATIVE_TIME_PERIODS],dtype=torch.float32)
    current=offsets.float()[...,None]*frequency
    previous=ages.float()[...,None]*frequency
    sin_current,cos_current=current.sin(),current.cos()
    sin_query,cos_query=query[...,2::2],query[...,3::2]
    scores=scores+torch.bmm(sin_query*sin_current+cos_query*cos_current,previous.cos().transpose(1,2))
    scores=scores+torch.bmm(sin_query*cos_current-cos_query*sin_current,previous.sin().transpose(1,2))
    return scores/math.sqrt(SONG_RELATIVE_TIME_DIM)


def _prediction_condition(prediction):
    """Keep predictions numerically exact but limit action-driven reshaping.

    Direct phrase/gap supervision still reads the original prediction tensor.
    Only its use as an action-decoder condition gets this gradient scale.
    """
    return prediction.detach() + float(QUALITY_ACTION_PREDICTION_GRAD_SCALE) * (prediction - prediction.detach())


class QualityContext(nn.Module):
    """Pair recent chart actions with their audio and predict a short plan.

    ``inputs`` follows ``quality_inputs.py``. ``anchor_embeddings`` is supplied
    by the main model, sharing its existing atom embedding/factor parameters.
    Symbolic dropout hides both lifecycle and paired anchors coherently. The
    audio-only residual never receives either of those two symbolic sources.
    """

    def __init__(self, width: int, condition_dim: int = 16, *, audio_dim: int = 120,
                 lifecycle_dim: int = QUALITY_LIFECYCLE_DIM, hidden: int = 64,
                 future_evidence: bool = False, paired_v3: bool = False):
        super().__init__()
        self.paired_v3 = paired_v3
        self.phrase_channels = 9 if paired_v3 else len(QUALITY_PHRASE_CHANNELS)
        self.width = int(width)
        self.hidden = min(int(hidden), self.width)
        self.audio_dim = int(audio_dim)
        h = self.hidden
        self.query = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, h), nn.GELU())
        self.condition = nn.Linear(condition_dim, h, bias=False)
        self.anchor_audio = nn.Sequential(nn.LayerNorm(audio_dim), nn.Linear(audio_dim, h, bias=False), nn.LayerNorm(h))
        self.anchor_symbol = nn.Sequential(nn.Linear(width, h, bias=False), nn.LayerNorm(h))
        self.anchor_age = nn.Linear(4, h, bias=False)
        self.anchor_norm = nn.LayerNorm(h)
        self.anchor_query = nn.Linear(h, h, bias=False)
        self.pair_time_query = nn.Linear(h, QUALITY_PAIR_TIME_DIM, bias=False)
        self.pair = nn.Sequential(nn.Linear(h * 3 + QUALITY_PAIR_TIME_DIM, h, bias=False), nn.GELU(), nn.LayerNorm(h))
        self.lifecycle = nn.Sequential(nn.Linear(lifecycle_dim, h, bias=False), nn.GELU(), nn.LayerNorm(h))
        # A per-window network makes every candidate in the same fixed phrase
        # read exactly the same prediction, independent of candidate packing.
        self.phrase = nn.Sequential(
            nn.Linear(CONTEXT_DIM + condition_dim, h), nn.GELU(),
            nn.Linear(h, h), nn.GELU(), nn.Linear(h, len(QUALITY_PHRASE_CHANNELS)),
        )
        if paired_v3:
            # The song trunk owns phrase prediction; no independent anchor branch.
            for name in ('anchor_audio', 'anchor_symbol', 'anchor_age', 'anchor_norm',
                         'anchor_query', 'pair_time_query', 'pair', 'phrase'):
                delattr(self, name)
            self.song_query = nn.Linear(96, h, bias=False)
            self.music_query_norm = nn.LayerNorm(h)
            self.relative_time_query = nn.Linear(h, SONG_RELATIVE_TIME_DIM, bias=False)
            self.history_audio = nn.Linear(104 + 96, h, bias=False)
            self.history_symbol = nn.Linear(width, h, bias=False)
            self.history_clock = nn.Linear(2, h, bias=False)
            self.history_pair = nn.Sequential(nn.Linear(h * 3, h), nn.GELU(), nn.LayerNorm(h))
            self.song_state = nn.Linear(96, h, bias=False)
        self.phrase_embedding = nn.Linear(self.phrase_channels, h, bias=False)
        self.gap_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, QUALITY_GAP_CLASSES))
        self.gap_embedding = nn.Linear(QUALITY_GAP_CLASSES, h, bias=False)
        self.future_evidence_projection = None
        if future_evidence:
            from .future_gap_evidence import FUTURE_GAP_EVIDENCE_DIM
            self.future_evidence_projection = nn.Linear(FUTURE_GAP_EVIDENCE_DIM, h, bias=False)
            nn.init.zeros_(self.future_evidence_projection.weight)
        self.residual_norm = nn.LayerNorm(h)
        self.output = nn.Linear(h, width, bias=False)
        nn.init.zeros_(self.output.weight)

    def _paired_context(self, query, inputs, anchor_embeddings):
        ids = inputs['quality_anchor_ids']
        valid = inputs['quality_anchor_mask'].bool()
        atom_valid = ids.ne(0)
        # Every anchor remains an event/set; slot ordering is not a hand label.
        symbols = (anchor_embeddings * atom_valid[..., None]).sum(-2)
        symbols = symbols / atom_valid.sum(-1, keepdim=True).clamp_min(1)
        audio = bounded_history_memory(self.anchor_audio(inputs['quality_anchor_audio']), ~valid, 4.)
        symbols = bounded_history_memory(self.anchor_symbol(symbols), ~valid, 4.)
        age = inputs['quality_anchor_ages'].float().clamp_min(0.)
        age_features = torch.stack((
            torch.log1p(age) / math.log(65.), torch.exp(-age / 4.),
            torch.sin(age * (2. * math.pi)), torch.cos(age * (2. * math.pi)),
        ), dim=-1)
        memory = self.anchor_norm(audio + symbols + self.anchor_age(age_features.to(audio.dtype)))
        memory = bounded_history_memory(memory, ~valid, 4.)
        scores = torch.bmm(self.anchor_query(query).float(), memory.float().transpose(1, 2))
        scores = scores / math.sqrt(self.hidden)
        # Ages end at the last committed event. Add the actual candidate
        # offset so a near pickup and a far phrase entry are distinct queries.
        # Only six scalars per pair are materialized, never [B,C,A,width].
        delta = inputs['quality_candidate_offsets'].float()[..., None] + age[:, None]
        positive_delta = delta.clamp_min(0.)
        temporal = torch.stack((
            torch.log1p(positive_delta) / math.log(65.), torch.exp(-positive_delta / 4.),
            torch.sin(delta * math.pi), torch.cos(delta * math.pi),
            torch.sin(delta * (2. * math.pi)), torch.cos(delta * (2. * math.pi)),
        ), dim=-1)
        scores = scores + torch.einsum('bcaf,bcf->bca', temporal,
            self.pair_time_query(query).float()) / math.sqrt(QUALITY_PAIR_TIME_DIM)
        weights = scores.masked_fill(~valid[:, None], -1e4).softmax(-1)
        weights = weights * valid[:, None]
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        paired_audio = torch.bmm(weights.to(audio.dtype), audio)
        paired_symbols = torch.bmm(weights.to(symbols.dtype), symbols)
        paired_time = torch.einsum('bca,bcaf->bcf', weights, temporal).to(query.dtype)
        # Difference and product expose the relationship to the old sound,
        # while its paired action says how that sound was previously written.
        pair = self.pair(torch.cat((query - paired_audio, query * paired_audio, paired_symbols, paired_time), -1))
        return pair * valid.any(-1)[:, None, None]

    def forward(self, audio_context: torch.Tensor, condition: torch.Tensor,
                inputs: Mapping[str, torch.Tensor], *, anchor_embeddings: torch.Tensor,
                symbolic_keep: torch.Tensor | None = None,
                valid_mask: torch.Tensor | None = None, context_strength: float = 1.,
                plan_strength: float = 1., placement_probs: torch.Tensor | None = None,
                attack: torch.Tensor | None = None, song_state=None, phrase_prediction=None,
                history_memory=None, history_padding=None, history_song=None,
                history_reference_beat=None) -> dict[str, torch.Tensor]:
        """Return residuals and predictions; target/weight fields are unused.

        audio_context [B,C,D], anchor_embeddings [B,8,S,D]. Phrase audio is
        [B,P,104], and phrase_indices [B,C] selects one fixed window per beat.
        The caller adds ``quality_residual`` to the full action context and
        ``quality_audio_residual`` to its separate audio-only auxiliary path.
        """
        batch, count, _ = audio_context.shape
        query = self.query(audio_context)
        if not self.paired_v3:
            phrase_audio = inputs['quality_phrase_audio']
            phrase_condition = condition[:, None].expand(-1, phrase_audio.size(1), -1)
            phrase_prediction = self.phrase(torch.cat((phrase_audio, phrase_condition), -1)).sigmoid()
        indices = inputs['quality_phrase_indices'].long()
        selected_phrase = phrase_prediction.gather(
            1, indices[..., None].expand(-1, -1, self.phrase_channels))
        phrase_context = float(plan_strength) * self.phrase_embedding(selected_phrase)
        action_phrase_context = float(plan_strength) * self.phrase_embedding(_prediction_condition(selected_phrase))
        # Keep phrase->gap auxiliary learning intact. The action route through
        # gap is scaled at gap's output below, so it cannot bypass the 0.1 limit
        # or accidentally receive the scale twice (0.01).
        audio_base = query + self.condition(condition)[:, None] + phrase_context
        if self.future_evidence_projection is not None:
            if placement_probs is None or attack is None or valid_mask is None:
                raise ValueError('Future gap evidence requires placement, attack and explicit valid_mask')
            from .future_gap_evidence import future_gap_evidence
            evidence = future_gap_evidence(inputs['quality_candidate_offsets'], valid_mask,
                                           placement_probs, attack)
            audio_base = audio_base + self.future_evidence_projection(evidence.to(query.dtype))
        if self.paired_v3:
            projected_song = self.song_state(song_state)
            action_phrase_context = action_phrase_context + projected_song
            audio_base = audio_base + projected_song
            paired = torch.zeros_like(query)
            if history_memory is not None and history_memory.size(1):
                valid_history = ~history_padding
                music = self.history_audio(torch.cat((inputs['quality_history_audio'], history_song), -1))
                clock = inputs['quality_history_clock']
                music = music + self.history_clock(torch.sign(clock) * torch.log1p(clock.abs()))
                music = bounded_history_memory(music, history_padding, 4.)
                symbols = bounded_history_memory(self.history_symbol(history_memory), history_padding, 4.)
                current = self.music_query_norm(query + self.song_query(song_state))
                scores = torch.bmm(current.float(), music.float().transpose(1, 2)) / math.sqrt(self.hidden)
                positions = torch.arange(valid_history.size(1), device=valid_history.device)[None]
                last = positions.masked_fill(~valid_history, -1).max(1).values.clamp_min(0)
                last_beat = (clock[..., 0].gather(1, last[:, None]) if history_reference_beat is None
                             else history_reference_beat.reshape(-1, 1))
                ages=last_beat-clock[...,0]
                scores=scores+song_relative_time_scores(self.relative_time_query(current),
                    inputs['quality_candidate_offsets'],ages)
                weights = scores.masked_fill(~valid_history[:, None], -1e4).softmax(-1) * valid_history[:, None]
                weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
                old_music = torch.bmm(weights.to(music.dtype), music)
                old_symbol = torch.bmm(weights.to(symbols.dtype), symbols)
                paired = self.history_pair(torch.cat((current-old_music, current*old_music, old_symbol), -1))
                paired = paired * valid_history.any(-1)[:, None, None]
        else:
            paired = self._paired_context(query, inputs, anchor_embeddings)
        lifecycle = self.lifecycle(inputs['quality_lifecycle'])
        keep = (query.new_ones((batch, 1, 1)) if symbolic_keep is None
                else symbolic_keep.to(query.dtype).reshape(batch, 1, 1))
        symbolic = float(context_strength) * (paired + lifecycle) * keep
        gap_logits = self.gap_head(audio_base + symbolic)
        audio_gap_logits = self.gap_head(audio_base)
        gap_context = self.gap_embedding(_prediction_condition(gap_logits.float().softmax(-1)).to(query.dtype))
        audio_gap_context = self.gap_embedding(_prediction_condition(audio_gap_logits.float().softmax(-1)).to(query.dtype))
        residual = self.output(self.residual_norm(action_phrase_context + symbolic + gap_context))
        audio_residual = self.output(self.residual_norm(action_phrase_context + audio_gap_context))
        if valid_mask is not None:
            residual = residual * valid_mask[..., None]
            audio_residual = audio_residual * valid_mask[..., None]
        return {
            'quality_residual': residual,
            'quality_audio_residual': audio_residual,
            'quality_phrase_prediction': phrase_prediction,
            'quality_gap_logits': gap_logits,
            'quality_gap_audio_logits': audio_gap_logits,
        }


def quality_context_loss(output: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor], *,
                         phrase_weight: float = .1, gap_weight: float = .1):
    """Confidence-weighted phrase regression and selected-event interval CE.

    Tail/unknown next events have zero ``quality_gap_weight``. Recovery batches
    also set these label weights to zero rather than inventing labels for their
    generated histories. No target tensor is needed by ``QualityContext``.
    """
    prediction = output['quality_phrase_prediction']
    logits = output['quality_gap_logits']
    sample_weight = batch.get('sample_weight', prediction.new_ones(prediction.size(0))).float()
    phrase_mask = batch['quality_phrase_weight'].float() * sample_weight[:, None]
    phrase_error = F.smooth_l1_loss(prediction.float(), batch['quality_phrase_target'].float(), reduction='none').mean(-1)
    phrase_sum = (phrase_error * phrase_mask).sum()
    phrase_mass = phrase_mask.sum()
    phrase_loss = phrase_sum / phrase_mass.clamp_min(1e-6)

    index = batch['candidate_target'].long()
    target = batch['quality_gap_target'].long()
    valid = (index >= 0) & (index < logits.size(1)) & (target >= 0) & (target < QUALITY_GAP_CLASSES)
    selected = logits[torch.arange(logits.size(0), device=logits.device), index.clamp(0, logits.size(1) - 1)]
    gap_error = F.cross_entropy(selected.float(), target.clamp(0, QUALITY_GAP_CLASSES - 1), reduction='none')
    # Give the audio-only decoder the same interval objective: otherwise the
    # full head could learn only its symbolic route, leaving the audio-only
    # predicted-gap conditioning untrained. Split, rather than double, weight.
    audio_logits = output['quality_gap_audio_logits']
    audio_selected = audio_logits[torch.arange(logits.size(0), device=logits.device),
                                  index.clamp(0, logits.size(1) - 1)]
    audio_gap_error = F.cross_entropy(audio_selected.float(),
                                     target.clamp(0, QUALITY_GAP_CLASSES - 1), reduction='none')
    gap_mask = batch['quality_gap_weight'].float() * sample_weight * valid
    gap_sum = (.5 * (gap_error + audio_gap_error) * gap_mask).sum()
    gap_mass = gap_mask.sum()
    gap_loss = gap_sum / gap_mass.clamp_min(1e-6)
    loss = float(phrase_weight) * phrase_loss + float(gap_weight) * gap_loss
    return loss, {
        'quality_phrase_loss': phrase_loss.detach(), 'quality_gap_loss': gap_loss.detach(),
        'quality_gap_full_loss': ((gap_error * gap_mask).sum() / gap_mass.clamp_min(1e-6)).detach(),
        'quality_gap_audio_loss': ((audio_gap_error * gap_mask).sum() / gap_mass.clamp_min(1e-6)).detach(),
        'quality_phrase_error_sum': phrase_sum.detach(), 'quality_phrase_weight': phrase_mass.detach(),
        'quality_gap_ce_sum': gap_sum.detach(), 'quality_gap_ce_weight': gap_mass.detach(),
    }
