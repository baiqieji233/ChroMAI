"""Continuous phrase intentions from audio and completed causal references.

Targets are deliberately absent from this module's forward input selection.
Audio sampling is bounded (nine points per interval), reusing encoded banks.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from .stream_phrase import PHRASE_FEATURE_DIM


class OrderedPhraseContext(nn.Module):
    """Configuration-only sequence memory plus a separate cached music readout.

    Reference tables are encoded once per unique segment. Target tables are
    exclusively encoded by ordered_phrase_auxiliary_loss. Contrastive label
    gradients never turn future configurations into decoder inputs.
    """
    def __init__(self, model_width, width, *, aligned=False):
        super().__init__()
        self.width = width
        self.audio_keys = nn.Linear(model_width, 64)
        self.audio_summary = nn.Sequential(nn.Linear(4*64, width), nn.GELU(), nn.LayerNorm(width))
        self.sequence_input = nn.Sequential(nn.Linear(4*model_width+17, width), nn.GELU(), nn.LayerNorm(width))
        self.sequence_gru = nn.GRU(width, width, batch_first=True)
        self.sequence_norm = nn.LayerNorm(width)
        self.reference_fusion = nn.Sequential(nn.Linear(2*width, width), nn.GELU(), nn.LayerNorm(width))
        self.reference_output = nn.Linear(width, width, bias=False)
        self.anchor_output = nn.Linear(width, width, bias=False)
        self.actor_input = nn.Sequential(nn.Linear(model_width, width), nn.LayerNorm(width))
        self.interaction = nn.Sequential(nn.Linear(4*width+1, width), nn.GELU(), nn.LayerNorm(width))
        self.interaction_output = nn.Linear(width, model_width, bias=False)
        self.predict = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.LayerNorm(width), nn.Linear(width, width))
        self.aligned = aligned
        if aligned:
            self.alignment_query = nn.Linear(2*width+1,width)
            self.alignment_key = nn.Linear(width,width)
            self.alignment_null = nn.Parameter(torch.zeros(width))
            self.alignment_output = nn.Linear(width,model_width,bias=False)
            nn.init.zeros_(self.alignment_output.weight)
        for layer in (self.reference_output, self.anchor_output, self.interaction_output):
            nn.init.zeros_(layer.weight)

    def encode_sequences(self, model, batch, prefix='phrase_sequence_', *, return_states=False):
        kinds = batch[prefix+'kind']
        if kinds.size(0) == 0:
            empty = self.sequence_norm.weight.new_zeros(0, self.width)
            return (empty,empty.new_zeros(0,kinds.size(1),self.width)) if return_states else empty
        mask = batch[prefix+'mask'].bool()
        atoms = batch[prefix+'atoms']
        atom_mask = atoms.ne(0)
        # Sum complete unordered atoms without materializing [U,32,A,d_model].
        # Every original atom ID still contributes, including literal paths.
        atom_sum = F.embedding_bag(atoms.flatten(0, 1), model.atom_embedding.weight,
                                   mode='sum', padding_idx=0).reshape(*atoms.shape[:2], -1)
        atom_mean = atom_sum/atom_mask.sum(-1).clamp_min(1)[..., None]
        phase_mask = batch[prefix+'phase_mask'].to(atom_mean)
        phase_values = (phase_mask @ model.phase_embedding.weight)/phase_mask.sum(-1).clamp_min(1)[..., None]
        # Absolute song position is not a label. Retain exact relative timing
        # and actual gaps, including timing differences between sampled events.
        beats = batch[prefix+'beats'].double()
        seconds = batch[prefix+'seconds'].double()
        first = mask.long().argmax(-1, keepdim=True)
        relative_beats = beats-beats.gather(1, first)
        relative_seconds = seconds-seconds.gather(1, first)
        gap_beats = batch[prefix+'gap_beats'].double()
        gap_seconds = batch[prefix+'gap_seconds'].double()
        clocks = torch.stack((relative_beats, relative_seconds, gap_beats, gap_seconds), -1)
        scaled = clocks.sign()*clocks.abs().log1p()
        periods = beats.new_tensor([.25,.5,1.])
        angles = 2*math.pi*relative_beats[..., None]/periods
        gap_angles = 2*math.pi*gap_beats[..., None]/periods
        timing = torch.cat((scaled, angles.sin(), angles.cos(), gap_angles.sin(), gap_angles.cos(),
                            atom_mask.sum(-1).float().log1p()[..., None]), -1).float()
        features = torch.cat((model.kind_embedding(kinds), model.shape_embedding(batch[prefix+'shape']),
                              phase_values, atom_mean, timing), -1)
        inputs = self.sequence_input(features).masked_fill(~mask[..., None], 0.)
        sequence, _ = self.sequence_gru(inputs)
        last = (mask.long().sum(-1)-1).clamp_min(0)
        pooled = sequence.gather(1, last[:, None, None].expand(-1, 1, self.width))[:, 0]
        pooled = self.sequence_norm(pooled).masked_fill(~mask.any(-1)[:, None], 0.)
        return (pooled,self.sequence_norm(sequence).masked_fill(~mask[...,None],0.)) if return_states else pooled

    def aligned_read(self, batch, states, anchor, style, valid, projected_keys=None):
        """Read bounded reference event states near the landing-relative beat.

        Every reference has its own duration. A learned null supports novel
        writing even when acoustically similar references are available.
        """
        indices = batch['phrase_sequence_inverse']
        mask_bank = torch.cat((batch['phrase_sequence_mask'].new_zeros(1,states.size(1)),
                               batch['phrase_sequence_mask']),0)
        event_mask = mask_bank[(indices+1).clamp_min(0)] & valid[...,None]
        beats_bank = torch.cat((batch['phrase_sequence_beats'].new_zeros(1,states.size(1)),
                                batch['phrase_sequence_beats']),0)
        event_beats = beats_bank[(indices+1).clamp_min(0)]
        start,end = batch['phrase_memory_start_beat'],batch['phrase_memory_end_beat']
        event_position = ((event_beats-start[...,None])/(end-start).clamp_min(1e-8)[...,None]).float()
        landing = ((batch['step_beat']-batch['phrase_start_beat'])/
                   (batch['phrase_end_beat']-batch['phrase_start_beat']).clamp_min(1e-8)).float().clamp(0,1)
        query = self.alignment_query(torch.cat((anchor,style,landing[...,None]),-1))
        distance = (event_position-landing[...,None,None]).abs().masked_fill(~event_mask,torch.inf)
        # Find only the two nearest timed events per reference using scalar
        # positions. Never broadcast all 32 hidden vectors to every query.
        proximity,chosen = distance.topk(2,dim=-1,largest=False)
        unique_keys = self.alignment_key(states) if projected_keys is None else projected_keys
        sequence_bank = torch.cat((states.new_zeros(1,*states.shape[1:]),states),0)
        key_bank = torch.cat((unique_keys.new_zeros(1,*unique_keys.shape[1:]),unique_keys),0)
        selected = (indices+1).clamp_min(0)[...,None].expand_as(chosen)
        events = sequence_bank[selected,chosen]
        keys = key_bank[selected,chosen]
        scores = (query[...,None,None,:]*keys).sum(-1)/math.sqrt(self.width)
        # Smooth proximity retains off-beat responses and unequal phrase spans.
        scores = (scores-4.*proximity).masked_fill(~torch.isfinite(proximity),-torch.inf).flatten(-2)
        null_score = (query*self.alignment_null).sum(-1)[...,None]/math.sqrt(self.width)
        scores = torch.cat((null_score,scores),-1)
        weights = scores.float().softmax(-1).to(events)
        values = torch.cat((self.alignment_null.expand(*anchor.shape[:-1],1,self.width),
                            events.flatten(-3,-2)), -2)
        return (weights[...,None]*values).sum(-2),weights[...,0],unique_keys

    def prepare_audio(self, cache):
        if 'ordered_configuration_keys' in cache:
            return cache
        return {**cache, 'ordered_configuration_keys': self.audio_keys(cache['local'])}

    def music(self, model, cache, inverse, start, end):
        fractions = start.new_tensor([0., .125, .25, .375, .5, .625, .75, .875, 1.])
        positions = start[..., None]+(end-start).clamp_min(0)[..., None]*fractions
        values = model._read_audio(cache, positions, inverse, clock='beats', field='ordered_configuration_keys')
        pooled = torch.cat((values[..., :3, :].mean(-2), values[..., 3:5, :].mean(-2),
                            values[..., 5:7, :].mean(-2), values[..., 7:, :].mean(-2)), -1)
        return self.audio_summary(pooled)

    def music_table(self, cache, batch):
        """Read every unique configuration-music interval once, not once/T.

        Flattened clock search avoids copying a full-song clock per unique
        interval. The old parent's delta32 music path deliberately stays intact.
        """
        from .stream_duration import _read_intervals
        start,end=batch['phrase_music_start_beat'],batch['phrase_music_end_beat']
        inverse=batch['phrase_music_song_inverse']
        fractions=start.new_tensor([0.,.125,.25,.375,.5,.625,.75,.875,1.])
        positions=start[:,None]+(end-start).clamp_min(0)[:,None]*fractions
        # _read_intervals uses a seconds-named clock with a nonnegative origin;
        # an affine shift supplies the exact same ordered beat coordinates.
        beats=cache['beats'].double()
        origin=beats.masked_fill(~cache['mask'],torch.inf).amin().clamp_max(0)
        shifted={key:value for key,value in cache.items() if key!='_duration_clock'}
        shifted['seconds']=beats-origin
        values=_read_intervals(shifted,(positions-origin)[:,None,:],inverse,
                              field='ordered_configuration_keys')[:,0]
        pooled=torch.cat((values[:,:3].mean(-2),values[:,3:5].mean(-2),
                          values[:,5:7].mean(-2),values[:,7:].mean(-2)),-1)
        return self.audio_summary(pooled)

    def actor_residual(self, hidden, output, batch):
        intent, progress = output['phrase_intent'], output['phrase_progress_repr']
        start, end = batch['phrase_start_beat'], batch['phrase_end_beat']
        remaining = ((end-batch['step_beat'])/(end-start).clamp_min(1e-8)).clamp(0, 1).float()
        prefix = self.actor_input(hidden)
        features = torch.cat((intent, progress, intent*progress, prefix, remaining[..., None]), -1)
        residual = self.interaction_output(self.interaction(features)).to(hidden)
        if self.aligned and 'phrase_aligned_read' in output:
            residual = residual+self.alignment_output(output['phrase_aligned_read']).to(hidden)
        return residual*batch['step_mask'][..., None]


class CausalPhraseContext(nn.Module):
    def __init__(self, width, condition_dim, *, ordered=False, aligned=False):
        super().__init__()
        model_width, width = width, min(width, 128)
        self.phrase_width = width
        self.audio = nn.Sequential(nn.Linear(32 * 4, width), nn.GELU(), nn.LayerNorm(width))
        self.writing = nn.Sequential(nn.Linear(PHRASE_FEATURE_DIM, width), nn.GELU(), nn.LayerNorm(width))
        self.shared = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(), nn.LayerNorm(width))
        self.style = nn.Sequential(nn.Linear(condition_dim, width), nn.LayerNorm(width))
        self.query = nn.Linear(width * 2, width)
        self.key = nn.Linear(width, width)
        self.null_reference = nn.Parameter(torch.zeros(width))
        self.intent = nn.Sequential(nn.Linear(width * 3, width), nn.GELU(), nn.LayerNorm(width))
        self.progress = nn.Sequential(nn.Linear(PHRASE_FEATURE_DIM, width), nn.GELU(), nn.LayerNorm(width))
        self.predict = nn.Linear(width, PHRASE_FEATURE_DIM)
        self.intent_output = nn.Linear(width, model_width, bias=False)
        self.progress_output = nn.Linear(width, model_width, bias=False)
        self.style_output = nn.Linear(width, model_width, bias=False)
        self.tracking = nn.Linear(width, width)
        self.tracking_acoustic = nn.Linear(width, 32)
        self.tracking_output = nn.Linear(width, model_width, bias=False)
        for layer in (self.intent_output, self.progress_output, self.style_output, self.tracking_output):
            nn.init.zeros_(layer.weight)
        if ordered or aligned:
            self.ordered = OrderedPhraseContext(model_width, width, aligned=aligned)

    def _music(self, model, cache, inverse, start, end):
        fractions = start.new_tensor([0., .125, .25, .375, .5, .625, .75, .875, 1.])
        positions = start[..., None] + (end-start).clamp_min(0)[..., None]*fractions
        if 'delta_keys' in cache:
            values = model._read_audio(cache, positions, inverse, clock='beats', field='delta_keys')
        else:
            # Slice the shared bank before gathering any phrase intervals.
            source = cache['local'][..., :32]
            if source.size(-1) < 32: source = F.pad(source, (0, 32-source.size(-1)))
            compact = {**cache, 'phrase_audio_keys': source}
            values = model._read_audio(compact, positions, inverse, clock='beats', field='phrase_audio_keys')
        # Preserve coarse temporal order instead of collapsing the phrase to a mean.
        pooled = torch.stack((values[..., :3, :].mean(-2), values[..., 3:5, :].mean(-2),
                              values[..., 5:7, :].mean(-2), values[..., 7:, :].mean(-2)), -2)
        return self.audio(pooled.flatten(-2))

    def forward(self, model, cache, inverse, batch, tracking_strength=0., *, encoded_cache=None):
        start, end = batch['phrase_start_beat'], batch['phrase_end_beat']
        features = batch['phrase_memory_features'].float()
        if features.size(-1) != PHRASE_FEATURE_DIM or features.size(-2) > 4:
            raise ValueError('Phrase memory requires PHRASE_FEATURE_DIM features and at most four references')
        memory_start, memory_end = batch['phrase_memory_start_beat'], batch['phrase_memory_end_beat']
        valid = batch['phrase_memory_mask'].bool() & (memory_end <= start[..., None]) & (memory_end > memory_start)
        anchor = self._music(model, cache, inverse, start, end)
        music = self._music(model, cache, inverse, memory_start, memory_end)
        # Mask before normalization so masked recovery rows cannot poison the path.
        writing = self.writing(torch.nan_to_num(features).masked_fill(~valid[..., None], 0.))
        references = self.shared(torch.cat((music, writing), -1))
        ordered_output = {}
        if hasattr(self, 'ordered'):
            ordered_cache = self.ordered.prepare_audio(cache)
            if 'phrase_music_anchor_inverse' in batch:
                music_bank=self.ordered.music_table(ordered_cache,batch)
                anchor_music=music_bank[batch['phrase_music_anchor_inverse']]
                reference_music=music_bank[batch['phrase_music_reference_inverse']]
            else:
                anchor_music=self.ordered.music(model, ordered_cache, inverse, start, end)
                reference_music=self.ordered.music(model, ordered_cache, inverse, memory_start, memory_end)
            anchor_increment = self.ordered.anchor_output(anchor_music)
            anchor = anchor+anchor_increment
            if self.ordered.aligned and encoded_cache is not None and batch.get('phrase_sequence_shared',False):
                encoded = encoded_cache
            else:
                encoded = self.ordered.encode_sequences(model, batch, return_states=self.ordered.aligned)
            sequence,sequence_states = encoded[:2] if self.ordered.aligned else (encoded,None)
            projected_keys = encoded[2] if self.ordered.aligned and len(encoded)>2 else None
            indices = batch['phrase_sequence_inverse']
            sequence_bank = torch.cat((sequence.new_zeros(1, self.phrase_width), sequence), 0)
            sequence_values = sequence_bank[(indices+1).clamp_min(0)]
            paired = self.ordered.reference_fusion(torch.cat((reference_music, sequence_values), -1))
            reference_increment = self.ordered.reference_output(paired).masked_fill(~valid[..., None], 0.)
            references = references+reference_increment
            ordered_output = dict(phrase_ordered_anchor_residual_rms=anchor_increment.float().square().mean(-1).sqrt(),
                phrase_ordered_reference_residual_rms=reference_increment.float().square().mean((-1,-2)).sqrt())
        references = references.masked_fill(~valid[..., None], 0.)
        null = self.null_reference.expand(*anchor.shape[:-1], 1, anchor.size(-1))
        references = torch.cat((null, references), -2)
        style = self.style(batch['condition'].float())[:, None].expand_as(anchor)
        if hasattr(self,'ordered') and self.ordered.aligned:
            aligned_read,null_weight,projected_keys = self.ordered.aligned_read(
                batch,sequence_states,anchor,style,valid,projected_keys)
            ordered_output.update(phrase_aligned_read=aligned_read,
                                  phrase_aligned_null_weight=null_weight,
                                  _phrase_encoding_cache=(sequence,sequence_states,projected_keys))
        query = self.query(torch.cat((anchor, style), -1))
        scores = (query[..., None, :] * self.key(references)).sum(-1)/math.sqrt(anchor.size(-1))
        allowed = torch.cat((torch.ones_like(valid[..., :1]), valid), -1)
        scores = scores.masked_fill(~allowed, -torch.inf)
        weights = scores.float().softmax(-1).to(references)
        retrieved = (weights[..., None] * references).sum(-2)
        intent = self.intent(torch.cat((anchor, retrieved, style), -1))
        progress = self.progress(torch.nan_to_num(batch['phrase_progress'].float()))
        residual = self.intent_output(intent) + self.progress_output(progress) + self.style_output(style)
        # This attends to encoded acoustic subspaces, not separated instruments.
        strength = torch.as_tensor(tracking_strength, device=anchor.device, dtype=anchor.dtype)
        if strength.ndim == 1: strength = strength[:, None]
        if strength.ndim == 2: strength = strength[..., None]
        tracked = anchor * self.tracking(intent).sigmoid()
        residual = residual + strength * self.tracking_output(tracked)
        prediction = self.predict(intent)
        representation = self.shared(torch.cat((anchor, self.writing(prediction)), -1))
        if hasattr(self, 'ordered'):
            ordered_output.update(phrase_ordered_prediction=self.ordered.predict(intent), phrase_progress_repr=progress)
        return residual, dict(phrase_intent=intent, phrase_reference_weights=weights,
            phrase_prediction=prediction, phrase_repr=representation,
            phrase_reference_repr=references, phrase_reference_mask=allowed,
            phrase_residual_rms=residual.float().square().mean(-1).sqrt(), **ordered_output)

    @staticmethod
    def _candidate_view(batch):
        candidate = {key[len('candidate_'):]:value for key,value in batch.items()
                     if key.startswith('candidate_')}
        candidate['condition'] = batch['condition']
        candidate['step_mask'] = batch['step_mask'] if batch['step_mask'].shape == candidate['phrase_start_beat'].shape else batch['step_mask'][:,-1:]
        if 'step_beat' not in candidate:
            raise ValueError('candidate_step_beat is required')
        return candidate

    def at_candidate(self, model, cache, inverse, batch, tracking_strength=0., *, encoded_cache=None):
        """Evaluate only the phrase branch at the selected landing clock.

        The caller retains its departure-clock main hidden and timing output.
        Candidate inputs have an independent namespace so target clocks cannot
        silently replace timing inputs in the ordinary forward path.
        """
        if not hasattr(self,'ordered') or not self.ordered.aligned:
            raise ValueError('at_candidate requires aligned phrase context')
        return self.forward(model,cache,inverse,self._candidate_view(batch),tracking_strength,
                            encoded_cache=encoded_cache)

    def candidate_actor_residual(self, hidden, output, batch):
        """Apply the same actor branch at landing, without main trunk replay."""
        if not hasattr(self,'ordered') or not self.ordered.aligned:
            raise ValueError('candidate actor requires aligned phrase context')
        return self.ordered.actor_residual(hidden,output,self._candidate_view(batch))


def phrase_auxiliary_loss(output, batch):
    """One weighted feature loss; no duplicate event budget or recovery labels."""
    prediction = output['phrase_prediction'].float()
    if 'target_phrase_features' not in batch:
        return prediction.sum()*0.
    target = batch['target_phrase_features'].float()
    mask = batch['target_phrase_mask'].bool()
    weight = batch['target_phrase_weight'].float()
    if mask.ndim == prediction.ndim-1: mask = mask[..., None]
    if weight.ndim == prediction.ndim-1: weight = weight[..., None]
    active = mask & batch['step_mask'].bool()[..., None] & torch.isfinite(target)
    mass = torch.nan_to_num(weight, nan=0., posinf=0., neginf=0.).clamp_min(0).expand_as(prediction)
    mass = mass.masked_fill(~active, 0.)
    target = torch.nan_to_num(target).masked_fill(~active, 0.)
    errors = F.smooth_l1_loss(prediction, target, reduction='none')
    # Dataset weights already split a phrase's budget over its repeated queries.
    # Renormalizing by their sum would cancel that split (and chart balancing).
    return (errors * mass).sum()/max(1, prediction.size(0)*prediction.size(-1))


def ordered_phrase_auxiliary_loss(model, output, batch):
    """Cosine teacher and differentiable two-tower multi-positive matching.

    Repeated per-query targets retain their dataset-assigned budget. No target
    table is ever passed to the decoder or the forward reference encoder.
    """
    prediction = output['phrase_ordered_prediction'].float()
    zero = prediction.sum()*0.
    empty = dict(phrase_ordered_cosine_loss=zero, phrase_ordered_match_loss=zero,
                 phrase_ordered_query_match_loss=zero, phrase_ordered_target_match_loss=zero,
                 phrase_ordered_code_variance=zero)
    if 'target_phrase_sequence_inverse' not in batch or batch['target_phrase_sequence_kind'].size(0) == 0:
        return zero, empty
    labels = model.phrase_context.ordered.encode_sequences(model, batch, 'target_phrase_sequence_').float()
    labels = F.normalize(labels, dim=-1)
    mapping = batch['target_phrase_sequence_inverse']
    active = mapping.ge(0) & batch['target_phrase_mask'].bool() & batch['step_mask'].bool()
    mass = torch.nan_to_num(batch['target_phrase_weight'].float(), nan=0., posinf=0., neginf=0.).clamp_min(0)
    mass = mass.masked_fill(~active, 0.)
    target = mapping.clamp_min(0)
    normalized = F.normalize(prediction, dim=-1)
    cosine = 1-(normalized*labels.detach()[target]).sum(-1)
    scores = normalized.flatten(0, 1) @ labels.T / .1
    positive_ids = batch['target_phrase_sequence_positive_ids']
    positive = positive_ids[target.flatten(), None].eq(positive_ids[None])
    query_match = scores.logsumexp(-1)-scores.masked_fill(~positive, -torch.inf).logsumexp(-1)
    # Reverse retrieval excludes padded/zero-budget queries as negatives. An
    # all-masked batch and unused target columns receive finite dummy support,
    # then zero mass, avoiding inf*0 and all-inf logsumexp gradients.
    query_valid = active.flatten() & mass.flatten().gt(0)
    first_query = torch.arange(scores.size(0), device=scores.device).eq(0)
    safe_query = query_valid | (~query_valid.any() & first_query)
    reverse_positive = positive & query_valid[:, None]
    has_positive = reverse_positive.any(0)
    reverse_positive = reverse_positive | (~has_positive[None] & first_query[:, None])
    reverse_support = safe_query[:, None] | (~has_positive[None] & first_query[:, None])
    reverse_denominator = scores.masked_fill(~reverse_support, -torch.inf).logsumexp(0)
    reverse_numerator = scores.masked_fill(~reverse_positive, -torch.inf).logsumexp(0)
    reverse_match = (reverse_denominator-reverse_numerator).masked_fill(~has_positive, 0.)
    target_match = reverse_match[target.flatten()]
    query_match = query_match.reshape_as(mapping)/max(1., math.log(max(1, labels.size(0))))
    target_match = target_match.reshape_as(mapping)/torch.log(query_valid.float().sum().clamp_min(math.e))
    denominator = max(1, prediction.size(0))
    cosine_loss = (cosine*mass).sum()/denominator
    query_loss, target_loss = (query_match*mass).sum()/denominator, (target_match*mass).sum()/denominator
    match_loss = .5*(query_loss+target_loss)
    code_variance = labels.detach().var(0, unbiased=False).mean()
    return .5*(cosine_loss+match_loss), dict(phrase_ordered_cosine_loss=cosine_loss,
        phrase_ordered_match_loss=match_loss, phrase_ordered_query_match_loss=query_loss,
        phrase_ordered_target_match_loss=target_loss, phrase_ordered_code_variance=code_variance)
