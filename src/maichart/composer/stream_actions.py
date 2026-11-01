"""Semantic event probabilities, independent of spelling/catalog cardinality.

The five bits use EventMetadata.type_counts order: Tap, Slide, Hold, Touch,
TouchHold.  They describe the complete event, not a generation quota.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from .events import event_metadata


TYPE_BITS = {'tap': 1, 'slide': 2, 'hold': 4, 'touch': 8, 'touch_hold': 16}


def semantic_catalog(codec):
    atoms = [0]
    for text in codec.atom_catalog:
        atoms.append(sum(1 << i for i, count in enumerate(event_metadata(text).type_counts) if count))
    shapes = [0]
    for shape in codec.shape_catalog[1:]:
        value = 0
        for kind, _ in shape:
            value |= TYPE_BITS[kind]
        shapes.append(value)
    return torch.tensor(atoms, dtype=torch.long), torch.tensor(shapes, dtype=torch.long)


def grouped_event_log_probs(route_scores, family_scores, route_families, supported, *, temperature=1.):
    """P(type set) * P(regular shape or literal route | type set).

    A positive detail temperature changes only the conditional routes.  The
    semantic family probabilities stay at their calibrated training scale.
    This avoids a large fallback bucket gaining mass merely because the
    ordinary event alternatives are split into more shapes.
    """
    raw = route_scores.float()
    if temperature > 0:
        raw = raw / float(temperature)
    groups = (route_families - 1).expand_as(raw)
    shape = (*raw.shape[:-1], 31)
    maxima = raw.new_full(shape, -torch.inf).scatter_reduce(-1, groups, raw, reduce='amax', include_self=True)
    safe_maxima = maxima.masked_fill(~torch.isfinite(maxima), 0.)
    sums = raw.new_zeros(shape).scatter_add(-1, groups, (raw-safe_maxima.gather(-1, groups)).exp())
    lse = safe_maxima + sums.clamp_min(torch.finfo(raw.dtype).tiny).log()
    legal = sums.gt(0) & supported
    family_raw=family_scores.float().masked_fill(~legal,-torch.inf)
    normalizer=family_raw.logsumexp(-1,keepdim=True)
    family=family_raw-normalizer.masked_fill(~torch.isfinite(normalizer),0.)
    return raw-lse.gather(-1, groups)+family.gather(-1, groups)


def family_preserving_template_log_probs(base, keep_mask, route_families, correction, *, residual=True):
    """Repartition template retention within its family, preserving family mass.

    A reference template identifies at most one semantic family. Every other
    family keeps its exact joint probability, including under a large keep
    residual. The same joint distribution serves training and generation.
    """
    from .stream_transition import residual_keep_logit, two_group_log_probs

    scores=base.float()
    keep=torch.as_tensor(keep_mask,device=scores.device,dtype=torch.bool).expand_as(scores)
    family=(keep.long()*route_families).amax(-1)
    same=route_families.eq(family[...,None]) & family.gt(0)[...,None]
    valid=torch.isfinite(scores)&scores.gt(-1e4)
    supported=same&valid
    present=supported.any(-1)
    selected=scores.masked_fill(~supported,-torch.inf)
    safe_mass=torch.where(present[...,None],selected,torch.zeros_like(selected)).logsumexp(-1)
    gate=(residual_keep_logit(scores,keep,correction,gain=8.,allowed=same)
          if residual else correction)
    conditional=two_group_log_probs(scores,keep,gate,allowed=same)
    return torch.where(supported,conditional+safe_mass[...,None],scores)


def family_masks_from_atoms(atom_ids, atom_families):
    """Union observed atom types without adding future information to inputs."""
    ids = atom_ids.clamp(0, atom_families.numel()-1)
    values = atom_families[ids] * atom_ids.gt(0)
    bits = values.new_tensor([1, 2, 4, 8, 16])
    return ((values[..., None] & bits).ne(0).any(-2).long()*bits).sum(-1)


def prefix_union(values, bits):
    """Bitwise union of tokens strictly before each teacher-forced slot."""
    previous = torch.cat((values.new_zeros(*values.shape[:-1], 1), values[..., :-1]), -1)
    present = (previous[..., None] & bits).ne(0).long().cumsum(-2).gt(0)
    return (present.long()*bits).sum(-1)


def refresh_family_log_probs(audio_logits,continuity_logits,previous_family,has_previous,*,detach_audio=False):
    """Shared retention plus a fresh audio draw that includes the last type.

    For fixed p and family-independent rho, K=rho*I+(1-rho)*1*p has p as a
    stationary marginal. Changing music instead gives a lagged mixture; this
    does not guarantee instantaneous/finite-song class proportions.

    Retention is a latent mixture factor, not a directly supervised 'same'
    label: fresh draws can also equal the previous family.
    """
    fresh=audio_logits.float().log_softmax(-1)
    if detach_audio:fresh=fresh.detach()
    gate=continuity_logits.float()
    previous=F.one_hot((previous_family-1).clamp(0,audio_logits.size(-1)-1),audio_logits.size(-1)).bool()
    copy=F.logsigmoid(gate)[...,None].expand_as(fresh).masked_fill(~previous,-torch.inf)
    refresh=F.logsigmoid(-gate)[...,None]+fresh
    mixture=torch.logaddexp(copy,refresh)
    return torch.where(has_previous[...,None],mixture,fresh)


class StatePhraseFamilyResidual(torch.nn.Module):
    """Small candidate-conditioned correction; no previous-note identity input.

    STATE50 is sampled at the causal cursor. Its bounded log-seconds fields
    are advanced to the actual candidate, rather than treating cursor occupancy
    as candidate occupancy. Spatial slide pads are deliberately not extrapolated.
    """
    def __init__(self, model_width, phrase_width):
        super().__init__()
        self.phrase_width=phrase_width
        width=min(model_width,128)
        self.features=torch.nn.Sequential(torch.nn.Linear(model_width+phrase_width+11,width),
            torch.nn.GELU(),torch.nn.LayerNorm(width))
        self.output=torch.nn.Linear(width,32)
        torch.nn.init.zeros_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)

    @staticmethod
    def candidate_state(batch,event_seconds):
        import math
        gap=(event_seconds.double()-batch['step_second'].double()).clamp_min(0).float()
        exact='candidate_action_state' in batch
        state=batch.get('candidate_action_state',batch.get('action_state'))
        if state is None:
            return gap.new_zeros(*gap.shape,9)
        state=torch.nan_to_num(state.float()).clamp(0,1)
        valid=~batch['candidate_action_state_mask' if exact else 'action_state_mask'].bool()
        # STATE50: types 0:4; waiting 45; remaining 46; until-trace 47.
        scale=math.log1p(60.)
        shift=0. if exact else gap[...,None]
        remaining=(torch.expm1(state[...,46]*scale)-shift).clamp_min(0)
        until=(torch.expm1(state[...,47]*scale)-shift).clamp_min(0)
        valid=valid & remaining.gt(0)
        count=valid.sum(-1).float()
        norm=count.clamp_min(1)
        remaining=remaining.log1p()/scale
        until=until.log1p()/scale
        types=(state[...,:4]*valid[...,None]).sum(-2)/norm[...,None]
        avg=(remaining*valid).sum(-1)/norm
        maximum=remaining.masked_fill(~valid,0).amax(-1)
        waiting=(state[...,45].gt(0)&until.gt(0)&valid).sum(-1)/norm
        until_avg=(until*valid).sum(-1)/norm
        return torch.cat((types,(count.log1p()/math.log1p(32.))[...,None],
            avg[...,None],maximum[...,None],waiting[...,None],until_avg[...,None]),-1)

    def forward(self,music,event_seconds,event_beats,batch,phrase_output=None):
        intent=None if phrase_output is None else phrase_output.get('phrase_intent')
        if intent is None:intent=music.new_zeros(*music.shape[:-1],self.phrase_width)
        # Intent is the stable phrase plan, not the autoregressive decoder hidden.
        state=self.candidate_state(batch,event_seconds).to(music).expand(*music.shape[:-1],9)
        intent=intent.expand(*music.shape[:-1],self.phrase_width)
        gap_seconds=(event_seconds.double()-batch['step_second'].double()).clamp_min(0).log1p().float()
        gap_beats=(event_beats.double()-batch['step_beat'].double()).clamp_min(0).log1p().float()
        values=torch.cat((music,intent.to(music),state,gap_seconds[...,None].to(music),
            gap_beats[...,None].to(music)),-1)
        return self.output(self.features(values)).float()
