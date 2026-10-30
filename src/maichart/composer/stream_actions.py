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
