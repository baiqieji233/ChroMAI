"""Observed future evidence, not an event hazard or a coverage guarantee."""
from __future__ import annotations

import torch

FUTURE_GAP_EVIDENCE_POLICY = 'observed_max_presence_attack_support_7bins_v1'
FUTURE_GAP_EVIDENCE_DIM = 21


def future_gap_evidence(offsets, valid, placement_probs, attack):
    """Return [B,C,21]: max presence, max attack and support per gap bin.

    The final (>4 beat) bin describes only observed candidates. Support means
    at least one observed candidate, never complete coverage of an interval.
    Inputs are detached so this conditioning creates no placement shortcut.
    Only scalar [B,C,C] pair tensors are constructed, with no feature axis.
    """
    if offsets.ndim != 2 or valid.shape != offsets.shape:
        raise ValueError('Future gap offsets and valid must have shape [B,C]')
    if placement_probs.shape != (*offsets.shape, 5) or attack.shape != offsets.shape:
        raise ValueError('Future gap evidence requires placement [B,C,5] and attack [B,C]')
    offsets = offsets.detach().float()
    valid = valid.bool()
    # Unknown/padded entries are removed before arithmetic, including NaNs.
    safe_offsets = torch.where(valid, offsets, torch.zeros_like(offsets))
    delta = safe_offsets[:, None, :] - safe_offsets[:, :, None]
    pairs = valid[:, :, None] & valid[:, None, :] & delta.gt(0.)
    edges = offsets.new_tensor((.125, .25, .5, 1., 2., 4.))
    bins = torch.bucketize(delta.contiguous(), edges, right=False)
    # Channel 0 is trained on event presence. A noisy type/modifier prediction
    # must not manufacture stronger onset evidence by taking max over kinds.
    presence = placement_probs.detach().float()[..., 0]
    attack = attack.detach().float()
    features = []
    for values in (presence, attack, torch.ones_like(offsets)):
        source = torch.where(pairs, values[:, None, :], torch.zeros_like(delta))
        pooled = offsets.new_zeros((*offsets.shape, 7))
        pooled.scatter_reduce_(-1, bins, source, reduce='amax', include_self=True)
        features.append(pooled)
    # Bin-major ordering: [presence, attack, has_support] for each gap class.
    return torch.stack(features, -1).flatten(-2)
