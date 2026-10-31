"""Learned continue/exit decisions with explicit probability conservation.

The event alternatives remain an ordinary conditional distribution inside
each group. Its number of alternatives must not change the group's mass or
the greedy decision to continue versus leave a pattern.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

MASKED_LOGIT = -1e4


def _float_scores(base):
    """Promote logits while preserving the finite training-mask contract.

    BF16 stores -10000 as -9984. A plain FP32 cast would make that masked
    value larger than the legal-support threshold. Restore exactly that
    sentinel on device; ordinary values and their gradients are unchanged.
    """
    scores=base.float()
    if base.dtype==torch.bfloat16:
        scores=scores.masked_fill(scores.eq(-9984.),MASKED_LOGIT)
    return scores


def _support(base, allowed=None):
    valid=torch.isfinite(base)&base.gt(MASKED_LOGIT)
    if allowed is not None:valid=valid&torch.as_tensor(allowed,device=base.device,dtype=torch.bool)
    return valid


def _conditional_log_probs(scores,mask):
    present=mask.any(-1)
    masked=scores.masked_fill(~mask,-torch.inf)
    # All-empty rows need a finite, input-independent normalizer for backward.
    safe=torch.where(present[...,None],masked,torch.zeros_like(masked))
    return scores-safe.logsumexp(-1,keepdim=True),present


def residual_keep_logit(base,keep_mask,correction,gain=8.,allowed=None):
    """Keep the base odds and learn a residual in log-odds space.

    The base group masses remain in the gradient graph. Therefore zero
    correction restores both the original probability and its learning path
    when this value is passed to two_group_log_probs. Positive correction
    increases retention, negative correction increases exit probability.

    A single supported group is already deterministic; empty/single-group
    rows return a finite zero gate and have no correction gradient. Masks
    and leading correction dimensions follow two_group_log_probs.
    """
    scores=_float_scores(base)
    valid=_support(scores,allowed)
    keep=torch.as_tensor(keep_mask,device=scores.device,dtype=torch.bool).expand_as(scores)&valid
    leave=(~keep)&valid
    def log_mass(mask):
        present=mask.any(-1)
        values=scores.masked_fill(~mask,-torch.inf)
        safe=torch.where(present[...,None],values,torch.zeros_like(values))
        return safe.logsumexp(-1),present
    keep_mass,has_keep=log_mass(keep)
    leave_mass,has_leave=log_mass(leave)
    residual=torch.as_tensor(correction,device=scores.device).float()
    if residual.ndim==scores.ndim and residual.size(-1)==1:residual=residual.squeeze(-1)
    residual=residual.expand(scores.shape[:-1])
    gate=keep_mass-leave_mass+float(gain)*residual
    return torch.where(has_keep&has_leave,gate,torch.zeros_like(gate))


def two_group_log_probs(base,keep_mask,keep_logit,*,allowed=None):
    """Return P(group) * P(alternative | group), in FP32.

    ``base`` is (..., C); boolean ``keep_mask`` and ``allowed`` broadcast to
    that shape. ``keep_logit`` broadcasts to the leading dimensions (a final
    singleton is also accepted). Positive logit increases continuation mass.

    Empty groups give their entire mass to the other supported group. With
    no legal alternatives the row retains its input mask. Both -inf and the
    finite -1e4 training sentinel are recognized and preserved, so padding
    does not create nonfinite values in an otherwise finite training output.
    """
    scores=_float_scores(base)
    keep=torch.as_tensor(keep_mask,device=scores.device,dtype=torch.bool).expand_as(scores)
    valid=_support(scores,allowed)
    keep=keep&valid
    leave=(~keep)&valid
    within_keep,has_keep=_conditional_log_probs(scores,keep)
    within_leave,has_leave=_conditional_log_probs(scores,leave)
    gate=torch.as_tensor(keep_logit,device=scores.device).float()
    if gate.ndim==scores.ndim and gate.size(-1)==1:gate=gate.squeeze(-1)
    gate=gate.expand(scores.shape[:-1])
    both=has_keep&has_leave
    keep_mass=torch.where(both,F.logsigmoid(gate),torch.zeros_like(gate))
    leave_mass=torch.where(both,F.logsigmoid(-gate),torch.zeros_like(gate))
    result=torch.where(keep,within_keep+keep_mass[...,None],within_leave+leave_mass[...,None])
    # Preserve -inf inference masks and finite training masks. An additional
    # explicit support restriction uses the caller's existing sentinel style.
    original_mask=torch.where(torch.isfinite(scores),scores,torch.full_like(scores,-torch.inf))
    if allowed is not None:
        extra=~torch.as_tensor(allowed,device=scores.device,dtype=torch.bool).expand_as(scores)
        original_mask=torch.where(extra&torch.isfinite(scores),torch.full_like(scores,MASKED_LOGIT),original_mask)
    return torch.where(valid,result,original_mask)


def sample_two_group(log_probs,keep_mask,temperature=1.,generator=None,*,allowed=None,
                     within_group_ids=None,within_group_count=None):
    """Select one macro group first, then one alternative inside it.

    A single row (C or 1xC) is accepted. At zero temperature both decisions
    are greedy. At positive temperature the learned group mass stays at T=1
    and only within-group alternatives are tempered. Optional static IDs add
    a nested grid decision inside the chosen binary group: grid mass and
    within-grid alternatives use T, preserving the existing grid contract.
    At zero temperature this becomes binary -> grid -> candidate argmax.
    Rows without supported entries return None without consuming RNG. Input
    tensors are never mutated.
    """
    scores=_float_scores(log_probs)
    if scores.ndim<1 or scores.numel()!=scores.size(-1):
        raise ValueError('Sample one transition row at a time')
    scores=scores.reshape(-1)
    grid_ids=None
    if within_group_ids is not None:
        if within_group_count is None or int(within_group_count)<1:
            raise ValueError('Nested transition grids require a positive static group count')
        grid_ids=torch.as_tensor(within_group_ids,device=scores.device,dtype=torch.long)
        if grid_ids.ndim==3 and grid_ids.size(0)==1:grid_ids=grid_ids[0,-1]
        grid_ids=grid_ids.reshape(-1)
        if grid_ids.numel()!=scores.numel():raise ValueError('Nested grid IDs have the wrong width')
    mask=torch.as_tensor(keep_mask,device=scores.device,dtype=torch.bool).reshape(-1)
    if mask.numel()!=scores.numel():raise ValueError('Transition mask has the wrong width')
    valid=_support(scores,None if allowed is None else torch.as_tensor(allowed,device=scores.device).reshape(-1))
    if not bool(valid.any()):return None
    keep=mask&valid;leave=(~mask)&valid
    groups=torch.stack((scores.masked_fill(~keep,-torch.inf).logsumexp(-1),
                        scores.masked_fill(~leave,-torch.inf).logsumexp(-1)))
    if temperature<=0:group=int(groups.argmax())
    else:group=int(torch.multinomial(groups.softmax(-1),1,generator=generator))
    chosen=keep if group==0 else leave
    values=scores.masked_fill(~chosen,-torch.inf)
    if grid_ids is not None:
        from .stream_timing import grouped_logsumexp
        grid_scores=grouped_logsumexp(values,grid_ids,groups=int(within_group_count))
        if temperature<=0:grid=int(grid_scores.argmax())
        else:grid=int(torch.multinomial((grid_scores/float(temperature)).softmax(-1),1,generator=generator))
        values=values.masked_fill(grid_ids.ne(grid),-torch.inf)
    if temperature<=0:return int(values.argmax())
    return int(torch.multinomial((values/float(temperature)).softmax(-1),1,generator=generator))
