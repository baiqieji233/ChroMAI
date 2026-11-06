"""One discrete first-arrival law, including the censored no-event tail.

The public conditional NOTE timing and NOTE/WAIT decision are factorizations
of these same masses, not independently trainable vetoes.
"""
import torch
from torch.nn import functional as F


def first_arrival(log_rates, seconds, start, stop, allowed):
    """Sorted candidate cells with real seconds exposure; no Python time loop.

    A legal zero-beat BOS onset owns the first positive cell, not a second
    zero-width atom. PAD/duplicate/illegal candidates own no exposure.
    """
    seconds=seconds.double().expand_as(log_rates)
    start=start.double().expand(log_rates.shape[:-1]);stop=stop.double().expand(log_rates.shape[:-1]).maximum(start)
    valid=allowed.bool() & (seconds>=start[...,None]-1e-9) & (seconds<=stop[...,None]+1e-9)
    seconds=seconds.minimum(stop[...,None]).maximum(start[...,None])
    # Absolute phase catalogs wrap around the current beat, so their stored
    # order is NOT chronological. Keep vocabulary IDs intact after integration.
    order=seconds.masked_fill(~valid,torch.inf).argsort(dim=-1,stable=True)
    seconds=seconds.gather(-1,order)
    log_rates=log_rates.gather(-1,order)
    valid=valid.gather(-1,order)
    # Midpoint cells partition the complete observable future window. The
    # candidate axis is chronological (AbsoluteTimingHead contract).
    future=seconds.masked_fill(~valid,torch.inf)
    following=torch.cat((future[...,1:],torch.full_like(stop[...,None],torch.inf)),-1)
    right=torch.where(torch.isfinite(following),.5*(seconds+following),stop[...,None])
    right=torch.minimum(right,stop[...,None])
    right=right.masked_fill(~valid,-torch.inf).cummax(-1).values
    previous=torch.cat((start[...,None],right[...,:-1]),-1)
    previous=torch.where(torch.isfinite(previous),previous,start[...,None])
    width=(right-previous).clamp_min(0).masked_fill(~valid,0.)
    mass=F.softplus(log_rates.float()).double()*width
    integral=mass.sum(-1)
    before=mass.cumsum(-1)-mass
    # log(1-exp(-x)) with a finite dummy for masked rows.
    onset=(-torch.expm1(-mass)).clamp_min(1e-30).log()-before
    onset=onset.masked_fill(~valid,-1e4)
    note=(-torch.expm1(-integral)).clamp_min(1e-30).log()
    conditional=onset-note[...,None]
    conditional=conditional.masked_fill(~valid,-1e4)
    return dict(log_note=note.float(),log_tail=(-integral).float(),
                conditional=torch.empty_like(conditional).scatter(-1,order,conditional).float(),
                joint=torch.empty_like(onset).scatter(-1,order,onset).float(),
                expected_hazard=integral.float(),widths=torch.empty_like(width).scatter(-1,order,width))
