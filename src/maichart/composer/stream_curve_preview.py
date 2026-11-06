"""Detached count readout for the editor, independent of arrival decisions."""
import math

import torch
from torch.nn import functional as F

PREVIEW_BIN_SECONDS=2.


def preview_count_loss(logits,batch,rows,*,audio_grid_seconds):
    """Aggregate native cells once into approximately two-second exposures.

    Whole cells belong to their left-edge bin. A cell straddling a nominal
    bin boundary is never split into two copies of the same event count.
    Both counts and their real exposure follow that same ownership rule.
    """
    target=batch['audio_plan_target'].index_select(0,rows)[...,1:3].float()
    exposure=batch['audio_plan_exposure'].index_select(0,rows).float()
    trusted=batch['audio_plan_weight'].index_select(0,rows).gt(0)
    exposure=exposure*trusted
    target=target*trusted[...,None]
    inverse=batch['song_inverse'].index_select(0,rows)
    seconds=batch['song_seconds'].index_select(0,inverse)
    bins=max(1,math.ceil(logits.size(1)*float(audio_grid_seconds)/PREVIEW_BIN_SECONDS)+1)
    owner=(seconds/PREVIEW_BIN_SECONDS).floor().long().clamp(0,bins-1)
    index=owner[...,None].expand(-1,-1,2)
    expected=logits.new_zeros(logits.size(0),bins,2,dtype=torch.float32).scatter_add(
        1,index,F.softplus(logits.float())*exposure[...,None])
    counts=target.new_zeros(target.size(0),bins,2).scatter_add(1,index,target)
    widths=exposure.new_zeros(exposure.size(0),bins).scatter_add(1,owner,exposure)
    nll=expected-counts*expected.clamp_min(1e-8).log()+torch.lgamma(counts+1.)
    nll=nll.masked_fill(widths.le(0)[...,None],0.)
    per_chart=nll.sum(1)/widths.sum(1).clamp_min(1e-8)[:,None]
    mass=batch.get('unified_profile_chart_weights',batch.get('profile_chart_weights'))
    if mass is None:mass=per_chart.new_full((len(rows),),1/max(1,len(rows)))
    mass=mass.to(per_chart)
    loss=(per_chart.sum(-1)*mass).sum()
    return loss,dict(curve_preview_head_nll=(per_chart[:,0]*mass).sum(),
        curve_preview_event_nll=(per_chart[:,1]*mass).sum(),
        curve_preview_exposure_seconds=(widths.sum(1)*mass).sum(),
        curve_preview_target_heads=(counts[...,0].sum(1)*mass).sum(),
        curve_preview_target_events=(counts[...,1].sum(1)*mass).sum())
