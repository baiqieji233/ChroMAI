"""Inference-only protection against symbolic Touch probability feedback.

This does not change model parameters, training targets or recovery sampling.
The audio branch is a learned reference, not a promised musical ground truth.
"""
import math
import torch
from torch.nn import functional as F

TOUCH_HISTORY_GUIDANCE_POLICY='safe_audio_anchored_touch_odds_v2'
TOUCH_HISTORY_LOG_ODDS_BOUND=1.


def constrain_touch_history(full_logits,audio_logits,touch_mask,*,log_odds_bound=TOUCH_HISTORY_LOG_ODDS_BOUND):
    """Cap positive Touch/nonTouch log-odds amplification before legal masks.

    Preserve STOP probability, total nonSTOP mass and within-group relative
    probabilities. Lower Touch predictions are unchanged. Both paths must use
    the same candidate and same within-event prefix. Degenerate/absent groups
    retain the original logits. Physical masks applied later remain authoritative.
    """
    if (full_logits.ndim!=1 or full_logits.shape!=audio_logits.shape
            or full_logits.shape!=touch_mask.shape or not full_logits.numel()):
        raise ValueError('Touch guidance requires matching nonempty 1D logits and mask')
    bound=float(log_odds_bound)
    if not math.isfinite(bound) or bound<0:
        raise ValueError('Touch history log-odds bound must be finite and nonnegative')
    full=full_logits.float();audio=audio_logits.to(device=full.device,dtype=torch.float32)
    touch=touch_mask.to(device=full.device,dtype=torch.bool).clone();touch[0]=False
    non_touch=~touch;non_touch[0]=False
    ft=torch.logsumexp(full[touch],0);fn=torch.logsumexp(full[non_touch],0)
    at=torch.logsumexp(audio[touch],0);an=torch.logsumexp(audio[non_touch],0)
    total=torch.logaddexp(ft,fn)
    odds=ft-fn;reference=at-an;target=torch.minimum(odds,reference+bound)
    t_shift=total-F.softplus(-target)-ft
    n_shift=total-F.softplus(target)-fn
    corrected=torch.where(touch,full+t_shift,torch.where(non_touch,full+n_shift,full))
    valid=torch.isfinite(torch.stack((ft,fn,at,an))).all()&(odds>reference+bound)
    return torch.where(valid,corrected.to(full_logits.dtype),full_logits)


def action_logits_with_touch_guard(model,context,slot,selected_ids,*,audio_context=None,
                                  history_strength=1.,enabled=False,touch_mask=None,diagnostics=None,
                                  log_odds_bound=TOUCH_HISTORY_LOG_ODDS_BOUND,previous_heads=None):
    extra={} if previous_heads is None else {'previous_heads':previous_heads}
    full=model.action_slot(context,slot,selected_ids,audio_context=audio_context,history_strength=history_strength,**extra)
    if not enabled or audio_context is None:
        return full
    if touch_mask is None:
        raise ValueError('Touch guidance requires its catalog mask')
    reference=model.action_slot(audio_context,slot,selected_ids)
    corrected=constrain_touch_history(full,reference,touch_mask,log_odds_bound=log_odds_bound)
    if diagnostics is not None and not torch.equal(corrected,full):
        diagnostics['applied']=True
    return corrected
