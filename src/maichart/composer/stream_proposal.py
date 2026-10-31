"""Bounded, shared time/action retries without rerunning the audio encoder."""
from __future__ import annotations

from .stream_sampling import sample_logits, sample_guarded_event,delta_sampling_logits
from .stream_timing import temper_grid_logits
from .stream_transition import sample_two_group


def sample_timed_event(model, timing_logits, deltas, cursor, action_context,
                       reference_phase, reference_phase_mask, guard, generator,
                       *, pointer_temperature=.8, action_temperature=.8,
                       time_attempts=3, action_attempts=8, accepts_extra=None,
                       motion_phase_distribution=None,repeat_delta_id=-1,timing_grid_ids=None,timing_group_count=None,
                       timing_keep_mask=None):
    """Try distinct onset times before declaring a forced WAIT.

    ``action_context(delta_id, beat)`` reads the already encoded audio at the
    chosen time. All failed proposals leave the committed prefix untouched.
    Generation and recovery deliberately share the same bounded search.
    """
    if not 1 <= time_attempts <= 3:
        raise ValueError('Time attempts must be between 1 and 3')
    scores=timing_logits.clone()
    failed_config=failed_timing=0
    for _ in range(time_attempts):
        if timing_keep_mask is not None:
            keep=timing_keep_mask
            if keep.ndim==3 and keep.size(0)==1:keep=keep[0,-1]
            delta_id=sample_two_group(scores,keep,temperature=pointer_temperature,generator=generator,
                within_group_ids=timing_grid_ids,within_group_count=timing_group_count)
        else:
            tempered=(temper_grid_logits(scores,pointer_temperature,timing_grid_ids,timing_group_count) if timing_grid_ids is not None
                      else delta_sampling_logits(scores,pointer_temperature,repeat_delta_id))
            delta_id=sample_logits(tempered,0. if pointer_temperature<=0 else 1.,generator)
        if delta_id is None:break
        beat=cursor+deltas[delta_id]
        action=action_context(delta_id,beat)
        context=action['context'] if isinstance(action,dict) else action
        load=action.get('button_load') if isinstance(action,dict) else None
        family_options=({key:action[key] for key in ('family_context','family_logits','motion_keep_logit','motion_reference_shape','template_keep_logit','template_reference_shape') if key in action} if isinstance(action,dict) else {})
        point,failed=sample_guarded_event(model,context,
            reference_phase,beat,guard,generator,temperature=action_temperature,
            attempts=action_attempts,accepts_extra=accepts_extra,
            reference_phase_mask=reference_phase_mask,
            motion_phase_distribution=motion_phase_distribution,button_load=load,**family_options)
        failed_config+=failed
        if point is not None:return point,delta_id,failed_config,failed_timing
        scores[delta_id]=-float('inf')
        failed_timing+=1
    return None,None,failed_config,failed_timing
