"""Versioned short-rollout contract; generation controls remain independent."""
import math
from . import ACTION_SAMPLING_POLICY,LEGACY_ACTION_SAMPLING_POLICY

RECOVERY_SAMPLING_POLICIES={ACTION_SAMPLING_POLICY,LEGACY_ACTION_SAMPLING_POLICY}
RETRY_POLICIES={'legacy_first_atom_v1','failed_prefix_v2'}


def resolve_recovery_sampling(requested='auto',parent=None,*,resume=False,architecture_profile='audio_grounded_v3'):
    if not isinstance(requested,str) or requested not in {'auto',*RECOVERY_SAMPLING_POLICIES}:
        raise ValueError('Unknown recovery_sampling_policy')
    inherited=(parent or {}).get('action_sampling_policy') or LEGACY_ACTION_SAMPLING_POLICY
    if resume and inherited not in RECOVERY_SAMPLING_POLICIES:raise ValueError('Unsupported parent recovery sampling policy')
    default=ACTION_SAMPLING_POLICY if architecture_profile=='audio_grounded_v3' else LEGACY_ACTION_SAMPLING_POLICY
    resolved=(inherited if resume else default) if requested=='auto' else requested
    if resume and resolved!=inherited:raise ValueError('Resume inherits recovery_sampling_policy; use finetune')
    if architecture_profile!='audio_grounded_v3' and resolved!=LEGACY_ACTION_SAMPLING_POLICY:
        raise ValueError('Aligned recovery sampling requires audio_grounded_v3')
    return resolved


def recovery_sampling_settings(policy,pointer_temperature=1.,action_temperature=.85):
    if policy not in RECOVERY_SAMPLING_POLICIES:raise ValueError('Unknown recovery_sampling_policy')
    if any(not math.isfinite(float(v)) or not 0<=float(v)<=2 for v in (pointer_temperature,action_temperature)):
        raise ValueError('Recovery temperatures must be finite and in [0,2]')
    aligned=policy==ACTION_SAMPLING_POLICY
    if not aligned and (float(pointer_temperature)!=1. or float(action_temperature)!=.85):
        raise ValueError('Legacy recovery temperatures are fixed; select the aligned recovery policy')
    return {'policy':policy,'pointer_temperature':float(pointer_temperature),'action_temperature':float(action_temperature),
        'pointer_sampler':'grouped' if aligned else 'global',
        'touch_history_guard':aligned and float(action_temperature)>0,'touch_history_bound':1.,
        'action_type_temperature':1.,'action_type_min_p':0.,'min_same_head_ms':50.,
        'retry_policy':'failed_prefix_v2' if aligned else 'legacy_first_atom_v1'}
