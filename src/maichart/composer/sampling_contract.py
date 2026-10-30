"""Versioned short-rollout contract; generation controls remain independent."""
import math
from . import (ACTION_SAMPLING_POLICY,MASKED_TOUCH_ACTION_SAMPLING_POLICY,PREVIOUS_ACTION_SAMPLING_POLICY,
               LEGACY_ACTION_SAMPLING_POLICY)

MICROSTEP_ACTION_SAMPLING_POLICY='generation_matched_recovery_microstep_tails_v6'
RECOVERY_SAMPLING_POLICIES={MICROSTEP_ACTION_SAMPLING_POLICY,ACTION_SAMPLING_POLICY,MASKED_TOUCH_ACTION_SAMPLING_POLICY,
                            PREVIOUS_ACTION_SAMPLING_POLICY,LEGACY_ACTION_SAMPLING_POLICY}
# Policies whose rollouts use the generation-matched grouped sampler and prefix retry.
ALIGNED_RECOVERY_SAMPLING_POLICIES=(MICROSTEP_ACTION_SAMPLING_POLICY,ACTION_SAMPLING_POLICY,MASKED_TOUCH_ACTION_SAMPLING_POLICY,
                                    PREVIOUS_ACTION_SAMPLING_POLICY)
RETRY_POLICIES={'legacy_first_atom_v1','failed_prefix_v2'}
# Generation-time sampler policies (shared by generate, recovery and the WebUI).
POINTER_GRID_POLICY='grid_family_coherent_v3'
POINTER_GRID_POLICIES=('grid_family_coherent_v3','grid_family_coherent_v2','grid_level_mass_v1','off')
REPETITION_GUARD_POLICY='official_tail_v2'
REPETITION_GUARD_POLICIES=('official_tail_v2','legacy_v1','off')


def resolve_recovery_sampling(requested='auto',parent=None,*,resume=False,architecture_profile='audio_grounded_v3'):
    if not isinstance(requested,str) or requested not in {'auto',*RECOVERY_SAMPLING_POLICIES}:
        raise ValueError('Unknown recovery_sampling_policy')
    inherited=(parent or {}).get('action_sampling_policy') or LEGACY_ACTION_SAMPLING_POLICY
    if resume and inherited not in RECOVERY_SAMPLING_POLICIES:raise ValueError('Unsupported parent recovery sampling policy')
    default=MICROSTEP_ACTION_SAMPLING_POLICY if architecture_profile=='audio_grounded_v3' else LEGACY_ACTION_SAMPLING_POLICY
    resolved=(inherited if resume else default) if requested=='auto' else requested
    if resume and resolved!=inherited:raise ValueError('Resume inherits recovery_sampling_policy; use finetune')
    if architecture_profile!='audio_grounded_v3' and resolved!=LEGACY_ACTION_SAMPLING_POLICY:
        raise ValueError('Aligned recovery sampling requires audio_grounded_v3')
    return resolved


def recovery_sampling_settings(policy,pointer_temperature=1.,action_temperature=.85):
    if policy not in RECOVERY_SAMPLING_POLICIES:raise ValueError('Unknown recovery_sampling_policy')
    if any(not math.isfinite(float(v)) or not 0<=float(v)<=2 for v in (pointer_temperature,action_temperature)):
        raise ValueError('Recovery temperatures must be finite and in [0,2]')
    aligned=policy in ALIGNED_RECOVERY_SAMPLING_POLICIES
    musical=policy in (MICROSTEP_ACTION_SAMPLING_POLICY,ACTION_SAMPLING_POLICY)
    if not aligned and (float(pointer_temperature)!=1. or float(action_temperature)!=.85):
        raise ValueError('Legacy recovery temperatures are fixed; select the aligned recovery policy')
    return {'policy':policy,'pointer_temperature':float(pointer_temperature),'action_temperature':float(action_temperature),
        'pointer_sampler':'grouped' if aligned else 'global',
        'touch_history_guard':aligned and float(action_temperature)>0,'touch_history_bound':1.,
        'touch_legal_mask':policy in (MICROSTEP_ACTION_SAMPLING_POLICY,ACTION_SAMPLING_POLICY,MASKED_TOUCH_ACTION_SAMPLING_POLICY),
        'pointer_grid_policy':(POINTER_GRID_POLICY if policy==MICROSTEP_ACTION_SAMPLING_POLICY else 'grid_family_coherent_v2') if musical else 'off',
        'repetition_guard':REPETITION_GUARD_POLICY if musical else 'legacy_v1',
        'action_type_temperature':1.,'action_type_min_p':0.,'min_same_head_ms':50.,
        'retry_policy':'failed_prefix_v2' if aligned else 'legacy_first_atom_v1'}
