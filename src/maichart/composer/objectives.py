"""Versioned action learning objectives and comparable configuration metrics."""
from __future__ import annotations

import torch
from torch.nn import functional as F

from . import GROUNDED_TRAINING_SEMANTICS, TRAINING_SEMANTICS

LEGACY_ACTION_OBJECTIVE = 'event_balanced_v1'
JOINT_ACTION_OBJECTIVE = 'joint_event_nll_v2'
RECOVERY_ACTION_OBJECTIVE = 'joint_event_recovery_v3'
ACTION_OBJECTIVES = {LEGACY_ACTION_OBJECTIVE, JOINT_ACTION_OBJECTIVE, RECOVERY_ACTION_OBJECTIVE}
JOINT_TRAINING_SEMANTICS = 'chart_composer_training_v12_joint_event_likelihood'
RECOVERY_TRAINING_SEMANTICS = 'chart_composer_training_v13_configuration_recovery'


def resolve_action_objective(requested='auto', parent_args=None, *, resume=False,
                             architecture_profile='audio_grounded_v3'):
    if not isinstance(requested,str) or requested not in {'auto', *ACTION_OBJECTIVES}:
        raise ValueError('Unknown action_objective_policy')
    inherited = parent_args.get('action_objective_policy', LEGACY_ACTION_OBJECTIVE) if parent_args is not None else None
    if inherited is not None and inherited not in ACTION_OBJECTIVES:
        raise ValueError('Invalid saved action_objective_policy')
    # Fresh grounded training must supervise configurations on its own
    # generated histories as well as authored histories. Existing parents
    # retain their explicit training objective unless the caller changes it.
    default = RECOVERY_ACTION_OBJECTIVE if architecture_profile == 'audio_grounded_v3' else LEGACY_ACTION_OBJECTIVE
    resolved = (inherited or default) if requested == 'auto' else requested
    if resolved != LEGACY_ACTION_OBJECTIVE and architecture_profile != 'audio_grounded_v3':
        raise ValueError(f'{resolved} requires audio_grounded_v3')
    if resume and inherited is not None and resolved != inherited:
        raise ValueError('Resume inherits action_objective_policy; use finetune with a new directory')
    return resolved


def training_semantics_for_objective(architecture_profile, policy=LEGACY_ACTION_OBJECTIVE):
    policy = resolve_action_objective(policy, architecture_profile=architecture_profile)
    if architecture_profile == 'audio_grounded_v3':
        if policy == RECOVERY_ACTION_OBJECTIVE:
            return RECOVERY_TRAINING_SEMANTICS
        return JOINT_TRAINING_SEMANTICS if policy == JOINT_ACTION_OBJECTIVE else GROUNDED_TRAINING_SEMANTICS
    return TRAINING_SEMANTICS


def action_objective(logits, targets, sample_weights, *, policy=LEGACY_ACTION_OBJECTIVE):
    """Return the selected loss and diagnostics in fixed, comparable units.

    Joint NLL sums every authored action and its first STOP. Every sequence
    keeps its sample weight; its future length never rescales earlier labels.
    There is no STOP class reweighting. The legacy path remains exact for old
    training contracts. Total losses from different policies are not comparable.
    """
    if policy not in ACTION_OBJECTIVES:
        raise ValueError('Unknown action_objective_policy')
    per_slot = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                               reduction='none').reshape_as(targets)
    active = torch.ones_like(targets, dtype=torch.bool)
    if targets.size(1) > 1:
        active[:, 1:] = targets[:, :-1].ne(0)
    legacy_weights = torch.where(targets.ne(0), 1., .35) * active.to(per_slot)
    denominator = sample_weights.sum().clamp_min(1e-6)
    legacy = (((per_slot * legacy_weights).sum(-1) / legacy_weights.sum(-1).clamp_min(1e-6))
              * sample_weights).sum() / denominator
    sequence_active = torch.ones_like(active)
    if targets.size(1) > 1:
        sequence_active[:, 1:] = targets[:, :-1].ne(0).to(torch.int64).cumprod(-1).bool()
    joint = ((per_slot * sequence_active).sum(-1) * sample_weights).sum() / denominator
    non_stop = sequence_active & targets.ne(0)
    stop = sequence_active & targets.eq(0)
    decision_weights = sample_weights[:, None]
    diagnostics = {
        'legacy_ce': legacy, 'joint_nll': joint,
        'first_slot_ce': (per_slot[:, 0] * sample_weights).sum() / denominator,
        'nonstop_ce': (per_slot * non_stop * decision_weights).sum() / (non_stop * decision_weights).sum().clamp_min(1e-6),
        'stop_ce': (per_slot * stop * decision_weights).sum() / (stop * decision_weights).sum().clamp_min(1e-6),
    }
    # Separate real-note continuations from the easy STOP prediction. Pool
    # numerators/masses across batches rather than averaging conditional means.
    first_non_stop = non_stop.clone()
    first_non_stop[:, 1:] = False
    later_non_stop = non_stop.clone()
    later_non_stop[:, 0] = False
    for name, mask in (('first_nonstop', first_non_stop), ('later_nonstop', later_non_stop)):
        mass = (mask * decision_weights).sum()
        numerator = (per_slot * mask * decision_weights).sum()
        diagnostics[name + '_ce'] = numerator / mass.clamp_min(1e-6)
        diagnostics[name + '_ce_sum'] = numerator
        diagnostics[name + '_ce_weight'] = mass
    return (legacy if policy == LEGACY_ACTION_OBJECTIVE else joint), diagnostics


def recovery_objective(output, batch, *, policy=LEGACY_ACTION_OBJECTIVE, weight=.1):
    """Train legal reference continuation after actual sampled chart events.

    Only v3 adds configuration supervision. Reference labels are examples of
    legal continuation, not an optimal musical recovery oracle. Other authored
    auxiliary targets must remain masked after the generated prefix changes.
    """
    if policy not in ACTION_OBJECTIVES:
        raise ValueError('Unknown action_objective_policy')
    importance = batch['sample_weight'] * batch['pointer_weight']
    per_sample = F.cross_entropy(output['candidate_logits'], batch['candidate_target'], reduction='none')
    pointer = (per_sample * importance).sum() / importance.sum().clamp_min(1e-6)
    action = pointer.new_zeros(())
    action_metrics = {}
    if policy == RECOVERY_ACTION_OBJECTIVE:
        action, diagnostics = action_objective(output['action_logits'], batch['action_target'],
            batch['sample_weight'] * batch['action_weight'], policy=policy)
        action_metrics = {'action_' + name: value.detach() for name, value in diagnostics.items()
                          if name in ('first_nonstop_ce', 'later_nonstop_ce',
                                      'first_nonstop_ce_weight', 'later_nonstop_ce_weight')}
    return (pointer + action) * weight, {'pointer_loss': pointer.detach(), 'action_loss': action.detach(),
                                       **action_metrics}
