"""Versioned action learning objectives and comparable configuration metrics."""
from __future__ import annotations

import torch
from torch.nn import functional as F

from . import GROUNDED_TRAINING_SEMANTICS, TRAINING_SEMANTICS
from .conserved_actions import group_logsumexp

LEGACY_ACTION_OBJECTIVE = 'event_balanced_v1'
JOINT_ACTION_OBJECTIVE = 'joint_event_nll_v2'
RECOVERY_ACTION_OBJECTIVE = 'joint_event_recovery_v3'
ACTION_OBJECTIVES = {LEGACY_ACTION_OBJECTIVE, JOINT_ACTION_OBJECTIVE, RECOVERY_ACTION_OBJECTIVE}
JOINT_TRAINING_SEMANTICS = 'chart_composer_training_v12_joint_event_likelihood'
RECOVERY_TRAINING_SEMANTICS = 'chart_composer_training_v13_configuration_recovery'
OBJECTIVE_RECIPE_VERSION = 'composer_objective_recipe_v1_precision_audio_macro'
PLACEMENT_ALIGNMENT_SCALE = 0.5
RECOVERY_EOS_SCALE = 0.15


def action_group_loss_kwargs(model):
    """Read constructor-cached group counts without inspecting device tensors."""
    grouped = getattr(model, 'conserved_action_head', None)
    return {
        'macro_group_ids': getattr(grouped, 'atom_group_id', None),
        'macro_num_groups': getattr(grouped, 'num_groups', None),
        'audio_macro_group_ids': getattr(grouped, 'audio_macro_group_id', None),
        'audio_macro_num_groups': getattr(grouped, 'num_audio_groups', None),
    }


def training_objective_recipe(model, args):
    """Describe the effective objective separately from architecture compatibility.

    This is provenance, not a new resume gate. Old checkpoints can still load;
    the trainer reports any recipe change before the next update.
    """
    from .conserved_actions import AUDIO_MACRO_GROUP_POLICY
    from .audio_lens import PLACEMENT_PROXIMITY_SIGMA_SEC, PLACEMENT_TOLERANCE_SEC
    from .quality_inputs import quality_input_policy_for_policy
    from .quality_context import QUALITY_ACTION_PREDICTION_GRAD_SCALE
    grouped = getattr(model, 'conserved_action_head', None)
    grounded = model.config.architecture_profile == 'audio_grounded_v3'
    compact = getattr(model.config,'action_head_policy','flat_v1') == 'compact_grouped_v3'
    integrated = getattr(model.config,'timing_distribution_policy','hierarchical_v1') == 'survival_integrated_v2'
    policy = args.get('action_objective_policy', LEGACY_ACTION_OBJECTIVE)
    interval = int(args.get('recovery_interval', 0))
    recovery_weight = float(args.get('recovery_weight', .1))
    recovery_enabled = grounded and interval > 0 and recovery_weight > 0
    eos_enabled = recovery_enabled and policy == RECOVERY_ACTION_OBJECTIVE
    if grounded:
        from .grounded_model import PLACEMENT_GRAD_SCALE
    return {
        'version': OBJECTIVE_RECIPE_VERSION,
        'action_objective': policy,
        'sample_sequence': {
            'policy': args.get('sample_sequence_policy','independent_events_v1'),
            'phrase_max_beats': 16 if args.get('sample_sequence_policy')=='phrase_groups_v1' else None,
            'phrase_max_targets': 8 if args.get('sample_sequence_policy')=='phrase_groups_v1' else None,
            'history_encoding': 'each_target_own_causal_prefix',
            'shared_song_encoder': getattr(model.config,'configuration_quality_policy','off')=='song_paired_v3',
        },
        'audio_action_objective': AUDIO_MACRO_GROUP_POLICY if grouped is not None else 'full_atom_v1',
        'action_macro_groups': int(grouped.num_groups) if grouped is not None else None,
        'audio_macro_groups': int(grouped.num_audio_groups) if grouped is not None else None,
        'audio_auxiliary_weight': float(args.get('audio_pointer_weight', .15)),
        'placement': {
            'occupancy': 'type_union_within_25ms_v1' if grounded else 'off',
            'alignment': 'nearest_event_type_gaussian_positive_occupancy_v1' if grounded else 'off',
            'occupancy_tolerance_sec': PLACEMENT_TOLERANCE_SEC if grounded else None,
            'sigma_sec': PLACEMENT_PROXIMITY_SIGMA_SEC if grounded else None,
            'occupancy_weight': float(args.get('placement_weight', .15)) if grounded else 0.,
            'alignment_relative_weight': PLACEMENT_ALIGNMENT_SCALE if grounded else 0.,
            'alignment_weight': float(args.get('placement_weight', .15)) * PLACEMENT_ALIGNMENT_SCALE if grounded else 0.,
            'main_path_gradient_scale': PLACEMENT_GRAD_SCALE if grounded else 0.,
        },
        'recovery': {
            'enabled': recovery_enabled, 'interval': interval,
            'samples': int(args.get('recovery_samples', 2)),
            'steps': int(args.get('recovery_steps', 2)), 'weight': recovery_weight,
            'eos': 'reference_after_selected_event_v1' if eos_enabled else 'off',
            'eos_relative_weight': RECOVERY_EOS_SCALE if eos_enabled else 0.,
        },
        'candidate_competition': 'neighbor_residual_v1' if getattr(model, 'candidate_competition', None) is not None else 'off',
        'local_lead_in': 'multiscale_exact_beat_v2' if getattr(model, 'local_lead_in', None) is not None else 'off',
        'history_clock': 'unclipped_relative_beats_fp32_v1',
        'timing_distribution_policy': getattr(model.config,'timing_distribution_policy','hierarchical_v1'),
        'timing_rate_contract': {
            'grid_seconds': .05 if integrated else None,
            'audio': 'existing_104d_multiscale_context' if integrated else 'candidate_workload',
            'integral': 'fixed_grid_absolute_seconds_right_endpoint_queries' if integrated else 'legacy',
            'tail': 'unchanged_by_rhythm_history_and_temperature' if integrated else 'joint_log_softmax',
            'planner_rest': 'detached_soft_rate_prior' if integrated else 'indirect_context',
            'count_supervision': 'known_fixed_cells_poisson' if integrated else 'uniform_quarter_second_probes',
            'rate_loss_reference_exposure_sec': .25,
            'planner_rest_objective': 'bernoulli_nll' if integrated else 'smooth_l1_after_sigmoid',
            'audio_timing_score': 'separate_distance_classifier_and_unclipped_within_band_onsets' if integrated else 'legacy_tanh_shape',
            'audio_pointer_auxiliary': 'direct_candidate_ce' if integrated else 'legacy_joint_likelihood',
        },
        'button_load_policy': getattr(model.config,'button_load_policy','off'),
        'button_load_context': 'mean_max_to_existing_timing_and_all_action_slots' if hasattr(model,'button_load_context') else 'off',
        'fine_timing_policy':getattr(model.config,'fine_timing_policy','off'),
        'density_feedback_policy':getattr(model.config,'density_feedback_policy','off'),
        'density_feedback_source':'causal_committed_starts_vs_detached_1s_4s_rate_integral',
        'fine_timing_source':'existing_event_ctx_fine_note_ratio_no_new_target',
        'same_event_order': 'canonical_or_reverse_50pct_train_canonical_eval' if getattr(model.config,'action_head_policy','flat_v1')=='compact_grouped_v3' else 'canonical',
        'audio_reference': 'independent_macro_uniform_detail' if getattr(model.config,'action_head_policy','flat_v1')=='compact_grouped_v3' else 'shared_atom_head',
        'future_audio_readout': ('off' if getattr(model.config,'future_horizon_policy','off')=='off' else
                                 'ten_fixed_beat_probes' if compact else 'candidate_attention'),
        'empty_gap_supervision': {
            'policy': 'covered_by_fixed_grid_counts' if integrated else 'four_real_audio_zero_count_probes' if compact else 'off',
            'weight': 0. if integrated else .2 if compact else 0.,
            'event_margin_sec': .025 if compact and not integrated else None,
            'decoder_input': False,
        },
        'configuration_quality': {
            'policy': getattr(model.config,'configuration_quality_policy','off'),
            'input_policy': quality_input_policy_for_policy(getattr(model.config,'configuration_quality_policy','off')) if getattr(model,'quality_context',None) is not None else 'off',
            'action_to_prediction_gradient_scale': QUALITY_ACTION_PREDICTION_GRAD_SCALE if getattr(model,'quality_context',None) is not None else 0.,
            'phrase_weight': .1 if getattr(model,'quality_context',None) is not None else 0.,
            'gap_weight': .1 if getattr(model,'quality_context',None) is not None else 0.,
            'decoder_uses': 'predicted_phrase_and_gap_only',
            'music_context': 'whole_song_shared_encoder' if getattr(model.config,'configuration_quality_policy','off')=='song_paired_v3' else 'legacy_independent_planner',
            'history_pairing': 'each_committed_event_at_its_own_audio_and_clock' if getattr(model.config,'configuration_quality_policy','off')=='song_paired_v3' else 'selected_anchors',
            'relative_time_features': 'log_decay_and_1_2_3_4_beat_sin_cos_10d' if getattr(model.config,'configuration_quality_policy','off')=='song_paired_v3' else 'legacy_anchor_clock',
            'fine_rhythm_target': 'existing_event_ctx_fine_note_ratio' if getattr(model.config,'configuration_quality_policy','off')=='song_paired_v3' else 'off',
            'fine_rhythm_target_rule': 'reuse_existing_target_no_duplicate_delta_loss' if getattr(model.config,'configuration_quality_policy','off')=='song_paired_v3' else 'off',
            'extra_delta_loss': False,
            'recovery_auxiliary_labels': 'masked',
            'future_gap_evidence_policy': 'replaced_by_fixed_audio_probes' if compact else ('observed_max_presence_attack_support_7bins_v1' if getattr(model.config,'configuration_quality_policy','off') in {'cooperative_v2','song_paired_v3'} else 'off'),
            'future_gap_evidence_gradient': 'detached',
            'same_event_geometry': 'kind_mass_conserved' if getattr(model.config,'configuration_quality_policy','off') in {'cooperative_v2','song_paired_v3'} else 'legacy_raw_addition',
            'first_slot_min_legal_mass': .01 if getattr(model.config,'configuration_quality_policy','off') in {'cooperative_v2','song_paired_v3'} else 0.,
        },
        'generation_recovery_decode_rules': 'rhythm_blank_guard_v2',
        'metric_contract': {
            'action_macro_ce': 'main_head_kind_and_start_pad_nonstop_nats',
            'action_detail_ce': 'main_within_group_nonstop_nats',
            'audio_action_loss': 'macro_objective_not_comparable_to_old_full_atom_loss' if grouped is not None else 'full_atom_objective',
            'atom_nonstop_ce': 'full_atom_nonstop_nats_weighted_by_decisions',
            'atom_first_slot_ce': 'full_atom_first_slot_nats_weighted_by_samples',
            'atom_legacy_ce': 'full_atom_event_balanced_nats_with_stop_weight_0.35',
            'atom_joint_nll': 'full_atom_sequence_nats_including_first_stop',
            'old_audio_action_loss_comparison': (
                'audio_action_atom_legacy_ce' if policy == LEGACY_ACTION_OBJECTIVE
                else 'audio_action_atom_joint_nll'),
            'old_audio_action_loss_comparison_scope': 'same_action_objective_policy_sample_weights_catalog_and_evaluation_rows',
            'musical_quality': 'requires_closed_loop_generation_review',
        },
    }


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


def action_objective(logits, targets, sample_weights, *, policy=LEGACY_ACTION_OBJECTIVE, slot_indices=None,
                     macro_group_ids=None, macro_num_groups=None, macro_only=False):
    """Return the selected loss and diagnostics in fixed, comparable units.

    Joint NLL sums every authored action and its first STOP. Every sequence
    keeps its sample weight; its future length never rescales earlier labels.
    There is no STOP class reweighting. The legacy path remains exact for old
    training contracts. Total losses from different policies are not comparable.
    """
    if policy not in ACTION_OBJECTIVES:
        raise ValueError('Unknown action_objective_policy')
    if macro_only and macro_group_ids is None:
        raise ValueError('macro_only requires macro_group_ids')
    if slot_indices is None:
        flat_logits = logits.reshape(-1, logits.size(-1))
        flat_targets = targets.reshape(-1)
    else:
        rows, slots = slot_indices.unbind(-1)
        flat_logits = logits
        flat_targets = targets[rows, slots]

    def restore(values):
        if slot_indices is None:
            return values.reshape_as(targets)
        # Only the scalar CE grid is restored; never a [B,S,V] vocabulary grid.
        return values.new_zeros(targets.shape).index_put((rows, slots), values)

    # The macro auxiliary never consumes full-vocabulary CE. Do not compute
    # and discard its softmax/backward graph before replacing it with group CE.
    per_slot = None if macro_only else restore(F.cross_entropy(flat_logits, flat_targets, reduction='none'))
    macro_per_slot = None
    detail_per_slot = None
    if macro_group_ids is not None:
        gids = macro_group_ids.to(device=flat_logits.device, dtype=torch.long)
        if macro_num_groups is not None:
            num_groups = int(macro_num_groups)
        elif macro_group_ids.device.type == 'cpu':
            # Compatibility for standalone CPU callers; training passes the
            # constructor-cached group count, including on CUDA.
            num_groups = int(macro_group_ids.max()) + 1 if macro_group_ids.numel() else 1
        else:
            # Missing cached metadata must not trigger a GPU/CPU sync. Empty
            # groups have -inf logits and exactly zero probability.
            num_groups = max(1, macro_group_ids.numel())
        group_lse = group_logsumexp(flat_logits, gids, num_groups)
        target_groups = gids[flat_targets]
        flat_macro = F.cross_entropy(group_lse, target_groups, reduction='none')
        flat_detail = (
            group_lse.gather(-1, target_groups[:, None]).squeeze(-1)
            - flat_logits.float().gather(-1, flat_targets[:, None]).squeeze(-1)
        ).clamp_min(0.0)
        macro_per_slot = restore(flat_macro)
        detail_per_slot = restore(flat_detail)
        if macro_only:
            per_slot = macro_per_slot
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
    # The decomposition gives the same full-atom NLL without a second
    # vocabulary softmax. Keep these fixed-unit diagnostics when the auxiliary
    # switches from atom labels to macro labels, so a renamed easier objective
    # cannot masquerade as better atom prediction.
    with torch.no_grad():
        atom_per_slot = (macro_per_slot + detail_per_slot) if macro_only else per_slot
        atom_mass = (non_stop * decision_weights).sum()
        atom_sum = (atom_per_slot * non_stop * decision_weights).sum()
        diagnostics['atom_nonstop_ce'] = atom_sum / atom_mass.clamp_min(1e-6)
        diagnostics['atom_nonstop_ce_sum'] = atom_sum
        diagnostics['atom_nonstop_ce_weight'] = atom_mass
        diagnostics['atom_first_slot_ce'] = (atom_per_slot[:, 0] * sample_weights).sum() / denominator
        diagnostics['atom_legacy_ce'] = (
            (atom_per_slot * legacy_weights).sum(-1)
            / legacy_weights.sum(-1).clamp_min(1e-6) * sample_weights
        ).sum() / denominator
        diagnostics['atom_joint_nll'] = (
            (atom_per_slot * sequence_active).sum(-1) * sample_weights
        ).sum() / denominator
    if macro_per_slot is not None and detail_per_slot is not None:
        nonstop_mass = (non_stop * decision_weights).sum()
        macro_sum = (macro_per_slot * non_stop * decision_weights).sum()
        detail_sum = (detail_per_slot * non_stop * decision_weights).sum()
        diagnostics['macro_ce'] = macro_sum / nonstop_mass.clamp_min(1e-6)
        diagnostics['macro_ce_sum'] = macro_sum
        diagnostics['macro_ce_weight'] = nonstop_mass
        diagnostics['detail_ce'] = detail_sum / nonstop_mass.clamp_min(1e-6)
        diagnostics['detail_ce_sum'] = detail_sum
        diagnostics['detail_ce_weight'] = nonstop_mass
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
    per_sample = (-output['next_event_log_probs'].gather(1,batch['candidate_target'][:,None]).squeeze(1)
                  if 'next_event_log_probs' in output else
                  F.cross_entropy(output['candidate_logits'], batch['candidate_target'], reduction='none'))
    pointer = (per_sample * importance).sum() / importance.sum().clamp_min(1e-6)
    action = pointer.new_zeros(())
    eos_loss = pointer.new_zeros(())
    action_metrics = {}
    if policy == RECOVERY_ACTION_OBJECTIVE:
        action, diagnostics = action_objective(output['action_logits'], batch['action_target'],
            batch['sample_weight'] * batch['action_weight'], policy=policy,
            slot_indices=output.get('action_slot_indices'))
        action_metrics = {'action_' + name: value.detach() for name, value in diagnostics.items()
                          if name in ('first_nonstop_ce', 'later_nonstop_ce',
                                      'first_nonstop_ce_weight', 'later_nonstop_ce_weight')}
        if 'eos_logits' in output and 'eos_weight' in batch:
            eos_logits = output['eos_logits']
            if eos_logits.ndim > 1:
                rows = torch.arange(batch['candidate_target'].size(0), device=eos_logits.device)
                eos_logits = eos_logits[rows, batch['candidate_target']]
            eos_weights = batch['sample_weight'] * batch['eos_weight']
            eos_target = batch['eos_target']
            positives = (eos_target * eos_weights).sum().clamp_min(1.)
            negatives = ((1. - eos_target) * eos_weights).sum().clamp_min(1.)
            eos_bce = F.binary_cross_entropy_with_logits(
                eos_logits, eos_target,
                pos_weight=(negatives / positives).clamp(1., 64.),
                reduction='none',
            )
            eos_loss = (eos_bce * eos_weights).sum() / eos_weights.sum().clamp_min(1e-6)
            action_metrics['eos_loss'] = eos_loss.detach()
    return (pointer + action + RECOVERY_EOS_SCALE * eos_loss) * weight, {
        'pointer_loss': pointer.detach(),
        'action_loss': action.detach(),
        **action_metrics,
    }
