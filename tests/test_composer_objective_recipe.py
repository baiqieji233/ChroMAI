"""Fixed-unit diagnostics and synchronization-free grouped loss contracts."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from maichart.composer.objectives import (
    JOINT_ACTION_OBJECTIVE,
    RECOVERY_ACTION_OBJECTIVE,
    OBJECTIVE_RECIPE_VERSION,
    action_group_loss_kwargs,
    action_objective,
    recovery_objective,
    training_objective_recipe,
)


@pytest.mark.parametrize('packed', [False, True])
def test_macro_only_skips_atom_ce_and_preserves_loss_gradients_and_atom_diagnostics(monkeypatch, packed):
    torch.manual_seed(89)
    gids = torch.tensor([0, 1, 1, 2, 2, 2])
    targets = torch.tensor([[1, 3, 0, 0], [4, 0, 0, 0]])
    weights = torch.tensor([1., .7])
    active = torch.tensor([[True, True, True, False], [True, True, False, False]])
    indices = active.nonzero() if packed else None
    dense = torch.randn(2, 4, 6)
    x = (dense[indices[:, 0], indices[:, 1]] if packed else dense).clone().requires_grad_(True)
    reference = x.detach().clone().requires_grad_(True)
    flat = reference.reshape(-1, 6)
    grouped = torch.stack([flat[:, gids == g].logsumexp(-1) for g in range(3)], -1)
    macro_grid = grouped.reshape(2, 4, 3) if not packed else grouped
    expected, _ = action_objective(macro_grid, gids[targets], weights,
        policy=JOINT_ACTION_OBJECTIVE, slot_indices=indices)
    _, atom_diagnostics = action_objective(reference, targets, weights,
        policy=JOINT_ACTION_OBJECTIVE, slot_indices=indices)
    original_ce = F.cross_entropy
    ce_widths = []
    def record_ce(values, *args, **kwargs):
        ce_widths.append(values.shape[-1])
        return original_ce(values, *args, **kwargs)
    monkeypatch.setattr(F, 'cross_entropy', record_ce)
    # This catches an accidental max().item() in the grouped loss path even
    # on CPU, where a CUDA synchronization would otherwise go unnoticed.
    with monkeypatch.context() as local:
        def forbidden_item(*args, **kwargs):
            raise AssertionError('loss must use the cached group count')
        local.setattr(torch.Tensor, 'item', forbidden_item)
        result, diagnostics = action_objective(x, targets, weights,
            policy=JOINT_ACTION_OBJECTIVE, slot_indices=indices,
            macro_group_ids=gids, macro_num_groups=3, macro_only=True)
    assert ce_widths == [3]
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(diagnostics['atom_nonstop_ce'], atom_diagnostics['nonstop_ce'])
    torch.testing.assert_close(diagnostics['atom_first_slot_ce'], atom_diagnostics['first_slot_ce'])
    torch.testing.assert_close(diagnostics['atom_legacy_ce'], atom_diagnostics['legacy_ce'])
    torch.testing.assert_close(diagnostics['atom_joint_nll'], atom_diagnostics['joint_nll'])
    result.backward()
    expected.backward()
    torch.testing.assert_close(x.grad, reference.grad)


def _model(grouped=True, competition=False):
    head = SimpleNamespace(num_groups=91, num_audio_groups=17,
        atom_group_id=torch.tensor([0, 1]), audio_macro_group_id=torch.tensor([0, 1])) if grouped else None
    return SimpleNamespace(config=SimpleNamespace(architecture_profile='audio_grounded_v3'),
        conserved_action_head=head, candidate_competition=object() if competition else None)


def test_recipe_records_effective_objectives_weights_and_old_model_competition():
    model = _model()
    args = {'action_objective_policy': RECOVERY_ACTION_OBJECTIVE,
        'recovery_interval': 100, 'recovery_weight': .1, 'placement_weight': .2}
    recipe = training_objective_recipe(model, args)
    assert recipe['version'] == OBJECTIVE_RECIPE_VERSION
    assert recipe['audio_macro_groups'] == 17
    assert recipe['candidate_competition'] == 'off'
    assert recipe['placement']['alignment_weight'] == pytest.approx(.1)
    assert recipe['placement']['main_path_gradient_scale'] == .15
    assert recipe['recovery']['eos_relative_weight'] == .15
    assert recipe['metric_contract']['audio_action_loss'].startswith('macro_objective_not_comparable')
    assert recipe['metric_contract']['old_audio_action_loss_comparison'] == 'audio_action_atom_joint_nll'
    kwargs = action_group_loss_kwargs(model)
    assert kwargs['macro_num_groups'] == 91 and kwargs['audio_macro_num_groups'] == 17
    assert kwargs['macro_group_ids'] is model.conserved_action_head.atom_group_id
    off = training_objective_recipe(model, {**args, 'recovery_interval': 0})
    assert off['recovery']['eos'] == 'off'
    flat = training_objective_recipe(_model(grouped=False), args)
    assert flat['audio_action_objective'] == 'full_atom_v1'


def test_compact_recipe_records_gap_auxiliary_and_replaced_future_evidence():
    model = _model()
    model.config.action_head_policy = 'compact_grouped_v3'
    model.config.configuration_quality_policy = 'song_paired_v3'
    recipe = training_objective_recipe(model, {})
    assert recipe['empty_gap_supervision'] == {
        'policy': 'four_real_audio_zero_count_probes', 'weight': .2,
        'event_margin_sec': .025, 'decoder_input': False}
    assert recipe['configuration_quality']['future_gap_evidence_policy'] == 'replaced_by_fixed_audio_probes'
    assert recipe['same_event_order'] == 'canonical_or_reverse_50pct_train_canonical_eval'


def test_recipe_is_written_and_inspected_without_relabeling_architecture_contract(tmp_path, monkeypatch):
    from maichart.composer import checkpoint
    from maichart.composer.model import ComposerConfig, build_model
    atoms = ['1', '2', 'A1']
    config = ComposerConfig(atom_vocab_size=4, d_model=16, nhead=4, layers=1,
        dim_feedforward=32, history_events=4, candidate_count=4, action_slots=4,
        architecture_profile='audio_grounded_v3', head_timing_dim=99,
        lens_width=16, action_head_policy='conserved_grouped_v2')
    model = build_model(config, atoms)
    monkeypatch.setattr(checkpoint, '_write_payload', lambda *args: None)
    payload = checkpoint.save(tmp_path / 'model.pt', model, None, epoch=1,
        args={'action_objective_policy': RECOVERY_ACTION_OBJECTIVE, 'recovery_interval': 100},
        manifest=tmp_path / 'manifest.json', atom_catalog=atoms)
    assert payload['training_objective_recipe']['version'] == OBJECTIVE_RECIPE_VERSION
    assert payload['args']['training_objective_recipe'] == payload['training_objective_recipe']
    report = checkpoint.inspect_payload(payload)
    assert report['valid'], report['error']
    assert report['training_objective_recipe'] == payload['training_objective_recipe']


def test_zero_recovery_eos_weights_have_zero_loss_and_gradient():
    output = {'candidate_logits': torch.zeros(1, 2, requires_grad=True),
        'action_logits': torch.zeros(1, 2, 3, requires_grad=True),
        'eos_logits': torch.tensor([.8], requires_grad=True)}
    batch = {'sample_weight': torch.ones(1), 'pointer_weight': torch.ones(1),
        'action_weight': torch.ones(1), 'candidate_target': torch.tensor([0]),
        'action_target': torch.tensor([[1, 0]]), 'eos_weight': torch.zeros(1),
        'eos_target': torch.zeros(1)}
    loss, metrics = recovery_objective(output, batch, policy=RECOVERY_ACTION_OBJECTIVE)
    assert metrics['eos_loss'] == 0
    loss.backward()
    torch.testing.assert_close(output['eos_logits'].grad, torch.zeros(1))
