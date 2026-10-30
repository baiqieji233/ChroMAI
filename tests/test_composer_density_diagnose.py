from types import SimpleNamespace

import torch

from maichart.composer import diagnose


def test_diagnostic_inputs_forward_absolute_grid_without_authored_counts():
    names = ('survival_audio', 'survival_edges', 'survival_candidate_seconds',
             'survival_start_seconds', 'survival_plan_indices', 'survival_plan_mix')
    batch = {name: torch.zeros(1) for name in names}
    batch['survival_count_target'] = torch.ones(1)
    batch['survival_count_weight'] = torch.ones(1)
    inputs = diagnose._inputs(batch, 'cpu')
    assert set(inputs) == set(names)


def test_diagnostic_inputs_keep_causal_counts_separate_from_count_labels():
    prefix = torch.tensor([[[1., 2.]]])
    inputs = diagnose._inputs({'survival_prefix_counts': prefix,
                              'survival_count_target': torch.ones(1)}, 'cpu')
    assert set(inputs) == {'survival_prefix_counts'}
    assert torch.equal(inputs['survival_prefix_counts'], prefix)


def test_diagnostic_dataset_inherits_integrated_timing_policy(monkeypatch):
    captured = {}

    def dataset(*args, **kwargs):
        captured.update(kwargs)
        return captured

    monkeypatch.setattr(diagnose, 'ComposerDataset', dataset)
    config = SimpleNamespace(candidate_count=96, history_events=128, action_slots=8,
        architecture_profile='audio_grounded_v3', action_state_policy='selected_prefix_v3',
        head_timing_policy='merged_pads_v1', motif_memory_policy='off', motif_max_events=16,
        timing_distribution_policy='survival_integrated_v2', density_feedback_policy='causal_count_debt_v1')
    args = SimpleNamespace(manifest='unused', cache_dir='unused', limit_charts=0)
    diagnose._diagnostic_dataset(args, SimpleNamespace(config=config))
    assert captured['timing_distribution_policy'] == 'survival_integrated_v2'
    assert captured['density_feedback_policy'] == 'causal_count_debt_v1'
    del config.timing_distribution_policy
    diagnose._diagnostic_dataset(args, SimpleNamespace(config=config))
    assert captured['timing_distribution_policy'] == 'hierarchical_v1'
