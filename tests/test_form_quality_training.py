"""Versioned training objectives and interrupted migration readiness."""
from types import SimpleNamespace

import pytest
import torch

from maichart.form import checkpoint, train
from test_form_rational_events import source as source


def test_quality_settings_can_restore_saved_configs_without_instantiated_models():
    args = SimpleNamespace(model_family='song_form_v2', resume='checkpoint', quality_profile='auto')
    old = {'model_config': {'quality_profile': 'musical_evidence_v1', 'curve_bin_seconds': .5,
                            'position_alignment_tolerance_ms': 22.},
           'blueprint_config': {'context_seconds': 1.5},
           'args': {'position_alignment_loss_weight': .2}}
    train.resolve_quality_training_settings(args, old)
    assert (args.quality_profile, args.curve_bin_seconds, args.position_alignment_tolerance_ms,
            args.planner_context_seconds, args.position_alignment_loss_weight) == (
                'musical_evidence_v1', .5, 22., 1.5, .2)


@pytest.mark.parametrize('change', ['quality', 'curve_bin'])
def test_interrupted_finetune_does_not_inherit_layout_readiness_under_new_inputs(source, tmp_path, monkeypatch, change):
    torch_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        base = ['--manifest', str(source[1]), '--cache-dir', str(tmp_path/'cache'),
                '--device', 'cpu', '--precision', 'fp32', '--epochs', '1', '--batch-size', '1',
                '--max-steps', '2', '--d-model', '16', '--planner-width', '16', '--layers', '1',
                '--nhead', '2', '--max-tokens', '256', '--max-memory-tokens', '64',
                '--max-region-seconds', '3', '--condition-drop-prob', '0']
        original = tmp_path/'original'
        source_profile = 'legacy' if change == 'quality' else 'musical_evidence_v1'
        assert train.main(base + ['--checkpoint-dir', str(original), '--quality-profile', source_profile]) == 0
        path = original/'form_last.pt'
        before = torch.load(path, weights_only=False)
        assert checkpoint.task_is_ready(before)
        save = train._atomic_save
        upgraded = tmp_path/'interrupted'

        def stop_after_planner(payload, destination):
            save(payload, destination)
            if destination.parent == upgraded and payload['training_position']['completed'] == 0:
                raise InterruptedError('durable planner-only new-contract checkpoint')

        monkeypatch.setattr(train, '_atomic_save', stop_after_planner)
        options = ['--quality-profile', 'musical_evidence_v1']
        if change == 'curve_bin':
            options += ['--curve-bin-seconds', '.5']
        with pytest.raises(InterruptedError, match='planner-only'):
            train.main(base + ['--checkpoint-dir', str(upgraded), '--init-checkpoint', str(path)] + options)
        saved = torch.load(upgraded/'form_last.pt', weights_only=False)
        assert saved['planner_effective_steps'] == 1
        assert saved['task_training'] == before['task_training']
        assert saved['capabilities']['draft']['steps'] == 0
        assert saved['capabilities']['revision']['steps'] == 0
        assert not checkpoint.task_is_ready(saved)
        with pytest.raises(ValueError, match='effective training'):
            checkpoint.load(upgraded/'form_last.pt', inference=True)
    finally:
        torch.set_num_threads(torch_threads)


def test_enhanced_mid_epoch_resume_matches_uninterrupted_updates(source, tmp_path, monkeypatch):
    """New auxiliary heads, dropout and perturbations must survive interruption."""
    thread_count = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        common = ['--manifest', str(source[1]), '--cache-dir', str(tmp_path/'cache'),
                  '--device', 'cpu', '--precision', 'fp32', '--epochs', '1',
                  '--quality-profile', 'musical_evidence_v1', '--batch-size', '1',
                  '--grad-accumulation', '2', '--max-steps', '4', '--d-model', '16',
                  '--planner-width', '16', '--layers', '1', '--nhead', '2',
                  '--max-tokens', '256', '--max-memory-tokens', '64', '--max-region-seconds', '2',
                  '--checkpoint-interval-steps', '1', '--dropout', '.1', '--seed', '33',
                  '--perturb-curve-noise-std', '.03', '--perturb-history-corruption-prob', '.02',
                  '--perturb-apply-prob', '.5']
        complete_dir = tmp_path/'complete'
        assert train.main(common + ['--checkpoint-dir', str(complete_dir),
                                   '--layout-prefetch-batches', '0']) == 0
        complete = torch.load(complete_dir/'form_last.pt', weights_only=False)
        atomic_save = train._atomic_save

        def interrupt_after_update(value, path):
            atomic_save(value, path)
            cursor = value['training_position']
            if cursor['phase'] == 'layout_revision' and cursor['completed'] == 2:
                raise InterruptedError('enhanced checkpoint after a complete accumulation group')

        partial_dir = tmp_path/'partial'
        monkeypatch.setattr(train, '_atomic_save', interrupt_after_update)
        with pytest.raises(InterruptedError, match='accumulation group'):
            train.main(common + ['--checkpoint-dir', str(partial_dir), '--layout-prefetch-batches', '2'])
        monkeypatch.setattr(train, '_atomic_save', atomic_save)
        resumed_dir = tmp_path/'resumed'
        assert train.main(common + ['--checkpoint-dir', str(resumed_dir),
                                   '--resume', str(partial_dir/'form_last.pt')]) == 0
        resumed = torch.load(resumed_dir/'form_last.pt', weights_only=False)
        assert resumed['optimizer_steps'] == complete['optimizer_steps']
        assert resumed['capabilities'] == complete['capabilities']
        assert resumed['input_contract'] == complete['input_contract']
        assert resumed['planner_effective_steps'] == complete['planner_effective_steps']
        assert resumed['training_position'] == complete['training_position']
        for key in ('model_state_dict', 'blueprint_state_dict'):
            for name, weight in complete[key].items():
                torch.testing.assert_close(resumed[key][name], weight, rtol=0, atol=0)
    finally:
        torch.set_num_threads(thread_count)
