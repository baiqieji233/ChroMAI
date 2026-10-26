import importlib.util
from pathlib import Path
import hashlib
import json

import pytest
import torch


def _runner():
    path = Path(__file__).resolve().parents[1] / 'scripts/audit_composer_capabilities.py'
    spec = importlib.util.spec_from_file_location('capability_audit_runner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixture_rejects_same_size_different_codec_and_state_policy():
    runner = _runner()
    config = dict(architecture_profile='audio_grounded_v3', action_state_policy='committed_causal_v2',
                  history_events=128, candidate_count=96, action_slots=33)
    payload = {'atom_catalog': ['1', '2'], 'model_config': config}
    catalog_hash = hashlib.sha256(json.dumps(['1', '2'], ensure_ascii=False).encode()).hexdigest()
    provenance = {'catalog_sha256': catalog_hash, 'config': dict(config)}
    assert runner.verify_fixture(payload, provenance) == catalog_hash
    with pytest.raises(ValueError, match='catalog'):
        runner.verify_fixture({**payload, 'atom_catalog': ['2', '1']}, provenance)
    provenance['config']['action_state_policy'] = 'selected_prefix_v3'
    with pytest.raises(ValueError, match='action_state_policy'):
        runner.verify_fixture(payload, provenance)


def test_enabled_motif_requires_matching_fixture_or_explicit_missing_input_diagnostic():
    runner = _runner()
    config = dict(architecture_profile='audio_grounded_v3', action_state_policy='committed_causal_v2',
                  history_events=128, candidate_count=96, action_slots=33,
                  motif_memory_policy='audio_recurrence_archive_v1', motif_max_events=16)
    payload = {'atom_catalog': ['1'], 'model_config': config}
    fixture_config = {**config, 'motif_memory_policy': 'off'}
    provenance = {'catalog_sha256': hashlib.sha256(json.dumps(['1']).encode()).hexdigest(),
                  'config': fixture_config}
    with pytest.raises(ValueError, match='allow-missing-motif'):
        runner.verify_fixture(payload, provenance)
    assert runner.verify_fixture(payload, provenance, allow_missing_motif=True)


def test_snapshots_preserve_source_and_never_replace_previous_baselines(tmp_path):
    runner = _runner()
    source = tmp_path / 'live.pt'
    source.write_bytes(b'first saved state')
    first, first_hash = runner.freeze_checkpoint(source, tmp_path / 'snapshots')
    source.write_bytes(b'next saved state')
    second, second_hash = runner.freeze_checkpoint(source, tmp_path / 'snapshots')
    assert first != second and first_hash != second_hash
    assert first.read_bytes() == b'first saved state'
    assert second.read_bytes() == source.read_bytes() == b'next saved state'


def test_intervention_hooks_restore_after_exception_and_do_not_modify_parameters():
    runner = _runner()
    model = torch.nn.Module()
    model.action_relation = torch.nn.Linear(8, 4)
    data = torch.ones(2, 8)
    before = model.action_relation(data).detach()
    weights = model.action_relation.weight.detach().clone()
    with pytest.raises(RuntimeError, match='probe failure'):
        with runner.intervention(model, 'relation_zero'):
            assert torch.count_nonzero(model.action_relation(data)) == 0
            raise RuntimeError('probe failure')
    torch.testing.assert_close(model.action_relation(data), before)
    torch.testing.assert_close(model.action_relation.weight, weights)


def test_malformed_padding_is_rejected_instead_of_resurrecting_actions():
    from maichart.composer.action_factors import CompositionalActions
    from maichart.composer.capability_metrics import CapabilityMetrics

    meter = CapabilityMetrics(CompositionalActions(['1'], 4))
    with pytest.raises(ValueError, match='after the first STOP'):
        meter.update(torch.zeros(1, 3, 2), {'action_target': torch.tensor([[1, 0, 1]])})


def test_cli_writes_traceable_cpu_report_without_changing_checkpoint(tmp_path):
    from test_composer_audio_grounded_v3 import ATOMS, config, inputs
    from maichart.composer.model import build_model
    from maichart.composer.checkpoint import save

    torch.set_num_threads(1)
    runner = _runner()
    model = build_model(config(), ATOMS).eval()
    checkpoint = tmp_path / 'composer.pt'
    save(checkpoint, model, None, epoch=1, args={}, manifest=tmp_path / 'manifest.json',
         atom_catalog=ATOMS, optimizer_steps=10)
    batch = inputs()
    batch['candidate_padding_mask'] = batch.pop('candidate_mask')
    batch.update(candidate_target=torch.tensor([1, 2]), action_target=torch.tensor([[1, 0], [2, 0]]))
    fixture = tmp_path / 'batches.pt'
    torch.save({'heldout': [batch]}, fixture)
    provenance = tmp_path / 'provenance.json'
    provenance.write_text(json.dumps({'catalog_sha256': hashlib.sha256(
        json.dumps(ATOMS, ensure_ascii=False).encode()).hexdigest(), 'config': model.config.to_dict(),
        'fixture_sha256': runner.file_hash(fixture)}))
    destination = tmp_path / 'report.json'
    source_hash = runner.file_hash(checkpoint)
    runner.main(['--checkpoint', str(checkpoint), '--snapshot-dir', str(tmp_path / 'snapshots'),
                 '--batch-file', str(fixture), '--provenance', str(provenance), '--output', str(destination),
                 '--variants', 'normal', 'relation_zero'])
    report = json.loads(destination.read_text())
    assert report['checkpoint']['sha256'] == source_hash == runner.file_hash(checkpoint)
    assert report['checkpoint']['optimizer_steps'] == 10
    assert report['fixture']['fixture_hash_verified'] is True
    assert report['evaluations']['normal']['slot0']['row_count'] == 2
    assert report['evaluations']['normal'] == report['evaluations']['relation_zero']
