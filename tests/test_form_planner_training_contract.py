"""Planner updates and strict resume describe effective, reproducible training."""
import json

import pytest
import torch

from maichart.form import checkpoint
from maichart.form.resume_contract import validate_resume_cursor


def _ready_payload(**updates):
    value = dict(family='song_form_v2', input_contract=checkpoint.RATIONAL_INPUT_CONTRACT,
                 optimizer_steps=2, blueprint_steps=3, task_training={'draft': 1}, args={})
    value.update(updates)
    return value


def test_rational_readiness_distinguishes_real_and_zero_objective_planner_history():
    assert checkpoint.task_is_ready(_ready_payload())  # Old no-argument metadata.
    assert checkpoint.task_is_ready(_ready_payload(args={'load_curve_loss_weight': 1.}))
    assert not checkpoint.task_is_ready(_ready_payload(args={'load_curve_loss_weight': 0.}))
    assert not checkpoint.task_is_ready(_ready_payload(args={'blueprint_loss_weight': 0.}))
    assert not checkpoint.task_is_ready(_ready_payload(planner_effective_steps=0))
    assert checkpoint.task_is_ready(_ready_payload(planner_effective_steps=2,
                                                   args={'load_curve_loss_weight': 0.}))
    assert checkpoint.task_is_ready(_ready_payload(args={'load_curve_loss_weight': 0.},
        capabilities={'load_curve': {'steps': 1, 'contract': 'load_curve_v1'}}))
    assert checkpoint.task_is_ready(_ready_payload(family='song_form_v1',
        input_contract=checkpoint.LEGACY_INPUT_CONTRACT, args={'load_curve_loss_weight': 0.}))


def test_new_resume_cursor_validates_planner_batch_but_legacy_cursor_remains_usable():
    cursor = dict(epoch=2, phase='blueprint', completed=4, layout_total=8, plan_total=5,
                  batch_size=2, max_steps=0, blueprint_batch_size=4)
    options = dict(epoch=1, layout_total=8, plan_total=5, batch_size=2, max_steps=0)
    validate_resume_cursor(cursor, **options, blueprint_batch_size=4)
    with pytest.raises(ValueError, match='blueprint_batch_size'):
        validate_resume_cursor(cursor, **options, blueprint_batch_size=1)
    del cursor['blueprint_batch_size']
    validate_resume_cursor(cursor, **options, blueprint_batch_size=4)


def _tiny_run(tmp_path, monkeypatch):
    from maichart.form import data, train
    from test_form_pipeline import fixture_audio

    folder = tmp_path / 'song'
    folder.mkdir()
    (folder / 'track.wav').write_bytes(b'fixture')
    (folder / 'maidata.txt').write_text(
        '&title=fixture\n&wholebpm=120\n&lv_5=13\n&inote_5=(120){4}1,2,3,4,E\n', encoding='utf-8')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'songs': [dict(song_id='fixture',
        maidata_path='song/maidata.txt', audio_path='song/track.wav',
        difficulties=[dict(difficulty_index=5, level_raw='13')])]}), encoding='utf-8')
    real_dataset = data.FormDataset
    audio = fixture_audio(3)
    monkeypatch.setattr(train, 'FormDataset', lambda *a, **kw: real_dataset(
        *a, **kw, audio_loader=lambda *a, **kw: audio))
    common = ['--quality-profile', 'legacy', '--manifest', str(manifest), '--cache-dir', str(tmp_path / 'cache'),
        '--model-family', 'song_form_v2', '--rhythm-geometry-profile', 'none',
        '--epochs', '1', '--batch-size', '1', '--max-steps', '1', '--d-model', '16',
        '--nhead', '2', '--layers', '1', '--dropout', '0', '--max-tokens', '128',
        '--max-memory-tokens', '64', '--max-region-seconds', '3', '--precision', 'fp32']
    return train, common


def test_zero_objective_does_not_decay_planner_or_claim_generation_readiness(tmp_path, monkeypatch):
    from maichart.form import rational_planner
    from maichart.webui.form_api import inspect_form_checkpoint

    train, common = _tiny_run(tmp_path, monkeypatch)
    original = rational_planner.RationalPlanner
    initial = {}
    class CapturedPlanner(original):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            if not initial:
                initial.update({k: v.detach().clone() for k, v in self.state_dict().items()})
    monkeypatch.setattr(rational_planner, 'RationalPlanner', CapturedPlanner)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        output = tmp_path / 'zero'
        train.main(common + ['--checkpoint-dir', str(output), '--load-curve-loss-weight', '0'])
        saved = torch.load(output / 'form_last.pt', weights_only=False)
        assert saved['optimizer_steps'] > 0
        assert saved['blueprint_steps'] == saved['planner_effective_steps'] == 0
        assert 'load_curve' not in saved['capabilities']
        for name, value in saved['blueprint_state_dict'].items():
            torch.testing.assert_close(value, initial[name], rtol=0, atol=0)
        assert not checkpoint.task_is_ready(saved)
        assert not inspect_form_checkpoint(output / 'form_last.pt')['can_generate']
        with pytest.raises(ValueError, match='effective training'):
            checkpoint.load(output / 'form_last.pt', inference=True)
    finally:
        torch.set_num_threads(threads)


def test_cli_resume_inherits_planner_batch_even_when_another_size_is_requested(tmp_path, monkeypatch):
    train, common = _tiny_run(tmp_path, monkeypatch)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        source = tmp_path / 'source'
        train.main(common + ['--checkpoint-dir', str(source), '--blueprint-batch-size', '4'])
        resumed = tmp_path / 'resumed'
        train.main(common + ['--checkpoint-dir', str(resumed), '--resume', str(source / 'form_last.pt'),
                             '--blueprint-batch-size', '1'])
        saved = torch.load(resumed / 'form_last.pt', weights_only=False)
        assert saved['args']['blueprint_batch_size'] == 4
        assert saved['training_position']['blueprint_batch_size'] == 4
        assert saved['planner_effective_steps'] == saved['blueprint_steps'] == 2
        frozen = tmp_path / 'frozen_finetune'
        train.main(common + ['--checkpoint-dir', str(frozen),
                             '--init-checkpoint', str(resumed / 'form_last.pt'),
                             '--load-curve-loss-weight', '0'])
        frozen_saved = torch.load(frozen / 'form_last.pt', weights_only=False)
        assert frozen_saved['planner_effective_steps'] == frozen_saved['blueprint_steps'] == 2
        assert checkpoint.task_is_ready(frozen_saved)
        for name, value in saved['blueprint_state_dict'].items():
            torch.testing.assert_close(frozen_saved['blueprint_state_dict'][name], value, rtol=0, atol=0)
    finally:
        torch.set_num_threads(threads)
