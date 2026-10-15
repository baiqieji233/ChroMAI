"""Verify UI training options reach CLI and perturb evidence rather than answers."""
from fractions import Fraction as F
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from maichart.event_timing import EventPoint
from maichart.form import data, train
from maichart.form.training_input_perturbation import (
    PerturbationConfig, QUALITY_TRAINING_DEFAULTS, perturb_load_curve, perturb_history_notes,
)
from maichart.webui import form_api, server as webui
from test_webui import webui_api
from test_form_pipeline import fixture_audio


def test_quality_options_reach_cli_and_effective_settings(webui_api, monkeypatch, tmp_path):
    manifest = tmp_path / 'manifest.json'; manifest.write_text('{}')
    calls = []
    monkeypatch.setattr(webui, '_run_subprocess_task', lambda task, args, **kw: calls.append((task, args)))
    values = {**QUALITY_TRAINING_DEFAULTS, 'perturb_apply_prob': .5,
        'perturb_curve_noise_std': .03, 'perturb_curve_dropout_prob': .02,
        'perturb_history_corruption_prob': .03, 'diversity_loss_weight': .01}
    status, result = webui_api('/api/form/train', {
        'manifest': str(manifest), 'checkpoint_dir': str(tmp_path / 'out'),
        'model_family': 'song_form_v2', 'mode': 'fresh', **values})
    assert status == 200, result
    task, command = calls[0]
    options = train.parser().parse_args(command[2:])
    for name, value in values.items():
        assert getattr(options, name) == value
        assert task.result['effective_settings'][name] == value


def test_quality_resume_settings_have_explicit_legacy_defaults():
    options = form_api.form_resume_settings({'args': {'perturb_curve_noise_std': .04}})
    for name, default in QUALITY_TRAINING_DEFAULTS.items():
        assert options[name] == (.04 if name == 'perturb_curve_noise_std' else default)


def test_api_rejects_repetition_penalty_for_new_runs(webui_api, monkeypatch, tmp_path):
    manifest = tmp_path / 'manifest.json'; manifest.write_text('{}')
    calls = []
    monkeypatch.setattr(webui, '_run_subprocess_task', lambda task, args, **kw: calls.append(args))
    status, result = webui_api('/api/form/train', {
        'manifest': str(manifest), 'checkpoint_dir': str(tmp_path / 'out'),
        'model_family': 'song_form_v2', 'mode': 'fresh', 'repetition_penalty_weight': .1})
    assert status != 200 and '重复惩罚' in json.dumps(result, ensure_ascii=False)
    assert not calls


@pytest.mark.parametrize('name,value', [('perturb_curve_noise_std', float('nan')),
    ('perturb_onset_shift_ms', 51), ('diversity_loss_weight', -.1)])
def test_cli_quality_options_validate_bounds(name, value, tmp_path):
    options = train.parser().parse_args(['--manifest', 'fixture.json', '--checkpoint-dir', str(tmp_path)])
    setattr(options, name, value)
    with pytest.raises(ValueError, match=name):
        train._validate(options)


def test_curve_noise_is_reproducible_independent_of_numpy_global_rng():
    curve = np.ones((8, 4), dtype=np.float32) * .4
    saved = curve.copy()
    first = perturb_load_curve(curve, noise_std=.1, dropout_prob=.3, rng=random.Random(33))
    np.random.seed(984)
    second = perturb_load_curve(curve, noise_std=.1, dropout_prob=.3, rng=random.Random(33))
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(curve, saved)


def test_history_corruption_preserves_sustains_and_chart_positions():
    points = [EventPoint(F(i), text) for i, text in enumerate(['1', '2b', '1h[1:2]', '1-5[8:1]', '1/5'])]
    changed = perturb_history_notes(points, corruption_prob=1, rng=random.Random(1))
    assert [p.beat for p in changed] == [p.beat for p in points]
    assert changed[0].note_text != '1' and changed[1].note_text != '2b'
    assert changed[2:] == points[2:]


def test_diversity_loss_ignores_padding_and_respects_microbatch_normalization():
    from maichart.form.repetition_penalty import compute_pattern_diversity_loss
    torch.manual_seed(7)
    hidden = torch.randn(2, 32, 6, requires_grad=True)
    mask = torch.arange(32)[None] < torch.tensor([16, 32])[:, None]
    losses = compute_pattern_diversity_loss(hidden, diversity_weight=.05, valid_mask=mask, reduction='none')
    short = compute_pattern_diversity_loss(hidden[:1, :16], diversity_weight=.05)
    torch.testing.assert_close(losses[0], short)
    (losses * torch.tensor([1., 0.])).sum().backward()
    assert hidden.grad[1].abs().sum() == 0
    assert hidden.grad[0, 16:].abs().sum() == 0


def test_dataset_perturbs_v1_skeleton_but_keeps_targets_and_varies_by_epoch(tmp_path):
    folder = tmp_path / 'song'; folder.mkdir()
    (folder / 'maidata.txt').write_text('&wholebpm=120\n&lv_5=13\n&inote_5=(120){8}' + ','.join('1234' * 8) + ',E\n', encoding='utf-8')
    (folder / 'track.wav').write_bytes(b'fixture')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'songs': [dict(song_id='fixture', maidata_path='song/maidata.txt',
        audio_path='song/track.wav', difficulties=[dict(difficulty_index=5, level_raw='13')])]}))
    common = dict(cache_dir=tmp_path/'cache', max_tokens=512, max_region_seconds=3,
                  audio_loader=lambda *a, **k: fixture_audio(9))
    clean = data.FormDataset(manifest, **common)
    altered = data.FormDataset(manifest, **common, perturbation_config=PerturbationConfig(
        onset_drop_prob=.5, curve_noise_std=.1, history_corruption_prob=.3, apply_prob=1))
    before, first = clean[0], altered[0]
    assert torch.equal(before['target'], first['target'])
    assert len(first['skeleton']) < len(before['skeleton'])
    assert not torch.equal(first['load_curve'], before['load_curve'])
    second = altered[0]
    assert torch.equal(second['load_curve'], first['load_curve'])
    assert altered.sample_store.stats()['entries'] == 0
    altered.seed += 100003
    third = altered[0]
    assert not torch.equal(third['load_curve'], first['load_curve'])
    assert torch.equal(third['target'], before['target'])
