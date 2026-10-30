"""Diagnostic tooling must use the same causal quality inputs as generation."""
from types import SimpleNamespace

import numpy as np
import torch

from maichart.composer.diagnose import _diagnostic_dataset, _inputs, _omit_last_prefix
from maichart.composer.quality_inputs import QUALITY_FORWARD_KEYS
from test_composer_quality_integration import _fixture
from test_composer_event_views import dataset_fixture, points


def test_diagnostic_inputs_enable_new_model_without_passing_supervision():
    model, data = _fixture()
    batch = {key: value for key, value in data.items() if key != 'quality_inputs'}
    batch.update(data['quality_inputs'])
    batch['candidate_padding_mask'] = data['candidate_mask']
    batch['history_beat_ages'] = torch.tensor([[1., 0.], [1., 0.]])
    packed = _inputs(batch, 'cpu')
    assert set(packed['quality_inputs']) == set(QUALITY_FORWARD_KEYS) & set(batch)
    assert 'quality_phrase_target' not in packed['quality_inputs']
    torch.testing.assert_close(packed['history_beat_ages'], batch['history_beat_ages'])
    result = model(**packed, decode_actions=False)
    assert result['quality_gap_logits'].shape == (2, 6, 7)
    assert torch.isfinite(result['candidate_context']).all()


def test_diagnostic_dataset_inherits_quality_policy(monkeypatch):
    import maichart.composer.diagnose as module
    model, _ = _fixture()
    seen = {}
    monkeypatch.setattr(module, 'ComposerDataset', lambda *args, **kwargs: seen.update(kwargs))
    _diagnostic_dataset(SimpleNamespace(manifest='m', cache_dir=None, limit_charts=0), model)
    assert seen['quality_inputs_enabled'] is True
    model.config.configuration_quality_policy = 'off'
    _diagnostic_dataset(SimpleNamespace(manifest='m', cache_dir=None, limit_charts=0), model)
    assert seen['quality_inputs_enabled'] is False


def test_omitted_event_rebuilds_anchors_and_lifecycle_and_maps_catalog(tmp_path, monkeypatch):
    dataset = dataset_fixture(tmp_path, monkeypatch,
        source=points((0, '1'), (1, '2h[1:1]'), (2, '3'), (3, '4')))
    dataset.quality_inputs_enabled = True
    sample_index = next(i for i, sample in enumerate(dataset.samples) if sample.event_index == 2)
    sample = dataset.samples[sample_index]
    item = dataset[sample_index]
    assert item['quality_anchor_mask'].sum() == 2
    assert item['quality_lifecycle'][:, 41].max() > 0
    # A deliberately different checkpoint order proves anchors are remapped.
    id_map = np.arange(len(dataset.codec_atoms) + 1, dtype=np.int64)
    id_map[1:] = id_map[1:][::-1]
    changed, _seconds, _beats = _omit_last_prefix(dataset, sample, item, id_map)
    assert changed['quality_anchor_mask'].sum() == 1
    assert changed['quality_lifecycle'].count_nonzero() == 0
    expected_id = id_map[dataset.atom_to_id['1']]
    assert changed['quality_anchor_ids'][0, 0] == expected_id
    assert changed['quality_anchor_ids'][1:].count_nonzero() == 0
    assert changed['quality_phrase_weight'].count_nonzero() == 0
    assert changed['quality_gap_weight'] == 0
    assert len(changed['history_beat_ages']) == len(changed['history']) == 1


def test_old_diagnostic_inputs_need_no_quality_fields():
    batch = {'history': torch.zeros(1, 0, 3, dtype=torch.long),
             'history_padding_mask': torch.zeros(1, 0, dtype=torch.bool)}
    assert 'quality_inputs' not in _inputs(batch, 'cpu')


def test_omitted_event_rebuilds_song_paired_history_clock(tmp_path, monkeypatch):
    dataset = dataset_fixture(tmp_path, monkeypatch,
        source=points((0, '1'), (1, '2'), (2, '3'), (3, '4')))
    dataset.architecture_profile = 'audio_grounded_v3'
    dataset.quality_inputs_enabled = True
    dataset.quality_inputs_policy = 'song_paired_v3'
    index = next(i for i, sample in enumerate(dataset.samples) if sample.event_index == 2)
    sample, item = dataset.samples[index], dataset[index]
    assert item['quality_history_clock'].size(0) == 2
    changed, _, _ = _omit_last_prefix(dataset, sample, item, np.arange(len(dataset.codec_atoms)+1))
    assert changed['quality_history_audio'].size(0) == changed['history'].size(0) == 1
    assert changed['quality_history_clock'].tolist() == [[0., 0.]]
    assert 'quality_anchor_ids' not in changed
