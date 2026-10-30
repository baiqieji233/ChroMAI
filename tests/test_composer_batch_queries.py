"""Exact CPU batch assembly against the existing per-sample contract."""
import json
import random

import numpy as np
import pytest
import torch

from maichart.composer.data import ComposerDataset, collate_composer
from maichart.composer.model import GROUNDED_PROFILE
from maichart.composer import structure
from maichart.composer.workload import ExactWorkloadQuery


@pytest.fixture(scope='module')
def batch_manifest(tmp_path_factory):
    import soundfile as sf

    root = tmp_path_factory.mktemp('composer-batch')
    songs = []
    sr = 8000
    seconds = np.arange(sr * 4) / sr
    for index in range(2):
        audio = root / f'track-{index}.wav'
        sf.write(audio, (.1 * np.sin(2 * np.pi * (330 + index * 110) * seconds)).astype(np.float32), sr)
        chart = root / f'maidata-{index}.txt'
        chart.write_text('&wholebpm=120\n&lv_5=12\n&inote_5=(120){4}'
                         '1,2,3,4,(150)5,6,7,8,1/2,3,4,5,E\n', encoding='utf8')
        songs.append(dict(song_id=f'batch-fixture-{index}', maidata_path=str(chart),
                          audio_path=str(audio), difficulties=[dict(difficulty_index=5, level_raw='12')]))
    manifest = root / 'manifest.json'
    manifest.write_text(json.dumps(dict(songs=songs)), encoding='utf8')
    return manifest, root / 'cache'


def _assert_tensors_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        if torch.is_tensor(value):
            assert torch.is_tensor(actual[key]), key
            assert actual[key].dtype == value.dtype, key
            torch.testing.assert_close(actual[key], value, rtol=0, atol=0, msg=key)


@pytest.mark.parametrize('deferred', [False, True])
@pytest.mark.parametrize('policy', ['off', 'metrical_context_v1', 'metrical_context_v2'])
def test_batch_queries_match_scalar_all_tensor_fields_order_rng_and_query_counts(
        batch_manifest, monkeypatch, deferred, policy):
    manifest, cache = batch_manifest
    dataset = ComposerDataset(manifest, cache_dir=cache, candidate_count=16,
        history_events=8, action_slots=4, preparation_workers=1,
        architecture_profile=GROUNDED_PROFILE, quality_enabled=True,
        quality_inputs_enabled=True, defer_audio_queries=deferred,
        structure_policy=policy, structure_backend='native', seed=191)
    dataset.epoch = 3
    groups = {}
    for index, sample in enumerate(dataset.samples):
        groups.setdefault(sample.song_index, []).append(index)
    assert len(groups) == 2
    first, second = groups.values()
    assert min(len(first), len(second)) >= 3
    indices = [second[-1], first[len(first) // 2], second[0], first[-1], second[-1], first[0]]
    expected = [dataset[index] for index in indices]
    workload_calls, structure_calls = [], []
    query = ExactWorkloadQuery.query
    structure_batch = structure.structure_features_and_transition_batch

    def count_workload(self, seconds, *args, **kwargs):
        workload_calls.append(len(seconds))
        return query(self, seconds, *args, **kwargs)

    def count_structure(song, beats, seconds, *args, **kwargs):
        structure_calls.append(np.asarray(beats).shape)
        return structure_batch(song, beats, seconds, *args, **kwargs)

    monkeypatch.setattr(ExactWorkloadQuery, 'query', count_workload)
    monkeypatch.setattr(structure, 'structure_features_and_transition_batch', count_structure)
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.get_rng_state().clone()
    actual = dataset.__getitems__(indices)
    assert random.getstate() == python_rng
    after_numpy = np.random.get_state()
    assert after_numpy[0] == numpy_rng[0] and after_numpy[2:] == numpy_rng[2:]
    np.testing.assert_array_equal(after_numpy[1], numpy_rng[1])
    torch.testing.assert_close(torch.get_rng_state(), torch_rng, rtol=0, atol=0)
    assert len(actual) == len(expected)
    for row, reference in zip(actual, expected):
        _assert_tensors_equal(row, reference)
    _assert_tensors_equal(collate_composer(actual), collate_composer(expected))
    # Each source has three requested rows, including the repeated last row.
    assert workload_calls == [3 * (dataset.candidate_count + 16)] * 2
    assert structure_calls == ([(3, dataset.candidate_count)] * 2 if policy != 'off' else [])
    workload_calls.clear()
    structure_calls.clear()
    assert dataset.__getitems__([]) == []
    assert not workload_calls and not structure_calls
