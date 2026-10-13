from fractions import Fraction as F
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader

from maichart.event_timing import EventPoint
from maichart.form.codec import RationalCodec
from maichart.form.data import _AudioStore, _LazyAudio, _SampleStore
from maichart.form.timing import metrical_features
from maichart.generation.event_audio import ACOUSTIC_DIM, EventAudio


def _audio(value):
    features = np.full((4, ACOUSTIC_DIM), value, dtype=np.float32)
    return EventAudio(features, np.arange(4, dtype=np.float32), 1., 120.)


def _pickle_audio_loader(path, **_kwargs):
    return _audio(7)


def _dataset_audio_loader(path, **_kwargs):
    times = np.arange(150, dtype=np.float32) * 0.02
    features = np.zeros((len(times), ACOUSTIC_DIM), dtype=np.float32)
    return EventAudio(features, times, 3., 120.)


def test_bounded_memory_stops_encoding_after_the_first_over_budget_cell():
    codec = RationalCodec()
    points = [EventPoint(F(i), '123456789') for i in range(40)]
    original = [codec.cell(point, origin=F(0)) for point in points]

    def reference(values, newest):
        selected, used = [], 2
        for cell in reversed(values) if newest else values:
            if used + len(cell) > 32:
                break
            selected.append(cell)
            used += len(cell)
        if newest:
            selected.reverse()
        return [codec.bos_id] + [token for cell in selected for token in cell] + [codec.eos_id] if selected else [0]

    assert codec.bounded_memory(points, max_tokens=32, newest=False) == reference(original, False)
    assert codec.bounded_memory(points, max_tokens=32, newest=True) == reference(original, True)


def test_unknown_meter_uses_fast_path_with_exact_quarter_beat_features():
    beats = [F(0), F(1, 4), F(1, 2), F(3, 4), F(5, 3)]
    actual = metrical_features(beats, [])
    expected = np.zeros((len(beats), 8), dtype=np.float32)
    for row, beat in enumerate(beats):
        angle = 2 * np.pi * float(beat % 1)
        expected[row, :2] = (np.sin(angle), np.cos(angle))
    np.testing.assert_array_equal(actual, expected)


def test_audio_lru_reports_hits_misses_evictions_and_pickles_lock():
    calls = []

    def load(path, **_kwargs):
        calls.append(str(path))
        return _audio(len(calls))

    per_entry = _audio(1).features.nbytes + _audio(1).times.nbytes
    store = _AudioStore(load, 'cache', budget=per_entry)
    first = store.get('a')
    assert store.get('a') is first
    store.get('b')
    assert store.stats() == {'hits': 1, 'misses': 2, 'loads': 2,
                             'evictions': 1, 'entries': 1,
                             'bytes': per_entry, 'budget_bytes': per_entry}

    restored = pickle.loads(pickle.dumps(_AudioStore(_pickle_audio_loader, 'cache', budget=per_entry)))
    assert restored.stats()['budget_bytes'] == per_entry


def test_lazy_audio_is_safe_to_pickle_before_first_attribute_access():
    store = _AudioStore(_pickle_audio_loader, 'cache')
    store.get('asset')
    lazy = pickle.loads(pickle.dumps(_LazyAudio(store, 'asset')))
    assert lazy.features[0, 0] == 7
    assert lazy.store.stats()['entries'] == 1


def test_sample_cache_is_bounded_and_returns_independent_tensors():
    cache = _SampleStore(40)
    source = {'value': np.array([1, 2]), 'tensor': torch.tensor([3, 4])}
    cache.put('a', source)
    first = cache.get('a')
    assert cache.stats()['hits'] == 1
    first['tensor'][0] = 99
    assert cache.get('a')['tensor'][0].item() == 3
    cache.put('b', {'tensor': torch.zeros(8)})
    assert cache.stats()['entries'] == 1
    assert cache.stats()['evictions'] == 1
    spawned = pickle.loads(pickle.dumps(cache))
    assert spawned.stats()['entries'] == 0
    assert spawned.stats()['budget_bytes'] == 40


def test_spawn_workers_reproduce_revision_and_causal_memory(tmp_path):
    import json
    from maichart.form.data import FormDataset, collate_form

    song_dir = tmp_path / 'song'
    song_dir.mkdir()
    (song_dir / 'maidata.txt').write_text(
        '&title=fixture\n&wholebpm=120\n&lv_5=13\n'
        '&inote_5=(120){12}1,2,3,1,2,3,1h[4:4],2,3,1,2,3,E\n',
        encoding='utf-8',
    )
    (song_dir / 'track.wav').write_bytes(b'fixture')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'songs': [{
        'song_id': 'fixture', 'maidata_path': 'song/maidata.txt',
        'audio_path': 'song/track.wav',
        'difficulties': [{'difficulty_index': 5, 'level_raw': '13'}],
    }]}), encoding='utf-8')
    dataset = FormDataset(manifest, cache_dir=tmp_path / 'cache',
                          audio_loader=_dataset_audio_loader, max_region_seconds=3.)
    revision_index = next(i for i, item in enumerate(dataset.items) if item[3] == 1)
    expected = dataset[revision_index]
    loader = DataLoader(dataset, batch_size=1, sampler=[revision_index],
                        num_workers=2, multiprocessing_context='spawn',
                        persistent_workers=False, collate_fn=collate_form)
    iterator = iter(loader)
    try:
        actual = next(iterator)
    finally:
        iterator._shutdown_workers()

    for key in ('history', 'motif', 'draft', 'right', 'skeleton', 'action_state'):
        assert torch.equal(actual[key][0, :len(expected[key])], expected[key])
