from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import os
import pickle
import threading

import torch

from maichart.form.data import FormDataset, _LazyAudio, _SampleStore


def test_sample_store_concurrent_mutations_preserve_budget_and_copies():
    cache = _SampleStore(256)
    barrier = threading.Barrier(6)
    def work(worker):
        barrier.wait()
        for turn in range(120):
            key = (turn+worker)%11
            original = {'value': torch.full((8,), float(worker)), 'meta': [worker]}
            cache.put(key, original)
            original['value'].fill_(-100)
            actual = cache.get(key)
            if actual is not None:
                assert torch.all(actual['value'] >= 0)
                actual['value'].fill_(-200)
            stats = cache.stats()
            assert 0 <= stats['bytes'] <= stats['budget_bytes']
        return True
    with ThreadPoolExecutor(max_workers=6) as pool:
        assert all(pool.map(work, range(6)))
    with cache._lock:
        assert cache.bytes == sum(size for _, size in cache.entries.values())
        assert len(cache.entries) <= 8
        assert all(torch.all(value['value'] >= 0) for value, _ in cache.entries.values())
    assert cache.stats()['hits']+cache.stats()['misses'] == 720


def test_sample_store_spawn_serialization_retains_empty_independent_cache():
    cache = _SampleStore(256)
    cache.put('sample', {'tensor': torch.ones(4)})
    restored = pickle.loads(pickle.dumps(cache))
    assert restored.stats() == dict(hits=0, misses=0, evictions=0, entries=0,
                                    bytes=0, budget_bytes=256)
    restored.put('other', {'tensor': torch.zeros(4)})
    assert restored.get('other')['tensor'].sum() == 0
    assert cache.get('sample')['tensor'].sum() == 4


def _dataset(*audios):
    dataset = FormDataset.__new__(FormDataset)
    dataset.songs = [(SimpleNamespace(audio=audio),) for audio in audios]
    dataset.items = [(i, 0, [], 0) for i in range(len(audios))]
    return dataset


def test_prepare_concurrency_key_is_same_for_same_lazy_source_without_loading(tmp_path):
    class NoLoading:
        def get(self, path):
            raise AssertionError('Scheduling must not resolve or load audio')
    a = tmp_path/'song'/'track.mp3'
    dataset = _dataset(_LazyAudio(NoLoading(), a), _LazyAudio(NoLoading(), str(a)),
                       _LazyAudio(NoLoading(), tmp_path/'other.mp3'))
    key = dataset.prepare_concurrency_key(0)
    assert key == os.path.normcase(os.path.abspath(str(a)))
    assert dataset.prepare_concurrency_key(1) == key
    assert dataset.prepare_concurrency_key(2) != key


def test_prepare_concurrency_key_supports_loaded_and_in_memory_audio(tmp_path):
    source = tmp_path/'track.mp3'
    shared = SimpleNamespace(features=torch.zeros(1))
    dataset = _dataset(SimpleNamespace(source=str(source)),
                       SimpleNamespace(metadata={'source': str(source)}), shared, shared,
                       SimpleNamespace(features=torch.zeros(1)))
    assert dataset.prepare_concurrency_key(0) == dataset.prepare_concurrency_key(1)
    assert dataset.prepare_concurrency_key(2) == dataset.prepare_concurrency_key(3)
    assert dataset.prepare_concurrency_key(2) != dataset.prepare_concurrency_key(4)
