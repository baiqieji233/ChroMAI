"""CPU prefetch must preserve order/RNG and stop before the next epoch."""
import threading
import time

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from maichart.form.layout_prefetch import LayoutBatchPrefetch


class SeededRows(Dataset):
    def __init__(self, n=12, fail=None):
        self.n, self.fail = n, fail
        self.owner = set()

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        self.owner.add(threading.current_thread().ident)
        if index == self.fail:
            raise ValueError('original preparation failure')
        return {'value': torch.rand(3, generator=torch.Generator().manual_seed(77+index)),
                'index': torch.tensor(index)}

    def prepare_concurrency_key(self, index):
        return index % 3


def loader(dataset):
    return DataLoader(dataset, batch_size=2, num_workers=0,
                      generator=torch.Generator().manual_seed(33))


def test_prefetch_values_order_and_training_rng_equal_synchronous_loader():
    reference = list(loader(SeededRows()))
    dataset = SeededRows()
    before = torch.get_rng_state().clone()
    with LayoutBatchPrefetch(loader(dataset), depth=2) as stream:
        actual = list(stream)
        stats = stream.stats()
    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
    assert len(dataset.owner) == 1
    assert threading.current_thread().ident not in dataset.owner
    assert stats['prepared_batches'] == 6
    assert stats['max_buffered_tensor_bytes'] <= 3*(2*3*4+2*8)
    for a, b in zip(reference, actual):
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)


def test_prefetch_does_not_leave_producer_running_on_early_exit():
    dataset = SeededRows(100)
    with LayoutBatchPrefetch(loader(dataset), depth=2) as stream:
        assert next(stream)['index'].tolist() == [0, 1]
    count = stream.stats()['prepared_batches']
    time.sleep(.03)
    assert stream.stats()['prepared_batches'] == count
    assert count <= 3


def test_prefetch_propagates_original_preparation_error():
    with pytest.raises(ValueError, match='original preparation failure'):
        with LayoutBatchPrefetch(loader(SeededRows(fail=2)), depth=2) as stream:
            assert next(stream)['index'].tolist() == [0, 1]
            next(stream)


def test_prefetch_handles_empty_dataset():
    with LayoutBatchPrefetch(loader(SeededRows(0)), depth=2) as stream:
        assert list(stream) == []


@pytest.mark.parametrize('depth', [0, -1, 17, True])
def test_prefetch_depth_is_explicitly_bounded(depth):
    with pytest.raises(ValueError, match='depth'):
        LayoutBatchPrefetch(loader(SeededRows()), depth=depth)


def test_parallel_sample_preparation_preserves_order_values_and_rng():
    expected = list(loader(SeededRows(20)))
    before = torch.get_rng_state().clone()
    dataset = SeededRows(20)
    with LayoutBatchPrefetch(loader(dataset), depth=2, sample_workers=4) as stream:
        actual = list(stream)
        assert stream.stats()['sample_workers'] == 4
    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
    for a, b in zip(expected, actual):
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)


def test_parallel_sample_preparation_serializes_the_same_source():
    class GuardedRows(SeededRows):
        def __init__(self):
            super().__init__(20)
            self.active = set()
            self.lock = threading.Lock()
            self.maximum_active = 0

        def __getitem__(self, index):
            key = self.prepare_concurrency_key(index)
            with self.lock:
                assert key not in self.active
                self.active.add(key)
                self.maximum_active = max(self.maximum_active, len(self.active))
            time.sleep(.01)
            result = super().__getitem__(index)
            with self.lock:
                self.active.remove(key)
            return result

    dataset = GuardedRows()
    multi_loader = DataLoader(dataset, batch_size=8, num_workers=0,
                              generator=torch.Generator().manual_seed(33))
    with LayoutBatchPrefetch(multi_loader, depth=2, sample_workers=4) as stream:
        assert sum(len(batch['index']) for batch in stream) == 20
    assert dataset.maximum_active > 1
    assert not dataset.active


def test_parallel_preparation_propagates_errors_and_joins_workers():
    with pytest.raises(ValueError, match='original preparation failure'):
        with LayoutBatchPrefetch(loader(SeededRows(fail=2)), depth=2, sample_workers=4) as stream:
            list(stream)
    assert not any(thread.name.startswith('song-form-sample') for thread in threading.enumerate())


@pytest.mark.parametrize('workers', [0, -1, 9, True])
def test_sample_worker_count_is_bounded(workers):
    with pytest.raises(ValueError, match='sample workers'):
        LayoutBatchPrefetch(loader(SeededRows()), sample_workers=workers)


@pytest.mark.parametrize('workers', [1, 4])
def test_worker_gate_pauses_and_auto_resumes_on_next_and_close(workers):
    expected = list(loader(SeededRows(12)))
    dataset = SeededRows(12)
    actual = []
    with LayoutBatchPrefetch(loader(dataset), depth=2, sample_workers=workers) as stream:
        for idx in range(len(expected)):
            batch = next(stream)
            stream.pause_workers()
            if idx % 2 == 0:
                stream.resume_workers()
            actual.append(batch)
        stream.pause_workers()
    for a, b in zip(expected, actual):
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)


@pytest.mark.parametrize('workers', [1, 4])
@pytest.mark.parametrize('batch_enabled', [False, True])
def test_optional_dataset_batch_fetch_preserves_order_and_bounds_source_chunks(workers, batch_enabled):
    class BatchedRows(SeededRows):
        def __init__(self):
            super().__init__(41)
            self.batch_calls = []
            self.scalar_calls = []
            if not batch_enabled:
                self.__getitems__ = None

        def __getitem__(self, index):
            self.scalar_calls.append(index)
            return super().__getitem__(index)

        def __getitems__(self, indices):
            self.batch_calls.append(list(indices))
            # Deliberately bypass __getitem__ so this proves which path ran.
            return [SeededRows.__getitem__(self, index) for index in indices]

    dataset = BatchedRows()
    order = list(range(40, -1, -1))
    source = DataLoader(dataset, batch_sampler=[order], num_workers=0,
                        generator=torch.Generator().manual_seed(33))
    before = torch.get_rng_state().clone()
    with LayoutBatchPrefetch(source, depth=1, sample_workers=workers) as stream:
        batches = list(stream)
    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
    assert len(batches) == 1
    assert batches[0]['index'].tolist() == order
    for pos, index in enumerate(order):
        torch.testing.assert_close(batches[0]['value'][pos],
                                   SeededRows.__getitem__(dataset, index)['value'], rtol=0, atol=0)
    if batch_enabled:
        assert not dataset.scalar_calls
        assert sorted(index for chunk in dataset.batch_calls for index in chunk) == list(range(41))
        if workers == 1:
            assert dataset.batch_calls == [order[start:start + 8] for start in range(0, len(order), 8)]
        else:
            assert len(dataset.batch_calls) == 6
            assert all(1 <= len(chunk) <= 8 for chunk in dataset.batch_calls)
            assert all(len({dataset.prepare_concurrency_key(index) for index in chunk}) == 1
                       for chunk in dataset.batch_calls)
    else:
        assert not dataset.batch_calls
        assert sorted(dataset.scalar_calls) == list(range(41))


def test_batch_source_group_checks_gate_before_each_chunk():
    dataset = SeededRows(19)
    calls = []

    class CountingGate:
        waits = 0

        def wait(self):
            self.waits += 1

        def set(self):
            pass

    source = loader(dataset)
    stream = LayoutBatchPrefetch(source, sample_workers=2)
    gate = CountingGate()
    stream._worker_gate = gate

    def fetch(indices):
        calls.append((gate.waits, list(indices)))
        return [SeededRows.__getitem__(dataset, index) for index in indices]

    dataset.__getitems__ = fetch
    try:
        rows = stream._sample_group(('same-source', list(enumerate(range(19)))))
        assert [pos for pos, _ in rows] == list(range(19))
        assert [(waits, len(indices)) for waits, indices in calls] == [(1, 8), (2, 8), (3, 3)]
    finally:
        stream.close()
