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
