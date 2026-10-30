import numpy as np
import pytest

from maichart.composer.workload import prefix_ledger, prefix_ledger_batch


def test_batched_ledger_uses_each_rows_committed_prefix_without_future_events():
    times = np.asarray([.1, .5, .5, 1., 2., 5.])
    sums = np.r_[0., np.cumsum([1., 2., 1., 3., 1., 2.])]
    queries = np.asarray([[-.2, .1, .5, 2., 8.]] * 7)
    counts = np.arange(7)
    expected = np.stack([prefix_ledger(times, sums, row, committed_count=int(count))
                         for row, count in zip(queries, counts)])
    actual = prefix_ledger_batch(times, sums, queries, counts)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('shape,counts', [((0, 3), []), ((2, 0), [0, 0]), ((2, 3), [0, 0])])
def test_batched_ledger_empty_prefix_and_queries(shape, counts):
    actual = prefix_ledger_batch([], [0.], np.zeros(shape), counts)
    np.testing.assert_array_equal(actual, np.zeros((*shape, 6), dtype=np.float32))
