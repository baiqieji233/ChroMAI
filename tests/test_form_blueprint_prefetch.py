from __future__ import annotations

import threading
import time

import pytest

from maichart.form.blueprint_prefetch import (
    OrderedBlueprintPrefetch,
    iter_prefetched_blueprints,
    prefetch_blueprints,
)


def test_ordered_prefetch_preserves_order_and_bounds_queue():
    active = 0
    maximum = 0
    lock = threading.Lock()

    def prepare(index):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        # Reverse completion order to prove that yields remain input ordered.
        time.sleep(.002 * (5 - (index % 5)))
        with lock:
            active -= 1
        return index

    indices = list(range(15))
    with OrderedBlueprintPrefetch(indices, prepare, workers=3, depth=4) as stream:
        result = list(stream)
        stats = stream.stats
    assert result == indices
    assert maximum <= 3
    assert stats.submitted == len(indices)
    assert stats.completed == len(indices)
    assert stats.max_in_flight <= 4


def test_prefetch_preparation_error_is_propagated_and_workers_close():
    def prepare(index):
        if index == 2:
            raise ValueError("bad chart")
        return index

    stream = OrderedBlueprintPrefetch(range(10), prepare, workers=2, depth=3)
    with pytest.raises(ValueError, match="bad chart"):
        list(stream)
    assert stream.stats.completed < stream.stats.submitted


def test_zero_workers_is_synchronous_and_ordered():
    calls = []

    def prepare(index):
        calls.append(index)
        return index * 2

    with OrderedBlueprintPrefetch(range(4), prepare, workers=0) as stream:
        assert list(stream) == [0, 2, 4, 6]
    assert calls == [0, 1, 2, 3]


def test_single_worker_prepares_ahead_while_consumer_is_busy():
    prepared_second = threading.Event()
    release_second = threading.Event()

    def prepare(index):
        if index == 1:
            prepared_second.set()
            assert release_second.wait(2)
        return index

    with OrderedBlueprintPrefetch(range(3), prepare, workers=1, depth=2) as stream:
        assert next(stream) == 0
        assert prepared_second.wait(2)
        release_second.set()
        assert list(stream) == [1, 2]


def test_early_close_cancels_pending_and_waits_for_producer():
    release = threading.Event()

    def prepare(index):
        if index == 0:
            assert release.wait(2)
        return index

    stream = OrderedBlueprintPrefetch(range(20), prepare, workers=1, depth=2)
    release.set()
    stream.close(cancel=True)
    assert list(stream) == []
    assert stream.stats.submitted == 2


@pytest.mark.parametrize("kwargs", [{"workers": -1}, {"depth": 0}])
def test_invalid_prefetch_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        OrderedBlueprintPrefetch([], lambda x: x, **kwargs)


def test_public_indexed_interface_yields_indices_and_prepared_rows():
    with iter_prefetched_blueprints([3, 1, 2], lambda index: index * 10,
                                    workers=1, depth=2) as stream:
        rows = list(stream)
        stats = stream.stats
    assert rows == [(3, 30), (1, 10), (2, 20)]
    assert stats.submitted == stats.completed == 3
    assert stats.max_in_flight == 2


def test_training_loop_compatibility_api_yields_indexed_results_and_can_close():
    stream = prefetch_blueprints([4, 7], lambda index: index + 1,
                                 workers=1, depth=2)
    assert next(stream) == (4, 5)
    assert next(stream) == (7, 8)
    stream.close()

