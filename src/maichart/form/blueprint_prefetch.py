"""Bounded, ordered CPU preparation for whole-song Blueprint training.

Blueprint inputs are relatively expensive to assemble (audio queries, curve
resampling and descriptors), while the planner forward itself is small.  This
module overlaps that CPU work with the caller's GPU work without changing the
dataset order or moving CUDA objects into worker threads.

The producer is deliberately a :class:`ThreadPoolExecutor`: ``FormDataset``
owns process-local audio/sample caches and the audio feature libraries release
the GIL during their numeric work.  A bounded queue prevents a full epoch from
being materialized in memory and preserves deterministic resume cursors.
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from collections import deque
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, TypeVar


T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True)
class BlueprintPrefetchStats:
    """Counters useful for logging without exposing executor internals."""

    submitted: int = 0
    completed: int = 0
    max_in_flight: int = 0


class OrderedBlueprintPrefetch(Iterator[R]):
    """Prepare indexed Blueprint records in order with a bounded lookahead.

    ``prepare`` is called as ``prepare(index)`` on worker threads.  Results are
    yielded in the exact order of ``indices`` even if a later chart completes
    first.  At most ``depth`` futures are admitted, so a slow chart cannot
    cause unbounded memory growth.  ``workers=0`` uses a synchronous iterator;
    the default single producer overlaps preparation and training without
    scheduling concurrent operations on shared mutable audio view caches.
    """

    def __init__(self, indices: Iterable[T], prepare: Callable[[T], R], *,
                 workers: int = 1, depth: int | None = None,
                 return_index: bool = False):
        if workers < 0:
            raise ValueError("workers must be nonnegative")
        if depth is not None and depth < 1:
            raise ValueError("depth must be positive when provided")
        self._indices = iter(indices)
        self._prepare = prepare
        self._return_index = bool(return_index)
        self._workers = int(workers)
        # A small multiple of workers is enough to hide a single slow audio
        # query without retaining a full epoch of tensors.
        self._depth = max(1, int(depth if depth is not None else max(2, workers * 2)))
        self._executor: ThreadPoolExecutor | None = None
        self._pending: deque[Future[R]] = deque()
        self._closed = False
        self._submitted = self._completed = self._max_in_flight = 0
        if self._workers:
            self._executor = ThreadPoolExecutor(
                max_workers=self._workers, thread_name_prefix="song-form-blueprint"
            )
            self._fill()

    @property
    def stats(self) -> BlueprintPrefetchStats:
        return BlueprintPrefetchStats(self._submitted, self._completed,
                                      self._max_in_flight)

    def _fill(self) -> None:
        if self._executor is None:
            return
        while len(self._pending) < self._depth:
            try:
                index = next(self._indices)
            except StopIteration:
                break
            self._pending.append(self._executor.submit(self._prepare_index, index))
            self._submitted += 1
            self._max_in_flight = max(self._max_in_flight, len(self._pending))

    def _prepare_index(self, index):
        result = self._prepare(index)
        return (index, result) if self._return_index else result

    def __iter__(self) -> "OrderedBlueprintPrefetch[R]":
        return self

    def __next__(self) -> R:
        if self._closed:
            raise StopIteration
        if self._executor is None:
            try:
                index = next(self._indices)
            except StopIteration:
                self.close()
                raise
            self._submitted += 1
            try:
                result = self._prepare(index)
            except BaseException:
                self.close(cancel=True)
                raise
            self._completed += 1
            return (index, result) if self._return_index else result
        if not self._pending:
            self.close()
            raise StopIteration
        future = self._pending.popleft()
        try:
            result = future.result()
        except BaseException:
            # The caller must see the original preparation exception.  Cancel
            # work that has not started, then shut down before re-raising.
            self.close(cancel=True)
            raise
        self._completed += 1
        self._fill()
        return result

    def close(self, *, cancel: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        if cancel:
            for future in self._pending:
                future.cancel()
        self._pending.clear()
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

    def __enter__(self) -> "OrderedBlueprintPrefetch[R]":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close(cancel=exc_type is not None)


def iter_prefetched_blueprints(indices: Iterable[T], prepare: Callable[[T], R], *,
                               workers: int = 1, depth: int | None = None) -> Iterator[tuple[T, R]]:
    """Context-manager iterator yielding ``(index, prepared)`` in order.

    The returned iterator must be used with ``with`` so an interrupted training
    run reliably cancels pending preparation futures.  The index is yielded
    alongside the result to make logging and resume cursor handling explicit.
    """
    class _Context:
        def __init__(self):
            self._indices = iter(indices)
            self._prepare = prepare
            self._stream = None

        def _start(self):
            if self._stream is None:
                self._stream = OrderedBlueprintPrefetch(
                    self._indices, self._prepare, workers=workers, depth=depth,
                    return_index=True)
            return self._stream

        def __enter__(self):
            self._start().__enter__()
            return self

        def __iter__(self):
            return self

        def __next__(self):
            return next(self._start())

        @property
        def stats(self):
            return self._start().stats

        def close(self, *, cancel=False):
            if self._stream is not None:
                self._stream.close(cancel=cancel)

        def __exit__(self, exc_type, exc_value, traceback):
            self.close(cancel=exc_type is not None)

    return _Context()


def prefetch_blueprints(indices: Iterable[T], prepare: Callable[[T], R], *,
                        workers: int = 1, depth: int | None = None) -> Iterator[tuple[T, R]]:
    """Return the indexed prefetch stream used by the v2 training loop.

    This legacy-friendly name intentionally matches
    ``iter_prefetched_blueprints``: each yielded row is ``(index, result)``.
    """
    return iter_prefetched_blueprints(indices, prepare, workers=workers, depth=depth)

