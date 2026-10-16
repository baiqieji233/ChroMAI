"""Bounded single-producer layout batches sharing resident audio in-process.

Only CPU samples are prepared ahead. The caller remains the sole owner of
CUDA forward/backward and the consumed-batch checkpoint cursor. Closing joins
the producer before a new epoch can change the dataset seed.
"""
from __future__ import annotations

import threading
import time

import torch

from maichart.form.blueprint_prefetch import OrderedBlueprintPrefetch


def _tensor_bytes(batch):
    return sum(value.numel()*value.element_size() for value in batch.values()
               if isinstance(value, torch.Tensor))


class LayoutBatchPrefetch:
    """Prefetch a zero-worker DataLoader without copying its dataset or audio.

    One producer serializes dataset caches and random per-sample preparation.
    DataLoader's explicit generator and the dataset's per-index RNG preserve
    training RNG and order. ``depth`` bounds pending batches, not whole songs.
    """

    def __init__(self, loader, *, depth=2):
        if getattr(loader, 'num_workers', 0):
            raise ValueError('Layout thread prefetch requires loader_workers=0')
        if isinstance(depth, bool) or not isinstance(depth, int) or not 1 <= depth <= 16:
            raise ValueError('Layout prefetch depth must be in [1, 16]')
        self._source = iter(loader)
        self._length = len(loader)
        self.depth = depth
        self._lock = threading.Lock()
        self._prepared = 0
        self._prepare_seconds = self._wait_seconds = 0.
        self._buffered_bytes = self._max_buffered_bytes = 0
        self._stream = None

    def _prepare(self, _index):
        started = time.perf_counter()
        batch = next(self._source)
        size = _tensor_bytes(batch)
        with self._lock:
            self._prepared += 1
            self._prepare_seconds += time.perf_counter()-started
            self._buffered_bytes += size
            self._max_buffered_bytes = max(self._max_buffered_bytes, self._buffered_bytes)
        return batch

    def __enter__(self):
        self._stream = OrderedBlueprintPrefetch(range(self._length), self._prepare,
                                                workers=1, depth=self.depth)
        return self

    def __iter__(self):
        return self

    def __next__(self):
        if self._stream is None:
            raise RuntimeError('Use LayoutBatchPrefetch as a context manager')
        started = time.perf_counter()
        batch = next(self._stream)
        with self._lock:
            self._wait_seconds += time.perf_counter()-started
            self._buffered_bytes -= _tensor_bytes(batch)
        return batch

    def stats(self):
        with self._lock:
            return {'mode': 'single_thread_shared_audio', 'depth': self.depth,
                    'prepared_batches': self._prepared,
                    'prepare_sec': round(self._prepare_seconds, 3),
                    'consumer_wait_sec': round(self._wait_seconds, 3),
                    'buffered_tensor_bytes': self._buffered_bytes,
                    'max_buffered_tensor_bytes': self._max_buffered_bytes}

    def close(self):
        if self._stream is not None:
            self._stream.close(cancel=True)
            self._stream = None
        close = getattr(self._source, 'close', None)
        if close is not None:
            close()

    def __exit__(self, *_exc):
        self.close()
