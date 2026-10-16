"""Bounded CPU layout preparation sharing resident audio in-process.

Only CPU samples are prepared ahead. The caller remains the sole owner of
CUDA forward/backward and the consumed-batch checkpoint cursor. Closing joins
the producer before a new epoch can change the dataset seed.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import torch

from maichart.form.blueprint_prefetch import OrderedBlueprintPrefetch


def _tensor_bytes(batch):
    return sum(value.numel()*value.element_size() for value in batch.values()
               if isinstance(value, torch.Tensor))


class LayoutBatchPrefetch:
    """Prefetch a zero-worker DataLoader without copying its dataset or audio.

    One producer preserves batch order. Optional sample threads prepare
    independent audio sources concurrently, serializing each shared source.
    DataLoader's explicit generator and the dataset's per-index RNG preserve
    training RNG and order. ``depth`` bounds pending batches, not whole songs.
    """

    def __init__(self, loader, *, depth=2, sample_workers=1):
        if getattr(loader, 'num_workers', 0):
            raise ValueError('Layout thread prefetch requires loader_workers=0')
        if isinstance(depth, bool) or not isinstance(depth, int) or not 1 <= depth <= 16:
            raise ValueError('Layout prefetch depth must be in [1, 16]')
        if (isinstance(sample_workers, bool) or not isinstance(sample_workers, int)
                or not 1 <= sample_workers <= 8):
            raise ValueError('Layout sample workers must be in [1, 8]')
        if sample_workers > 1 and not callable(getattr(loader.dataset, 'prepare_concurrency_key', None)):
            raise ValueError('Concurrent layout preparation requires explicit dataset source keys')
        self._source = iter(loader)
        self._loader = loader
        self._length = len(loader)
        self.depth = depth
        self.sample_workers = sample_workers
        self._lock = threading.Lock()
        self._source_locks_guard = threading.Lock()
        self._source_locks = {}
        self._samples = None
        self._prepared = 0
        self._prepare_seconds = self._wait_seconds = 0.
        self._buffered_bytes = self._max_buffered_bytes = 0
        self._stream = None

    def _sample(self, index):
        key = self._loader.dataset.prepare_concurrency_key(index)
        with self._source_locks_guard:
            lock = self._source_locks.setdefault(key, threading.Lock())
        with lock:
            return self._loader.dataset[index]

    def _prepare(self, indices):
        started = time.perf_counter()
        if self._samples is None:
            batch = next(self._source)
        else:
            # The batch sampler owns exact sample order. Futures finish in any
            # order, but map returns in that order before the ordinary collate.
            batch = self._loader.collate_fn(list(self._samples.map(self._sample, indices)))
            # Match the zero-worker DataLoader's actual pin policy. This does
            # not move audio or model work onto CUDA sample threads.
            if self._source._pin_memory:
                from torch.utils.data._utils.pin_memory import pin_memory
                batch = pin_memory(batch, self._source._pin_memory_device)
        size = _tensor_bytes(batch)
        with self._lock:
            self._prepared += 1
            self._prepare_seconds += time.perf_counter()-started
            self._buffered_bytes += size
            self._max_buffered_bytes = max(self._max_buffered_bytes, self._buffered_bytes)
        return batch

    def __enter__(self):
        if self.sample_workers > 1:
            self._samples = ThreadPoolExecutor(max_workers=self.sample_workers,
                                               thread_name_prefix='song-form-sample')
        indices = self._loader.batch_sampler if self._samples is not None else range(self._length)
        self._stream = OrderedBlueprintPrefetch(indices, self._prepare,
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
            return {'mode': 'shared_audio_ordered_batches', 'depth': self.depth,
                    'sample_workers': self.sample_workers,
                    'prepared_batches': self._prepared,
                    'prepare_sec': round(self._prepare_seconds, 3),
                    'consumer_wait_sec': round(self._wait_seconds, 3),
                    'buffered_tensor_bytes': self._buffered_bytes,
                    'max_buffered_tensor_bytes': self._max_buffered_bytes}

    def close(self):
        if self._stream is not None:
            self._stream.close(cancel=True)
            self._stream = None
        if self._samples is not None:
            self._samples.shutdown(wait=True, cancel_futures=True)
            self._samples = None
        close = getattr(self._source, 'close', None)
        if close is not None:
            close()

    def __exit__(self, *_exc):
        self.close()
