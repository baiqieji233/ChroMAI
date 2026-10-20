"""Ordered batches with bounded work scheduled across batch boundaries."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import threading,time

from maichart.form.layout_prefetch import _tensor_bytes


def _batch_bytes(batch):
    from .training_batch import PackedBatch
    return batch.nbytes if isinstance(batch, PackedBatch) else _tensor_bytes(batch)


class StreamBatchPrefetch:
    def __init__(self,loader,*,depth=2,workers=4,prepare_batch=None):
        if loader.num_workers or not 1<=depth<=16 or not 1<=workers<=8:
            raise ValueError('Invalid stream prefetch settings')
        self.loader,self.depth,self.workers=loader,depth,workers
        self.prepare_batch=prepare_batch
        # Consume the same DataLoader base RNG seed as synchronous iteration.
        self.source=iter(loader);self.indices=iter(loader.batch_sampler)
        self.pending=deque();self.source_locks={};self.guard=threading.Lock()
        self.samples=self.collator=None
        self.prepared=0;self.prepare_seconds=self.wait_seconds=0.
        self.buffered_bytes=self.peak_bytes=0;self.exhausted=False

    def _sample(self,index):
        key=self.loader.dataset.prepare_concurrency_key(index)
        with self.guard:lock=self.source_locks.setdefault(key,threading.Lock())
        started=time.perf_counter()
        with lock:result=self.loader.dataset[index]
        with self.guard:self.prepare_seconds+=time.perf_counter()-started
        return result

    def _collate(self,futures):
        batch=self.loader.collate_fn([future.result() for future in futures])
        if self.prepare_batch is not None:
            batch=self.prepare_batch(batch)
        elif self.source._pin_memory:
            from torch.utils.data._utils.pin_memory import pin_memory
            batch=pin_memory(batch,self.source._pin_memory_device)
        size=_batch_bytes(batch)
        with self.guard:
            self.prepared+=1;self.buffered_bytes+=size;self.peak_bytes=max(self.peak_bytes,self.buffered_bytes)
        return batch

    def _fill(self):
        while len(self.pending)<self.depth and not self.exhausted:
            try:indices=next(self.indices)
            except StopIteration:self.exhausted=True;break
            futures=[self.samples.submit(self._sample,index) for index in indices]
            self.pending.append(self.collator.submit(self._collate,futures))

    def __enter__(self):
        self.samples=ThreadPoolExecutor(self.workers,thread_name_prefix='stream-input')
        self.collator=ThreadPoolExecutor(1,thread_name_prefix='stream-collate')
        self._fill();return self

    def __iter__(self):return self
    def __next__(self):
        if not self.pending:raise StopIteration
        started=time.perf_counter();batch=self.pending.popleft().result()
        with self.guard:
            self.wait_seconds+=time.perf_counter()-started;self.buffered_bytes-=_batch_bytes(batch)
        self._fill();return batch

    def stats(self):
        with self.guard:return {'mode':'ordered_cross_batch_samples','depth':self.depth,'sample_workers':self.workers,
            'prepared_batches':self.prepared,'sample_prepare_sec':round(self.prepare_seconds,3),
            'consumer_wait_sec':round(self.wait_seconds,3),'buffered_tensor_bytes':self.buffered_bytes,
            'max_buffered_tensor_bytes':self.peak_bytes}

    def __exit__(self,*_):
        for future in self.pending:future.cancel()
        # Cancel samples before waiting for the collator; failed/cancelled
        # sample futures unblock its result() call and preserve the exception.
        self.samples.shutdown(wait=True,cancel_futures=True)
        self.collator.shutdown(wait=True,cancel_futures=True)
        self.pending.clear()
