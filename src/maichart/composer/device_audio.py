"""Bounded device residency and exact-clock audio queries for training.

Only the training thread calls this module: dataset workers retain CPU banks.
Float64 clocks and acoustic integrals deliberately match the reference bank.
"""
from collections import OrderedDict

import numpy as np
import torch

from .audio_lens import LENS_OFFSETS


class _DeviceBank:
    def __init__(self, context, device):
        def copy(value):
            return torch.tensor(np.asarray(value), device=device)
        self.times = copy(context.times)
        self.data = copy(context.data)
        self.prefix = copy(context.prefix)
        self.slopes = copy(getattr(context, 'slopes', np.zeros((0, 26))))
        self.chroma = copy(getattr(context, 'chroma', np.zeros((0, 12))))
        self.global_mean = copy(context.global_mean)
        self.duration = context.duration
        lens = getattr(context, 'lens_bank', None)
        self.lens_times = copy(lens.times) if lens is not None else None
        self.lens_values = copy(lens.values) if lens is not None else None
        self.lens_duration = lens.duration if lens is not None else 0.
        self.nbytes = sum(value.numel() * value.element_size()
                          for value in vars(self).values() if torch.is_tensor(value))

    def integral(self, seconds):
        if len(self.times) < 2:
            return seconds.new_zeros((*seconds.shape, 26))
        seconds = seconds.clamp(self.times[0], self.times[-1])
        left = (torch.searchsorted(self.times, seconds.contiguous(), right=True) - 1).clamp(0, len(self.times)-2)
        delta = (seconds-self.times[left])[..., None]
        return self.prefix[left] + self.data[left]*delta + .5*self.slopes[left]*delta.square()

    def at(self, seconds):
        widths = seconds.new_tensor([.5, 2., 8.])[:, None]
        lo = (seconds[None]-widths/2).clamp(0., self.duration)
        hi = (seconds[None]+widths/2).clamp(0., self.duration)
        mean = (self.integral(hi)-self.integral(lo))/(hi-lo).clamp_min(1e-6)[..., None]
        return torch.nan_to_num(torch.cat((mean.transpose(0, 1).reshape(len(seconds), 78),
            self.global_mean.expand(len(seconds), -1)), -1), nan=0., posinf=0., neginf=0.).float()

    def trailing(self, seconds):
        end = self.integral(seconds)
        return torch.cat((end-self.integral(seconds-1.),
                          (end-self.integral(seconds-4.))/4.), -1).float()

    def interval(self, previous, seconds):
        start = previous.clamp(0., self.duration)
        stop = torch.minimum(torch.maximum(seconds, start), seconds.new_tensor(self.duration))
        area = self.integral(stop)-self.integral(start)
        mean = area[..., :14]/(stop-start).clamp_min(1e-6)[..., None]
        if len(self.times):
            def endpoints(clock):
                if len(self.times) == 1:
                    return self.chroma[0].expand(*clock.shape, 12)
                clock = clock.clamp(self.times[0], self.times[-1])
                idx = (torch.searchsorted(self.times, clock.contiguous(), right=True)-1).clamp(0, len(self.times)-2)
                slope = (self.chroma[idx+1]-self.chroma[idx])/(self.times[idx+1]-self.times[idx])[..., None]
                values = self.chroma[idx]+slope*(clock-self.times[idx])[..., None]
                return torch.where((clock == self.times[-1])[..., None], self.chroma[-1], values)
            begin, end = endpoints(start), endpoints(stop)
            norm = torch.linalg.vector_norm(end, dim=-1)*torch.linalg.vector_norm(begin, dim=-1)
            novelty = torch.where(norm > 1e-9, 1.-(end*begin).sum(-1)/norm.clamp_min(1e-9), 0.)
        else:
            novelty = torch.zeros_like(stop)
        return torch.cat((mean, torch.log1p(area[..., :1].clamp_min(0.)), novelty[..., None]), -1).float()

    def lens(self, seconds):
        clock = seconds[:, None]+seconds.new_tensor(LENS_OFFSETS)[None]
        result = torch.zeros((*clock.shape, 26), device=seconds.device, dtype=torch.float32)
        times = self.lens_times
        if times is None or not len(times):
            return result, torch.zeros_like(clock, dtype=torch.bool)
        valid = (clock >= 0.) & (clock < self.lens_duration) & (clock >= times[0]) & (clock <= times[-1])
        if len(times) == 1:
            return torch.where(valid[..., None], self.lens_values[0], result), valid
        # Query all positions and mask afterward; avoids variable-size CUDA
        # nonzero synchronization while retaining float32 interpolation order.
        left = (torch.searchsorted(times, clock.contiguous(), right=True)-1).clamp(0, len(times)-2)
        fraction = ((clock-times[left])/(times[left+1]-times[left])).float()
        base = self.lens_values[left]
        values = base+fraction[..., None]*(self.lens_values[left+1]-base)
        return values.masked_fill(~valid[..., None], 0.), valid


class AudioQueryMaterializer:
    """LRU CPU-bank identity cache, bounded in bytes per device.

    Oversized songs are materialized transiently, never retained above budget.
    Strong CPU references prevent recycled Python ids from aliasing a bank.
    """
    def __init__(self, budget_bytes=256*1024**2):
        self.budget_bytes = max(0, int(budget_bytes))
        self.cache = OrderedDict()
        self.resident_bytes = 0

    def _get(self, context, device):
        key = (str(device), id(context))
        cached = self.cache.get(key)
        if cached is not None:
            self.cache.move_to_end(key)
            return cached[1]
        bank = _DeviceBank(context, device)
        while self.cache and self.resident_bytes+bank.nbytes > self.budget_bytes:
            _, (_, evicted) = self.cache.popitem(last=False)
            self.resident_bytes -= evicted.nbytes
        if bank.nbytes <= self.budget_bytes:
            self.cache[key] = (context, bank)
            self.resident_bytes += bank.nbytes
        return bank

    @torch.no_grad()
    def materialize(self, batch, tensors):
        if '_audio_banks' not in batch:
            return tensors
        query = tensors['audio_query_seconds']
        candidates = tensors['candidate_features'].shape[1]
        for context, rows in zip(batch['_audio_banks'], batch['_audio_bank_rows']):
            bank = self._get(context, query.device)
            row_index = torch.tensor(rows, device=query.device, dtype=torch.long)
            seconds = query.index_select(0, row_index)
            flat = seconds.reshape(-1)
            outputs = {'workload_audio': bank.at(flat).reshape(*seconds.shape, 104)}
            if batch['_audio_music']:
                outputs['trailing_audio'] = bank.trailing(flat).reshape(*seconds.shape, 52)
                previous = tensors['audio_previous_seconds'].index_select(0, row_index)[:, None]
                outputs['interval_audio'] = bank.interval(previous, seconds[:, :candidates])
            if batch['_audio_grounded']:
                patches, mask = bank.lens(flat)
                outputs['audio_lens'] = patches.reshape(*seconds.shape, 31, 26)
                outputs['audio_lens_mask'] = mask.reshape(*seconds.shape, 31)
            for name, value in outputs.items():
                if name not in tensors:
                    tensors[name] = value.new_empty((len(query), *value.shape[1:]))
                tensors[name].index_copy_(0, row_index, value)
        return tensors


Materializer = AudioQueryMaterializer
