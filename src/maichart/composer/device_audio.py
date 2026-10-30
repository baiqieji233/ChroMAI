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
            return torch.as_tensor(np.ascontiguousarray(value)).to(device)
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
        self._widths = torch.tensor([.5, 2., 8.], device=device, dtype=self.times.dtype)[:, None]
        self._duration = torch.tensor(self.duration, device=device, dtype=self.times.dtype)
        self._lens_offsets = torch.tensor(LENS_OFFSETS, device=device, dtype=self.times.dtype)[None]
        self._time_min = float(context.times[0]) if len(context.times) else 0.
        self._time_max = float(context.times[-1]) if len(context.times) else 0.
        self._max_idx = max(0, len(context.times) - 2)
        self._half_slopes = .5 * self.slopes
        if len(context.times) > 1 and len(self.chroma) > 1:
            self._chroma_slopes = (self.chroma[1:] - self.chroma[:-1]) / (self.times[1:] - self.times[:-1])[:, None]
            self._chroma_last = self.chroma[-1]
        else:
            self._chroma_slopes = None
            self._chroma_last = self.chroma[-1] if len(self.chroma) else None
        if lens is not None and len(lens.times):
            self._lens_t_min = float(lens.times[0])
            self._lens_t_max = float(lens.times[-1])
            self._lens_lo = max(0., self._lens_t_min)
            self._lens_max_idx = max(0, len(lens.times) - 2)
            if len(lens.times) > 1:
                self._lens_inv_dt = 1. / (self.lens_times[1:] - self.lens_times[:-1])
                self._lens_diffs = self.lens_values[1:] - self.lens_values[:-1]
            else:
                self._lens_inv_dt = None
                self._lens_diffs = None
        else:
            self._lens_t_min = self._lens_t_max = self._lens_lo = 0.
            self._lens_max_idx = 0
            self._lens_inv_dt = self._lens_diffs = None
        # Include the new slope/difference banks. Count storage once so views
        # such as _chroma_last neither disappear from nor inflate the budget.
        storages = {value.untyped_storage().data_ptr(): value.untyped_storage().nbytes()
                    for value in vars(self).values() if torch.is_tensor(value)}
        self.nbytes = sum(storages.values())

    def integral(self, seconds):
        if len(self.times) < 2:
            return seconds.new_zeros((*seconds.shape, 26))
        seconds = seconds.clamp(self._time_min, self._time_max)
        left = (torch.searchsorted(self.times, seconds.contiguous(), right=True) - 1).clamp(0, self._max_idx)
        delta = (seconds-self.times[left])[..., None]
        return self.prefix[left] + self.data[left]*delta + self._half_slopes[left]*delta.square()

    def at(self, seconds):
        widths = (self._widths if self._widths.device == seconds.device and self._widths.dtype == seconds.dtype
                  else seconds.new_tensor([.5, 2., 8.])[:, None])
        lo = (seconds[None]-widths/2).clamp(0., self.duration)
        hi = (seconds[None]+widths/2).clamp(0., self.duration)
        hi_int, lo_int = self.integral(torch.stack((hi, lo), dim=0)).unbind(0)
        mean = (hi_int-lo_int)/(hi-lo).clamp_min(1e-6)[..., None]
        return torch.nan_to_num(torch.cat((mean.transpose(0, 1).reshape(len(seconds), 78),
            self.global_mean.expand(len(seconds), -1)), -1), nan=0., posinf=0., neginf=0.).float()

    def trailing(self, seconds):
        end, prev1, prev4 = self.integral(torch.stack((seconds, seconds-1., seconds-4.), dim=0)).unbind(0)
        return torch.cat((end-prev1, (end-prev4)/4.), -1).float()

    def _at_and_trailing(self, seconds):
        widths = (self._widths if self._widths.device == seconds.device and self._widths.dtype == seconds.dtype
                  else seconds.new_tensor([.5, 2., 8.])[:, None])
        lo = (seconds[None]-widths/2).clamp(0., self.duration)
        hi = (seconds[None]+widths/2).clamp(0., self.duration)
        trail_t = torch.stack((seconds, seconds-1., seconds-4.), dim=0)
        all_int = self.integral(torch.cat((hi, lo, trail_t), dim=0))
        hi_int, lo_int = all_int[:3], all_int[3:6]
        end, prev1, prev4 = all_int[6], all_int[7], all_int[8]
        mean = (hi_int-lo_int)/(hi-lo).clamp_min(1e-6)[..., None]
        at_out = torch.nan_to_num(torch.cat((mean.transpose(0, 1).reshape(len(seconds), 78),
            self.global_mean.expand(len(seconds), -1)), -1), nan=0., posinf=0., neginf=0.).float()
        tr_out = torch.cat((end-prev1, (end-prev4)/4.), -1).float()
        return at_out, tr_out

    def interval(self, previous, seconds):
        start = previous.clamp(0., self.duration)
        dur = (self._duration if self._duration.device == seconds.device and self._duration.dtype == seconds.dtype
               else seconds.new_tensor(self.duration))
        stop = torch.minimum(torch.maximum(seconds, start), dur)
        start_exp = start.expand_as(stop)
        both = torch.stack((start_exp, stop), dim=0)
        if len(self.times) >= 2:
            clock = both.clamp(self._time_min, self._time_max)
            idx = (torch.searchsorted(self.times, clock.contiguous(), right=True) - 1).clamp(0, self._max_idx)
            delta = (clock - self.times[idx])[..., None]
            start_int, stop_int = (self.prefix[idx] + self.data[idx]*delta + self._half_slopes[idx]*delta.square()).unbind(0)
            area = stop_int - start_int
            values = self.chroma[idx] + self._chroma_slopes[idx] * delta
            begin, end = torch.where((clock == self._time_max)[..., None], self._chroma_last, values).unbind(0)
            norm = torch.linalg.vector_norm(end, dim=-1)*torch.linalg.vector_norm(begin, dim=-1)
            novelty = torch.where(norm > 1e-9, 1.-(end*begin).sum(-1)/norm.clamp_min(1e-9), 0.)
        elif len(self.times) == 1:
            area = stop.new_zeros((*stop.shape, 26))
            begin = end = self.chroma[0].expand(*stop.shape, 12)
            norm = torch.linalg.vector_norm(end, dim=-1)*torch.linalg.vector_norm(begin, dim=-1)
            novelty = torch.where(norm > 1e-9, 1.-(end*begin).sum(-1)/norm.clamp_min(1e-9), 0.)
        else:
            area = stop.new_zeros((*stop.shape, 26))
            novelty = torch.zeros_like(stop)
        mean = area[..., :14]/(stop-start_exp).clamp_min(1e-6)[..., None]
        return torch.cat((mean, torch.log1p(area[..., :1].clamp_min(0.)), novelty[..., None]), -1).float()

    def lens(self, seconds):
        offsets = (self._lens_offsets if self._lens_offsets.device == seconds.device and self._lens_offsets.dtype == seconds.dtype
                   else seconds.new_tensor(LENS_OFFSETS)[None])
        clock = seconds[:, None]+offsets
        times = self.lens_times
        if times is None or not len(times):
            return torch.zeros((*clock.shape, 26), device=seconds.device, dtype=torch.float32), torch.zeros_like(clock, dtype=torch.bool)
        valid = (clock >= self._lens_lo) & (clock < self.lens_duration) & (clock <= self._lens_t_max)
        if len(times) == 1:
            result = torch.zeros((*clock.shape, 26), device=seconds.device, dtype=torch.float32)
            return torch.where(valid[..., None], self.lens_values[0], result), valid
        # Query all positions and mask afterward; avoids variable-size CUDA
        # nonzero synchronization while retaining float32 interpolation order.
        left = (torch.searchsorted(times, clock.contiguous(), right=True)-1).clamp(0, self._lens_max_idx)
        fraction = ((clock-times[left])*self._lens_inv_dt[left]).float()
        base = self.lens_values[left]
        values = base+fraction[..., None]*self._lens_diffs[left]
        return values.masked_fill(~valid[..., None], 0.), valid


class AudioQueryMaterializer:
    """LRU CPU-bank identity cache, bounded in bytes per device.

    Oversized songs are materialized transiently, never retained above budget.
    Strong CPU references prevent recycled Python ids from aliasing a bank.
    """
    def __init__(self, budget_bytes=512*1024**2):
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
        raw_rows = batch['_audio_bank_rows']
        unique_splits = batch.get('_audio_unique_splits')
        compact = unique_splits is not None
        if compact:
            unique_times = tensors['audio_unique_seconds'].split(unique_splits)
            query_inverse = tensors['audio_query_inverse']
            lens_patches, lens_masks = [], []
            unique_offset = 0
        slices = [
            (rows[0], rows[-1] + 1)
            if rows and rows[-1] - rows[0] + 1 == len(rows) and rows == tuple(range(rows[0], rows[-1] + 1))
            else None
            for rows in raw_rows
        ]
        if all(slc is not None for slc in slices):
            row_tensors = (None,) * len(raw_rows)
        else:
            packed_rows = tensors.get('_audio_bank_row_indices')
            splits = batch.get('_audio_bank_row_splits')
            if (packed_rows is not None and splits is not None
                    and packed_rows.device == query.device and sum(splits) == packed_rows.numel()):
                row_tensors = packed_rows.split(splits)
            else:
                row_tensors = tuple(torch.tensor(rows, device=query.device, dtype=torch.long)
                                    for rows in raw_rows)
        for bank_index, (context, slc, row_index) in enumerate(zip(batch['_audio_banks'], slices, row_tensors)):
            bank = self._get(context, query.device)
            seconds = query[slc[0]:slc[1]] if slc is not None else query.index_select(0, row_index)
            flat = seconds.reshape(-1)
            if compact:
                flat = unique_times[bank_index]
                local_inverse = (query_inverse[slc[0]:slc[1]] if slc is not None
                                 else query_inverse.index_select(0, row_index)) - unique_offset
                unique_offset += unique_splits[bank_index]
            if batch['_audio_music']:
                at_val, tr_val = bank._at_and_trailing(flat)
                previous = (tensors['audio_previous_seconds'][slc[0]:slc[1], None]
                            if slc is not None
                            else tensors['audio_previous_seconds'].index_select(0, row_index)[:, None])
                outputs = {
                    'workload_audio': at_val[local_inverse] if compact else at_val.reshape(*seconds.shape, 104),
                    'trailing_audio': tr_val[local_inverse] if compact else tr_val.reshape(*seconds.shape, 52),
                    'interval_audio': bank.interval(previous, seconds[:, :candidates]),
                }
            else:
                at_val = bank.at(flat)
                outputs = {'workload_audio': at_val[local_inverse] if compact else at_val.reshape(*seconds.shape, 104)}
            if batch['_audio_grounded']:
                patches, mask = bank.lens(flat)
                if compact:
                    lens_patches.append(patches); lens_masks.append(mask)
                else:
                    outputs['audio_lens'] = patches.reshape(*seconds.shape, 31, 26)
                    outputs['audio_lens_mask'] = mask.reshape(*seconds.shape, 31)
            if slc == (0, len(query)):
                tensors.update(outputs)
            elif slc is not None:
                for name, value in outputs.items():
                    if name not in tensors:
                        tensors[name] = value.new_empty((len(query), *value.shape[1:]))
                    tensors[name][slc[0]:slc[1]].copy_(value)
            else:
                for name, value in outputs.items():
                    if name not in tensors:
                        tensors[name] = value.new_empty((len(query), *value.shape[1:]))
                    tensors[name].index_copy_(0, row_index, value)
        if compact and batch['_audio_grounded']:
            tensors['audio_lens'] = torch.cat(lens_patches) if len(lens_patches) > 1 else lens_patches[0]
            tensors['audio_lens_mask'] = torch.cat(lens_masks) if len(lens_masks) > 1 else lens_masks[0]
            tensors['audio_lens_inverse'] = query_inverse
        return tensors


Materializer = AudioQueryMaterializer
