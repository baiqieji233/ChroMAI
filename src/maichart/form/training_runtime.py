"""Numerically correct updates and observable runtime for song-form training."""
from __future__ import annotations

from contextlib import nullcontext
import json
import math
import time

import torch


TRAINING_SEMANTICS = 'song_form_training_v3'


def emit(event, **fields):
    print(event + ' ' + json.dumps(fields, ensure_ascii=False, allow_nan=False), flush=True)


def resolve_precision(requested, device):
    if requested not in {'auto', 'fp32', 'bf16', 'fp16'}:
        raise ValueError('precision must be auto, fp32, bf16 or fp16')
    if device.type != 'cuda':
        if requested not in {'auto', 'fp32'}:
            raise ValueError('CPU training uses fp32; choose precision=auto or fp32')
        return 'fp32'
    if requested == 'auto':
        return 'bf16' if torch.cuda.is_bf16_supported() else 'fp16'
    if requested == 'bf16' and not torch.cuda.is_bf16_supported():
        raise ValueError('This GPU does not support bf16; choose auto, fp16 or fp32')
    return requested


class UpdateEngine:
    """Accumulate loss sums, normalize once by actual sample count, then step.

    Dividing by a fixed accumulation factor loses the tail group's scale and
    gives a smaller final batch the same weight as a full one. This uses the
    actual examples in each update, including partial accumulation groups.
    """

    def __init__(self, optimizer, parameters, device, precision='fp32'):
        self.optimizer = optimizer
        self.parameters = list(parameters)
        self.device = device
        self.precision = resolve_precision(precision, device)
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.precision == 'fp16')
        self.samples = 0
        self.microbatches = 0
        self.optimizer.zero_grad(set_to_none=True)

    def autocast(self):
        if self.precision == 'fp32':
            return nullcontext()
        return torch.autocast('cuda', dtype=torch.bfloat16 if self.precision == 'bf16' else torch.float16)

    def backward(self, loss_sum, samples):
        if samples < 1:
            raise ValueError('Update sample count must be positive')
        self.scaler.scale(loss_sum).backward()
        self.samples += int(samples)
        self.microbatches += 1

    def step(self):
        if not self.samples:
            raise ValueError('Cannot update without accumulated examples')
        samples, microbatches = self.samples, self.microbatches
        self.scaler.unscale_(self.optimizer)
        for parameter in self.parameters:
            if parameter.grad is not None:
                parameter.grad.div_(samples)
        norm = torch.nn.utils.clip_grad_norm_(self.parameters, 1.0)
        finite = bool(torch.isfinite(norm))
        if self.scaler.is_enabled():
            if finite:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Even finite individual gradients can overflow the norm
                # reduction. Never let clipping's NaNs reach Adam in that case.
                self.scaler.update(new_scale=self.scaler.get_scale() * .5)
        elif finite:
            self.optimizer.step()
        else:
            self.optimizer.zero_grad(set_to_none=True)
            self.samples = self.microbatches = 0
            raise ValueError('Nonfinite gradients; update was not applied. Try fp32 and inspect loss components.')
        self.optimizer.zero_grad(set_to_none=True)
        self.samples = self.microbatches = 0
        return {'applied': finite, 'samples': samples, 'microbatches': microbatches,
                'grad_norm': float(norm) if finite else None,
                'amp_scale': float(self.scaler.get_scale()) if self.scaler.is_enabled() else None}


class EpochBatches:
    """Reproducible shuffled batches with an O(1)-cost resume cursor.

    No dataset fetches are made for skipped batches. This sampler has its own
    generator, so creating workers or inspecting a sample cannot alter order.
    """

    def __init__(self, size, batch_size, seed, epoch, *, start=0, limit=0):
        self.size, self.batch_size = int(size), int(batch_size)
        self.seed, self.epoch, self.start = int(seed), int(epoch), int(start)
        self.total = math.ceil(self.size / self.batch_size)
        if limit:
            self.total = min(self.total, int(limit))
        if not 0 <= self.start <= self.total:
            raise ValueError('Checkpoint batch cursor is outside current loader')

    def __len__(self):
        return self.total - self.start

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch * 100003)
        order = torch.randperm(self.size, generator=generator).tolist()
        for batch in range(self.start, self.total):
            yield order[batch * self.batch_size:(batch + 1) * self.batch_size]


def validate_cpu_batch(batch):
    """Catch corrupt source tensors before H2D without GPU host syncs."""
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f'Nonfinite training input: {key}')
    task = batch.get('task')
    if task is not None:
        for key in ('draft', 'right'):
            ids = batch.get(key)
            if ids is not None and torch.any((task == 0)[:, None] & (ids != 0)):
                raise ValueError(f'{key} targets cannot appear in draft task input')


def memory_snapshot(device):
    if device.type != 'cuda':
        return {}
    gib = 1024 ** 3
    free, total = torch.cuda.mem_get_info(device)
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    return {key: round(value / gib, 3) for key, value in {
        'allocated_gib': allocated,
        'reserved_gib': reserved,
        'inactive_cached_gib': max(0, reserved-allocated),
        'device_free_gib': free, 'device_total_gib': total,
        'peak_allocated_gib': torch.cuda.max_memory_allocated(device),
    }.items()}


class Progress:
    def __init__(self, epoch, plan_total, layout_total, interval, device, precision):
        self.epoch, self.plan_total, self.layout_total = epoch, plan_total, layout_total
        self.interval, self.device, self.precision = interval, device, precision
        self.started = time.perf_counter()
        self.phase = None
        self.last_logged = self.started
        self.phase_started = self.started
        self.phase_start_count = 0
        self.timing = dict(data_wait_sec=0., h2d_sec=0., compute_sec=0., checkpoint_sec=0.)
        self.loss_sums = {}
        self.loss_counts = {}
        self.metric_counts = {}

    def begin(self, phase, completed=0):
        self.phase, self.phase_start_count = phase, completed
        self.phase_started = time.perf_counter()
        self.last_logged = self.phase_started
        self.last_count = completed
        self.loss_sums.clear()
        self.loss_counts.clear()
        self.metric_counts.clear()
        self.phase_samples = 0
        self.phase_tokens = 0
        self.timing = dict(data_wait_sec=0., h2d_sec=0., compute_sec=0., checkpoint_sec=0.)

    def add(self, losses, weight=1, *, weights=None):
        # Keep detached scalar totals on the GPU until an actual log interval.
        for key, value in losses.items():
            item_weight = (weights or {}).get(key, weight)
            value = value.detach().float() if isinstance(value, torch.Tensor) else float(value)
            self.loss_sums[key] = self.loss_sums.get(key, 0.) + value * item_weight
            self.loss_counts[key] = self.loss_counts.get(key, 0) + item_weight
            self.metric_counts[key] = self.metric_counts.get(key, 0) + item_weight

    def log(self, completed, total, *, force=False, **extra):
        now = time.perf_counter()
        if not force and completed not in (self.phase_start_count + 1, total) and now-self.last_logged < self.interval:
            return
        elapsed = now-self.phase_started
        processed = completed-self.phase_start_count
        rate = processed/max(elapsed, 1e-9)
        scalar = lambda value: float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)
        losses = {key: scalar(value)/max(1, scalar(self.loss_counts[key]))
                  for key,value in self.loss_sums.items()}
        if not all(math.isfinite(value) for value in losses.values()):
            raise ValueError(f'Nonfinite training losses: {losses}')
        metric_counts = {key: int(round(scalar(value))) for key, value in self.metric_counts.items()}
        fraction = ((completed if self.phase == 'blueprint' else self.plan_total + completed)
                    / max(1, self.plan_total + self.layout_total))
        emit('form_progress', epoch=self.epoch, phase=self.phase, completed=completed, total=total,
             unit='charts' if self.phase == 'blueprint' else 'batches',
             phase_fraction=completed/max(1,total), overall_fraction=fraction,
             elapsed_sec=round(now-self.started, 2), phase_elapsed_sec=round(elapsed, 2),
             items_per_sec=round(rate, 3),
             eta_phase_sec=round((total-completed)/rate, 1) if processed else None,
             samples_per_sec=round(self.phase_samples/max(elapsed,1e-9), 2),
             tokens_per_sec=round(self.phase_tokens/max(elapsed,1e-9), 1),
             losses=losses, metric_counts=metric_counts,
             timing={k:round(v,3) for k,v in self.timing.items()},
             timing_scope='host_wall_including_synchronization',
             precision=self.precision, memory=memory_snapshot(self.device), **extra)
        self.last_logged, self.last_count = now, completed


def capture_rng(device):
    import random
    import numpy as np
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None}


def restore_rng(state, device):
    import random
    import numpy as np
    if not state:
        return
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if device.type == 'cuda' and state.get('cuda'):
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])
