"""Authored-position supervision over existing audio evidence, never output grids.

Targets remain exact Fraction events. Queries are constructed without chart
events; unmatched targets are explicitly excluded instead of teaching an
incorrect nearest-frame position. Both losses use independent finite masks.
"""
from __future__ import annotations

import math
import numpy as np
import torch
from torch.nn import functional as F

from maichart.event_timing import seconds_at_beat


def _times(points, tempo, first):
    return np.asarray([seconds_at_beat(p.beat, tempo, first) for p in points], dtype=np.float64)


def _query_times(times):
    values = np.asarray(times, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any(np.diff(values) < 0):
        raise ValueError('Audio query times must be finite, ordered and one-dimensional')
    return values


def _tolerance(value):
    value = float(value)
    if not math.isfinite(value) or not 1 <= value <= 100:
        raise ValueError('position_alignment_tolerance_ms must be in [1, 100]')
    return value / 1000.


def build_alignment_targets(points, tempo, first, times, *, tolerance_ms=15.):
    radius = _tolerance(tolerance_ms)
    queries = _query_times(times)
    stamps = _times(points, tempo, first)
    labels = np.zeros(len(queries), dtype=np.float32)
    mask = np.isfinite(queries) & (queries >= 0.)
    if len(stamps):
        for second in stamps:
            if second < 0:
                continue
            lo = np.searchsorted(queries, second-radius, side='left')
            hi = np.searchsorted(queries, second+radius, side='right')
            if hi > lo:
                # This radius is a labeling uncertainty, not a model time step.
                labels[lo:hi] = np.maximum(labels[lo:hi], np.exp(
                    -.5 * ((queries[lo:hi]-second)/(radius*.5))**2))
    return {'labels': torch.from_numpy(labels), 'mask': torch.from_numpy(mask)}


def build_event_alignment_targets(points, tempo, first, times, decoder_input, codec, *, tolerance_ms=15.):
    radius = _tolerance(tolerance_ms)
    queries = _query_times(times)
    prefix = torch.as_tensor(decoder_input).tolist()
    # Every BOS/END immediately before an AT predicts its next event, without
    # consuming any digit of that event's Fraction.
    indices = [i for i, token in enumerate(prefix[:-1])
               if token in (codec.bos_id, codec.end_id) and prefix[i+1] == codec.at_id]
    stamps = _times(points, tempo, first)
    if len(indices) != len(stamps):
        raise ValueError('Event alignment targets do not match the causal decoder prefix')
    targets, masks = [], []
    valid_indices = np.flatnonzero(queries >= 0.)
    available = queries[valid_indices]
    for second in stamps:
        right = int(np.searchsorted(available, second))
        choices = [int(valid_indices[i]) for i in (right-1, right) if 0 <= i < len(available)]
        nearest = min(choices, key=lambda i: abs(queries[i]-second)) if choices else 0
        targets.append(nearest)
        masks.append(bool(choices and second >= 0 and abs(queries[nearest]-second) <= radius))
    return {'alignment_event_indices': torch.tensor(indices, dtype=torch.long),
            'alignment_audio_index': torch.tensor(targets, dtype=torch.long),
            'alignment_event_mask': torch.tensor(masks, dtype=torch.bool)}


def position_alignment_loss(prediction, batch):
    """Draft-only event CE and audio BCE, independently normalized per row.

    Revision can read an authored draft with correct future positions. Its
    alignment accuracy does not measure audio-dependent free generation, so
    neither this auxiliary loss nor its metrics train on revision rows.
    """
    logits = prediction['alignment_event_logits'].float()
    event_mask = batch['alignment_event_mask'].bool()
    labels = batch['alignment_audio_index'].long()
    if logits.ndim != 3 or logits.shape[:2] != labels.shape or labels.shape != event_mask.shape:
        raise ValueError('Event alignment output/target shapes differ')
    indices = batch['alignment_event_indices'].long()
    if indices.shape != labels.shape:
        raise ValueError('Event alignment decoder indices/target shapes differ')
    draft = batch.get('task', torch.zeros(logits.size(0), device=logits.device, dtype=torch.long)).eq(0)
    if draft.shape != (logits.size(0),):
        raise ValueError('Alignment task must have shape [B]')
    audio_padding = batch.get('audio_padding_mask')
    valid_audio = (torch.ones((logits.size(0), logits.size(2)), dtype=torch.bool, device=logits.device)
                   if audio_padding is None else ~audio_padding.bool())
    if valid_audio.shape != (logits.size(0), logits.size(2)):
        raise ValueError('Alignment audio padding shape differs from query logits')
    eligible = indices.ge(0) & draft[:, None]
    event_mask = event_mask & eligible & labels.ge(0) & labels.lt(logits.size(2))
    if logits.size(1) and logits.size(2):
        safe_labels = labels.clamp(0, logits.size(2)-1)
        event_mask = event_mask & valid_audio.gather(1, safe_labels)
        safe_logits = logits.masked_fill(~valid_audio[:, None], -1e4)
        safe_logits = safe_logits.masked_fill(~event_mask[..., None], 0.)
        loss = F.cross_entropy(safe_logits.transpose(1, 2), safe_labels, reduction='none')
        event = (loss*event_mask).sum(1)/event_mask.sum(1).clamp_min(1)
        accuracy = ((safe_logits.argmax(-1)==safe_labels)&event_mask).sum(1)/event_mask.sum(1).clamp_min(1)
    else:
        event_mask = torch.zeros_like(event_mask)
        event = logits.masked_fill(torch.ones_like(logits, dtype=torch.bool), 0.).sum((1, 2))
        accuracy = event.detach()
    scores = prediction['position_alignment_logits'].float()
    target = batch['position_alignment_target'].to(scores)
    mask = batch['position_alignment_mask'].bool()
    padding = batch.get('audio_padding_mask')
    if padding is not None:
        mask = mask & ~padding.bool()
    mask = mask & draft[:, None]
    if scores.shape != target.shape or target.shape != mask.shape:
        raise ValueError('Audio alignment output/target shapes differ')
    target = target.masked_fill(~mask, 0.)
    scores = scores.masked_fill(~mask, 0.)
    positive = (target*mask).sum(1)
    negative = ((1-target)*mask).sum(1)
    pos_weight = (negative/positive.clamp_min(1)).clamp(1, 12)[:, None]
    selection = F.binary_cross_entropy_with_logits(scores, target, reduction='none', pos_weight=pos_weight)
    selection = (selection*mask).sum(1)/mask.sum(1).clamp_min(1)
    coverage = event_mask.sum(1)/eligible.sum(1).clamp_min(1)
    return event+.25*selection, {'position_alignment_event_accuracy': accuracy.detach(),
                                'position_alignment_coverage': coverage.detach()}


def temporal_neighbors(hidden, times, padding=None, *, seconds=1., validate_inputs=True):
    """Separate left/right means on the actual nonuniform clock, O(B*T*D)."""
    if hidden.ndim != 3 or not math.isfinite(seconds) or seconds < 0:
        raise ValueError('Temporal neighbors require hidden [B,T,D] and finite nonnegative seconds')
    times = times.to(device=hidden.device, dtype=torch.float64).contiguous()
    if times.shape != hidden.shape[:2]:
        raise ValueError('Temporal audio timestamps must match audio nodes')
    valid = (torch.ones_like(times, dtype=torch.bool) if padding is None
             else ~padding.to(device=hidden.device, dtype=torch.bool))
    if valid.shape != times.shape:
        raise ValueError('Temporal audio padding must match timestamps')
    # Forward filling ignored nodes keeps search order even with internal
    # padding holes. Valid-count prefixes, rather than array indices, supply
    # the denominator. No interpolation or regular frame clock is introduced.
    search_times = times.masked_fill(~valid, -float('inf')).cummax(1).values.contiguous()
    if validate_inputs:
        invalid = (~torch.isfinite(times) & valid).any()
        invalid = invalid | (~torch.isfinite(hidden) & valid[..., None]).any()
        invalid = invalid | ((times < search_times) & valid).any()
        if invalid:
            raise ValueError('Unmasked temporal audio inputs must be finite and times ordered')
    values = hidden.masked_fill(~valid[..., None], 0.)
    if seconds == 0:
        return torch.cat((values, values), -1)
    query_times = times.masked_fill(~valid, 0.)
    left = torch.searchsorted(search_times, (query_times-seconds).contiguous())
    right = torch.searchsorted(search_times, (query_times+seconds).contiguous(), right=True)
    index = torch.arange(times.size(1), device=times.device)[None].expand_as(left)
    # AMP projections are bf16/fp16. Subtracting nearby large prefixes in
    # those dtypes can turn a constant .3 neighborhood into 0 or 2. Accumulate
    # and divide in fp32, keeping higher precision inputs unchanged.
    cumulative_values = values.float() if values.dtype in (torch.bfloat16, torch.float16) else values
    prefix = torch.cat((cumulative_values.new_zeros(values.size(0), 1, values.size(2)),
                        cumulative_values.cumsum(1)), 1)
    def gather(i): return prefix.gather(1, i[..., None].expand(-1, -1, hidden.size(2)))
    counts = torch.cat((valid.new_zeros(valid.size(0), 1).long(), valid.long().cumsum(1)), 1)
    before_count = counts.gather(1, index)-counts.gather(1, left)
    after_count = counts.gather(1, right)-counts.gather(1, index+1)
    before = (gather(index)-gather(left))/before_count.clamp_min(1)[..., None]
    after = (gather(right)-gather(index+1))/after_count.clamp_min(1)[..., None]
    return torch.cat((before, after), -1).masked_fill(~valid[..., None], 0).to(hidden.dtype)
