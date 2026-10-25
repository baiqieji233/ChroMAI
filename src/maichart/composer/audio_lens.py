"""Ordered local audio evidence and authored, condition-specific occupancy.

Placement targets describe this reference chart, not a universal decision that
an audible event must or must not receive a note. No HPSS/PCEN is inferred from
the existing normalized audio bank.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from .event_views import EventViews
from .events import event_metadata


LENS_DIM = 26
LENS_OFFSETS = tuple(index * .02 for index in range(-10, 21))
PLACEMENT_TOLERANCE_SEC = .025
PLACEMENT_CHANNELS = ('presence', 'hold', 'slide', 'touch', 'break_ex')


class LensBank:
    """Read 31 local frames on the source clock without per-patch scaling.

    Inputs are existing [N,120] bank rows and strictly increasing source times.
    Query output is float32 [Q,31,26] plus bool [Q,31]. Frames outside either
    the source-clock support or [0,duration) are zero, never endpoint copies.
    """

    def __init__(self, values, times, duration):
        values = np.asarray(values)
        self.times = np.asarray(times, dtype=np.float64).reshape(-1)
        self.duration = float(duration)
        if not math.isfinite(self.duration) or self.duration < 0.:
            raise ValueError('Lens duration must be finite and nonnegative')
        if values.ndim != 2 or values.shape != (len(self.times), 120):
            raise ValueError('Lens source values must have shape [N,120]')
        if not np.isfinite(self.times).all() or np.any(np.diff(self.times) <= 0.):
            raise ValueError('Lens source times must be finite and strictly increasing')
        source = np.asarray(values[:, :98], dtype=np.float32)
        if not np.isfinite(source).all():
            raise ValueError('Lens source values must be finite')
        self.values = np.concatenate((source[:, 92:98],
            source[:, :80].reshape(-1, 8, 10).mean(-1), source[:, 80:92]), axis=-1)

    def query(self, seconds):
        seconds = np.asarray(seconds, dtype=np.float64).reshape(-1)
        if not np.isfinite(seconds).all():
            raise ValueError('Lens query times must be finite')
        clock = seconds[:, None] + np.asarray(LENS_OFFSETS)[None]
        result = np.zeros((*clock.shape, LENS_DIM), dtype=np.float32)
        valid = np.zeros(clock.shape, dtype=bool)
        if not len(self.times) or not len(seconds):
            return result, valid
        valid = ((clock >= 0.) & (clock < self.duration)
                 & (clock >= self.times[0]) & (clock <= self.times[-1]))
        flat_clock = clock[valid]
        if len(self.times) == 1:
            result[valid] = self.values[0]
            return result, valid
        left = np.clip(np.searchsorted(self.times, flat_clock, side='right') - 1,
                       0, len(self.times) - 2)
        fraction = ((flat_clock - self.times[left])
                    / (self.times[left + 1] - self.times[left])).astype(np.float32)
        result[valid] = (self.values[left]
            + fraction[:, None] * (self.values[left + 1] - self.values[left]))
        return result, valid


class _TemporalBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.depthwise = nn.Conv1d(width, width, 3, padding=dilation,
                                   dilation=dilation, groups=width)
        self.pointwise = nn.Conv1d(width, width, 1)
        self.activation = nn.GELU()

    def forward(self, hidden, valid):
        hidden = hidden + self.pointwise(self.activation(self.depthwise(hidden)))
        return hidden.masked_fill(~valid, 0.)


class AudioLens(nn.Module):
    """Shared noncausal TCN with ordered pooling: [B,Q,31,26] -> [B,Q,W].

    Candidate chunking bounds convolution activations without moving tensors
    to CPU or reading device scalars. Pool weights have a distinct parameter
    per temporal position; reversing a patch does not preserve its embedding.
    """

    def __init__(self, width=64, chunk_size=128):
        super().__init__()
        if int(width) <= 0 or int(chunk_size) <= 0:
            raise ValueError('Lens width and chunk_size must be positive')
        self.width = int(width)
        self.chunk_size = int(chunk_size)
        self.input_projection = nn.Conv1d(LENS_DIM, self.width, 1)
        self.blocks = nn.ModuleList(_TemporalBlock(self.width, dilation)
                                    for dilation in (1, 2, 4))
        self.position_pool = nn.Linear(len(LENS_OFFSETS), 1, bias=False)
        self.output = nn.Sequential(nn.LayerNorm(self.width), nn.Linear(self.width, self.width), nn.GELU())

    def forward(self, patches, valid):
        if patches.ndim != 4 or patches.shape[-2:] != (len(LENS_OFFSETS), LENS_DIM):
            raise ValueError('Lens patches must have shape [B,Q,31,26]')
        if valid.shape != patches.shape[:-1]:
            raise ValueError('Lens validity must have shape [B,Q,31]')
        batch, candidates = patches.shape[:2]
        if candidates == 0 or batch == 0:
            return patches.new_zeros((batch, candidates, self.width))
        chunks = []
        for start in range(0, candidates, self.chunk_size):
            stop = min(start + self.chunk_size, candidates)
            mask = valid[:, start:stop].reshape(-1, 1, len(LENS_OFFSETS)).bool()
            source = patches[:, start:stop].reshape(-1, len(LENS_OFFSETS), LENS_DIM).transpose(1, 2)
            source = (source if torch.is_autocast_enabled() else source.to(self.input_projection.weight.dtype)).masked_fill(~mask, 0.)
            hidden = self.input_projection(source).masked_fill(~mask, 0.)
            for block in self.blocks:
                hidden = block(hidden, mask)
            pooled = self.position_pool(hidden).squeeze(-1)
            embedded = self.output(pooled)
            embedded = embedded.masked_fill(~mask.any(-1), 0.)
            chunks.append(embedded.reshape(batch, stop - start, self.width))
        return torch.cat(chunks, dim=1)


class PlacementTargets:
    """Reference-chart occupancy within an inclusive +/-25ms neighborhood.

    Every equivalent nearby query receives a positive target. If several
    authored events match, their type flags are unioned, rather than inventing
    an exclusive nearest-event label. Unknown neighborhoods receive zero
    weight in all channels. Audio boundaries clip the known support; query
    centers outside [0,end_sec) have zero targets and weights.
    """

    def __init__(self, points, tempo, first, end_sec, event_views=None):
        self.end_sec = float(end_sec)
        if not math.isfinite(self.end_sec) or self.end_sec < 0.:
            raise ValueError('Placement end must be finite and nonnegative')
        self.views = (event_views if event_views is not None
                      else EventViews.build(points, tempo, first, self.end_sec))
        if not math.isclose(self.views.end_sec, self.end_sec, rel_tol=0., abs_tol=1e-9):
            raise ValueError('Placement and event-view audio ends must match')
        self.times = self.views.seconds[self.views.trusted]
        flags = []
        for point, trusted in zip(self.views.context_points, self.views.trusted):
            if trusted:
                meta = event_metadata(str(point.note_text))
                flags.append((1., meta.hold, meta.slide, meta.touch, meta.break_ex))
        values = np.asarray(flags, dtype=np.float32).reshape(-1, len(PLACEMENT_CHANNELS))
        self.prefix = np.vstack((np.zeros((1, len(PLACEMENT_CHANNELS)), dtype=np.float32),
                                 np.cumsum(values, axis=0)))

    def query(self, query_seconds):
        seconds = np.asarray(query_seconds, dtype=np.float64).reshape(-1)
        if not np.isfinite(seconds).all():
            raise ValueError('Placement query times must be finite')
        lower = seconds - PLACEMENT_TOLERANCE_SEC
        upper = seconds + PLACEMENT_TOLERANCE_SEC
        # Include exact tolerance endpoints despite decimal clock roundoff.
        left = np.searchsorted(self.times, lower - 1e-9, side='left')
        right = np.searchsorted(self.times, upper + 1e-9, side='right')
        targets = ((self.prefix[right] - self.prefix[left]) > 0.).astype(np.float32)
        valid = (seconds >= 0.) & (seconds < self.end_sec)
        targets[~valid] = 0.
        confidence = valid.astype(np.float32)
        for lo, hi in self.views.unknown_intervals:
            confidence[(upper >= lo) & (lower < hi)] = 0.
        for lo, hi, weight in self.views.regions:
            if weight > 0.:
                confidence[(upper >= lo) & (lower < hi)] *= min(1., weight)
        weights = np.broadcast_to(confidence[:, None], targets.shape).copy()
        return targets, weights
