"""Section-local type multipliers from the planner's audio-predicted load curve.

The curve channels are (event_density, tap share, sustain share, slide/touch
share), predicted from audio alone. A multiplier is the local share relative
to its density-weighted song mean, so the requested whole-song style target
keeps its mean while concentrating where the music supports each type.
"""
from __future__ import annotations

import math

import numpy as np

# HEAD_TYPES = tap, slide, hold, touch, touch_hold.
_EPS = .02


class SectionTypeProfile:
    def __init__(self, curve, times, *, window_seconds=4.):
        curve = np.asarray(curve, dtype=np.float64)
        times = np.asarray(times, dtype=np.float64)
        if curve.ndim != 2 or curve.shape[1] != 4 or len(curve) != len(times) or not len(times):
            raise ValueError('Section profile requires a [T,4] curve on its time axis')
        if not math.isfinite(window_seconds) or window_seconds <= 0:
            raise ValueError('window_seconds must be finite and positive')
        order = np.argsort(times, kind='stable')
        self.times = times[order]
        values = np.clip(curve[order], 0., 1.)
        density = values[:, 0]
        shares = values[:, 1:]
        weight = density.sum()
        self.mean_density = float(density.mean())
        self.density_cumulative = np.concatenate(([0.], np.cumsum(density)))
        self.reference = ((shares*density[:, None]).sum(0)/weight if weight > 1e-9 else shares.mean(0))
        self.cumulative = np.vstack((np.zeros((1, 3)), np.cumsum(shares, 0)))
        self.window = float(window_seconds)

    def density_factor(self, second):
        left = int(np.searchsorted(self.times, second-self.window, side='left'))
        right = int(np.searchsorted(self.times, second+self.window, side='right'))
        if right <= left:
            return 1.
        local = float((self.density_cumulative[right]-self.density_cumulative[left])/(right-left))
        return float(np.clip((local+_EPS)/(self.mean_density+_EPS), .25, 2.5))

    def __call__(self, second):
        left = int(np.searchsorted(self.times, second-self.window, side='left'))
        right = int(np.searchsorted(self.times, second+self.window, side='right'))
        if right <= left:
            return np.ones(5)
        local = (self.cumulative[right]-self.cumulative[left])/(right-left)
        tap, sustain, slide_touch = (local+_EPS)/(self.reference+_EPS)
        mixed = math.sqrt(sustain*slide_touch)
        return np.asarray([tap, mixed, sustain, slide_touch, mixed])
