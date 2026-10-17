"""Quality weights without deleting events or inventing authored rests."""
from __future__ import annotations

import numpy as np


def region_weight_at(second, annotation):
    for region in (annotation or {}).get('regions', []):
        if float(region['start_sec']) <= second < float(region['end_sec']):
            return float(region.get('weight', 1.))
    return 1.


def interval_weight(start, end, annotation):
    """Conservative supervision for a decision that spans an ignored region.

    A jump or EOS over any excluded interval has no trusted timing label.
    Positive regional weights otherwise act as duration-averaged multipliers.
    """
    if end <= start:
        return region_weight_at(end, annotation)
    result = end-start
    for region in (annotation or {}).get('regions', []):
        overlap = max(0., min(end, float(region['end_sec']))-max(start, float(region['start_sec'])))
        if overlap:
            weight = float(region.get('weight', 1.))
            if weight <= 0:
                return 0.
            result += overlap*(weight-1.)
    return max(0., result/(end-start))


def event_weight(previous_second, second, annotation):
    return min(region_weight_at(second, annotation), interval_weight(previous_second, second, annotation))


def curve_weights(song, times, annotation):
    """An interpolated label is trusted only if both source bins are trusted."""
    centers = np.asarray(song.load_curve_times_sec, dtype=np.float64) + float(song.first)
    if not (annotation or {}).get('regions'):
        return np.ones(len(times), dtype=np.float32)
    width = float(song.curve_bin_seconds)
    weights = np.asarray([interval_weight(float(song.first)+i*width,
        min(float(song.audio.duration_sec), float(song.first)+(i+1)*width), annotation)
        for i in range(len(centers))], dtype=np.float32)
    values = np.asarray(times, dtype=np.float64)
    right = np.searchsorted(centers, values, side='left').clip(0, len(centers)-1)
    left = (right-1).clip(0)
    # At a bin center the right bin alone supplies the target. Outside the
    # range np.interp extends the nearest bin, so its mask must also extend.
    left = np.where(values == centers[right], right, left)
    left = np.where(values >= centers[-1], len(centers)-1, left)
    return np.minimum(weights[left], weights[right])
