"""Select real score peaks and suppress near-duplicate onset candidates."""
from bisect import bisect_left
from fractions import Fraction
import math

import numpy as np


def local_peak_mask(scores):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError('Onset scores must be a finite one-dimensional series')
    if not len(scores):
        return np.zeros(0, dtype=np.bool_)
    # A plateau has one representative, not one event per plateau frame.
    return np.r_[True, scores[1:] > scores[:-1]] & np.r_[scores[:-1] >= scores[1:], True]


def merge_close_onsets(records, window_ms):
    """Keep the strongest event within a time radius; never average its beat.

    Work on the final tempo-integrated positions so rational snapping cannot
    reintroduce close pairs. Greedy suppression avoids transitively combining
    an arbitrarily long run of genuine rapid notes into one event.
    """
    window_ms = float(window_ms)
    if not math.isfinite(window_ms) or not 0 <= window_ms <= 250:
        raise ValueError('onset_merge_ms must be in [0, 250]')
    if window_ms == 0:
        return sorted(records, key=lambda item: Fraction(item['beat']))
    radius = window_ms / 1000
    ranked = sorted(records, key=lambda item: (-float(item['score']), float(item['second']),
                                               Fraction(item['beat'])))
    selected, times = [], []
    for item in ranked:
        stamp = float(item['second'])
        index = bisect_left(times, stamp)
        left = index and stamp - times[index-1] < radius - 1e-9
        right = index < len(times) and times[index] - stamp < radius - 1e-9
        if left or right:
            continue
        times.insert(index, stamp)
        selected.insert(index, item)
    return selected


def interval_statistics(records):
    times = sorted(float(item['second']) for item in records)
    intervals = np.diff(times)
    return {'count': len(records),
            'minimum_interval_ms': float(intervals.min()*1000) if len(intervals) else None,
            'pairs_below_30ms': int(np.sum(intervals < .03-1e-9)),
            'pairs_below_50ms': int(np.sum(intervals < .05-1e-9))}
