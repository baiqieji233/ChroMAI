"""Authored workload supervision and causal, seconds-based prefix statistics."""
from __future__ import annotations

from fractions import Fraction
from functools import lru_cache

import numpy as np

from maichart.notes import parse_timing_point_notes
from maichart.tempo_clock import TempoClock
from maichart.timing import RawTimingPoint
from .events import event_metadata


WORKLOAD_DIM = 5
WORKLOAD_CHANNELS = (
    'event_rate', 'head_rate', 'hold_fraction', 'slide_fraction', 'touch_fraction',
)
LEDGER_DIM = 6
LEDGER_CHANNELS = (
    'event_rate_1s', 'head_rate_1s', 'event_rate_4s', 'head_rate_4s',
    'log_seconds_since_last', 'history_span_4s',
)


@lru_cache(maxsize=65536)
def event_statistics(note_text: str, bpm: float = 120.) -> tuple[int, float, float, float]:
    """Return distinct struck heads and event-level Hold/Slide/Touch flags.

    Shared-head Slide branches count once. Compact chords and positioned Touch
    notes retain their parsed heads; a Touch Hold belongs to Hold and Touch.
    These are authored head statistics, not a number of required player hands.
    """
    parsed = parse_timing_point_notes(RawTimingPoint(
        index=0, raw=note_text, note_text=note_text, beat=Fraction(0), tick=0,
        time_sec=0., division=4, bpm=float(bpm),
    ))
    if parsed.unknown_tokens:
        raise ValueError(f'Cannot derive workload from unsupported notes: {note_text!r}')
    heads = {str(note.position) for note in parsed.notes}
    kinds = {note.note_type for note in parsed.notes}
    return (len(heads), float(bool(kinds & {'hold', 'touch_hold'})),
            float('slide' in kinds), float(bool(kinds & {'touch', 'touch_hold'})))


def _triangle_integral(position):
    position = np.clip(np.asarray(position, dtype=np.float64), -1., 1.)
    u = 1. - np.abs(position)
    half_u2 = .5 * (u * u)
    return np.where(position < 0., half_u2, 1. - half_u2)


def build_workload_targets(points, tempo, first, centers_seconds, duration_sec):
    """Smooth authored rates and type fractions around absolute audio seconds.

    A triangular kernel covers each center +/- one second. Rates use its true
    area after clipping to the audio extent, including intro/outro silence.
    Prefix sums of counts and count*time avoid rescanning events per query.
    This function creates labels only; none of its chart data are audio input.
    """
    centers = np.asarray(centers_seconds, dtype=np.float64).reshape(-1)
    output = np.zeros((len(centers), WORKLOAD_DIM), dtype=np.float32)
    duration = float(duration_sec)
    if not len(centers) or duration <= 0.:
        return output
    clock = TempoClock(tempo, float(first))
    timestamps = []
    statistics = []
    for point in points:
        second = float(clock.at(point.beat))
        if not (0. <= second < duration):
            continue
        heads, hold, slide, touch = event_statistics(str(point.note_text), clock.bpm(point.beat))
        if not heads:
            continue
        timestamps.append(second)
        statistics.append((1., heads, hold, slide, touch))
    if not timestamps:
        return output
    times = np.asarray(timestamps, dtype=np.float64)
    order = np.argsort(times, kind='stable')
    times = times[order]
    values = np.asarray(statistics, dtype=np.float64)[order]
    counts = np.concatenate((np.zeros((1, WORKLOAD_DIM)), values.cumsum(axis=0)))
    moments = np.concatenate((np.zeros((1, WORKLOAD_DIM)),
                              (values * times[:, None]).cumsum(axis=0)))
    lower = np.searchsorted(times, centers - 1., side='left')
    middle = np.searchsorted(times, centers, side='right')
    upper = np.searchsorted(times, centers + 1., side='right')
    left = ((1. - centers[:, None]) * (counts[middle] - counts[lower])
            + moments[middle] - moments[lower])
    right = ((1. + centers[:, None]) * (counts[upper] - counts[middle])
             - moments[upper] + moments[middle])
    weighted = np.maximum(left + right, 0.)
    area = _triangle_integral(duration - centers) - _triangle_integral(-centers)
    valid = (centers >= 0.) & (centers < duration) & (area > 0.)
    output[valid, :2] = np.log1p(weighted[valid, :2] / area[valid, None])
    active = valid & (weighted[:, 0] > 1e-9)
    output[active, 2:] = np.clip(weighted[active, 2:] / weighted[active, :1], 0., 1.)
    return np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


class ExactWorkloadQuery:
    """Reusable exact +/-1s triangular workload labels for one authored chart.

    Unlike ``build_workload_targets``, this object parses the authored event
    list once and answers arbitrary query times directly from prefix sums; it
    never interpolates labels sampled on a grid. ``parsed=True`` uses the
    shared Composer event semantics; ``parsed=False`` preserves the legacy
    workload parser semantics for callers that explicitly need them.
    Points should already be filtered to the caller's valid/trusted view.
    """
    def __init__(self, points, tempo, first, duration_sec, *, parsed=True,
                 unknown_intervals=(), regions=()):
        self.duration = float(duration_sec)
        if not np.isfinite(self.duration) or self.duration < 0:
            raise ValueError('Workload duration must be finite and nonnegative')
        self.clock = TempoClock(tempo, float(first))
        unknown = []
        weighted_regions = []
        for lo, hi in unknown_intervals:
            lo, hi = float(lo), float(hi)
            if not np.isfinite(lo) or not np.isfinite(hi) or hi < lo:
                raise ValueError('Invalid unknown workload interval')
            lo, hi = max(0., lo), min(self.duration, hi)
            if hi > lo:
                unknown.append((lo, hi))
        for row in regions:
            lo, hi, weight = map(float, row)
            if not all(np.isfinite(x) for x in (lo, hi, weight)) or hi < lo or weight < 0.:
                raise ValueError('Invalid workload confidence region')
            lo, hi = max(0., lo), min(self.duration, hi)
            if hi <= lo:
                continue
            if weight <= 0.:
                unknown.append((lo, hi))
            else:
                weighted_regions.append((lo, hi, min(1., weight)))
        merged = []
        for lo, hi in sorted(unknown):
            if merged and lo <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(hi, merged[-1][1]))
            else:
                merged.append((lo, hi))
        self.unknown_intervals = tuple(merged)
        self.regions = tuple(weighted_regions)
        cuts=sorted({0.,self.duration,*[x for pair in merged for x in pair]})
        segments=[]
        for lo,hi in zip(cuts,cuts[1:]):
            middle=(lo+hi)*.5
            confidence=0. if any(a<=middle<b for a,b in merged) else 1.
            if hi>lo and confidence>0:segments.append((lo,hi,confidence))
        self.mass_segments=np.asarray(segments,dtype=np.float64).reshape(-1,3)
        rows = []
        for point in points:
            second = float(self.clock.at(point.beat))
            if not np.isfinite(second):
                raise ValueError('Workload event time must be finite')
            if not (0. <= second < self.duration):
                continue
            if any(lo <= second < hi for lo, hi in self.unknown_intervals):
                continue
            if parsed:
                meta = event_metadata(str(point.note_text))
                heads = len(meta.heads)
                hold, slide, touch = meta.hold, meta.slide, meta.touch
            else:
                heads, hold, slide, touch = event_statistics(
                    str(point.note_text), self.clock.bpm(point.beat))
            if heads:
                rows.append((second, 1., heads, float(hold), float(slide), float(touch)))
        rows.sort(key=lambda row: row[0])
        if rows:
            values = np.asarray(rows, dtype=np.float64)
            self.times = values[:, 0]
            self.prefix = np.vstack((np.zeros((1, WORKLOAD_DIM)), np.cumsum(values[:, 1:], axis=0)))
            self.moments = np.vstack((np.zeros((1, WORKLOAD_DIM)),
                np.cumsum(values[:, 1:] * self.times[:, None], axis=0)))
        else:
            self.times = np.empty(0, dtype=np.float64)
            self.prefix = np.zeros((1, WORKLOAD_DIM), dtype=np.float64)
            self.moments = np.zeros((1, WORKLOAD_DIM), dtype=np.float64)

    def weight_at(self, centers_seconds):
        """Confidence for each query; unknown-supported kernels are masked whole."""
        centers = np.asarray(centers_seconds, dtype=np.float64).reshape(-1)
        weights = np.ones(len(centers), dtype=np.float32)
        valid = np.isfinite(centers) & (centers >= 0.) & (centers < self.duration)
        weights[~valid] = 0.
        for lo, hi in self.unknown_intervals:
            # A triangular kernel is positive throughout the open interval
            # (center-1, center+1); touching only at a zero-mass endpoint is safe.
            weights[(centers + 1. > lo) & (centers - 1. < hi)] = 0.
        for lo, hi, value in self.regions:
            weights[(centers + 1. > lo) & (centers - 1. < hi)] *= value
        return weights

    def _known_kernel_mass(self, centers):
        """Integrate the triangular kernel over known time, independent of confidence."""
        if not self.unknown_intervals:
            return _triangle_integral(self.duration-centers)-_triangle_integral(-centers)
        if not len(self.mass_segments):return np.zeros(len(centers),dtype=np.float64)
        left,right,weights=self.mass_segments.T
        area=(_triangle_integral(right[None]-centers[:,None])
              -_triangle_integral(left[None]-centers[:,None]))
        return area@weights

    def query(self, centers_seconds):
        """Return exact labels at arbitrary absolute seconds as float32 [N,5]."""
        centers = np.asarray(centers_seconds, dtype=np.float64).reshape(-1)
        output = np.zeros((len(centers), WORKLOAD_DIM), dtype=np.float32)
        if not len(centers) or self.duration <= 0. or not len(self.times):
            return output
        lower = np.searchsorted(self.times, centers - 1., side='left')
        middle = np.searchsorted(self.times, centers, side='right')
        upper = np.searchsorted(self.times, centers + 1., side='right')
        left = ((1. - centers[:, None]) * (self.prefix[middle] - self.prefix[lower])
                + self.moments[middle] - self.moments[lower])
        right = ((1. + centers[:, None]) * (self.prefix[upper] - self.prefix[middle])
                 - self.moments[upper] + self.moments[middle])
        weighted = np.maximum(left + right, 0.)
        area = self._known_kernel_mass(centers)
        valid = np.isfinite(centers) & (centers >= 0.) & (centers < self.duration) & (area > 0.)
        output[valid, :2] = np.log1p(weighted[valid, :2] / area[valid, None])
        active = valid & (weighted[:, 0] > 1e-9)
        output[active, 2:] = np.clip(weighted[active, 2:] / weighted[active, :1], 0., 1.)
        if not np.isfinite(output).all():
            np.nan_to_num(output, copy=False, nan=0., posinf=0., neginf=0.)
        return output


def exact_workload_targets(points, tempo, first, centers_seconds, duration_sec, *, parsed=True):
    """Convenience one-shot exact query; use ``ExactWorkloadQuery`` to reuse."""
    return ExactWorkloadQuery(points, tempo, first, duration_sec, parsed=parsed).query(centers_seconds)


def causal_ledger(history_seconds, history_head_counts, query_seconds):
    """Query committed-prefix event/head rates without adding future events.

    Windows are left-open and right-closed. A future candidate clock can age
    the existing prefix, but cannot invent a candidate event or its head count.
    Queries preceding committed events see only the already elapsed prefix.
    """
    times = np.asarray(history_seconds, dtype=np.float64).reshape(-1)
    heads = np.asarray(history_head_counts, dtype=np.float64).reshape(-1)
    queries = np.asarray(query_seconds, dtype=np.float64).reshape(-1)
    if len(times) != len(heads):
        raise ValueError('Committed event times and head counts must have the same length')
    result = np.zeros((len(queries), LEDGER_DIM), dtype=np.float32)
    if not len(times) or not len(queries):
        return result
    order = np.argsort(times, kind='stable')
    times, heads = times[order], heads[order]
    return prefix_ledger(times, np.concatenate(([0.], heads.cumsum())), queries)


def prefix_ledger(times, sums, queries, *, committed_count=None):
    """Query an already sorted authored prefix without re-sorting per sample.

    ``committed_count`` bounds BOTH counts and head-prefix queries; future
    authored events in the reusable table cannot enter a training input.
    """
    times = np.asarray(times, dtype=np.float64)
    sums = np.asarray(sums, dtype=np.float64)
    queries = np.asarray(queries, dtype=np.float64).reshape(-1)
    result = np.zeros((len(queries), LEDGER_DIM), dtype=np.float32)
    count = len(times) if committed_count is None else int(committed_count)
    if count < 0 or count > len(times):
        raise ValueError('Invalid committed prefix length')
    times = times[:count]
    if not len(times) or not len(queries):
        return result
    right = np.searchsorted(times, queries, side='right')
    left1 = np.searchsorted(times, queries - 1., side='right')
    left4 = np.searchsorted(times, queries - 4., side='right')
    result[:, 0] = right - left1
    result[:, 1] = sums[right] - sums[left1]
    result[:, 2] = (right - left4) * .25
    result[:, 3] = (sums[right] - sums[left4]) * .25
    visible = right > 0
    if visible.any():
        vis_idx = right[visible] - 1
        last = times[vis_idx]
        result[visible, 4] = np.log1p(np.maximum(0., queries[visible] - last))
        result[visible, 5] = np.clip((last - times[0]) * .25, 0., 1.)
    if not np.isfinite(result).all():
        np.nan_to_num(result, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return result


def prefix_ledger_batch(times, sums, queries, committed_counts):
    """Exact causal ledger for same-chart [rows, candidates] query grids.

    Searching the full clock once and clipping each row's indices to its own
    committed count is identical to searching separate truncated prefixes.
    Future authored events can therefore never contribute to these inputs.
    """
    times = np.asarray(times, dtype=np.float64)
    sums = np.asarray(sums, dtype=np.float64)
    queries = np.asarray(queries, dtype=np.float64)
    counts = np.asarray(committed_counts, dtype=np.intp)
    if queries.ndim != 2 or counts.shape != (len(queries),):
        raise ValueError('Ledger batch requires a query grid and one committed count per row')
    if np.any((counts < 0) | (counts > len(times))):
        raise ValueError('Invalid committed prefix length')
    result = np.zeros((*queries.shape, LEDGER_DIM), dtype=np.float32)
    if not len(times) or not queries.size:
        return result
    right = np.minimum(np.searchsorted(times, queries, side='right'), counts[:, None])
    left1 = np.minimum(np.searchsorted(times, queries - 1., side='right'), counts[:, None])
    left4 = np.minimum(np.searchsorted(times, queries - 4., side='right'), counts[:, None])
    result[..., 0] = right - left1
    result[..., 1] = sums[right] - sums[left1]
    result[..., 2] = (right - left4) * .25
    result[..., 3] = (sums[right] - sums[left4]) * .25
    visible = right > 0
    if visible.any():
        last = times[right[visible] - 1]
        result[..., 4][visible] = np.log1p(np.maximum(0., queries[visible] - last))
        result[..., 5][visible] = np.clip((last - times[0]) * .25, 0., 1.)
    if not np.isfinite(result).all():
        np.nan_to_num(result, copy=False, nan=0., posinf=0., neginf=0.)
    return result


def trailing_workload_targets(times, head_prefix, queries):
    """Label log1p rates in exactly the same past 1s/4s windows as ledger."""
    times = np.asarray(times, dtype=np.float64)
    sums = np.asarray(head_prefix, dtype=np.float64)
    queries = np.asarray(queries, dtype=np.float64).reshape(-1)
    result = np.zeros((len(queries), 4), dtype=np.float32)
    if not len(times) or not len(queries):
        return result
    right = np.searchsorted(times, queries, side='right')
    left1 = np.searchsorted(times, queries - 1., side='right')
    left4 = np.searchsorted(times, queries - 4., side='right')
    result[:, 0] = right - left1
    result[:, 1] = sums[right] - sums[left1]
    result[:, 2] = (right - left4) * .25
    result[:, 3] = (sums[right] - sums[left4]) * .25
    np.log1p(result, out=result)
    if not np.isfinite(result).all():
        np.nan_to_num(result, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return result
