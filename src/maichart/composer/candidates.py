"""Shared candidate-time generation for training and free generation."""
from __future__ import annotations

from bisect import bisect_right
from functools import lru_cache
from fractions import Fraction
import math

import numpy as np

from maichart.form.rational_audio import beat_at_second
from maichart.event_timing import seconds_at_beat
from maichart.tempo_clock import TempoClock


_FINE_OFFSETS: tuple[Fraction, ...] = tuple(sorted({
    Fraction(index, denominator)
    for denominator in (1, 2, 3, 4, 6, 8, 12, 16)
    for index in range(1, denominator * 2 + 1)
}))
_NEAR_REQUIRED: tuple[Fraction, ...] = (
    Fraction(1, 16), Fraction(1, 12), Fraction(1, 8), Fraction(1, 6), Fraction(1, 4),
    Fraction(1, 3), Fraction(3, 8), Fraction(1, 2), Fraction(2, 3), Fraction(3, 4), Fraction(1),
)
_FAR_OFFSETS: tuple[Fraction, ...] = (
    Fraction(2), Fraction(3), Fraction(4), Fraction(6), Fraction(8), Fraction(12), Fraction(16),
)
_PRIORITY_TAIL: tuple[Fraction, ...] = (
    tuple(offset for offset in _FINE_OFFSETS if offset not in _NEAR_REQUIRED)
    + (
        Fraction(7, 3), Fraction(9, 4), Fraction(17, 8), Fraction(5, 2), Fraction(11, 4), Fraction(3),
        Fraction(13, 4), Fraction(7, 2), Fraction(15, 4), Fraction(4), Fraction(6), Fraction(8),
        Fraction(12), Fraction(16),
    )
)
_ALL_CANONICAL_OFFSETS: tuple[Fraction, ...] = tuple(sorted({
    value for value in (
        list(_FINE_OFFSETS)
        + [Fraction(index, 4) for index in range(6, 65)]
        + [Fraction(index, 3) for index in range(7, 49)]
        + [
            Fraction(5, 3), Fraction(7, 4), Fraction(11, 6), Fraction(7, 3),
            Fraction(17, 8), Fraction(19, 8), Fraction(23, 8), Fraction(25, 8),
            Fraction(5, 2), Fraction(11, 4), Fraction(13, 4), Fraction(7, 2),
            Fraction(15, 4), Fraction(4), Fraction(6), Fraction(8), Fraction(12), Fraction(16),
        ]
    ) if value > 0
}))


def _tempo_step_seconds(tempo, first, beat, fraction: Fraction) -> float:
    before = float(seconds_at_beat(beat, tempo, first))
    after = float(seconds_at_beat(beat + fraction, tempo, first))
    return max(0., after - before)


def _canonical_offsets(max_rest_beats: Fraction) -> list[Fraction]:
    """Return a bounded musical proposal grid shared by train and inference.

    The fine grid uses denominators supported by common maimai subdivisions.
    It deliberately excludes arbitrary 1/48-beat steps, which would create
    192nd-note proposals.  Coarser quarter/half-beat anchors preserve phrase
    movement and long rests without converting wall-clock seconds into
    fractions.
    """
    if max_rest_beats <= 0:
        return []
    return list(_ALL_CANONICAL_OFFSETS[:bisect_right(_ALL_CANONICAL_OFFSETS, max_rest_beats)])


@lru_cache(maxsize=4096)
def _select_candidate_offsets(
    first_event: bool,
    valid_len: int,
    count: int,
) -> tuple[tuple[Fraction, ...], dict[Fraction, int], np.ndarray]:
    offsets = _ALL_CANONICAL_OFFSETS[:valid_len]
    values = ([Fraction(0)] if first_event else []) + list(offsets)
    if len(values) > count:
        priority_offsets = (
            (([Fraction(0)] if first_event else []) + list(_NEAR_REQUIRED) + list(_PRIORITY_TAIL))
        )
        by_offset = {value: value for value in values}
        far = [by_offset[offset] for offset in _FAR_OFFSETS if offset in by_offset]
        far_slots = min(len(far), max(0, count // 4))
        near_slots = max(0, count - far_slots)
        selected = [by_offset[offset] for offset in priority_offsets if offset in by_offset][:near_slots]
        remaining_far = [value for value in far if value not in selected]
        if far_slots and remaining_far:
            indices = sorted(set(
                round(i * (len(remaining_far) - 1) / max(1, far_slots - 1))
                for i in range(far_slots)
            ))
            selected.extend(remaining_far[index] for index in indices)
        remaining = [value for value in values if value not in selected]
        slots = count - len(selected)
        if slots and remaining:
            indices = sorted(set(
                round(i * (len(remaining) - 1) / max(1, slots - 1))
                for i in range(slots)
            ))
            selected.extend(remaining[index] for index in indices)
        values = sorted(set(selected))
    selected_tuple = tuple(values)
    lookup = {offset: idx for idx, offset in enumerate(selected_tuple)}
    floats = np.asarray([float(offset) for offset in selected_tuple], dtype=np.float64)
    return selected_tuple, lookup, floats


def candidate_spec(
    previous: Fraction | None,
    *,
    tempo,
    first: float,
    audio_end_sec: float,
    candidate_count: int = 96,
    max_rest_sec: float = 16.,
    clock: TempoClock | None = None,
) -> tuple[bool, Fraction, int, tuple[Fraction, ...], dict[Fraction, int], np.ndarray]:
    first_event = previous is None
    previous_beat = Fraction(0) if first_event else Fraction(previous)
    count = int(candidate_count)
    if count < 1:
        raise ValueError("candidate_count must be positive")
    max_rest_sec = max(0.0, float(max_rest_sec))
    timeline = clock if clock is not None else TempoClock(tempo, first)
    clock_previous = max(Fraction(0), previous_beat)
    previous_second = float(timeline.at(clock_previous))
    max_canonical_last = _ALL_CANONICAL_OFFSETS[-1]
    if float(timeline.at(clock_previous + max_canonical_last)) - previous_second <= max_rest_sec - 1e-12:
        hi = len(_ALL_CANONICAL_OFFSETS)
    else:
        max_rest_beats = beat_at_second(previous_second + max_rest_sec, tempo, first) - clock_previous
        max_rest_beats = max(Fraction(1, 48), max_rest_beats)
        hi = bisect_right(_ALL_CANONICAL_OFFSETS, max_rest_beats)
    end_sec = float(audio_end_sec)
    if hi > 0 and float(timeline.at(previous_beat + _ALL_CANONICAL_OFFSETS[hi - 1])) >= end_sec:
        if float(timeline.at(previous_beat + _ALL_CANONICAL_OFFSETS[0])) >= end_sec:
            hi = 0
        else:
            lo = 0
            while lo < hi:
                mid = (lo + hi) // 2
                if float(timeline.at(previous_beat + _ALL_CANONICAL_OFFSETS[mid])) < end_sec:
                    lo = mid + 1
                else:
                    hi = mid
            hi = lo
    selected, lookup, floats = _select_candidate_offsets(first_event, hi, count)
    return first_event, previous_beat, hi, selected, lookup, floats


def candidate_beats(previous: Fraction | None, *, tempo, first: float, audio_end_sec: float,
                    candidate_count: int = 96, max_rest_sec: float = 16.,
                    target: Fraction | None = None, clock: TempoClock | None = None) -> list[Fraction]:
    """Return deterministic bounded-fraction candidates.

    ``target`` is accepted for API compatibility but is deliberately never
    injected into the proposal set.  Training and inference must see the same
    candidates; callers should record or reject authored events that are not
    recalled by this grid instead of silently teaching a special candidate.
    ``max_rest_sec`` is converted only into an upper beat horizon.  Candidate
    positions themselves always come from the canonical musical grid.
    """
    del target
    _, previous_beat, _, selected, _, _ = candidate_spec(
        previous,
        tempo=tempo,
        first=first,
        audio_end_sec=audio_end_sec,
        candidate_count=candidate_count,
        max_rest_sec=max_rest_sec,
        clock=clock,
    )
    if previous_beat == 0:
        return list(selected)
    return [previous_beat + offset for offset in selected]


def candidate_index(target: Fraction, candidates: list[Fraction]) -> int:
    if target in candidates:
        return candidates.index(target)
    return -1


def candidate_features(
    candidates,
    previous,
    tempo,
    first,
    audio_times=None,
    *,
    clock: TempoClock | None = None,
    candidate_seconds: np.ndarray | None = None,
    beat_floats: np.ndarray | None = None,
):
    """Numeric time features; audio probes are attached by the dataset/model."""
    del audio_times
    if (beat_floats is None and not candidates) or (beat_floats is not None and len(beat_floats) == 0):
        return np.zeros((0, 5), dtype=np.float32)
    timeline = clock if clock is not None else TempoClock(tempo, first)
    previous_beat = Fraction(0) if previous is None else max(Fraction(0), Fraction(previous))
    previous_second = float(timeline.at(previous_beat))
    if beat_floats is None:
        beat_floats = np.asarray([float(beat) for beat in candidates], dtype=np.float64)
    if candidate_seconds is None:
        candidate_seconds = np.asarray([float(timeline.at(beat)) for beat in candidates], dtype=np.float64)
    delta_beats = (
        np.asarray([float(beat - previous_beat) for beat in candidates], dtype=np.float64)
        if candidates is not None and len(candidates) == len(beat_floats)
        else beat_floats - float(previous_beat)
    )
    delta_sec = np.maximum(0.0, candidate_seconds - previous_second)
    angles = 2.0 * math.pi * beat_floats
    out = np.empty((len(beat_floats), 5), dtype=np.float32)
    out[:, 0] = delta_beats
    out[:, 1] = delta_sec
    out[:, 2] = np.log1p(delta_sec)
    out[:, 3] = np.sin(angles)
    out[:, 4] = np.cos(angles)
    return out


def interp_candidate_audio(values: np.ndarray, times: np.ndarray, seconds: np.ndarray) -> np.ndarray:
    """Vectorized 2D linear interpolation matching np.interp(..., left=0., right=0.)."""
    seconds = np.asarray(seconds, dtype=np.float64)
    n_queries = len(seconds)
    n_times = len(times)
    if n_times == 0 or n_queries == 0:
        return np.zeros((n_queries, values.shape[1]), dtype=np.float32)
    if n_times == 1:
        out = np.repeat(values[:1].astype(np.float32, copy=False), n_queries, axis=0)
        out[(seconds < times[0]) | (seconds > times[0])] = 0.0
        return out
    right = np.searchsorted(times, seconds, side='left')
    right_clipped = np.clip(right, 1, n_times - 1)
    left_clipped = right_clipped - 1
    t0 = times[left_clipped]
    t1 = times[right_clipped]
    denom = t1 - t0
    safe_denom = np.where(denom > 0.0, denom, 1.0)
    weight = np.where(denom > 0.0, (seconds - t0) / safe_denom, 0.0)
    weight = np.clip(weight, 0.0, 1.0)[:, None]
    v0 = values[left_clipped].astype(np.float64, copy=False)
    v1 = values[right_clipped].astype(np.float64, copy=False)
    out = (v0 + weight * (v1 - v0)).astype(np.float32)
    outside = (seconds < times[0]) | (seconds > times[-1])
    if outside.any():
        out[outside] = 0.0
    return out


def composer_audio_nodes(song) -> tuple[np.ndarray, np.ndarray]:
    """Build the 120-dim composer feature bank directly from cached RationalAudio nodes."""
    from maichart.form.timing import normalize_meter_map
    audio = song.audio
    acoustic = np.asarray(audio.features, dtype=np.float32)
    times = np.asarray(audio.times, dtype=np.float64)
    n = len(times)
    if n == 0:
        return np.zeros((0, 120), dtype=np.float32), times
    duration = max(float(audio.duration_sec), 1e-6)
    changes = sorted(song.tempo, key=lambda t: t.beat)
    t_beats = np.asarray([float(c.beat) for c in changes], dtype=np.float64)
    t_bpms = np.asarray([float(c.bpm) for c in changes], dtype=np.float64)
    t_secs = np.empty(len(changes), dtype=np.float64)
    t_secs[0] = float(song.first)
    for i in range(1, len(changes)):
        t_secs[i] = t_secs[i - 1] + (t_beats[i] - t_beats[i - 1]) * 60.0 / t_bpms[i - 1]
    t_idx = np.clip(np.searchsorted(t_secs, times, side='right') - 1, 0, len(changes) - 1)
    beat_values = t_beats[t_idx] + (times - t_secs[t_idx]) * t_bpms[t_idx] / 60.0
    bpm_values = t_bpms[t_idx]

    regions = getattr(song, 'regions', None)
    spans = getattr(song, 'spans', None)
    extent = float(regions[-1].end if regions else (spans[-1].end if spans else max(1.0, float(beat_values[-1]))))
    progress = np.clip(times / duration, 0.0, 1.0)
    rhythm = np.column_stack((acoustic[:, 92], acoustic[:, 92], acoustic[:, 94], acoustic[:, 93]))
    context = np.column_stack((
        progress,
        np.full(n, min(duration / 300.0, 1.0)),
        bpm_values / 300.0,
        np.sin(2.0 * np.pi * beat_values),
        np.cos(2.0 * np.pi * beat_values),
        beat_values / max(extent, 1.0),
        np.zeros((n, 4)),
    ))
    metrical = np.zeros((n, 8), dtype=np.float32)
    q_angles = 2.0 * np.pi * np.mod(beat_values, 1.0)
    metrical[:, 0] = np.sin(q_angles)
    metrical[:, 1] = np.cos(q_angles)
    meter_map = normalize_meter_map(getattr(song, 'meters', None))
    if meter_map:
        m_beats = np.asarray([float(m.beat) for m in meter_map], dtype=np.float64)
        m_idx = np.searchsorted(m_beats, beat_values, side='right') - 1
        for i_m, active in enumerate(meter_map):
            sel = m_idx == i_m
            if not np.any(sel):
                continue
            b_sub = beat_values[sel]
            bar = float(active.beats_per_bar)
            within_bar = np.mod(b_sub - float(active.beat), bar)
            bar_angle = 2.0 * np.pi * (within_bar / bar)
            metrical[sel, 2] = np.sin(bar_angle)
            metrical[sel, 3] = np.cos(bar_angle)
            metrical[sel, 6] = 1.0
            if active.grouping:
                g_start = 0.0
                g_phase = np.zeros_like(within_bar)
                for g_size in active.grouping:
                    g_len = float(g_size * 4) / float(active.denominator)
                    in_g = (within_bar >= g_start) & (within_bar < g_start + g_len)
                    g_phase[in_g] = (within_bar[in_g] - g_start) / g_len
                    g_start += g_len
                g_angle = 2.0 * np.pi * g_phase
                metrical[sel, 4] = np.sin(g_angle)
                metrical[sel, 5] = np.cos(g_angle)
                metrical[sel, 7] = 1.0
    values = np.concatenate((acoustic, rhythm, context, metrical), axis=1).astype(np.float32, copy=False)
    outside = (times < 0.0) | (times >= float(audio.duration_sec))
    if outside.any():
        values = values.copy()
        values[outside] = 0.0
    return values, times
