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


# Composer feature policy is part of the training/inference contract.  The
# attack-derived columns are confidence-weighted at candidate time so a sparse
# evidence peak cannot become a full-width synthetic onset plateau.
from .acoustics import FEATURE_POLICY as COMPOSER_AUDIO_FEATURE_POLICY

CANDIDATE_POLICY = 'absolute_recovery_v4'
MUSIC_CANDIDATE_POLICY = 'absolute_recovery_v5'
MUSIC_CANDIDATE_POLICY_V6 = 'absolute_recovery_v6_phase_safe'
_MIN_MUSIC_RECOVERY_GAP = Fraction(1, 16)
CANONICAL_BEAT_DENOMINATORS = frozenset({1, 2, 3, 4, 6, 8})


def is_playable_candidate_transition(
    beat: Fraction,
    committed_previous: Fraction | None,
    policy: str = MUSIC_CANDIDATE_POLICY_V6,
    *,
    min_gap: Fraction = _MIN_MUSIC_RECOVERY_GAP,
) -> bool:
    """Check whether ``beat`` is a legal phase-safe candidate after ``committed_previous``.

    Under ``MUSIC_CANDIDATE_POLICY_V6``, candidate beats must lie on the
    canonical absolute grid ``{1, 2, 3, 4, 6, 8}`` and maintain at least
    ``min_gap`` (1/16 beat) from ``committed_previous`` so binary/ternary
    transitions cannot create 1/24 or 1/48 micro-bridge notes, while still
    allowing direct return from ternary phases (e.g. 1/3, 1/6) to valid binary
    subdivisions (e.g. 3/4, 7/8).
    """
    if policy != MUSIC_CANDIDATE_POLICY_V6:
        return True
    beat = beat if isinstance(beat, Fraction) else Fraction(beat)
    if beat.denominator not in CANONICAL_BEAT_DENOMINATORS:
        return False
    if committed_previous is None:
        return True
    prev = committed_previous if isinstance(committed_previous, Fraction) else Fraction(committed_previous)
    delta = beat - prev
    if delta < min_gap:
        return False
    if prev.denominator not in CANONICAL_BEAT_DENOMINATORS:
        return True
    if delta.denominator not in CANONICAL_BEAT_DENOMINATORS:
        # Cross-grid binary/ternary transitions must never create <1/4 beat
        # micro-bridges such as 1/24 ({96}) or 1/12 ({48}, e.g. 2/3 -> 3/4).
        return delta >= Fraction(1, 4)
    if beat.denominator == 8 and prev.denominator in {1, 2, 4, 8}:
        # Enter a {32} (1/8-beat) off-beat only via a genuine 1/8-beat step,
        # never across 3/8, 5/8, 7/8 empty syncopation gaps.
        return delta == Fraction(1, 8)
    if prev.denominator == 8 and beat.denominator in {1, 2, 4}:
        # Exit a {32} off-beat either by completing the 1/8-beat pair onto the
        # 1/4 grid, landing on the next integer beat (e.g. 0->1/8 then 1), or
        # recovering to a >=1-beat anchor, never via a 3/8 hiccup ({32}x,y,{16},).
        return (
            delta == Fraction(1, 8)
            or (beat.denominator == 1 and delta >= Fraction(5, 8))
            or (beat.denominator == 2 and delta >= Fraction(1, 1))
        )
    return True


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
_LONG_REST_OFFSETS = tuple(Fraction(x) for x in (24,32,48,64,96,128))
_MUSIC_OFFSETS = tuple(sorted(set(_ALL_CANONICAL_OFFSETS + _LONG_REST_OFFSETS)))


def offset_universe(policy):
    return _MUSIC_OFFSETS if policy in {MUSIC_CANDIDATE_POLICY, MUSIC_CANDIDATE_POLICY_V6} else _ALL_CANONICAL_OFFSETS


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


def _select_candidate_offsets(
    first_event: bool,
    valid_len: int,
    count: int,
    previous: Fraction | None = None,
    policy: str = CANDIDATE_POLICY,
) -> tuple[tuple[Fraction, ...], dict[Fraction, int], np.ndarray]:
    phase = Fraction(0) if previous is None else Fraction(previous) % 1
    return _cached_candidate_offsets(first_event, valid_len, count, phase, policy)


@lru_cache(maxsize=32768)
def _cached_candidate_offsets(first_event, valid_len, count, phase, policy):
    offsets = offset_universe(policy)[:valid_len]
    values = ([Fraction(0)] if first_event else []) + list(offsets)
    recovery = []
    if policy in {CANDIDATE_POLICY, MUSIC_CANDIDATE_POLICY, MUSIC_CANDIDATE_POLICY_V6} and offsets and phase:
        # Include ways back to absolute grids instead of declaring mixed
        # fractions illegal. This also makes an existing /48 phase escapable.
        for denominator in (1, 2, 3, 4, 6, 8, 12, 16):
            tick = phase * denominator
            for step in (1, 2, 4):
                delta = Fraction(tick.numerator // tick.denominator + step, denominator) - phase
                if 0 < delta <= offsets[-1]:
                    recovery.append(delta)
        if policy in {MUSIC_CANDIDATE_POLICY, MUSIC_CANDIDATE_POLICY_V6}:
            # Off-grid generation must also recover to later integer/half-
            # beat anchors, not just the next few fine subdivisions. These
            # are clock proposals, never compulsory chart events.
            for denominator in (1, 2):
                tick = phase * denominator
                for step in (2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256):
                    delta = Fraction(tick.numerator // tick.denominator + step, denominator) - phase
                    if 0 < delta <= offsets[-1]: recovery.append(delta)
        if policy == MUSIC_CANDIDATE_POLICY_V6:
            # Switching between binary and ternary grids must not invent a
            # tiny bridge event (e.g. 5/16 -> 1/3 is only 1/48 beat). Later
            # anchors still permit recovery without forcing that bridge.
            recovery = [delta for delta in recovery if delta >= _MIN_MUSIC_RECOVERY_GAP]
        recovery = sorted(set(recovery))
        values = sorted(set(values + recovery))
    if len(values) > count:
        priority_offsets = (
            (([Fraction(0)] if first_event else []) + list(_NEAR_REQUIRED)
             + recovery + list(_PRIORITY_TAIL))
        )
        if policy in {MUSIC_CANDIDATE_POLICY, MUSIC_CANDIDATE_POLICY_V6}:
            # Reserve a bounded recovery share, interleaving coarse and fine
            # grids so pruning cannot remove every useful escape route.
            coarse = [x for x in recovery if (phase+x).denominator <= 2]
            recovery_priority = list(dict.fromkeys(coarse + recovery))[:max(2, count//4)]
            priority_offsets = (([Fraction(0)] if first_event else []) + list(_NEAR_REQUIRED)
                                + recovery_priority + list(_PRIORITY_TAIL) + recovery)
        by_offset = set(values)
        far = [offset for offset in (_FAR_OFFSETS+_LONG_REST_OFFSETS if policy in {MUSIC_CANDIDATE_POLICY, MUSIC_CANDIDATE_POLICY_V6} else _FAR_OFFSETS) if offset in by_offset]
        far_slots = min(len(far), max(0, count // 4))
        near_slots = max(0, count - far_slots)
        selected = list(dict.fromkeys(offset for offset in priority_offsets if offset in by_offset))[:near_slots]
        selected_set = set(selected)
        remaining_far = [value for value in far if value not in selected_set]
        if far_slots and remaining_far:
            indices = sorted(set(
                round(i * (len(remaining_far) - 1) / max(1, far_slots - 1))
                for i in range(far_slots)
            ))
            for index in indices:
                val = remaining_far[index]
                selected.append(val)
                selected_set.add(val)
        remaining = [value for value in values if value not in selected_set]
        slots = count - len(selected)
        if slots and remaining:
            indices = sorted(set(
                round(i * (len(remaining) - 1) / max(1, slots - 1))
                for i in range(slots)
            ))
            selected.extend(remaining[index] for index in indices)
        values = sorted(set(selected))
    if policy == 'phase_v3':
        filtered = [offset for offset in values if _phase_transition_allowed(phase, offset)]
        values = filtered or values
    selected_tuple = tuple(values)
    lookup = {offset: idx for idx, offset in enumerate(selected_tuple)}
    floats = np.asarray([float(offset) for offset in selected_tuple], dtype=np.float64)
    return selected_tuple, lookup, floats


def _mixed_denominator(value: Fraction) -> bool:
    denominator = int(value.denominator)
    return denominator % 2 == 0 and denominator % 3 == 0


def _phase_transition_allowed(previous: Fraction, offset: Fraction) -> bool:
    """Prevent a mixed binary/ternary phase from becoming persistent.

    Direct fine candidates such as 1/12 or 1/16 remain available from an
    aligned integer phase. Once a ternary phase is active, adding a binary
    offset that creates a mixed denominator is rejected; if a mixed phase was
    reached by an authored fine event, only candidates that escape it remain.
    This blocks the observed /48 -> {192} lock without deleting valid grids.
    """
    candidate = previous + offset
    if not _mixed_denominator(candidate):
        return True
    if _mixed_denominator(previous):
        return False
    previous_has_ternary = previous.denominator % 3 == 0
    offset_has_binary = offset.denominator % 2 == 0
    return not (previous_has_ternary and offset_has_binary)


def candidate_spec(
    previous: Fraction | None,
    *,
    tempo,
    first: float,
    audio_end_sec: float,
    candidate_count: int = 96,
    max_rest_sec: float | None = None,
    clock: TempoClock | None = None,
    policy: str = CANDIDATE_POLICY,
) -> tuple[bool, Fraction, int, tuple[Fraction, ...], dict[Fraction, int], np.ndarray]:
    first_event = previous is None
    previous_beat = Fraction(0) if first_event else Fraction(previous)
    count = int(candidate_count)
    if count < 1:
        raise ValueError("candidate_count must be positive")
    max_rest_sec = max(0.0, float(max_rest_sec if max_rest_sec is not None else 64. if policy in {MUSIC_CANDIDATE_POLICY, MUSIC_CANDIDATE_POLICY_V6} else 16.))
    universe=offset_universe(policy)
    timeline = clock if clock is not None else TempoClock(tempo, first)
    clock_previous = max(Fraction(0), previous_beat)
    previous_second = float(timeline.at(clock_previous))
    max_canonical_last = universe[-1]
    if float(timeline.at(clock_previous + max_canonical_last)) - previous_second <= max_rest_sec - 1e-12:
        hi = len(universe)
    else:
        max_rest_beats = beat_at_second(previous_second + max_rest_sec, tempo, first) - clock_previous
        max_rest_beats = max(Fraction(1, 48), max_rest_beats)
        hi = bisect_right(universe, max_rest_beats)
    end_sec = float(audio_end_sec)
    if hi > 0 and float(timeline.at(previous_beat + universe[hi - 1])) >= end_sec:
        if float(timeline.at(previous_beat + universe[0])) >= end_sec:
            hi = 0
        else:
            lo = 0
            while lo < hi:
                mid = (lo + hi) // 2
                if float(timeline.at(previous_beat + universe[mid])) < end_sec:
                    lo = mid + 1
                else:
                    hi = mid
            hi = lo
    selected, lookup, floats = _select_candidate_offsets(first_event, hi, count, previous_beat, policy)
    return first_event, previous_beat, hi, selected, lookup, floats


def candidate_beats(previous: Fraction | None, *, tempo, first: float, audio_end_sec: float,
                    candidate_count: int = 96, max_rest_sec: float | None = None,
                    target: Fraction | None = None, clock: TempoClock | None = None,
                    policy: str = CANDIDATE_POLICY) -> list[Fraction]:
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
        policy=policy,
    )
    if previous_beat == 0:
        return list(selected)
    return [previous_beat + offset for offset in selected]


# Rhythm grid level of an absolute candidate beat, used by timing samplers to
# keep the model's learned mass per grid level (integer / half / quarter /
# eighth / third / sixth) at T=1. Temperature and truncation then act only
# inside one level, so a sharp temperature cannot erase a whole rhythm family
# (e.g. every triplet position) just because it is the minority in its bin.
GRID_LEVEL_COUNT = 7
_GRID_LEVEL_BY_DENOMINATOR = {1: 0, 2: 1, 4: 2, 8: 3, 3: 4, 6: 5}


def candidate_grid_levels(candidates, width: int | None = None) -> np.ndarray:
    """Return int64 grid levels [width]; padding and off-grid beats use level 6."""
    size = len(candidates) if width is None else int(width)
    levels = np.full(size, GRID_LEVEL_COUNT - 1, dtype=np.int64)
    for index, beat in enumerate(candidates[:size]):
        value = beat if isinstance(beat, Fraction) else Fraction(beat)
        levels[index] = _GRID_LEVEL_BY_DENOMINATOR.get(value.denominator, GRID_LEVEL_COUNT - 1)
    return levels


# Rhythm family of the recent committed prefix. Official Master charts (341
# charts, 2026-10-29): after a ternary onset within 2 beats the next onset is
# ternary 41.6% of the time; after a binary one, 1.6%; with only integer onsets,
# 1.8%. Only 6.2% of ternary onsets are isolated (no other ternary onset within
# one beat). Integer beats belong to both families and never set the state.
RHYTHM_FAMILY_WINDOW = Fraction(2)
RHYTHM_FAMILY_STATES = ('binary', 'ternary', 'neutral')
_BINARY_LEVELS = (1, 2, 3)
_TERNARY_LEVELS = (4, 5)


def rhythm_family_state(committed_beats, window: Fraction = RHYTHM_FAMILY_WINDOW) -> str:
    """Family of the most recent non-integer committed onset within ``window`` beats of the last one."""
    beats = list(committed_beats)
    if not beats:
        return 'neutral'
    last = Fraction(beats[-1])
    for beat in reversed(beats):
        beat = Fraction(beat)
        if last - beat > window:
            break
        level = _GRID_LEVEL_BY_DENOMINATOR.get(beat.denominator)
        if level in _TERNARY_LEVELS:
            return 'ternary'
        if level in _BINARY_LEVELS:
            return 'binary'
    return 'neutral'


def family_switch_levels(state: str, *, entering_microstep: bool = False) -> np.ndarray:
    """Boolean [GRID_LEVEL_COUNT]: grid levels that would switch away from ``state``.

    Neutral (only integer onsets recently) defaults to binary, the common case.
    Integer beats are shared and never a switch; off-canonical levels always are.
    """
    if state not in RHYTHM_FAMILY_STATES:
        raise ValueError('Unknown rhythm family state')
    switch = np.zeros(GRID_LEVEL_COUNT, dtype=bool)
    switch[list(_BINARY_LEVELS if state == 'ternary' else _TERNARY_LEVELS)] = True
    switch[GRID_LEVEL_COUNT - 1] = True
    if entering_microstep:
        switch[3] = True
    return switch


def candidate_index(target: Fraction, candidates: list[Fraction]) -> int:
    if target in candidates:
        return candidates.index(target)
    return -1


_HARMONICS = np.asarray((2, 3, 4, 6, 8, 12, 16), dtype=np.float64)[None, :]


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
    previous_second: float | None = None,
    rich_time: bool = False,
):
    """Numeric time features; audio probes are attached by the dataset/model."""
    del audio_times
    if (beat_floats is None and not candidates) or (beat_floats is not None and len(beat_floats) == 0):
        return np.zeros((0, 5), dtype=np.float32)
    timeline = clock if clock is not None else TempoClock(tempo, first)
    previous_beat = Fraction(0) if previous is None else max(Fraction(0), previous if type(previous) is Fraction else Fraction(previous))
    if previous_second is None:
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
    angles = (2.0 * math.pi) * beat_floats
    out = np.empty((len(beat_floats), 21 if rich_time else 5), dtype=np.float32)
    out[:, 0] = delta_beats
    out[:, 1] = delta_sec
    out[:, 2] = np.log1p(delta_sec)
    out[:, 3] = np.sin(angles)
    out[:, 4] = np.cos(angles)
    if not rich_time:
        return out
    phases = angles[:, None] * _HARMONICS
    out[:, 5:12] = np.sin(phases)
    out[:, 12:19] = np.cos(phases)
    out[:, 19] = np.log1p(np.maximum(delta_beats, 0.))
    out[:, 20] = np.log1p(1. / np.maximum(delta_beats, 1e-6))
    return out


def interp_candidate_audio(values: np.ndarray, times: np.ndarray, seconds: np.ndarray,
                           *, suppress_attack_bridges: bool = False) -> np.ndarray:
    """Vectorized 2D linear interpolation matching np.interp(..., left=0., right=0.)."""
    values = np.asarray(values).view(np.ndarray)
    times = np.asarray(times, dtype=np.float64).view(np.ndarray)
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
    if suppress_attack_bridges and out.shape[1] >= 120 and len(times) > 1:
        # The bank clock contains sparse evidence nodes.  Keep mel/RMS/chroma
        # interpolation smooth, but attenuate only narrow attack-derived
        # columns when a query is far from every evidence node.
        nearest = np.minimum(
            np.abs(seconds - times[np.clip(np.searchsorted(times, seconds, side='left'), 0, n_times - 1)]),
            np.abs(seconds - times[np.clip(np.searchsorted(times, seconds, side='right') - 1, 0, n_times - 1)]),
        )
        confidence = np.clip(1.0 - nearest / 0.012, 0.0, 1.0).astype(np.float32)
        for column in (92, 95, 96, 97, 98, 99):
            out[:, column] *= confidence
    return out.astype(np.float32, copy=False)


def composer_audio_nodes(song, *, policy=COMPOSER_AUDIO_FEATURE_POLICY) -> tuple[np.ndarray, np.ndarray]:
    """Build the 120-dim composer feature bank directly from cached RationalAudio nodes."""
    from maichart.form.timing import normalize_meter_map
    audio = song.audio
    acoustic = np.asarray(audio.features, dtype=np.float32)
    times = np.asarray(audio.times, dtype=np.float64)
    if policy == COMPOSER_AUDIO_FEATURE_POLICY:
        from .acoustics import native_dynamics
        # Sampling is an input analysis grid, never an output-note clock.
        # 20ms is the existing Composer evidence resolution; native short
        # windows still preserve attack measurements without multiplying the
        # mapped bank to a 5ms full-song grid.
        times = np.unique(np.r_[times, np.arange(0., audio.duration_sec, .02)])
        acoustic = interp_candidate_audio(audio.features, audio.times, times)
        measured = native_dynamics(audio, times)
        acoustic[:, 92:95] = measured[:, :3]
        reference = np.asarray(audio.features[:, 93])
        median = max(float(np.median(reference[reference > 0])) if np.any(reference > 0) else 1., 1e-5)
        acoustic[:, 95] = acoustic[:, 93] / median
        acoustic[:, 96] = measured[:, 3]
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
