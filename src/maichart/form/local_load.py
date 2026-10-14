"""Causal local chord-pressure guidance for song-form decoding."""
from __future__ import annotations

from collections import deque
from fractions import Fraction
import heapq
import math

import numpy as np

from maichart.event_timing import seconds_at_beat
from maichart.notes import parse_timing_point_notes
from maichart.timing import RawTimingPoint

HEAD_TYPES = ('tap', 'slide', 'hold', 'touch', 'touch_hold')
WINDOW_SECONDS = 2.0
TARGET_PSEUDO_EVENTS = 4.0
CHORD_HEADROOM = 0.2
CURVE_FACTOR_RANGE = (0.5, 1.5)
MAX_LOCAL_PENALTY = 4.0


def validate_strength(value, name='local_chord_strength'):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 3:
        raise ValueError(f'{name} must be finite and between 0 and 3')
    return value


def _counts_by_type(values):
    counts = np.asarray(values, dtype=np.float64)
    if counts.shape != (len(HEAD_TYPES),) or not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError('note counts must contain five finite nonnegative type counts')
    return counts


class LocalChordEvent:
    def __init__(self, pressure, strength):
        self.pressure = float(pressure)
        self.strength = float(strength)

    def penalty(self, button_heads):
        heads = np.asarray(button_heads, dtype=np.float64)
        if not np.isfinite(heads).all() or (heads < 0).any():
            raise ValueError('button_heads must contain finite nonnegative counts')
        head_scale = np.maximum(0.0, heads - 1.0)
        penalties = np.minimum(MAX_LOCAL_PENALTY, self.strength * self.pressure * head_scale)
        return float(penalties) if penalties.ndim == 0 else penalties


class LocalChordGuard:
    """Penalize excess physical chords using only the recent committed events.

    Counts use ``(tap, slide, hold, touch, touch_hold)`` order. Touch and
    TouchHold heads are excluded from physical chord detection. Cue seconds map
    musical beats onto the song timeline; ``curve_factor`` is a relative load
    multiplier, not an absolute events-per-second estimate.
    """

    def __init__(self, cues, target_chord_ratio=0.2, strength=1.0,
                 window_seconds=WINDOW_SECONDS):
        self.strength = validate_strength(strength)
        self.target_chord_share = float(target_chord_ratio)
        if not math.isfinite(self.target_chord_share) or not 0 <= self.target_chord_share <= 1:
            raise ValueError('target_chord_ratio must be finite and between 0 and 1')
        self.window_seconds = float(window_seconds)
        if not math.isfinite(self.window_seconds) or self.window_seconds <= 0:
            raise ValueError('window_seconds must be finite and positive')
        self.cues = {}
        for cue in cues:
            beat = Fraction(cue['beat'])
            second = float(cue['second'])
            curve_factor = float(cue.get('curve_factor', 1.0))
            if not math.isfinite(second) or not math.isfinite(curve_factor):
                raise ValueError('cue second and curve_factor must be finite')
            self.cues[beat] = (second, curve_factor)
        self._history = deque()
        self._recent_events = 0
        self._recent_chord_events = 0
        self._last_commit_second = None
        self.events = 0
        self.chord_events = 0
        self.button_heads = 0
        self.max_button_heads_per_event = 0
        self.peak_local_chord_share = 0.0
        self.pressure_events = 0
        self.current_pressure_run = 0
        self.max_pressure_run = 0
        self._last_event_second = None
        self.diagnostics = deque(maxlen=128)

    def _cue(self, beat):
        try:
            return self.cues[Fraction(beat)]
        except KeyError as exc:
            raise ValueError(f'No local chord cue for beat {beat}') from exc

    def _prune(self, second):
        cutoff = second - self.window_seconds
        while self._history and self._history[0][0] < cutoff:
            _, chorded, _ = self._history.popleft()
            self._recent_events -= 1
            self._recent_chord_events -= chorded

    def _local_share(self):
        posterior = (self._recent_chord_events + TARGET_PSEUDO_EVENTS * self.target_chord_share) / (
            self._recent_events + TARGET_PSEUDO_EVENTS)
        return posterior, self._recent_events, self._recent_chord_events

    def event(self, beat):
        second, curve_factor = self._cue(beat)
        if (self._last_event_second is not None
                and second - self._last_event_second > self.window_seconds):
            self.current_pressure_run = 0
        self._prune(second)
        observed_share, _, _ = self._local_share()
        curve_factor = min(CURVE_FACTOR_RANGE[1], max(CURVE_FACTOR_RANGE[0], curve_factor))
        local_ceiling = min(1.0, self.target_chord_share * curve_factor + CHORD_HEADROOM)
        pressure = max(0.0, observed_share - local_ceiling) * 4.0
        self.peak_local_chord_share = max(self.peak_local_chord_share, observed_share)
        self.pressure_events += int(pressure > 0)
        self.current_pressure_run = self.current_pressure_run + 1 if pressure > 0 else 0
        self.max_pressure_run = max(self.max_pressure_run, self.current_pressure_run)
        self._last_event_second = second
        self.diagnostics.append({
            'beat': str(Fraction(beat)), 'second': float(second),
            'observed_local_chord_share': float(observed_share),
            'local_chord_ceiling': float(local_ceiling),
            'pressure': float(pressure),
        })
        return LocalChordEvent(pressure, self.strength)

    def commit(self, counts_by_type, beat):
        counts = _counts_by_type(counts_by_type)
        second, _ = self._cue(beat)
        if self._last_commit_second is not None and second < self._last_commit_second:
            raise ValueError('local chord events must be committed in nondecreasing time order')
        self._prune(second)
        button_heads = int(counts[:3].sum())
        chorded = button_heads >= 2
        self._history.append((second, int(chorded), button_heads))
        self._recent_events += 1
        self._recent_chord_events += int(chorded)
        self._last_commit_second = second
        self.events += 1
        self.chord_events += int(chorded)
        self.button_heads += button_heads
        self.max_button_heads_per_event = max(self.max_button_heads_per_event, button_heads)

    def report(self):
        latest = self._last_commit_second
        if latest is not None:
            self._prune(latest)
        observed_share, recent_events, recent_chords = self._local_share()
        self.peak_local_chord_share = max(self.peak_local_chord_share, observed_share)
        return {
            'schema': 'song_form_local_chord_guidance_v1',
            'applied': self.strength > 0,
            'enabled': self.strength > 0,
            'strength': self.strength,
            'target_chord_share': self.target_chord_share,
            'window_seconds': self.window_seconds,
            'curve_factor_range': list(CURVE_FACTOR_RANGE),
            'curve_factor_semantics': 'relative_multiplier',
            'events': self.events,
            'chord_events': self.chord_events,
            'button_heads': self.button_heads,
            'recent_events': recent_events,
            'recent_chord_events': recent_chords,
            'observed_local_chord_share': float(observed_share),
            'peak_observed_local_chord_share': float(self.peak_local_chord_share),
            'max_button_heads_per_event': self.max_button_heads_per_event,
            'history_events': len(self._history),
            'pressure_events': self.pressure_events,
            'max_pressure_run': self.max_pressure_run,
            'diagnostics': list(self.diagnostics),
            'penalty_bound': MAX_LOCAL_PENALTY,
            'feedback_scope': 'causal_rolling_window',
            'touch_heads_excluded': True,
        }


def local_load_statistics(points, tempo, first, target_chord_ratio=None, window_seconds=WINDOW_SECONDS):
    """Measure local head load and physical chord bursts without changing quality controls.

    Peak rates use one-second windows ending at occupied event times, matching
    the existing half-open peak convention. Burst intervals use two-second
    windows starting at occupied event times and require at least eight events.
    Touch and TouchHold count toward all-head rate, but never toward physical
    button-chord share.
    """
    window_seconds = float(window_seconds)
    if not math.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError('window_seconds must be finite and positive')
    if target_chord_ratio is not None:
        target_chord_ratio = float(target_chord_ratio)
        if not math.isfinite(target_chord_ratio) or not 0 <= target_chord_ratio <= 1:
            raise ValueError('target_chord_ratio must be finite and between 0 and 1')

    grouped = {}
    previous_beat = None
    for point in points:
        beat = Fraction(point.beat)
        if previous_beat is not None and beat < previous_beat:
            raise ValueError('points must be ordered by nondecreasing beat')
        previous_beat = beat
        raw = str(point.note_text)
        raw_point = RawTimingPoint(index=0, raw=raw, note_text=raw, beat=beat, tick=0,
                                   time_sec=float(seconds_at_beat(beat, tempo, first)),
                                   division=4, bpm=120.0)
        parsed = parse_timing_point_notes(raw_point)
        if parsed.unknown_tokens:
            continue
        entry = grouped.setdefault(beat, [raw_point.time_sec, 0, 0])
        entry[1] += len(parsed.notes)
        entry[2] += sum(note.note_type in {'tap', 'slide', 'hold'} for note in parsed.notes)

    groups = [
        (values[0], values[1], values[2], int(values[2] >= 2))
        for values in grouped.values() if values[1] > 0
    ]
    if not groups:
        return {
            'schema': 'song_form_local_load_statistics_v1', 'enabled': True,
            'events': 0, 'all_heads': 0, 'button_heads': 0, 'physical_chords': 0,
            'peak_window_seconds': 1.0, 'peak_all_head_nps': 0.0,
            'peak_button_head_nps': 0.0, 'peak_occupied_events': 0,
            'peak_occupied_events_nps': 0.0, 'peak_window': None,
            'burst_window_seconds': window_seconds, 'minimum_burst_events': 8,
            'target_chord_ratio': target_chord_ratio,
            'burst_threshold': max(0.6, target_chord_ratio + 0.25) if target_chord_ratio is not None else 0.6,
            'burst_intervals': [], 'burst_interval_count': 0,
            'burst_intervals_truncated': False, 'burst_interval_limit': 16,
            'burst_interval_selection': 'merge_overlapping_or_contiguous_windows',
            'touch_heads_excluded_from_button_chords': True,
        }

    times = np.asarray([item[0] for item in groups], dtype=np.float64)
    all_heads = np.asarray([item[1] for item in groups], dtype=np.int64)
    button_heads = np.asarray([item[2] for item in groups], dtype=np.int64)
    chord_events = np.asarray([item[3] for item in groups], dtype=np.int64)
    event_count = len(groups)

    # Prefix totals keep all three peak calculations linear in chart length.
    all_prefix = np.concatenate(([0], np.cumsum(all_heads)))
    button_prefix = np.concatenate(([0], np.cumsum(button_heads)))
    chord_prefix = np.concatenate(([0], np.cumsum(chord_events)))
    peak_start, peak_end = 0.0, 0.0
    peak_all, peak_buttons, peak_events = 0, 0, 0
    left = 0
    for right, end_second in enumerate(times):
        while end_second - times[left] >= 1.0 - 1e-9:
            left += 1
        window_events = right - left + 1
        window_all = int(all_prefix[right + 1] - all_prefix[left])
        window_buttons = int(button_prefix[right + 1] - button_prefix[left])
        if window_all > peak_all:
            peak_all = window_all
            peak_start, peak_end = float(end_second - 1.0), float(end_second)
        peak_buttons = max(peak_buttons, window_buttons)
        peak_events = max(peak_events, window_events)

    burst_threshold = max(0.6, target_chord_ratio + 0.25) if target_chord_ratio is not None else 0.6
    right = 0
    component_end = None
    component_best = None
    top_components = []
    burst_interval_count = 0

    def save_component(best):
        nonlocal burst_interval_count
        if best is None:
            return
        burst_interval_count += 1
        # Keep only the strongest 16 merged episodes; heap size is constant.
        rank = (best['button_chord_share'], best['events'], best['button_heads'], -best['start_sec'])
        item = (rank, best)
        if len(top_components) < 16:
            heapq.heappush(top_components, item)
        elif rank > top_components[0][0]:
            heapq.heapreplace(top_components, item)

    for left, start_second in enumerate(times):
        if right < left:
            right = left
        limit = start_second + window_seconds
        while right < event_count and times[right] < limit:
            right += 1
        count = right - left
        if count < 8:
            continue
        chords = int(chord_prefix[right] - chord_prefix[left])
        share = chords / count
        if share <= burst_threshold:
            continue
        total_all = int(all_prefix[right] - all_prefix[left])
        total_buttons = int(button_prefix[right] - button_prefix[left])
        candidate = {
            'start_sec': float(start_second), 'end_sec': float(start_second + window_seconds),
            'window_seconds': window_seconds, 'events': count, 'all_heads': total_all,
            'button_heads': total_buttons, 'chords': chords,
            'button_chord_share': float(share), 'all_head_nps': total_all / window_seconds,
            'button_head_nps': total_buttons / window_seconds,
            'occupied_event_nps': count / window_seconds,
        }
        # Adjacent qualifying windows describe one burst episode. Keep its strongest
        # fixed-width window so reported intervals remain bounded to window_seconds.
        if component_end is None:
            component_end = candidate['end_sec']
            component_best = candidate
        elif candidate['start_sec'] <= component_end + 1e-9:
            component_end = max(component_end, candidate['end_sec'])
            if (candidate['button_chord_share'], candidate['events'], candidate['button_heads']) > (
                    component_best['button_chord_share'], component_best['events'], component_best['button_heads']):
                component_best = candidate
        else:
            save_component(component_best)
            component_end = candidate['end_sec']
            component_best = candidate
    save_component(component_best)
    burst_intervals = [item[1] for item in sorted(top_components, key=lambda item: item[0], reverse=True)]

    return {
        'schema': 'song_form_local_load_statistics_v1', 'enabled': True,
        'events': event_count, 'all_heads': int(all_heads.sum()),
        'button_heads': int(button_heads.sum()), 'physical_chords': int(chord_events.sum()),
        'peak_window_seconds': 1.0, 'peak_all_head_nps': float(peak_all),
        'peak_button_head_nps': float(peak_buttons),
        'peak_occupied_events': int(peak_events),
        'peak_occupied_events_nps': float(peak_events),
        'peak_window': {'start_sec': peak_start, 'end_sec': peak_end},
        'burst_window_seconds': window_seconds, 'minimum_burst_events': 8,
        'target_chord_ratio': target_chord_ratio, 'burst_threshold': burst_threshold,
        'burst_intervals': burst_intervals, 'burst_interval_count': burst_interval_count,
        'burst_intervals_truncated': burst_interval_count > 16, 'burst_interval_limit': 16,
        'burst_interval_selection': 'merge_overlapping_or_contiguous_windows',
        'touch_heads_excluded_from_button_chords': True,
    }
