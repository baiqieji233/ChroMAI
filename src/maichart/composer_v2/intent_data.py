"""Bounded, same-chart demonstrations for the phrase intent experiment."""
from bisect import bisect_left, bisect_right
from collections import OrderedDict
from copy import copy
from fractions import Fraction
from functools import lru_cache
import math

import numpy as np
import torch

from maichart.composer.event_set_codec import _atoms
from .codec import KINDS, atom_components
from .data import _beats_at_seconds
from .intent_policy import POLICY
from .motion_writing import CONTACT, MOVEMENT, PATHS, contact
from .ordered_configuration import ConfigurationQuality
from .slide_geometry import path_features


PHRASE_BEATS = 8
MIN_SUPPORT_BEATS = 2
STEPS = 2 * PHRASE_BEATS
MAX_PHRASES = 5
AUDIO_DIM = 24
FIELDS = (
    'head_density', 'chord', 'head_cardinality', 'mixed_rim_touch',
    *(f'pulse_{i}' for i in range(4)),
    'native_phase_16_only', 'native_phase_24_only',
    'gap_beats', 'gap_seconds',
    *(f'{edge}_{kind}' for edge in ('first', 'last') for kind in KINDS),
    *(f'chord_separation_{i}' for i in range(1, 5)),
    *(f'rim_step_{i}' for i in range(8)),
    *(f'{edge}_rim_step_{i}' for edge in ('first', 'last') for i in range(8)),
    *(f'slide_{path}' for path in PATHS), 'slide_clockwise_sweep', 'slide_counterclockwise_sweep',
    *(f'{kind}_{part}' for kind in ('hold', 'touch_hold', 'slide')
      for part in ('occupancy', 'release', 'seconds')),
    'slide_waiting', 'slide_trace_start', 'slide_wait_seconds',
)
TARGET_DIM = len(FIELDS)
_INDEX = {field: i for i, field in enumerate(FIELDS)}
_CANDIDATE_LIMIT = 16
_FUTURE_CANDIDATE_LIMIT = 8
FUTURE_VIEW_PERIOD = 4
FUTURE_MIN_GAP_BEATS = 2
_CHART_CACHE_SIZE = 8
_PHRASE_CACHE_SIZE = 128
_QUARTER_OFFSETS = (np.arange(4, dtype=np.float64) + .5) / 4


@lru_cache(maxsize=8192)
def _event(text):
    kinds, heads = set(), set()
    for atom in _atoms(text):
        components = atom_components(atom)
        kind = components[0][2:]
        kinds.add(kind)
        pads = [part[2:] for part in components if part.startswith('P:')]
        heads.update(pads if kind == 'tap' else pads[:1])
    paths, _ = contact(text)
    path_start = len(CONTACT) + len(MOVEMENT)
    return (tuple(KINDS.index(kind) for kind in kinds), tuple(sorted(heads)),
            tuple(field - path_start for field in paths if field >= path_start))


def empty_item():
    return {
        'intent_phrase_bounds': torch.zeros(MAX_PHRASES, 2, dtype=torch.float32),
        'intent_phrase_audio': torch.zeros(MAX_PHRASES, STEPS, AUDIO_DIM, dtype=torch.float32),
        'intent_phrase_targets': torch.zeros(MAX_PHRASES, STEPS, TARGET_DIM, dtype=torch.float32),
        'intent_phrase_weights': torch.zeros(MAX_PHRASES, STEPS, TARGET_DIM, dtype=torch.float32),
        'intent_phrase_valid': torch.zeros(MAX_PHRASES, STEPS, dtype=torch.bool),
        'intent_support_mask': torch.zeros(MAX_PHRASES, dtype=torch.bool),
        'intent_future_support_mask': torch.zeros(MAX_PHRASES, dtype=torch.bool),
        'intent_query_mask': torch.zeros(MAX_PHRASES, dtype=torch.bool),
    }


def collate(items, batch):
    if any('intent_phrase_bounds' in item for item in items):
        default = empty_item()
        rows = [item if 'intent_phrase_bounds' in item else default for item in items]
        batch.update({key: torch.stack([row[key] for row in rows]) for key in default})
        query = batch['intent_query_mask']
        support = batch['intent_support_mask']
        future = batch['intent_future_support_mask']
        if query[:, 1:].any() or support[:, 0].any():
            raise ValueError('Phrase intent query and support must occupy separate reserved slots')
        if (future & ~support).any():
            raise ValueError('Phrase intent future support requires a valid support slot')
        start, stop = batch['intent_phrase_bounds'].unbind(-1)
        overlap = (start[:, 1:] < stop[:, :1]) & (stop[:, 1:] > start[:, :1])
        if (overlap & support[:, 1:] & query[:, :1]).any():
            raise ValueError('Phrase intent support must be outside the query interval')


def _audio_rows(song, starts, stops):
    # Mel-only reads avoid importing a full MERT frame for each short sample.
    centers = starts[:, None] + (stops - starts)[:, None] * _QUARTER_OFFSETS
    seconds = song.clock.at_floats(centers.ravel())
    ticks = np.column_stack((np.maximum(0., seconds - .02), seconds)).ravel()
    audio = song.audio
    if hasattr(audio, 'mel_times_sec'):
        from .audio import _interpolate
        mel = _interpolate(audio.mel, audio.mel_times_sec, ticks)
    else:
        mel = audio.sample(ticks)[0]
    bands = mel[:, :96].reshape(len(seconds), 2, 12, 8).mean(-1)
    power = bands[:, 1].reshape(-1, 4, 12).mean(1)
    flux = np.maximum(0., bands[:, 1] - bands[:, 0]).reshape(-1, 4, 12).mean(1)
    return np.column_stack((power, flux)).astype(np.float32)


def _heads(song, begin, end, start, starts, target, weights):
    old_beat = None
    old_rim = None
    counts = np.zeros(len(starts), dtype=np.int32)
    gaps = np.zeros(len(starts), dtype=np.bool_)
    ring_steps = np.zeros(len(starts), dtype=np.bool_)
    for point in song.points[begin:end]:
        row = int((point.beat - start) * 2)
        values = target[row]
        kinds, heads, paths = _event(point.note_text)
        rim = [int(head) for head in heads if head.isdigit()]
        pulse = min(3, int(((point.beat - start) * 2 % 1) * 4))
        values[_INDEX[f'pulse_{pulse}']] = 1.
        on_16 = (point.beat * 4).denominator == 1
        on_24 = (point.beat * 6).denominator == 1
        if on_16 != on_24:
            values[_INDEX['native_phase_16_only' if on_16 else 'native_phase_24_only']] = 1.
        values[_INDEX['head_cardinality']] = max(values[_INDEX['head_cardinality']], min(1., len(heads) / 4))
        values[_INDEX['chord']] = max(values[_INDEX['chord']], len(heads) > 1)
        values[_INDEX['mixed_rim_touch']] = max(values[_INDEX['mixed_rim_touch']], 0 < len(rim) < len(heads))
        for i, left in enumerate(rim):
            for right in rim[i + 1:]:
                separation = min(abs(right - left), 8 - abs(right - left))
                values[_INDEX[f'chord_separation_{separation}']] = 1.
        if counts[row] == 0:
            values[list(_INDEX['first_tap'] + kind for kind in kinds)] = 1.
        values[_INDEX['last_tap']:_INDEX['last_touch_hold'] + 1] = 0.
        values[list(_INDEX['last_tap'] + kind for kind in kinds)] = 1.
        counts[row] += 1
        values[list(_INDEX['slide_straight'] + path for path in paths)] = 1.
        if old_beat is not None:
            gaps[row] = True
            gap = min(1., float(point.beat - old_beat) / 4)
            sec_gap = min(1., (song.clock.at(point.beat) - song.clock.at(old_beat)) / 2)
            values[_INDEX['gap_beats']] = max(values[_INDEX['gap_beats']], gap)
            values[_INDEX['gap_seconds']] = max(values[_INDEX['gap_seconds']], sec_gap)
            if old_rim is not None and len(heads) == len(rim) == 1:
                direction = (rim[0] - old_rim) % 8
                values[_INDEX['rim_step_0'] + direction] = 1.
                if not ring_steps[row]:
                    values[_INDEX['first_rim_step_0'] + direction] = 1.
                    ring_steps[row] = True
                values[_INDEX['last_rim_step_0']:_INDEX['last_rim_step_7'] + 1] = 0.
                values[_INDEX['last_rim_step_0'] + direction] = 1.
        old_rim = rim[0] if len(heads) == len(rim) == 1 else None
        old_beat = point.beat
    target[:, _INDEX['head_density']] = np.minimum(1., counts / 4)
    # The first head, a chord and a headless step need not establish a gap or
    # rim transition within the reviewed phrase.
    weights[~gaps, _INDEX['gap_beats']:_INDEX['gap_seconds'] + 1] = 0.
    weights[~ring_steps, _INDEX['rim_step_0']:_INDEX['last_rim_step_7'] + 1] = 0.


def _sustains(markers, song, starts, stops, target, weights):
    steps = len(starts)
    for marker in markers:
        source = song.clock.at(Fraction(marker['source_beat']))
        release = float(marker['end_second'])
        kind = 'slide' if marker['type'] in ('slide', 'wifi') else marker['type']
        trace = float(marker['slide_trace_second']) if kind == 'slide' else source
        occupancy = np.maximum(0., np.minimum(stops, release) - np.maximum(starts, trace)) / (stops - starts)
        column = _INDEX[f'{kind}_occupancy']
        target[:, column] = np.maximum(target[:, column], occupancy)
        observed_release = release < stops[-1]
        if observed_release:
            target[(starts <= release) & (stops > release), _INDEX[f'{kind}_release']] = 1.
        onset = min(steps - 1, max(0, int(np.searchsorted(starts, source, side='right')) - 1))
        column = _INDEX[f'{kind}_seconds']
        if observed_release:
            target[onset, column] = max(target[onset, column], np.clip((release - trace) / 4, 0., 1.))
        else:
            weights[onset, column] = 0.
        if kind == 'slide':
            geometry = path_features(marker['slide_track_key'])
            sweep = slice(_INDEX['slide_clockwise_sweep'], _INDEX['slide_counterclockwise_sweep'] + 1)
            if geometry is not None:
                target[onset, sweep] = np.maximum(target[onset, sweep], geometry[5:7])
            else:
                weights[onset, sweep] = 0.
            waiting = np.maximum(0., np.minimum(stops, trace) - np.maximum(starts, source)) / (stops - starts)
            column = _INDEX['slide_waiting']
            target[:, column] = np.maximum(target[:, column], waiting)
            target[(starts <= trace) & (stops > trace), _INDEX['slide_trace_start']] = 1.
            column = _INDEX['slide_wait_seconds']
            if trace < stops[-1]:
                target[onset, column] = max(target[onset, column], np.clip((trace - source) / 2, 0., 1.))
            else:
                weights[onset, column] = 0.


def _mask_incoming_sustains(markers, starts, stops, weights):
    """Absent local heads cannot imply that an external sustain has released."""
    for marker in markers:
        release = float(marker['end_second'])
        kind = 'slide' if marker['type'] in ('slide', 'wifi') else marker['type']
        trace = float(marker['slide_trace_second']) if kind == 'slide' else starts[0]
        weights[(stops > trace) & (starts < release), _INDEX[f'{kind}_occupancy']] = 0.
        weights[(starts <= release) & (stops > release), _INDEX[f'{kind}_release']] = 0.
        if kind == 'slide':
            weights[starts < trace, _INDEX['slide_waiting']] = 0.
            weights[(starts <= trace) & (stops > trace), _INDEX['slide_trace_start']] = 0.


class _Chart:
    def __init__(self, song):
        self.song = song
        if getattr(song, '_legacy_zero_unmarked_resume', False):
            song = copy(song)
            song._legacy_zero_unmarked_resume = False
        self.quality = ConfigurationQuality(song)
        chart_end = getattr(song, 'authored_end_beat', None)
        chart_end = song.end_beat if chart_end is None else min(chart_end, song.end_beat)
        audio_ends = _beats_at_seconds(song.clock, [0., song.audio.duration_sec])
        self.audio_start = max(Fraction(0), Fraction(float(audio_ends[0])))
        self.end = min(chart_end, Fraction(float(audio_ends[1])))
        self.markers = sorted((mark for mark in (song.sustain.markers if song.sustain is not None else ())
                               if not mark.get('is_mine') and not mark.get('track_mine')),
                              key=lambda mark: mark['source_beat'])
        self.marker_beats = [mark['source_beat'] for mark in self.markers]
        self.latest_release = np.maximum.accumulate([float(mark['end_second']) for mark in self.markers])
        self.supports = OrderedDict()
        self.approved = self._approved_supports()
        self.support_starts = [left for left, _, _ in self.approved]
        self.support_ends = [right for _, right, _ in self.approved]

    def incoming_sustain(self, beat):
        index = bisect_left(self.marker_beats, beat) - 1
        return index >= 0 and self.latest_release[index] > self.song.clock.at(beat) + 1e-8

    def covered(self, start, stop):
        if stop <= start or start < self.audio_start:
            return 0.
        a, b = self.song.clock.at_floats([float(start), float(stop)])
        # The inverse audio endpoint can round by one unit in the last place.
        if abs(b - self.quality.duration) < 1e-10:
            b = self.quality.duration
        return self.quality.covered(max(0., a), b)

    def _approved_supports(self):
        usable = self.quality.weights[:-1].min(1) > 0
        changes = np.flatnonzero(np.diff(np.r_[False, usable, False]))
        beats = _beats_at_seconds(self.song.clock, self.quality.boundaries[changes])
        result = []
        for a, b in zip(beats[::2], beats[1::2]):
            # Round into the reviewed interval. Outer windows may be shorter
            # than eight beats, with right padding represented by validity.
            left = max(self.audio_start, Fraction(math.ceil(a * 2), 2))
            right = min(self.end, Fraction(math.floor(b * 2), 2))
            if right - left < MIN_SUPPORT_BEATS:
                continue
            internal = range((left // PHRASE_BEATS + 1) * PHRASE_BEATS, math.ceil(right), PHRASE_BEATS)
            bounds = [left, *internal, right]
            chunks = [(start, stop) for start, stop in zip(bounds, bounds[1:])
                      if stop - start >= MIN_SUPPORT_BEATS]
            for start, stop in chunks or [(left, right)]:
                trust = self.covered(start, stop)
                if trust > 0:
                    result.append((start, stop, trust))
        return result

    def description(self, start, stop, begin, end, trust):
        beats = self.song.point_beats
        starts = np.asarray([float(start + Fraction(i, 2)) for i in range(STEPS)])
        rows = int(np.count_nonzero(starts < float(stop)))
        starts = starts[:rows]
        stops = np.minimum(starts + .5, float(stop))
        a, b = self.song.clock.at_floats(starts), self.song.clock.at_floats(stops)
        target = np.zeros((rows, TARGET_DIM), dtype=np.float32)
        weights = np.full_like(target, trust) * np.float32(2) * (stops - starts)[:, None].astype(np.float32)
        _heads(self.song, begin, end, start, starts, target, weights)
        lo = bisect_left(self.marker_beats, beats[begin]) if begin < len(beats) else len(self.markers)
        hi = bisect_right(self.marker_beats, beats[end - 1], lo=lo) if begin < end else lo
        _sustains(self.markers[lo:hi], self.song, a, b, target, weights)
        if lo and self.latest_release[lo - 1] >= a[0]:
            last_source = self.marker_beats[lo - 1]
            incoming = (mark for mark in self.song.sustain._candidate_markers(a[0])
                        if mark['source_beat'] <= last_source and float(mark['end_second']) >= a[0]
                        and not mark.get('is_mine') and not mark.get('track_mine'))
            _mask_incoming_sustains(incoming, a, b, weights)
        audio = _audio_rows(self.song, starts, stops)
        return audio, target, weights

    def support(self, start, stop, trust):
        if start not in self.supports:
            beats = self.song.point_beats
            begin, end = bisect_left(beats, start), bisect_left(beats, stop)
            self.supports[start] = self.description(start, stop, begin, end, trust)
            if len(self.supports) > _PHRASE_CACHE_SIZE:
                self.supports.popitem(last=False)
        self.supports.move_to_end(start)
        return self.supports[start]


def _copy(out, row, bounds, values):
    out['intent_phrase_bounds'][row] = torch.tensor(tuple(map(float, bounds)), dtype=torch.float32)
    for name, value in zip(('audio', 'targets', 'weights'), values):
        out[f'intent_phrase_{name}'][row, :len(value)] = torch.from_numpy(value)
    out['intent_phrase_valid'][row, :len(values[0])] = True


def _similarity(query, source):
    length = min(len(query), len(source))
    left, right = query[:length].ravel(), source[:length].ravel()
    # A music-only proposal shortlist. Labels never choose the demonstration.
    return float(left @ right / max(float(np.linalg.norm(left) * np.linalg.norm(right)), 1e-8))


def _candidate(chart, query, left, right, trust, blocked_beats, blocked_spans):
    if any(left <= beat < right for beat in blocked_beats):
        return None
    if any(left < stop and start < right for start, stop in blocked_spans):
        return None
    values = chart.support(left, right, trust)
    return _similarity(query[0], values[0]), left, right, values


def _shortlist(chart, query, first, stop, budget, blocked_beats, blocked_spans, *, future=False):
    count = stop - first
    if count <= 0:
        return []
    anchors = np.linspace(first, stop - 1, min(budget, count), dtype=np.int64)
    candidates = []
    for anchor in anchors:
        left, right, trust = chart.approved[anchor]
        if future and chart.incoming_sustain(left):
            continue
        candidate = _candidate(chart, query, left, right, trust, blocked_beats, blocked_spans)
        if candidate is not None:
            candidates.append(candidate)
    return sorted(candidates, key=lambda row: (row[0], row[1]), reverse=True)


def add_item(item, dataset, si, song, begin, end, *, query_cursor=None, support_before=None,
             allow_future=False):
    """Add a held-out query and separate, teacher-only authored references."""
    blocked_beats, blocked_spans = item.pop('_intent_blocked', ((), ()))
    out = empty_item()
    item.update(out)
    quality = song.quality or {}
    if begin == end or quality.get('label') in ('excluded', 'negative') or float(item['sample_weight']) <= 0:
        return
    cache = dataset._intent_chart_cache
    if si not in cache:
        cache[si] = _Chart(song)
        if len(cache) > _CHART_CACHE_SIZE:
            cache.popitem(last=False)
    cache.move_to_end(si)
    chart = cache[si]
    cursor = item['cursor'] if query_cursor is None else query_cursor
    start = max(cursor, chart.audio_start)
    limit = song.point_beats[end] if end < len(song.points) else chart.end
    stop = min(start + PHRASE_BEATS, chart.end, limit)
    trust = chart.covered(start, stop)
    if trust <= 0:
        return
    first = bisect_left(song.point_beats, start, lo=begin, hi=end)
    last = bisect_left(song.point_beats, stop, lo=first, hi=end)
    query = chart.description(start, stop, first, last, trust)
    _copy(out, 0, (start, stop), query)
    out['intent_query_mask'][0] = True
    cutoff = min(cursor - dataset.history_beats, chart.end)
    if support_before is not None:
        cutoff = min(cutoff, support_before)
    past = _shortlist(chart, query, 0, bisect_right(chart.support_ends, cutoff), _CANDIDATE_LIMIT,
                      blocked_beats, blocked_spans)
    future = []
    if allow_future:
        first = max(limit, stop, *(right for _, right in blocked_spans)) + FUTURE_MIN_GAP_BEATS
        first_block = bisect_left(chart.support_starts, first)
        future = _shortlist(chart, query, first_block, len(chart.approved),
                            _FUTURE_CANDIDATE_LIMIT, blocked_beats, blocked_spans, future=True)[:1]
    # Include a lower-similarity option when available so support cannot be
    # treated as a forced assignment to a single matching phrase.
    past_count = MAX_PHRASES - 1 - len(future)
    if len(past) > past_count:
        past = [*past[:past_count - 1], past[-1]]
    for row, (_, left, right, values) in enumerate([*past, *future], 1):
        _copy(out, row, (left, right), values)
        out['intent_support_mask'][row] = True
        out['intent_future_support_mask'][row] = row > len(past)
