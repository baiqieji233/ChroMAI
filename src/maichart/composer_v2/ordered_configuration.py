"""Sparse supervision over native future events and successive beat intervals."""
from bisect import bisect_left
from functools import lru_cache

import numpy as np
import torch

from maichart.composer.event_set_codec import _atoms
from .codec import KINDS, atom_components
from .defaults import FUTURE_CONFIGURATION_TRAIN_VIEW_FRACTION as TRAINING_PROBABILITY
from .quality_data import stream_region_weight_fn

POLICY = 'ordered_future_v1'
HORIZONS = ('next_event_2', 'next_event_4', 'beats_0_2', 'beats_2_4', 'beats_4_8')
BEAT_INTERVALS = ((0, 2), (2, 4), (4, 8))
POSITIONS = tuple(str(i) for i in range(1, 9)) + tuple(
    f'{zone}{i}' for zone in 'ABDE' for i in range(1, 9)) + ('C',)
SUSTAINS = ('hold', 'touch_hold', 'slide')
FEATURES = (
    'rest', *(f'kind_{name}' for name in KINDS), *(f'position_{name}' for name in POSITIONS),
    *(f'button_step_{i}' for i in range(5)),
    *(f'{name}_{feature}' for name in SUSTAINS for feature in ('occupied', 'release', 'duration')),
)
TARGET_DIM = len(FEATURES)
_POSITION_START = 1 + len(KINDS)
_STEP_START = _POSITION_START + len(POSITIONS)
_SUSTAIN_START = _STEP_START + 5
FEATURE_GROUPS = (
    slice(0, 1), slice(1, _POSITION_START), slice(_POSITION_START, _STEP_START),
    slice(_STEP_START, _SUSTAIN_START), slice(_SUSTAIN_START, TARGET_DIM),
)
MAX_QUERIES = 8


@lru_cache(maxsize=8192)
def _event_atoms(note_text):
    kinds, positions, buttons = set(), set(), set()
    for atom in _atoms(note_text):
        components = atom_components(atom)
        kind = components[0][2:]
        kinds.add(KINDS.index(kind))
        pads = [part[2:] for part in components if part.startswith('P:')]
        positions.update(POSITIONS.index(pad) for pad in pads)
        heads = pads if kind == 'tap' else pads[:1]
        buttons.update(int(pad) for pad in heads if pad.isdigit())
    return tuple(kinds), tuple(positions), tuple(buttons)


class ConfigurationQuality:
    """The same component weights as ordinary CE, indexed over exact audio time."""

    def __init__(self, song):
        quality = song.quality or {}
        baseline = float(quality.get('unmarked_weight', 1.))
        regions = quality.get('regions', ())
        self.duration = float(song.audio.duration_sec)
        self.boundaries = np.asarray(sorted({0., self.duration, *(
            float(region[key]) for region in regions for key in ('start_sec', 'end_sec')
            if 0 <= float(region[key]) <= self.duration)}))
        self.weights = np.full((len(self.boundaries), 2), baseline, dtype=np.float32)
        self.weights[-1] = 0
        rule = stream_region_weight_fn(baseline, getattr(song, '_legacy_zero_unmarked_resume', False))
        for region in regions:
            selected = ((self.boundaries >= float(region['start_sec']))
                        & (self.boundaries < min(self.duration, float(region['end_sec']))))
            self.weights[selected] = (0., 0.) if region.get('training_use') == 'none' else rule(region)

    def at(self, seconds):
        seconds = np.asarray(seconds)
        rows = np.searchsorted(self.boundaries, seconds, side='right') - 1
        return np.where(np.expand_dims(rows >= 0, -1), self.weights[np.maximum(rows, 0)], 0.)

    def covered(self, start, stop, channel=None):
        if start < 0 or stop > self.duration or start >= stop:
            return 0.
        first = np.searchsorted(self.boundaries, start, side='right') - 1
        last = np.searchsorted(self.boundaries, stop, side='left')
        covered = self.weights[first:last]
        return float((covered if channel is None else covered[:, channel]).min())

    def coverage_pairs(self, starts, stops):
        first = np.maximum(0,np.searchsorted(self.boundaries, starts, side='right') - 1)
        last = np.searchsorted(self.boundaries, stops, side='left')
        result = self.weights[first].copy()
        for i in np.flatnonzero(last > first+1):
            result[i] = self.weights[first[i]:last[i]].min(0)
        result[(starts < 0) | (stops > self.duration) | (starts >= stops)] = 0
        return result


class ChartConfiguration:
    """One compact native descriptor per authored onset, plus its sustain intervals."""

    def __init__(self, song):
        self.beats = song.point_beats
        self.clock = song.clock
        self.seconds = song.clock.at_floats(np.asarray(self.beats, dtype=np.float64))
        self.quality = ConfigurationQuality(song)
        self.event_weights = self.quality.at(self.seconds)
        self.events = np.zeros((len(self.beats), _SUSTAIN_START), dtype=np.bool_)
        previous = ()
        for index, point in enumerate(song.points):
            kinds, positions, buttons = _event_atoms(point.note_text)
            self.events[index, [1 + kind for kind in kinds]] = True
            self.events[index, [_POSITION_START + pad for pad in positions]] = True
            if index and self.event_weights[index - 1, 1] > 0:
                for before in previous:
                    for after in buttons:
                        distance = abs(after - before)
                        self.events[index, _STEP_START + min(distance, 8 - distance)] = True
            previous = buttons

        native = song.sustain.markers if song.sustain is not None else ()
        intervals = []
        for marker in native:
            kind = marker['type']
            kind = 'slide' if kind == 'wifi' else kind
            start = float(marker['slide_trace_second'] if kind == 'slide' else marker['second'])
            stop = float(marker['end_second'])
            if kind in SUSTAINS and stop > start:
                source = float(self.clock.at_floats(float(marker['source_beat'])))
                intervals.append((source, start, stop, SUSTAINS.index(kind)))
        intervals.sort()
        self.sustains = np.asarray(intervals, dtype=np.float64).reshape(-1, 4)
        self._latest_end = np.maximum.accumulate(self.sustains[:, 2])

    def _overlapping_sustains(self, start, stop):
        first = np.searchsorted(self._latest_end, start, side='left')
        last = np.searchsorted(self.sustains[:, 0], stop, side='left')
        return self.sustains[first:last]

    def has_sustain(self, start, stop):
        intervals = self._overlapping_sustains(start, stop)
        return bool(np.any((intervals[:, 1] < stop) & (intervals[:, 2] > start)))

    def sustain_features(self, target, weights, start, stop, *, duration_from=None):
        duration_from = start if duration_from is None else duration_from
        for source, onset, release, kind in self._overlapping_sustains(start, stop):
            if release < start or (onset >= stop and source < duration_from):
                continue
            field = _SUSTAIN_START + 3 * int(kind)
            if self.quality.at(source)[1] <= 0:
                weights[0] = 0
                weights[field:field + 3] = 0
                continue
            overlap = max(0., min(release, stop) - max(onset, start))
            if overlap:
                # The maximum coverage of one native action is bounded and
                # does not multiply the interval by simultaneous actions.
                target[field] = max(target[field], overlap / (stop - start))
                target[0] = 0
            target[field + 1] = max(target[field + 1], float(start <= release < stop))
            if duration_from <= source < stop:
                if self.quality.covered(source, release) > 0:
                    target[field + 2] = max(target[field + 2], min(1., (release - onset) / 8))
                else:
                    weights[field + 2] = 0

    def target_at(self, first, cursor):
        values = np.zeros((len(HORIZONS), TARGET_DIM), dtype=np.float32)
        weights = np.zeros_like(values)
        query_sec = float(self.clock.at_floats(float(cursor)))
        for row, offset in enumerate((1, 3)):
            index = first + offset
            if index >= len(self.beats):
                continue
            end_sec = np.nextafter(self.seconds[index], np.inf)
            trust = self.quality.covered(query_sec, end_sec)
            if trust <= 0:
                continue
            values[row, :_SUSTAIN_START] = self.events[index]
            weights[row] = trust
            start_sec = self.seconds[index - 1]
            self.sustain_features(values[row], weights[row], start_sec, end_sec,
                                  duration_from=self.seconds[index])
            weights[row, 0] = 0  # event-count targets have an onset by definition

        for row, (left, right) in enumerate(BEAT_INTERVALS, start=2):
            start_sec, end_sec = self.clock.at_floats(np.asarray([float(cursor + left), float(cursor + right)]))
            trust = self.quality.covered(start_sec, end_sec)
            if trust <= 0:
                continue
            a = bisect_left(self.beats, cursor + left, lo=first)
            b = bisect_left(self.beats, cursor + right, lo=a)
            events = self.events[a:b]
            weights[row] = trust
            if len(events):
                values[row, :_SUSTAIN_START] = events.mean(0)
                # A step into the first event may originate before this interval.
                if a and self.event_weights[a - 1, 1] <= 0:
                    weights[row, _STEP_START:_SUSTAIN_START] = 0
            else:
                values[row, 0] = 1
            self.sustain_features(values[row], weights[row], start_sec, end_sec)
        return values, weights


def add_item(item, configuration, codec, begin, end):
    mass = float(item['token_weights'].sum() * item['sample_weight'])
    if begin == end or mass <= 0:
        return
    ids = item['input_ids'].numpy()
    start = np.flatnonzero(ids == codec.tgt_id)[0]
    boundaries = np.concatenate(([start], np.flatnonzero(ids[start + 1:] == codec.event_end_id) + start + 1))
    count = end - begin
    ordinals = np.linspace(0, count - 1, min(count, MAX_QUERIES), dtype=np.int64)
    positions, targets, weights = [], [], []
    for ordinal in ordinals:
        first = begin + int(ordinal)
        cursor = configuration.beats[first - 1] if first else item['cursor']
        target, weight = configuration.target_at(first, cursor)
        if weight.any():
            positions.append(boundaries[ordinal])
            targets.append(target)
            weights.append(weight)
    if positions:
        weights = np.stack(weights)
        # Equalize feature families before distributing this window's CE mass.
        # Each available horizon contributes, including legitimately blank spans.
        available = np.clip(weights, 0., 1.)
        counts = np.full((1, len(HORIZONS), 1), len(FEATURE_GROUPS), dtype=np.float32)
        counts[:, :2] -= 1  # rest is undefined for a direct event descriptor
        for group in FEATURE_GROUPS:
            divisor = (group.stop - group.start) * counts
            available[..., group] /= divisor
            weights[..., group] /= divisor
        coverage = available.sum() / (len(ordinals) * len(HORIZONS))
        weights *= mass * coverage / weights.sum()
        item['future_configuration_positions'] = torch.as_tensor(np.asarray(positions))
        item['future_configuration_targets'] = torch.from_numpy(np.stack(targets))
        item['future_configuration_weights'] = torch.from_numpy(weights)


def collate(items, result):
    positions, targets, weights = [], [], []
    for index, item in enumerate(items):
        position = item.get('future_configuration_positions')
        if position is None:
            continue
        positions.append(torch.stack((torch.full_like(position, index), position), dim=1))
        targets.append(item['future_configuration_targets'])
        weights.append(item['future_configuration_weights'])
    if positions:
        result['future_configuration_positions'] = torch.cat(positions)
        result['future_configuration_targets'] = torch.cat(targets)
        result['future_configuration_weights'] = torch.cat(weights)
