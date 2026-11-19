"""Training targets at real event boundaries and sparse coverage between onsets."""
import math
from fractions import Fraction

import numpy as np
import torch

from . import motion_writing


GRID_FIELDS = ('head_rate', 'first_accent_phase')
DECISION_FIELDS = ('gap_half_beat', 'gap_one_beat', 'gap_two_beats', 'continue_motion',
                   'next_chord', 'next_hold', 'next_slide', 'next_touch')
HALF_BEAT = Fraction(1, 2)


def missing_beats(first, last, observed, offset=.5):
    """Select at most two representatives and count how many beats each represents."""
    boundaries = [first - 1, *sorted(b for b in observed if first <= b <= last), last + 1]
    gaps = np.maximum(0, np.diff(boundaries) - 1)
    ends = gaps.cumsum()
    total = int(ends[-1])
    count = min(2, total)
    if not count:
        return [], []
    edges = [i * total // count for i in range(count + 1)]
    weights = np.diff(edges)
    ranks = edges[:-1] + np.floor(weights * offset).astype(np.int64)
    indices = np.searchsorted(ends, ranks, side='right')
    beats = np.asarray(boundaries)[indices] + 1 + ranks - (ends - gaps)[indices]
    return beats.tolist(), weights.tolist()


def grid_targets(item, configuration, song, dataset, targets, weights, tiles, spans, offset):
    from .local_writing import BEAT_INTERVALS, HORIZONS, _motion_intervals

    anchors = item['anchors']
    observed = [anchors[i] for i in tiles]
    last = math.floor(item.get('writing_stop_beat', max(observed, default=item['cursor'])))
    extra, counts = [], []
    if observed and dataset is not None:
        extra, counts = missing_beats(math.ceil(item['cursor']), last, observed, offset)
    if extra:
        item['writing_extra'] = dataset.writing_features(song, extra)
    extra_spans = [(beat + left, right - left) for beat in extra for left, right in BEAT_INTERVALS]
    spans = [*spans, *extra_spans]
    size = len(anchors) + len(extra)
    result = np.zeros((size, HORIZONS, targets.shape[-1] + len(GRID_FIELDS)), dtype=np.float32)
    trust = np.zeros_like(result)
    result[:len(anchors), :, :-2] = targets
    trust[:len(anchors), :, :-2] = weights
    if not spans:
        return result, trust

    chart = configuration
    if extra_spans:
        cache = _motion_intervals(chart, song, extra_spans)
        extra_targets, extra_weights = zip(*(cache[span] for span in extra_spans))
        dims = (-1, HORIZONS, targets.shape[-1])
        result[len(anchors):, :, :-2] = np.reshape(extra_targets, dims)
        trust[len(anchors):, :, :-2] = np.reshape(extra_weights, dims)
    if not hasattr(chart, '_accent_beats'):
        contact = chart._ordered_motion[0][:, :len(motion_writing.CONTACT)]
        # All contact fields except plain Tap mark a timed articulation.
        marked = np.diff(contact[:, 1:].sum(1)) > 0
        chart._accent_beats = chart._motion_beats[marked]
    rows = np.asarray([*tiles, *range(len(anchors), size)], dtype=int)
    starts, widths = np.asarray(spans, dtype=float).T
    stops = starts + widths
    heads = np.searchsorted(chart._motion_beats, stops) - np.searchsorted(chart._motion_beats, starts)
    coverage = chart.quality.coverage_pairs(song.clock.at_floats(starts), song.clock.at_floats(stops))
    result[rows, :, -2] = (heads / (heads + 4 * widths)).reshape(-1, HORIZONS)
    trust[rows, :, -2] = coverage[:, 0].reshape(-1, HORIZONS) * .05
    if len(chart._accent_beats):
        index = np.searchsorted(chart._accent_beats, starts)
        accent = chart._accent_beats[np.minimum(index, len(chart._accent_beats) - 1)]
        valid = (index < len(chart._accent_beats)) & (accent < stops)
        result[rows, :, -1] = np.where(valid, (accent - starts) / widths, 0.).reshape(-1, HORIZONS)
        trust[rows, :, -1] = (valid * coverage.min(1) * .05).reshape(-1, HORIZONS)
    trust[:, :, :-2] *= .9
    if counts:
        trust[len(anchors):] *= np.asarray(counts, dtype=np.float32)[:, None, None]
    # Use the same CE mass for gradient accumulation across unequal microbatches.
    mass = float(item['token_weights'].sum() * item['sample_weight'])
    trust *= .7 * mass / (max(1, len(tiles) + sum(counts)) * HORIZONS)
    return result, trust


def add_decisions(item, configuration, song, codec, begin, end, *, layout=True):
    """Predict the successor strictly after the committed event, with independent quality masks."""
    ids = item['input_ids'].numpy()
    start = int(np.flatnonzero(ids == codec.tgt_id)[0])
    completed = start + 1 + np.flatnonzero(ids[start + 1:] == codec.event_end_id)
    boundaries = np.r_[start, completed][:end - begin]
    if not len(boundaries):
        return
    rows = np.arange(begin, begin + len(boundaries))
    beats = configuration.beats[begin:begin + len(rows)]
    prior = [item['cursor'], *beats[:-1]]
    a = song.clock.at_floats(np.asarray(prior, dtype=np.float64))
    b = song.clock.at_floats(np.asarray(beats, dtype=np.float64))
    coverage = configuration.quality.coverage_pairs(a, np.nextafter(b, np.inf))
    values = np.zeros((len(rows), len(DECISION_FIELDS)), dtype=np.float32)
    weights = np.zeros_like(values)
    gaps = [beat - previous for beat, previous in zip(beats, prior)]
    values[:, :3] = [(gap >= HALF_BEAT, gap >= 1, gap >= 2) for gap in gaps]
    weights[:, :3] = coverage[:, :1] / 6
    if layout:
        for j, index in enumerate(rows):
            fields, key = motion_writing.contact(song.points[index].note_text)
            names = [motion_writing.EVENT_FIELDS[field] for field in fields]
            values[j, 4:] = ('chord' in names, 'hold' in names, 'slide' in names,
                             'touch' in names or 'touch_hold' in names)
            weights[j, 4:] = coverage[j, 1] / 12
            if index < 3 or configuration.event_weights[index - 3:index, 1].min() <= 0:
                continue
            x, y, z = [motion_writing.contact(p.note_text)[1] for p in song.points[index - 3:index]]
            if x and y and z and (y - x) % 8 == (z - y) % 8:
                values[j, 3] = bool(key and (key - z) % 8 == (z - y) % 8)
                weights[j, 3] = coverage[j, 1] / 6

    kind_queries = (start + 2 + np.flatnonzero(ids[start + 1:] == codec.phase_id))[:len(rows)]
    token_weights = item['token_weights'].numpy()
    timing_active = token_weights[boundaries] > 0
    layout_active = token_weights[kind_queries] > 0
    weights[:, :3] *= timing_active[:, None]
    weights[:, 3:] *= layout_active[:, None]
    supervised = timing_active | layout_active
    if 'layout_replay' in item and supervised.any():
        # The two retained keys determine the outgoing step. The third preceding
        # key used to establish its earlier direction may have been replaced.
        weights[np.flatnonzero(supervised)[0], 3] = 0.
    active = supervised & (weights.max(1) > 0)
    if not active.any():
        return
    mass = float(item['sample_weight'] * item['token_weights'].sum())
    weights *= .3 * mass / max(1, int(supervised.sum()))
    # The first digit query has seen no part of the next time. The first kind
    # query has seen the chosen time but none of the next action.
    frontier = boundaries[active]
    positions = np.concatenate((frontier, frontier + 1, kind_queries[active]))
    both = weights[active] * .5
    timing = both.copy()
    timing[:, 3:] = 0
    action = both.copy()
    action[:, :3] = 0
    matrix = np.concatenate((both, timing, action))
    keep = matrix.max(1) > 0
    item['decision_positions'] = torch.from_numpy(positions[keep])
    item['decision_targets'] = torch.from_numpy(np.tile(values[active], (3, 1))[keep])
    item['decision_weights'] = torch.from_numpy(matrix[keep])


def collate(items, batch):
    rows = [(index, item) for index, item in enumerate(items) if 'decision_positions' in item]
    if rows:
        positions = [torch.stack((torch.full_like(item['decision_positions'], index),
                                  item['decision_positions']), dim=1) for index, item in rows]
        batch['decision_positions'] = torch.cat(positions)
        for name in ('decision_targets', 'decision_weights'):
            batch[name] = torch.cat([item[name] for _, item in rows])
