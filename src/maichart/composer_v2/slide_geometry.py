"""Auxiliary disk geometry of unambiguous authored MiaCode slide traces."""
from collections import Counter
from functools import lru_cache

import numpy as np
import torch
from torch.nn import functional as F

from maichart.generation.muri.data import reference_data
from .ordered_configuration import BEAT_INTERVALS


FIELDS = ('travel', 'radius_early', 'radius_middle', 'radius_late', 'central_travel',
          'clockwise_sweep', 'counterclockwise_sweep', 'endpoint_displacement')
HORIZONS = len(BEAT_INTERVALS)
# MiaCode's native slide endpoints lie on a radius-240 circle.
DISK_RADIUS = 240.


@lru_cache(maxsize=1024)
def path_features(key):
    """Summarize the actual directed polyline, sampled by travelled distance."""
    entry = reference_data()['slides'].get(key, {})
    points = entry.get('real_path_samples', ())
    if len(points) < 2:
        return None
    xy = np.asarray([(point['x'], point['y']) for point in points], dtype=np.float64) / DISK_RADIUS
    if not np.isfinite(xy).all():
        return None
    start, end = xy[:-1], xy[1:]
    steps = np.linalg.norm(end - start, axis=1)
    distance = np.r_[0., steps.cumsum()]
    length = distance[-1]
    if length <= 0:
        return None
    fractions = length * np.array([.25, .5, .75])
    landmarks = np.column_stack([np.interp(fractions, distance, axis) for axis in xy.T])
    radii = np.linalg.norm(landmarks, axis=1)
    mid_radius = np.linalg.norm((start + end) * .5, axis=1)
    central_travel = steps[mid_radius <= 1 / 3].sum() / length
    # Twice the swept area relative to each directed step. The native y axis
    # increases downwards, so clockwise sweeps have positive sign.
    sweep = start[:, 0] * end[:, 1] - start[:, 1] * end[:, 0]
    values = (length / (2 * np.pi), *radii, central_travel,
              sweep.clip(0).sum() / (2 * np.pi), -sweep.clip(max=0).sum() / (2 * np.pi),
              np.linalg.norm(xy[-1] - xy[0]) * .5)
    return tuple(np.clip(values, 0., 1.))


def _chart_rows(configuration, song):
    """Index trusted single paths and their weighted descriptor prefix sums."""
    sustain = song.sustain
    if sustain is None:
        return None
    slides = [m for m in sustain.markers if m['type'] in ('slide', 'wifi')
              and not m.get('is_mine') and not m.get('track_mine')]
    if not slides:
        return None
    counts = Counter(m['source_beat'] for m in slides)
    slides.sort(key=lambda marker: marker['slide_trace_second'])
    starts = np.asarray([m['slide_trace_second'] for m in slides], dtype=np.float64)
    stops = np.asarray([m['end_second'] for m in slides], dtype=np.float64)
    prior_end = np.r_[-np.inf, np.maximum.accumulate(stops[:-1])]
    next_start = np.r_[starts[1:], np.inf]
    # Consecutive traces may share a boundary. They cannot share elapsed time.
    isolated = ((starts >= prior_end - 1e-7) & (stops <= next_start + 1e-7)
                & np.isfinite(starts) & np.isfinite(stops) & (stops > starts))
    paths = []
    for index in np.flatnonzero(isolated):
        marker = slides[index]
        beat = marker['source_beat']
        keys = marker['slide_segment_keys']
        if marker['type'] == 'slide' and len(keys) == 1 and counts[beat] == 1:
            paths.append((beat, keys[0], stops[index]))
    if not paths:
        return None

    paths.sort()
    beats = np.asarray([beat for beat, _, _ in paths], dtype=np.float64)
    traces_end = np.asarray([stop for _, _, stop in paths], dtype=np.float64)
    source_seconds = song.clock.at_floats(beats)
    trust = configuration.quality.coverage_pairs(source_seconds, traces_end)[:, 1].clip(0., 1.)
    known = []
    for index in np.flatnonzero(trust > 0):
        descriptor = path_features(paths[index][1])
        if descriptor is not None:
            known.append((index, descriptor))
    if not known:
        return None
    indices, values = zip(*known)
    indices = list(indices)
    trust = trust[indices, None]
    rows = np.hstack((np.asarray(values) * trust, trust))
    totals = np.vstack((np.zeros(rows.shape[1]), np.cumsum(rows, axis=0)))
    return beats[indices], totals


def add_targets(item, configuration, song, kind_ids):
    """Write sparse loss-only rows in the same tile order as the audio planner."""
    active = (item['token_weights'].numpy() > 0) & np.isin(item['labels'].numpy(), kind_ids)
    tiles = np.unique(np.asarray(item['tile_ids'])[active])
    if not len(tiles):
        return
    mass = float(item['token_weights'].sum() * item['sample_weight'])
    if mass <= 0:
        return
    if not hasattr(configuration, '_slide_geometry_rows'):
        configuration._slide_geometry_rows = _chart_rows(configuration, song)
    rows = configuration._slide_geometry_rows
    if rows is None:
        return
    beats, totals = rows
    spans = (np.asarray(item['anchors'])[tiles, None, None] + np.asarray(BEAT_INTERVALS)).reshape(-1, 2)
    starts, stops = spans.astype(np.float64).T
    first = np.searchsorted(beats, starts)
    last = np.searchsorted(beats, stops)
    populated = np.flatnonzero(last > first)
    if not len(populated):
        return
    starts_s = song.clock.at_floats(starts[populated])
    stops_s = song.clock.at_floats(stops[populated])
    coverage = configuration.quality.coverage_pairs(starts_s, stops_s)[:, 1].clip(0., 1.)
    active = coverage > 0
    if not active.any():
        return
    selected = populated[active]
    count = last[selected] - first[selected]
    values = totals[last[selected]] - totals[first[selected]]
    trust = values[:, -1]
    weights = np.minimum(coverage[active], trust / count) * mass / (len(tiles) * HORIZONS)
    positions = tiles[selected // HORIZONS] * HORIZONS + selected % HORIZONS
    item['slide_geometry_positions'] = torch.from_numpy(positions)
    item['slide_geometry_targets'] = torch.from_numpy((values[:, :-1] / trust[:, None]).astype(np.float32))
    item['slide_geometry_weights'] = torch.from_numpy(weights.astype(np.float32))


def collate(items, out):
    positions, targets, weights = [], [], []
    offset = 0
    # Extra grid queries follow all base tiles. They carry planner inputs and
    # grid labels, while geometry here is supervised on authored kind queries.
    for item in items:
        if 'slide_geometry_positions' in item:
            positions.append(item['slide_geometry_positions'] + offset)
            targets.append(item['slide_geometry_targets'])
            weights.append(item['slide_geometry_weights'])
        offset += len(item['anchors']) * HORIZONS
    if positions:
        out['slide_geometry_positions'] = torch.cat(positions)
        out['slide_geometry_targets'] = torch.cat(targets)
        out['slide_geometry_weights'] = torch.cat(weights)


def loss(logits, batch, mass):
    prediction = logits.flatten(0, 1).index_select(0, batch['slide_geometry_positions']).float()
    errors = F.binary_cross_entropy_with_logits(prediction, batch['slide_geometry_targets'], reduction='none')
    return (errors.mean(-1) * batch['slide_geometry_weights']).sum() / mass
