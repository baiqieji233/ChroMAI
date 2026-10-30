"""Immutable audio on a fixed absolute-time grid for integrated survival."""
from collections import OrderedDict
import threading
import weakref

import numpy as np

SURVIVAL_GRID_SECONDS = .05
_CACHE_BUDGET = 128 * 1024 * 1024
_CACHE = OrderedDict()
_CACHE_BYTES = 0
_LOCK = threading.RLock()


def _audio_grid(context, duration):
    global _CACHE_BYTES
    key = (weakref.ref(context), float(duration))
    with _LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            _CACHE.move_to_end(key)
            return cached
        # Include a possibly shorter final cell, never an extra zero-width cell.
        count = max(1, int(np.ceil(duration / SURVIVAL_GRID_SECONDS)))
        edges = np.minimum(np.arange(count + 1, dtype=np.float64) * SURVIVAL_GRID_SECONDS, duration)
        audio = np.ascontiguousarray(context.at((edges[:-1] + edges[1:]) * .5), dtype=np.float32)
        edges.setflags(write=False)
        audio.setflags(write=False)
        size = edges.nbytes + audio.nbytes
        while _CACHE and _CACHE_BYTES + size > _CACHE_BUDGET:
            _, (old_audio, old_edges) = _CACHE.popitem(last=False)
            _CACHE_BYTES -= old_audio.nbytes + old_edges.nbytes
        if size <= _CACHE_BUDGET:
            _CACHE[key] = (audio, edges)
            _CACHE_BYTES += size
        return audio, edges


def survival_grid_inputs(context, absolute_start, absolute_candidate_seconds, duration,
                         plan_grid=None, views=None, point_seconds=None, prefix_seconds=None):
    """Return a window of the global grid; only authored callers supply labels.

    Counts use half-open cells [left, right). The start can fall inside the
    first cell: the model integrates that cell only after the exact cursor.
    """
    duration = float(duration)
    start = float(absolute_start)
    candidates = np.asarray(absolute_candidate_seconds, dtype=np.float64).reshape(-1)
    if not np.isfinite(duration) or duration <= 0 or not np.isfinite(start) or not np.isfinite(candidates).all():
        raise ValueError('Survival grid requires finite clocks and positive duration')
    audio, edges = _audio_grid(context, duration)
    start = float(np.clip(start, 0., duration))
    stop = float(np.clip(candidates.max() if candidates.size else start, start, duration))
    window_start = max(0., start - 4.) if prefix_seconds is not None else start
    first = min(len(audio) - 1, int(np.floor(window_start / SURVIVAL_GRID_SECONDS)))
    last = min(len(audio), max(first + 1, int(np.ceil(stop / SURVIVAL_GRID_SECONDS))))
    window_edges = edges[first:last + 1].copy()
    midpoints = (window_edges[:-1] + window_edges[1:]) * .5
    count = len(midpoints)
    indices, mix = (plan_grid.locate(midpoints) if plan_grid is not None else
                    (np.zeros((count, 2), dtype=np.int64), np.zeros(count, dtype=np.float32)))
    targets = np.zeros(count, dtype=np.float32)
    weights = np.zeros(count, dtype=np.float32)
    if views is not None and point_seconds is not None:
        points = np.asarray(point_seconds, dtype=np.float64)
        targets = (np.searchsorted(points, window_edges[1:], side='left') -
                   np.searchsorted(points, window_edges[:-1], side='left')).astype(np.float32)
        # EventViews has inclusive right endpoints; these cells are half-open.
        weights = np.asarray(views.window_weight(window_edges[:-1],
                            np.nextafter(window_edges[1:], window_edges[:-1])), dtype=np.float32)
    result = {
        'survival_audio': audio[first:last].copy(),
        'survival_edges': window_edges,
        'survival_candidate_seconds': candidates.copy(),
        'survival_start_seconds': np.asarray(start, dtype=np.float64),
        'survival_plan_indices': np.asarray(indices, dtype=np.int64),
        'survival_plan_mix': np.asarray(mix, dtype=np.float32),
        'survival_count_target': targets,
        'survival_count_weight': weights,
        'survival_widths': np.diff(window_edges).astype(np.float32),
    }
    if prefix_seconds is not None:
        prefix = np.asarray(prefix_seconds, dtype=np.float64).reshape(-1)
        if not np.isfinite(prefix).all():
            raise ValueError('Survival prefix requires finite event clocks')
        # Only committed history is an input. Authored future counts above
        # remain labels and never participate in this causal rolling count.
        prefix = np.sort(prefix[(prefix >= 0.) & (prefix <= start)])
        end = np.searchsorted(prefix, midpoints, side='left')
        result['survival_prefix_counts'] = np.column_stack([
            end - np.searchsorted(prefix, midpoints - width, side='left')
            for width in (1., 4.)]).astype(np.float32)
    return result
