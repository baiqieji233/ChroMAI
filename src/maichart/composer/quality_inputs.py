"""Small causal configuration inputs shared by training and generation.

Audio is immutable and may describe the whole song. Symbolic inputs always
describe committed events only; future authored notes appear only in labels.
"""
from __future__ import annotations

from fractions import Fraction
from collections import OrderedDict
import math
import threading

import numpy as np
import torch

from maichart.generation.muri.model import CENTERS, PADS, RELEASE
from .candidates import interp_candidate_audio
from .events import event_metadata

QUALITY_INPUT_POLICY = 'native_lifecycle_paired_candidate_clock_phrase_v2'
SONG_QUALITY_INPUT_POLICY = 'exact_history_audio_plan_clock_song_paired_v3'


def quality_input_policy_for_policy(policy):
    return SONG_QUALITY_INPUT_POLICY if policy == 'song_paired_v3' else QUALITY_INPUT_POLICY

QUALITY_FORWARD_KEYS = ('quality_lifecycle', 'quality_anchor_audio', 'quality_anchor_ids',
                        'quality_anchor_ages', 'quality_anchor_mask', 'quality_phrase_audio',
                        'quality_phrase_indices', 'quality_candidate_offsets')
SONG_QUALITY_FORWARD_KEYS = ('quality_history_audio', 'quality_history_clock',
    'quality_history_plan_indices', 'quality_history_plan_mix',
    'quality_phrase_plan_indices', 'quality_phrase_plan_mix', 'quality_plan_clock')
QUALITY_FORWARD_KEYS = QUALITY_FORWARD_KEYS + SONG_QUALITY_FORWARD_KEYS
QUALITY_FORWARD_KEYS += ('quality_future_audio', 'quality_future_mask', 'quality_future_offsets')


def quality_forward_inputs(batch):
    keys = (tuple(k for k in QUALITY_FORWARD_KEYS if not k.startswith('quality_anchor_'))
            if 'quality_history_audio' in batch else tuple(k for k in QUALITY_FORWARD_KEYS if k not in SONG_QUALITY_FORWARD_KEYS))
    return {key: batch[key] for key in keys if key in batch}


QUALITY_LIFECYCLE_DIM = 88
QUALITY_ANCHORS = 8
QUALITY_PHRASE_BEATS = 4.
QUALITY_FUTURE_OFFSETS = (.25, .5, .75, 1., 1.5, 2., 3., 4., 6., 8.)
QUALITY_GAP_UPPER_BEATS = np.asarray((.125, .25, .5, 1., 2., 4.), dtype=np.float64)
_PAD_INDEX = {pad: i for i, pad in enumerate(PADS)}
_EPS = 1e-7


def _pad_sector_weights():
    # D/E sensors lie halfway between button sectors. Do not conflate their
    # numeric suffix with the button of the same number.
    result = {}
    for pad, (x, y) in CENTERS.items():
        row = np.zeros(8, dtype=np.float32)
        if pad != 'C':
            coordinate = ((math.atan2(y, x) + math.radians(67.5)) / math.radians(45.)) % 8.
            left = int(math.floor(coordinate))
            mix = coordinate - left
            row[left % 8] = 1. - mix
            row[(left + 1) % 8] += mix
        result[pad] = row
    return result


_PAD_SECTOR_WEIGHTS = _pad_sector_weights()
_PAD_SECTOR_MATRIX = np.stack([_PAD_SECTOR_WEIGHTS[pad] for pad in PADS])


def _lane(marker, name='lane'):
    value = marker.get(name)
    return int(value) - 1 if value is not None and str(value).isdigit() and 1 <= int(value) <= 8 else None


def lifecycle_features(state_index, candidate_seconds, *, visible_before_beat=None):
    """Aggregate every visible sustain using native wait/segment/end clocks.

    Layout: slide onset heads, waiting heads, tracing path sectors, tracing
    endpoints, landing endpoints, active button Holds (6 * 8); active Touch
    Holds (33); six stage counts and tracing center occupancy (7). Counts use
    log1p rather than truncating to an arbitrary number of active actions.
    Native slide path sampling includes segmented slides and Wifi branches.
    """
    seconds = np.asarray(candidate_seconds, dtype=np.float64)
    output = np.zeros((len(seconds), QUALITY_LIFECYCLE_DIM), dtype=np.float32)
    if not len(seconds) or state_index is None:
        return output
    cutoff = None if visible_before_beat is None else Fraction(visible_before_beat)
    lower, upper = float(seconds.min()), float(seconds.max())
    # Reuse the native one-second interval index. A sample queries only its
    # candidate buckets, not every sustain in the full authored song.
    if hasattr(state_index, '_candidate_markers'):
        selected = {}
        for bucket in np.unique(np.floor(seconds)):
            for marker in state_index._candidate_markers(float(bucket)):
                selected[id(marker)] = marker
        markers = selected.values()
    else:
        markers = state_index.markers
    for marker in markers:
        # Missing source identity is not allowed to expose a future marker.
        if cutoff is not None and (marker.get('source_beat') is None
                                   or Fraction(marker['source_beat']) >= cutoff):
            continue
        start, end = float(marker['second']), float(marker['end_second'])
        if start > upper + _EPS or end + RELEASE < lower - _EPS:
            continue
        kind = marker['type']
        head = _lane(marker)
        active = (seconds >= start - _EPS) & (seconds < end - _EPS)
        if kind in {'slide', 'wifi'}:
            trace = float(marker['slide_trace_second'])
            onset = (seconds >= start - _EPS) & (seconds <= start + RELEASE + _EPS)
            waiting = active & (seconds < trace - _EPS)
            tracing = active & ~waiting
            landing = (seconds >= start - _EPS) & (np.abs(seconds - end) <= RELEASE + _EPS)
            if head is not None:
                output[onset, head] += 1.
                output[waiting, 8 + head] += 1.
            terminals = state_index._terminal_lanes(marker)
            for lane in terminals:
                output[tracing, 24 + lane - 1] += 1.
                output[landing, 32 + lane - 1] += 1.
            rows = np.flatnonzero(tracing)
            if len(rows) and hasattr(state_index, '_slide_pad_matrix'):
                pads = state_index._slide_pad_matrix(marker, seconds[rows])
                sectors = np.where(pads[:, :, None], _PAD_SECTOR_MATRIX[None, :, :], 0.)
                output[rows, 16:24] += sectors.max(axis=1)
                output[rows, 87] += pads[:, _PAD_INDEX['C']]
            else:
                for row in rows:
                    pads = state_index._slide_pads(marker, float(seconds[row]))
                    sector_rows = [_PAD_SECTOR_WEIGHTS[pad] for pad in pads if pad in _PAD_SECTOR_WEIGHTS]
                    if sector_rows:
                        output[row, 16:24] += np.maximum.reduce(sector_rows)
                    output[row, 87] += float('C' in pads)
            for column, selection in enumerate((onset, waiting, tracing, landing)):
                output[selection, 81 + column] += 1.
        elif kind == 'hold':
            if head is not None:
                output[active, 40 + head] += 1.
            output[active, 85] += 1.
        elif kind == 'touch_hold':
            pad = str(marker.get('touch_pad', ''))
            if pad.startswith('C'):
                pad = 'C'
            if pad in _PAD_INDEX:
                output[active, 48 + _PAD_INDEX[pad]] += 1.
            output[active, 86] += 1.
    np.log1p(output, out=output)
    return output


def select_history_anchors(beats):
    """Last four events plus four beat-spaced earlier anchors, without RNG."""
    beats = np.asarray(beats, dtype=np.float64)
    if len(beats) <= QUALITY_ANCHORS:
        return np.arange(len(beats), dtype=np.int64)
    selected = set(range(len(beats) - 4, len(beats)))
    for age in (2., 4., 8., 16.):
        candidate = int(np.searchsorted(beats, beats[-1] - age, side='right')) - 1
        if candidate >= 0:
            selected.add(candidate)
    # Dense histories or very short prefixes can collapse beat anchors onto
    # the same event. Fill deterministically from the available older range.
    for candidate in np.linspace(0, len(beats) - 5, QUALITY_ANCHORS, dtype=int)[::-1]:
        if len(selected) >= QUALITY_ANCHORS:
            break
        selected.add(int(candidate))
    return np.asarray(sorted(selected), dtype=np.int64)


def phrase_audio_bank(acoustic_context, clock, audio_duration):
    """A cached four-beat grid built exclusively from the existing audio bank."""
    identity = (tuple(clock.beats), tuple(clock.bpms), tuple(clock.seconds), float(audio_duration))
    cache = getattr(acoustic_context, '_quality_phrase_banks', None)
    if cache is None:
        cache = acoustic_context._quality_phrase_banks = {}
    if identity in cache:
        return cache[identity]
    # Invert the piecewise tempo clock; song end is an audio fact, not a label.
    segment = max(0, int(np.searchsorted(clock.seconds, audio_duration, side='right')) - 1)
    end_beat = float(clock.beats[segment]) + (float(audio_duration) - clock.seconds[segment]) * clock.bpms[segment] / 60.
    count = max(1, int(math.ceil(max(0., end_beat) / QUALITY_PHRASE_BEATS)))
    edges = np.arange(count + 1, dtype=np.float64) * QUALITY_PHRASE_BEATS
    edge_seconds = clock.at_floats(edges)
    centers = np.clip(clock.at_floats((edges[:-1] + edges[1:]) * .5), 0., max(0., audio_duration - 1e-6))
    bank = {'audio': torch.from_numpy(acoustic_context.at(centers)), 'edges': edges,
            'seconds': edge_seconds}
    cache[identity] = bank
    return bank


def _anchor_audio(acoustic_context, values, times, seconds):
    """Bounded CPU cache of exact event-time audio rows, shared across samples."""
    cache_state = getattr(acoustic_context, '_quality_anchor_cache', None)
    if cache_state is None:
        cache_state = (OrderedDict(), threading.RLock())
        acoustic_context._quality_anchor_cache = cache_state
    cache, lock = cache_state
    keys = [float(second) for second in seconds]
    with lock:
        missing = list(dict.fromkeys(key for key in keys if key not in cache))
        if missing:
            rows = interp_candidate_audio(values, times, np.asarray(missing))[:, :120]
            for key, row in zip(missing, rows):
                cache[key] = row
        result = np.stack([cache[key] for key in keys])
        for key in keys:
            cache.move_to_end(key)
        while len(cache) > 1024:
            cache.popitem(last=False)
    return result


def exact_history_audio(acoustic_context, seconds):
    """Bounded immutable CPU summaries at the actual committed event seconds."""
    seconds = np.asarray(seconds, dtype=np.float64)
    if not len(seconds):
        return np.zeros((0, 104), dtype=np.float32)
    state = getattr(acoustic_context, '_quality_history_cache', None)
    if state is None:
        state = acoustic_context._quality_history_cache = (OrderedDict(), threading.RLock())
    cache, lock = state
    keys = [float(value) for value in seconds]
    with lock:
        missing = list(dict.fromkeys(key for key in keys if key not in cache))
        if missing:
            for key, row in zip(missing, acoustic_context.at(np.asarray(missing))):
                cache[key] = row.copy()
        result = np.stack([cache[key] for key in keys])
        for key in keys:
            cache.move_to_end(key)
        while len(cache) > 8192:
            cache.popitem(last=False)
    return result


def fixed_future_audio_inputs(acoustic_context, clock, candidate_beats, audio_duration):
    """Query actual audio at fixed beat offsets, independently of candidate keys.

    True masks denote valid probes. The shared immutable event-time cache
    deduplicates exact seconds within and across samples; no learned values
    or authored future events enter this bank.
    """
    offsets = np.asarray(QUALITY_FUTURE_OFFSETS, dtype=np.float64)
    beats = np.asarray(candidate_beats, dtype=np.float64)
    seconds = clock.at_floats((beats[:, None] + offsets[None, :]).reshape(-1)).reshape(len(beats), len(offsets))
    valid = np.isfinite(seconds) & (seconds >= 0.) & (seconds <= float(audio_duration))
    audio = np.zeros((*seconds.shape, 104), dtype=np.float32)
    if valid.any():
        audio[valid] = exact_history_audio(acoustic_context, seconds[valid])
    return {'quality_future_audio': torch.from_numpy(audio),
            'quality_future_mask': torch.from_numpy(valid),
            'quality_future_offsets': torch.tensor(QUALITY_FUTURE_OFFSETS, dtype=torch.float32)}


def shared_raw_plan_bank(acoustic_context, clock, centers):
    """Share immutable audio/clock tensors across charts of one actual source.

    The owning AcousticContext establishes audio identity. Exact grid and tempo
    identity prevent merging different offsets or BPM maps, even if shapes match.
    Chart-authored targets and all learned states are deliberately excluded.
    """
    seconds = np.ascontiguousarray(centers, dtype=np.float64)
    key = (tuple(clock.beats), tuple(clock.bpms), tuple(clock.seconds), seconds.shape, seconds.tobytes())
    cache, lock = acoustic_context.__dict__.setdefault('_quality_raw_plan_cache', (OrderedDict(), threading.RLock()))
    with lock:
        bank = cache.get(key)
        if bank is None:
            audio = acoustic_context.at(seconds)
            segment = np.clip(np.searchsorted(clock.seconds, seconds, side='right') - 1, 0, len(clock.seconds)-1)
            beats = np.asarray(clock.beats, dtype=np.float64)[segment] + (seconds-np.asarray(clock.seconds)[segment])*np.asarray(clock.bpms)[segment]/60.
            bank = dict(audio=audio, audio_tensor=torch.from_numpy(audio),
                        mask=torch.zeros(len(seconds), dtype=torch.bool),
                        clock=torch.from_numpy(np.column_stack((beats, seconds)).astype(np.float32)))
            cache[key] = bank
        cache.move_to_end(key)
        while len(cache) > 4:
            cache.popitem(last=False)
        return bank


def plan_clock_features(plan_grid, clock):
    cached = getattr(plan_grid, '_quality_plan_clock', None)
    if cached is None:
        seconds = np.asarray(plan_grid.centers, dtype=np.float64)
        segment = np.clip(np.searchsorted(clock.seconds, seconds, side='right') - 1, 0, len(clock.seconds)-1)
        beats = np.asarray(clock.beats, dtype=np.float64)[segment] + (seconds-np.asarray(clock.seconds)[segment])*np.asarray(clock.bpms)[segment]/60.
        cached = plan_grid._quality_plan_clock = torch.from_numpy(np.column_stack((beats, seconds)).astype(np.float32))
    return cached


def quality_context_inputs(*, points, atom_ids, candidate_beats, candidate_seconds,
                           state_index, audio_features, audio_times, acoustic_context,
                           clock, audio_duration, visible_before_beat=None,
                           point_beats=None, point_seconds=None, policy=None, plan_grid=None, history_audio=None, history_plan_mapping=None, include_future=False):
    """Build one unbatched row; ``points`` and ``atom_ids`` are committed only.

    ``state_index`` may contain an authored future only when the explicit
    ``visible_before_beat`` cutoff is provided (the training target's beat).
    Generation/recovery pass their IncrementalState over the actual prefix.
    """
    beats = np.asarray([float(p.beat) for p in points] if point_beats is None else point_beats, dtype=np.float64)
    seconds = clock.at_floats(beats) if point_seconds is None else np.asarray(point_seconds, dtype=np.float64)
    ids = np.asarray(atom_ids, dtype=np.int64)
    if ids.ndim != 2 or len(ids) != len(beats) or len(seconds) != len(beats):
        raise ValueError('Quality audio anchors must pair the same committed events and atom rows')
    future = fixed_future_audio_inputs(acoustic_context, clock, candidate_beats, audio_duration) if include_future else {}
    if policy == 'song_paired_v3':
        if plan_grid is None:
            raise ValueError('song_paired_v3 requires the existing PlanGrid')
        phrase = phrase_audio_bank(acoustic_context, clock, audio_duration)
        candidates = np.asarray(candidate_beats, dtype=np.float64)
        if '_song_plan_mapping' not in phrase or phrase['_song_plan_mapping'][0] is not plan_grid:
            phrase_beats = (phrase['edges'][:-1] + phrase['edges'][1:]) * .5
            phrase_seconds = np.clip(clock.at_floats(phrase_beats), 0., max(0., audio_duration-1e-6))
            pi, pm = plan_grid.locate(phrase_seconds)
            phrase['_song_plan_mapping'] = (plan_grid, torch.from_numpy(pi), torch.from_numpy(pm))
        hi, hm = plan_grid.locate(seconds) if history_plan_mapping is None else history_plan_mapping
        _, pi, pm = phrase['_song_plan_mapping']
        audio = exact_history_audio(acoustic_context, seconds) if history_audio is None else np.asarray(history_audio, dtype=np.float32)
        if audio.shape != (len(beats), 104):
            raise ValueError('History audio must align with the exact symbolic history tail')
        return {
            **future,
            'quality_lifecycle': torch.from_numpy(lifecycle_features(state_index, candidate_seconds, visible_before_beat=visible_before_beat)),
            'quality_history_audio': torch.from_numpy(audio),
            'quality_history_clock': torch.from_numpy(np.column_stack((beats, seconds)).astype(np.float32)),
            'quality_history_plan_indices': torch.from_numpy(hi),
            'quality_history_plan_mix': torch.from_numpy(hm),
            'quality_phrase_plan_indices': pi,
            'quality_phrase_plan_mix': pm,
            'quality_plan_clock': plan_clock_features(plan_grid, clock),
            'quality_phrase_audio': phrase['audio'],
            'quality_phrase_indices': torch.from_numpy(np.clip(np.floor(candidates/QUALITY_PHRASE_BEATS).astype(np.int64), 0, len(phrase['audio'])-1)),
            'quality_candidate_offsets': torch.from_numpy((candidates-(beats[-1] if len(beats) else 0.)).astype(np.float32)),
        }
    selected = select_history_anchors(beats)
    anchor_audio = np.zeros((QUALITY_ANCHORS, 120), dtype=np.float32)
    anchor_ids = np.zeros((QUALITY_ANCHORS, ids.shape[1]), dtype=np.int64)
    ages = np.zeros(QUALITY_ANCHORS, dtype=np.float32)
    valid = np.zeros(QUALITY_ANCHORS, dtype=bool)
    if len(selected):
        count = len(selected)
        anchor_audio[:count] = _anchor_audio(acoustic_context, audio_features, audio_times, seconds[selected])
        anchor_ids[:count] = ids[selected]
        ages[:count] = beats[-1] - beats[selected]
        valid[:count] = True
    phrase = phrase_audio_bank(acoustic_context, clock, audio_duration)
    candidates = np.asarray(candidate_beats, dtype=np.float64)
    # This is an unclipped beat distance with an explicit anchor clock,
    # independent of candidate-feature packing. Empty prefixes use beat zero.
    candidate_offsets = (candidates - (beats[-1] if len(beats) else 0.)).astype(np.float32)
    phrase_indices = np.clip(np.floor(candidates / QUALITY_PHRASE_BEATS).astype(np.int64),
                             0, len(phrase['audio']) - 1)
    return {
        **future,
        'quality_lifecycle': torch.from_numpy(lifecycle_features(state_index, candidate_seconds,
            visible_before_beat=visible_before_beat)),
        'quality_anchor_audio': torch.from_numpy(anchor_audio),
        'quality_anchor_ids': torch.from_numpy(anchor_ids),
        'quality_anchor_ages': torch.from_numpy(ages),
        'quality_anchor_mask': torch.from_numpy(valid),
        'quality_phrase_audio': phrase['audio'],
        'quality_phrase_indices': torch.from_numpy(phrase_indices),
        'quality_candidate_offsets': torch.from_numpy(candidate_offsets),
    }


def quality_targets(points, clock, phrase_bank, views, *, policy=None):
    """Labels only: descriptive four-beat windows and true next-event gaps."""
    beats = np.asarray([float(p.beat) for p in points], dtype=np.float64)
    targets = np.zeros((len(phrase_bank['audio']), 9 if policy == 'song_paired_v3' else 8), dtype=np.float32)
    edges = phrase_bank['edges']
    seconds = phrase_bank['seconds']
    # Phrase labels cover [lo, hi); EventViews windows have a closed right
    # edge, so exclude the next phrase and permit an exact audio-end boundary.
    weights = np.asarray(views.window_weight(seconds[:-1], np.nextafter(seconds[1:], -np.inf)), dtype=np.float32)
    # A partially truncated first/last phrase is not a complete label window.
    weights[(seconds[:-1] < 0.) | (seconds[1:] > views.end_sec)] = 0.
    metadata = [event_metadata(str(p.note_text)) for p in points]
    for index, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        begin, stop = np.searchsorted(beats, (lo, hi), side='left')
        events = metadata[begin:stop]
        if policy == 'song_paired_v3':
            from maichart.training.music_plan_targets import fine_note_ratio
            targets[index, 8] = fine_note_ratio([p.beat for p in points[begin:stop]])
        count = len(events)
        targets[index, 7] = .5
        if not count:
            continue
        heads = np.asarray([len(meta.heads) for meta in events], dtype=np.float32)
        targets[index, :6] = (min(1., count / QUALITY_PHRASE_BEATS / 8.),
            min(1., float(heads.mean()) / 4.), np.mean([m.chord for m in events]),
            np.mean([m.hold for m in events]), np.mean([m.slide for m in events]),
            np.mean([m.touch for m in events]))
        lanes = np.zeros(8, dtype=np.float64)
        for meta in events:
            for head in meta.heads:
                if str(head) in '12345678' and len(str(head)) == 1:
                    lanes[int(head) - 1] += 1.
        if lanes.sum():
            probability = lanes[lanes > 0] / lanes.sum()
            targets[index, 6] = -float(np.sum(probability * np.log(probability))) / math.log(8.)
        first = float(heads[beats[begin:stop] < (lo + hi) * .5].sum())
        second = float(heads.sum()) - first
        targets[index, 7] = second / max(first + second, 1.)
    gap = np.zeros(len(points), dtype=np.int64)
    gap_weight = np.zeros(len(points), dtype=np.float32)
    if len(points) > 1:
        gap[:-1] = np.searchsorted(QUALITY_GAP_UPPER_BEATS, np.diff(beats), side='left')
        event_seconds = clock.at_floats(beats)
        gap_weight[:-1] = views.window_weight(event_seconds[:-1], event_seconds[1:])
        gap_weight[:-1] *= np.asarray(views.trusted[:-1]) & np.asarray(views.trusted[1:])
    return {'phrase_target': torch.from_numpy(targets), 'phrase_weight': torch.from_numpy(weights),
            'gap_target': torch.from_numpy(gap), 'gap_weight': torch.from_numpy(gap_weight)}
