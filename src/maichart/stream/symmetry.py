"""Cabinet geometry for chart_stream: button lanes, hand kinematics, mirrors.

Only the four transforms offered by the game's own mirror option are used
(identity, left-right, up-down, 180-degree rotation). They preserve the
cabinet's left/right hand split up to a swap, so a mirrored authored chart is
exactly as playable as the original. Arbitrary 45/90-degree rotations are
deliberately excluded: they move patterns across the player's body axis.
"""
from __future__ import annotations

from functools import lru_cache
import math

import numpy as np
import torch

TRANSFORMS = ('identity', 'mirror_lr', 'mirror_ud', 'rotate_180')
KINEMATIC_DIM = 65
LANES = 8

# Button k sits at angle (k-0.5)*45 degrees clockwise from the top.
# Left-right mirror (vertical axis): 1<->8, 2<->7, 3<->6, 4<->5.
# Up-down mirror (horizontal axis):  1<->4, 2<->3, 5<->8, 6<->7.
# D/E sensors sit on the boundaries between buttons (D1 at the top).
_BUTTON = {
    'identity': lambda k: k,
    'mirror_lr': lambda k: 9 - k,
    'mirror_ud': lambda k: (4 - k) % 8 + 1,
    'rotate_180': lambda k: (k + 3) % 8 + 1,
}
_BOUNDARY = {
    'identity': lambda k: k,
    'mirror_lr': lambda k: (9 - k) % 8 + 1,
    'mirror_ud': lambda k: (5 - k) % 8 + 1,
    'rotate_180': lambda k: (k + 3) % 8 + 1,
}
# '<'/'>' mean "start toward the player's left/right", so they swap whenever
# left and right swap. p/q are absolute rotation senses and swap under any
# reflection. s/z are mirror images of each other and swap under reflections.
_SHAPES = {
    'identity': {},
    'mirror_lr': {'<': '>', '>': '<', 'p': 'q', 'q': 'p', 's': 'z', 'z': 's'},
    'mirror_ud': {'p': 'q', 'q': 'p', 's': 'z', 'z': 's'},
    'rotate_180': {'<': '>', '>': '<'},
}


@lru_cache(maxsize=262144)
def transform_text(text, transform):
    """Mirror one Simai cell; durations inside [...] are copied verbatim."""
    if transform == 'identity':
        return text
    button, boundary, shapes = _BUTTON[transform], _BOUNDARY[transform], _SHAPES[transform]
    result = []
    depth = 0
    previous = ''
    for char in text:
        if char == '[':
            depth += 1
        elif char == ']':
            depth = max(0, depth - 1)
        if depth or char in '[]':
            result.append(char)
        elif char.isdigit() and '1' <= char <= '8':
            if previous in ('D', 'E'):
                result.append(str(boundary(int(char))))
            elif previous == 'C':
                result.append(char)
            else:
                result.append(str(button(int(char))))
        else:
            result.append(shapes.get(char, char))
        previous = char if not depth else previous
    return ''.join(result)


def _pad_permutation(transform):
    """Permutation of the 50-dim sustain state derived from pad geometry."""
    from maichart.generation.muri.model import CENTERS, PADS
    from maichart.generation.action_state_spec import STATE_DIM
    from maichart.generation import sustain_state as sustain
    sign = {'identity': (1, 1), 'mirror_lr': (-1, 1), 'mirror_ud': (1, -1), 'rotate_180': (-1, -1)}[transform]
    names = list(PADS)
    index = np.arange(STATE_DIM)
    for source, name in enumerate(names):
        x, y = CENTERS[name]
        target = (sign[0] * x, sign[1] * y)
        match = min(names, key=lambda other: math.dist(CENTERS[other], target))
        if math.dist(CENTERS[match], target) > 1e-3 or match[0] != name[0]:
            raise ValueError(f'Pad geometry is not symmetric under {transform}: {name}')
        index[sustain._PAD_OFFSET + names.index(match)] = sustain._PAD_OFFSET + source
    for lane in range(1, LANES + 1):
        index[sustain._LANE_OFFSET + _BUTTON[transform](lane) - 1] = sustain._LANE_OFFSET + lane - 1
    return torch.as_tensor(index, dtype=torch.long)


@lru_cache(maxsize=8)
def state_permutation(transform):
    # new_state[:, i] = old_state[:, permutation[i]]
    return _pad_permutation(transform)


def transform_state(values, transform):
    if transform == 'identity' or values.numel() == 0:
        return values
    permutation = state_permutation(transform).to(values.device)
    width = permutation.numel()
    head = values[..., :width].index_select(-1, permutation)
    return torch.cat((head, values[..., width:]), -1)


@lru_cache(maxsize=262144)
def event_lanes(text):
    """Buttons struck at an event's head time (taps, holds and slide stars)."""
    lanes = set()
    for part in str(text).split('/'):
        part = part.strip()
        if not part or part[0] in 'ABCDE':
            continue
        if len(part) > 1 and all('1' <= c <= '8' for c in part):
            lanes.update(int(c) for c in part)
        elif '1' <= part[0] <= '8':
            lanes.add(int(part[0]))
    return tuple(sorted(lanes))


def _angle(lanes):
    angles = [(lane - .5) * math.pi / 4 for lane in lanes]
    x = sum(math.sin(a) for a in angles); y = sum(math.cos(a) for a in angles)
    return math.atan2(x, y) if abs(x) + abs(y) > 1e-6 else None


SLIDE, HOLD = 1, 0


def sustain_trajectories(markers, event_seconds, *, tolerance=1e-3):
    """Per event: ``(kind, head_lane, end_lane, trace_second, end_second)``.

    ``kind`` is 1 for Slide/Wifi and 0 for Hold. A Hold leaves on its own
    button; a Slide/Wifi starts moving at ``trace_second`` and leaves on its end
    button at ``end_second``. This is the cabinet fact behind "slide into the
    next Tap" (一笔画): the next star or Tap is usually written while the
    previous Slide is still in flight, so the destination must be visible
    *before* the Slide ends, not only after. Touch actions are not included.
    """
    seconds = np.asarray(event_seconds, dtype=np.float64)
    result = [[] for _ in range(len(seconds))]
    if not len(seconds):
        return []
    for marker in markers or ():
        kind = marker.get('type')
        if kind not in ('hold', 'slide', 'wifi') or marker.get('is_mine') or marker.get('track_mine'):
            continue
        try:
            head = int(marker.get('lane'))
            end_lane = head if kind == 'hold' else int(marker.get('end_lane'))
        except (TypeError, ValueError):
            continue
        if not (1 <= head <= LANES and 1 <= end_lane <= LANES):
            continue
        onset, end = float(marker.get('second', math.nan)), float(marker.get('end_second', math.nan))
        if not (math.isfinite(onset) and math.isfinite(end)):
            continue
        trace = onset if kind == 'hold' else float(marker.get('slide_trace_second', onset))
        if not math.isfinite(trace):
            trace = onset
        right = int(np.searchsorted(seconds, onset))
        candidates = [i for i in (right - 1, right) if 0 <= i < len(seconds)]
        index = min(candidates, key=lambda i: abs(seconds[i] - onset))
        if abs(seconds[index] - onset) <= tolerance:
            result[index].append((HOLD if kind == 'hold' else SLIDE, head, end_lane, trace, end))
    return [tuple(sorted(item)) for item in result]


def transform_trajectories(trajectories, transform):
    if transform == 'identity':
        return trajectories
    mapping = _BUTTON[transform]
    return [tuple(sorted((kind, mapping(head), mapping(end_lane), trace, end)
                         for kind, head, end_lane, trace, end in item)) for item in trajectories]


def _effective_lanes(lanes, trajectories):
    """Where the hands will be once an event's own Slides have been played."""
    slides = [t for t in (trajectories or ()) if t[0] == SLIDE]
    if not slides:
        return lanes
    heads = {head for _kind, head, _end_lane, _trace, _end in slides}
    ends = {end_lane for _kind, _head, end_lane, _trace, _end in slides}
    return tuple(sorted((set(lanes) - heads) | ends))


def kinematic_rows(lane_sets, event_seconds, committed, seconds, *, horizon=16, trajectories=None):
    """Causal hand features for queries; only committed events are read.

    Pure-Tap passages leave no sustained action, so this is the decoder's only
    explicit record of which buttons were just struck and how hands travelled.
    ``trajectories`` (see :func:`sustain_trajectories`) adds where sustained
    actions already ended (36:44) and, for Slides committed but not finished,
    their destination (44:52), origin phase (52:60), count and time to arrival
    (60:62), and the hand step measured on effective lanes (62:65).
    """
    committed = np.asarray(committed, dtype=np.int64)
    seconds = np.asarray(seconds, dtype=np.float64)
    event_seconds = np.asarray(event_seconds, dtype=np.float64)
    result = np.zeros((len(committed), KINEMATIC_DIM), dtype=np.float32)
    # Formerly every query rescanned every committed trajectory to find the
    # longest sustain: O(queries * song_events) Python/GIL work per sample.
    # One causal prefix maximum preserves masked histories and long Slides;
    # a query reads only its own committed prefix, never a future duration.
    duration_prefix = None
    if trajectories is not None:
        total = min(len(trajectories),len(event_seconds),len(lane_sets),
                    max(0,int(committed.max())) if len(committed) else 0)
        durations=np.zeros(total,dtype=np.float64)
        for event in range(total):
            onset=float(event_seconds[event])
            durations[event]=max((max(0.,float(item[4])-onset)
                                  for item in trajectories[event]),default=0.)
        duration_prefix=np.r_[0.,np.maximum.accumulate(durations)]
    cache = {}
    for row, (count, now) in enumerate(zip(committed, seconds)):
        count = min(int(count), len(lane_sets), len(event_seconds))
        if count <= 0:
            continue
        key = (count, float(now))
        if key in cache:
            result[row] = cache[key]; continue
        values = np.zeros(KINEMATIC_DIM, dtype=np.float32)
        start = max(0, count - horizon)
        last_hit = np.full(LANES, -np.inf)
        for event in range(start, count):
            for lane in lane_sets[event]:
                last_hit[lane - 1] = event_seconds[event]
        age = np.maximum(0., now - last_hit)
        values[:8] = np.where(np.isfinite(last_hit), np.exp(-age / .5), 0.)
        last = lane_sets[count - 1]
        for lane in last:
            values[8 + lane - 1] = 1.
        values[16] = math.log1p(max(0., now - event_seconds[count - 1])) / math.log(9.)
        if count > 1:
            prev = lane_sets[count - 2]
            for lane in prev:
                values[24 + lane - 1] = 1.
            values[17] = math.log1p(max(0., event_seconds[count - 1] - event_seconds[count - 2])) / math.log(9.)
            a, b = _angle(prev), _angle(last)
            if a is not None and b is not None:
                step = b - a
                values[18], values[19], values[20] = math.sin(step), math.cos(step), 1.
            if count > 2:
                a2 = _angle(lane_sets[count - 3])
                if a2 is not None and a is not None:
                    step2 = a - a2
                    values[32], values[33] = math.sin(step2), math.cos(step2)
        run = 0
        for event in range(count - 1, start - 1, -1):
            if lane_sets[event] == last and last:
                run += 1
            else:
                break
        values[21] = min(run, 8) / 8.
        values[22] = min(len(last), 4) / 4.
        values[23] = float(not last)
        sweep_dir = 0
        sweep_run = 0
        for event in range(count - 1, max(0, start), -1):
            cur_lanes, prev_lanes = lane_sets[event], lane_sets[event - 1]
            if len(cur_lanes) != 1 or len(prev_lanes) != 1:
                break
            delta = (cur_lanes[0] - prev_lanes[0]) % LANES
            step_dir = 1 if delta == 1 else (-1 if delta == LANES - 1 else 0)
            if step_dir == 0:
                break
            if sweep_dir == 0:
                sweep_dir = step_dir
            if step_dir == sweep_dir:
                sweep_run += 1
            else:
                break
        values[34] = sweep_dir * min(sweep_run, 8) / 8.
        trill_run = 0
        for event in range(count - 1, max(1, start + 1), -1):
            cur_lanes, prev_lanes, alt_lanes = lane_sets[event], lane_sets[event - 1], lane_sets[event - 2]
            if cur_lanes and prev_lanes and cur_lanes == alt_lanes and cur_lanes != prev_lanes:
                trill_run += 1
            else:
                break
        values[35] = min(trill_run, 8) / 8.
        if trajectories is not None:
            # A Slide can remain in flight far beyond the recent strike
            # horizon.  This scan is intentionally over all committed events;
            # limiting it to ``4 * horizon`` makes an old, long Slide vanish
            # from the causal state while it is still moving.  The event bound
            # keeps malformed/masked trajectory arrays from leaking or raising.
            released = np.full(LANES, -np.inf)
            destination = np.zeros(LANES)
            origin = np.zeros(LANES)
            flying, soonest = 0, math.inf
            trajectory_count = min(count, len(trajectories))
            max_duration = float(duration_prefix[trajectory_count])
            # Release channels decay over roughly 0.5 s. Keep five half-lives
            # in addition to the maximum track duration so a just-released
            # action remains visible without scanning an entire long song.
            relevant_horizon = max_duration + 2.5
            start_event = max(0, int(np.searchsorted(
                event_seconds[:trajectory_count], now - relevant_horizon - 1e-9, side='left')))
            for event in range(start_event, trajectory_count):
                for kind, head, end_lane, trace, end in trajectories[event]:
                    if end <= now + 1e-9:
                        if end > released[end_lane - 1]:
                            released[end_lane - 1] = end
                    elif kind == SLIDE:
                        # Committed and not finished: the hand is on its way
                        # to end_lane, which is where the next star or Tap of
                        # a one-stroke figure is usually written.
                        remaining = end - now
                        destination[end_lane - 1] = max(destination[end_lane - 1], .5 + .5 * math.exp(-remaining / .5))
                        origin[head - 1] = max(origin[head - 1], 1. if now < trace else .5)
                        flying += 1
                        soonest = min(soonest, remaining)
            release_age = np.maximum(0., now - released)
            values[36:44] = np.where(np.isfinite(released), np.exp(-release_age / .5), 0.)
            values[44:52] = destination
            values[52:60] = origin
            values[60] = min(flying, 4) / 4.
            values[61] = math.log1p(min(soonest, 8.)) / math.log(9.) if flying else 0.
            # The step the hands really took: Slide heads count where the Slide ends.
            current_trajectory = trajectories[count - 1] if count <= len(trajectories) else ()
            now_lanes = _effective_lanes(last, current_trajectory)
            if count > 1:
                previous_trajectory = trajectories[count - 2] if count - 1 <= len(trajectories) else ()
                a_eff = _angle(_effective_lanes(lane_sets[count - 2], previous_trajectory))
                b_eff = _angle(now_lanes)
                if a_eff is not None and b_eff is not None:
                    values[62], values[63], values[64] = math.sin(b_eff - a_eff), math.cos(b_eff - a_eff), 1.
        cache[key] = values
        result[row] = values
    return torch.from_numpy(result)
