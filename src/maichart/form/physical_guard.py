"""Causal physical-feasibility guard for song_form_v2 rational decoding.

The Muri layer (``generation/muri/runtime.py``, ``generation/muri/static.py``)
and the cabinet geometry (``generation/maimai_physics.py``) already exist, but
they only run *after* a chart is complete.  A decoder therefore keeps writing
notes no player can execute: it will happily place a Tap on a button that the
Hold it emitted two beats earlier still occupies, because nothing in the decode
loop knows the button is held.

This module supplies the missing causal half.  It stores only already committed
events, so it can answer at every decoder step without knowing the future.

Two conservative heuristics are modelled:

* **Occupancy** -- a Hold occupies its button until the reference parser's end
  time. Slide proximity follows the reference moving path, only after its wait;
  its head and endpoint are never reserved for the whole Slide duration.
* **Travel** -- the hands must reach the next buttons in the elapsed time.  Two
  hand positions are tracked through the chart, and the required speed is
  bounded by ``MAX_HAND_SPEED`` in the same normalised cabinet units as
  ``PAD_DISTANCES`` (the ``A`` ring radius is 1.0).

Both are *soft*.  The guard returns a bounded penalty added to candidate logits.
It never removes a token from the decoder's candidate set: an empty candidate
set turns one questionable note into a hard decode failure, which is the failure
mode ``_NoteGrammar.head_closure`` exists to avoid, and it would replace the
model's judgement with a blacklist.  Strength 0 disables the guard entirely.

Nothing here is a fixed note grid.  It constrains *where a hand can be*, not
which subdivision the music may use, so tuplet-like rhythms and mixed meters
remain fully representable.
"""
from __future__ import annotations

from collections import deque
from fractions import Fraction
from functools import lru_cache
import math

import numpy as np

from maichart.generation.judgment_system import pad_center
from maichart.generation.muri.model import CENTERS, RADII, has_head
from maichart.generation.muri.parser import parse_markers
from maichart.generation.muri.runtime import build_actions

LANES = 8
# Normalised cabinet units per second.  Crossing the whole cabinet (2.0 units)
# takes 0.2 s at this bound: fast, but clearly human.
MAX_HAND_SPEED = 10.0
# Below this gap a hand cannot reposition at all, so dividing by the real
# elapsed time would produce a meaningless speed.
MIN_TRAVEL_SECONDS = 0.05
# How far apart two lanes may be for one hand to cover both at once.  Adjacent
# lanes are 0.765 units apart and the next-nearest are 1.414, so 0.90 admits a
# palm hitting neighbouring buttons and rejects anything wider.
MAX_HAND_SPAN = 0.90
# Bounded additive logit penalties.  A collision is the stronger signal: the
# note is not merely difficult, it is impossible.
OCCUPANCY_PENALTY = 8.0
# A moving Slide can share judged sensors with other actions. Geometric
# proximity is weaker evidence than a held button, not a hard collision.
SLIDE_PROXIMITY_PENALTY = 2.0
TRAVEL_PENALTY = 4.0
MAX_PHYSICAL_PENALTY = 12.0
MAX_HAND_STATES = 16
# Above this many simultaneous buttons the exhaustive two-hand split is
# skipped; a hand cannot hold that many anyway and the split is arbitrary.
MAX_EXACT_CHORD = 6
EPS = 1e-9
# Speed-excess equivalent of dropping a Slide with all of its path still to go
# (TRAVEL_PENALTY * this * strength logits). Deliberately modest: a figure that
# repeats Slides from one button with alternating hands is legitimate.
ABANDON_EXCESS = 0.6
# Slide launch taps are judged within this many seconds of the launch (1/90 s).
LAUNCH_WINDOW = 1.0 / 90.0

_PAD_CENTERS = np.asarray([pad_center(lane) for lane in range(1, LANES + 1)], dtype=np.float64)
# Distance between every pair of hand lanes, in cabinet units.
LANE_DISTANCE = np.linalg.norm(_PAD_CENTERS[:, None, :] - _PAD_CENTERS[None, :, :], axis=-1)
# MiaCode reference paths use screen pixels with +y downwards; cabinet units
# put the A ring at radius 1.0 with +y upwards.
_CABINET_SCALE = 1.0 / math.hypot(*CENTERS['A1'])


def _to_cabinet(point):
    return (float(point[0]) * _CABINET_SCALE, -float(point[1]) * _CABINET_SCALE)


class _SustainTrack:
    """The hand that struck a Hold or Slide head stays bound to that action.

    * Hold: the hand is pinned on its button until release.
    * Slide/Wifi: during the star wait the hand rests on the head as before;
      while tracing it follows the reference path; at the end it is on the
      end button, free from the end time onward. This is what makes a Tap
      next to a Slide's end ("一笔画") cheap and a Tap back at its head costly.
    Identity hashing is intentional: tracks are per committed action.
    """
    __slots__ = ('kind', 'lane', 'end_lane', 'trace', 'end', 'actions')

    def __init__(self, kind, lane, end_lane, trace, end, actions=()):
        self.kind, self.lane, self.end_lane = kind, lane, end_lane
        self.trace, self.end, self.actions = float(trace), float(end), tuple(actions)

    def resolve(self, when, hand, previous):
        if when >= self.end - EPS:
            return self.end_lane, self.end
        if self.kind == 'hold':
            return self.lane, when
        if when < self.trace - EPS:
            return hand, previous
        samples = [action.sample(when) for action in self.actions]
        samples = [_to_cabinet(sample[0]) for sample in samples if sample is not None]
        if not samples:
            return self.end_lane, when
        return (sum(p[0] for p in samples) / len(samples), sum(p[1] for p in samples) / len(samples)), when

    def abandon_cost(self, group, when, rebinding=False):
        """Speed-excess equivalent of pulling this hand off an unfinished Slide.

        A plain tap is free when it is where the Slide is going anyway (its end
        lane: a one-stroke hand-off) or is the launch tap on its own head. A hand
        that is re-bound to a new Slide or Hold head (``rebinding``) really
        drops this Slide, so those exemptions do not apply. Holds are free
        (their button is pinned by the occupancy term). Otherwise the cost is
        proportional to the path still to travel, halved while the star waits.
        A second hand taking the note costs nothing, so legitimate two-hand
        figures, including repeated same-head Slides, are not penalised.
        """
        if self.kind == 'hold' or when >= self.end - EPS:
            return 0.0
        if not rebinding and (self.end_lane in group
                              or (self.lane in group and abs(when - self.trace) <= LAUNCH_WINDOW + EPS)):
            return 0.0
        if when < self.trace - EPS:
            return ABANDON_EXCESS * .5
        return ABANDON_EXCESS * min(1.0, max(0.0, (self.end - when) / max(self.end - self.trace, EPS)))


def validate_physical_strength(value, name="physical_feasibility_strength"):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 3:
        raise ValueError(f"{name} must be finite and between 0 and 3")
    return value


def _lane(value):
    """Return a hand lane 1..8, or None for Touch/Center positions."""
    if isinstance(value, bool):
        return None
    try:
        lane = int(value)
    except (TypeError, ValueError):
        return None
    return lane if 1 <= lane <= LANES else None


def lane_mask(lanes):
    """Boolean mask over the eight hand lanes.

    Accepts whatever the note grammar reports as a position: ``1``..``8`` for a
    tap or slide head, but also ``"A4"``/``"C"`` for Touch notes.  Touch is not
    a hand on a button, so anything that is not a plain lane number is dropped.
    """
    mask = np.zeros(LANES, dtype=bool)
    for lane in lanes:
        value = _lane(lane)
        if value is not None:
            mask[value - 1] = True
    return mask


def _group_travel(hand, group):
    """Largest distance this hand must cover to reach every lane in ``group``."""
    if not group or hand is None:
        return 0.0
    if isinstance(hand, tuple):
        return max(math.dist(hand, _PAD_CENTERS[lane - 1]) for lane in group)
    return max(float(LANE_DISTANCE[hand - 1, lane - 1]) for lane in group)


def _medoid(group):
    """The lane in ``group`` closest to all the others, as a hand position."""
    if not group:
        return None
    return min(group, key=lambda lane: sum(float(LANE_DISTANCE[lane - 1, other - 1])
                                           for other in group))


def _span_ok(group, span=MAX_HAND_SPAN):
    """Whether one hand can cover every lane in ``group`` at once."""
    if len(group) < 2:
        return True
    widest = max(float(LANE_DISTANCE[a - 1, b - 1]) for a in group for b in group)
    return widest <= span + EPS


def assign_hands(hands, lanes, span=MAX_HAND_SPAN):
    """Move the two hands onto ``lanes``, minimising the larger travel distance.

    Returns ``(new_hands, cost)``.  ``hands`` may contain ``None`` for an
    unknown position; an unknown hand costs 0 so the guard stays silent until
    the chart gives it something to track.

    A split that would put two far-apart lanes on one hand is rejected first:
    a five-button chord is two hands working in parallel, not one wide hand.
    Without that check the cheapest split is always "everything on one hand",
    because an unplaced hand contributes no distance.
    """
    values = sorted({int(lane) for lane in lanes if lane is not None and 1 <= int(lane) <= LANES})
    if not values:
        return hands, 0.0
    left, right = hands
    if len(values) > MAX_EXACT_CHORD:
        # Such a chord already exceeds two hands; keep a stable arbitrary split.
        half = len(values) // 2
        splits = [(tuple(values[:half]), tuple(values[half:]))]
    else:
        splits = [(tuple(lane for index, lane in enumerate(values) if (mask >> index) & 1),
                   tuple(lane for index, lane in enumerate(values) if not (mask >> index) & 1))
                  for mask in range(1 << len(values))]
    best = None
    for group_a, group_b in splits:
        # Rank: span-feasible splits first, then least travel, then the most
        # balanced split so a spread chord does not collapse onto one hand.
        rank = (0 if _span_ok(group_a, span) and _span_ok(group_b, span) else 1,
                max(_group_travel(left, group_a), _group_travel(right, group_b)),
                -min(len(group_a), len(group_b)))
        if best is None or rank < best[0]:
            best = (rank, group_a, group_b)
    rank, group_a, group_b = best
    return ((_medoid(group_a) if group_a else left), (_medoid(group_b) if group_b else right)), rank[1]


@lru_cache(maxsize=256)
def _lane_splits(lanes):
    values = tuple(sorted(set(lanes)))
    splits = []
    for bits in range(1 << len(values)):
        groups = (tuple(lane for index, lane in enumerate(values) if (bits >> index) & 1),
                  tuple(lane for index, lane in enumerate(values) if not (bits >> index) & 1))
        if all(_span_ok(group) for group in groups):
            splits.append(groups)
    return tuple(splits)


def _hand_excess(hands, seconds, groups, when, speed):
    """Minimum speed excess uses each hand's own previous action time."""
    return max((max(0.0, _group_travel(hand, group) /
                    max(MIN_TRAVEL_SECONDS, when - previous) / speed - 1.0)
                if hand is not None and previous is not None and group else 0.0)
               for hand, previous, group in zip(hands, seconds, groups))


class PhysicalEvent:
    """A per-event view bound to one decode second, handed to the note-cell bias."""

    def __init__(self, guard, second):
        self.guard = guard
        self.second = second

    def penalty(self, lanes):
        """Bounded penalty for a single known button set."""
        return self.guard.lane_penalty(lanes, self.second)

    def vector_penalty(self, matrix):
        """Penalties for a ``(n, 8)`` boolean matrix of candidate lane sets."""
        return self.guard.matrix_penalty(matrix, self.second)


class PhysicalFeasibilityGuard:
    """Causal occupancy and hand-travel state over committed events only."""

    def __init__(self, strength=1.0, *, max_hand_speed=MAX_HAND_SPEED):
        self.strength = validate_physical_strength(strength)
        self.max_hand_speed = float(max_hand_speed)
        if not math.isfinite(self.max_hand_speed) or self.max_hand_speed <= 0:
            raise ValueError("max_hand_speed must be finite and positive")
        self.tempo = None
        self.first = 0.0
        self._cues = {}
        self._sustains = deque()
        self._slide_actions = []
        self._hands = (None, None)
        self._hand_seconds = (None, None)
        # Preserve alternative feasible assignments instead of committing a
        # greedy left/right choice that can falsely constrain the next event.
        # (positions, last free seconds, bound sustain tracks, cost)
        self._hand_states = [(self._hands, self._hand_seconds, (None, None), 0.0)]
        self.tracked_sustains = 0
        self._last_second = None
        self._lane_set_cache = {}
        self._travel_cache = {}
        self.events = 0
        self.occupancy_blocks = 0
        self.travel_blocks = 0
        self.committed_collisions = 0
        self.slide_proximity_events = 0
        self.parse_failures = 0
        self.max_penalty = 0.0
        self.diagnostics = deque(maxlen=64)

    # -- causal clock ----------------------------------------------------
    def register(self, beat, second, tempo=None, first=None):
        """Record the song second for a beat, and the tempo map that extends it."""
        self._cues[Fraction(beat)] = float(second)
        if tempo is not None:
            self.tempo = list(tempo)
            self.first = float(first or 0.0)

    def event(self, beat):
        second = self._cues.get(Fraction(beat))
        if second is None:
            return None
        # Speculative candidate times may be evaluated out of order. Pruning
        # here loses held actions if a later candidate is rejected.
        return PhysicalEvent(self, second)

    # -- committed state -------------------------------------------------
    def _prune(self, second):
        # Notes are committed by onset; their ends need not be sorted.
        self._sustains = deque((end, held) for end, held in self._sustains
                               if end > second + EPS)
        self._slide_actions = [action for action in self._slide_actions
                               if action.end > second + EPS]

    def occupied_mask(self, second):
        """Definite held-button occupancy; moving Slides use a separate bias."""
        mask = np.zeros(LANES, dtype=bool)
        for end_second, held in self._sustains:
            if float(second) < end_second - EPS:
                mask |= held
        return mask

    def slide_proximity_mask(self, second):
        """Reference path footprints at this instant, excluding Slide wait."""
        result = np.zeros(LANES, dtype=bool)
        for action in self._slide_actions:
            sampled = action.sample(float(second))
            if sampled is None:
                continue
            position, _tangent = sampled
            for index in range(LANES):
                if math.dist(position, CENTERS[f'A{index + 1}']) <= action.radius + RADII['A'] + EPS:
                    result[index] = True
        return result

    # -- scoring ---------------------------------------------------------
    def _raw_penalty(self, matrix, second):
        matrix = np.asarray(matrix, dtype=bool)
        if matrix.ndim != 2 or matrix.shape[1] != LANES:
            raise ValueError("candidate lane matrix must have shape (n, 8)")
        count = matrix.shape[0]
        values = np.zeros(count, dtype=np.float64)
        if not count:
            return values
        occupied = self.occupied_mask(second)
        if occupied.any():
            values += OCCUPANCY_PENALTY * matrix[:, occupied].sum(1)
        slide = self.slide_proximity_mask(second) & ~occupied
        if slide.any():
            values += SLIDE_PROXIMITY_PENALTY * matrix[:, slide].sum(1)
        if self._last_second is not None:
            # Each distinct candidate lane-set is scored once: the grammar
            # repeats cells heavily, so the same chord arrives many times.
            sets = self._lane_sets(matrix)
            costs = {}
            for lanes in sets:
                if lanes not in costs:
                    key = (lanes, float(second))
                    if key not in self._travel_cache:
                        self._travel_cache[key] = self._travel_excess(lanes, float(second))
                    costs[lanes] = self._travel_cache[key]
            excess = np.asarray([costs[lanes] for lanes in sets], dtype=np.float64)
            values += np.minimum(TRAVEL_PENALTY * 2.0, TRAVEL_PENALTY * excess)
        return np.minimum(MAX_PHYSICAL_PENALTY, values) * self.strength

    def _lane_sets(self, matrix):
        """Lane indices for each row of a boolean matrix, memoised per call."""
        result = []
        for row in matrix:
            key = row.tobytes()
            lanes = self._lane_set_cache.get(key)
            if lanes is None:
                lanes = tuple(int(index) + 1 for index in np.nonzero(row)[0])
                self._lane_set_cache[key] = lanes
            result.append(lanes)
        return result

    @staticmethod
    def _resolve(hands, times, tracks, second):
        """Effective (position, free-since) per hand at ``second``."""
        resolved = [(hand, previous) if track is None else track.resolve(second, hand, previous)
                    for hand, previous, track in zip(hands, times, tracks)]
        return tuple(item[0] for item in resolved), tuple(item[1] for item in resolved)

    @staticmethod
    def _settle(hands, times, tracks, second):
        """Materialise sustains that already ended: the hand is free at the end button."""
        if all(track is None or track.end > second + EPS for track in tracks):
            return hands, times, tracks
        values = [((track.end_lane, track.end, None) if track is not None and track.end <= second + EPS
                   else (hand, previous, track)) for hand, previous, track in zip(hands, times, tracks)]
        return tuple(v[0] for v in values), tuple(v[1] for v in values), tuple(v[2] for v in values)

    @staticmethod
    def _abandonment(tracks, groups, second):
        """Cost of pulling hands off unfinished Slides to strike ``groups``."""
        return sum(track.abandon_cost(group, second) for track, group in zip(tracks, groups)
                   if track is not None and group)

    def _travel_excess(self, lanes, second):
        """Best plausible two-hand speed excess over a bounded assignment beam."""
        if not lanes:
            return 0.0
        splits = _lane_splits(tuple(sorted(lanes)))
        best = math.inf
        for hands, times, tracks, cost in self._hand_states:
            positions, free = self._resolve(hands, times, tracks, second)
            for groups in splits:
                best = min(best, cost + _hand_excess(positions, free, groups, second, self.max_hand_speed)
                           + self._abandonment(tracks, groups, second))
        return best

    @staticmethod
    def _group_track(group, heads):
        """A hand striking a sustained head becomes bound to the longest such action."""
        bound = [heads[lane] for lane in group if lane in heads]
        return max(bound, key=lambda track: track.end) if bound else None

    @staticmethod
    def _overbound(group, heads):
        """Extra cost when one hand would have to follow two moving sustains.

        Two adjacent Holds under one palm are fine; two Slides, or a Slide
        plus a Hold, cannot be followed by the same hand.
        """
        bound = [heads[lane] for lane in group if lane in heads]
        if len(bound) < 2 or all(track.kind == 'hold' for track in bound):
            return 0.0
        return float(len(bound) - 1)

    def _commit_hands(self, lanes, second, heads=None):
        if not lanes:
            return
        heads = heads or {}
        next_states = {}
        for hands, times, tracks, previous_cost in self._hand_states:
            hands, times, tracks = self._settle(hands, times, tracks, second)
            positions, free = self._resolve(hands, times, tracks, second)
            for groups in _lane_splits(tuple(sorted(lanes))):
                bound = tuple(self._group_track(group, heads) if group else None for group in groups)
                excess = _hand_excess(positions, free, groups, second, self.max_hand_speed)
                excess += sum(self._overbound(group, heads) for group in groups)
                excess += sum(track.abandon_cost(group, second, rebinding=new is not None)
                              for track, group, new in zip(tracks, groups, bound)
                              if track is not None and group)
                new_hands = tuple(_medoid(group) if group else hand
                                  for group, hand in zip(groups, hands))
                new_times = tuple(second if group else previous
                                  for group, previous in zip(groups, times))
                # A hand re-bound to a new sustained head releases its earlier
                # Slide (charged above). A plain tap on that Slide's end lane or
                # launch keeps the hand on it; any other tap leaves it free.
                new_tracks = tuple(track if not group else new if new is not None
                                   else track if (track is not None and track.kind != 'hold'
                                                  and second < track.end - EPS
                                                  and track.abandon_cost(group, second) == 0.0) else None
                                   for group, track, new in zip(groups, tracks, bound))
                # Hands have no intrinsic identity here. Canonicalize mirrored
                # assignments to use the bounded beam for distinct histories.
                key = tuple(sorted(zip(new_hands, new_times, new_tracks),
                                   key=lambda item: (item[0] or 0,
                                                     -math.inf if item[1] is None else item[1],
                                                     0 if item[2] is None else id(item[2]))))
                candidate = (tuple(item[0] for item in key), tuple(item[1] for item in key),
                             tuple(item[2] for item in key), previous_cost + excess)
                if key not in next_states or candidate[3] < next_states[key][3]:
                    next_states[key] = candidate
        if not next_states:
            # A questionable chord must not overwrite the last plausible hand
            # histories and manufacture further impossible travel constraints.
            return
        ordered = sorted(next_states.values(), key=lambda state: state[3])[:MAX_HAND_STATES]
        minimum = ordered[0][3]
        self._hand_states = [(hands, times, tracks, cost - minimum) for hands, times, tracks, cost in ordered]
        self._hands, self._hand_seconds, _, _ = self._hand_states[0]

    def matrix_penalty(self, matrix, second):
        values = self._raw_penalty(matrix, second)
        if values.size:
            self.max_penalty = max(self.max_penalty, float(values.max()))
        return values

    def lane_penalty(self, lanes, second=None):
        if self.strength <= 0:
            return 0.0
        mask = lane_mask(lanes)
        if not mask.any():
            return 0.0
        when = self._last_second if second is None else float(second)
        if when is None:
            return 0.0
        return float(self.matrix_penalty(mask[None], when)[0])

    # -- commit ----------------------------------------------------------
    def commit(self, text, beat, second=None):
        if self.strength <= 0 or not text:
            return
        beat = Fraction(beat)
        if second is None:
            second = self._cues.get(beat)
        if second is None:
            return
        second = float(second)
        self._prune(second)
        bpm = 120.0
        if self.tempo:
            # The reference fixes each duration from the BPM at its head;
            # later tempo changes move later heads, not an already shot Slide.
            bpm = float(self.tempo[0].bpm)
            for change in self.tempo:
                if Fraction(change.beat) > beat:
                    break
                bpm = float(change.bpm)
        try:
            markers, errors, _warnings = parse_markers(f'({bpm}){{4}}{text},', first=second)
        except (ValueError, TypeError, KeyError, OverflowError):
            self.parse_failures += 1
            return
        if errors:
            self.parse_failures += 1
            self.diagnostics.append({'beat': str(beat), 'second': second, 'text': text,
                                     'reason': 'reference_parse_failed', 'errors': errors[:3]})
            return
        occupied = self.occupied_mask(second)
        slide_proximity = self.slide_proximity_mask(second)
        lanes = set()
        # build_actions uses the same cached reference paths as MiaCode, with
        # the parser's exact wait, segment timing and seconds/BPM overrides.
        actions = build_actions(markers, [])
        heads = {}
        for marker in markers:
            if marker.get('is_mine') or marker.get('head_mine'):
                continue
            kind = marker['type']
            head = _lane(marker.get('lane'))
            struck = head is not None and (kind in {'tap', 'hold'} or has_head(marker))
            if struck:
                lanes.add(head)
            end = float(marker.get('end_second', second))
            if kind == 'hold' and head is not None and end > second + EPS:
                self._sustains.append((end, lane_mask([head])))
            # Only a struck head binds a hand; head-less slides stay unbound.
            if struck and end > second + EPS and kind in {'hold', 'slide', 'wifi'}:
                if kind == 'hold':
                    track = _SustainTrack('hold', head, head, second, end)
                else:
                    end_lane = _lane(marker.get('end_lane'))
                    trace = float(marker.get('slide_trace_second', second))
                    if end_lane is None or not math.isfinite(trace):
                        continue
                    track = _SustainTrack(kind, head, end_lane, trace, end,
                                          [action for action in actions if action.marker is marker])
                if head not in heads or track.end > heads[head].end:
                    heads[head] = track
        self.tracked_sustains += len(heads)
        self._slide_actions.extend(actions)
        mask = lane_mask(lanes)
        collisions = mask & occupied
        if collisions.any():
            # The decoder chose an impossible note anyway, meaning every offered
            # continuation was costly. Recorded rather than silently accepted.
            self.occupancy_blocks += 1
            self.committed_collisions += 1
            self.diagnostics.append({
                "beat": str(beat), "second": second, "text": text,
                "colliding_lanes": [int(index) + 1 for index in np.nonzero(collisions)[0]],
                "reason": "committed_on_active_sustain",
            })
        if (mask & slide_proximity).any():
            self.slide_proximity_events += 1
        if self._last_second is not None and lanes:
            # Use the same cost model the candidate penalty uses, so this count
            # describes exactly what the guard was steering against.
            if self._travel_excess(lanes, second) > 0:
                self.travel_blocks += 1
        self._commit_hands(lanes, second, heads)
        self._travel_cache.clear()
        self._last_second = second
        self.events += 1

    def report(self):
        return {
            "schema": "song_form_physical_feasibility_v3",
            "enabled": self.strength > 0,
            "strength": self.strength,
            "applied": self.strength > 0,
            "max_hand_speed_units_per_second": self.max_hand_speed,
            "min_travel_seconds": MIN_TRAVEL_SECONDS,
            "occupancy_penalty": OCCUPANCY_PENALTY * self.strength,
            "travel_penalty": TRAVEL_PENALTY * self.strength,
            "penalty_bound": MAX_PHYSICAL_PENALTY * self.strength,
            "events": self.events,
            "active_sustains": len(self._sustains),
            "active_slide_actions": len(self._slide_actions),
            "occupancy_blocks": self.occupancy_blocks,
            "travel_blocks": self.travel_blocks,
            "committed_collisions": self.committed_collisions,
            "slide_proximity_events": self.slide_proximity_events,
            "reference_parse_failures": self.parse_failures,
            "max_observed_penalty": self.max_penalty,
            "hand_positions": [int(hand) if hand is not None else None for hand in self._hands],
            "hand_last_seconds": list(self._hand_seconds),
            "hand_hypotheses": len(self._hand_states),
            "tracked_sustain_heads": self.tracked_sustains,
            "sustain_hand_model": "hold_pinned_slide_follows_path_ends_on_end_lane",
            "duration_semantics": "miacode_head_bpm_v1",
            "slide_proximity_penalty": SLIDE_PROXIMITY_PENALTY * self.strength,
            "diagnostics": list(self.diagnostics),
            "feedback_scope": "causal_committed_prefix_only",
            "removes_candidates": False,
            "touch_excluded": True,
            "limitations": [
                "Soft logit bias only; it never removes a candidate token.",
                "Slide proximity samples the reference moving footprint after its wait; "
                "it is a weak bias, not a claim of an impossible collision.",
                "Hand travel keeps bounded alternative assignments and independent clocks. "
                "A hand that struck a Hold stays on it, a Slide hand follows the reference "
                "path and ends on the end button; there is no left/right body model.",
                "Touch notes are deliberately excluded: authored Touch usage is too "
                "free-form to be a reliable hand-position signal.",
                "Cabinet units follow the pinned MiaCode reference geometry, not a player model.",
            ],
        }


__all__ = ["PhysicalFeasibilityGuard", "PhysicalEvent", "assign_hands", "lane_mask",
           "validate_physical_strength", "LANE_DISTANCE", "MAX_HAND_SPEED",
           "OCCUPANCY_PENALTY", "TRAVEL_PENALTY", "MAX_PHYSICAL_PENALTY"]
