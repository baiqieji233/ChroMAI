"""Causal physical-feasibility guard for song_form_v2 rational decoding.

The Muri layer (``generation/muri/runtime.py``, ``generation/muri/static.py``)
and the cabinet geometry (``generation/maimai_physics.py``) already exist, but
they only run *after* a chart is complete.  A decoder therefore keeps writing
notes no player can execute: it will happily place a Tap on a button that the
Hold it emitted two beats earlier still occupies, because nothing in the decode
loop knows the button is held.

This module supplies the missing causal half.  It stores only already committed
events, so it can answer at every decoder step without knowing the future.

Two constraints are modelled.  Both are physics rather than stylistic
preference, which is what makes them safe to steer decoding with:

* **Occupancy** -- a Hold occupies its button, and a Slide occupies its head and
  target buttons, for the whole declared duration.  A later note striking one of
  those buttons inside that window cannot be pressed; the native checker reports
  this family as ``Overlap`` / ``TapOnSlide``.
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
import math

import numpy as np

from maichart.event_timing import seconds_at_beat
from maichart.generation.judgment_system import pad_center
from maichart.notes import parse_timing_point_notes
from maichart.timing import RawTimingPoint

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
TRAVEL_PENALTY = 4.0
MAX_PHYSICAL_PENALTY = 12.0
# Above this many simultaneous buttons the exhaustive two-hand split is
# skipped; a hand cannot hold that many anyway and the split is arbitrary.
MAX_EXACT_CHORD = 6
SUSTAIN_TYPES = ("hold", "slide")
EPS = 1e-9

_PAD_CENTERS = np.asarray([pad_center(lane) for lane in range(1, LANES + 1)], dtype=np.float64)
# Distance between every pair of hand lanes, in cabinet units.
LANE_DISTANCE = np.linalg.norm(_PAD_CENTERS[:, None, :] - _PAD_CENTERS[None, :, :], axis=-1)


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


def _sustain_window(note, *, start_second, tempo, first, beat):
    """Return ``(end_second, lanes)`` for a sustained note, else None."""
    if note.note_type not in SUSTAIN_TYPES:
        return None
    duration = note.duration_beats
    if duration is None or float(duration) <= 0:
        return None
    end_second = float(seconds_at_beat(Fraction(beat) + Fraction(duration), tempo, first))
    if end_second <= start_second + EPS:
        return None
    lanes = set()
    head = _lane(note.position)
    if head is not None:
        lanes.add(head)
    if note.note_type == "slide":
        tail = _lane(getattr(note, "end_position", None))
        if tail is not None:
            lanes.add(tail)
        for segment in (getattr(note, "slide_segments", None) or ()):
            for key in ("position", "end_position"):
                lane = _lane(segment.get(key))
                if lane is not None:
                    lanes.add(lane)
    if not lanes:
        return None
    return end_second, frozenset(lanes)


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
        self._hands = (None, None)
        self._last_second = None
        self._lane_set_cache = {}
        self.events = 0
        self.occupancy_blocks = 0
        self.travel_blocks = 0
        self.committed_collisions = 0
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
        self._prune(second)
        return PhysicalEvent(self, second)

    # -- committed state -------------------------------------------------
    def _prune(self, second):
        while self._sustains and self._sustains[0][0] <= second + EPS:
            self._sustains.popleft()

    def occupied_mask(self, second):
        mask = np.zeros(LANES, dtype=bool)
        for end_second, held in self._sustains:
            if float(second) < end_second - EPS:
                mask |= held
        return mask

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
        if self._last_second is not None:
            gap = max(MIN_TRAVEL_SECONDS, float(second) - self._last_second)
            # Each distinct candidate lane-set is scored once: the grammar
            # repeats cells heavily, so the same chord arrives many times.
            sets = self._lane_sets(matrix)
            costs = {}
            for lanes in sets:
                if lanes not in costs:
                    costs[lanes] = self._travel_cost(lanes)
            travel = np.asarray([costs[lanes] for lanes in sets], dtype=np.float64)
            speed = travel / gap
            excess = np.maximum(0.0, speed / self.max_hand_speed - 1.0)
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

    def _travel_cost(self, lanes):
        """Least distance any hand must cover to press exactly ``lanes``.

        ``math.inf`` means no two-hand assignment can cover the chord at all:
        one hand would have to span more than one palm, or a lane is beyond
        reach.  Only the already placed hands are considered; an unplaced hand
        contributes no distance, which keeps the guard silent until the chart
        has given it something to track.
        """
        if not lanes:
            return 0.0
        values = sorted(lanes)
        if len(values) > 10:
            # Beyond two hands regardless of timing; the exact split is moot.
            return math.inf
        left, right = self._hands
        best = math.inf
        for mask in range(1 << len(values)):
            first = tuple(lane for index, lane in enumerate(values) if (mask >> index) & 1)
            second = tuple(lane for index, lane in enumerate(values) if not (mask >> index) & 1)
            if not _span_ok(first) or not _span_ok(second):
                continue
            cost = max(_group_travel(left, first), _group_travel(right, second))
            if cost < best:
                best = cost
        return best

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
        try:
            parsed = parse_timing_point_notes(RawTimingPoint(
                index=0, raw=text, note_text=text, beat=beat, tick=0,
                time_sec=second, division=4, bpm=120.0))
        except Exception:
            return
        if parsed.unknown_tokens and not parsed.notes:
            return
        occupied = self.occupied_mask(second)
        lanes = set()
        for note in parsed.notes:
            head = _lane(note.position)
            if head is not None:
                lanes.add(head)
            if self.tempo is not None:
                window = _sustain_window(note, start_second=second, tempo=self.tempo,
                                         first=self.first, beat=beat)
                if window is not None:
                    end_second, held = window
                    self._sustains.append((end_second, lane_mask(held)))
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
        if self._last_second is not None and lanes:
            # Use the same cost model the candidate penalty uses, so this count
            # describes exactly what the guard was steering against.
            travel_cost = self._travel_cost(sorted(lanes))
            if math.isfinite(travel_cost) and travel_cost > 0:
                gap = max(MIN_TRAVEL_SECONDS, second - self._last_second)
                if travel_cost / gap > self.max_hand_speed:
                    self.travel_blocks += 1
        self._hands, _ = assign_hands(self._hands, lanes)
        self._last_second = second
        self.events += 1

    def report(self):
        return {
            "schema": "song_form_physical_feasibility_v1",
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
            "occupancy_blocks": self.occupancy_blocks,
            "travel_blocks": self.travel_blocks,
            "committed_collisions": self.committed_collisions,
            "max_observed_penalty": self.max_penalty,
            "hand_positions": [int(hand) if hand is not None else None for hand in self._hands],
            "diagnostics": list(self.diagnostics),
            "feedback_scope": "causal_committed_prefix_only",
            "removes_candidates": False,
            "touch_excluded": True,
            "limitations": [
                "Soft logit bias only; it never removes a candidate token.",
                "Slide occupancy uses head and target lanes for the declared duration; "
                "the exact swept path stays with the post-generation MiaCode checker.",
                "Hand positions are representative lanes, not a body simulation.",
                "Cabinet units follow the pinned MiaCode reference geometry, not a player model.",
            ],
        }


__all__ = ["PhysicalFeasibilityGuard", "PhysicalEvent", "assign_hands", "lane_mask",
           "validate_physical_strength", "LANE_DISTANCE", "MAX_HAND_SPEED",
           "OCCUPANCY_PENALTY", "TRAVEL_PENALTY", "MAX_PHYSICAL_PENALTY"]
