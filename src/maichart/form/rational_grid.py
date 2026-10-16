"""Conservative post-decode simplification for near-common Fraction beats.

The song-form v2 decoder keeps arbitrary rational positions.  A weak or
high-temperature checkpoint can nevertheless spell a normal grid with a
small numerical drift (for example 35.24468 instead of 35.25).  The lossless
Simai compiler must preserve that drift and consequently emits large grid
directives such as ``{376}``.  This module only removes that local drift; it
does not impose a denominator whitelist and leaves isolated unusual rhythms
alone unless they have local sequence support.
"""
from __future__ import annotations

from fractions import Fraction
import math

from maichart.event_timing import EventPoint, seconds_at_beat


DEFAULT_DIVISIONS = (4, 6, 8, 12, 16, 24, 32, 48, 64, 96)


def normalize_rational_grid_divisions(values):
    if isinstance(values, str):
        parts = [part.strip() for part in values.split(',') if part.strip()]
        if any(not part.isascii() or not part.isdecimal() for part in parts):
            raise ValueError('Rational grid divisions must be comma-separated positive integers')
        values = [int(part) for part in parts]
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError('Select between 1 and 32 rational grid divisions')
    if any(isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 768
           for value in values):
        raise ValueError('Rational grid divisions must be integers in [1, 768]')
    result = tuple(sorted(set(values)))
    if len(result) > 32:
        raise ValueError('Select between 1 and 32 rational grid divisions')
    return result


def _near_grid_candidates(beat, tempo, first, divisions, tolerance_sec, *,
                          start_beat=None, end_beat=None):
    """Return (shift_seconds, division, candidate_beat) near ``beat``."""
    result = []
    for division in divisions:
        step = Fraction(4, division)
        cell = beat // step
        for index in (cell - 1, cell, cell + 1, cell + 2):
            candidate = index * step
            if (candidate < 0 or (start_beat is not None and candidate < start_beat)
                    or (end_beat is not None and candidate >= end_beat)):
                continue
            shift = abs(float(seconds_at_beat(candidate, tempo, first) -
                             seconds_at_beat(beat, tempo, first)))
            if shift <= tolerance_sec + 1e-9:
                result.append((shift, division, candidate))
    result.sort(key=lambda value: (round(value[0], 9), value[1], value[2]))
    unique = []
    seen = set()
    for item in result:
        if item[2] in seen:
            continue
        seen.add(item[2])
        unique.append(item)
    return unique


def simplify_rational_events(points, tempo, first, *, tolerance_ms=8.0,
                             divisions=DEFAULT_DIVISIONS, min_spacing_ms=0.0,
                             start_beat=None, end_beat=None, valid_position=None,
                             immutable_prefix=0):
    """Move only locally-supported, near-grid Fraction positions.

    The original note text and event count are preserved.  A position is a
    candidate only when its reduced denominator is 17 or larger; this protects
    isolated 5/7/9/11/13-tuplet-like rhythms and other intentional small-
    denominator timings.  At least one
    adjacent event must also be on a common grid or have a compatible nearby
    candidate.  All moves are checked for monotonicity, rests/boundaries and
    the configured physical minimum spacing. ``immutable_prefix`` freezes that
    many leading events while retaining them as spacing and rhythm evidence.
    ``start_beat`` is inclusive and ``end_beat`` is exclusive, so cleanup never
    moves a newly decoded event into a neighboring compute span.
    """
    tolerance_ms = float(tolerance_ms)
    if not math.isfinite(tolerance_ms) or not 0 <= tolerance_ms <= 50:
        raise ValueError('rational_grid_snap_ms must be in [0, 50]')
    min_spacing_ms = float(min_spacing_ms)
    if not math.isfinite(min_spacing_ms) or min_spacing_ms < 0:
        raise ValueError('min_spacing_ms must be finite and nonnegative')
    divisions = normalize_rational_grid_divisions(divisions)
    values = [EventPoint(Fraction(point.beat), point.note_text) for point in points]
    values.sort(key=lambda point: point.beat)
    if (isinstance(immutable_prefix, bool) or not isinstance(immutable_prefix, int)
            or not 0 <= immutable_prefix <= len(values)):
        raise ValueError('immutable_prefix must be an event count within points')
    start_beat = Fraction(start_beat) if start_beat is not None else None
    end_beat = Fraction(end_beat) if end_beat is not None else None
    if start_beat is not None and end_beat is not None and start_beat >= end_beat:
        raise ValueError('Rational grid bounds must have positive duration')
    if any(a.beat >= b.beat for a, b in zip(values, values[1:])):
        raise ValueError('Rational grid simplification requires sorted, unique beats')
    stats = {
        'enabled': tolerance_ms > 0,
        'algorithm': 'near_common_rational_grid_v1',
        'tolerance_ms': tolerance_ms,
        'divisions': list(divisions),
        'min_spacing_ms': min_spacing_ms,
        'immutable_prefix': immutable_prefix,
        'changed': 0,
        'isolated_complex_skipped': 0,
        'collision_blocked': 0,
        'boundary_blocked': 0,
        'maximum_shift_ms': 0.0,
        'changed_indices': [],
    }
    if len(values) < 2 or tolerance_ms == 0:
        stats['count_before'] = stats['count_after'] = len(values)
        return values, stats

    # The threshold is deliberately independent of the candidate list.  A
    # user may add division 28 or 40, but a small denominator such as 7 or 9
    # is still treated as a meaningful unusual rhythm unless the user turns
    # this cleanup off or supplies a tighter downstream policy.
    complex_denominator_min = 17
    tolerance_sec = tolerance_ms / 1000.0
    minimum_sec = min_spacing_ms / 1000.0
    candidates = []
    for index, point in enumerate(values):
        # Only the final two frozen neighbors can support a mutable event.
        # Avoid recomputing tempo/grid queries over the entire song prefix on
        # every resource pass while keeping the exact same local evidence.
        if index < immutable_prefix - 2:
            candidates.append([])
            continue
        candidates.append(_near_grid_candidates(point.beat, tempo, first, divisions,
                                                 tolerance_sec, start_beat=start_beat,
                                                 end_beat=end_beat))

    # Keep support local.  This avoids moving an isolated intentional 1/7 or
    # 1/11 rhythm merely because a nearby common grid happens to be within a
    # few milliseconds.
    supported = []
    for index, point in enumerate(values):
        if point.beat.denominator < complex_denominator_min:
            supported.append(True)
            continue
        near = candidates[index]
        if not near:
            supported.append(False)
            continue
        support = False
        for neighbor in (index - 2, index - 1, index + 1, index + 2):
            if 0 <= neighbor < len(values):
                if values[neighbor].beat.denominator < complex_denominator_min or candidates[neighbor]:
                    support = True
                    break
        supported.append(support)

    output = []
    occupied = set()
    for index, point in enumerate(values):
        chosen = point.beat
        if index >= immutable_prefix and point.beat.denominator >= complex_denominator_min:
            if not supported[index] or not candidates[index]:
                stats['isolated_complex_skipped'] += 1
            else:
                for shift, _division, candidate in candidates[index]:
                    if candidate in occupied:
                        continue
                    if output and candidate <= output[-1].beat:
                        continue
                    following = values[index + 1].beat if index + 1 < len(values) else None
                    if following is not None and candidate >= following:
                        continue
                    if valid_position is not None and not valid_position(candidate):
                        stats['boundary_blocked'] += 1
                        continue
                    if output and float(seconds_at_beat(candidate, tempo, first) -
                                       seconds_at_beat(output[-1].beat, tempo, first)) < minimum_sec - 1e-9:
                        stats['collision_blocked'] += 1
                        continue
                    if following is not None and float(seconds_at_beat(following, tempo, first) -
                                                       seconds_at_beat(candidate, tempo, first)) < minimum_sec - 1e-9:
                        stats['collision_blocked'] += 1
                        continue
                    chosen = candidate
                    if chosen != point.beat:
                        stats['changed'] += 1
                        stats['changed_indices'].append(index)
                        stats['maximum_shift_ms'] = max(stats['maximum_shift_ms'], shift * 1000.0)
                    break
        occupied.add(chosen)
        output.append(EventPoint(chosen, point.note_text))
    stats['count_before'] = stats['count_after'] = len(output)
    stats['changed_indices'] = stats['changed_indices'][:128]
    stats['changed_indices_truncated'] = stats['changed'] > 128
    return output, stats
