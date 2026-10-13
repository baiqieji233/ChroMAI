"""Optional, bounded approximation of selected onsets to common Simai grids."""
from bisect import bisect_right
from fractions import Fraction
import math

from maichart.event_timing import seconds_at_beat


COMMON_DIVISIONS = (4, 8, 12, 16, 24, 32)


def normalize_divisions(values):
    if isinstance(values, str):
        parts = [part.strip() for part in values.split(',') if part.strip()]
        if any(not part.isascii() or not part.isdecimal() for part in parts):
            raise ValueError('Rhythm divisions must be comma-separated positive integers')
        values = [int(part) for part in parts]
    if not isinstance(values,(list,tuple)) or not values:
        raise ValueError('Select between 1 and 32 rhythm divisions')
    if any(isinstance(value,bool) or not isinstance(value,int) or not 1 <= value <= 768 for value in values):
        raise ValueError('Rhythm divisions must be integers in [1, 768]')
    unique = tuple(sorted(set(values)))
    if len(unique) > 32:
        raise ValueError('Select between 1 and 32 rhythm divisions')
    return unique


def validate_snap_ms(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 50:
        raise ValueError('rhythm_snap_ms must be in [0, 50]')
    return value


def approximate_onsets(records, tempo, first, *, tolerance_ms=15., min_interval_ms=0.,
                       end_beat=None, duration_sec=None, rest_regions=(),divisions=COMMON_DIVISIONS):
    """Move only safe, nearby points; never combine cells or replace note text.

    The grids refer to 4/division quarter-note beats, independent of meter.
    Every move is checked against the exact tempo clock, neighboring onsets,
    known tempo boundaries, silence regions, and the original frame timestamp.
    """
    tolerance = validate_snap_ms(tolerance_ms)/1000
    divisions = normalize_divisions(divisions)
    result = [dict(item) for item in records]
    beats = [Fraction(item['beat']) for item in result]
    if beats != sorted(set(beats)):
        raise ValueError('Rhythm approximation requires sorted, unique beats')
    tempo_boundaries = sorted(Fraction(item.beat) for item in tempo)
    stats = dict(enabled=True, divisions=list(divisions), tolerance_ms=float(tolerance_ms),
                 changed=0, already_common=0, outside_tolerance=0, blocked=0, maximum_shift_ms=0.)
    occupied = set(beats)
    for index, item in enumerate(result):
        beat = beats[index]
        stamp = float(seconds_at_beat(beat,tempo,first))
        if any((beat*division/4).denominator == 1 for division in divisions):
            stats['already_common'] += 1
            continue
        if tolerance == 0:
            stats['outside_tolerance'] += 1
            continue
        candidates = {}
        for division in divisions:
            step = Fraction(4,division)
            cell = beat // step
            for position in (cell*step, (cell+1)*step):
                if position >= 0:
                    candidates[position] = min(division,candidates.get(position,division))
        ranked = sorted(((abs(seconds_at_beat(b,tempo,first)-stamp),d,b) for b,d in candidates.items()))
        nearby = [(error,division,position) for error,division,position in ranked if error <= tolerance+1e-9]
        if not nearby:
            stats['outside_tolerance'] += 1
            continue
        changed = False
        for error, division, position in nearby:
            second = float(seconds_at_beat(position,tempo,first))
            if position in occupied or position in tempo_boundaries or beat in tempo_boundaries:
                continue
            if bisect_right(tempo_boundaries,position) != bisect_right(tempo_boundaries,beat):
                continue
            if end_beat is not None and position >= end_beat:
                continue
            if second < 0 or (duration_sec is not None and second >= duration_sec):
                continue
            if any(float(region['start_sec']) <= second < float(region['end_sec']) for region in rest_regions):
                continue
            if 'source_second' in item and abs(second-float(item['source_second'])) > tolerance+1e-9:
                continue
            previous = result[index-1] if index else None
            following = result[index+1] if index+1 < len(result) else None
            if previous is not None and position <= Fraction(previous['beat']):
                continue
            if following is not None and position >= Fraction(following['beat']):
                continue
            minimum = float(min_interval_ms)/1000
            if previous is not None and second-float(previous['second']) < minimum-1e-9:
                continue
            if following is not None and float(following['second'])-second < minimum-1e-9:
                continue
            occupied.remove(beat)
            occupied.add(position)
            item.update(pre_snap_beat=str(beat),pre_snap_second=stamp,beat=str(position),second=second,
                        rhythm_snap_shift_ms=(second-stamp)*1000,rhythm_snap_division=division)
            stats['changed'] += 1
            stats['maximum_shift_ms'] = max(stats['maximum_shift_ms'],error*1000)
            changed = True
            break
        if not changed:
            stats['blocked'] += 1
    stats['count_before'] = stats['count_after'] = len(result)
    return result, stats
