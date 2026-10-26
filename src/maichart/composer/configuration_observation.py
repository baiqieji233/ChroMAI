"""Descriptive event repetition; these observations are not quality gates."""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
from itertools import groupby

from .events import event_metadata


def _patterns(values, beats, *, maximum_gap=None):
    counts = Counter()
    for size, name in ((2, 'same'), (3, 'ABA'), (4, 'ABAB')):
        for start in range(len(values) - size + 1):
            window = values[start:start + size]
            if any(value is None for value in window):
                continue
            if maximum_gap is not None and any(
                    not 0 < beats[i + 1] - beats[i] <= maximum_gap
                    for i in range(start, start + size - 1)):
                continue
            counts[f'windows_{size}'] += 1
            a, b = window[:2]
            matched = a == b if size == 2 else a == window[2] and a != b
            if size == 4:
                matched = matched and b == window[3]
            if matched:
                counts[name] += 1
                if size > 2:
                    for label, pair in (('lanes_1_8', {'1', '8'}), ('lanes_4_5', {'4', '5'})):
                        if {a, b} == pair:
                            counts[f'{label}_{name}'] += 1
            if size == 2 and a == b == '1':
                counts['lane_1_same'] += 1
    definitions = [('same', 2), ('ABA', 3), ('ABAB', 4), ('lane_1_same', 2)]
    definitions += [(f'{label}_{pattern}', size)
                    for label in ('lanes_1_8', 'lanes_4_5')
                    for pattern, size in (('ABA', 3), ('ABAB', 4))]
    return {
        'windows': {str(size): counts[f'windows_{size}'] for size in (2, 3, 4)},
        'counts': {name: counts[name] for name, _ in definitions},
        'rates': {name: counts[name] / counts[f'windows_{size}'] if counts[f'windows_{size}'] else None
                  for name, size in definitions},
    }


def configuration_observation(points):
    """Measure committed/exported events, preserving simultaneous note groups.

    A singleton Tap window breaks at a chord, Touch, Hold or Slide: filtering
    those events out would invent adjacency. Half-beat windows also break at
    rests longer than half a beat; they include eighth notes and faster notes.
    Unsupported event syntax is reported and excluded from Tap windows.
    """
    points = list(points)
    texts = [point.note_text for point in points]
    beats = [point.beat for point in points]
    frequencies = Counter(texts)
    tap_lanes, unsupported = [], 0
    for text in texts:
        try:
            meta = event_metadata(text)
        except ValueError:
            unsupported += 1
            tap_lanes.append(None)
            continue
        tap_lanes.append(meta.heads[0] if meta.type_counts == (1, 0, 0, 0, 0)
                         and len(meta.heads) == 1 else None)
    return {
        'unique_texts': len(frequencies), 'top_texts': frequencies.most_common(5),
        'max_consecutive_same_text': max((sum(1 for _ in group) for _, group in groupby(texts)), default=0),
        'repetition': {
            'schema': 'composer_event_repetition_v1', 'event_count': len(points),
            'exact_event_text': _patterns(texts, beats),
            'consecutive_single_tap_lanes': _patterns(tap_lanes, beats),
            'single_tap_lanes_at_most_half_beat': _patterns(tap_lanes, beats, maximum_gap=Fraction(1, 2)),
            'unsupported_event_count': unsupported,
            'scope': 'adjacent_exported_events; raw note text equality; singleton Tap windows exclude other events; descriptive only',
            'named_pairs': '1/8 and 4/5 mean alternating singleton Tap heads, not every possible configuration using those lanes',
        },
    }
