"""Exact stream timing, musical diagnostics, and a simple non-note EOF."""
from __future__ import annotations

from fractions import Fraction
import math
import re

from maichart.event_timing import seconds_at_beat
from maichart.form.rational_generate import PositionGrammar


class CanonicalPositionGrammar(PositionGrammar):
    """Use the training spelling in both completion and reachability checks."""
    def complete(self, text):
        if not super().complete(text):
            return False
        return str(Fraction(text)) == text


def export_end_timing(audio_end, tempo, first):
    # E is not a played event. Using a separately approximated 1/960 beat
    # here multiplied the last note's denominator (78 -> {49920}). An integer
    # terminal beat cannot introduce a new factor; all note times stay exact.
    end = Fraction(math.ceil(audio_end))
    return end, {'audio_end_beat': str(audio_end), 'export_end_beat': str(end),
        'end_extension_sec': float(seconds_at_beat(end, tempo, first)
                                   - seconds_at_beat(audio_end, tempo, first)),
        'policy': 'next_integer_beat_eof_only', 'notes_unchanged': True}


def timing_diagnostics(points, tempo, first, duration, text):
    times = [float(seconds_at_beat(p.beat, tempo, first)) for p in points]
    previous = 0.
    gaps = []
    for point, second in zip(points, times):
        gaps.append({'start_sec': previous, 'end_sec': second, 'duration_sec': second-previous,
                     'next_beat': str(point.beat)})
        previous = second
    gaps.append({'start_sec': previous, 'end_sec': duration,
                 'duration_sec': max(0., duration-previous), 'next_beat': None})
    intervals = [b.beat-a.beat for a, b in zip(points, points[1:])]
    return {'event_count': len(points), 'first_event_sec': times[0] if times else None,
        'last_event_sec': times[-1] if times else None,
        'max_gap_sec': max((g['duration_sec'] for g in gaps), default=duration),
        'largest_gaps': sorted(gaps, key=lambda g: g['duration_sec'], reverse=True)[:12],
        'max_beat_denominator': max((p.beat.denominator for p in points), default=0),
        'max_interval_denominator': max((d.denominator for d in intervals), default=0),
        'max_export_division': max(map(int, re.findall(r'\{(\d+)\}', text)), default=0),
        'large_gap_is_not_automatically_an_error': True}
