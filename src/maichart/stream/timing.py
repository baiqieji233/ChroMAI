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


def interval_consistency_cost(delta, prediction, strength):
    """Compare exact delta to the censored interval head without capping delta.

    The training label saturates at eight beats. A saturated prediction means
    'eight or more', so arbitrarily long rests remain unpenalized in that case.
    Below saturation, a 161-beat jump cannot cost the same as a nine-beat one.
    """
    if not strength or prediction is None or not math.isfinite(prediction):
        return 0.
    target = max(0., min(1., float(prediction))) * 8.
    actual = max(0., float(delta))
    if target >= 8. and actual >= 8.:
        return 0.
    # One quarter beat prevents an unstable relative error at a near-zero
    # prediction. It is a score scale, never an output grid or allowed gap.
    difference = math.log((actual+.25)/(target+.25))
    return float(strength) * .5 * difference * difference


def export_end_timing(audio_end, tempo, first):
    # E is not a played event. Using a separately approximated 1/960 beat
    # here multiplied the last note's denominator (78 -> {49920}). An integer
    # terminal beat cannot introduce a new factor; all note times stay exact.
    end = max(Fraction(0),Fraction(math.ceil(audio_end)))
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
