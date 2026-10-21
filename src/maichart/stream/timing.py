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


def interval_reliability(metrics=None):
    """A conservative score multiplier, not a calibrated probability.

    Old checkpoints have no interval residual telemetry. Keep their auxiliary
    scalar weak. New training logs MAE in the very log-ratio space scored here.
    Training residuals are optimistic; never let this single head dominate CE.
    """
    metrics=metrics or {}
    error=metrics.get('interval_log_mae')
    count=metrics.get('interval_targets',0)
    if (not isinstance(error,(int,float)) or not math.isfinite(error) or error<0
            or not isinstance(count,(int,float)) or count<32):
        return {'multiplier':.25,'source':'legacy_or_insufficient_telemetry'}
    return {'multiplier':min(.75,1./(1.+4.*error)),
            'source':'teacher_forced_log_residual','log_mae':float(error),'targets':int(count)}


def interval_consistency_cost(delta, prediction, strength, gap_penalty_strength=0.,*,reliability=1.,cap=None):
    """Compare exact delta to the censored interval head without capping delta.

    The training label saturates at eight beats. A saturated prediction means
    'eight or more', so arbitrarily long rests remain unpenalized in that case.
    Below saturation, a 161-beat jump cannot cost the same as a nine-beat one.
    """
    if prediction is None or not math.isfinite(prediction) or float(prediction) < 0.:
        return 0.
    if not math.isfinite(reliability) or not 0<=reliability<=1:
        raise ValueError('Interval reliability must be in [0,1]')
    if cap is not None and (not math.isfinite(cap) or cap<0):
        raise ValueError('Interval shaping cap must be nonnegative')
    target = max(0., min(1., float(prediction))) * 8.
    actual = max(0., float(delta))
    if target >= 8. and actual >= 8.:
        return 0.
    cost = 0.
    if strength:
        # One quarter beat prevents an unstable relative error at a near-zero
        # prediction. It is a score scale, never an output grid or allowed gap.
        difference = math.log((actual+.25)/(target+.25))
        shape=.5*difference*difference
        if cap is not None:shape=min(cap,shape)
        # Predictions below a quarter beat are especially vulnerable to raw
        # regression error. Attenuate, do not snap or prohibit fine rhythms.
        confidence=reliability*(min(1.,target/.25) if cap is not None else 1.)
        cost += float(strength)*confidence*shape
    if gap_penalty_strength and target < 8.:
        threshold = max(target * 2., target + 4.)
        if actual > threshold:
            cost += float(gap_penalty_strength) * .35 * (actual - threshold)
    return cost


def rhythm_cycle_cost(beat, points, previous, strength, max_period=24):
    """Penalize continuing a non-uniform rhythm loop past two full cycles.

    Uniform subdivision streams (steady 4th/8th/16th runs where len(set(cycle))==1)
    and two-phrase echoes remain unpenalized. Only composite multi-interval patterns
    that have already repeated for at least two full periods or two 4-beat bars
    incur a bounded nonnegative cost.
    """
    if not strength or previous is None or previous < 0 or len(points) < 4:
        return 0.
    strength = float(strength)
    if not math.isfinite(strength) or strength <= 0:
        return 0.
    beat = Fraction(beat)
    previous = Fraction(previous)
    delta = beat - previous
    if delta <= 0:
        return 0.
    recent = points[-max(72, 3 * max_period + 1):]
    deltas = [b.beat - a.beat for a, b in zip(recent, recent[1:])]
    best_reps = 0
    limit = min(max_period, len(deltas) // 2)
    for period in range(2, limit + 1):
        cycle = deltas[-period:]
        if delta != cycle[0] or len(set(cycle)) <= 1:
            continue
        if deltas[-2 * period:-period] != cycle:
            continue
        reps = 2
        while (reps + 1) * period <= len(deltas) and deltas[-(reps + 1) * period:-reps * period] == cycle:
            reps += 1
        if reps > best_reps:
            best_reps = reps
    bar_reps = 0
    bar_intervals = [d for p, d in zip(recent[1:], deltas) if p.beat > previous - 4]
    if len(set(bar_intervals)) > 1:
        delta_at = {b.beat: b.beat - a.beat for a, b in zip(recent, recent[1:])}
        k = 1
        while delta_at.get(beat - 4 * k) == delta:
            bar_reps += 1
            k += 1
    if best_reps < 2 and bar_reps < 2:
        return 0.
    cost = .55 * max(0, best_reps - 1) + .45 * max(0, bar_reps - 1)
    return strength * min(3.5, cost)



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
