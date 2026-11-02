"""Exact onset subdivision diagnostics, without generation preferences.

Absolute metrical phase and inter-onset spacing answer different questions:
an integer-beat onset can close a sixteenth-note run. Report both separately.
Conditions are caller-supplied strata (e.g. authored density or musical section),
never inferred quality labels. Compare generated/reference results in the same
strata; this metric does not declare dense rhythms better than sparse rhythms.
"""
from collections import Counter
from fractions import Fraction


def conditioned_rhythm_summary(beats, conditions=None):
    """Summarize strictly increasing, unique NOTE onset beats.

    ``conditions[i]`` labels onset i; its preceding interval is assigned to
    that landing's stratum. The first onset has no gap. Stratum boundaries do
    not invent gaps between nonadjacent notes. Beat values should be Fractions
    or exact strings from the chart, rather than rounded playback seconds.
    """
    beats = tuple(Fraction(str(value)) for value in beats)
    if any(right <= left for left, right in zip(beats, beats[1:])):
        raise ValueError('Rhythm diagnostics require unique increasing onsets')
    labels = ('all',)*len(beats) if conditions is None else tuple(map(str, conditions))
    if len(labels) != len(beats):
        raise ValueError('One condition label is required per onset')

    def new_bucket():
        return dict(onsets=0, gaps=0, phase=Counter(), gap=Counter(), short=[0, 0, 0], total_gap=Fraction(0))

    all_rows = new_bucket()
    strata = {}
    thresholds = (Fraction(1, 2), Fraction(1, 4), Fraction(1, 8))
    for i, (beat, label) in enumerate(zip(beats, labels)):
        bucket = strata.setdefault(label, new_bucket())
        for target in (all_rows, bucket):
            target['onsets'] += 1
            target['phase'][str((beat % 1).denominator)] += 1
            if i:
                gap = beat-beats[i-1]
                target['gaps'] += 1
                target['gap'][str(gap.denominator)] += 1
                target['total_gap'] += gap
                for j, threshold in enumerate(thresholds):
                    target['short'][j] += int(gap <= threshold)

    def finish(bucket):
        count = bucket['gaps']
        return dict(onsets=bucket['onsets'], gaps=count,
                    phase_denominator_counts=dict(sorted(bucket['phase'].items(), key=lambda item: int(item[0]))),
                    gap_denominator_counts=dict(sorted(bucket['gap'].items(), key=lambda item: int(item[0]))),
                    mean_gap_beats=float(bucket['total_gap']/count) if count else None,
                    gap_at_most_half_beat_fraction=bucket['short'][0]/count if count else None,
                    gap_at_most_quarter_beat_fraction=bucket['short'][1]/count if count else None,
                    gap_at_most_eighth_beat_fraction=bucket['short'][2]/count if count else None)

    return dict(overall=finish(all_rows), conditions={key: finish(value) for key, value in sorted(strata.items())})
