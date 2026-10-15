"""Describe joint-event generation failures without assuming a frame planner."""
from __future__ import annotations

from fractions import Fraction

from maichart.event_timing import seconds_at_beat


def rational_diagnostics(points, records, tempo, first, duration_sec):
    times = [float(seconds_at_beat(point.beat, tempo, first)) for point in points]
    edges = [0.] + times + [float(duration_sec)]
    gaps = sorted(({'start_sec': a, 'end_sec': b, 'duration_sec': b-a}
                   for a, b in zip(edges, edges[1:]) if b-a >= 4.),
                  key=lambda gap: gap['duration_sec'], reverse=True)
    incomplete = [row['span'] for row in records if row['status'].get('incomplete')
                  or row['status']['reason'] in ('invalid_position_prefix','position_search_exhausted')]
    empty_eos = [row['span'] for row in records
                 if row['onsets'] == 0 and row['status']['reason'] == 'eos'
                 and row.get('has_allowed_positions', True)]
    forced_empty = [row['span'] for row in records
                    if row['onsets'] == 0 and row['status'].get('reason') == 'eos'
                    and row.get('has_allowed_positions', True)
                    and bool(row['status'].get('event_start_allowed', True))]
    return {
        'schema': 'song_form_rational_diagnostics_v1',
        'event_count': len(points), 'span_count': len(records),
        'empty_eos_spans': len(empty_eos), 'empty_eos_span_indices': empty_eos,
        'forced_empty_eos_spans': forced_empty,
        'incomplete_spans': incomplete,
        'position_recoveries': sum(row['status'].get('position_recoveries', 0) for row in records),
        'gap_threshold_sec': 4., 'gap_count': len(gaps), 'gaps': gaps[:8],
        'max_gap_sec': max((b-a for a, b in zip(edges, edges[1:])), default=0.),
        'large_denominator_events': sum(Fraction(point.beat).denominator > 32 for point in points),
        'large_denominator_reference': 32,
        'incomplete_reasons': [row['status'].get('reason') for row in records
                               if row['status'].get('incomplete')],
        'empty_eos_forced_flags': [bool(row['status'].get('eos_forced', False)) for row in records
                                   if row['onsets'] == 0 and row['status'].get('reason') == 'eos'],
        'musical_quality_verified': False,
        'limitations': [
            'Gaps may be intentional rests; larger denominators may be legitimate rhythms.',
            'EOS is chosen by the joint event decoder, not a frame onset threshold.',
            'These counters cannot establish musical alignment or playable chart quality.',
        ],
    }
