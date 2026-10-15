from fractions import Fraction as F

from maichart.event_timing import EventPoint, EventTempo
from maichart.form.rational_grid import simplify_rational_events


def test_near_common_grid_drift_is_simplified_without_whitelisting_fraction_positions():
    points = [
        EventPoint(F(35), '1'),
        EventPoint(F(3313, 94), '2'),  # 35.24468, about 2.5 ms from 35.25 at 130 BPM
        EventPoint(F(71, 2), '3'),
        EventPoint(F(1211, 34), '4'),
        EventPoint(F(143, 4), '5'),
    ]
    result, stats = simplify_rational_events(
        points, [EventTempo(F(0), 130)], 0,
        tolerance_ms=8, min_spacing_ms=20, end_beat=F(40))
    assert result[1].beat == F(141, 4)
    assert result[3].beat == F(285, 8)
    assert stats['changed'] >= 2
    assert all(a.note_text == b.note_text for a, b in zip(points, result))


def test_small_denominator_unusual_rhythm_is_kept_when_isolated():
    points = [EventPoint(F(0), '1'), EventPoint(F(1, 7), '2'), EventPoint(F(1), '3')]
    result, stats = simplify_rational_events(
        points, [EventTempo(F(0), 180)], 0,
        tolerance_ms=12, min_spacing_ms=0, end_beat=F(2))
    assert result[1].beat == F(1, 7)
    assert stats['changed'] == 0


def test_zero_tolerance_preserves_all_exact_positions():
    points = [EventPoint(F(0), '1'), EventPoint(F(3313, 94), '2')]
    result, stats = simplify_rational_events(
        points, [EventTempo(F(0), 130)], 0,
        tolerance_ms=0, end_beat=F(40))
    assert [point.beat for point in result] == [point.beat for point in points]
    assert stats['enabled'] is False
