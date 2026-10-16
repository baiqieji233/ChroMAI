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


def test_near_grid_cleanup_cannot_move_last_event_into_next_compute_span():
    points = [EventPoint(F(2), '1'), EventPoint(F(287, 96), '2')]
    result, stats = simplify_rational_events(
        points, [EventTempo(F(0), 185)], 0,
        tolerance_ms=8, start_beat=F(0), end_beat=F(3))
    assert result == points
    assert all(point.beat < F(3) for point in result)
    assert stats['changed'] == 0


def test_cleanup_preserves_committed_prefix_while_using_it_as_neighbor_evidence():
    points = [EventPoint(F(35), '1'), EventPoint(F(3313, 94), '2'),
              EventPoint(F(1211, 34), '3'), EventPoint(F(143, 4), '4')]
    result, stats = simplify_rational_events(
        points, [EventTempo(F(0), 130)], 0,
        tolerance_ms=8, immutable_prefix=2,
        start_beat=F(71, 2), end_beat=F(36))
    assert result[:2] == points[:2]
    assert result[2] == EventPoint(F(285, 8), '3')
    assert stats['changed_indices'] == [2]
    assert stats['changed'] == 1


def test_exact_common_position_is_not_reported_as_a_change():
    points = [EventPoint(F(0), '1'), EventPoint(F(1, 24), '2')]
    result, stats = simplify_rational_events(
        points, [EventTempo(F(0), 120)], 0,
        divisions=(96,), tolerance_ms=8, end_beat=F(1))
    assert result == points
    assert stats['changed'] == 0
    assert stats['changed_indices'] == []
    assert stats['maximum_shift_ms'] == 0


def test_cleanup_cannot_snap_before_mutable_span_start():
    points = [EventPoint(F(0), '1'), EventPoint(F(97, 96), '2')]
    result, stats = simplify_rational_events(
        points, [EventTempo(F(0), 185)], 0,
        start_beat=F(97, 96), end_beat=F(2), immutable_prefix=1)
    assert result == points
    assert stats['changed'] == 0
