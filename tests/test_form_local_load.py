from fractions import Fraction

import numpy as np
import pytest

from maichart.form.local_load import LocalChordGuard, MAX_LOCAL_PENALTY
from maichart.form.local_load import local_load_statistics
from maichart.event_timing import EventPoint, EventTempo


def cues(seconds, curve_factors=None):
    curve_factors = curve_factors or [1.0] * len(seconds)
    return [
        {'beat': str(index), 'second': second, 'curve_factor': factor}
        for index, (second, factor) in enumerate(zip(seconds, curve_factors))
    ]


def guard(seconds, *, factors=None, target=0.2, strength=1.0):
    return LocalChordGuard(cues(seconds, factors), target_chord_ratio=target, strength=strength)


def test_zero_strength_returns_zero_penalty_for_scalar_and_candidate_vector():
    guide = guard([0.0], strength=0)
    event = guide.event(Fraction(0))
    assert event.penalty(2) == 0
    np.testing.assert_array_equal(event.penalty([1, 2, 8]), [0, 0, 0])
    assert guide.report()['applied'] is False


def test_initial_intentional_doubles_pass_and_singles_never_get_catchup_reward():
    guide = guard([0.0, 0.1, 0.2, 0.3, 0.4])
    event = guide.event(0)
    assert event.penalty(2) == 0
    guide.commit([2, 0, 0, 0, 0], 0)
    assert guide.event(1).penalty(2) == 0
    guide.commit([1, 0, 0, 0, 0], 1)
    assert guide.event(2).penalty(1) == 0
    assert guide.event(2).penalty(2) >= 0


def test_local_double_burst_builds_pressure_and_cools_after_window_gap():
    guide = guard([0.0, 0.1, 0.2, 0.3, 0.4, 2.5])
    for beat in range(4):
        guide.commit([2, 0, 0, 0, 0], beat)
    assert guide.report()['history_events'] == 4
    hot = guide.event(4)
    assert hot.penalty(2) > 0
    assert hot.penalty(3) > hot.penalty(2)
    cool = guide.event(5)
    assert cool.penalty(2) == 0
    report = guide.report()
    assert report['pressure_events'] > 0
    assert report['max_pressure_run'] > 0
    assert report['recent_events'] == 0
    assert report['history_events'] == 0


def test_touch_and_touch_hold_clusters_do_not_count_as_physical_chords():
    guide = guard([0.0, 0.1, 0.2])
    guide.commit([0, 0, 0, 2, 0], 0)
    guide.commit([0, 0, 0, 1, 1], 1)
    event = guide.event(2)
    assert event.penalty(0) == 0
    report = guide.report()
    assert report['chord_events'] == 0
    assert report['button_heads'] == 0
    assert report['max_button_heads_per_event'] == 0
    assert report['touch_heads_excluded'] is True


def test_high_relative_curve_factor_relaxes_local_chord_pressure():
    guide = guard([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], factors=[1, 1, 1, 1, 1, 1.5])
    for beat in range(4):
        guide.commit([2, 0, 0, 0, 0], beat)
    ordinary = guide.event(4)
    relaxed = guide.event(5)
    assert ordinary.penalty(2) > relaxed.penalty(2) > 0
    assert ordinary.pressure == pytest.approx(0.8)
    assert relaxed.pressure == pytest.approx(0.4)


def test_penalty_is_bounded_and_diagnostics_are_capped():
    event_cues = cues([index * 0.05 for index in range(150)])
    guide = LocalChordGuard(event_cues, target_chord_ratio=0, strength=3)
    for beat in range(20):
        guide.commit([8, 0, 0, 0, 0], beat)
    event = guide.event(20)
    values = event.penalty([1, 2, 3, 100])
    assert np.all(values >= 0)
    assert np.all(values <= MAX_LOCAL_PENALTY)
    report = guide.report()
    assert len(report['diagnostics']) <= 128
    assert report['penalty_bound'] == MAX_LOCAL_PENALTY
    assert report['max_button_heads_per_event'] == 8


@pytest.mark.parametrize('target', [-0.1, 1.1, float('nan')])
def test_invalid_target_chord_ratio_is_rejected(target):
    with pytest.raises(ValueError, match='target_chord_ratio'):
        LocalChordGuard(cues([0]), target_chord_ratio=target)


def test_local_load_statistics_separate_head_peaks_from_event_rate_and_merge_bursts():
    points = []
    for index in list(range(8)) + list(range(16, 24)):
        points.append(EventPoint(Fraction(index, 2), '1/2'))
    points.extend([EventPoint(Fraction(1, 4), 'E1/E2')])
    points.sort(key=lambda point: point.beat)
    stats = local_load_statistics(points, [EventTempo(Fraction(0), 120)], 0)
    assert stats['enabled'] is True
    assert stats['events'] == 17
    assert stats['all_heads'] == 34
    assert stats['button_heads'] == 32
    assert stats['physical_chords'] == 16
    assert stats['peak_all_head_nps'] == 10
    assert stats['peak_button_head_nps'] == 8
    assert stats['peak_occupied_events'] == 5
    assert stats['peak_occupied_events_nps'] == 5
    assert len(stats['burst_intervals']) == 2
    assert len(stats['burst_intervals']) <= stats['burst_interval_limit'] == 16
    assert stats['burst_interval_count'] == 2
    assert stats['burst_intervals_truncated'] is False
    assert all(item['window_seconds'] == 2.0 for item in stats['burst_intervals'])
    assert all(item['events'] >= 8 for item in stats['burst_intervals'])
    assert all(item['button_chord_share'] > stats['burst_threshold'] for item in stats['burst_intervals'])
    assert all({'button_heads', 'chords', 'button_head_nps', 'occupied_event_nps'} <= item.keys()
               for item in stats['burst_intervals'])


def test_burst_statistics_require_eight_events_and_use_strict_target_threshold():
    tempo = [EventTempo(Fraction(0), 120)]
    seven = [EventPoint(Fraction(index, 2), '1/2') for index in range(7)]
    assert local_load_statistics(seven, tempo, 0)['burst_intervals'] == []

    points = [EventPoint(Fraction(index, 2), '1/2' if index < 6 else '1') for index in range(8)]
    stats = local_load_statistics(points, tempo, 0, target_chord_ratio=0.5)
    assert stats['burst_threshold'] == 0.75
    assert stats['burst_intervals'] == []


def test_touch_only_cluster_counts_all_heads_but_not_button_chords():
    points = [EventPoint(Fraction(index, 2), 'E1/E2') for index in range(8)]
    stats = local_load_statistics(points, [EventTempo(Fraction(0), 120)], 0)
    assert stats['all_heads'] == 16
    assert stats['button_heads'] == 0
    assert stats['physical_chords'] == 0
    assert stats['peak_all_head_nps'] == 8
    assert stats['peak_button_head_nps'] == 0
    assert stats['burst_intervals'] == []


def test_one_second_peak_metrics_track_independent_maxima():
    points = [EventPoint(Fraction(index, 5), '1/2') for index in range(3)]
    points.extend(EventPoint(Fraction(4, 1) + Fraction(index, 5), '1') for index in range(5))
    stats = local_load_statistics(points, [EventTempo(Fraction(0), 120)], 0)
    assert stats['peak_all_head_nps'] == 6
    assert stats['peak_button_head_nps'] == 6
    assert stats['peak_occupied_events'] == 5
    assert stats['peak_occupied_events_nps'] == 5


def test_burst_episode_summary_reports_truncation_after_sixteen():
    points = [
        EventPoint(Fraction(cluster * 6, 1) + Fraction(index, 2), '1/2')
        for cluster in range(17) for index in range(8)
    ]
    stats = local_load_statistics(points, [EventTempo(Fraction(0), 120)], 0)
    assert stats['burst_interval_count'] == 17
    assert len(stats['burst_intervals']) == 16
    assert stats['burst_intervals_truncated'] is True
