from fractions import Fraction

import numpy as np
import pytest

from maichart.composer.workload import (
    LEDGER_CHANNELS, LEDGER_DIM, WORKLOAD_DIM, build_workload_targets,
    causal_ledger, event_statistics,
)
from maichart.event_timing import EventPoint, EventTempo


def test_event_statistics_counts_compact_chords_touch_and_shared_slide_heads():
    assert event_statistics('12') == (2, 0., 0., 0.)
    assert event_statistics('1h[4:1]/C') == (2, 1., 0., 1.)
    assert event_statistics('Ch[4:1]') == (1, 1., 0., 1.)
    assert event_statistics('1-4[4:1]*-6[4:1]') == (1, 0., 1., 0.)
    assert event_statistics('1/1-4[4:1]') == (1, 0., 1., 0.)
    with pytest.raises(ValueError):
        event_statistics('unrecognized')


def test_workload_uses_audio_seconds_and_smooth_authored_rates():
    tempo = [EventTempo(Fraction(0), 120.)]
    points = [EventPoint(Fraction(0), '1'), EventPoint(Fraction(1), '2h[4:1]'),
              EventPoint(Fraction(2), '3-6[4:1]'), EventPoint(Fraction(3), 'C')]
    target = build_workload_targets(points, tempo, .25, [1., 2.9], 3.)
    assert target.shape == (2, WORKLOAD_DIM)
    assert target.dtype == np.float32
    # At one second, kernel weights are .25, .75, .75 and .25; full area is 1.
    np.testing.assert_allclose(target[0, :2], np.log1p([2., 2.]), atol=1e-6)
    np.testing.assert_allclose(target[0, 2:], [.375, .375, .125], atol=1e-6)
    np.testing.assert_array_equal(target[1], np.zeros(WORKLOAD_DIM))


def test_workload_clips_kernel_area_to_real_audio_boundary():
    tempo = [EventTempo(Fraction(0), 60.)]
    points = [EventPoint(Fraction(0), '1/2')]
    target = build_workload_targets(points, tempo, 0., [-.1, 0., 3., 4.], 4.)
    # Only half the triangle is inside the file at zero: event rate=2, heads=4.
    np.testing.assert_allclose(target[1, :2], np.log1p([2., 4.]), atol=1e-6)
    np.testing.assert_array_equal(target[[0, 2, 3]], np.zeros((3, WORKLOAD_DIM)))


def test_workload_uses_tempo_map_and_excludes_outside_file_heads():
    tempo = [EventTempo(Fraction(0), 120.), EventTempo(Fraction(2), 60.)]
    points = [EventPoint(Fraction(0), '1'), EventPoint(Fraction(2), '2/3'),
              EventPoint(Fraction(3), '4')]
    target = build_workload_targets(points, tempo, -.5, [.5, 1.5], 1.)
    # First event is before audio; third and second query are beyond its end.
    np.testing.assert_allclose(target[0, :2], np.log1p([4. / 3., 8. / 3.]), atol=1e-6)
    np.testing.assert_array_equal(target[1], np.zeros(WORKLOAD_DIM))


def test_ledger_only_counts_prefix_events_before_each_candidate():
    target = causal_ledger([0., .5, 2.], [1, 2, 3], [-1., .5, 1.5, 2.5, 7.])
    assert target.shape == (5, LEDGER_DIM)
    assert target.dtype == np.float32
    assert LEDGER_CHANNELS[:4] == ('event_rate_1s', 'head_rate_1s', 'event_rate_4s', 'head_rate_4s')
    np.testing.assert_array_equal(target[0], np.zeros(LEDGER_DIM))
    np.testing.assert_allclose(target[1, :4], [2., 3., .5, .75])
    np.testing.assert_allclose(target[2, :4], [0., 0., .5, .75])
    np.testing.assert_allclose(target[3, :4], [1., 3., .75, 1.5])
    np.testing.assert_allclose(target[3, 4:], [np.log1p(.5), .5])
    np.testing.assert_allclose(target[4, :4], np.zeros(4))
    np.testing.assert_allclose(target[4, 4:], [np.log1p(5.), .5])


def test_empty_ledger_does_not_invent_a_first_note():
    np.testing.assert_array_equal(causal_ledger([], [], [0., 2.]), np.zeros((2, LEDGER_DIM)))
    with pytest.raises(ValueError):
        causal_ledger([0.], [], [0.])
