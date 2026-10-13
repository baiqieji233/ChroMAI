from fractions import Fraction

import numpy as np
import pytest

from maichart.event_timing import EventPoint, EventTempo
from maichart.form.curve import build_load_curve, resample_load_curve


def test_build_load_curve_uses_audio_relative_bins_and_four_channels():
    tempo = [EventTempo(Fraction(0), 120.0)]
    points = [
        EventPoint(Fraction(0), "1"),
        EventPoint(Fraction(1), "2h[1:2]"),
        EventPoint(Fraction(3), "3-5[1:2]"),
    ]
    times, values = build_load_curve(points, tempo, first=-0.5, duration_sec=4.5)
    assert times.tolist() == [1.0, 3.0, 4.5]
    assert values.shape == (3, 4)
    assert np.all((values >= 0) & (values <= 1))
    assert values[0, 0] > 0
    assert values[0, 2] > 0


def test_resample_load_curve_interpolates_and_rejects_bad_shape():
    times = np.asarray([1.0, 3.0], dtype=np.float32)
    values = np.asarray([[0.0, 0.1, 0.2, 0.3], [1.0, 0.4, 0.5, 0.6]], dtype=np.float32)
    output = resample_load_curve(times, values, [0.0, 2.0, 4.0])
    assert output.shape == (3, 4)
    assert np.allclose(output[:, 0], [0.0, 0.5, 1.0])
    with pytest.raises(ValueError, match="shape"):
        resample_load_curve(times, values[:, :3], [1.0])


def test_load_curve_preserves_exact_bins_and_converts_each_event_once(monkeypatch):
    from maichart.form import curve
    from maichart.event_timing import seconds_at_beat
    tempo = [EventTempo(Fraction(0), 120.0), EventTempo(Fraction(4), 60.0)]
    points = [EventPoint(Fraction(0), "1/2"), EventPoint(Fraction(4), "3h[4:1]"),
              EventPoint(Fraction(6), "C"), EventPoint(Fraction(9), "4")]
    calls = []
    def timed(beat, *args):
        calls.append(beat)
        return seconds_at_beat(beat, *args)
    monkeypatch.setattr(curve, "seconds_at_beat", timed)
    times, values = curve.build_load_curve(points, tempo, first=-0.5, duration_sec=6.5)
    assert calls == [point.beat for point in points]
    np.testing.assert_array_equal(times, [1.0, 3.0, 5.0, 6.5])
    np.testing.assert_array_equal(values, [[0, 0, 0, 0], [0.0625, 0, 1, 0],
                                          [0.0625, 0, 0, 1], [0, 0, 0, 0]])


@pytest.mark.parametrize(('first', 'duration', 'beats', 'expected_times', 'active_bins'), [
    (-1.0, 4.0, [2, 4, 5], [1.0, 3.0, 4.5], [1, 2]),
    (1.0, 5.0, [0, 3, 4], [1.0, 3.0], [0, 1]),
])
def test_load_curve_covers_audio_tail_and_excludes_events_outside_audio(
        first, duration, beats, expected_times, active_bins):
    tempo = [EventTempo(Fraction(0), 60.0)]
    points = [EventPoint(Fraction(beat), '1') for beat in beats]
    times, values = build_load_curve(points, tempo, first=first, duration_sec=duration)
    np.testing.assert_array_equal(times, expected_times)
    assert np.flatnonzero(values[:, 0]).tolist() == active_bins
