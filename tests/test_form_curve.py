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
    assert times.tolist() == [1.0, 3.0, 4.25]
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
