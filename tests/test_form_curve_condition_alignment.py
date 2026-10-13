from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from maichart.event_timing import EventPoint, EventTempo, seconds_at_beat
from maichart.form.curve import LOAD_CURVE_CHANNELS, build_load_curve, load_curve_bin_centers
from maichart.form.generate import _curve_condition


def profile(times, values, first=0., intensity=.5):
    return {'time_axis': 'audio_absolute_v2', 'curve_resolution_sec': 2.,
            'points': [{'time_sec': float(time+first), 'intensity': intensity} for time in times],
            'components': {name: values[:, i].tolist() for i, name in enumerate(LOAD_CURVE_CHANNELS)}}


@pytest.mark.parametrize('first', [0., .5, -1.82])
def test_decoder_curve_uses_training_bin_centers_and_selection(first):
    tempo = [EventTempo(Fraction(0), 120.)]
    song = SimpleNamespace(audio=SimpleNamespace(duration_sec=9.3), tempo=tempo, first=first)
    points = [EventPoint(Fraction(i), '1') for i in range(2, 16)]
    times, values = build_load_curve(points, tempo, first, 9.3)
    for lo, hi in [(Fraction(5), Fraction(13)), (Fraction(14), Fraction(18))]:
        span = SimpleNamespace(start=lo, end=hi)
        low = seconds_at_beat(lo, tempo, first)-first
        high = seconds_at_beat(hi, tempo, first)-first
        selected = (times >= low) & (times < high)
        actual, mask, local_times = _curve_condition(song, span, profile(times, values, first), strength=0.)
        np.testing.assert_allclose(local_times, times[selected]-low, atol=1e-6)
        np.testing.assert_allclose(actual, values[selected], atol=1e-6)
        assert mask.all()


def test_short_span_does_not_import_neighbor_curve_tokens():
    song = SimpleNamespace(audio=SimpleNamespace(duration_sec=8),
                           tempo=[EventTempo(Fraction(0), 120.)], first=0.)
    span = SimpleNamespace(start=Fraction(4), end=Fraction(5))
    times = np.asarray([1., 3., 5., 7.], dtype=np.float32)
    values = np.full((4, 4), .5, dtype=np.float32)
    actual, mask, local_times = _curve_condition(song, span, profile(times, values))
    assert actual.shape == mask.shape == (0, 4)
    assert local_times.numel() == 0


def test_intensity_changes_density_without_corrupting_type_proportions():
    song = SimpleNamespace(audio=SimpleNamespace(duration_sec=4),
                           tempo=[EventTempo(Fraction(0), 120.)], first=0.)
    span = SimpleNamespace(start=Fraction(0), end=Fraction(8))
    times = np.asarray([1., 3.], dtype=np.float32)
    values = np.asarray([[.4, .8, .2, .1], [.2, .7, .3, .2]], dtype=np.float32)
    actual, _, _ = _curve_condition(song, span, profile(times, values, intensity=1.),
                                    strength=1., span_limit=.5)
    np.testing.assert_allclose(actual[:, 0], values[:, 0]*1.5)
    np.testing.assert_array_equal(actual[:, 1:], values[:, 1:])


@pytest.mark.parametrize('duration', [160.7368707482993, 1.111111111, .01, 4.])
def test_editor_timestamps_keep_exact_eof_and_partial_bin_midpoint(duration):
    from maichart.generation.music_intensity import validate_intensity_profile
    times = load_curve_bin_centers(duration)
    assert times.dtype == np.float64
    assert times[-1] < duration
    assert times[-1] == pytest.approx((2*(len(times)-1)+duration)*.5)
    editor_times = np.unique(np.concatenate(([0.], times, [duration])))
    saved = {'schema': 'maichart-intensity-curve-v1', 'duration_sec': duration,
             'points': [{'time_sec': float(time), 'intensity': .5} for time in editor_times]}
    normalized = validate_intensity_profile(saved)
    assert normalized['points'][-1]['time_sec'] == duration
    assert all(a['time_sec'] < b['time_sec'] for a, b in zip(normalized['points'], normalized['points'][1:]))
