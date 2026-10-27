from fractions import Fraction

import numpy as np
import pytest

from maichart.event_timing import EventTempo
from maichart.form import recurrence
from maichart.form.preparation_cache import PreparationCache
from maichart.form.timing import MetricSpan
from maichart.generation.event_audio import ACOUSTIC_DIM, EventAudio


def span(index, start, end):
    return MetricSpan(index, Fraction(start), Fraction(end), 4, 4, (), Fraction(4), (), False, 1., 'test')


def test_integrated_sampling_captures_attack_away_from_bin_center():
    times = np.arange(201, dtype=np.float32) / 200
    features = np.zeros((len(times), ACOUSTIC_DIM), np.float32)
    features[(times >= .095) & (times <= .105), 92] = 1
    audio = EventAudio(features, times, 1., 120.)
    tempo = [EventTempo(Fraction(0), 120.)]
    interval = span(0, 0, 2)
    integrated = recurrence._integrated_samples(audio, interval, tempo, 0., 4)
    center = recurrence._sample(audio, np.array([.125, .375, .625, .875]))
    assert center[:, 92].max() == 0
    assert integrated[0, 92] > 0
    assert integrated[1:, 92].max() == 0


def test_integrated_native_path_uses_cached_spectra_without_fft(monkeypatch):
    import maichart.form.native_dynamics as dynamics
    import maichart.form.rational_audio as rational
    times = np.array([0., .5, .99, 1.], np.float32)
    features = np.zeros((4, ACOUSTIC_DIM), np.float32)
    features[:, 0] = 2 * times
    audio = EventAudio(features, times, 1., 120., metadata={'profile': 'rational_audio_queries_v1'})
    monkeypatch.setattr(dynamics, 'native_dynamics', lambda _audio, stamps: np.zeros((len(stamps), 3)))
    def forbidden(*args, **kwargs):
        raise AssertionError('integrated cached spectrum must not recompute acoustic FFT')
    monkeypatch.setattr(rational, 'acoustic_at_times', forbidden)
    monkeypatch.setattr(np.fft, 'rfft', forbidden)
    values = recurrence._integrated_samples(audio, span(0, 0, 2),
                                            [EventTempo(Fraction(0), 120.)], 0., 2)
    np.testing.assert_allclose(values[:, 0], [.5, 1.5], atol=.02)


def test_adjacent_span_texture_cannot_contaminate_current_span():
    times = np.arange(101, dtype=np.float32) / 50
    features = np.zeros((len(times), ACOUSTIC_DIM), np.float32)
    features[times < 1., 0] = 2.
    features[times >= 1., 0] = 100.
    features[times < 1., 92] = .25
    features[times >= 1., 92] = 1.
    audio = EventAudio(features, times, 2., 120.)
    result = recurrence._integrated_samples(audio, span(0, 0, 2),
                                             [EventTempo(Fraction(0), 120.)], 0., 24)
    np.testing.assert_allclose(result[:, 0], 2.)
    np.testing.assert_allclose(result[:, 92], .25)


def test_integrated_filters_incompatible_beat_duration_pairs(monkeypatch):
    monkeypatch.setattr(recurrence, '_describe', lambda samples, bins, **kw:
                        (np.ones(bins * recurrence.DESCRIPTOR_CHANNELS, np.float32) /
                         np.sqrt(bins * recurrence.DESCRIPTOR_CHANNELS), True))
    audio = EventAudio(np.zeros((401, ACOUSTIC_DIM), np.float32),
                       np.arange(401, dtype=np.float32) / 50, 8., 120.)
    spans = [span(0, 0, 4), span(1, 4, 7), span(2, 7, 9)]
    result = recurrence.analyze_recurrence(audio, spans, [EventTempo(Fraction(0), 120.)],
                                            0., min_separation=0, min_similarity=0.)
    pairs = {(link.source, link.target) for link in result.links}
    assert (0, 1) in pairs  # inclusive lower bound 3/4
    assert (0, 2) not in pairs
    assert result.audit['rejected_duration_ratios'] >= 1


def test_preparation_identity_keeps_legacy_code_independent_and_tracks_parameters(tmp_path, monkeypatch):
    chart, audio = tmp_path/'chart.txt', tmp_path/'audio.wav'
    chart.write_bytes(b'chart'); audio.write_bytes(b'audio')
    cache = PreparationCache(tmp_path/'cache')
    options = dict(meters=[], max_tokens=512, audio_profile='legacy_50hz',
                   max_region_seconds=30., reconstruction=False)
    def identity(opts):
        return cache.identity(chart, audio, 5, '13', None, opts)
    legacy_key, legacy = identity(options)
    assert 'recurrence_parameters' not in legacy['settings']
    cache.code_hash = 'another implementation revision'
    assert identity(options)[0] == legacy_key
    integrated = {**options, 'recurrence_descriptor_policy': recurrence.RECURRENCE_DESCRIPTOR_POLICY}
    key, metadata = identity(integrated)
    assert key != legacy_key
    assert metadata['settings']['recurrence_parameters']['spectral_sample_seconds'] == recurrence.SPECTRAL_SAMPLE_SECONDS
    monkeypatch.setattr(recurrence, 'SPECTRAL_SAMPLE_SECONDS', recurrence.SPECTRAL_SAMPLE_SECONDS * 2)
    assert identity(integrated)[0] != key
    assert identity(options)[0] == legacy_key
