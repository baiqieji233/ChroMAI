from fractions import Fraction as F
from types import SimpleNamespace

import numpy as np
import pytest

from maichart.event_timing import EventTempo
from maichart.form import rational_audio as audio_module
from maichart.form.recurrence import analyze_recurrence
from maichart.form.timing import MetricSpan
from maichart.generation.event_audio import EventAudio


def _span(index, start, end):
    return MetricSpan(index, F(start), F(end), 4, 4, (), F(4), (), False, 1., 'test')


def _song(profile, *, name, duration=16., cache_dir=None):
    times = np.arange(31, int(duration*1000)-31, 73, dtype=np.float64)/1000
    features = np.zeros((len(times), 98), dtype=np.float32)
    features[:, 80] = 1.
    features[:, 92] = np.where(times < duration/2, 10., .01)
    features[:, 93] = np.where(times < duration/2, 1., .1)
    meta = dict(profile=audio_module.PROFILE, audio_sha256=name,
        normalization_scale=[1.]*95, sample_rate=1000)
    audio = audio_module.RationalAudio(features, times, duration, 120., meta, 'virtual.wav',
        str(cache_dir) if cache_dir is not None else None,
        feature_policy=audio_module.NATIVE_FEATURE_POLICY, evidence_profile=profile)
    audio._views.clear()
    return SimpleNamespace(audio=audio, tempo=[EventTempo(F(0), 120)], first=0., meters=[],
        spans=[_span(0, 0, int(duration*2))])


@pytest.fixture
def measured_queries(monkeypatch):
    def measure(audio, times, **kwargs):
        return np.column_stack([np.interp(times, audio.times, audio.features[:, column])
                                for column in range(98)]).astype(np.float32)
    monkeypatch.setattr(audio_module, 'acoustic_at_times', measure)


def test_query_budget_preserves_quiet_later_sections_instead_of_only_strong_early_attacks(measured_queries):
    balanced = _song('balanced_evidence_v1', name='balanced-weak-late')
    legacy = _song('legacy', name='legacy-weak-late')
    _, balanced_times, balanced_beats = audio_module.audio_queries(balanced, F(0), F(32), max_nodes=24)
    _, legacy_times, legacy_beats = audio_module.audio_queries(legacy, F(0), F(32), max_nodes=24)
    later_balanced = [time for time, beat in zip(balanced_times, balanced_beats)
                      if time >= 8 and beat.denominator > 1]
    later_legacy = [time for time, beat in zip(legacy_times, legacy_beats)
                    if time >= 8 and beat.denominator > 1]
    assert len(balanced_beats) <= 24
    assert len(later_balanced) >= 4
    assert later_legacy == []


def test_query_budget_retains_low_attack_harmonic_changes(measured_queries):
    song = _song('balanced_evidence_v1', name='balanced-harmonic', duration=8.)
    index = int(np.searchsorted(song.audio.times, 6.1))
    song.audio.features[index:, 80] = 0.
    song.audio.features[index:, 84] = 1.
    _, times, beats = audio_module.audio_queries(song, F(0), F(16), max_nodes=32)
    assert np.any(np.isclose(song.audio.times[index], times, rtol=0., atol=1e-12))
    assert len(beats) <= 32
    assert song.audio.features[index, 92] == pytest.approx(.01)


def test_balanced_queries_keep_tempo_and_meter_anchors_and_transient_slots(measured_queries):
    from maichart.form.timing import MeterChange
    song = _song('balanced_evidence_v1', name='balanced-many-anchors', duration=80.)
    song.tempo.append(EventTempo(F(7, 3), 120.))
    song.meters = [MeterChange(F(0), 3, 4), MeterChange(F(11, 2), 4, 4)]
    _, _, beats = audio_module.audio_queries(song, F(0), F(160), max_nodes=32)
    assert {F(0), F(7, 3), F(11, 2)} <= set(beats)
    assert len(beats) == 32
    assert sum(beat.denominator > 1 for beat in beats) >= 12


def test_balanced_raw_attack_budget_preserves_weak_late_peaks(monkeypatch):
    monkeypatch.setattr(audio_module, 'MAX_EVIDENCE_NODES', 64)
    sr = 4000
    signal = np.zeros(8*sr, dtype=np.float32)
    for center in range(40, len(signal)-40, 52):
        amplitude = 1. if center < len(signal)//2 else .02
        signal[center:center+12] = amplitude*np.hanning(12)
    balanced = audio_module.attack_samples(signal, sr, evidence_profile='balanced_evidence_v1')
    legacy = audio_module.attack_samples(signal, sr)
    assert len(balanced) == 64
    assert (balanced >= len(signal)//2).sum() >= 24
    assert (legacy >= len(signal)//2).sum() == 0


def test_legacy_raw_peak_selection_is_unchanged(monkeypatch):
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import find_peaks
    monkeypatch.setattr(audio_module, 'MAX_EVIDENCE_NODES', 32)
    sr = 1000
    signal = np.random.default_rng(11).normal(size=8000).astype(np.float32)
    energy = uniform_filter1d(signal.astype(np.float64)**2, max(1, round(sr*.003)))
    shift = max(1, round(sr*.004))
    previous = np.roll(energy, shift)
    previous[:shift] = 0
    novelty = np.maximum(energy-previous, 0)
    height = max(float(np.percentile(novelty, 85)), float(novelty.max())*.002, 1e-12)
    expected, props = find_peaks(novelty, height=height, distance=max(1, round(sr*.006)))
    chosen = np.argsort(props['peak_heights'], kind='stable')[-32:]
    expected = np.sort(expected[chosen]).astype(np.int64)
    assert np.array_equal(audio_module.attack_samples(signal, sr), expected)


def test_spectral_proposals_find_harmonic_change_even_without_an_attack(monkeypatch):
    def spectra(signal, sr, times):
        result = np.zeros((len(times), 98), dtype=np.float32)
        result[:, 80] = np.asarray(times) < .5
        result[:, 84] = np.asarray(times) >= .5
        result[:, 93] = 1.
        return result
    monkeypatch.setattr(audio_module, 'query_spectra', spectra)
    samples, features = audio_module._spectral_evidence(np.ones(1000, dtype=np.float32), 1000)
    assert np.any(np.abs(samples/1000-.5) <= .05)
    assert np.all(features[:, 92] == 0.)


def test_raw_cache_profiles_are_separate_and_reused_without_rebuilding(tmp_path, monkeypatch):
    import soundfile as sf
    sr = 22050
    times = np.arange(sr)/sr
    signal = (.1*np.sin(2*np.pi*np.where(times < .5, 220., 440.)*times)).astype(np.float32)
    source = tmp_path/'song.wav'
    sf.write(source, signal, sr)
    cache = tmp_path/'cache'
    monkeypatch.setattr(audio_module, '_tempo_hint', lambda signal, sr: 120.)
    legacy = audio_module.extract_rational_audio(source, cache)
    balanced = audio_module.extract_rational_audio(source, cache, evidence_profile='balanced_evidence_v1')
    legacy_path = audio_module.cache_path(source, cache)
    balanced_path = audio_module.cache_path(source, cache, evidence_profile='balanced_evidence_v1')
    assert legacy_path.name == legacy.metadata['audio_sha256']+'.npz'
    assert balanced_path.name == legacy.metadata['audio_sha256']+'.balanced_evidence_v1.npz'
    assert legacy_path.is_file() and balanced_path.is_file()
    assert balanced.evidence_profile == balanced.metadata['evidence_profile'] == 'balanced_evidence_v1'
    assert legacy.evidence_profile == 'legacy' and 'evidence_profile' not in legacy.metadata
    assert legacy_path.resolve() not in audio_module._cache_candidates(
        source, cache, evidence_profile='balanced_evidence_v1')
    monkeypatch.setattr('maichart.form.waveform_cache.mapped_waveform',
        lambda *args: pytest.fail('A valid raw cache must avoid decoding the waveform'))
    again = audio_module.extract_rational_audio(source, cache, evidence_profile='balanced_evidence_v1')
    assert np.array_equal(again.features, balanced.features)
    assert again.evidence_profile == 'balanced_evidence_v1'


def test_query_cache_identity_separates_profiles_with_same_audio_hash(tmp_path, measured_queries):
    legacy = _song('legacy', name='shared-hash-query', cache_dir=tmp_path)
    balanced = _song('balanced_evidence_v1', name='shared-hash-query', cache_dir=tmp_path)
    _, legacy_times, _ = audio_module.audio_queries(legacy, F(0), F(32), max_nodes=24)
    _, balanced_times, _ = audio_module.audio_queries(balanced, F(0), F(32), max_nodes=24)
    assert not np.array_equal(legacy_times, balanced_times)
    assert len(list((tmp_path/audio_module.PROFILE).glob('*.queries.npz'))) == 2


def test_layout_dynamics_distinguish_quiet_and_loud_regions_without_changing_match_descriptors():
    stamps = np.arange(601, dtype=np.float32)/100
    values = np.zeros((len(stamps), 98), dtype=np.float32)
    phase = stamps % 2
    scale = np.where((stamps >= 2) & (stamps < 4), .1, 1.)
    values[:, 80] = phase < 1
    values[:, 84] = phase >= 1
    values[:, 92] = (phase < .5)*scale
    values[:, 93] = scale
    values[:, 94] = .3*scale
    audio = EventAudio(values, stamps, 6., 120.)
    spans = [_span(0, 0, 4), _span(1, 4, 8), _span(2, 8, 12)]
    tempo = [EventTempo(F(0), 120)]
    legacy = analyze_recurrence(audio, spans, tempo, 0.)
    enhanced = analyze_recurrence(audio, spans, tempo, 0., quality_profile='musical_evidence_v1')
    assert legacy.layout_descriptors is None
    assert enhanced.layout_descriptors.shape == (3, 512)
    assert np.array_equal(enhanced.descriptors, legacy.descriptors)
    assert enhanced.links == legacy.links
    assert enhanced.layout_descriptors[0, -8] > 5*enhanced.layout_descriptors[1, -8]
    assert enhanced.layout_descriptors[0, -6] > 5*enhanced.layout_descriptors[1, -6]
    assert np.array_equal(enhanced.layout_descriptors[:, :504], enhanced.descriptors)
