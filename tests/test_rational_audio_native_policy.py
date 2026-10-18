import numpy as np
from fractions import Fraction as F
from types import SimpleNamespace

from maichart.event_timing import EventTempo
from maichart.form import rational_audio
from maichart.form.recurrence import _sample


def _audio(policy):
    features = np.zeros((2, rational_audio.ACOUSTIC_DIM), dtype=np.float32)
    features[0, 80] = 1.0
    features[1, 81] = 1.0
    features[:, 93] = 1.0
    metadata = {
        "profile": rational_audio.PROFILE,
        "audio_sha256": "fixture",
        "normalization_scale": [1.0] * 95,
        "sample_rate": 100,
    }
    return rational_audio.RationalAudio(
        features, np.asarray([0.0, 10.0]), 10.0, 120.0, metadata,
        "fixture.wav", feature_policy=policy,
    )


def test_native_policy_does_not_interpolate_sparse_attack_rows(monkeypatch):
    monkeypatch.setattr(
        rational_audio, "waveform",
        lambda *_args, **_kwargs: (np.zeros(1001, dtype=np.float32), 100),
    )
    monkeypatch.setattr(
        rational_audio, "query_spectra",
        lambda _signal, _sr, seconds: np.zeros((len(seconds), rational_audio.ACOUSTIC_DIM), dtype=np.float32),
    )
    legacy = _audio(rational_audio.LEGACY_FEATURE_POLICY)
    native = _audio(rational_audio.NATIVE_FEATURE_POLICY)
    legacy.features[0, 95] = 1.0
    legacy.features[1, 95] = 0.0
    native.features[0, 95] = 1.0
    native.features[1, 95] = 0.0

    old = rational_audio.acoustic_at_times(legacy, [5.0])
    measured = rational_audio.acoustic_at_times(native, [5.0])
    np.testing.assert_allclose(old[0, 95], 0.5)
    np.testing.assert_allclose(measured[0, 95], 0.0)


def test_recurrence_uses_native_policy_without_changing_legacy(monkeypatch):
    monkeypatch.setattr(
        rational_audio, "waveform",
        lambda *_args, **_kwargs: (np.zeros(1001, dtype=np.float32), 100),
    )
    monkeypatch.setattr(
        rational_audio, "query_spectra",
        lambda _signal, _sr, seconds: np.zeros((len(seconds), rational_audio.ACOUSTIC_DIM), dtype=np.float32),
    )
    legacy = _audio(rational_audio.LEGACY_FEATURE_POLICY)
    native = _audio(rational_audio.NATIVE_FEATURE_POLICY)
    np.testing.assert_allclose(_sample(legacy, np.asarray([5.0]))[0, 80:82], [0.5, 0.5])
    np.testing.assert_allclose(_sample(native, np.asarray([5.0]))[0, 80:82], [0.0, 0.0])


def test_native_policy_keeps_silence_and_outside_file_zero(monkeypatch):
    audio = _audio(rational_audio.NATIVE_FEATURE_POLICY)
    monkeypatch.setattr(
        rational_audio, "waveform",
        lambda *_args, **_kwargs: (np.zeros(1001, dtype=np.float32), 100),
    )
    monkeypatch.setattr(
        rational_audio, "query_spectra",
        lambda _signal, _sr, seconds: np.zeros((len(seconds), rational_audio.ACOUSTIC_DIM), dtype=np.float32),
    )

    values = rational_audio.acoustic_at_times(audio, [-1.0, 0.0, 5.0, 10.0, 11.0])

    assert np.isfinite(values).all()
    assert not values.any()


def test_native_policy_derives_columns_after_the_shared_normalization(monkeypatch):
    audio = _audio(rational_audio.NATIVE_FEATURE_POLICY)
    audio.metadata["normalization_scale"] = [2.0] * 95
    audio.features[:, 93] = [1.0, 3.0]
    monkeypatch.setattr(
        rational_audio, "waveform",
        lambda *_args, **_kwargs: (np.ones(1001, dtype=np.float32), 100),
    )

    def spectra(_signal, _sr, seconds):
        seconds = np.asarray(seconds, dtype=np.float32)
        values = np.zeros((len(seconds), rational_audio.ACOUSTIC_DIM), dtype=np.float32)
        values[:, 80] = seconds
        values[:, 93] = seconds
        return values

    monkeypatch.setattr(rational_audio, "query_spectra", spectra)
    values = rational_audio.acoustic_at_times(audio, [1.0])

    # Center RMS is 1/2 after the shared scale, then divided by the stored
    # normalized-anchor median of 2.  The slope and chroma delta use those
    # same normalized neighboring measurements.
    np.testing.assert_allclose(values[0, 95], 0.25, atol=1e-6)
    np.testing.assert_allclose(values[0, 96], 0.5, atol=1e-6)
    np.testing.assert_allclose(values[0, 97], 0.01, atol=1e-6)


def test_native_policy_gets_an_independent_query_view(monkeypatch):
    audio = _audio(rational_audio.LEGACY_FEATURE_POLICY)
    audio.duration_sec = 10.0
    monkeypatch.setattr(
        rational_audio, "waveform",
        lambda *_args, **_kwargs: (np.zeros(1001, dtype=np.float32), 100),
    )
    monkeypatch.setattr(
        rational_audio, "query_spectra",
        lambda _signal, _sr, seconds: np.zeros((len(seconds), rational_audio.ACOUSTIC_DIM), dtype=np.float32),
    )
    span = SimpleNamespace(start=F(0), end=F(1))
    song = SimpleNamespace(audio=audio, first=0.0, tempo=[EventTempo(F(0), 60.0)],
                           meters=[], spans=[span])

    rational_audio.audio_queries(song, F(0), F(1))
    audio.feature_policy = rational_audio.NATIVE_FEATURE_POLICY
    rational_audio.audio_queries(song, F(0), F(1))

    assert len(audio._views) == 2


def test_native_queries_read_a_bounded_waveform_window(monkeypatch, tmp_path):
    from maichart.form import waveform_cache
    audio = _audio(rational_audio.NATIVE_FEATURE_POLICY)
    audio.duration_sec = 30.
    source = tmp_path / 'track.wav'
    source.write_bytes(b'fixture')
    audio.source = str(source)
    audio.metadata['audio_sha256'] = None
    calls = []
    monkeypatch.setattr(waveform_cache, 'mapped_waveform',
        lambda *args: (np.zeros(30000,dtype=np.float32),1000))
    def spectra(signal, sr, times, *, sample_offset, component='all'):
        calls.append((len(signal),sample_offset,times.copy()))
        return np.ones((len(times),rational_audio.ACOUSTIC_DIM),dtype=np.float32)
    monkeypatch.setattr(
        rational_audio, "query_spectra",
        spectra,
    )
    values = rational_audio.acoustic_at_times(audio, [10.1, 10.4], derived_policy='native_measured_v2')
    assert calls and calls[0][0] < 2500
    assert calls[0][1] > 0
    assert values.shape == (2, rational_audio.ACOUSTIC_DIM)
    assert values.any() and np.isfinite(values).all()


def test_query_views_are_process_bounded_and_clearable(monkeypatch):
    audio = _audio(rational_audio.LEGACY_FEATURE_POLICY)
    span = SimpleNamespace(start=F(0), end=F(1))
    song = SimpleNamespace(audio=audio, first=0.0, tempo=[EventTempo(F(0), 60.0)],
                           meters=[], spans=[span])
    calls = []
    monkeypatch.setattr(
        rational_audio, "waveform",
        lambda *_args, **_kwargs: (np.zeros(1001, dtype=np.float32), 100),
    )

    def spectra(_signal, _sr, seconds):
        calls.append(len(seconds))
        return np.zeros((len(seconds), rational_audio.ACOUSTIC_DIM), dtype=np.float32)

    monkeypatch.setattr(rational_audio, "query_spectra", spectra)
    audio._views.clear()
    rational_audio.audio_queries(song, F(0), F(1))
    assert calls
    calls.clear()
    rational_audio.audio_queries(song, F(0), F(1))
    assert not calls, "a repeated query should use the shared process cache"

    # Tiny budgets leave only a lightweight local key and force a fresh
    # measurement instead of retaining a full query tensor.
    audio._views.clear()
    monkeypatch.setattr(rational_audio, "QUERY_CACHE_MAX_BYTES", 1)
    rational_audio.audio_queries(song, F(0), F(1))
    calls.clear()
    rational_audio.audio_queries(song, F(0), F(1))
    assert calls

