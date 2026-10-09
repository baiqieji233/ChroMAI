from fractions import Fraction

import numpy as np
import pytest

from maichart.generation.event_audio import (
    ACOUSTIC_DIM,
    FEATURE_KEYS,
    FEATURE_VERSION,
    GLOBAL_MEMORY_FRAMES,
    INPUT_DIM,
    EventAudio,
    audio_segment_tensor,
    compress_event_audio,
    extract_event_audio,
    load_event_audio,
    load_event_audio_npz,
    save_event_audio_npz,
)


def test_rhythm_windows_match_reference_reduction():
    from maichart.generation.event_audio import _beat_normalized_rhythm

    rng = np.random.default_rng(7)
    acoustic = np.zeros((41, ACOUSTIC_DIM), dtype=np.float32)
    acoustic[:, 92] = rng.random(41)
    acoustic[:, 93] = rng.random(41)
    acoustic[:, 94] = rng.random(41)
    times = np.arange(41, dtype=np.float64) / 50.0
    bpms = np.linspace(90.0, 210.0, len(times))
    got = _beat_normalized_rhythm(acoustic, times, np.zeros_like(times), bpms)
    onset, rms, flux = acoustic[:, 92], acoustic[:, 93], acoustic[:, 94]
    expected = []
    for index, stamp in enumerate(times):
        local = max(60.0 / max(bpms[index], 1e-6), 1.0 / 50.0)
        narrow = (times >= stamp - .25 * local) & (times <= stamp + .25 * local)
        wide = (times >= stamp - .5 * local) & (times <= stamp + .5 * local)
        expected.append((onset[narrow].mean() if narrow.any() else onset[index],
                         onset[wide].mean() if wide.any() else onset[index],
                         flux[narrow].mean() if narrow.any() else flux[index],
                         rms[wide].mean() if wide.any() else rms[index]))
    np.testing.assert_allclose(got, np.asarray(expected, dtype=np.float32), rtol=0, atol=1e-6)


def test_audio_segment_zero_fills_beyond_file_duration():
    audio = EventAudio(
        np.full((5, ACOUSTIC_DIM), 3.0, dtype=np.float32),
        np.arange(5, dtype=np.float32) / 50.0,
        0.1,
        120.0,
    )
    segment = audio_segment_tensor(
        audio, -0.04, 0.16, song_duration=0.1, bpm=120.0,
        first=-0.04, start_beat=0.0,
    )
    times = -0.04 + np.arange(segment.size(0), dtype=np.float32) / 50.0
    assert np.allclose(segment[:2, :ACOUSTIC_DIM].numpy(), 0.0)
    assert np.allclose(segment[times > audio.times[-1], :ACOUSTIC_DIM].numpy(), 0.0)


def test_global_memory_zero_fills_after_file_duration():
    audio = EventAudio(
        np.full((5, ACOUSTIC_DIM), 3.0, dtype=np.float32),
        np.arange(5, dtype=np.float32) / 50.0,
        0.1,
        120.0,
    )
    memory = compress_event_audio(audio, song_duration=0.2, bpm=120.0, frames=20)
    assert np.allclose(memory[10:, :ACOUSTIC_DIM].numpy(), 0.0)
from maichart.generation.event_audio import _beat_normalized_rhythm, _sample_audio


def _audio_dependencies():
    return pytest.importorskip("soundfile"), pytest.importorskip("librosa")


def _write_wav(path, *, duration=1.0, sample_rate=22050):
    sf, _ = _audio_dependencies()
    times = np.arange(max(1, int(duration * sample_rate)), dtype=np.float32) / sample_rate
    signal = 0.12 * np.sin(2 * np.pi * 440 * times)
    for onset in np.arange(0.1, duration, 0.25):
        start = int(onset * sample_rate)
        signal[start : start + 32] += np.hanning(min(32, len(signal) - start)) * 0.7
    sf.write(path, signal, sample_rate)


def _write_click_track(path, *, bpm, duration=12.0, sample_rate=22050):
    """Write an audible pulse train at an exact tempo."""

    sf, _ = _audio_dependencies()
    total = int(duration * sample_rate)
    signal = np.zeros(total, dtype=np.float32)
    period = 60.0 / bpm
    click = np.hanning(220).astype(np.float32) * 0.8
    for index in range(int(duration / period)):
        start = int(index * period * sample_rate)
        if start + len(click) > total:
            break
        signal[start : start + len(click)] += click
    sf.write(path, signal, sample_rate)


@pytest.mark.parametrize("bpm", [95.0, 128.0, 134.0, 160.0, 182.0, 205.0])
def test_detected_bpm_matches_a_click_track(tmp_path, bpm):
    """The tempo clock must be the real tempo, not an autocorrelation lag bin.

    The previous detector returned values quantized to 3000/integer-lag, so at
    these tempos the nearest representable values were several BPM away.

    Doubling is tolerated because a chart at 2x tempo still places every note on
    a true beat and is a legitimate stylistic reading; what must never happen is
    a tempo that is unrelated to the song, which is what the lag quantisation
    produced.
    """

    _audio_dependencies()

    audio_path = tmp_path / f"click_{bpm}.wav"
    _write_click_track(audio_path, bpm=bpm)
    audio = extract_event_audio(audio_path, cache_dir=tmp_path / "cache")

    detected = float(audio.tempo_bpm)
    folded = detected
    while folded / bpm >= 1.5:
        folded /= 2.0
    while folded / bpm <= 0.67:
        folded *= 2.0
    assert abs(folded - bpm) <= 1.0, (
        f"detected {detected} for a {bpm} BPM click track (folded {folded})"
    )


def test_tempo_policy_change_invalidates_the_cache(tmp_path, monkeypatch):
    """A cache written by an older detector must never be reused.

    Stale caches store the tempo, so reusing one would hand the chart the very
    BPM the fix corrected.  The policy string participates in the cache key,
    which is what makes that impossible.
    """

    _write_wav(tmp_path / "song.wav")
    cache = tmp_path / "cache"
    extract_event_audio(tmp_path / "song.wav", cache_dir=cache)
    original = sorted(item.name for item in cache.glob("*.event_audio.json"))
    assert len(original) == 1

    from maichart.generation import event_audio

    monkeypatch.setattr(event_audio, "TEMPO_POLICY", "some_older_detector")
    extract_event_audio(tmp_path / "song.wav", cache_dir=cache)
    updated = sorted(item.name for item in cache.glob("*.event_audio.json"))
    # A second cache appears under a new key; the stale one is neither reused nor
    # overwritten, so a freshly computed tempo can never be shadowed by it.
    assert len(updated) == 2
    assert original[0] in updated
    assert len(set(original) & set(updated)) == 1


def test_feature_contract_is_not_the_legacy_seven_dimensions():
    # 80 log-mel + 12 chroma + onset/rms/flux + three song-relative structure
    # channels, then ten timing values.
    assert ACOUSTIC_DIM == 98
    assert INPUT_DIM == 112
    assert len(FEATURE_KEYS) == INPUT_DIM
    assert FEATURE_KEYS[-10:] == (
        "song_progress",
        "song_duration",
        "local_tempo_bpm",
        "beat_phase_sin",
        "beat_phase_cos",
        "beat_progress",
        "bar_phase_sin",
        "bar_phase_cos",
        "phrase_phase_sin",
        "phrase_phase_cos",
    )
    assert FEATURE_KEYS[92:98] == (
        "onset_strength", "rms", "spectral_flux",
        "relative_rms", "intensity_slope", "harmonic_novelty",
    )


def test_extract_shape_short_audio_and_cache_config(tmp_path):
    audio_path = tmp_path / "short.wav"
    _write_wav(audio_path, duration=0.01)
    cache = tmp_path / "cache"

    first = extract_event_audio(audio_path, cache_dir=cache, sample_rate=22050, hop_length=441)
    assert first.features.dtype == np.float32
    assert first.features.ndim == 2
    assert first.features.shape[1] == ACOUSTIC_DIM
    assert first.times.shape == (first.features.shape[0],)
    assert first.metadata["feature_config"]["hop_length"] == 441
    assert list(cache.glob("*.event_audio.json"))

    second = extract_event_audio(audio_path, cache_dir=cache, sample_rate=22050, hop_length=220)
    assert second.metadata["feature_config"]["hop_length"] == 220
    assert len(list(cache.glob("*.event_audio.json"))) == 2


def test_cache_round_trip_and_config_metadata_rejection(tmp_path):
    audio_path = tmp_path / "song.wav"
    _write_wav(audio_path)
    cache = tmp_path / "cache"
    features = extract_event_audio(audio_path, cache_dir=cache)
    cache_path = next(cache.glob("*.event_audio.json"))
    loaded = load_event_audio(cache_path)
    np.testing.assert_allclose(loaded.features, features.features)
    assert loaded.metadata["feature_version"] == FEATURE_VERSION

    payload = cache_path.read_text(encoding="utf-8").replace('"hop_length": 441', '"hop_length": 220')
    cache_path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="configuration"):
        load_event_audio(cache_path, sample_rate=22050, hop_length=441)


def test_segment_and_global_memory_keep_context_shape_and_phase(tmp_path):
    audio_path = tmp_path / "song.wav"
    _write_wav(audio_path, duration=2.0)
    audio = extract_event_audio(audio_path, cache_dir=tmp_path / "cache")
    tempo = [(Fraction(0), 120.0), (Fraction(4), 150.0)]
    segment = audio_segment_tensor(
        audio,
        0.5,
        1.5,
        song_duration=2.0,
        bpm=120.0,
        start_beat=Fraction(1),
        first=0.0,
        tempo_changes=tempo,
    )
    assert tuple(segment.shape) == (50, INPUT_DIM)
    assert str(segment.dtype) == "torch.float32"
    memory = compress_event_audio(audio, song_duration=2.0, first=0.0, tempo_changes=tempo)
    assert tuple(memory.shape) == (GLOBAL_MEMORY_FRAMES, INPUT_DIM)
    # phase is a unit circle, not a linearly averaged scalar.
    phase_norm = (memory[:, -2:] ** 2).sum(dim=1)
    assert np.allclose(phase_norm.numpy(), 1.0, atol=2e-4)


def test_gt_tempo_map_changes_local_context_without_resetting_phase():
    audio = EventAudio(
        np.zeros((200, ACOUSTIC_DIM), dtype=np.float32),
        np.arange(200, dtype=np.float32) / 50.0,
        4.0,
        120.0,
    )
    tensor = audio_segment_tensor(
        audio,
        2.0,
        2.1,
        song_duration=4.0,
        first=0.0,
        start_beat=4,
        tempo_changes=[(Fraction(0), 120.0), (Fraction(4), 150.0)],
    )
    assert tensor.shape == (5, INPUT_DIM)
    assert tensor[0, FEATURE_KEYS.index("local_tempo_bpm")] == pytest.approx(150.0 / 240.0)
    assert tensor[0, FEATURE_KEYS.index("beat_phase_sin")] == pytest.approx(0.0, abs=1e-5)
    assert tensor[1, FEATURE_KEYS.index("beat_phase_sin")] == pytest.approx(np.sin(2 * np.pi * 0.05), abs=1e-5)


def test_global_memory_respects_nonzero_first_and_empty_audio():
    audio = EventAudio(
        np.zeros((2, ACOUSTIC_DIM), dtype=np.float32),
        np.array([0.0, 0.02], dtype=np.float32),
        0.0,
        120.0,
    )
    memory = compress_event_audio(audio, song_duration=0.0, first=-0.25, bpm=120.0, frames=64)
    assert memory.shape == (64, INPUT_DIM)
    assert memory[0, FEATURE_KEYS.index("beat_phase_cos")] == pytest.approx(-1.0, abs=1e-5)


def test_audio_sampling_zero_fills_before_and_after_file():
    features = np.full((3, ACOUSTIC_DIM), 7.0, dtype=np.float32)
    audio = EventAudio(features, np.array([1.0, 1.02, 1.04], dtype=np.float32), 1.04, 120.0)
    times, sampled = _sample_audio(audio, 0.98, 1.08)
    assert np.all(sampled[times < 1.0] == 0.0)
    assert np.all(sampled[times > 1.04] == 0.0)
    assert np.all(sampled[(times >= 1.0) & (times <= 1.04)] > 0.0)


def test_compress_event_audio_zero_fills_outside_file_range():
    features = np.full((3, ACOUSTIC_DIM), 5.0, dtype=np.float32)
    audio = EventAudio(features, np.array([1.0, 1.02, 1.04], dtype=np.float32), 1.04, 120.0)
    memory = compress_event_audio(audio, song_duration=1.2, first=0.0, frames=120).numpy()
    acoustic = memory[:, :ACOUSTIC_DIM]
    assert np.all(acoustic[:100] == 0.0)
    assert np.all(acoustic[105:] == 0.0)
    assert np.any(acoustic[100:105] > 0.0)


def test_beat_normalized_rhythm_matches_window_means():
    times = np.arange(100, dtype=np.float64) / 50.0
    acoustic = np.zeros((100, ACOUSTIC_DIM), dtype=np.float32)
    acoustic[:, 92] = np.arange(100, dtype=np.float32)
    acoustic[:, 93] = 2.0 * np.arange(100, dtype=np.float32)
    acoustic[:, 94] = 3.0 * np.arange(100, dtype=np.float32)
    result = _beat_normalized_rhythm(acoustic, times, np.zeros(100), np.full(100, 120.0))
    index = 50
    narrow = (times >= times[index] - 0.125) & (times <= times[index] + 0.125)
    wide = (times >= times[index] - 0.25) & (times <= times[index] + 0.25)
    np.testing.assert_allclose(result[index], [acoustic[narrow, 92].mean(), acoustic[wide, 92].mean(), acoustic[narrow, 94].mean(), acoustic[wide, 93].mean()])


def test_npz_cache_round_trip(tmp_path):
    audio = EventAudio(np.zeros((3, ACOUSTIC_DIM), dtype=np.float32), np.array([0, .02, .04]), .05, 120.0)
    path = tmp_path / "audio.npz"
    save_event_audio_npz(audio, path)
    loaded = load_event_audio_npz(path)
    np.testing.assert_allclose(loaded.features, audio.features)
    assert loaded.metadata["feature_version"] == FEATURE_VERSION


def test_tempo_is_conservative_for_small_estimation_drift(tmp_path):
    audio_path = tmp_path / "drift.wav"
    _write_wav(audio_path, duration=2.0)
    audio = extract_event_audio(audio_path, cache_dir=tmp_path / "cache")
    # This is a direct metadata-level guard: candidate output must not be used
    # as an asserted map for a small 129 -> 132 BPM estimate drift.
    assert all(not (128 <= item["bpm"] <= 133) or item.get("candidate") for item in audio.tempo_changes)
    assert audio.metadata["tempo_policy"] == "continuous_acf_harmonic_comb_v2"


def test_tempo_candidate_helper_does_not_promote_129_to_132_drift():
    from maichart.generation.tempo import tempo_change_candidates

    beat_times = [index * 60.0 / 129.0 for index in range(8)]
    beat_times.extend([beat_times[-1] + index * 60.0 / 132.0 for index in range(1, 9)])
    candidates = tempo_change_candidates(beat_times, 129.0)
    assert len(candidates) == 1


def test_event_audio_validation_rejects_wrong_shape():
    with pytest.raises(ValueError, match="98"):
        EventAudio(np.zeros((2, 7), dtype=np.float32), np.zeros(2), 1.0, 120.0)
