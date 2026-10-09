import json
from pathlib import Path

import numpy as np
import pytest

from maichart.generation.music_intensity import (
    SCHEMA,
    analyze_intensity_features,
    audio_sha256,
    curve_window_values,
    generate_intensity_profile,
    load_intensity_profile,
    validate_intensity_profile,
)


def _deps():
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    pytest.importorskip("librosa")
    return np, sf


def _write_wav(path: Path, *, mode: str = "mixed", duration: float = 8.0) -> None:
    np, sf = _deps()
    sr = 16000
    t = np.arange(int(sr * duration), dtype=np.float64) / sr
    base = 0.16 * np.sin(2 * np.pi * 220.0 * t)
    if mode == "dip":
        y = base.copy()
        # Equal RMS outside the middle, with both the beat-like bursts and upper
        # harmonics removed in that interval. The full file is RMS-normalized.
        for onset in np.arange(0.25, duration, 0.5):
            if 3.0 <= onset < 5.0:
                continue
            index = int(onset * sr)
            burst = np.hanning(int(0.04 * sr)) * 0.45
            y[index : index + len(burst)] += burst
        y[3 * sr : 5 * sr] = 0.16 * np.sin(2 * np.pi * 220.0 * t[3 * sr : 5 * sr])
        y *= 0.18 / np.sqrt(np.mean(y * y))
    elif mode == "silent":
        y = np.zeros_like(t)
    elif mode == "onsets":
        y = base.copy()
        for onset in (1.0, 2.0, 3.0, 5.0, 6.0, 7.0):
            index = int(onset * sr)
            burst = np.hanning(int(0.08 * sr)) * 0.8
            y[index : index + len(burst)] += burst
    elif mode == "timbre":
        y = base.copy()
        y[int(4 * sr) :] += 0.12 * np.sin(2 * np.pi * 2600.0 * t[int(4 * sr) :])
    else:
        y = base
    if mode != "silent":
        y *= 0.12 / max(float(np.sqrt(np.mean(y * y))), 1e-12)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, y.astype(np.float32), sr)


def test_profile_schema_components_and_real_spectrogram(tmp_path):
    path = tmp_path / "song.wav"
    _write_wav(path, mode="onsets")
    profile = generate_intensity_profile(path, resolution_sec=1.0, local_context_sec=2.0)
    assert profile["schema"] == SCHEMA
    assert profile["method"] == "multifeature_v1"
    assert profile["audio_sha256"] == audio_sha256(path)
    assert profile["points"][0]["time_sec"] == pytest.approx(0.0)
    assert profile["points"][-1]["time_sec"] == pytest.approx(profile["duration_sec"])
    assert len(profile["points"]) <= 1024
    assert all(0.0 <= p["intensity"] <= 1.0 for p in profile["points"])
    assert profile["spectrogram"]["kind"] == "log_mel_db"
    assert profile["spectrogram"]["db_min"] == -80
    assert profile["spectrogram"]["db_max"] == 0
    values = np.asarray(profile["spectrogram"]["values"])
    assert values.ndim == 2 and values.shape[0] <= 64 and values.shape[1] <= 768
    assert np.issubdtype(values.dtype, np.integer)
    assert int(values.min()) >= 0 and int(values.max()) <= 255
    for values in profile["components"].values():
        assert len(values) == len(profile["points"])


def test_silence_is_valid_flat_zero_curve(tmp_path):
    path = tmp_path / "silence.wav"
    _write_wav(path, mode="silent", duration=2.0)
    profile = generate_intensity_profile(path)
    assert all(point["intensity"] == pytest.approx(0.0) for point in profile["points"])


def test_same_rms_onsets_and_timbre_change_proxy_features(tmp_path):
    quiet = tmp_path / "steady.wav"
    active = tmp_path / "active.wav"
    timbre = tmp_path / "timbre.wav"
    _write_wav(quiet, mode="mixed")
    _write_wav(active, mode="onsets")
    _write_wav(timbre, mode="timbre")
    np, sf = _deps()
    rms_values = [float(np.sqrt(np.mean(sf.read(path)[0] ** 2))) for path in (quiet, active, timbre)]
    assert max(rms_values) - min(rms_values) < 1e-5
    steady_features = analyze_intensity_features(quiet)
    active_features = analyze_intensity_features(active)
    timbre_features = analyze_intensity_features(timbre)
    assert np.std(active_features["components"]["onset_activity"]) > np.std(steady_features["components"]["onset_activity"])
    assert max(timbre_features["components"]["timbre_change"]) > 0.55


def test_equal_rms_middle_arrangement_dip_is_detectable(tmp_path):
    path = tmp_path / "dip.wav"
    _write_wav(path, mode="dip")
    profile = generate_intensity_profile(path, resolution_sec=0.5, local_context_sec=2.0)
    at = [point["intensity"] for point in profile["points"] if 3.25 <= point["time_sec"] <= 4.75]
    outside = [point["intensity"] for point in profile["points"] if point["time_sec"] < 2.5 or 5.5 < point["time_sec"] < 7.5]
    assert at and outside
    assert np.mean(at) < np.mean(outside)


def test_cached_event_audio_does_not_gate_on_normalized_rms():
    from types import SimpleNamespace

    features = np.zeros((100, 98), dtype=np.float32)
    features[:, 92] = -0.2
    features[:, 93] = -1.5  # Robust-normalized RMS can be negative while audible.
    features[:, 94] = 0.1
    features[:, 95] = 0.8
    features[:, 97] = 0.05
    audio = SimpleNamespace(
        features=features,
        times=np.arange(100, dtype=np.float32) / 10,
        duration_sec=10.0,
        metadata={"audio_sha256": "a" * 64},
    )
    from maichart.generation.music_intensity import intensity_profile_from_event_audio

    profile = intensity_profile_from_event_audio(audio, resolution_sec=1.0)
    assert any(point["intensity"] > 0.0 for point in profile["points"])
    assert profile["audio_sha256"] == "a" * 64


def test_local_dip_and_time_integrated_window_mean():
    profile = {
        "schema": SCHEMA,
        "duration_sec": 10.0,
        "points": [
            {"time_sec": 0.0, "intensity": 0.8},
            {"time_sec": 2.0, "intensity": 0.8},
            {"time_sec": 4.0, "intensity": 0.2},
            {"time_sec": 6.0, "intensity": 0.2},
            {"time_sec": 10.0, "intensity": 0.8},
        ],
    }
    checked = validate_intensity_profile(profile)
    assert curve_window_values(checked, 2.0, 6.0) == pytest.approx(0.35)
    assert curve_window_values(checked, -5.0, 2.0) == pytest.approx(0.8)
    assert curve_window_values(checked, 9.0, 20.0) == pytest.approx(0.725)


def test_validation_rejects_nan_and_duration_bounds():
    base = {"schema": SCHEMA, "duration_sec": 2.0, "points": [{"time_sec": 0, "intensity": 0.5}, {"time_sec": 2, "intensity": 0.5}]}
    with pytest.raises(ValueError):
        validate_intensity_profile({**base, "points": [{"time_sec": 0, "intensity": float("nan")}, base["points"][1]]})
    with pytest.raises(ValueError):
        validate_intensity_profile(base, duration_sec=3.0)
    with pytest.raises(ValueError):
        validate_intensity_profile({**base, "points": [{"time_sec": 0, "intensity": 0.5}, {"time_sec": 3, "intensity": 0.5}]})


def test_load_profile_round_trip(tmp_path):
    path = tmp_path / "profile.json"
    payload = {"schema": SCHEMA, "duration_sec": 1.0, "points": [{"time_sec": 0.0, "intensity": 0.0}, {"time_sec": 1.0, "intensity": 1.0}]}
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_intensity_profile(path)["points"][-1]["intensity"] == 1.0
