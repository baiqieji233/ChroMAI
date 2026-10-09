"""Multi-feature musical tension profiles for generation-side guidance.

This module is deliberately independent from the event model's 112-input audio
contract.  It extracts inexpensive, explainable *proxies* for musical tension:
onset/rhythmic activity, spectral flux and brightness/occupancy, timbral change,
harmonic novelty, and a deliberately limited loudness contribution.  The result
is a small JSON-safe profile suitable for an editor with a spectrogram backdrop.

The curve is not emotion recognition.  ``0.5`` means the song's baseline proxy
tension, not a target difficulty grade.  A quiet or silent recording is a valid
profile whose curve is flat at zero; a strong, steady sound remains around the
baseline rather than being mistaken for rest merely because it has few onsets.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "maichart-intensity-curve-v1"
METHOD = "multifeature_v1"
SAMPLE_RATE = 22050
HOP_LENGTH = 512
N_FFT = 2048
N_MELS = 64
MAX_DURATION_SEC = 30.0 * 60.0
MAX_CURVE_KNOTS = 1024
MAX_VALIDATION_POINTS = 2048

# Loudness is intentionally unable to dominate the proxy.  The remaining
# weights describe musical activity/contrast, not a claimed psychological model.
DEFAULT_WEIGHTS = {
    "loudness": 0.15,
    "onset_activity": 0.22,
    "spectral_flux": 0.18,
    "spectral_brightness": 0.10,
    "timbre_change": 0.12,
    "harmonic_novelty": 0.13,
    "local_contrast": 0.10,
}


def audio_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the content SHA256 used to bind a profile to its source audio."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while True:
            chunk = handle.read(int(chunk_size))
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _librosa():
    try:
        import librosa
    except ImportError as exc:  # pragma: no cover - exercised by optional-dep tests
        raise RuntimeError(
            'Music intensity analysis requires librosa; install the optional audio dependencies with '
            '`pip install -e ".[audio]"`.'
        ) from exc
    return librosa


def _finite_array(values: Any, *, fallback: float = 0.0) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = np.where(np.isfinite(array), array, fallback)
    return array


def _normalize(values: np.ndarray, *, fallback: float = 0.5) -> np.ndarray:
    """Robustly map a feature to 0..1, preserving a steady non-zero signal."""

    values = _finite_array(values)
    if not values.size:
        return values
    low, high = np.percentile(values, [10.0, 90.0])
    span = float(high - low)
    if span <= 1e-9:
        return np.full(values.shape, float(np.clip(fallback, 0.0, 1.0)))
    return np.clip((values - low) / span, 0.0, 1.0)


def _rolling_mean(values: np.ndarray, width: int) -> np.ndarray:
    if values.size == 0 or width <= 1:
        return values.astype(np.float64, copy=True)
    width = min(int(width), int(values.size))
    kernel = np.ones(width, dtype=np.float64) / float(width)
    pad_left = width // 2
    pad_right = width - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _local_contrast(values: np.ndarray, radius: int) -> np.ndarray:
    """Positive local contrast against a centred context median."""

    values = _finite_array(values)
    if values.size == 0 or radius <= 0:
        return np.zeros(values.shape, dtype=np.float64)
    result = np.zeros_like(values)
    for index in range(values.size):
        low = max(0, index - radius)
        high = min(values.size, index + radius + 1)
        context = np.concatenate((values[low:index], values[index + 1 : high]))
        if context.size:
            scale = float(np.std(context)) + 1e-6
            result[index] = (values[index] - float(np.median(context))) / scale
    # Map signed contrast to [0, 1], retaining local dips below baseline.
    return np.clip(0.5 + 0.25 * result, 0.0, 1.0)


def _resample_columns(values: np.ndarray, columns: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("spectrogram must be a 2D matrix")
    columns = max(1, int(columns))
    if values.shape[1] <= columns:
        return values
    edges = np.linspace(0, values.shape[1], columns + 1, dtype=int)
    result = np.empty((values.shape[0], columns), dtype=np.float64)
    for index in range(columns):
        lo, hi = int(edges[index]), int(edges[index + 1])
        hi = max(lo + 1, hi)
        result[:, index] = values[:, lo:min(hi, values.shape[1])].mean(axis=1)
    return result


def _feature_analysis(audio_path: str | Path) -> dict[str, Any]:
    """Extract frame-level proxies and a display spectrogram (internal/public helper)."""

    source = Path(audio_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    librosa = _librosa()
    try:
        metadata_duration = float(librosa.get_duration(path=source))
    except Exception as exc:
        raise ValueError(f"cannot inspect audio duration: {source}") from exc
    if not np.isfinite(metadata_duration) or metadata_duration <= 0.0:
        raise ValueError("audio duration must be finite and positive")
    if metadata_duration > MAX_DURATION_SEC:
        raise ValueError(f"audio duration exceeds {MAX_DURATION_SEC:g} seconds")
    y, sr = librosa.load(source, sr=SAMPLE_RATE, mono=True)
    y = np.asarray(y, dtype=np.float64)
    if not y.size:
        raise ValueError("audio file contains no samples")
    duration = float(librosa.get_duration(y=y, sr=sr))
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("audio duration must be finite and positive")
    if duration > MAX_DURATION_SEC:
        raise ValueError(f"audio duration exceeds {MAX_DURATION_SEC:g} seconds")

    mel_power = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, power=2.0
    )
    logmel = librosa.power_to_db(mel_power, ref=np.max, top_db=80.0)
    times = librosa.frames_to_time(np.arange(logmel.shape[1]), sr=sr, hop_length=HOP_LENGTH)
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH, center=True)[0]
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
    spectrum = np.abs(librosa.stft(y=y, n_fft=N_FFT, hop_length=HOP_LENGTH, center=True))
    flux = np.sqrt(np.mean(np.maximum(0.0, np.diff(spectrum, axis=1, prepend=spectrum[:, :1])) ** 2, axis=0))
    centroid = librosa.feature.spectral_centroid(S=np.maximum(spectrum, 1e-12), sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=np.maximum(spectrum, 1e-12), sr=sr)[0]
    # Occupancy is the fraction of spectral energy above each frame's median bin.
    # A median-bin threshold is almost always exactly 0.5 occupancy.  Use a
    # perceptual relative level instead: how much of the spectrum remains within
    # 24 dB of the frame peak.
    spectrum_db = librosa.amplitude_to_db(np.maximum(spectrum, 1e-12), ref=np.max, top_db=80.0)
    occupancy = np.mean(spectrum_db >= (np.max(spectrum_db, axis=0, keepdims=True) - 24.0), axis=0)
    # Chroma cosine distance to a nearby past frame is a cheap harmonic novelty proxy.
    chroma = librosa.feature.chroma_stft(S=np.maximum(spectrum, 1e-12), sr=sr, hop_length=HOP_LENGTH)
    lag = max(1, int(round(2.0 * sr / HOP_LENGTH)))
    harmonic_novelty = np.zeros(chroma.shape[1], dtype=np.float64)
    for index in range(lag, chroma.shape[1]):
        left = chroma[:, index]
        right = chroma[:, index - lag]
        denom = float(np.linalg.norm(left) * np.linalg.norm(right))
        harmonic_novelty[index] = 1.0 if denom <= 1e-12 else float(1.0 - np.dot(left, right) / denom)
    brightness = centroid / (float(sr) / 2.0)
    # Align all frame series to the shortest common frame clock.
    frame_count = min(len(times), len(rms), len(onset), len(flux), len(brightness), len(occupancy), len(harmonic_novelty))
    times = np.asarray(times[:frame_count], dtype=np.float64)
    raw = {
        "loudness": _finite_array(rms[:frame_count]),
        "onset_activity": _finite_array(onset[:frame_count]),
        "spectral_flux": _finite_array(flux[:frame_count]),
        "spectral_brightness": np.clip(_finite_array(brightness[:frame_count]), 0.0, 1.0),
        "spectral_occupancy": np.clip(_finite_array(occupancy[:frame_count]), 0.0, 1.0),
        "timbre_change": _finite_array(np.abs(np.diff(centroid[:frame_count], prepend=centroid[:1])) / (float(sr) / 2.0)),
        "harmonic_novelty": np.clip(_finite_array(harmonic_novelty[:frame_count]), 0.0, 1.0),
    }
    # Onset density uses a short local average, while other channels retain their
    # temporal detail for later resolution/context aggregation.
    raw["onset_activity"] = _rolling_mean(raw["onset_activity"], max(1, int(round(0.5 * sr / HOP_LENGTH))))
    raw["spectral_flux"] = _rolling_mean(raw["spectral_flux"], max(1, int(round(0.25 * sr / HOP_LENGTH))))
    raw["timbre_change"] = _rolling_mean(raw["timbre_change"], max(1, int(round(0.5 * sr / HOP_LENGTH))))
    normalized = {key: _normalize(value) for key, value in raw.items() if key != "spectral_occupancy"}
    normalized["spectral_brightness"] = np.clip(
        0.7 * normalized["spectral_brightness"] + 0.3 * _normalize(raw["spectral_occupancy"]), 0.0, 1.0
    )
    normalized["local_contrast"] = _local_contrast(
        0.55 * normalized["onset_activity"]
        + 0.25 * normalized["spectral_flux"]
        + 0.20 * normalized["harmonic_novelty"],
        max(1, int(round(12.0 * sr / HOP_LENGTH))),
    )
    rms_level = float(np.sqrt(np.mean(np.square(y))))
    return {
        "duration_sec": duration,
        "times": times,
        "raw": raw,
        "normalized": normalized,
        "logmel": logmel,
        "sr": int(sr),
        "rms_level": rms_level,
        "audible": np.asarray(raw["loudness"] >= max(float(np.max(raw["loudness"])) * 1e-4, 1e-7)),
    }


def analyze_intensity_features(audio_path: str | Path) -> dict[str, Any]:
    """Expose frame-level proxy features for diagnostics and synthetic tests."""

    result = _feature_analysis(audio_path)
    return {
        "duration_sec": float(result["duration_sec"]),
        "times": result["times"].tolist(),
        "components": {key: value.tolist() for key, value in result["normalized"].items()},
    }


def _build_spectrogram(logmel: np.ndarray, duration: float, sr: int) -> dict[str, Any]:
    columns = min(768, max(1, int(logmel.shape[1])))
    display = _resample_columns(logmel, columns)
    quantized = np.rint(np.clip((display + 80.0) / 80.0, 0.0, 1.0) * 255.0).astype(np.int16)
    frequencies = librosa_mel_frequencies(sr, N_MELS).tolist()
    times = np.linspace(0.0, float(duration), columns).tolist()
    return {
        "times": [float(value) for value in times],
        "frequencies_hz": [float(value) for value in frequencies],
        "values": quantized.astype(int).tolist(),
        "db_min": -80,
        "db_max": 0,
        "kind": "log_mel_db",
    }


def librosa_mel_frequencies(sr: int, n_mels: int) -> np.ndarray:
    """Keep spectrogram metadata creation lazy and dependency-local."""

    librosa = _librosa()
    return np.asarray(librosa.mel_frequencies(n_mels=n_mels, fmin=0.0, fmax=float(sr) / 2.0), dtype=np.float64)


def generate_intensity_profile(
    audio_path: str | Path,
    *,
    resolution_sec: float = 2.0,
    local_context_sec: float = 12.0,
) -> dict[str, Any]:
    """Generate a validated, JSON-safe multi-feature tension profile."""

    resolution_sec = float(resolution_sec)
    local_context_sec = float(local_context_sec)
    if not np.isfinite(resolution_sec) or resolution_sec <= 0.0:
        raise ValueError("resolution_sec must be finite and positive")
    if not np.isfinite(local_context_sec) or local_context_sec < 0.0:
        raise ValueError("local_context_sec must be finite and non-negative")
    result = _feature_analysis(audio_path)
    duration = float(result["duration_sec"])
    point_count = min(MAX_CURVE_KNOTS, max(2, int(math.ceil(duration / resolution_sec)) + 1))
    point_times = np.linspace(0.0, duration, point_count)
    frame_times = result["times"]
    local_radius = max(0, int(round(local_context_sec * SAMPLE_RATE / HOP_LENGTH)))
    def sample(values: np.ndarray) -> np.ndarray:
        sampled = np.empty(point_count, dtype=np.float64)
        half = max(resolution_sec * 0.5, (frame_times[1] - frame_times[0]) if len(frame_times) > 1 else resolution_sec)
        for index, center in enumerate(point_times):
            mask = (frame_times >= center - half) & (frame_times <= center + half)
            if not np.any(mask):
                sampled[index] = float(np.interp(center, frame_times, values))
            else:
                selected = values[mask]
                sampled[index] = float(0.75 * np.mean(selected) + 0.25 * np.percentile(selected, 90.0))
        return np.clip(sampled, 0.0, 1.0)

    components: dict[str, np.ndarray] = {key: sample(values) for key, values in result["normalized"].items()}
    local_frame = _local_contrast(
        0.55 * result["normalized"]["onset_activity"]
        + 0.25 * result["normalized"]["spectral_flux"]
        + 0.20 * result["normalized"]["harmonic_novelty"],
        local_radius,
    )
    components["local_contrast"] = sample(local_frame)
    # An RMS gate only identifies truly silent audio. It never suppresses a
    # steady, audible section whose activity features happen to be flat.
    silent = float(result["rms_level"]) <= 1e-5
    if silent:
        intensity = np.zeros(point_count, dtype=np.float64)
        for key in components:
            components[key] = np.zeros(point_count, dtype=np.float64)
    else:
        intensity = np.zeros(point_count, dtype=np.float64)
        for key, weight in DEFAULT_WEIGHTS.items():
            intensity += float(weight) * components.get(key, np.full(point_count, 0.5))
        intensity = np.clip(intensity, 0.0, 1.0)
        audible_points = sample(result["audible"].astype(np.float64)) > 0.05
        intensity[~audible_points] = 0.0
        for key in components:
            components[key][~audible_points] = 0.0
        # Local contrast is a signed comparison proxy; pull the baseline toward
        # 0.5 so stable strong passages are not interpreted as rests.
        if float(np.std(intensity)) < 1e-8:
            intensity.fill(0.5)
    points = [
        {"time_sec": float(time), "intensity": float(value)}
        for time, value in zip(point_times, intensity)
    ]
    profile = {
        "schema": SCHEMA,
        "audio_sha256": audio_sha256(audio_path),
        "duration_sec": duration,
        "audio_path": str(Path(audio_path)),
        "method": METHOD,
        "points": points,
        "spectrogram": _build_spectrogram(result["logmel"], duration, int(result["sr"])),
        "components": {key: [float(value) for value in values] for key, values in components.items()},
        "weights": dict(DEFAULT_WEIGHTS),
        "limitations": [
            "Features are explainable musical proxies, not emotion recognition.",
            "0.5 denotes the song's baseline proxy tension, not a target difficulty level.",
            "Audio alone cannot fully recover arranger intent; edit the curve when a deliberate rest is desired.",
        ],
    }
    return validate_intensity_profile(profile)


def intensity_profile_from_event_audio(
    audio: Any,
    *,
    resolution_sec: float = 2.0,
    local_context_sec: float = 12.0,
) -> dict[str, Any]:
    """Build an approximate multi-feature curve from an existing EventAudio cache.

    This path never decodes the source file and therefore omits the raw-dB
    spectrogram. It accepts current 98-column acoustic tensors and legacy
    95/99-column cache-like objects by using the stable mel/chroma/onset/RMS/
    flux prefix and whichever structure channels are present.
    """

    resolution_sec = float(resolution_sec)
    local_context_sec = float(local_context_sec)
    if not np.isfinite(resolution_sec) or resolution_sec <= 0.0:
        raise ValueError("resolution_sec must be finite and positive")
    if not np.isfinite(local_context_sec) or local_context_sec < 0.0:
        raise ValueError("local_context_sec must be finite and non-negative")
    features = np.asarray(audio.features, dtype=np.float64)
    times = np.asarray(audio.times, dtype=np.float64).reshape(-1)
    if features.ndim != 2 or features.shape[0] != times.size or features.shape[1] < 95:
        raise ValueError("audio.features must contain at least the legacy 95 acoustic columns")
    duration = float(audio.duration_sec)
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("audio.duration_sec must be finite and positive")
    order = np.argsort(times)
    times, features = times[order], features[order]
    mel = np.nan_to_num(features[:, :80], nan=0.0, posinf=0.0, neginf=0.0)
    chroma = np.nan_to_num(features[:, 80:92], nan=0.0, posinf=0.0, neginf=0.0)
    onset, rms, flux = (np.nan_to_num(features[:, index], nan=0.0, posinf=0.0, neginf=0.0) for index in (92, 93, 94))
    relative = np.nan_to_num(features[:, 95], nan=0.0, posinf=0.0, neginf=0.0) if features.shape[1] > 95 else rms
    novelty = np.nan_to_num(features[:, 97], nan=0.0, posinf=0.0, neginf=0.0) if features.shape[1] > 97 else np.zeros_like(rms)
    # Cached mel values are robust-normalized z-like proxies, not dB. Shift each
    # frame positive before computing a relative high-frequency centre of mass.
    mel_positive = mel - np.min(mel, axis=1, keepdims=True)
    brightness_raw = np.sum(mel_positive * np.linspace(0.0, 1.0, 80)[None, :], axis=1) / np.maximum(np.sum(mel_positive, axis=1), 1e-6)
    timbre_raw = np.linalg.norm(np.diff(mel, axis=0, prepend=mel[:1]), axis=1)
    raw = {
        "loudness": relative,
        "onset_activity": _rolling_mean(np.maximum(onset, 0.0), max(1, int(round(0.5 / max(times[1] - times[0], 1e-3))) if len(times) > 1 else 1)),
        "spectral_flux": _rolling_mean(np.maximum(flux, 0.0), max(1, int(round(0.25 / max(times[1] - times[0], 1e-3))) if len(times) > 1 else 1)),
        "spectral_brightness": np.clip(brightness_raw, 0.0, 1.0),
        "timbre_change": timbre_raw,
        "harmonic_novelty": np.clip(novelty, 0.0, 1.0),
    }
    normalized = {key: _normalize(value) for key, value in raw.items()}
    radius = max(0, int(round(local_context_sec / max(float(np.median(np.diff(times))) if len(times) > 1 else 0.02, 1e-3))))
    normalized["local_contrast"] = _local_contrast(
        0.55 * normalized["onset_activity"] + 0.25 * normalized["spectral_flux"] + 0.20 * normalized["harmonic_novelty"],
        radius,
    )
    point_count = min(MAX_CURVE_KNOTS, max(2, int(math.ceil(duration / resolution_sec)) + 1))
    point_times = np.linspace(0.0, duration, point_count)
    def sample(values: np.ndarray) -> np.ndarray:
        output = np.empty(point_count, dtype=np.float64)
        half = max(resolution_sec * 0.5, float(np.median(np.diff(times))) if len(times) > 1 else resolution_sec)
        for index, center in enumerate(point_times):
            mask = (times >= center - half) & (times <= center + half)
            output[index] = float(np.mean(values[mask])) if np.any(mask) else float(np.interp(center, times, values, left=float(values[0]), right=float(values[-1])))
        return np.clip(output, 0.0, 1.0)
    components = {key: sample(value) for key, value in normalized.items()}
    intensity = np.zeros(point_count, dtype=np.float64)
    for key, weight in DEFAULT_WEIGHTS.items():
        intensity += float(weight) * components.get(key, np.full(point_count, 0.5))
    if features.shape[1] > 95:
        audible_frames = relative > max(float(np.max(relative)) * 1e-4, 1e-4)
    else:
        # Legacy caches have no readable amplitude channel; never infer silence
        # from robust-normalized RMS, which may be negative for audible music.
        audible_frames = np.ones_like(rms, dtype=bool)
    audible = sample(audible_frames.astype(np.float64)) > 0.05
    intensity = np.clip(intensity, 0.0, 1.0)
    intensity[~audible] = 0.0
    for key in components:
        components[key][~audible] = 0.0
    metadata = getattr(audio, "metadata", {}) or {}
    profile = {
        "schema": SCHEMA,
        "audio_sha256": str(metadata.get("audio_sha256", "")),
        "duration_sec": duration,
        "audio_path": str(metadata.get("audio_path", "")),
        "method": "multifeature_v1_cached_feature_proxies",
        "points": [{"time_sec": float(t), "intensity": float(v)} for t, v in zip(point_times, intensity)],
        "components": {key: [float(v) for v in value] for key, value in components.items()},
        "weights": dict(DEFAULT_WEIGHTS),
        "limitations": [
            "Derived from EventAudio cache; no raw log-mel dB spectrogram is included.",
            "Features are explainable musical proxies, not emotion recognition.",
            "0.5 denotes the song's baseline proxy tension, not a target difficulty level.",
        ],
    }
    return validate_intensity_profile(profile)


def validate_intensity_profile(profile: dict[str, Any], *, duration_sec: float | None = None) -> dict[str, Any]:
    """Validate and normalize a profile, rejecting NaN and malformed timing."""

    if not isinstance(profile, dict):
        raise ValueError("intensity profile must be a mapping")
    if profile.get("schema") != SCHEMA:
        raise ValueError(f"unsupported intensity profile schema: {profile.get('schema')!r}")
    try:
        duration = float(profile["duration_sec"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("profile duration_sec is required") from exc
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("profile duration_sec must be finite and positive")
    if duration_sec is not None:
        bound = float(duration_sec)
        if not np.isfinite(bound) or bound < 0.0 or abs(duration - bound) > 1e-3:
            raise ValueError("profile duration does not match duration_sec")
    points = profile.get("points")
    if not isinstance(points, list) or len(points) < 2 or len(points) > MAX_VALIDATION_POINTS:
        raise ValueError("profile must contain 2..2048 points")
    normalized_points: list[dict[str, float]] = []
    previous = -math.inf
    for point in points:
        if not isinstance(point, dict):
            raise ValueError("each intensity point must be a mapping")
        try:
            time = float(point["time_sec"])
            value = float(point["intensity"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("points require time_sec and intensity") from exc
        if not np.isfinite(time) or not np.isfinite(value):
            raise ValueError("point values must be finite")
        if time < 0.0 or time <= previous or time > duration + 1e-6:
            raise ValueError("point times must be strictly increasing and within duration")
        if value < 0.0 or value > 1.0:
            raise ValueError("point intensity must be in 0..1")
        normalized_points.append({"time_sec": max(0.0, time), "intensity": value})
        previous = time
    if abs(normalized_points[0]["time_sec"]) > 1e-6 or abs(normalized_points[-1]["time_sec"] - duration) > 1e-3:
        raise ValueError("profile points must start at 0 and end at duration_sec")
    normalized = dict(profile)
    normalized["duration_sec"] = duration
    normalized["points"] = normalized_points
    regions = profile.get("rest_regions", [])
    if not isinstance(regions, list) or len(regions) > 256:
        raise ValueError("rest_regions must contain at most 256 time intervals")
    rests = []
    for region in regions:
        if not isinstance(region, dict):
            raise ValueError("rest_regions require start_sec and end_sec")
        try:
            low, high = float(region["start_sec"]), float(region["end_sec"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("rest_regions require numeric start_sec and end_sec") from exc
        if not np.isfinite([low, high]).all() or not 0 <= low < high <= duration:
            raise ValueError("rest_regions must be finite, increasing and within the song")
        rests.append({"start_sec": low, "end_sec": high})
    merged = []
    for region in sorted(rests, key=lambda item: item["start_sec"]):
        if merged and region["start_sec"] <= merged[-1]["end_sec"]:
            merged[-1]["end_sec"] = max(merged[-1]["end_sec"], region["end_sec"])
        else:
            merged.append(dict(region))
    # Optional extension: old profiles keep their exact serialized shape.
    if regions or "rest_regions" in profile:
        normalized["rest_regions"] = merged
    if "audio_sha256" in normalized:
        normalized["audio_sha256"] = str(normalized["audio_sha256"])
    components = normalized.get("components")
    if components is not None:
        if not isinstance(components, dict):
            raise ValueError("components must be a mapping")
        for key, values in components.items():
            array = np.asarray(values, dtype=np.float64).reshape(-1)
            if not np.all(np.isfinite(array)):
                raise ValueError(f"component {key!r} contains non-finite values")
            if len(array) != len(normalized_points) or np.any(array < 0.0) or np.any(array > 1.0):
                raise ValueError(f"component {key!r} must match points and stay in 0..1")
            normalized["components"][key] = [float(value) for value in array]
    return normalized


def load_intensity_profile(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return validate_intensity_profile(payload)


def curve_window_values(profile: dict[str, Any], start_sec: float, end_sec: float) -> float:
    """Return the time-integrated mean intensity over a clamped interval."""

    profile = validate_intensity_profile(profile)
    start, end = float(start_sec), float(end_sec)
    if not np.isfinite(start) or not np.isfinite(end) or end <= start:
        raise ValueError("curve window must have finite end_sec > start_sec")
    duration = float(profile["duration_sec"])
    start, end = np.clip([start, end], 0.0, duration)
    if end <= start:
        return float(profile["points"][0]["intensity"])
    times = np.asarray([point["time_sec"] for point in profile["points"]], dtype=np.float64)
    values = np.asarray([point["intensity"] for point in profile["points"]], dtype=np.float64)
    def integrate(low, high):
        interior = times[(times > low) & (times < high)]
        grid = np.concatenate(([low], interior, [high]))
        sampled = np.interp(grid, times, values)
        return float(np.sum((sampled[:-1] + sampled[1:]) * np.diff(grid) * 0.5))
    integral = integrate(start, end)
    for region in profile.get("rest_regions", []):
        low, high = max(start, region["start_sec"]), min(end, region["end_sec"])
        if high > low:
            integral -= integrate(low, high)
    return max(0.0, integral) / (end - start)


__all__ = [
    "SCHEMA",
    "METHOD",
    "DEFAULT_WEIGHTS",
    "audio_sha256",
    "analyze_intensity_features",
    "generate_intensity_profile",
    "intensity_profile_from_event_audio",
    "validate_intensity_profile",
    "load_intensity_profile",
    "curve_window_values",
]
