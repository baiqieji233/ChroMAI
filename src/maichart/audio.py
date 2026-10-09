"""Optional audio feature extraction for V2 datasets."""

from __future__ import annotations

import importlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

AUDIO_FEATURES_SCHEMA = "maichart-audio-features-v1"
DEFAULT_SAMPLE_RATE = 22050
DEFAULT_HOP_LENGTH = 512
DEFAULT_MAX_FEATURE_FRAMES = 20000


class AudioFeatureDependencyError(RuntimeError):
    """Raised when optional audio dependencies are not installed."""


@dataclass(slots=True)
class AudioBeat:
    """One estimated beat time."""

    index: int
    time_sec: float


@dataclass(slots=True)
class AudioOnset:
    """One detected onset time and strength."""

    index: int
    time_sec: float
    strength: float


@dataclass(slots=True)
class AudioTempoChange:
    """A locally estimated tempo segment boundary.

    ``time_sec`` is authoritative because audio analysis does not know the
    chart's ``&first`` offset. ``beat`` is the index of the detected beat at
    which the segment starts and is retained for diagnostics.
    """

    beat: float
    time_sec: float
    bpm: float
    confidence: float = 0.0


@dataclass(slots=True)
class AudioFeatureFrame:
    """One fixed-hop audio feature frame."""

    frame_index: int
    time_sec: float
    onset_strength: float
    rms: float
    percussive_rms: float
    harmonic_rms: float
    spectral_centroid: float
    spectral_bandwidth: float
    zero_crossing_rate: float


@dataclass(slots=True)
class AudioFeatureSet:
    """Top-level audio feature payload for one song audio file."""

    schema: str
    audio_path: str
    sample_rate: int
    duration_sec: float
    tempo_bpm: float
    beats: list[AudioBeat] = field(default_factory=list)
    tempo_changes: list[AudioTempoChange] = field(default_factory=list)
    onsets: list[AudioOnset] = field(default_factory=list)
    feature_frames: list[AudioFeatureFrame] = field(default_factory=list)
    hop_length: int = DEFAULT_HOP_LENGTH
    division: int = 16


def analyze_audio_file(
    audio_path: str | Path,
    division: int = 16,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    *,
    hop_length: int = DEFAULT_HOP_LENGTH,
    max_feature_frames: int = DEFAULT_MAX_FEATURE_FRAMES,
    max_duration_sec: float | None = None,
) -> AudioFeatureSet:
    """Extract basic audio features from one audio file.

    Optional audio dependencies are imported lazily so parser-only users do not
    need to install them.
    """

    librosa, np = _import_audio_dependencies()
    source = Path(audio_path)
    y, sr = librosa.load(source, sr=sample_rate, mono=True, duration=max_duration_sec)
    duration_sec = _float(librosa.get_duration(y=y, sr=sr))

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    tempo_raw, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
    )
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        units="frames",
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

    harmonic, percussive = librosa.effects.hpss(y)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    harmonic_rms = librosa.feature.rms(y=harmonic, hop_length=hop_length)[0]
    percussive_rms = librosa.feature.rms(y=percussive, hop_length=hop_length)[0]
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0]
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]

    frame_count = max(
        len(onset_env),
        len(rms),
        len(harmonic_rms),
        len(percussive_rms),
        len(spectral_centroid),
        len(spectral_bandwidth),
        len(zero_crossing_rate),
    )
    stride = max(1, math.ceil(frame_count / max_feature_frames)) if max_feature_frames > 0 else 1
    frame_indices = range(0, frame_count, stride)
    frame_times = librosa.frames_to_time(
        np.asarray(list(frame_indices), dtype=int),
        sr=sr,
        hop_length=hop_length,
    )

    tempo_bpm = _tempo_to_float(tempo_raw, np)
    return AudioFeatureSet(
        schema=AUDIO_FEATURES_SCHEMA,
        audio_path=str(source),
        sample_rate=int(sr),
        duration_sec=duration_sec,
        tempo_bpm=tempo_bpm,
        beats=[
            AudioBeat(index=index, time_sec=_float(time_sec))
            for index, time_sec in enumerate(beat_times)
        ],
        tempo_changes=detect_tempo_changes(
            [float(time_sec) for time_sec in beat_times],
            tempo_bpm,
        ),
        onsets=[
            AudioOnset(
                index=index,
                time_sec=_float(time_sec),
                strength=_float(_array_value(onset_env, int(frame), default=0.0)),
            )
            for index, (frame, time_sec) in enumerate(zip(onset_frames, onset_times))
        ],
        feature_frames=[
            AudioFeatureFrame(
                frame_index=int(frame_index),
                time_sec=_float(time_sec),
                onset_strength=_float(_array_value(onset_env, frame_index)),
                rms=_float(_array_value(rms, frame_index)),
                percussive_rms=_float(_array_value(percussive_rms, frame_index)),
                harmonic_rms=_float(_array_value(harmonic_rms, frame_index)),
                spectral_centroid=_float(_array_value(spectral_centroid, frame_index)),
                spectral_bandwidth=_float(_array_value(spectral_bandwidth, frame_index)),
                zero_crossing_rate=_float(_array_value(zero_crossing_rate, frame_index)),
            )
            for frame_index, time_sec in zip(frame_indices, frame_times)
        ],
        hop_length=int(hop_length),
        division=int(division),
    )


def audio_features_to_dict(features: AudioFeatureSet) -> dict[str, Any]:
    """Convert audio features to JSON-compatible primitives."""

    return asdict(features)


def audio_features_to_json(features: AudioFeatureSet, *, indent: int = 2) -> str:
    """Serialize audio features to JSON."""

    return json.dumps(audio_features_to_dict(features), ensure_ascii=False, indent=indent)


def audio_features_from_dict(data: dict[str, Any]) -> AudioFeatureSet:
    """Load audio features from JSON-compatible primitives."""

    return AudioFeatureSet(
        schema=str(data.get("schema", AUDIO_FEATURES_SCHEMA)),
        audio_path=str(data.get("audio_path", "")),
        sample_rate=int(data.get("sample_rate", DEFAULT_SAMPLE_RATE)),
        duration_sec=float(data.get("duration_sec", 0.0)),
        tempo_bpm=float(data.get("tempo_bpm", 0.0)),
        beats=[AudioBeat(**beat) for beat in data.get("beats", [])],
        tempo_changes=[
            AudioTempoChange(
                beat=float(change.get("beat", 0.0)),
                time_sec=float(change.get("time_sec", 0.0)),
                bpm=float(change.get("bpm", 0.0)),
                confidence=float(change.get("confidence", 0.0)),
            )
            for change in data.get("tempo_changes", [])
            if isinstance(change, dict) and float(change.get("bpm", 0.0)) > 0
        ],
        onsets=[AudioOnset(**onset) for onset in data.get("onsets", [])],
        feature_frames=[
            AudioFeatureFrame(**frame)
            for frame in data.get("feature_frames", [])
        ],
        hop_length=int(data.get("hop_length", DEFAULT_HOP_LENGTH)),
        division=int(data.get("division", 16)),
    )


def audio_features_from_json(payload: str) -> AudioFeatureSet:
    """Deserialize audio features from JSON text."""

    data = json.loads(payload)
    if not isinstance(data, dict):
        raise TypeError("Audio features JSON must decode to an object.")
    return audio_features_from_dict(data)


def save_audio_features_json(features: AudioFeatureSet, path: str | Path) -> None:
    """Write audio features JSON."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(audio_features_to_json(features), encoding="utf-8")


def load_audio_features_json(path: str | Path) -> AudioFeatureSet:
    """Read audio features JSON."""

    return audio_features_from_json(Path(path).read_text(encoding="utf-8"))


def build_audio_features_for_dataset_manifest(
    manifest_path: str | Path,
    *,
    out_dir: str | Path,
    division: int = 16,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    force: bool = False,
    output_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build audio feature files for every song with audio in a dataset manifest."""

    manifest_file = Path(manifest_path).resolve()
    manifest_base = manifest_file.parent
    output_path = Path(output_manifest_path).resolve() if output_manifest_path else manifest_file
    feature_dir = Path(out_dir).resolve()

    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    errors = manifest.setdefault("errors", [])
    summary = {
        "processed": 0,
        "skipped": 0,
        "failed": 0,
    }

    for song in manifest.get("songs", []):
        song_id = str(song.get("song_id") or "unknown")
        audio_path_text = song.get("audio_path")
        if not song.get("has_audio") or not audio_path_text:
            summary["skipped"] += 1
            song["audio_features_status"] = "skipped"
            song["audio_features_reason"] = "missing audio"
            continue

        features_path = feature_dir / f"{song_id}.audio_features.json"
        song["audio_features_path"] = _display_path(features_path, manifest_base)
        if features_path.exists() and not force:
            summary["skipped"] += 1
            song["audio_features_status"] = "skipped"
            song["audio_features_reason"] = "audio features already exist"
            continue

        try:
            features = analyze_audio_file(
                _resolve_manifest_path(audio_path_text, manifest_base),
                division=division,
                sample_rate=sample_rate,
            )
            save_audio_features_json(features, features_path)
            summary["processed"] += 1
            song["audio_features_status"] = "processed"
            song.pop("audio_features_reason", None)
        except Exception as exc:  # noqa: BLE001 - batch extraction must continue.
            summary["failed"] += 1
            song["audio_features_status"] = "failed"
            song["audio_features_reason"] = str(exc)
            errors.append(
                {
                    "stage": "audio_features",
                    "message": str(exc),
                    "song_id": song_id,
                    "audio_path": audio_path_text,
                    "exception_type": type(exc).__name__,
                }
            )

    manifest["audio_feature_summary"] = summary
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "manifest": manifest,
        **summary,
        "error_count": len(errors),
    }


def _import_audio_dependencies():
    try:
        librosa = importlib.import_module("librosa")
        np = importlib.import_module("numpy")
    except ImportError as exc:
        raise AudioFeatureDependencyError(
            "Audio feature extraction requires optional dependencies. "
            "Install them with: pip install -e \".[audio]\""
        ) from exc
    return librosa, np


def _array_value(values, index: int, default: float = 0.0) -> float:
    if index < 0 or index >= len(values):
        return default
    return values[index]


def _float(value: Any) -> float:
    return float(value)


def _tempo_to_float(value: Any, np) -> float:
    array = np.asarray(value).reshape(-1)
    if array.size == 0:
        return 0.0
    return _float(array[0])


def detect_tempo_changes(
    beat_times: list[float],
    global_bpm: float,
    *,
    min_change_ratio: float = 0.08,
    min_persist_beats: int = 3,
) -> list[AudioTempoChange]:
    """Estimate persistent local tempo changes from beat intervals.

    Beat trackers commonly return a half- or double-tempo interpretation. Each
    interval is therefore normalized to the closest of ``0.5x``, ``1x`` and
    ``2x`` the global estimate before change detection. A change is accepted
    only after several consecutive intervals agree, which avoids turning onset
    errors into BPM events. This helper is dependency-free so it can be tested
    with synthetic beat times.
    """

    times = [float(value) for value in beat_times]
    times = [value for index, value in enumerate(times) if index == 0 or value > times[index - 1]]
    if len(times) < 4:
        return []

    raw_bpms = [
        60.0 / (times[index + 1] - times[index])
        for index in range(len(times) - 1)
        if 0.25 <= times[index + 1] - times[index] <= 2.0
    ]
    if len(raw_bpms) < 3:
        return []
    reference = float(global_bpm) if global_bpm > 0 else _median(raw_bpms)
    normalized = [_normalize_tempo(value, reference) for value in raw_bpms]
    smoothed = [
        _median(normalized[max(0, index - 1) : min(len(normalized), index + 2)])
        for index in range(len(normalized))
    ]
    initial_count = min(4, len(smoothed))
    active = _median(smoothed[:initial_count])
    confidence = _tempo_confidence(smoothed[:initial_count], active)
    changes = [
        AudioTempoChange(
            beat=0.0,
            time_sec=times[0],
            bpm=round(active, 6),
            confidence=round(confidence, 6),
        )
    ]

    index = initial_count
    while index < len(smoothed):
        if not _tempo_is_different(smoothed[index], active, min_change_ratio):
            index += 1
            continue
        candidate = smoothed[index]
        end = index + 1
        while end < len(smoothed) and _tempo_is_close(smoothed[end], candidate, min_change_ratio):
            end += 1
        if end - index < max(2, int(min_persist_beats)):
            index = end
            continue
        next_bpm = _median(smoothed[index:end])
        changes.append(
            AudioTempoChange(
                beat=float(index),
                time_sec=times[index],
                bpm=round(next_bpm, 6),
                confidence=round(_tempo_confidence(smoothed[index:end], next_bpm), 6),
            )
        )
        active = next_bpm
        index = end
    return changes


def _normalize_tempo(value: float, reference: float) -> float:
    if reference <= 0:
        return value
    candidates = (value * 0.5, value, value * 2.0)
    return min(candidates, key=lambda candidate: abs(math.log(candidate / reference)))


def _tempo_is_different(value: float, reference: float, ratio: float) -> bool:
    return abs(value - reference) / max(reference, 1e-9) >= ratio


def _tempo_is_close(value: float, reference: float, ratio: float) -> bool:
    return abs(value - reference) / max(reference, 1e-9) < ratio


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _tempo_confidence(values: list[float], reference: float) -> float:
    if not values or reference <= 0:
        return 0.0
    error = sum(abs(value - reference) / reference for value in values) / len(values)
    return max(0.0, min(1.0, 1.0 - error * 4.0))


def _resolve_manifest_path(path: str | Path, base_path: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (base_path / candidate).resolve()


def _display_path(path: Path, base_path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(base_path.resolve()).as_posix()
    except ValueError:
        return str(resolved)
