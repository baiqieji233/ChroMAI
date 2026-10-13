"""Audio features and timing context for event-level chart generation.

This module is deliberately independent from :mod:`maichart.audio`.  The
older module stores a sparse JSON feature set for the frame-level baseline;
event generation needs a dense, fixed-rate representation and a precise
chart-clock context built from a caller-owned tempo map.

The acoustic base is 98 values at 50 Hz: 80 log-mel bins, 12 chroma bins,
three robust onset/RMS/flux channels, and three song-relative structure
channels (relative RMS, its one-second trend, and a harmonic-novelty signal
that rises at section changes).  Ten timing values are appended by
``audio_segment_tensor``: global progress, normalized duration, local BPM,
beat phase as sine/cosine, beat progress through the song, and bar/phrase
phase as sine/cosine, plus four beat-normalized rhythm channels.  Thus
``INPUT_DIM`` is 112 for the current event-audio release.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

import numpy as np

from maichart.generation.tempo import detect_global_bpm, tempo_change_candidates

FEATURE_VERSION = "event-audio-v4-rhythm"
FEATURE_RATE_HZ = 50
# Identifies the tempo-estimation algorithm.  It participates in the cache key
# via ``_feature_config``, so a detector change automatically invalidates every
# cache written by an older one -- without this, a chart would be generated
# against the stale BPM stored in an existing cache.
TEMPO_POLICY = "continuous_acf_harmonic_comb_v2"
N_MELS = 80
N_CHROMA = 12
# The original onset/RMS/flux block plus three song-relative structure
# channels.  ``relative_rms`` is 1.0 exactly at the song's median loudness,
# ``intensity_slope`` is its one-second change (build-up vs drop), and
# ``harmonic_novelty`` is ~0 while the harmonic content is stable and rises
# toward 1 around a section change.  These give the model an explicit,
# dedicated "how intense is this part relative to the rest of the song" signal
# instead of leaving it implicit in 95 normalized columns.
STRUCTURE_KEYS = ("relative_rms", "intensity_slope", "harmonic_novelty")
BASE_ACOUSTIC_KEYS = ("onset_strength", "rms", "spectral_flux")
ACOUSTIC_DIM = N_MELS + N_CHROMA + len(BASE_ACOUSTIC_KEYS) + len(STRUCTURE_KEYS)
# Column index of ``relative_rms`` inside the cached acoustic block; the
# acceptance metrics read the song-relative intensity straight from here.
RELATIVE_RMS_COLUMN = N_MELS + N_CHROMA + len(BASE_ACOUSTIC_KEYS)
CONTEXT_KEYS = (
    "song_progress", "song_duration", "local_tempo_bpm", "beat_phase_sin", "beat_phase_cos",
    "beat_progress", "bar_phase_sin", "bar_phase_cos", "phrase_phase_sin", "phrase_phase_cos",
)
RHYTHM_CONTEXT_KEYS = (
    "beat_onset_local", "beat_onset_half", "beat_flux_local", "beat_rms_local",
)
RHYTHM_OFFSET = ACOUSTIC_DIM
CONTEXT_OFFSET = ACOUSTIC_DIM + len(RHYTHM_CONTEXT_KEYS)
FEATURE_KEYS = tuple(
    [f"logmel_{index:02d}" for index in range(N_MELS)]
    + [f"chroma_{index:02d}" for index in range(N_CHROMA)]
    + list(BASE_ACOUSTIC_KEYS)
    + list(STRUCTURE_KEYS)
    + list(RHYTHM_CONTEXT_KEYS)
    + list(CONTEXT_KEYS)
)
INPUT_DIM = len(FEATURE_KEYS)
# Full-song memory resolution.  At 64 frames a 160 s song got one token per
# ~2.5 s (about eight beats at 195 BPM), which is too coarse to compare the
# current phrase with an earlier one.  256 frames gives roughly one token per
# two beats, at a memory cost of 256 x 112 x 4 B ~= 112 KB per song.
GLOBAL_MEMORY_FRAMES = 256

# The previous feature release had 95 acoustic columns (no structure channels)
# and 5 timing columns (no beat/bar/phrase phase), for 100 inputs.  The layout
# below is a strict prefix-plus-suffix of that one, so a 100-dim checkpoint can
# still be generated from: the base 95 columns are produced by the same
# normalization, and the first five timing columns are unchanged.
LEGACY_ACOUSTIC_DIM = 95
LEGACY_INPUT_DIM = 100


def legacy_feature_view(tensor: "np.ndarray") -> "np.ndarray":
    """Reduce the current feature tensor to the pre-v3 100-column layout.

    Used only to generate with checkpoints trained before the structure and
    phase channels existed.  The base 95 columns come from the same
    normalization and the first five timing columns are unchanged, so values are
    identical to what such a checkpoint was trained on.
    """

    if tensor.shape[-1] < ACOUSTIC_DIM + len(CONTEXT_KEYS):
        raise ValueError(
            f"legacy_feature_view expects at least {ACOUSTIC_DIM + len(CONTEXT_KEYS)} columns, "
            f"got {tensor.shape[-1]}"
        )
    left = tensor[..., :LEGACY_ACOUSTIC_DIM]
    right = tensor[..., ACOUSTIC_DIM + len(RHYTHM_CONTEXT_KEYS):
                   ACOUSTIC_DIM + len(RHYTHM_CONTEXT_KEYS) + 5]
    if tensor.__class__.__module__.startswith("torch"):
        import torch

        return torch.cat([left, right], dim=-1)
    return np.concatenate([left, right], axis=-1)

TempoEntry = tuple[Fraction, float]
TempoMapLike = Sequence[TempoEntry] | Callable[[float], float] | None


@dataclass(slots=True)
class EventAudio:
    """Dense audio features and conservative audio-time tempo metadata.

    ``features`` contains only the 98-dimensional acoustic base.  The
    timing-dependent five columns are intentionally produced by
    ``audio_segment_tensor`` because they depend on chart GT or a user BPM
    override, neither of which belongs in an audio cache.
    """

    features: np.ndarray
    times: np.ndarray
    duration_sec: float
    tempo_bpm: float
    tempo_changes: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.features = np.asarray(self.features, dtype=np.float32)
        self.times = np.asarray(self.times, dtype=np.float32)
        if self.features.ndim != 2 or self.features.shape[1] != ACOUSTIC_DIM:
            raise ValueError(f"features must have shape [T, {ACOUSTIC_DIM}].")
        if self.times.ndim != 1 or self.times.shape[0] != self.features.shape[0]:
            raise ValueError("times must have one value per feature frame.")
        self.duration_sec = max(0.0, float(self.duration_sec))
        self.tempo_bpm = max(0.0, float(self.tempo_bpm))
        self.tempo_changes = [_tempo_change_dict(item) for item in self.tempo_changes]


def audio_cache_dir(cache_dir: str | Path) -> Path:
    """Canonical location of the per-song feature cache inside a cache folder.

    Every writer and reader must agree on this, otherwise a song is extracted
    twice: the training dataset looked in ``<cache>/audio`` while the gold-stats
    and acceptance paths wrote directly into ``<cache>``, so those features were
    never found again (25 orphan files were sitting in one dataset's cache root
    while training re-extracted the same songs).
    """

    return Path(cache_dir) / "audio"


def extract_event_audio(
    audio_path: str | Path,
    cache_dir: str | Path | None = None,
    sample_rate: int = 22050,
    hop_length: int = 441,
) -> EventAudio:
    """Extract dense 50 Hz event features, using a content/configured cache.

    Cache keys include the complete file content digest and every feature
    configuration value.  A cache written for another sample rate, hop, or
    feature version is never reused, even if its filename happens to match.
    """

    source = Path(audio_path)
    if sample_rate <= 0 or hop_length <= 0:
        raise ValueError("sample_rate and hop_length must be positive.")
    content_sha256 = _sha256_file(source)
    config = _feature_config(sample_rate, hop_length)
    cache_path = _cache_path(cache_dir, source, content_sha256, config)
    if cache_path is not None and cache_path.exists():
        try:
            return load_event_audio(cache_path, sample_rate=sample_rate, hop_length=hop_length)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            # A partial/corrupt cache is replaceable; a mismatched cache is
            # rejected by load_event_audio rather than silently reused.
            pass

    librosa = _import_librosa()
    y, sr = librosa.load(source, sr=int(sample_rate), mono=True)
    y = np.asarray(y, dtype=np.float32)
    duration_sec = float(len(y)) / float(sr) if len(y) else 0.0
    # Empty and very short files still get a stable [T, 95] tensor.  Librosa
    # STFT/mel routines need enough samples for their default window.
    minimum_samples = max(2048, int(sr * 0.02))
    analysis_y = np.pad(y, (0, max(0, minimum_samples - len(y))))
    n_fft = 2048
    target_frames = max(1, int(math.ceil(max(duration_sec, 1.0 / FEATURE_RATE_HZ) * FEATURE_RATE_HZ)))
    mel = librosa.feature.melspectrogram(
        y=analysis_y,
        sr=sr,
        n_fft=n_fft,
        hop_length=int(hop_length),
        n_mels=N_MELS,
        power=2.0,
        center=True,
    )
    logmel = librosa.power_to_db(mel, ref=np.max).T
    chroma = librosa.feature.chroma_stft(
        y=analysis_y,
        sr=sr,
        n_fft=n_fft,
        hop_length=int(hop_length),
        n_chroma=N_CHROMA,
    ).T
    onset = np.asarray(librosa.onset.onset_strength(y=analysis_y, sr=sr, hop_length=hop_length), dtype=np.float32)
    rms = np.asarray(librosa.feature.rms(y=analysis_y, frame_length=n_fft, hop_length=hop_length, center=True)[0], dtype=np.float32)
    flux = _spectral_flux(analysis_y, sr, hop_length, n_fft)

    count = max(target_frames, logmel.shape[0], chroma.shape[0], len(onset), len(rms), len(flux))
    logmel_padded = _pad_or_trim(logmel, count)
    chroma_padded = _pad_or_trim(chroma, count)
    onset_padded = _pad_or_trim(onset.reshape(-1, 1), count).reshape(-1)
    rms_padded = _pad_or_trim(rms.reshape(-1, 1), count).reshape(-1)
    flux_padded = _pad_or_trim(flux.reshape(-1, 1), count).reshape(-1)
    base = np.column_stack(
        [logmel_padded, chroma_padded, onset_padded, rms_padded, flux_padded]
    )
    base = _robust_normalize(base)
    # The structure channels keep their own readable scales instead of joining
    # the robust normalization: 1.0 is defined as "the song's median loudness",
    # which stays true no matter how the per-column IQR behaves.
    structure = _structure_channels(rms_padded, chroma_padded, frame_rate=float(sr) / float(hop_length))
    acoustic = np.concatenate([base, structure], axis=1).astype(np.float32, copy=False)
    # 50 Hz is an explicit contract. Resampling after normalization keeps all
    # columns aligned even when librosa's hop does not exactly equal 20 ms.
    times = np.arange(target_frames, dtype=np.float32) / FEATURE_RATE_HZ
    source_times = np.arange(count, dtype=np.float32) * float(hop_length) / float(sr)
    features = _resample_rows(acoustic, source_times, times)
    tempo_bpm, tempo_changes = _conservative_tempo(librosa, analysis_y, sr, hop_length, duration_sec)
    result = EventAudio(
        features=features,
        times=times,
        duration_sec=duration_sec,
        tempo_bpm=tempo_bpm,
        tempo_changes=tempo_changes,
        metadata={
            "feature_version": FEATURE_VERSION,
            "sample_rate": int(sr),
            "hop_length": int(hop_length),
            "feature_rate_hz": FEATURE_RATE_HZ,
            "acoustic_dim": ACOUSTIC_DIM,
            "input_dim": INPUT_DIM,
            "audio_sha256": content_sha256,
            "feature_config": config,
            "tempo_policy": TEMPO_POLICY,
        },
    )
    if cache_path is not None:
        save_event_audio(result, cache_path)
    return result


def audio_segment_tensor(
    audio: EventAudio,
    start_sec: float,
    end_sec: float,
    *,
    song_duration: float | None = None,
    bpm: float | None = None,
    start_beat: float | Fraction = 0,
    first: float = 0.0,
    tempo_changes: TempoMapLike = None,
) -> "Any":
    """Return ``[T, INPUT_DIM]`` audio plus chart-clock context.

    ``first`` is the full-song audio timestamp of chart beat 0.  ``start_beat``
    is the full-song beat at ``start_sec``; it does not reset beat phase.
    Prefer an explicit GT ``tempo_changes`` list during training.  A callback
    receives full-song beat and returns local BPM, which is useful for a
    continuously warped training map.  A constant BPM is used only when no
    map is supplied.
    """

    torch = _import_torch()
    # A chart may legitimately start before the audio does: a negative `first`
    # (or a leading negative delay folded into it) expresses the blank space a
    # machine needs in front of the chart body, and that blank space behaves
    # exactly like silent audio prepended to the file.  Clamping the start to 0
    # instead folded those beats onto the very first frame, which shifted the
    # whole window's music relative to the notes; the pre-audio region is filled
    # with silence below.
    start = float(start_sec)
    end = max(start, float(end_sec))
    duration = max(float(song_duration if song_duration is not None else audio.duration_sec), 1e-9)
    if bpm is not None and float(bpm) <= 0:
        raise ValueError("bpm must be positive when supplied.")
    if not math.isfinite(float(first)):
        raise ValueError("first must be finite.")
    selected = _sample_audio(audio, start, end)
    times = selected[0]
    acoustic = selected[1]
    if start < 0.0:
        acoustic = acoustic.copy()
        acoustic[times < 0.0] = 0.0
    map_entries = _normalize_tempo_map(tempo_changes, bpm or audio.tempo_bpm or 120.0)
    context = _timing_context(
        times,
        duration=duration,
        first=first,
        start_sec=start,
        start_beat=float(start_beat),
        map_entries=map_entries,
        tempo_changes=tempo_changes,
        fallback_bpm=bpm or audio.tempo_bpm or 120.0,
    )
    beats = _beats_at_audio_offsets(np.asarray(times, dtype=np.float64) - float(first), map_entries)
    rhythm = _beat_normalized_rhythm(acoustic, np.asarray(times), beats, context[:, 2] * 240.0)
    rhythm[times < 0.0] = 0.0
    values = np.concatenate([acoustic, rhythm, context], axis=1).astype(np.float32, copy=False)
    if values.shape[1] != INPUT_DIM:
        raise AssertionError(f"event audio input dimension drifted: {values.shape[1]} != {INPUT_DIM}")
    return torch.from_numpy(values)


def compress_event_audio(
    audio: EventAudio,
    *,
    song_duration: float | None = None,
    bpm: float | None = None,
    first: float = 0.0,
    tempo_changes: TempoMapLike = None,
    frames: int = GLOBAL_MEMORY_FRAMES,
) -> "Any":
    """Create full-song global memory with the same ``INPUT_DIM`` columns.

    Acoustic rows are linearly resampled, while timing context is evaluated at
    each memory timestamp.  In particular, phase is never averaged over a
    long block and remains an exact chart-clock value at the memory centres.
    """

    if frames <= 0:
        raise ValueError("frames must be positive.")
    duration = max(float(song_duration if song_duration is not None else audio.duration_sec), 0.0)
    torch = _import_torch()
    fallback_bpm = bpm or audio.tempo_bpm or 120.0
    memory_tempo = _normalize_tempo_map(tempo_changes, fallback_bpm)
    # With a negative chart offset the song starts before the audio does, so the
    # memory begins at that earlier point and the missing part is silence --
    # otherwise the whole-song memory would start a second late relative to the
    # chart clock.
    origin = min(0.0, float(first))
    memory_start_beat = (
        _time_to_beat(origin - float(first), memory_tempo)
        if not callable(tempo_changes)
        else _callback_time_to_beat(origin - float(first), tempo_changes)
    )
    if duration <= 0:
        values = np.zeros((frames, ACOUSTIC_DIM), dtype=np.float32)
        times = np.zeros(frames, dtype=np.float32)
        context = _timing_context(
            times,
            duration=1.0,
            first=first,
            start_sec=origin,
            start_beat=memory_start_beat,
            map_entries=memory_tempo,
            tempo_changes=tempo_changes,
            fallback_bpm=fallback_bpm,
        )
        rhythm = np.zeros((frames, len(RHYTHM_CONTEXT_KEYS)), dtype=np.float32)
        return torch.from_numpy(np.concatenate([values, rhythm, context], axis=1).astype(np.float32))
    times = (origin + np.arange(frames, dtype=np.float32) * np.float32((duration - origin) / frames))
    acoustic = _resample_rows(audio.features, audio.times, times)
    if origin < 0.0:
        acoustic = acoustic.copy()
        acoustic[times < 0.0] = 0.0
    context = _timing_context(
        times,
        duration=duration,
        first=first,
        start_sec=0.0,
        start_beat=memory_start_beat,
        map_entries=memory_tempo,
        tempo_changes=tempo_changes,
        fallback_bpm=fallback_bpm,
    )
    beats = _beats_at_audio_offsets(np.asarray(times, dtype=np.float64) - float(first), memory_tempo)
    rhythm = _beat_normalized_rhythm(acoustic, np.asarray(times), beats, context[:, 2] * 240.0)
    if origin < 0.0:
        rhythm[times < 0.0] = 0.0
    return torch.from_numpy(np.concatenate([acoustic, rhythm, context], axis=1).astype(np.float32))


def save_event_audio(audio: EventAudio, path: str | Path) -> None:
    """Save an EventAudio cache as a JSON manifest plus compressed NPZ data."""

    target = Path(path)
    if target.suffix.lower() != ".json":
        raise ValueError("EventAudio cache path must use the .json suffix.")
    target.parent.mkdir(parents=True, exist_ok=True)
    npz_path = target.with_suffix(".npz")
    metadata = dict(audio.metadata)
    metadata.setdefault("feature_version", FEATURE_VERSION)
    metadata.setdefault("input_dim", INPUT_DIM)
    metadata.setdefault(
        "feature_config",
        _feature_config(
            int(metadata.get("sample_rate", 22050)),
            int(metadata.get("hop_length", 441)),
        ),
    )
    payload = {
        "schema": FEATURE_VERSION,
        "features_npz": npz_path.name,
        "times": audio.times.tolist(),
        "duration_sec": audio.duration_sec,
        "tempo_bpm": audio.tempo_bpm,
        "tempo_changes": audio.tempo_changes,
        "metadata": _jsonable(metadata),
    }
    _atomic_save_npz(npz_path, features=audio.features.astype(np.float32), times=audio.times.astype(np.float32))
    _atomic_write_text(target, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def event_audio_cache_path(
    audio_path: str | Path,
    cache_dir: str | Path | None,
    *,
    sample_rate: int = 22050,
    hop_length: int = 441,
) -> Path | None:
    """Return the cache manifest path ``extract_event_audio`` would use.

    A caller that keeps its own bounded cache of ``EventAudio`` objects can
    resolve this once per song and then reload a spilled song straight from the
    cache, skipping the source read and content hash on every miss.
    """

    if cache_dir is None:
        return None
    source = Path(audio_path)
    return _cache_path(cache_dir, source, _sha256_file(source), _feature_config(sample_rate, hop_length))


def load_event_audio(
    path: str | Path,
    *,
    sample_rate: int | None = None,
    hop_length: int | None = None,
) -> EventAudio:
    """Load and validate an EventAudio JSON/NPZ cache pair.

    ``sample_rate`` and ``hop_length`` may be supplied by callers that require
    an exact configuration.  The automatic extractor already enforces this
    through its content/configuration-addressed cache filename.
    """

    return _load_event_audio(path, sample_rate=sample_rate, hop_length=hop_length)


def save_event_audio_json(audio: EventAudio, path: str | Path) -> None:
    """Explicit JSON-manifest cache API; the matching NPZ is written beside it."""

    save_event_audio(audio, path)


def load_event_audio_json(
    path: str | Path,
    *,
    sample_rate: int | None = None,
    hop_length: int | None = None,
) -> EventAudio:
    """Explicit JSON-manifest cache API."""

    return load_event_audio(path, sample_rate=sample_rate, hop_length=hop_length)


def save_event_audio_npz(audio: EventAudio, path: str | Path) -> None:
    """Save a self-contained NPZ cache for pipelines that do not need JSON."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(audio.metadata)
    metadata.setdefault("feature_version", FEATURE_VERSION)
    metadata.setdefault("input_dim", INPUT_DIM)
    metadata.setdefault(
        "feature_config",
        _feature_config(int(metadata.get("sample_rate", 22050)), int(metadata.get("hop_length", 441))),
    )
    _atomic_save_npz(
        target,
        features=audio.features.astype(np.float32),
        times=audio.times.astype(np.float32),
        duration_sec=np.asarray(audio.duration_sec, dtype=np.float64),
        tempo_bpm=np.asarray(audio.tempo_bpm, dtype=np.float64),
        tempo_changes_json=np.asarray(json.dumps(_jsonable(audio.tempo_changes))),
        metadata_json=np.asarray(json.dumps(_jsonable(metadata))),
    )


def load_event_audio_npz(
    path: str | Path,
    *,
    sample_rate: int | None = None,
    hop_length: int | None = None,
) -> EventAudio:
    """Load a self-contained NPZ cache and validate its version/config."""

    with np.load(Path(path), allow_pickle=False) as archive:
        metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
        if metadata.get("feature_version") != FEATURE_VERSION:
            raise ValueError("EventAudio NPZ has missing or incompatible metadata version.")
        config = metadata.get("feature_config")
        if not isinstance(config, dict) or config != _feature_config(int(config.get("sample_rate", 0)), int(config.get("hop_length", 0))):
            raise ValueError("EventAudio NPZ has incomplete feature configuration metadata.")
        if sample_rate is not None and int(config["sample_rate"]) != int(sample_rate):
            raise ValueError("EventAudio NPZ feature configuration does not match sample_rate.")
        if hop_length is not None and int(config["hop_length"]) != int(hop_length):
            raise ValueError("EventAudio NPZ feature configuration does not match hop_length.")
        return EventAudio(
            features=np.asarray(archive["features"], dtype=np.float32),
            times=np.asarray(archive["times"], dtype=np.float32),
            duration_sec=float(np.asarray(archive["duration_sec"]).item()),
            tempo_bpm=float(np.asarray(archive["tempo_bpm"]).item()),
            tempo_changes=json.loads(str(np.asarray(archive["tempo_changes_json"]).item())),
            metadata=metadata,
        )


def _load_event_audio(
    path: str | Path,
    *,
    sample_rate: int | None = None,
    hop_length: int | None = None,
) -> EventAudio:
    """Implementation shared by the public loader and strict callers."""

    target = Path(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    if payload.get("schema") != FEATURE_VERSION:
        raise ValueError(f"Unsupported EventAudio schema: {payload.get('schema')!r}")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("feature_version") != FEATURE_VERSION:
        raise ValueError("EventAudio cache has missing or incompatible metadata version.")
    config = metadata.get("feature_config")
    if not isinstance(config, dict) or config != _feature_config(int(config.get("sample_rate", 0)), int(config.get("hop_length", 0))):
        raise ValueError("EventAudio cache has incomplete feature configuration metadata.")
    if sample_rate is not None and int(config.get("sample_rate", 0)) != int(sample_rate):
        raise ValueError("EventAudio cache feature configuration does not match sample_rate.")
    if hop_length is not None and int(config.get("hop_length", 0)) != int(hop_length):
        raise ValueError("EventAudio cache feature configuration does not match hop_length.")
    if sample_rate is not None and int(config.get("sample_rate", 0)) != int(sample_rate):
        raise ValueError("EventAudio cache feature configuration does not match sample_rate.")
    if hop_length is not None and int(config.get("hop_length", 0)) != int(hop_length):
        raise ValueError("EventAudio cache feature configuration does not match hop_length.")
    npz_path = target.parent / str(payload.get("features_npz", target.with_suffix(".npz").name))
    with np.load(npz_path, allow_pickle=False) as archive:
        features = np.asarray(archive["features"], dtype=np.float32)
        npz_times = np.asarray(archive["times"], dtype=np.float32)
    times = np.asarray(payload.get("times", npz_times), dtype=np.float32)
    if not np.array_equal(times, npz_times):
        raise ValueError("EventAudio JSON and NPZ times disagree.")
    return EventAudio(
        features=features,
        times=times,
        duration_sec=float(payload.get("duration_sec", 0.0)),
        tempo_bpm=float(payload.get("tempo_bpm", 0.0)),
        tempo_changes=list(payload.get("tempo_changes", [])),
        metadata=metadata,
    )


def _feature_config(sample_rate: int, hop_length: int) -> dict[str, Any]:
    return {
        "feature_version": FEATURE_VERSION,
        "sample_rate": int(sample_rate),
        "hop_length": int(hop_length),
        "feature_rate_hz": FEATURE_RATE_HZ,
        "n_fft": 2048,
        "n_mels": N_MELS,
        "n_chroma": N_CHROMA,
        "normalization": "median_iqr_per_channel_v1",
        "tempo_policy": TEMPO_POLICY,
    }


def _cache_path(cache_dir: str | Path | None, source: Path, digest: str, config: Mapping[str, Any]) -> Path | None:
    if cache_dir is None:
        return None
    token = hashlib.sha256(json.dumps({"sha256": digest, "config": config}, sort_keys=True).encode()).hexdigest()[:24]
    return Path(cache_dir) / f"{source.stem}.{token}.event_audio.json"


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _import_librosa():
    try:
        import librosa
    except ImportError as exc:  # pragma: no cover - depends on installation
        raise RuntimeError('EventAudio requires librosa; install the optional audio dependencies with `pip install -e ".[audio]"`.') from exc
    return librosa


def _import_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency boundary
        raise RuntimeError("audio_segment_tensor requires torch.") from exc
    return torch


def _spectral_flux(y: np.ndarray, sr: int, hop: int, n_fft: int) -> np.ndarray:
    librosa = _import_librosa()
    spectrum = np.abs(librosa.stft(y=y, n_fft=n_fft, hop_length=hop, center=True))
    normalized = spectrum / np.maximum(spectrum.sum(axis=0, keepdims=True), 1e-8)
    delta = np.diff(normalized, axis=1, prepend=normalized[:, :1])
    return np.sqrt(np.maximum(delta, 0.0).sum(axis=0)).astype(np.float32)


def _pad_or_trim(values: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(values)
    if values.shape[0] >= count:
        return values[:count]
    if values.shape[0] == 0:
        shape = (count,) + values.shape[1:]
        return np.zeros(shape, dtype=values.dtype if values.dtype != object else np.float32)
    padding = np.repeat(values[-1:], count - values.shape[0], axis=0)
    return np.concatenate([values, padding], axis=0)


def _beat_normalized_rhythm(acoustic: np.ndarray, times: np.ndarray, beats: np.ndarray, bpms: np.ndarray) -> np.ndarray:
    """Compute local rhythm over fractions of the current beat, not fixed seconds."""
    if len(acoustic) == 0:
        return np.zeros((0, len(RHYTHM_CONTEXT_KEYS)), dtype=np.float32)
    times = np.asarray(times, dtype=np.float64)
    beats = np.asarray(beats, dtype=np.float64)
    bpms = np.asarray(bpms, dtype=np.float64)
    onset = acoustic[:, N_MELS + N_CHROMA]
    flux = acoustic[:, N_MELS + N_CHROMA + 2]
    rms = acoustic[:, N_MELS + N_CHROMA + 1]
    # Derive the local beat length from the same beat clock used by the phase
    # features.  This makes the rhythm window follow tempo-map boundaries
    # exactly; the BPM column remains a safe fallback for degenerate/single
    # frame inputs.
    beat_sec = 60.0 / np.maximum(bpms, 1e-6)
    if len(times) > 1 and beats.shape == times.shape:
        derivative = np.gradient(beats, times, edge_order=1)
        from_beats = 1.0 / np.maximum(derivative, 1e-6)
        valid = np.isfinite(from_beats) & (derivative > 1e-6)
        beat_sec = np.where(valid, from_beats, beat_sec)
    result = np.empty((len(times), 4), dtype=np.float32)
    # Prefix sums keep reductions constant-time once monotone window endpoints
    # are known. The two-pointer sweep below makes the complete pass O(T),
    # avoiding a fresh O(T) mask for every frame.
    onset_sum = np.concatenate(([0.0], np.cumsum(onset, dtype=np.float64)))
    flux_sum = np.concatenate(([0.0], np.cumsum(flux, dtype=np.float64)))
    rms_sum = np.concatenate(([0.0], np.cumsum(rms, dtype=np.float64)))
    quarter_left = quarter_right = half_left = half_right = 0
    for index, stamp in enumerate(times):
        local = max(float(beat_sec[index]), 1.0 / FEATURE_RATE_HZ)
        quarter_start, quarter_end = stamp - 0.25 * local, stamp + 0.25 * local
        half_start, half_end = stamp - 0.5 * local, stamp + 0.5 * local
        while quarter_left < len(times) and times[quarter_left] < quarter_start:
            quarter_left += 1
        while quarter_right < len(times) and times[quarter_right] <= quarter_end:
            quarter_right += 1
        while half_left < len(times) and times[half_left] < half_start:
            half_left += 1
        while half_right < len(times) and times[half_right] <= half_end:
            half_right += 1
        narrow_count = quarter_right - quarter_left
        wide_count = half_right - half_left
        result[index] = (
            (onset_sum[quarter_right] - onset_sum[quarter_left]) / narrow_count
            if narrow_count else onset[index],
            (onset_sum[half_right] - onset_sum[half_left]) / wide_count
            if wide_count else onset[index],
            (flux_sum[quarter_right] - flux_sum[quarter_left]) / narrow_count
            if narrow_count else flux[index],
            (rms_sum[half_right] - rms_sum[half_left]) / wide_count
            if wide_count else rms[index],
        )
    return result

def _structure_channels(rms: np.ndarray, chroma: np.ndarray, *, frame_rate: float) -> np.ndarray:

    """Song-relative intensity, trend, and harmonic novelty."""

    frames = len(rms)
    if frames == 0:
        return np.zeros((0, len(STRUCTURE_KEYS)), dtype=np.float32)
    # RMS relative to the whole song's median: 1.0 == the song's own median
    # loudness, 2.0 == twice as loud as the median, 0.0 == silence.
    median_rms = float(np.median(rms))
    relative = np.clip(rms / max(median_rms, 1e-6), 0.0, 4.0)
    # One-second trend of the relative intensity.
    trend_lag = max(1, int(round(float(frame_rate))))
    slope = np.zeros_like(relative)
    if frames > trend_lag:
        slope[trend_lag:] = np.clip(relative[trend_lag:] - relative[:-trend_lag], -2.0, 2.0)
    # Harmonic novelty: cosine distance between the average chroma of the two
    # seconds *before* a frame and the two seconds *after* it.  Averaging the
    # windows (instead of comparing single frames) cancels the per-chord
    # movement inside a section, so this stays near 0 while a passage repeats
    # itself and rises only where the harmony actually changes -- a cheap,
    # robust section-boundary signal.  Cumulative sums keep it O(frames).
    novelty = np.zeros(frames, dtype=np.float32)
    novelty_window = max(1, int(round(2.0 * float(frame_rate))))
    if frames > 2 * novelty_window:
        positive = np.clip(chroma, 0.0, None).astype(np.float64)
        cumulative = np.vstack([np.zeros((1, positive.shape[1]), dtype=np.float64),
                                np.cumsum(positive, axis=0, dtype=np.float64)])
        index = np.arange(frames)
        before_start = np.maximum(0, index - novelty_window)
        after_end = np.minimum(frames, index + novelty_window)
        before = (cumulative[index] - cumulative[before_start]) / np.maximum(index - before_start, 1)[:, None]
        after = (cumulative[after_end] - cumulative[index]) / np.maximum(after_end - index, 1)[:, None]
        before_norm = np.linalg.norm(before, axis=1)
        after_norm = np.linalg.norm(after, axis=1)
        denominator = np.maximum(before_norm * after_norm, 1e-12)
        similarity = np.einsum("ij,ij->i", before, after) / denominator
        similarity = np.where(np.maximum(before_norm, after_norm) < 1e-6, 1.0, similarity)
        novelty = np.clip(1.0 - similarity, 0.0, 1.0).astype(np.float32)
    return np.column_stack([relative, slope, novelty]).astype(np.float32, copy=False)


def _robust_normalize(values: np.ndarray) -> np.ndarray:
    median = np.nanmedian(values, axis=0, keepdims=True)
    q25 = np.nanpercentile(values, 25, axis=0, keepdims=True)
    q75 = np.nanpercentile(values, 75, axis=0, keepdims=True)
    scale = np.maximum(q75 - q25, 1e-4)
    normalized = (np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0) - median) / scale
    return np.clip(normalized, -8.0, 8.0).astype(np.float32)


def _resample_rows(values: np.ndarray, source_times: np.ndarray, target_times: np.ndarray,
                   *, fill_outside: tuple[float, float] | None = None) -> np.ndarray:
    """Linear interpolation onto ``target_times``, all columns in one pass.

    ``np.interp`` handles one column at a time, so the old implementation made 98
    Python calls per sample; with the loader on a prefetch thread that is 98 GIL
    round-trips competing with the CUDA submissions on the main thread, and it
    showed up as the producer never getting ahead of the GPU.  Same arithmetic
    (float64, clamped outside the source range) in a single vectorised pass.
    """

    if len(values) == 1:
        targets = np.asarray(target_times, dtype=np.float64)
        rows = np.repeat(values, len(targets), axis=0).astype(np.float32)
        low, high = (map(float, fill_outside) if fill_outside is not None
                     else (float(np.asarray(source_times, dtype=np.float64)[0]),
                           float(np.asarray(source_times, dtype=np.float64)[0])))
        rows[(targets < low) | (targets > high)] = 0.0
        return rows
    source = np.asarray(source_times, dtype=np.float64)
    targets = np.asarray(target_times, dtype=np.float64)
    if targets.size == 0:
        return np.zeros((0, values.shape[1]), dtype=np.float32)
    if source.size == 1:
        rows = np.repeat(values[:1], targets.size, axis=0).astype(np.float32)
        low, high = (map(float, fill_outside) if fill_outside is not None
                     else (float(source[0]), float(source[0])))
        rows[(targets < low) | (targets > high)] = 0.0
        return rows
    index = np.clip(np.searchsorted(source, targets, side="left"), 1, source.size - 1)
    left_time = source[index - 1]
    span = source[index] - left_time
    weight = np.where(span > 0, (targets - left_time) / np.where(span > 0, span, 1.0), 0.0)
    low = values[index - 1].astype(np.float64)
    high = values[index].astype(np.float64)
    rows = low + (high - low) * weight[:, None]
    low, high = (map(float, fill_outside) if fill_outside is not None
                 else (float(source[0]), float(source[-1])))
    rows[(targets < low) | (targets > high)] = 0.0
    return rows.astype(np.float32)


def _conservative_tempo(librosa: Any, y: np.ndarray, sr: int, hop: int, duration: float) -> tuple[float, list[dict[str, Any]]]:
    """Detect the global tempo and report local candidates.

    The scalar comes from :func:`maichart.generation.tempo.detect_global_bpm`,
    which searches a continuous tempo axis and resolves the octave by harmonic
    summation.  Reading ``librosa.beat.beat_track``'s scalar directly instead
    returns an autocorrelation lag bin: measured against 25 authored charts that
    path matched none of them, because one lag step near 130 BPM is 5.4 BPM.

    Beat times are still taken from the tracker, since onset-level alignment is
    what they are good for, but the *clock* comes from the continuous estimate.
    """

    if duration < 0.5 or np.max(np.abs(y)) < 1e-5:
        return 0.0, []
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    fps = float(sr) / float(hop)
    detected, confidence = detect_global_bpm(np.asarray(onset, dtype=np.float64), fps=fps)
    if detected <= 0:
        return 0.0, []
    _, beats = librosa.beat.beat_track(onset_envelope=onset, sr=sr, hop_length=hop, units="frames")
    beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop)
    candidates = tempo_change_candidates([float(value) for value in beat_times], detected)
    if candidates:
        candidates[0]["confidence"] = round(
            min(1.0, max(float(candidates[0].get("confidence", 0.0)), confidence)), 5
        )
    return detected, candidates


def _tempo_change_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        beat = float(value.get("beat", 0.0))
        bpm = float(value.get("bpm", 0.0))
        result = dict(value)
    elif hasattr(value, "beat") and hasattr(value, "bpm"):
        beat = float(value.beat)
        bpm = float(value.bpm)
        result = {"beat": beat, "bpm": bpm}
    else:
        beat, bpm = value
        result = {"beat": float(beat), "bpm": float(bpm)}
    if bpm <= 0:
        raise ValueError("tempo_changes BPM values must be positive.")
    result["beat"] = beat
    result["bpm"] = bpm
    return result


def _normalize_tempo_map(value: TempoMapLike, fallback: float) -> list[TempoEntry]:
    if callable(value):
        return [(Fraction(0), float(fallback))]
    entries: list[TempoEntry] = []
    for raw in value or []:
        if isinstance(raw, Mapping):
            beat, bpm = raw.get("beat", 0), raw["bpm"]
        elif hasattr(raw, "beat") and hasattr(raw, "bpm"):
            beat, bpm = raw.beat, raw.bpm
        else:
            beat, bpm = raw
        bpm_value = float(bpm)
        if bpm_value <= 0:
            raise ValueError("tempo_changes BPM values must be positive.")
        entries.append((Fraction(beat), bpm_value))
    entries.sort(key=lambda item: item[0])
    if not entries:
        entries = [(Fraction(0), float(fallback))]
    elif entries[0][0] > 0:
        entries.insert(0, (Fraction(0), entries[0][1]))
    return entries


def _tempo_at_beat(beat: float, entries: Sequence[TempoEntry], fallback: float) -> float:
    result = float(fallback)
    for boundary, value in entries:
        if float(boundary) <= beat + 1e-9:
            result = value
        else:
            break
    return result


def _boundaries_are_sorted(entries: Sequence[TempoEntry]) -> bool:
    """True when tempo boundaries are non-decreasing (the parser's guarantee)."""

    previous = None
    for boundary, _ in entries:
        value = float(boundary)
        if previous is not None and value < previous:
            return False
        previous = value
    return True


def _beats_at_audio_offsets(offsets: "np.ndarray", entries: Sequence[TempoEntry]) -> "np.ndarray":
    """Vectorised :func:`_time_to_beat`.

    Identical arithmetic to the scalar version (same operand order, float64),
    just without the per-frame Python loop: the tempo map is piecewise constant,
    so beats are piecewise linear in time and one ``searchsorted`` finds the
    active interval for every frame at once.  The timing context is rebuilt for
    every training sample (420+ frames), so this loop used to dominate the
    dataloader -- 26 minutes of CPU per epoch on the full corpus.
    """

    import numpy as np

    values = np.asarray(offsets, dtype=np.float64)
    if not entries:
        return np.zeros_like(values)
    boundary_beats = np.array([float(boundary) for boundary, _ in entries], dtype=np.float64)
    boundary_bpms = np.array([float(value) for _, value in entries], dtype=np.float64)
    if len(entries) > 1:
        spans = np.maximum(0.0, np.diff(boundary_beats)) * 60.0 / boundary_bpms[:-1]
        elapsed = np.concatenate([[0.0], np.cumsum(spans)])
    else:
        elapsed = np.zeros(1, dtype=np.float64)
    result = np.empty_like(values)
    before = values <= 0
    result[before] = boundary_beats[0] + values[before] * boundary_bpms[0] / 60.0
    if not before.all():
        active = values[~before]
        index = np.clip(np.searchsorted(elapsed, active, side="right") - 1, 0, len(entries) - 1)
        result[~before] = (boundary_beats[index]
                           + (active - elapsed[index]) * boundary_bpms[index] / 60.0)
    return result


def _tempos_at_beats(beats: "np.ndarray", entries: Sequence[TempoEntry],
                     fallback: float) -> "np.ndarray":
    """Vectorised :func:`_tempo_at_beat` (last boundary at or before each beat)."""

    import numpy as np

    values = np.asarray(beats, dtype=np.float64)
    if not entries:
        return np.full(values.shape, float(fallback), dtype=np.float64)
    boundary_beats = np.array([float(boundary) for boundary, _ in entries], dtype=np.float64)
    boundary_bpms = np.array([float(value) for _, value in entries], dtype=np.float64)
    index = np.searchsorted(boundary_beats, values + 1e-9, side="right") - 1
    resolved = np.where(index >= 0, boundary_bpms[np.clip(index, 0, None)], float(fallback))
    return resolved.astype(np.float64)


def _timing_context(
    times: np.ndarray,
    *,
    duration: float,
    first: float,
    start_sec: float,
    start_beat: float,
    map_entries: Sequence[TempoEntry],
    tempo_changes: TempoMapLike,
    fallback_bpm: float,
) -> np.ndarray:
    # Callback tempo maps have no closed form, and an unordered boundary list
    # would break the break-on-first-greater semantics the scalar path encodes
    # (the parser always emits sorted boundaries, but a hand-built map may not).
    # Both fall back to the reference implementation rather than guessing.
    if callable(tempo_changes) or not _boundaries_are_sorted(map_entries):
        return _timing_context_scalar(
            times, duration=duration, first=first, start_sec=start_sec,
            start_beat=start_beat, map_entries=map_entries,
            tempo_changes=tempo_changes, fallback_bpm=fallback_bpm,
        )

    import numpy as np

    stamps = np.asarray(times, dtype=np.float64)
    total_beats = max(
        _beat_at_audio_time(float(duration), first, 0.0, 0.0, map_entries, tempo_changes),
        1.0,
    )
    mismatched = np.nonzero(np.abs(stamps - float(start_sec)) <= 1e-7)[0]
    if mismatched.size:
        resolved = _beats_at_audio_offsets(stamps[mismatched] - float(first), map_entries)
        if np.any(np.abs(resolved - float(start_beat)) > 0.05):
            raise ValueError(
                "start_beat does not agree with first and tempo_changes at start_sec; "
                "pass the full-song chart beat rather than resetting a segment to zero."
            )

    beats = _beats_at_audio_offsets(stamps - float(first), map_entries)
    local_bpms = _tempos_at_beats(beats, map_entries, fallback_bpm)
    if np.any(local_bpms <= 0):
        raise ValueError("tempo_changes callback must return a positive BPM.")

    phase = beats - np.floor(beats)
    context = np.empty((stamps.size, len(CONTEXT_KEYS)), dtype=np.float32)
    context[:, 0] = np.clip(stamps / max(duration, 1e-9), 0.0, 1.0)
    context[:, 1] = np.clip(duration / 300.0, 0.0, 1.0)
    context[:, 2] = np.clip(local_bpms / 240.0, 0.0, 2.0)
    context[:, 3] = np.sin(2.0 * np.pi * phase)
    context[:, 4] = np.cos(2.0 * np.pi * phase)
    context[:, 5] = np.clip(beats / total_beats, 0.0, 1.0)
    context[:, 6] = np.sin(2.0 * np.pi * beats / 4.0)
    context[:, 7] = np.cos(2.0 * np.pi * beats / 4.0)
    context[:, 8] = np.sin(2.0 * np.pi * beats / 16.0)
    context[:, 9] = np.cos(2.0 * np.pi * beats / 16.0)
    return context


def _timing_context_scalar(
    times: np.ndarray,
    *,
    duration: float,
    first: float,
    start_sec: float,
    start_beat: float,
    map_entries: Sequence[TempoEntry],
    tempo_changes: TempoMapLike,
    fallback_bpm: float,
) -> np.ndarray:
    """Reference implementation: one Python iteration per frame."""
    context = np.empty((len(times), len(CONTEXT_KEYS)), dtype=np.float32)
    # Total song length in beats, so every frame can report how far through
    # the *music* it is (beat progress), which differs from linear audio time
    # whenever the tempo map is not constant.
    total_beats = max(
        _beat_at_audio_time(float(duration), first, 0.0, 0.0, map_entries, tempo_changes),
        1.0,
    )
    for index, timestamp in enumerate(times):
        full_beat = _beat_at_audio_time(
            float(timestamp),
            first,
            start_beat,
            start_sec,
            map_entries,
            tempo_changes,
        )
        local_bpm = (
            float(tempo_changes(full_beat))
            if callable(tempo_changes)
            else _tempo_at_beat(full_beat, map_entries, fallback_bpm)
        )
        if local_bpm <= 0:
            raise ValueError("tempo_changes callback must return a positive BPM.")
        phase = full_beat - math.floor(full_beat)
        context[index] = (
            np.clip(float(timestamp) / max(duration, 1e-9), 0.0, 1.0),
            np.clip(duration / 300.0, 0.0, 1.0),
            np.clip(local_bpm / 240.0, 0.0, 2.0),
            math.sin(2.0 * math.pi * phase),
            math.cos(2.0 * math.pi * phase),
            # How far through the song in beats, then the phase within the
            # current 4-beat measure and the current 16-beat phrase.  A model
            # can use these to learn "measures start with taps" and "phrases
            # end with a rest", which linear time cannot express.
            np.clip(full_beat / total_beats, 0.0, 1.0),
            math.sin(2.0 * math.pi * full_beat / 4.0),
            math.cos(2.0 * math.pi * full_beat / 4.0),
            math.sin(2.0 * math.pi * full_beat / 16.0),
            math.cos(2.0 * math.pi * full_beat / 16.0),
        )
    return context


def _callback_time_to_beat(audio_offset: float, callback: Callable[[float], float]) -> float:
    """Integrate a beat->BPM callback from chart beat zero to an audio offset."""

    beat = 0.0
    elapsed = 0.0
    direction = 1.0 if audio_offset >= 0 else -1.0
    while (audio_offset - elapsed) * direction > 1e-7:
        delta = min(0.01, abs(audio_offset - elapsed)) * direction
        local_bpm = float(callback(beat))
        if local_bpm <= 0:
            raise ValueError("tempo_changes callback must return a positive BPM.")
        beat += delta * local_bpm / 60.0
        elapsed += delta
    return beat


def _beat_at_audio_time(timestamp: float, first: float, anchor_beat: float, start_sec: float, entries: Sequence[TempoEntry], original_map: TempoMapLike) -> float:
    """Resolve full-song beat phase from the declared timing anchors.

    For a list map, ``first`` is authoritative because it is the full-song
    beat-zero audio anchor. ``start_beat`` is checked at the segment start so a
    crop cannot accidentally reset its phase. Callback maps have no closed
    form, so they are advanced from the supplied segment anchor.
    """

    if callable(original_map):
        # Callback maps are evaluated with a small fixed-point integration.
        beat = anchor_beat
        current_time = start_sec
        step = 0.01
        direction = 1.0 if timestamp >= start_sec else -1.0
        while (timestamp - current_time) * direction > 1e-7:
            delta = min(step, abs(timestamp - current_time)) * direction
            local = max(1.0, float(original_map(beat)))
            beat += delta * local / 60.0
            current_time += delta
        return beat
    target = float(timestamp - first)
    resolved = _time_to_beat(target, entries)
    if abs(timestamp - start_sec) <= 1e-7 and abs(resolved - anchor_beat) > 0.05:
        raise ValueError(
            "start_beat does not agree with first and tempo_changes at start_sec; "
            "pass the full-song chart beat rather than resetting a segment to zero."
        )
    return resolved


def _time_to_beat(audio_offset: float, entries: Sequence[TempoEntry]) -> float:
    """Convert seconds from the full-song ``first`` anchor to chart beat.

    Each tempo boundary is a beat coordinate, so the seconds consumed before a
    boundary must use the BPM active in the preceding interval.  This is kept
    separate from the segment-anchor path to make both contracts explicit.
    """

    if not entries:
        return 0.0
    if audio_offset <= 0:
        return float(entries[0][0]) + audio_offset * entries[0][1] / 60.0
    elapsed = 0.0
    previous = float(entries[0][0])
    active_bpm = entries[0][1]
    for boundary, next_bpm in entries[1:]:
        boundary_beat = float(boundary)
        span = max(0.0, boundary_beat - previous) * 60.0 / active_bpm
        if elapsed + span >= audio_offset:
            return previous + (audio_offset - elapsed) * active_bpm / 60.0
        elapsed += span
        previous = boundary_beat
        active_bpm = next_bpm
    return previous + (audio_offset - elapsed) * active_bpm / 60.0


def _sample_audio(audio: EventAudio, start: float, end: float) -> tuple[np.ndarray, np.ndarray]:
    times = segment_frame_times(start, end)
    features = _resample_rows(audio.features, audio.times, times)
    return times, features


def segment_frame_times(start_sec: float, end_sec: float) -> np.ndarray:
    """Return the exact frame timestamps used by ``audio_segment_tensor``.

    Segment sampling is relative to its requested start, including negative
    starts. For a degenerate interval the feature path emits one frame at start.
    """
    start, end = float(start_sec), float(end_sec)
    if end <= start:
        return np.asarray([start], dtype=np.float32)
    count = max(1, int(math.ceil((end - start) * FEATURE_RATE_HZ - 1e-8)))
    return (start + np.arange(count, dtype=np.float32) / FEATURE_RATE_HZ).astype(np.float32)


def _atomic_write_text(path: Path, content: str) -> None:
    fd, temporary = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    """Write an npz atomically, using a unique temporary name per writer.

    Feature extraction is now run in parallel across songs, and two writers can
    legitimately reach the same destination at the same moment (duplicate audio
    files, or a retried batch).  A deterministic ``<name>.tmp`` would let them
    interleave into one file and publish a corrupt cache, so the temporary name
    carries the pid and a random suffix, and the final move stays atomic.
    """

    temporary = path.with_name(f"{path.name}.{os.getpid()}.{uuid4().hex[:8]}.tmp")
    generated = Path(str(temporary) + ".npz")
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(generated, path)
    finally:
        for leftover in (generated, temporary):
            if leftover.exists():
                try:
                    leftover.unlink()
                except OSError:
                    pass


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


__all__ = [
    "ACOUSTIC_DIM",
    "BASE_ACOUSTIC_KEYS",
    "CONTEXT_KEYS",
    "EventAudio",
    "FEATURE_KEYS",
    "FEATURE_RATE_HZ",
    "FEATURE_VERSION",
    "GLOBAL_MEMORY_FRAMES",
    "LEGACY_ACOUSTIC_DIM",
    "LEGACY_INPUT_DIM",
    "legacy_feature_view",
    "INPUT_DIM",
    "RELATIVE_RMS_COLUMN",
    "STRUCTURE_KEYS",
    "audio_segment_tensor",
    "compress_event_audio",
    "event_audio_cache_path",
    "extract_event_audio",
    "load_event_audio",
    "load_event_audio_json",
    "load_event_audio_npz",
    "save_event_audio",
    "save_event_audio_json",
    "save_event_audio_npz",
    "segment_frame_times",
]


