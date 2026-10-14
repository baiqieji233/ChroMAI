"""Nonuniform audio evidence queried on the exact musical clock.

Waveform samples are observations, never legal chart-position tokens. Acoustic
attacks and exact beat anchors supply memory; the decoder may emit any Fraction.
There is no frame-rate conversion, Gaussian chart-onset map, or grid snapping.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import tempfile

import numpy as np

from maichart.event_timing import seconds_at_beat

PROFILE = "rational_audio_queries_v1"
ACOUSTIC_DIM = 98
INPUT_DIM = 120
MAX_EVIDENCE_NODES = 8192
# Maximum native audio evidence nodes exposed to the whole-song Planner.  The
# same bound is used during training, curve prediction, and chart generation;
# it limits memory/latency without quantising the musical Fraction clock.
PLANNER_MAX_AUDIO_NODES = 1024


def _digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


@lru_cache(maxsize=2)
def _wave(path, size, mtime_ns, sha256):
    import librosa
    if _digest(path)!=sha256:
        raise ValueError('Audio content changed before waveform query; reload the audio source')
    signal, sr = librosa.load(path, sr=None, mono=True)
    return np.asarray(signal, dtype=np.float32), int(sr)


def waveform(path, sha256=None):
    path = Path(path).resolve()
    stat = path.stat()
    return _wave(str(path), stat.st_size, stat.st_mtime_ns,sha256 or _digest(path))


@lru_cache(maxsize=6)
def _filters(sr, size):
    import librosa
    return (librosa.filters.mel(sr=sr, n_fft=size, n_mels=80),
            librosa.filters.chroma(sr=sr, n_fft=size, n_chroma=12))


def query_spectra(signal, sr, seconds):
    """Measure spectra at requested centers, including a short transient view.

    Long and short analysis windows share a musical center; neither establishes
    a periodic output clock. Beyond-file centers are explicit silence.
    """
    seconds = np.asarray(seconds, dtype=np.float64)
    result = np.zeros((len(seconds), ACOUSTIC_DIM), dtype=np.float32)
    size = 2048
    mel_filter, chroma_filter = _filters(sr, size)
    offsets = np.arange(size) - size // 2
    window = np.hanning(size)
    short_size = max(16, round(sr * .006))
    for start in range(0, len(seconds), 128):
        centers = np.rint(seconds[start:start + 128] * sr).astype(np.int64)
        indices = centers[:, None] + offsets[None]
        inside = (indices >= 0) & (indices < len(signal))
        samples = signal[np.clip(indices, 0, max(0, len(signal) - 1))] * inside if len(signal) else np.zeros_like(indices, dtype=np.float32)
        spectrum = np.abs(np.fft.rfft(samples * window, axis=1)) ** 2
        mel = np.log1p(spectrum @ mel_filter.T)
        chroma = spectrum @ chroma_filter.T
        chroma /= np.maximum(chroma.sum(axis=1, keepdims=True), 1e-8)
        rms = np.sqrt(np.mean(samples ** 2, axis=1))
        middle = size // 2
        left = samples[:, max(0, middle-short_size):middle]
        right = samples[:, middle:min(size, middle+short_size)]
        attack = np.maximum(np.mean(right**2, axis=1)-np.mean(left**2, axis=1), 0) ** .5
        detail = np.sqrt(np.mean(np.diff(samples[:, middle-short_size:middle+short_size], axis=1)**2, axis=1))
        values = np.concatenate((mel, chroma, attack[:, None], rms[:, None], detail[:, None],
                                 np.zeros((len(samples), 3))), axis=1)
        values[(centers < 0) | (centers >= len(signal))] = 0
        result[start:start+len(samples)] = values
    return result


def attack_samples(signal, sr):
    """Find irregular acoustic evidence directly on the native sample clock."""
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import find_peaks
    if len(signal) < 2 or not np.any(signal):
        return np.zeros(0, dtype=np.int64)
    energy = uniform_filter1d(signal.astype(np.float64)**2, max(1, round(sr*.003)))
    previous = np.roll(energy, max(1, round(sr*.004)))
    previous[:max(1, round(sr*.004))] = 0
    novelty = np.maximum(energy-previous, 0)
    height = max(float(np.percentile(novelty, 85)), float(novelty.max())*.002, 1e-12)
    peaks, props = find_peaks(novelty, height=height, distance=max(1, round(sr*.006)))
    if len(peaks) > MAX_EVIDENCE_NODES:
        chosen = np.argsort(props["peak_heights"], kind="stable")[-MAX_EVIDENCE_NODES:]
        peaks = np.sort(peaks[chosen])
    return peaks.astype(np.int64)


def _tempo_hint(signal, sr):
    """Estimate a global musical tempo for metadata and UI hints.

    The attack-node spacing is deliberately *not* a tempo estimator: a chart or
    recording can contain eighth notes, triplets, fills, and several voices, so
    its median inter-attack interval routinely selects a faster subdivision
    (for example 189 BPM for a 150 BPM song).  We use the existing global tempo
    detector on a temporary 200 Hz onset envelope instead.  This envelope is
    only an analysis aid for the scalar hint; rational audio queries still use
    native sample attacks and exact Fraction beat positions.
    """
    try:
        import librosa
        from maichart.generation.tempo import detect_global_bpm
        hop = max(1, round(float(sr) / 200.0))
        onset = librosa.onset.onset_strength(
            y=np.asarray(signal, dtype=np.float32), sr=int(sr),
            hop_length=hop, aggregate=np.median,
        )
        bpm, confidence = detect_global_bpm(onset, fps=float(sr) / hop)
        if bpm > 0 and np.isfinite(bpm) and confidence >= 0:
            return float(bpm)
    except Exception:
        # Metadata estimation must not make an otherwise readable audio file
        # unusable.  Keep a conservative UI hint when the optional detector is
        # unavailable; no chart positions depend on this fallback.
        pass
    return 120.0


def cache_path(path, cache_dir):
    if cache_dir is None: return None
    identity = {"profile": PROFILE, "source": str(Path(path).resolve()), "audio_sha256": _digest(path)}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return Path(cache_dir) / PROFILE / (key+".npz")


@dataclass
class RationalAudio:
    features: np.ndarray
    times: np.ndarray
    duration_sec: float
    tempo_bpm: float
    metadata: dict
    source: str
    cache_dir: str | None = None
    tempo_changes: list = field(default_factory=list)
    _views: OrderedDict = field(default_factory=OrderedDict, repr=False)


def load_rational_audio(path):
    with np.load(path, allow_pickle=False) as saved:
        meta = json.loads(str(saved["metadata"].item()))
        features, times = saved["features"].copy(), saved["times"].copy()
    if meta.get("profile") != PROFILE or features.ndim != 2 or features.shape[1] != ACOUSTIC_DIM:
        raise ValueError("Rational audio cache contract mismatch")
    if len(times) != len(features) or not np.isfinite(features).all() or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Invalid rational audio cache")
    return RationalAudio(features, times, meta["duration_sec"], meta["tempo_bpm"], meta,
                         meta["source"], meta.get("cache_dir"))


def extract_rational_audio(path, cache_dir=None):
    path = Path(path).resolve()
    cached = cache_path(path, cache_dir)
    if cached is not None and cached.is_file():
        try: return load_rational_audio(cached)
        except (OSError, ValueError, KeyError): pass
    content_sha256=_digest(path)
    signal, sr = waveform(path,content_sha256)
    samples = np.unique(np.r_[0, attack_samples(signal, sr), max(0, len(signal)-1)])
    times = samples.astype(np.float64) / sr
    features = query_spectra(signal, sr, times)
    scale = np.maximum(np.percentile(features[:, :95], 95, axis=0), 1e-5)
    features[:, :95] /= scale
    rms = features[:, 93]
    features[:, 95] = rms / max(float(np.median(rms[rms > 0])) if np.any(rms > 0) else 1, 1e-5)
    if len(times) > 1:
        features[:, 96] = np.clip(np.gradient(rms, times), -8, 8)
        features[1:, 97] = np.linalg.norm(np.diff(features[:, 80:92], axis=0), axis=1)
    meta = dict(profile=PROFILE, audio_sha256=content_sha256, source=str(path), sample_rate=sr,
                duration_sec=len(signal)/sr, tempo_bpm=_tempo_hint(signal, sr),
                normalization_scale=scale.tolist(), cache_dir=str(cache_dir) if cache_dir is not None else None,
                timing="native_sample_attacks_and_exact_fraction_queries", feature_rate_hz=None)
    audio = RationalAudio(features, times, meta["duration_sec"], meta["tempo_bpm"], meta, str(path), meta["cache_dir"])
    if cached is not None:
        cached.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=cached.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, features=features, times=times, metadata=json.dumps(meta))
        temporary.replace(cached)
    return audio


def beat_at_second(second, tempo, first):
    """Integrate decimal BPM values with Fractions, without denominator snapping."""
    timestamp = Fraction(second) if isinstance(second, Fraction) else Fraction(str(second))
    remaining = timestamp-Fraction(str(first))
    cursor = Fraction(0)
    for index, entry in enumerate(tempo):
        bpm = Fraction(str(entry.bpm))
        if index+1 == len(tempo): return cursor+remaining*bpm/60
        stop = Fraction(tempo[index+1].beat)
        duration = (stop-cursor)*60/bpm
        if remaining < duration: return cursor+remaining*bpm/60
        remaining -= duration
        cursor = stop
    raise ValueError("Tempo map must be nonempty")


def audio_queries(song, start, end, *, max_nodes=None):
    """Return nonuniform evidence and a clock made entirely of Fraction beats."""
    import torch
    from maichart.form.timing import metrical_features
    audio = song.audio
    identity = dict(start=str(start), end=str(end), first=song.first, max_nodes=max_nodes,
                    tempo=[(str(t.beat), t.bpm) for t in song.tempo],
                    meter=[m.to_dict() for m in song.meters], profile=PROFILE)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if key in audio._views:
        audio._views.move_to_end(key)
        values, times, beats = audio._views[key]
        return torch.from_numpy(values.copy()), times.copy(), beats
    stored = Path(audio.cache_dir)/PROFILE/(audio.metadata["audio_sha256"]+"."+key+".queries.npz") if audio.cache_dir else None
    if stored is not None and stored.is_file():
        try:
            with np.load(stored, allow_pickle=False) as data:
                values, times = data["values"].copy(), data["times"].copy()
                beats = tuple(Fraction(str(b)) for b in data["beats"])
            if values.shape != (len(beats), INPUT_DIM) or not np.isfinite(values).all() or len(times) != len(beats):
                raise ValueError("Invalid musical query cache")
        except (OSError, ValueError, KeyError):
            stored.unlink(missing_ok=True)
            return audio_queries(song,start,end,max_nodes=max_nodes)
    else:
        positions = {Fraction(start)}
        positions.update(Fraction(i) for i in range(math.ceil(start), math.ceil(end)))
        positions.update(Fraction(t.beat) for t in song.tempo if start <= t.beat < end)
        positions.update(Fraction(m.beat) for m in song.meters if start <= m.beat < end)
        lo, hi = seconds_at_beat(start,song.tempo,song.first), seconds_at_beat(end,song.tempo,song.first)
        sr = audio.metadata["sample_rate"]
        selected=np.flatnonzero((audio.times >= lo) & (audio.times < hi))
        if max_nodes is not None and len(selected)+len(positions)>max_nodes:
            available=max_nodes-len(positions)
            if available<0: raise ValueError('Musical anchor count exceeds audio memory budget; reduce the resource span')
            chosen=np.argsort(audio.features[selected,92],kind='stable')[-available:] if available else []
            selected=selected[chosen]
        for time in audio.times[selected]:
            sample = round(float(time)*sr)
            beat = beat_at_second(Fraction(sample,sr),song.tempo,song.first)
            if start <= beat < end: positions.add(beat)
        beats = tuple(sorted(positions))
        times = np.asarray([seconds_at_beat(b,song.tempo,song.first) for b in beats],dtype=np.float64)
        signal, sr = waveform(audio.source,audio.metadata['audio_sha256'])
        acoustic = query_spectra(signal,sr,times)
        acoustic[:, :95] /= np.asarray(audio.metadata["normalization_scale"],dtype=np.float32)
        for column in range(95,98): acoustic[:,column] = np.interp(times,audio.times,audio.features[:,column],left=0,right=0)
        beat_values = np.asarray([float(b) for b in beats])
        bpm_values = np.asarray([next(t.bpm for t in reversed(song.tempo) if t.beat <= b) for b in beats])
        progress = np.clip(times/max(audio.duration_sec,1e-6),0,1)
        context = np.column_stack((progress, np.full(len(beats), min(audio.duration_sec/300,1)),
                                  bpm_values/300, np.sin(2*np.pi*beat_values), np.cos(2*np.pi*beat_values),
                                  beat_values/max(float(song.spans[-1].end),1), np.zeros((len(beats),4))))
        rhythm = np.column_stack((acoustic[:,92], acoustic[:,92], acoustic[:,94], acoustic[:,93]))
        values = np.concatenate((acoustic,rhythm,context,metrical_features(beats,song.meters)),axis=1).astype(np.float32)
        values[(times < 0)|(times >= audio.duration_sec)] = 0
        if stored is not None:
            stored.parent.mkdir(parents=True,exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=stored.parent,suffix=".tmp",delete=False) as stream:
                temporary = Path(stream.name)
                np.savez_compressed(stream,values=values,times=times,beats=np.asarray([str(b) for b in beats]))
            temporary.replace(stored)
    audio._views[key] = (values,times,beats)
    while len(audio._views)>4: audio._views.popitem(last=False)
    return torch.from_numpy(values.copy()), times.copy(), beats


def planner_audio_queries(song, start=None, end=None):
    """Return the bounded full-song query used by the Planner contract."""
    start = song.spans[0].start if start is None else start
    end = song.spans[-1].end if end is None else end
    return audio_queries(song, start, end, max_nodes=PLANNER_MAX_AUDIO_NODES)
