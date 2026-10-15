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
import threading

import numpy as np

from maichart.event_timing import seconds_at_beat

PROFILE = "rational_audio_queries_v1"
LEGACY_FEATURE_POLICY = "legacy_sparse_v1"
NATIVE_FEATURE_POLICY = "native_measured_v2"
FEATURE_POLICIES = (LEGACY_FEATURE_POLICY, NATIVE_FEATURE_POLICY)
ACOUSTIC_DIM = 98
INPUT_DIM = 120
MAX_EVIDENCE_NODES = 8192
# Maximum native audio evidence nodes exposed to the whole-song Planner.  The
# same bound is used during training, curve prediction, and chart generation;
# it limits memory/latency without quantising the musical Fraction clock.
PLANNER_MAX_AUDIO_NODES = 1024

# Query tensors are much larger than the sparse native feature arrays.  Keep
# them in one process-wide, byte-bounded LRU instead of retaining up to four
# full tensors on every RationalAudio object.  The per-audio ``_views`` field
# below stores only lightweight keys into this cache.
QUERY_CACHE_MAX_BYTES = 64 * 1024 * 1024
_QUERY_CACHE = OrderedDict()
_QUERY_CACHE_BYTES = 0
_QUERY_CACHE_LOCK = threading.RLock()


def _query_source_token(audio):
    metadata = getattr(audio, "metadata", {}) or {}
    return str(metadata.get("audio_sha256") or getattr(audio, "source", ""))


def _query_cache_get(token):
    with _QUERY_CACHE_LOCK:
        item = _QUERY_CACHE.get(token)
        if item is None:
            return None
        _QUERY_CACHE.move_to_end(token)
        return item[:3]


def _query_cache_put(token, values, times, beats):
    global _QUERY_CACHE_BYTES
    size = int(values.nbytes + times.nbytes + len(beats) * 64)
    if size > QUERY_CACHE_MAX_BYTES:
        return False
    with _QUERY_CACHE_LOCK:
        old = _QUERY_CACHE.pop(token, None)
        if old is not None:
            _QUERY_CACHE_BYTES -= old[3]
        # Own immutable storage: callers receive copies, and the global cache
        # must never freeze a mutable array still held by a builder.
        values = np.asarray(values).copy()
        times = np.asarray(times).copy()
        values.setflags(write=False)
        times.setflags(write=False)
        _QUERY_CACHE[token] = (values, times, beats, size)
        _QUERY_CACHE_BYTES += size
        while _QUERY_CACHE and _QUERY_CACHE_BYTES > QUERY_CACHE_MAX_BYTES:
            _, (_, _, _, removed) = _QUERY_CACHE.popitem(last=False)
            _QUERY_CACHE_BYTES -= removed
    return True


def _query_cache_clear_source(source_token):
    global _QUERY_CACHE_BYTES
    with _QUERY_CACHE_LOCK:
        for token in [token for token in _QUERY_CACHE if token[0] == source_token]:
            _, _, _, size = _QUERY_CACHE.pop(token)
            _QUERY_CACHE_BYTES -= size


class _QueryViewRefs(OrderedDict):
    """At most four lightweight query keys associated with one audio source."""

    def __init__(self, source_token):
        super().__init__()
        self.source_token = source_token

    def clear(self):
        _query_cache_clear_source(self.source_token)
        super().clear()


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
    feature_policy: str = LEGACY_FEATURE_POLICY

    def __post_init__(self):
        policy = self.feature_policy or (self.metadata or {}).get("feature_policy", LEGACY_FEATURE_POLICY)
        if policy not in FEATURE_POLICIES:
            raise ValueError(f"Unknown rational audio feature policy: {policy}")
        self.feature_policy = policy
        # Backward-compatible direct constructors may pass an ordinary
        # OrderedDict.  Replace it before any query cache interaction.
        if not isinstance(self._views, _QueryViewRefs):
            self._views = _QueryViewRefs(_query_source_token(self))


def load_rational_audio(path):
    with np.load(path, allow_pickle=False) as saved:
        meta = json.loads(str(saved["metadata"].item()))
        features, times = saved["features"].copy(), saved["times"].copy()
    if meta.get("profile") != PROFILE or features.ndim != 2 or features.shape[1] != ACOUSTIC_DIM:
        raise ValueError("Rational audio cache contract mismatch")
    if len(times) != len(features) or not np.isfinite(features).all() or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Invalid rational audio cache")
    return RationalAudio(features, times, meta["duration_sec"], meta["tempo_bpm"], meta,
                         meta["source"], meta.get("cache_dir"),
                         feature_policy=meta.get("feature_policy", LEGACY_FEATURE_POLICY))


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
    meta["feature_policy"] = LEGACY_FEATURE_POLICY
    audio = RationalAudio(features, times, meta["duration_sec"], meta["tempo_bpm"], meta, str(path), meta["cache_dir"],
                          feature_policy=LEGACY_FEATURE_POLICY)
    if cached is not None:
        cached.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=cached.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, features=features, times=times, metadata=json.dumps(meta))
        temporary.replace(cached)
    return audio


def acoustic_at_times(audio, times, *, derived_policy=None):
    """Measure rational audio features at arbitrary seconds.

    ``legacy_sparse_v1`` preserves the original cache/query behavior: the
    first 95 columns are measured from the waveform and columns 95--97 are
    interpolated from sparse attack nodes. ``native_measured_v2`` measures
    those derived columns from short neighboring waveform windows, so gaps
    between attack nodes cannot acquire synthetic ramps. This policy is
    explicit because it changes the model input distribution.
    """
    policy = derived_policy or getattr(audio, "feature_policy", None) or LEGACY_FEATURE_POLICY
    if policy not in FEATURE_POLICIES:
        raise ValueError(f"Unknown rational audio feature policy: {policy}")
    seconds = np.asarray(times, dtype=np.float64).reshape(-1)
    signal, sr = waveform(audio.source, audio.metadata.get("audio_sha256"))
    scale = np.asarray(audio.metadata["normalization_scale"], dtype=np.float32)
    if scale.shape != (95,):
        raise ValueError("Rational audio normalization scale must contain 95 values")
    if policy == LEGACY_FEATURE_POLICY:
        measured = query_spectra(signal, sr, seconds)
        measured[:, :95] /= scale
        for column in range(95, 98):
            measured[:, column] = np.interp(
                seconds, audio.times, audio.features[:, column], left=0.0, right=0.0
            )
        return measured

    # Center spectra are exact waveform measurements. Neighbor queries are
    # analysis probes only; they do not create an output clock.
    offsets = np.asarray([0.0, -0.04, 0.04, -0.01, 0.01], dtype=np.float64)
    measured = query_spectra(signal, sr, (seconds[:, None] + offsets).reshape(-1))
    measured[:, :95] /= scale
    measured = measured.reshape(len(seconds), len(offsets), ACOUSTIC_DIM)
    center = measured[:, 0, :].copy()
    rms = center[:, 93]
    reference = np.asarray(audio.features[:, 93], dtype=np.float64)
    positive = reference[reference > 0]
    median_rms = max(float(np.median(positive)) if len(positive) else 1.0, 1e-5)
    center[:, 95] = rms / median_rms
    center[:, 96] = np.clip((measured[:, 2, 93] - measured[:, 1, 93]) / 0.08, -8.0, 8.0)
    center[:, 97] = np.linalg.norm(measured[:, 4, 80:92] - measured[:, 3, 80:92], axis=1)
    outside = (seconds < 0.0) | (seconds >= len(signal) / float(sr))
    center[outside] = 0.0
    return center.astype(np.float32, copy=False)


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
    start, end = Fraction(start), Fraction(end)
    if end <= start:
        raise ValueError('Musical audio query range must have end > start')
    if max_nodes is not None and (isinstance(max_nodes, bool)
            or not isinstance(max_nodes, int) or max_nodes < 1):
        raise ValueError('max_nodes must be a positive integer or None')
    audio = song.audio
    policy = getattr(audio, "feature_policy", LEGACY_FEATURE_POLICY)
    if policy not in FEATURE_POLICIES:
        raise ValueError(f"Unknown rational audio feature policy: {policy}")
    identity = dict(start=str(start), end=str(end), first=song.first, max_nodes=max_nodes,
                    tempo=[(str(t.beat), t.bpm) for t in song.tempo],
                    meter=[m.to_dict() for m in song.meters], profile=PROFILE)
    if policy != LEGACY_FEATURE_POLICY:
        identity["feature_policy"] = policy
        identity['anchor_budget_policy'] = 'bounded_long_song_v1'
        regions=getattr(song,'regions',None)
        extent=regions[-1].end if regions else song.spans[-1].end
        identity['song_end_beat']=str(extent)
    else:
        extent=song.spans[-1].end
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache_token = (_query_source_token(audio), key)
    cached_view = _query_cache_get(cache_token)
    if cached_view is not None:
        values, times, beats = cached_view
        audio._views[cache_token] = None
        audio._views.move_to_end(cache_token)
        while len(audio._views) > 4:
            audio._views.popitem(last=False)
        return torch.from_numpy(values.copy()), times.copy(), beats
    stored = Path(audio.cache_dir)/PROFILE/(audio.metadata["audio_sha256"]+"."+key+".queries.npz") if audio.cache_dir else None
    if stored is not None and stored.is_file():
        try:
            with np.load(stored, allow_pickle=False) as data:
                values, times = data["values"].copy(), data["times"].copy()
                beats = tuple(Fraction(str(b)) for b in data["beats"])
            if (values.shape != (len(beats), INPUT_DIM) or not np.isfinite(values).all()
                    or times.ndim != 1 or len(times) != len(beats) or not len(beats)
                    or not np.isfinite(times).all() or np.any(np.diff(times)<=0)
                    or any(not start<=beat<end for beat in beats)
                    or any(b<=a for a,b in zip(beats,beats[1:]))
                    or (max_nodes is not None and len(beats)>max_nodes)
                    or not np.allclose(times,[seconds_at_beat(b,song.tempo,song.first) for b in beats],
                                       rtol=1e-12,atol=1e-10)):
                raise ValueError("Invalid musical query cache")
        except (OSError, ValueError, KeyError):
            stored.unlink(missing_ok=True)
            return audio_queries(song,start,end,max_nodes=max_nodes)
    else:
        positions = {Fraction(start)}
        positions.update(Fraction(i) for i in range(math.ceil(start), math.ceil(end)))
        positions.update(Fraction(t.beat) for t in song.tempo if start <= t.beat < end)
        positions.update(Fraction(m.beat) for m in song.meters if start <= m.beat < end)
        if max_nodes is not None and len(positions) > max_nodes and policy == NATIVE_FEATURE_POLICY:
            # Full-song queries can exceed 1024 quarter beats on long/fast
            # music. Keep exact tempo/meter boundaries and distributed beat
            # evidence, leaving room for transients. These are memory nodes,
            # never a list of legal chart positions.
            required = {start}
            required.update(Fraction(t.beat) for t in song.tempo if start <= t.beat < end)
            required.update(Fraction(m.beat) for m in song.meters if start <= m.beat < end)
            if len(required) > max_nodes:
                raise ValueError('Tempo/meter anchors exceed audio memory budget; reduce the resource span')
            optional = sorted(positions - required)
            count = min(len(optional), max(0, max(max_nodes // 2, len(required)) - len(required)))
            chosen = np.linspace(0, len(optional) - 1, count, dtype=int) if count else []
            positions = required | {optional[i] for i in chosen}
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
        if policy == LEGACY_FEATURE_POLICY:
            signal, sr = waveform(audio.source,audio.metadata['audio_sha256'])
            acoustic = query_spectra(signal,sr,times)
            acoustic[:, :95] /= np.asarray(audio.metadata["normalization_scale"],dtype=np.float32)
            for column in range(95,98): acoustic[:,column] = np.interp(times,audio.times,audio.features[:,column],left=0,right=0)
        else:
            acoustic = acoustic_at_times(audio, times, derived_policy=policy)
        beat_values = np.asarray([float(b) for b in beats])
        bpm_values = np.asarray([next(t.bpm for t in reversed(song.tempo) if t.beat <= b) for b in beats])
        progress = np.clip(times/max(audio.duration_sec,1e-6),0,1)
        context = np.column_stack((progress, np.full(len(beats), min(audio.duration_sec/300,1)),
                                  bpm_values/300, np.sin(2*np.pi*beat_values), np.cos(2*np.pi*beat_values),
                                  beat_values/max(float(extent),1), np.zeros((len(beats),4))))
        rhythm = np.column_stack((acoustic[:,92], acoustic[:,92], acoustic[:,94], acoustic[:,93]))
        values = np.concatenate((acoustic,rhythm,context,metrical_features(beats,song.meters)),axis=1).astype(np.float32)
        values[(times < 0)|(times >= audio.duration_sec)] = 0
        if stored is not None:
            stored.parent.mkdir(parents=True,exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=stored.parent,suffix=".tmp",delete=False) as stream:
                temporary = Path(stream.name)
                np.savez_compressed(stream,values=values,times=times,beats=np.asarray([str(b) for b in beats]))
            temporary.replace(stored)
    _query_cache_put(cache_token, values, times, beats)
    audio._views[cache_token] = None
    while len(audio._views)>4: audio._views.popitem(last=False)
    return torch.from_numpy(values.copy()), times.copy(), beats


def planner_audio_queries(song, start=None, end=None):
    """Return the bounded full-song query used by the Planner contract."""
    start = song.spans[0].start if start is None else start
    end = song.spans[-1].end if end is None else end
    return audio_queries(song, start, end, max_nodes=PLANNER_MAX_AUDIO_NODES)
