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
LEGACY_EVIDENCE_PROFILE = 'legacy'
BALANCED_EVIDENCE_PROFILE = 'balanced_evidence_v1'
EVIDENCE_PROFILES = (LEGACY_EVIDENCE_PROFILE, BALANCED_EVIDENCE_PROFILE)
ACOUSTIC_DIM = 98
INPUT_DIM = 120
MAX_EVIDENCE_NODES = 8192
# Maximum native audio evidence nodes exposed to the whole-song Planner.  The
# same bound is used during training, curve prediction, and chart generation;
# it limits memory/latency without quantising the musical Fraction clock.
PLANNER_MAX_AUDIO_NODES = 1024


class _ShortAudioRead(ValueError):
    """Decoder header advertises more frames than the stream yields."""

# Query tensors are much larger than the sparse native feature arrays.  Keep
# them in one process-wide, byte-bounded LRU instead of retaining up to four
# full tensors on every RationalAudio object.  The per-audio ``_views`` field
# below stores only lightweight keys into this cache.
QUERY_CACHE_MAX_BYTES = 64 * 1024 * 1024
_QUERY_CACHE = OrderedDict()
_QUERY_CACHE_BYTES = 0
_QUERY_CACHE_LOCK = threading.RLock()
_QUERY_FILE_LOCKS = {}
_QUERY_FILE_LOCKS_GUARD = threading.Lock()


def _query_file_lock(path):
    key = str(path)
    with _QUERY_FILE_LOCKS_GUARD:
        lock = _QUERY_FILE_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _QUERY_FILE_LOCKS[key] = lock
        return lock


def _publish_query_cache(temporary, stored):
    """Publish one query cache safely when Blueprint threads share a source.

    Windows can reject ``os.replace`` while another thread still has the
    destination open.  A per-target lock prevents in-process collisions; the
    short retry also tolerates a publisher from an older process.
    """
    import time
    with _query_file_lock(stored):
        if stored.is_file():
            temporary.unlink(missing_ok=True)
            return False
        last = None
        for attempt in range(8):
            try:
                temporary.replace(stored)
                return True
            except PermissionError as exc:
                last = exc
                if stored.is_file():
                    temporary.unlink(missing_ok=True)
                    return False
                time.sleep(0.025 * (attempt + 1))
        raise last


def _query_cache_candidates(cache_dir, filename):
    """Find reusable query caches across dataset-specific cache roots.

    Query filenames already contain the audio content hash and the complete
    musical-query identity, so sharing them across dataset directories is
    safe.  The current directory remains first; legacy sibling roots are only
    consulted on a miss.
    """
    if cache_dir:
        from maichart.form.cache_support import cache_roots
        for root in cache_roots(str(Path(cache_dir).resolve())):
            yield root / PROFILE / filename


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


def _source_digest(path, cache_dir=None):
    if cache_dir is None:
        return _digest(path)
    from maichart.form.cache_support import source_digests
    return source_digests(str(Path(cache_dir).resolve())).digest(path)


@lru_cache(maxsize=2)
def _wave(path, size, mtime_ns, sha256):
    if _digest(path)!=sha256:
        raise ValueError('Audio content changed before waveform query; reload the audio source')
    signal, sr, _ = _read_mono(path)
    return signal, sr


def waveform(path, sha256=None):
    path = Path(path).resolve()
    stat = path.stat()
    full_hash = sha256 or _digest(path)
    return _wave(str(path), stat.st_size, stat.st_mtime_ns, full_hash)


def _read_mono_fallback(path, start_sec=None, end_sec=None, *, padding_sec=1.0):
    """Use an independent decoder after libsndfile fails or reads short."""
    import audioread
    import librosa

    start = max(0., float(start_sec) - padding_sec) if start_sec is not None else 0.
    duration = max(0., float(end_sec) + padding_sec - start) if end_sec is not None else None
    try:
        decoder = audioread.audio_open(str(path))
        signal, rate = librosa.load(decoder, sr=None, mono=True, offset=start, duration=duration)
    except (audioread.DecodeError, OSError) as error:
        raise ValueError(f'Alternate audio decoder failed for {path}: {error}') from error
    if not len(signal) and duration != 0:
        raise ValueError(f'Alternate audio decoder returned no samples for {path}')
    return np.asarray(signal, dtype=np.float32), int(rate), int(start * rate)


def _read_mono(path, start_sec=None, end_sec=None, *, padding_sec=1.0):
    """Decode in small channel blocks; retain only mono native-rate samples."""
    import soundfile as sf
    try:
        source = sf.SoundFile(str(path))
    except (OSError, RuntimeError):
        return _read_mono_fallback(path, start_sec, end_sec, padding_sec=padding_sec)
    try:
        with source:
            sr = int(source.samplerate)
            # Include the complete analysis window and +/-40 ms native probes,
            # even for unusually low-rate source audio.
            padding = max(float(padding_sec), .04 + 1025. / sr)
            start = (min(source.frames, max(0, math.floor((float(start_sec) - padding) * sr)))
                     if start_sec is not None else 0)
            stop = (min(source.frames, max(start, math.ceil((float(end_sec) + padding) * sr)))
                    if end_sec is not None else source.frames)
            count = stop - start
            signal = np.empty(count, dtype=np.float32)
            if not count:
                return signal, sr, start
            source.seek(start)
            block = np.empty((min(65536, count), source.channels), dtype=np.float32)
            cursor = 0
            while cursor < count:
                chunk = source.read(out=block[:min(len(block), count-cursor)])
                if not len(chunk):
                    raise _ShortAudioRead(f'Audio ended before its declared frame count: {path}')
                # Match librosa.to_mono's channel mean at float32 precision.
                np.mean(chunk, axis=1, dtype=np.float32, out=signal[cursor:cursor+len(chunk)])
                cursor += len(chunk)
            return signal, sr, start
    except (sf.LibsndfileError, _ShortAudioRead):
        return _read_mono_fallback(path, start_sec, end_sec, padding_sec=padding_sec)


def _query_waveform_spectra(audio, seconds, offsets):
    """Measure identical source sample centers in bounded eight-second reads."""
    seconds = np.asarray(seconds, dtype=np.float64)
    offsets = np.asarray(offsets, dtype=np.float64)
    result = np.zeros((len(seconds), len(offsets), ACOUSTIC_DIM), dtype=np.float32)
    if not len(seconds):
        return result
    if not np.isfinite(seconds).all():
        raise ValueError('Acoustic query times must be finite')
    source = Path(audio.source).resolve()
    if not source.is_file():
        # Keep the existing virtual-waveform protocol for synthetic callers.
        signal, sr = waveform(audio.source, audio.metadata.get('audio_sha256'))
        return query_spectra(signal, sr, (seconds[:, None] + offsets).reshape(-1)).reshape(result.shape)
    from maichart.form.waveform_cache import mapped_waveform
    full_signal, sr = mapped_waveform(source, audio.metadata.get('audio_sha256'), audio.cache_dir)
    # Include just-outside-file centers: probes near zero/EOF still need the
    # real neighboring waveform, then the center is masked as before.
    indices = np.flatnonzero((seconds + offsets.max() >= 0)
                             & (seconds + offsets.min() < float(audio.duration_sec)))
    groups = np.floor(seconds[indices] / 8.).astype(np.int64)
    for group in np.unique(groups):
        selected = indices[groups == group]
        padding = max(1., .04 + 1025. / sr)
        sample_offset = max(0, math.floor((float(seconds[selected].min()) - padding) * sr))
        stop = min(len(full_signal), math.ceil((float(seconds[selected].max()) + padding) * sr))
        signal = full_signal[sample_offset:stop]
        times = (seconds[selected, None] + offsets).reshape(-1)
        if np.array_equal(offsets, [0., -.04, .04, -.01, .01]):
            # Native v2 consumes all center columns, RMS at +/-40ms and
            # chroma at +/-10ms. Computing 5 full mel spectra discarded four
            # mel projections and two FFTs for every evidence node.
            shaped=times.reshape(len(selected),len(offsets))
            for columns,component in (([0],'all'),([1,2],'rms'),([3,4],'chroma')):
                measured=query_spectra(signal,sr,shaped[:,columns].reshape(-1),
                    sample_offset=sample_offset,component=component)
                result[np.ix_(selected,columns)]=measured.reshape(len(selected),len(columns),ACOUSTIC_DIM)
        else:
            measured = query_spectra(signal, sr, times, sample_offset=sample_offset)
            result[selected] = measured.reshape(len(selected), len(offsets), ACOUSTIC_DIM)
    return result


@lru_cache(maxsize=6)
def _filters(sr, size):
    import librosa
    return (librosa.filters.mel(sr=sr, n_fft=size, n_mels=80),
            librosa.filters.chroma(sr=sr, n_fft=size, n_chroma=12))


def query_spectra(signal, sr, seconds, *, sample_offset=0, component='all'):
    """Measure spectra at requested centers, including a short transient view.

    Long and short analysis windows share a musical center; neither establishes
    a periodic output clock. Beyond-file centers are explicit silence.
    """
    if component not in {'all','rms','chroma'}:raise ValueError('Unknown acoustic component')
    # Keep mmap storage shared, without propagating the memmap subclass into
    # thousands of temporary advanced-index and arithmetic arrays.
    signal=np.asarray(signal)
    seconds = np.asarray(seconds, dtype=np.float64)
    result = np.zeros((len(seconds), ACOUSTIC_DIM), dtype=np.float32)
    size = 2048
    mel_filter, chroma_filter = _filters(sr, size)
    offsets = np.arange(size) - size // 2
    window = np.hanning(size)
    short_size = max(16, round(sr * .006))
    for start in range(0, len(seconds), 128):
        centers = np.rint(seconds[start:start + 128] * sr).astype(np.int64) - sample_offset
        indices = centers[:, None] + offsets[None]
        inside = (indices >= 0) & (indices < len(signal))
        samples = signal[np.clip(indices, 0, max(0, len(signal) - 1))] * inside if len(signal) else np.zeros_like(indices, dtype=np.float32)
        if component=='rms':
            values=np.sqrt(np.mean(samples**2,axis=1))
            values[(centers<0)|(centers>=len(signal))]=0
            result[start:start+len(samples),93]=values
            continue
        spectrum = np.abs(np.fft.rfft(samples * window, axis=1)) ** 2
        if component=='all':mel = np.log1p(spectrum @ mel_filter.T)
        chroma = spectrum @ chroma_filter.T
        chroma /= np.maximum(chroma.sum(axis=1, keepdims=True), 1e-8)
        if component=='chroma':
            chroma[(centers<0)|(centers>=len(signal))]=0
            result[start:start+len(samples),80:92]=chroma
            continue
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


def _validate_evidence_profile(profile):
    if profile not in EVIDENCE_PROFILES:
        raise ValueError('evidence_profile must be legacy or balanced_evidence_v1')
    return profile


def _balanced_indices(times, scores, budget, *, start_sec=None, end_sec=None):
    """Select evidence by real-time strata and alternating acoustic criteria.

    Each occupied stratum gets one choice before any receives a second. Four
    successive rounds rotate the selection criterion, so neither a loud early
    section nor an attack-only ranking can consume the entire memory budget.
    No observation outside ``times`` is invented and this is not an event grid.
    """
    times = np.asarray(times, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    budget = max(0, int(budget))
    if not budget or not len(times):
        return np.zeros(0, dtype=np.int64)
    if len(times) <= budget:
        return np.arange(len(times), dtype=np.int64)
    if scores.ndim == 1:
        scores = scores[:, None]
    if scores.ndim != 2 or scores.shape[0] != len(times) or not scores.shape[1]:
        raise ValueError('Evidence scores must have one nonempty row per time')
    low = float(times.min()) if start_sec is None else float(start_sec)
    high = float(times.max()) if end_sec is None else float(end_sec)
    strata = max(1, min(len(times), math.ceil(budget / scores.shape[1])))
    cells = np.minimum(np.maximum(((times-low)/max(high-low, 1e-9)*strata).astype(int), 0), strata-1)
    groups = []
    grouped = np.argsort(cells, kind='stable')
    edges = np.r_[0, np.flatnonzero(np.diff(cells[grouped]))+1, len(grouped)]
    for lower, upper in zip(edges[:-1], edges[1:]):
        indices = grouped[lower:upper]
        # Ranking each criterion locally also preserves quieter observations;
        # multiplying one musical section's amplitude cannot evict another.
        orders = [indices[np.argsort(-scores[indices, column], kind='stable')]
                  for column in range(scores.shape[1])]
        groups.append((orders, [0]*scores.shape[1]))
    selected = set()
    round_index = 0
    while len(selected) < budget:
        progress = False
        column = round_index % scores.shape[1]
        for orders, cursors in groups:
            order = orders[column]
            while cursors[column] < len(order) and int(order[cursors[column]]) in selected:
                cursors[column] += 1
            if cursors[column] < len(order):
                selected.add(int(order[cursors[column]]))
                cursors[column] += 1
                progress = True
            if len(selected) == budget:
                break
        if not progress:
            break
        round_index += 1
    return np.asarray(sorted(selected), dtype=np.int64)


def _evidence_scores(features):
    """Independent attack, harmonic, spectral and dynamics evidence."""
    values = np.asarray(features, dtype=np.float64)
    if not len(values):
        return np.zeros((0, 4), dtype=np.float64)
    chroma = values[:, 80:92]
    chroma = chroma / np.maximum(np.linalg.norm(chroma, axis=1, keepdims=True), 1e-8)
    mel = values[:, :80]
    mel = mel / np.maximum(np.linalg.norm(mel, axis=1, keepdims=True), 1e-8)
    harmonic = np.r_[0., np.linalg.norm(np.diff(chroma, axis=0), axis=1)]
    spectral = np.r_[0., np.linalg.norm(np.diff(mel, axis=0), axis=1)]
    rms = np.maximum(values[:, 93], 0.)
    dynamics = np.r_[0., np.abs(np.diff(rms))]
    return np.column_stack((np.maximum(values[:, 92], 0.),
        harmonic+np.maximum(values[:, 97], 0.),
        spectral+np.maximum(values[:, 94], 0.),
        dynamics+np.abs(values[:, 96])))


def attack_samples(signal, sr, *, evidence_profile=LEGACY_EVIDENCE_PROFILE):
    """Find irregular acoustic evidence directly on the native sample clock."""
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import find_peaks
    profile = _validate_evidence_profile(evidence_profile)
    if len(signal) < 2 or not np.any(signal):
        return np.zeros(0, dtype=np.int64)
    energy = uniform_filter1d(signal.astype(np.float64)**2, max(1, round(sr*.003)))
    previous = np.roll(energy, max(1, round(sr*.004)))
    previous[:max(1, round(sr*.004))] = 0
    novelty = np.maximum(energy-previous, 0)
    if profile == BALANCED_EVIDENCE_PROFILE:
        height = max(float(novelty.max())*1e-8, 1e-12)
    else:
        height = max(float(np.percentile(novelty, 85)), float(novelty.max())*.002, 1e-12)
    peaks, props = find_peaks(novelty, height=height, distance=max(1, round(sr*.006)))
    if profile == BALANCED_EVIDENCE_PROFILE and len(peaks):
        # Estimate a local amplitude floor on sparse candidates, not with an
        # additional full-song float64 envelope in every preparation worker.
        cells = peaks//max(1, round(sr*.4))
        local_maximum = np.zeros(int(cells[-1])+1, dtype=np.float64)
        np.maximum.at(local_maximum, cells, props['peak_heights'])
        keep = props['peak_heights'] >= local_maximum[cells]*.002
        peaks = peaks[keep]
        props['peak_heights'] = props['peak_heights'][keep]
    if len(peaks) > MAX_EVIDENCE_NODES:
        if profile == BALANCED_EVIDENCE_PROFILE:
            chosen = _balanced_indices(peaks/sr, props['peak_heights'], MAX_EVIDENCE_NODES,
                start_sec=0., end_sec=len(signal)/sr)
        else:
            chosen = np.argsort(props["peak_heights"], kind="stable")[-MAX_EVIDENCE_NODES:]
        peaks = np.sort(peaks[chosen])
    return peaks.astype(np.int64)


def _spectral_evidence(signal, sr):
    """Find non-percussive spectral changes using bounded native-sample probes.

    Probe windows are analysis observations only. Only local acoustic maxima
    become evidence nodes; neither these centers nor their spacing constrain
    chart positions. The probe budget is bounded independently of file length.
    """
    from scipy.signal import find_peaks
    count = min(MAX_EVIDENCE_NODES, max(2, math.ceil(len(signal)/sr/.04)))
    samples = np.unique(np.linspace(0, max(0, len(signal)-1), count, dtype=np.int64))
    features = query_spectra(signal, sr, samples/sr)
    scores = _evidence_scores(features)
    selected = set()
    for column in (1, 2, 3):
        values = scores[:, column]
        threshold = max(float(values.max(initial=0.))*1e-6, 1e-10)
        peaks, _ = find_peaks(values, height=threshold)
        selected.update(map(int, peaks))
    indices = np.asarray(sorted(selected), dtype=np.int64)
    return samples[indices], features[indices]


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


def cache_path(path, cache_dir, sha256=None, *, evidence_profile=LEGACY_EVIDENCE_PROFILE):
    evidence_profile = _validate_evidence_profile(evidence_profile)
    if cache_dir is None: return None
    suffix = '' if evidence_profile == LEGACY_EVIDENCE_PROFILE else '.'+evidence_profile
    return Path(cache_dir) / PROFILE / ((sha256 or _source_digest(path,cache_dir))+suffix+".npz")


def _cache_candidates(path, cache_dir, sha256=None, *, evidence_profile=LEGACY_EVIDENCE_PROFILE):
    """Current and legacy feature-cache locations for one content hash."""
    evidence_profile = _validate_evidence_profile(evidence_profile)
    if cache_dir is None:
        return []
    path = Path(path).resolve()
    digest = sha256 or _source_digest(path,cache_dir)
    current = Path(cache_dir).resolve()
    if evidence_profile != LEGACY_EVIDENCE_PROFILE:
        from maichart.form.cache_support import cache_roots
        roots = [current, *cache_roots(str(current))]
        return list(dict.fromkeys(root / PROFILE / (digest+'.'+evidence_profile+'.npz')
                                  for root in roots))
    candidates = [current / PROFILE / (digest + '.npz')]
    # Legacy key included the absolute source path.
    legacy_identity = {'profile': PROFILE, 'source': str(path), 'audio_sha256': digest}
    legacy_key = hashlib.sha256(json.dumps(legacy_identity, sort_keys=True).encode()).hexdigest()
    candidates.append(current / PROFILE / (legacy_key + '.npz'))
    from maichart.form.cache_support import cache_roots
    for root in cache_roots(str(current)):
        candidates.append(root / PROFILE / (digest+'.npz'))
        candidates.append(root / PROFILE / (legacy_key+'.npz'))
        try:
            relative = path.relative_to(current.parent)
        except ValueError:
            continue
        relocated = root.parent / relative
        identity = {'profile':PROFILE,'source':str(relocated),'audio_sha256':digest}
        relocated_key = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        candidates.append(root / PROFILE / (relocated_key+'.npz'))
    return list(dict.fromkeys(candidates))


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
    evidence_profile: str = LEGACY_EVIDENCE_PROFILE

    def __post_init__(self):
        policy = self.feature_policy or (self.metadata or {}).get("feature_policy", LEGACY_FEATURE_POLICY)
        if policy not in FEATURE_POLICIES:
            raise ValueError(f"Unknown rational audio feature policy: {policy}")
        self.feature_policy = policy
        self.evidence_profile = _validate_evidence_profile(self.evidence_profile)
        # Backward-compatible direct constructors may pass an ordinary
        # OrderedDict.  Replace it before any query cache interaction.
        if not isinstance(self._views, _QueryViewRefs):
            self._views = _QueryViewRefs(_query_source_token(self))


def load_rational_audio(path, *, source=None, cache_dir=None):
    with np.load(path, allow_pickle=False) as saved:
        meta = json.loads(str(saved["metadata"].item()))
        features, times = saved["features"].copy(), saved["times"].copy()
    if meta.get("profile") != PROFILE or features.ndim != 2 or features.shape[1] != ACOUSTIC_DIM:
        raise ValueError("Rational audio cache contract mismatch")
    if len(times) != len(features) or not np.isfinite(features).all() or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Invalid rational audio cache")
    if source is not None:
        meta['source'] = str(Path(source).resolve())
    if cache_dir is not None:
        meta['cache_dir'] = str(cache_dir)
    return RationalAudio(features, times, meta["duration_sec"], meta["tempo_bpm"], meta,
                         meta["source"], meta.get("cache_dir"),
                         feature_policy=meta.get("feature_policy", LEGACY_FEATURE_POLICY),
                         evidence_profile=meta.get('evidence_profile', LEGACY_EVIDENCE_PROFILE))


def extract_rational_audio(path, cache_dir=None, *, evidence_profile=LEGACY_EVIDENCE_PROFILE):
    evidence_profile = _validate_evidence_profile(evidence_profile)
    path = Path(path).resolve()
    content_sha256 = _source_digest(path,cache_dir)
    cached = cache_path(path, cache_dir, content_sha256, evidence_profile=evidence_profile)
    for candidate in _cache_candidates(path, cache_dir, content_sha256, evidence_profile=evidence_profile):
        if not candidate.is_file():
            continue
        try:
            audio = load_rational_audio(candidate,source=path,cache_dir=cache_dir)
            if audio.metadata.get('audio_sha256') != content_sha256:
                raise ValueError('Rational audio cache content mismatch')
            if audio.evidence_profile != evidence_profile:
                raise ValueError('Rational audio cache evidence profile mismatch')
            break
        except (OSError, ValueError, KeyError):
            audio = None
    else:
        audio = None
    if audio is not None:
        if round(float(audio.tempo_bpm), 4) != float(audio.tempo_bpm) and path.is_file():
            from maichart.form.waveform_cache import mapped_waveform
            signal, sr = mapped_waveform(path, content_sha256, cache_dir)
            audio.tempo_bpm = _tempo_hint(signal, sr)
            audio.metadata['tempo_bpm'] = audio.tempo_bpm
            target = cached if cached is not None else candidate
            if target is not None:
                target.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".tmp", delete=False) as stream:
                    temporary = Path(stream.name)
                    np.savez_compressed(stream, features=audio.features, times=audio.times,
                                        metadata=json.dumps(audio.metadata))
                temporary.replace(target)
        return audio
    from maichart.form.waveform_cache import mapped_waveform
    signal, sr = mapped_waveform(path,content_sha256,cache_dir)
    if evidence_profile == BALANCED_EVIDENCE_PROFILE:
        attacks = attack_samples(signal, sr, evidence_profile=evidence_profile)
        spectral_samples, spectral_features = _spectral_evidence(signal, sr)
        samples = np.unique(np.r_[0, attacks, spectral_samples, max(0, len(signal)-1)])
        features = np.zeros((len(samples), ACOUSTIC_DIM), dtype=np.float32)
        known = np.searchsorted(samples, spectral_samples)
        features[known] = spectral_features
        missing = np.ones(len(samples), dtype=bool)
        missing[known] = False
        features[missing] = query_spectra(signal, sr, samples[missing]/sr)
        if len(samples) > MAX_EVIDENCE_NODES:
            required = {0, len(samples)-1}
            optional = np.asarray(sorted(set(range(len(samples)))-required), dtype=np.int64)
            selected = _balanced_indices(samples[optional]/sr, _evidence_scores(features)[optional],
                MAX_EVIDENCE_NODES-len(required), start_sec=0., end_sec=len(signal)/sr)
            keep = np.asarray(sorted(required | set(map(int, optional[selected]))), dtype=np.int64)
            samples, features = samples[keep], features[keep]
        times = samples.astype(np.float64)/sr
    else:
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
    if evidence_profile != LEGACY_EVIDENCE_PROFILE:
        meta['evidence_profile'] = evidence_profile
        meta['evidence_selection'] = 'time_strata_acoustic_diversity_v1'
    audio = RationalAudio(features, times, meta["duration_sec"], meta["tempo_bpm"], meta, str(path), meta["cache_dir"],
                          feature_policy=LEGACY_FEATURE_POLICY, evidence_profile=evidence_profile)
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
    scale = np.asarray(audio.metadata["normalization_scale"], dtype=np.float32)
    if scale.shape != (95,):
        raise ValueError("Rational audio normalization scale must contain 95 values")
    if policy == LEGACY_FEATURE_POLICY:
        measured = _query_waveform_spectra(audio, seconds, [0.])[:, 0]
        measured[:, :95] /= scale
        for column in range(95, 98):
            measured[:, column] = np.interp(
                seconds, audio.times, audio.features[:, column], left=0.0, right=0.0
            )
        return measured

    # Center spectra are exact waveform measurements. Neighbor queries are
    # analysis probes only; they do not create an output clock.
    offsets = np.asarray([0.0, -0.04, 0.04, -0.01, 0.01], dtype=np.float64)
    measured = _query_waveform_spectra(audio, seconds, offsets)
    measured[:, :, :95] /= scale
    center = measured[:, 0, :].copy()
    rms = center[:, 93]
    reference = np.asarray(audio.features[:, 93], dtype=np.float64)
    positive = reference[reference > 0]
    median_rms = max(float(np.median(positive)) if len(positive) else 1.0, 1e-5)
    center[:, 95] = rms / median_rms
    center[:, 96] = np.clip((measured[:, 2, 93] - measured[:, 1, 93]) / 0.08, -8.0, 8.0)
    center[:, 97] = np.linalg.norm(measured[:, 4, 80:92] - measured[:, 3, 80:92], axis=1)
    # Keep EOF on the source clock, never on the local window duration.
    outside = (seconds < 0.0) | (seconds >= float(audio.duration_sec))
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
    from maichart.tempo_clock import TempoClock
    start, end = Fraction(start), Fraction(end)
    if end <= start:
        raise ValueError('Musical audio query range must have end > start')
    if max_nodes is not None and (isinstance(max_nodes, bool)
            or not isinstance(max_nodes, int) or max_nodes < 1):
        raise ValueError('max_nodes must be a positive integer or None')
    clock=TempoClock(song.tempo,song.first)
    audio = song.audio
    policy = getattr(audio, "feature_policy", LEGACY_FEATURE_POLICY)
    if policy not in FEATURE_POLICIES:
        raise ValueError(f"Unknown rational audio feature policy: {policy}")
    evidence_profile = _validate_evidence_profile(
        getattr(audio, 'evidence_profile', LEGACY_EVIDENCE_PROFILE))
    identity = dict(start=str(start), end=str(end), first=song.first, max_nodes=max_nodes,
                    tempo=[(str(t.beat), t.bpm) for t in song.tempo],
                    meter=[m.to_dict() for m in song.meters], profile=PROFILE)
    if evidence_profile != LEGACY_EVIDENCE_PROFILE:
        identity['evidence_profile'] = evidence_profile
        identity['evidence_selection'] = 'time_strata_acoustic_diversity_v1'
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
    filename = audio.metadata["audio_sha256"]+"."+key+".queries.npz"
    stored = next((candidate for candidate in _query_cache_candidates(audio.cache_dir, filename)
                   if candidate.is_file()), None) if audio.cache_dir else None
    if stored is not None:
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
                    or not np.allclose(times,[clock.at(b) for b in beats],
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
        omitted_anchors = []
        if (max_nodes is not None and
                ((evidence_profile == BALANCED_EVIDENCE_PROFILE and len(positions) > max_nodes//2)
                 or (len(positions) > max_nodes and policy == NATIVE_FEATURE_POLICY))):
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
            if evidence_profile == BALANCED_EVIDENCE_PROFILE:
                omitted_anchors = sorted(set(optional)-positions)
        lo, hi = clock.at(start), clock.at(end)
        sr = audio.metadata["sample_rate"]
        selected=np.flatnonzero((audio.times >= lo) & (audio.times < hi))
        if max_nodes is not None and len(selected)+len(positions)>max_nodes:
            available=max_nodes-len(positions)
            if available<0: raise ValueError('Musical anchor count exceeds audio memory budget; reduce the resource span')
            if evidence_profile == BALANCED_EVIDENCE_PROFILE:
                # Score before clipping so harmonic/level transitions at the
                # query boundary keep their evidence from the prior sample.
                scores = _evidence_scores(audio.features)[selected]
                chosen = _balanced_indices(audio.times[selected], scores, available,
                    start_sec=lo, end_sec=hi)
            else:
                chosen=np.argsort(audio.features[selected,92],kind='stable')[-available:] if available else []
            selected=selected[chosen]
        for time in audio.times[selected]:
            sample = round(float(time)*sr)
            beat = beat_at_second(Fraction(sample,sr),song.tempo,song.first)
            if start <= beat < end: positions.add(beat)
        if omitted_anchors and max_nodes is not None and len(positions) < max_nodes:
            optional = [beat for beat in omitted_anchors if beat not in positions]
            count = min(len(optional), max_nodes-len(positions))
            chosen = np.linspace(0, len(optional)-1, count, dtype=int) if count else []
            positions.update(optional[index] for index in chosen)
        beats = tuple(sorted(positions))
        times = np.asarray([clock.at(b) for b in beats],dtype=np.float64)
        if policy == LEGACY_FEATURE_POLICY:
            acoustic = acoustic_at_times(audio, times, derived_policy=policy)
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
        if audio.cache_dir:
            # Always publish new entries to the active root. A cache found in
            # a sibling/legacy root is read-only from this call's perspective.
            stored = Path(audio.cache_dir)/PROFILE/filename
            stored.parent.mkdir(parents=True,exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=stored.parent,suffix=".tmp",delete=False) as stream:
                temporary = Path(stream.name)
                np.savez_compressed(stream,values=values,times=times,beats=np.asarray([str(b) for b in beats]))
            _publish_query_cache(temporary, stored)
    _query_cache_put(cache_token, values, times, beats)
    audio._views[cache_token] = None
    while len(audio._views)>4: audio._views.popitem(last=False)
    return torch.from_numpy(values.copy()), times.copy(), beats


def planner_audio_queries(song, start=None, end=None):
    """Return the bounded full-song query used by the Planner contract."""
    start = song.spans[0].start if start is None else start
    end = song.spans[-1].end if end is None else end
    return audio_queries(song, start, end, max_nodes=PLANNER_MAX_AUDIO_NODES)
