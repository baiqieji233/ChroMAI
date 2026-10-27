"""Audio-only metrical evidence, with explicit uncertainty.

Layout: native beat phases (6), four tokens (10 each: weighted sin/cos
for harmonics 1/2, log bar length, log pulse length, posterior weight,
support, phase confidence, valid), then acoustic novelty and boundary support.
Candidates describe periods in chart-beat units, not asserted time signatures.
Equal durations with different denominator/grouping are intentionally unresolved.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
from collections import OrderedDict
from functools import lru_cache

import numpy as np

STRUCTURE_DIM = 48
STRUCTURE_POLICIES = frozenset(('off', 'native_periodic_v1', 'metrical_context_v1'))
SCHEMA = 'composer_structure_native_v4'
# Exact rational durations, avoiding artificial claims about time signatures.
PERIODS = np.array(sorted({Fraction(n * 4, d) for d in (2, 4, 8, 16)
                           for n in range(2, 33)}), dtype=np.float64)
_lock = threading.RLock()
_banks = OrderedDict()
_identity_bases = OrderedDict()
_key_locks = {}
_query_banks=OrderedDict()


@lru_cache(maxsize=64)
def _cache_directory(raw):
    return str(Path(raw).resolve()) if raw is not None else ''


def clear_structure_caches():
    """Explicit run-boundary reset; source data are immutable during a run."""
    from .structure_backend import clear_backend_identity_cache
    with _lock:
        songs=[value[0] for value in _query_banks.values()]
        songs.extend(value[0] for value in _identity_bases.values())
        for song in songs:
            try:delattr(song,'_composer_structure_identity_base')
            except (AttributeError,TypeError):pass
        _query_banks.clear();_banks.clear();_identity_bases.clear()
        _cache_directory.cache_clear()
    clear_backend_identity_cache()


def resolve_structure_backend(requested='auto'):
    """Resolve and freeze the backend in the caller's model configuration."""
    from .structure_backend import resolve_structure_backend as resolve
    return resolve(requested)


def _identity(song, backend):
    from .structure_backend import backend_identity
    audio = song.audio
    # Cache the immutable audio identity on the lightweight song object. This
    # avoids reloading an evicted lazy RationalAudio or scanning its features on
    # every training sample. The fallback is bounded for objects without attrs.
    try:
        base = vars(song).get('_composer_structure_identity_base')
    except TypeError:
        base = None
    if base is not None and base[0] is not audio:
        base = None
    if base is None:
        fallback = _identity_bases.get(id(song))
        if fallback is not None and fallback[0] is song and fallback[1] is audio:
            base = (audio, fallback[2])
            _identity_bases.move_to_end(id(song))
    if base is None:
        meta = getattr(audio, 'metadata', {}) or {}
        digest = meta.get('audio_sha256')
        if not digest:
            h = hashlib.sha256()
            for a in (audio.times, audio.features):
                arr = np.ascontiguousarray(a)
                h.update(str((arr.shape, arr.dtype)).encode())
                h.update(memoryview(arr).cast('B'))
            digest = h.hexdigest()
        details = {'audio_sha256': str(digest),
                   'audio_profile': str(getattr(audio, 'feature_policy', 'unknown')),
                   'evidence_profile': str(getattr(audio, 'evidence_profile', 'unknown')),
                   'duration': float(audio.duration_sec)}
        base = (audio, details)
        try:
            setattr(song, '_composer_structure_identity_base', base)
        except (AttributeError, TypeError):
            _identity_bases[id(song)] = (song, audio, details)
            while len(_identity_bases) > 128: _identity_bases.popitem(last=False)
    return {'schema': SCHEMA, **base[1], 'first': float(song.first),
            'tempo': [[str(t.beat), float(t.bpm)] for t in song.tempo],
            'backend_requested': backend, 'backend_effective': backend_identity(backend),
            'grouping': 'unknown', 'grid_beats': .25, 'window_beats': 32.}


def structure_fingerprint(song, cache_dir=None, backend='auto'):
    """Stable cache key; no analysis or network access is performed."""
    return hashlib.sha256(json.dumps(_identity(song, backend), sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def _seconds_at_beats(beats, song):
    b = np.array([float(t.beat) for t in song.tempo], dtype=np.float64)
    bpm = np.array([float(t.bpm) for t in song.tempo], dtype=np.float64)
    if len(b) == 0 or b[0] != 0 or np.any(np.diff(b) <= 0) or np.any(bpm <= 0):
        raise ValueError('Invalid tempo map')
    secs = np.r_[float(song.first), float(song.first) + np.cumsum(np.diff(b) * 60 / bpm[:-1])]
    i = np.clip(np.searchsorted(b, beats, side='right') - 1, 0, len(b)-1)
    return secs[i] + (beats-b[i]) * 60 / bpm[i]


def _harmonic_coefficients(phase, weighted_residual, residual_energy):
    """Separate complex harmonics; symmetric accents can cancel only z1."""
    active = residual_energy > 1e-7
    denom = np.maximum(residual_energy[None, :], 1e-8)
    first = ((phase.T @ weighted_residual) / denom)[:, active].mean(axis=1)
    second = (((phase * phase).T @ weighted_residual) / denom)[:, active].mean(axis=1)
    return first, second


def _build(song, cache_dir, backend_info):
    audio = song.audio
    times = np.asarray(audio.times, dtype=np.float64)
    data = np.asarray(audio.features)
    if data.ndim != 2 or len(times) != len(data) or data.shape[1] < 98:
        raise ValueError('Structure analysis requires RationalAudio features [N, >=98]')
    if len(times) < 2 or np.any(np.diff(times) <= 0) or not np.isfinite(times).all():
        raise ValueError('Invalid audio timestamps')
    # The inverse tempo mapping is vectorized and handles local tempo changes.
    tb = np.array([float(t.beat) for t in song.tempo]); bpm = np.array([float(t.bpm) for t in song.tempo])
    ts = _seconds_at_beats(tb, song)
    end_segment=int(np.clip(np.searchsorted(ts,float(audio.duration_sec),side='right')-1,0,len(ts)-1))
    end = max(0., tb[end_segment] + (float(audio.duration_sec)-ts[end_segment])*bpm[end_segment]/60.)
    grid = np.arange(0., max(end, .25)+.125, .25)
    seconds = _seconds_at_beats(grid, song)
    # Three independent acoustic channels; 92 is onset-strength in the native bank,
    # low mel and chroma contribute complementary repetition evidence.
    x = np.nan_to_num(np.asarray(data, dtype=np.float64), nan=0., posinf=0., neginf=0.)
    cols = np.column_stack((x[:, 92], np.mean(x[:, :20], axis=1), x[:, 80:92]))
    sampled = np.column_stack([np.interp(seconds, times, cols[:, i], left=0., right=0.) for i in range(cols.shape[1])])
    # Near-silence can still have normalized chroma; suppress its contribution
    # based on absolute attack and low-band energy before normalization.
    strength = max(float(np.percentile(np.abs(sampled[:, 0]), 90)),
                   float(np.percentile(np.abs(sampled[:, 1]), 90)))
    audibility = np.clip(strength / 1e-4, 0., 1.)
    # Spectral changes avoid treating sustained loudness as downbeat evidence.
    signal = np.column_stack((np.maximum(sampled[:, 0], 0.),
                              np.maximum(np.diff(sampled[:, 1], prepend=sampled[0, 1]), 0.),
                              np.linalg.norm(np.diff(sampled[:, 2:], axis=0, prepend=sampled[:1, 2:]), axis=1)))
    scale = np.percentile(signal, 90, axis=0)
    signal = np.clip(signal / np.maximum(scale, 1e-8), 0., 3.)
    signal[:, scale < 1e-8] = 0.
    from .structure_backend import beat_this_frames
    frames = beat_this_frames(audio, cache_dir, backend_info)
    if frames is not None:
        logits = np.asarray(frames, dtype=np.float64)
        probabilities = 1. / (1. + np.exp(-np.clip(logits, -30., 30.)))
        ft = np.arange(len(frames), dtype=np.float64) / 50.
        # Beat and downbeat remain independent. Use peak excess as evidence,
        # with silence / near-constant outputs contributing no support.
        neural = np.column_stack([np.maximum.reduce([np.interp(seconds + offset, ft, probabilities[:, j], left=0., right=0.) for offset in (-.04, -.02, 0., .02, .04)]) for j in range(2)])
        neural = np.maximum(neural - np.percentile(neural, 50, axis=0), 0.)
        signal = np.column_stack((signal, neural))
    # Section novelty compares acoustic context on either side, rather than
    # equating a single onset with a section boundary. Preserve a quarter-beat
    # output grid and use means over 2, 4, and 8 beats.
    low, high = np.percentile(sampled, [10, 90], axis=0)
    context = np.clip((sampled-low) / np.maximum(high-low, 1e-6), -1., 3.)
    context[:, high-low < 1e-6] = 0.
    prefix = np.vstack((np.zeros((1, context.shape[1])), np.cumsum(context, axis=0)))
    positions = np.arange(len(grid))
    changes=[]
    for width in (8, 16, 32):
        left=np.maximum(0, positions-width)
        right=np.minimum(len(grid), positions+width)
        before=(prefix[positions]-prefix[left])/np.maximum((positions-left)[:,None],1)
        after=(prefix[right]-prefix[positions])/np.maximum((right-positions)[:,None],1)
        delta=np.linalg.norm(after-before,axis=1)/np.sqrt(context.shape[1])
        valid=(positions-left >= width//2) & (right-positions >= width//2)
        changes.append(np.where(valid, np.clip(delta, 0., 1.), 0.))
    novelty=np.mean(changes,axis=0)*audibility
    centers = np.arange(0., max(end, .25)+4., 4.)
    # Store every candidate's period, weight, support, phase confidence and
    # Fourier coefficient at every anchor. One bounded window at a time.
    records = np.zeros((len(centers), len(PERIODS), 8), np.float32)
    offsets = np.arange(-64, 65)
    taper = np.hanning(len(offsets))
    for k, center in enumerate(centers):
        indices = np.rint(center * 4).astype(int) + offsets
        valid = (indices >= 0) & (indices < len(grid))
        if valid.sum() < 8: continue
        g = grid[indices[valid]]; w = taper[valid]
        sig = signal[indices[valid]]
        energy = np.sum(w[:, None]*sig, axis=0)
        if energy.max() < 1e-6: continue
        # Estimate a pulse separately, then remove its repeated per-pulse
        # profile. A uniform kick on every pulse is not a bar accent.
        pulse_options = np.array([.25, .5, .75, 1., 1.5, 2., 3.])
        # Compare repeated onset/acoustic profiles, with a weak chart-beat
        # reference prior to break aliases; no tempo map is altered.
        pulse_strength=[]
        for option in pulse_options:
            lag=int(round(option*4))
            left, right = sig[:-lag], sig[lag:]
            denom=np.sqrt(np.sum(left*left)*np.sum(right*right))
            coherence=float(np.sum(left*right)/denom) if denom > 1e-8 else 0.
            pulse_strength.append(coherence - .03*abs(np.log2(option)))
        pulse = float(pulse_options[np.argmax(pulse_strength)])
        pulse_bins = max(1, round(pulse * 4))
        residual = sig.copy()
        phase_bin = np.rint(g*4).astype(int) % pulse_bins
        for group in range(pulse_bins):
            mask = phase_bin == group
            if mask.any():
                baseline = np.sum(sig[mask]*w[mask,None], axis=0) / max(w[mask].sum(), 1e-8)
                residual[mask] = np.maximum(sig[mask] - baseline, 0.)
        residual_energy = np.sum(w[:, None]*residual, axis=0)
        contrast = np.clip(float(residual_energy.sum() / max(energy.sum(), 1e-8)), 0., 1.)
        if contrast < .02:
            continue
        phase = np.exp(-2j*np.pi*g[:, None]/PERIODS[None, :])
        combined, combined2 = _harmonic_coefficients(
            phase, w[:, None]*residual, residual_energy)
        concentration = np.maximum(np.abs(combined), np.abs(combined2))
        cycles = np.clip((g[-1]-g[0])/PERIODS / 3., 0., 1.)
        activity = np.clip(float(np.mean(energy / max(w.sum(), 1e-8))), 0., 1.)
        support = cycles * activity * contrast * audibility
        # Two-pulse bars remain a competing hypothesis in the free slot;
        # pulse-only energy has already been removed above.
        eligible = (PERIODS >= 3.*pulse - 1e-7) | np.isclose(PERIODS, 2.*pulse)
        score = concentration * support * eligible
        evidence = np.expm1(np.clip(score * 5., 0., 5.))
        weight = evidence / (8. + evidence.sum())
        records[k, :, 0] = weight
        records[k, :, 1] = support * eligible
        records[k, :, 2] = concentration * eligible
        records[k, :, 3] = combined.real
        records[k, :, 4] = combined.imag
        records[k, :, 5] = pulse
        records[k, :, 6] = combined2.real
        records[k, :, 7] = combined2.imag
    return centers, records, grid, novelty


def _select_order(records):
    order = np.zeros((len(records), 4), dtype=np.int32)
    for row in range(len(records)):
        taken = set()
        ranked = np.argsort(-records[row, :, 0], kind='stable')
        for slot, minimum in enumerate((3., 6., 3., 0.)):
            choices = [i for i in ranked if PERIODS[i] >= minimum and i not in taken]
            choice = choices[0]
            taken.add(choice)
            order[row, slot] = choice
    return order


def _bank(song, cache_dir, backend):
    from .structure_backend import backend_cache_token
    # Fractions and floats compare directly: no JSON/SHA or filesystem work.
    # The small tempo tuple detects in-place tempo edits as well as replacement.
    tempo=tuple((change.beat,change.bpm) for change in song.tempo)
    raw_dir=os.fspath(cache_dir) if cache_dir is not None else None
    audio=song.audio
    query_key=(id(song),id(audio),float(song.first),tempo,raw_dir,backend,SCHEMA,backend_cache_token())
    with _lock:
        hit=_query_banks.get(query_key)
        if hit is not None and hit[0] is song and hit[1] is audio:
            _query_banks.move_to_end(query_key)
            return hit[2]
    result=_cold_bank(song,cache_dir,backend)
    with _lock:
        _query_banks[query_key]=(song,audio,result)
        _query_banks.move_to_end(query_key)
        while len(_query_banks)>128:_query_banks.popitem(last=False)
    return result


def _cold_bank(song, cache_dir, backend):
    identity = _identity(song, backend)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    cache_key = (_cache_directory(os.fspath(cache_dir) if cache_dir is not None else None),key)
    with _lock:
        if cache_key in _banks:
            _banks.move_to_end(cache_key)
            return _banks[cache_key]
        entry = _key_locks.get(cache_key)
        if entry is None:
            entry = [threading.Lock(), 0]
            _key_locks[cache_key] = entry
        entry[1] += 1
    try:
        with entry[0]:
            with _lock:
                if cache_key in _banks:
                    _banks.move_to_end(cache_key)
                    return _banks[cache_key]
            result = _build_bank(song, cache_dir, identity, key)
            with _lock:
                _banks[cache_key] = result
                while len(_banks) > 8:
                    _banks.popitem(last=False)
            return result
    finally:
        with _lock:
            entry[1] -= 1
            if entry[1] == 0:
                _key_locks.pop(cache_key, None)


def _publish_structure_array(path, values):
    """Publish an immutable array, reusing a concurrent Windows publisher."""
    fd, temporary = tempfile.mkstemp(prefix=path.stem+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            np.save(stream, values, allow_pickle=False)
            stream.flush(); os.fsync(stream.fileno())
        try:
            os.replace(temporary, path)
        except PermissionError:
            # A different process can publish and mmap this same content key
            # before us. Windows then refuses replacement of the mapped file.
            # Reuse only a complete, valid array; other write failures surface.
            if not path.is_file(): raise
            existing = np.load(path, mmap_mode='r', allow_pickle=False)
            if (existing.shape != values.shape or existing.dtype != values.dtype
                    or not np.isfinite(existing).all()
                    or not np.array_equal(existing[:, 0], values[:, 0])):
                raise ValueError(f'Invalid concurrently published structure cache: {path}')
            return existing
        return np.load(path, mmap_mode='r', allow_pickle=False)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def _build_bank(song, cache_dir, identity, key):
    directory = Path(cache_dir) / 'composer_structure_v4' if cache_dir is not None else None
    path = directory / (key + '.npy') if directory else None
    novelty_path = directory / (key + '.novelty.npy') if directory else None
    width = 2 + len(PERIODS)*8
    array = None
    novelty_array = None
    if path and path.exists() and novelty_path.exists():
        try:
            candidate = np.load(path, mmap_mode='r', allow_pickle=False)
            if (candidate.ndim == 2 and candidate.shape[1] == width and candidate.shape[0] > 0
                    and candidate.dtype == np.float32 and np.isfinite(candidate).all()):
                novelty_candidate = np.load(novelty_path, mmap_mode='r', allow_pickle=False)
                if (novelty_candidate.ndim == 2 and novelty_candidate.shape[1] == 2 and len(novelty_candidate)
                        and novelty_candidate.dtype == np.float32 and np.isfinite(novelty_candidate).all()):
                    array, novelty_array = candidate, novelty_candidate
        except (OSError, ValueError): pass
    if array is None:
        centers, records, grid, novelty = _build(song, cache_dir, identity['backend_effective'])
        array = np.zeros((len(centers), width), dtype=np.float32)
        array[:, 0] = centers
        array[:, 1] = np.interp(centers, grid, novelty)
        array[:, 2:] = records.reshape(len(centers), -1)
        novelty_array = np.column_stack((grid, novelty)).astype(np.float32)
        if directory:
            directory.mkdir(parents=True, exist_ok=True)
            array = _publish_structure_array(path, array)
            novelty_array = _publish_structure_array(novelty_path, novelty_array)
            meta_path = directory / (key + '.json')
            fd, tmp = tempfile.mkstemp(prefix=key+'.', suffix='.json.tmp', dir=directory)
            try:
                with os.fdopen(fd, 'w', encoding='utf8') as stream:
                    json.dump(identity, stream, sort_keys=True)
                try:
                    os.replace(tmp, meta_path)
                except PermissionError:
                    if not meta_path.is_file(): raise
                    if json.loads(meta_path.read_text(encoding='utf8')) != identity:
                        raise ValueError(f'Conflicting structure cache identity: {meta_path}')
            finally:
                if os.path.exists(tmp): os.unlink(tmp)
    rec = np.asarray(array[:, 2:]).reshape(len(array), len(PERIODS), 8)
    order = _select_order(rec)
    selected = rec[np.arange(len(rec))[:, None], order]

    return array, novelty_array, order, selected


def structure_features(song, beats, seconds, *, policy='metrical_context_v1', cache_dir=None, backend='auto'):
    """Return [N,48] features; seconds and beats refer to the same positions."""
    if policy not in STRUCTURE_POLICIES: raise ValueError(f'Unknown structure policy: {policy}')
    beats = np.asarray(beats, dtype=np.float64).reshape(-1)
    seconds = np.asarray(seconds, dtype=np.float64).reshape(-1)
    if len(beats) != len(seconds) or not np.isfinite(beats).all() or not np.isfinite(seconds).all():
        raise ValueError('beats and seconds must be finite arrays of the same length')
    out = np.zeros((len(beats), STRUCTURE_DIM), dtype=np.float32)
    if policy == 'off': return out
    for i, period in enumerate((4., 8., 16.)):
        angle = 2*np.pi*beats/period
        out[:, 2*i] = np.sin(angle); out[:, 2*i+1] = np.cos(angle)
    if policy == 'native_periodic_v1' or not len(beats): return out
    bank, novelty, order, selected = _bank(song, cache_dir, backend)
    anchors = np.asarray(bank[:, 0], dtype=np.float64)
    # Select candidates from the nearest anchor, then interpolate evidence for
    # that same period from the surrounding anchors. This never blends two
    # different periods into a single invented token.
    rec = np.asarray(bank[:, 2:]).reshape(len(bank), len(PERIODS), 8)
    hi = np.clip(np.searchsorted(anchors, beats, side='right'), 1, len(anchors)-1) if len(anchors)>1 else np.zeros(len(beats), dtype=int)
    lo = np.maximum(0, hi-1)
    alpha = np.clip((beats-anchors[lo])/np.maximum(anchors[hi]-anchors[lo], 1e-8), 0., 1.) if len(anchors)>1 else np.zeros(len(beats))
    nearest = np.where(alpha < .5, lo, hi)
    for slot in range(4):
        ids = order[nearest, slot]
        v = rec[lo, ids] * (1.-alpha[:, None]) + rec[hi, ids] * alpha[:, None]
        p = PERIODS[ids]; wt = v[:, 0]
        angle = 2*np.pi*beats/p
        z1 = (v[:, 3] + 1j*v[:, 4]) * np.exp(1j*angle)
        z2 = (v[:, 6] + 1j*v[:, 7]) * np.exp(2j*angle)
        token = np.column_stack((wt*z1.imag, wt*z1.real, wt*z2.imag, wt*z2.real,
            np.log1p(p)*wt, np.log1p(v[:, 5])*wt, wt, v[:, 1], v[:, 2], wt > 1e-8))
        out[:, 6+10*slot:16+10*slot] = token
    out[:, 46] = np.interp(beats, novelty[:, 0], novelty[:, 1], left=0., right=0.)
    out[:, 47] = np.clip(out[:, 46] * np.sum(out[:, [12,22,32,42]], axis=1), 0., 1.)
    return out


def structure_features_and_transition(song,beats,seconds,previous_beat=None,previous_second=None,
                                      *,policy='metrical_context_v1',cache_dir=None,backend='auto'):
    """Query candidates and the committed predecessor in one vectorized read."""
    if (previous_beat is None)!=(previous_second is None):
        raise ValueError('Previous beat and second must both be supplied or absent')
    if previous_beat is None:
        features=structure_features(song,beats,seconds,policy=policy,cache_dir=cache_dir,backend=backend)
        return features,features.copy()
    b=np.asarray(beats,dtype=np.float64).reshape(-1)
    s=np.asarray(seconds,dtype=np.float64).reshape(-1)
    if len(b)!=len(s):raise ValueError('beats and seconds must have the same length')
    joined=structure_features(song,np.r_[b,float(previous_beat)],np.r_[s,float(previous_second)],
        policy=policy,cache_dir=cache_dir,backend=backend)
    features=joined[:-1]
    return features,features-joined[-1:]


def structure_segments(song, *, cache_dir=None, backend='auto', limit=4):
    """Inspectable local candidates; probabilities are heuristic, not calibrated.

    Ranges are anchor neighborhoods and overlap in evidence due to 32-beat
    windows. The grouping/time signature remains unknown. Unknown weight is
    the unassigned mass over all candidate periods, not just displayed slots.
    """
    if not 1 <= limit <= 4:
        raise ValueError('limit must be in [1,4]')
    bank, novelty, order, selected = _bank(song, cache_dir, backend)
    rec = np.asarray(bank[:, 2:]).reshape(len(bank), len(PERIODS), 8)
    anchors = np.asarray(bank[:, 0], dtype=np.float64)
    end = float(novelty[-1, 0])
    result=[]
    for row, center in enumerate(anchors):
        lo=max(0., (anchors[row-1]+center)/2.) if row else 0.
        hi=min(end, (center+anchors[row+1])/2.) if row+1<len(anchors) else end
        if hi <= lo: continue
        seconds = _seconds_at_beats(np.array([lo, hi]), song)
        candidates=[]
        for idx in order[row, :limit]:
            v=rec[row, idx]
            if v[0] <= 1e-8: continue
            candidates.append({'period_beats':float(PERIODS[idx]), 'pulse_beats':float(v[5]),
                'weight':float(v[0]), 'support':float(v[1]), 'phase_confidence':float(v[2]),
                'phase_radians':float(np.arctan2(v[4], v[3])),
                'first_harmonic_amplitude':float(np.hypot(v[3],v[4])),
                'second_harmonic_amplitude':float(np.hypot(v[6],v[7])), 'grouping':'unknown'})
        result.append({'start_beat':float(lo), 'end_beat':float(hi),
            'start_second':float(seconds[0]), 'end_second':float(seconds[1]),
            'unknown_weight':float(np.clip(1.-rec[row,:,0].sum(),0.,1.)),
            'candidates':candidates})
    return result
