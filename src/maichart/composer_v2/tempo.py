"""Audio-derived constant or piecewise tempo, with explicit manual precedence.

Only detector outputs are cached; requested clocks are resolved each call.
Piecewise clocks require periodic support and continuous phase boundaries.
"""
from __future__ import annotations
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import numpy as np
from maichart.event_timing import EventTempo
from maichart.tempo_clock import TempoClock

# This is the DETECTOR cache identity, not the clock fitter version. Cached
# peaks remain valid when fitting/diagnostics change; do not rebuild audio.
POLICY = 'composer_v2_global_beat_phase_v1'

def read_tempo_map_argument(value):
    """Accept the existing inline JSON syntax and portable @file arguments."""
    if value.startswith('@'):
        try:
            value = Path(value[1:]).read_text(encoding='utf-8-sig')
        except OSError as exc:
            raise ValueError(f'Cannot read tempo map: {exc}') from exc
    return json.loads(value)


def _finite(value, name, positive=False):
    if isinstance(value, bool):
        raise ValueError(name + ' must be a finite number')
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(name + ' must be finite' + (' and positive' if positive else ''))
    return value


def _digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _validate_beats(beats):
    beats = np.asarray(beats, dtype=np.float64)
    if beats.ndim != 1 or len(beats) < 8 or not np.isfinite(beats).all() or (beats < 0).any():
        raise ValueError('Tempo detection requires at least eight finite nonnegative beat times')
    beats = np.unique(beats)
    if len(beats) < 8:
        raise ValueError('Tempo detection requires at least eight distinct beats')
    return beats


def _beat_this(path, cache_dir, digest, identity, device):
    import torch
    from beat_this.inference import Postprocessor
    from maichart.composer import structure_backend as backend
    # Only release an instance this call creates on this thread. Existing
    # shared analyzers belong to their owners and must remain cached.
    thread = threading.get_ident()
    with backend._model_lock:
        previous = set(backend._models)
    try:
        frames = backend.beat_this_frames(SimpleNamespace(source=path,
            metadata={'audio_sha256': digest}), cache_dir, identity, device=device)
        beats, downbeats = Postprocessor(type='minimal', fps=identity['fps'])(
            torch.from_numpy(np.array(frames[:, 0], copy=True)),
            torch.from_numpy(np.array(frames[:, 1], copy=True)))
        downbeat_arr = np.asarray(downbeats, dtype=np.float64)
        downbeat_arr = (np.unique(downbeat_arr[np.isfinite(downbeat_arr) & (downbeat_arr >= 0)])
                        if downbeat_arr.ndim == 1 else np.empty(0, dtype=np.float64))
        return _validate_beats(beats), dict(downbeat_times_sec=downbeat_arr.tolist(),
                                            downbeat_count=int(len(downbeat_arr)))
    finally:
        with backend._model_lock:
            created = [key for key in backend._models if key not in previous and key[-1] == thread]
            for key in created:
                del backend._models[key]
        # Removing our reference releases tensors naturally; avoid global CUDA
        # allocator operations while another job may be training.


def _acoustic(path, cache_dir, digest):
    import librosa
    from maichart.form.waveform_cache import mapped_waveform
    from maichart.generation.tempo import detect_global_bpm
    waveform, rate = mapped_waveform(path, digest, cache_dir)
    if not len(waveform) or float(np.max(np.abs(waveform))) < 1e-6:
        raise ValueError('Silent or empty audio has no detectable tempo')
    target = 22050
    signal = librosa.resample(np.asarray(waveform), orig_sr=rate, target_sr=target) if rate != target else np.asarray(waveform)
    hop = 256
    onset = librosa.onset.onset_strength(y=signal, sr=target, hop_length=hop)
    if not np.isfinite(onset).all() or float(np.max(onset)) <= 1e-5:
        raise ValueError('No usable acoustic onset energy')
    bpm, confidence = detect_global_bpm(onset, fps=target / hop)
    if not math.isfinite(bpm) or bpm <= 0:
        raise ValueError('Acoustic global BPM detection failed')
    dynamic = librosa.feature.tempo(onset_envelope=onset,sr=target,hop_length=hop,
        start_bpm=bpm,aggregate=None,max_tempo=500.)
    _, frames = librosa.beat.beat_track(onset_envelope=onset, sr=target,
        hop_length=hop, bpm=dynamic, trim=False)
    beats = librosa.frames_to_time(frames, sr=target, hop_length=hop)
    return _validate_beats(beats), dict(acoustic_bpm=float(bpm), acoustic_confidence=float(confidence))


def _detect(path, cache_dir, device):
    from maichart.composer.structure_backend import backend_identity
    digest = _digest(path)
    primary = backend_identity('auto', refresh=True)
    try:
        acoustic_version = importlib.metadata.version('librosa')
    except importlib.metadata.PackageNotFoundError:
        acoustic_version = 'unavailable'
    acoustic = dict(name='librosa_flux_dynamic_bpm_v2',
        package_version=acoustic_version, waveform_policy='native_mono_float32_v1')
    identities = [primary, acoustic] if primary['name'] == 'beat_this_raw_logits_v1' else [acoustic]
    failures = [] if len(identities) == 2 else [primary.get('fallback', 'BeatThis unavailable')]
    for identity in identities:
        key = hashlib.sha256(json.dumps(dict(policy=POLICY, audio_sha256=digest,
            backend=identity), sort_keys=True).encode()).hexdigest()
        cached = Path(cache_dir) / 'composer_v2_timing_v1' / (key + '.json') if cache_dir else None
        if cached and cached.is_file():
            try:
                payload = json.loads(cached.read_text(encoding='utf-8'))
                beats = _validate_beats(payload['beat_times_sec'])
                return beats, dict(payload['diagnostics'], cache_hit=True, fallback_reasons=failures)
            except (OSError, ValueError, KeyError, TypeError):
                pass
        try:
            if identity['name'] == 'beat_this_raw_logits_v1':
                beats, extra = _beat_this(path, cache_dir, digest, identity, device)
            else:
                beats, extra = _acoustic(path, cache_dir, digest)
            report = dict(backend=identity, algorithm_policy=POLICY, audio_sha256=digest,
                beat_count=len(beats), detector_device=str(device) if identity is primary else 'cpu',
                **extra)
            if cached:
                cached.parent.mkdir(parents=True, exist_ok=True)
                descriptor, temporary = tempfile.mkstemp(dir=cached.parent, suffix='.tmp')
                try:
                    with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
                        json.dump(dict(beat_times_sec=beats.tolist(), diagnostics=report), stream, allow_nan=False)
                    os.replace(temporary, cached)
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
            return beats, dict(report, cache_hit=False, fallback_reasons=failures)
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            failures.append(identity['name'] + ': ' + type(exc).__name__ + ': ' + str(exc))
    raise ValueError('Automatic timing detection failed; specify BPM and first explicitly. ' + '; '.join(failures))


def _global_fit(beats):
    from .tempo_fit import fit_global_clock
    return fit_global_clock(_validate_beats(beats))


def _phase(beats, tempo):
    from .tempo_fit import fit_phase
    return fit_phase(_validate_beats(beats), TempoClock(tempo, 0.))


def _timing_envelope(path,cache_dir):
    """Read native-rate flux without downloading or running a music encoder."""
    from .audio import AudioFeatureStore
    store=AudioFeatureStore(cache_dir or 'cache/chart_composer_v2',backend='mert',device='cpu')
    try:
        try:audio=store.load(path)
        except (OSError,ValueError):
            store.close()
            store=AudioFeatureStore(cache_dir or 'cache/chart_composer_v2',backend='native',device='cpu')
            audio=store.prepare(path)
        return np.array(audio.mel_times_sec),np.array(audio.mel[:,96])
    finally:store.close()


def _recover_cached_downbeats(cache_dir, detected):
    raw = detected.get('downbeat_times_sec')
    if raw is not None:
        arr = np.asarray(raw, dtype=np.float64)
        return arr[np.isfinite(arr) & (arr >= 0)] if arr.ndim == 1 else np.empty(0, dtype=np.float64)
    backend_info = detected.get('backend') or {}
    digest = detected.get('audio_sha256')
    if not cache_dir or backend_info.get('name') != 'beat_this_raw_logits_v1' or not digest:
        return np.empty(0, dtype=np.float64)
    try:
        import torch
        from beat_this.inference import Postprocessor
        from maichart.composer.structure_backend import raw_cache_key
        key = raw_cache_key(digest, backend_info)
        cached_npy = Path(cache_dir) / 'composer_beat_this_v1' / (key + '.npy')
        if not cached_npy.is_file():
            return np.empty(0, dtype=np.float64)
        frames = np.load(cached_npy, mmap_mode='r', allow_pickle=False)
        if frames.ndim != 2 or frames.shape[1] != 2:
            return np.empty(0, dtype=np.float64)
        _, downbeats = Postprocessor(type='minimal', fps=backend_info.get('fps', 50))(
            torch.from_numpy(np.array(frames[:, 0], copy=True)),
            torch.from_numpy(np.array(frames[:, 1], copy=True)))
        arr = np.asarray(downbeats, dtype=np.float64)
        return np.unique(arr[np.isfinite(arr) & (arr >= 0)]) if arr.ndim == 1 else np.empty(0, dtype=np.float64)
    except Exception:
        return np.empty(0, dtype=np.float64)


def _reanchor_tempo_clock(tempo, old_first, *, scale=Fraction(1, 1), beat_shift=Fraction(0)):
    """Scale BPM/beats and shift origin while preserving every node's wall-clock second."""
    if any(change.beat * scale - beat_shift <= 0 for change in tempo[1:]):
        return None, float(old_first)
    bpm0_new = float(tempo[0].bpm) * float(scale)
    period_new = 60.0 / bpm0_new
    new_first = float(old_first) + float(beat_shift) * period_new
    new_tempo = [
        EventTempo(Fraction(0), bpm0_new),
        *[EventTempo(change.beat * scale - beat_shift, float(change.bpm) * float(scale)) for change in tempo[1:]],
    ]
    return new_tempo, new_first


def _canonical_beat_shift(first_sec, period_sec):
    m = -math.floor(float(first_sec) / period_sec)
    while float(first_sec) + m * period_sec >= period_sec:
        m -= 1
    while float(first_sec) + m * period_sec < 0.0:
        m += 1
    return m


def _calibrate_tempo_octave(tempo, effective_first, *, manual_first, downbeats, policy):
    raw_bpm = float(tempo[0].bpm)
    scale_frac = Fraction(1, 1)
    if raw_bpm < 85.0:
        for exp in range(1, 5):
            cand_scale = Fraction(1 << exp, 1)
            if 85.0 <= raw_bpm * float(cand_scale) <= 255.0:
                scale_frac = cand_scale
                break
    elif raw_bpm > 255.0:
        for exp in range(1, 5):
            cand_scale = Fraction(1, 1 << exp)
            if 85.0 <= raw_bpm * float(cand_scale) <= 255.0:
                scale_frac = cand_scale
                break
    factor = float(scale_frac)
    if scale_frac == 1:
        return tempo, float(effective_first), dict(policy=policy, applied=False, factor=1.0,
                                                   raw_bpm=raw_bpm, calibrated_bpm=raw_bpm)
    if manual_first is not None:
        calibrated, new_first = _reanchor_tempo_clock(
            tempo, effective_first, scale=scale_frac, beat_shift=Fraction(0))
        return calibrated, new_first, dict(policy=policy, applied=True, factor=factor,
                                           raw_bpm=raw_bpm, calibrated_bpm=float(calibrated[0].bpm))
    new_period = 60.0 / (raw_bpm * factor)
    m_norm = _canonical_beat_shift(effective_first, new_period)
    if scale_frac > 1:
        calibrated, new_first = _reanchor_tempo_clock(
            tempo, effective_first, scale=scale_frac, beat_shift=Fraction(m_norm))
        if calibrated is None:
            calibrated, new_first = _reanchor_tempo_clock(
                tempo, effective_first, scale=scale_frac, beat_shift=Fraction(0))
    else:
        denom = scale_frac.denominator
        valid_candidates = []
        for j in range(denom):
            cand_shift = Fraction(m_norm * denom + j, denom)
            while float(effective_first) + float(cand_shift) * new_period >= new_period:
                cand_shift -= 1
            while float(effective_first) + float(cand_shift) * new_period < 0.0:
                cand_shift += 1
            cand_tempo, cand_first = _reanchor_tempo_clock(
                tempo, effective_first, scale=scale_frac, beat_shift=cand_shift)
            if cand_tempo is not None:
                valid_candidates.append((j, cand_tempo, cand_first))
        if not valid_candidates:
            calibrated, new_first = _reanchor_tempo_clock(
                tempo, effective_first, scale=scale_frac, beat_shift=Fraction(0))
        elif len(downbeats) >= 2 and len(valid_candidates) > 1:
            def _hits(cand_tempo, cand_first):
                clk = TempoClock(cand_tempo, cand_first)
                ix = np.clip(np.searchsorted(clk._seconds_arr, downbeats, side='right') - 1, 0, len(cand_tempo) - 1)
                q = clk._beat_floats[ix] + (downbeats - clk._seconds_arr[ix]) / clk._scale_arr[ix]
                res = np.abs(q - np.rint(q)) * clk._scale_arr[ix]
                return int((res <= np.minimum(0.08, 0.20 * clk._scale_arr[ix])).sum())
            _, calibrated, new_first = max(
                valid_candidates, key=lambda item: (_hits(item[1], item[2]), -item[0]))
        else:
            _, calibrated, new_first = valid_candidates[0]
    return calibrated, new_first, dict(policy=policy, applied=True, factor=factor,
                                       raw_bpm=raw_bpm, calibrated_bpm=float(calibrated[0].bpm))


def _align_bar_downbeat(path, cache_dir, tempo, effective_first, beats, downbeats, envelope, *, policy):
    period = 60.0 / float(tempo[0].bpm)
    m_base = _canonical_beat_shift(effective_first, period)
    base_tempo, base_first = _reanchor_tempo_clock(
        tempo, effective_first, scale=Fraction(1, 1), beat_shift=Fraction(m_base))
    if base_tempo is None:
        m_base = 0
        base_tempo = tempo
        base_first = float(effective_first)
    clock0 = TempoClock(base_tempo, base_first)
    best_k = 0
    has_consensus = False
    source_kind = 'none'
    counts_list = None
    if len(downbeats) >= 2:
        ix = np.clip(np.searchsorted(clock0._seconds_arr, downbeats, side='right') - 1, 0, len(base_tempo) - 1)
        q = clock0._beat_floats[ix] + (downbeats - clock0._seconds_arr[ix]) / clock0._scale_arr[ix]
        nearest = np.rint(q).astype(np.int64)
        tol = np.minimum(0.08, 0.25 * clock0._scale_arr[ix])
        inliers = nearest[np.abs(q - nearest) * clock0._scale_arr[ix] <= tol]
        if len(inliers) >= 2:
            counts = np.bincount(np.mod(inliers, 4), minlength=4)
            counts_list = [int(v) for v in counts]
            best_k = int(np.argmax(counts))
            consensus = float(counts[best_k]) / float(counts.sum())
            has_consensus = int(counts[best_k]) >= 2 and consensus >= 0.35
            source_kind = 'beat_this_downbeats'
    if not has_consensus:
        if envelope is None:
            try:
                envelope = _timing_envelope(path, cache_dir)
            except (ImportError, OSError, ValueError, RuntimeError):
                envelope = None
        if envelope is not None:
            ft, flux = envelope
            if len(ft) and float(ft[-1]) > max(0.0, base_first) + 4.0 * period:
                max_beats = min(512, int((float(ft[-1]) - base_first) / period))
                if max_beats >= 8:
                    sample_sec = clock0.at_floats(np.arange(max_beats, dtype=np.float64))
                    valid_sec = sample_sec[(sample_sec >= 0.0) & (sample_sec <= float(ft[-1]))]
                    if len(valid_sec) >= 8:
                        vals = np.maximum(np.interp(valid_sec, ft, flux), 0.0)
                        scores = np.array([float(vals[k::4].mean()) if len(vals[k::4]) else 0.0 for k in range(4)])
                        best_k = int(np.argmax(scores))
                        med = float(np.median(scores))
                        has_consensus = scores[best_k] > 1e-5 and scores[best_k] > max(med * 1.08, med + 1e-6)
                        if has_consensus:
                            source_kind = 'acoustic_bar_flux'
    if not has_consensus or best_k == 0:
        return tempo, float(effective_first), envelope, dict(
            policy=policy, applied=False, source=source_kind, dominant_bar_phase=int(best_k),
            selected_bar_beat_offset=0, blocked_by_intro_pickup=False, phase_counts=counts_list)
    total_shift = Fraction(m_base + best_k)
    shifted_tempo, cand_first = _reanchor_tempo_clock(
        tempo, effective_first, scale=Fraction(1, 1), beat_shift=total_shift)
    piecewise_ok = shifted_tempo is not None
    if not piecewise_ok:
        cand_first = float(effective_first) + float(total_shift) * period
    beat_before = bool(len(beats) and float(beats[0]) < cand_first - 0.20 * period)
    flux_before = False
    if envelope is None and not beat_before:
        try:
            envelope = _timing_envelope(path, cache_dir)
        except (ImportError, OSError, ValueError, RuntimeError):
            envelope = None
    if envelope is not None:
        ft, flux = envelope
        if len(ft) and len(flux):
            pos_flux = np.maximum(np.asarray(flux, dtype=np.float64), 0.0)
            thresh = max(float(np.quantile(pos_flux, 0.90)) * 0.35, 1e-4)
            early = (ft >= 0.02) & (ft < cand_first - 0.06)
            flux_before = bool(np.any(pos_flux[early] > thresh))
    blocked = (not piecewise_ok) or beat_before or flux_before or cand_first < 0.0
    if blocked:
        return tempo, float(effective_first), envelope, dict(
            policy=policy, applied=False, source=source_kind, dominant_bar_phase=int(best_k),
            selected_bar_beat_offset=0, blocked_by_intro_pickup=bool(beat_before or flux_before),
            piecewise_boundary_blocked=not piecewise_ok, phase_counts=counts_list)
    return shifted_tempo, cand_first, envelope, dict(
        policy=policy, applied=True, source=source_kind, dominant_bar_phase=int(best_k),
        selected_bar_beat_offset=int(best_k), blocked_by_intro_pickup=False, phase_counts=counts_list)


def resolve_timing(audio_path, cache_dir, bpm=None, first=None, tempo_map=None, device='cpu',
                   tempo_detection='legacy', tempo_octave_policy='off', downbeat_alignment='off'):
    if tempo_detection not in ('auto','constant','legacy'):raise ValueError('Unknown tempo detection mode')
    if tempo_octave_policy not in ('off', 'maimai_pulse_v1'):raise ValueError('Unknown tempo octave policy')
    if downbeat_alignment not in ('off', 'bar_downbeat_v1'):raise ValueError('Unknown downbeat alignment policy')
    path = Path(audio_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if isinstance(tempo_map, dict):
        # An explicitly supplied --first always takes precedence over the file.
        if first is None:
            first = tempo_map.get('first')
        tempo_map = tempo_map.get('tempo_map', tempo_map.get('tempo_changes'))
        if not isinstance(tempo_map, list):
            raise ValueError('Tempo map object requires a tempo_map array')
    manual_bpm = None if bpm is None else _finite(bpm, 'BPM', positive=True)
    manual_first = None if first is None else _finite(first, 'first')
    if tempo_map is not None:
        if not isinstance(tempo_map, (list, tuple)) or not tempo_map:
            raise ValueError('tempo_map must be a nonempty list')
        tempo = [EventTempo(Fraction(str(value['beat'])), _finite(value['bpm'], 'tempo_map BPM', True))
                 if isinstance(value, dict) else EventTempo(Fraction(value.beat), _finite(value.bpm, 'tempo_map BPM', True))
                 for value in tempo_map]
        tempo = sorted(tempo, key=lambda value: value.beat)
        if len({value.beat for value in tempo}) != len(tempo):
            raise ValueError('tempo_map cannot contain duplicate beat positions')
        TempoClock(tempo, 0.)
    else:
        tempo = [EventTempo(Fraction(0), manual_bpm)] if manual_bpm is not None else None
    report = dict(bpm_auto=tempo is None, first_auto=manual_first is None,tempo_detection=tempo_detection,
        tempo_octave_policy=tempo_octave_policy, downbeat_alignment_policy=downbeat_alignment,
        tempo_source='manual_tempo_map' if tempo_map is not None else 'manual_bpm' if manual_bpm is not None else 'audio_global_consensus',
        first_source='manual' if manual_first is not None else 'detected_beat_phase',
        warnings=[], automatic_tempo_map_applied=False,limitations=['Automatic maps estimate periodic audio clocks; they do not recover an authored tempo map uniquely.',
            'Half/double tempo and integer beat/bar origin can be ambiguous; no meter is inferred.',
            'Automatic first is a canonical phase modulo the initial beat period, not the first sound time.'])
    if tempo is None or manual_first is None:
        beats, detected = _detect(path, cache_dir, device)
        report.update(detected)
        if tempo is None:
            measured, diagnostics = _global_fit(beats)
            report.update(diagnostics, detected_global_bpm=measured,
                pulse_alternatives_bpm=[measured/2.,measured,measured*2.])
            tempo = [EventTempo(Fraction(0), measured)]
            from .tempo_fit import inspect_tempo_regions
            report.update(inspect_tempo_regions(beats, measured))
            if report['variable_tempo_suspected']:
                report['warnings'].append('检测到局部脉冲速度变化，可能是真实变速或拍层切换；这些候选尚未自动应用，请核对时间轴。')
            if report['global_fit_coverage'] < .65:
                report['warnings'].append('恒速拟合的一致性偏低，可能存在变速、细分拍或漏拍；请核对 BPM 与半速／倍速候选。')
        phase, diagnostics = _phase(beats, tempo)
        report.update(diagnostics, detected_phase_first=phase)
        if report['phase_fit_coverage'] < .65:
            report['warnings'].append('首拍相位仅由主要一致节拍拟合；其他峰可能属于细分拍、变速或检测误差，请试听确认。')
        effective_first = phase if manual_first is None else manual_first
        envelope = None
        if tempo_detection!='legacy' and tempo_map is None:
            from .tempo_map_fit import refine_acoustic,fit_variable_clock
            try:
                ft,flux=_timing_envelope(path,cache_dir);envelope=(ft,flux)
                refined,audio_phase,calibration=refine_acoustic(ft,flux,tempo[0].bpm,phase,fixed_bpm=manual_bpm is not None)
                report['acoustic_clock_calibration']=calibration
                if manual_bpm is None:tempo=[EventTempo(Fraction(0),refined)]
                if manual_first is None:
                    effective_first=audio_phase
                    if calibration.get('applied'):report['first_source']='acoustic_refined_phase'
            except (ImportError,OSError,ValueError,RuntimeError) as exc:
                report['acoustic_clock_calibration']=dict(applied=False,error=str(exc))
                report['warnings'].append('细粒度起音校准不可用，沿用节拍峰拟合；请核对时间轴。')
            if tempo_detection=='auto' and manual_bpm is None:
                ft,flux=envelope if envelope is not None else (None,None)
                changes,map_first,analysis=fit_variable_clock(beats,tempo[0].bpm,effective_first,
                    flux,ft,first_override=manual_first)
                report['variable_clock_analysis']=analysis
                report['automatic_tempo_map_applied']=changes is not None
                if changes is not None:
                    tempo=changes;effective_first=map_first
                    report['tempo_source']='audio_piecewise_consensus'
                    if manual_first is None:report['first_source']='piecewise_audio_phase'
                    report['warnings']=[w for w in report['warnings'] if '尚未自动应用' not in w]
                    report['warnings'].append('已应用音频估计的变速表；边界在候选位置附近细化，搜索范围不代表真实误差界限，半速／倍速仍需试听核对。')
                elif report.get('variable_tempo_suspected'):
                    report['warnings'].append('变速候选未通过连续性或覆盖检查，保留候选段信息；本次仍使用恒速校准。')
        downbeats = None
        if tempo_octave_policy == 'maimai_pulse_v1' and manual_bpm is None and tempo_map is None:
            downbeats = _recover_cached_downbeats(cache_dir, detected)
            tempo, effective_first, oct_report = _calibrate_tempo_octave(
                tempo, effective_first, manual_first=manual_first, downbeats=downbeats, policy=tempo_octave_policy)
            report['tempo_octave_calibration'] = oct_report
            if oct_report['applied']:
                report['tempo_source'] = report['tempo_source'] + '_octave_calibrated'
        if downbeat_alignment == 'bar_downbeat_v1' and manual_first is None:
            if downbeats is None:
                downbeats = _recover_cached_downbeats(cache_dir, detected)
            tempo, effective_first, envelope, db_report = _align_bar_downbeat(
                path, cache_dir, tempo, effective_first, beats, downbeats, envelope, policy=downbeat_alignment)
            report['downbeat_alignment'] = db_report
            if db_report['applied']:
                report['first_source'] = 'bar_downbeat_aligned'
            elif db_report.get('blocked_by_intro_pickup'):
                report['warnings'].append('检测到强拍相位前存在前奏／弱起起音，为避免首音落入负拍已保留原首拍相位。')
        if (tempo_detection != 'legacy' and tempo_map is None) or tempo_octave_policy != 'off' or downbeat_alignment != 'off':
            clock=TempoClock(tempo,effective_first)
            ix=np.clip(np.searchsorted(clock._seconds_arr,beats,side='right')-1,0,len(tempo)-1)
            q=clock._beat_floats[ix]+(beats-clock._seconds_arr[ix])/clock._scale_arr[ix]
            residual=abs(q-np.round(q))*clock._scale_arr[ix]
            report['effective_clock_peak_coverage']=float((residual<.05).mean())
            report['effective_clock_residual_p90_sec']=float(np.quantile(residual,.9))
    else:
        effective_first = manual_first
        report.update(backend={'name': 'manual'}, cache_hit=False)
    report.update(effective_bpm=float(tempo[0].bpm), effective_first=float(effective_first),
        effective_tempo_map=[dict(beat=str(value.beat), bpm=float(value.bpm)) for value in tempo])
    return tempo, float(effective_first), report


def main(argv=None):
    """Measure timing without loading a Composer checkpoint or generating notes."""
    import argparse
    parser = argparse.ArgumentParser(description='Measure Composer v2 BPM/beat phase without generating a chart')
    parser.add_argument('--audio', required=True)
    parser.add_argument('--cache-dir', default='cache/chart_composer_v2')
    parser.add_argument('--bpm', type=float)
    parser.add_argument('--first', type=float)
    parser.add_argument('--tempo-map', type=read_tempo_map_argument)
    parser.add_argument('--tempo-detection',choices=('auto','constant','legacy'),default='auto')
    parser.add_argument('--tempo-octave-policy', choices=('off', 'maimai_pulse_v1'), default='off')
    parser.add_argument('--downbeat-alignment', choices=('off', 'bar_downbeat_v1'), default='off')
    parser.add_argument('--output', help='Optional JSON report path; no chart files are modified')
    parser.add_argument('--cpu-threads', type=int, default=2)
    args = parser.parse_args(argv)
    if args.cpu_threads < 1:
        parser.error('--cpu-threads must be positive')
    import torch
    torch.set_num_threads(args.cpu_threads)
    _, _, report = resolve_timing(args.audio, args.cache_dir, bpm=args.bpm,
        first=args.first, tempo_map=args.tempo_map, device='cpu',
        tempo_detection=args.tempo_detection,
        tempo_octave_policy=args.tempo_octave_policy,
        downbeat_alignment=args.downbeat_alignment)
    content = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content+'\n', encoding='utf-8')
    print(content, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
