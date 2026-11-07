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
        beats, _ = Postprocessor(type='minimal', fps=identity['fps'])(
            torch.from_numpy(np.array(frames[:, 0], copy=True)),
            torch.from_numpy(np.array(frames[:, 1], copy=True)))
        return _validate_beats(beats), {}
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


def resolve_timing(audio_path, cache_dir, bpm=None, first=None, tempo_map=None, device='cpu',tempo_detection='legacy'):
    if tempo_detection not in ('auto','constant','legacy'):raise ValueError('Unknown tempo detection mode')
    path = Path(audio_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
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
        if tempo_detection!='legacy' and tempo_map is None:
            from .tempo_map_fit import refine_acoustic,fit_variable_clock
            envelope=None
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
    parser.add_argument('--tempo-map', type=json.loads)
    parser.add_argument('--tempo-detection',choices=('auto','constant','legacy'),default='auto')
    parser.add_argument('--output', help='Optional JSON report path; no chart files are modified')
    parser.add_argument('--cpu-threads', type=int, default=2)
    args = parser.parse_args(argv)
    if args.cpu_threads < 1:
        parser.error('--cpu-threads must be positive')
    import torch
    torch.set_num_threads(args.cpu_threads)
    _, _, report = resolve_timing(args.audio, args.cache_dir, bpm=args.bpm,
        first=args.first, tempo_map=args.tempo_map, device='cpu',tempo_detection=args.tempo_detection)
    content = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content+'\n', encoding='utf-8')
    print(content, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
