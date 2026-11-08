"""Generate a Composer v2 chart using the checkpoint's sequence policy."""
from __future__ import annotations

import argparse
from copy import deepcopy
from fractions import Fraction
from functools import lru_cache
import json
import math
from pathlib import Path

import numpy as np
import torch

from maichart.event_timing import EventPoint, EventTempo
from maichart.tempo_clock import TempoClock
from maichart.generation.event_compile import compile_event_points
from maichart.composer.event_set_codec import _atoms
from maichart.composer.guard import EventGuard
from .checkpoint import read, inspect_payload
from .codec import ComponentCodec
from .codec import render_atom
from .data import ComposerV2Dataset, ComposerV2Song, collate_composer_v2, _beats_at_seconds
from .conditions import resolve_generation_condition


def parser():
    from .defaults import GENERATION_DEFAULTS
    result = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'audio', 'output'):
        result.add_argument('--' + name, required=True)
    result.add_argument('--cache-dir', default='cache/chart_composer_v2')
    result.add_argument('--bpm', type=float, default=None, help='Omit for automatic tempo analysis (detects variable tempo when --tempo-detection=auto, or constant BPM when constant); manual --tempo-map overrides')
    result.add_argument('--first', type=float, default=None, help='Omit to detect beat phase; explicit zero is preserved')
    result.add_argument('--level', type=float, default=13.)
    result.add_argument('--condition', type=json.loads, default=None,
                        help='Explicit Radar/Style overrides as a JSON object; omitted keys are automatic')
    result.add_argument('--intensity-curve',default=None,help='Edited Composer v2 rate profile JSON; omitted uses the trained audio prediction')
    result.add_argument('--intensity-strength',type=float,default=1.,help='Blend of edited and automatically predicted curves, 0–1')
    result.add_argument('--seed', type=int, default=0)
    result.add_argument('--temperature', type=float, default=GENERATION_DEFAULTS['temperature'],
                        help='Timing temperature and the default configuration temperature')
    result.add_argument('--top-p', type=float, default=GENERATION_DEFAULTS['top_p'],
                        help='Timing nucleus threshold and the default configuration threshold')
    result.add_argument('--action-temperature', type=float,
                        help='Override configuration temperature for a continuous-event checkpoint')
    result.add_argument('--action-top-p', type=float,
                        help='Override configuration nucleus threshold for a continuous-event checkpoint')
    result.add_argument('--onset-strength', type=float, default=None,
                        help='Trained phase-only acoustic reference: 0 off, 1 trained setting, up to 2 experimental emphasis')
    result.add_argument('--intercept-tap-on-slide',action=argparse.BooleanOptionalAction,default=GENERATION_DEFAULTS['intercept_tap_on_slide'],help='Enforce native hard TapOnSlide findings and clean residual affected notes')
    result.add_argument('--tap-on-slide-threshold-ms',type=float,default=GENERATION_DEFAULTS['tap_on_slide_threshold_ms'])
    result.add_argument('--candidates', type=int, choices=(1, 2, 4), default=GENERATION_DEFAULTS['candidates'])
    result.add_argument('--count', type=int, default=GENERATION_DEFAULTS['count'], help='Number of complete charts; sequential seeds, shared model/audio')
    result.add_argument('--batch-manifest', help=argparse.SUPPRESS)
    result.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    result.add_argument('--cpu-threads', type=int, default=2)
    result.add_argument('--max-events', type=int, default=0)
    result.add_argument('--tempo-map', type=json.loads)
    result.add_argument('--tempo-detection',choices=('auto','constant','legacy'),default='auto')
    result.add_argument('--meter-map', type=json.loads)
    result.add_argument('--window-retries', type=int, default=16,
                        help='Proposal attempts per event for continuous models or per window for older models')
    result.add_argument('--rollback-windows', type=int, default=2,
                        help='Recovery budget per generation frontier for continuous models or per chart for older models')
    result.add_argument('--reference-selection',choices=('auto','aligned'),default='auto',
        help='auto: the reference the checkpoint was trained with; aligned: beat-synchronous aligned choice '
             '(training-free for older continuous checkpoints, already built into aligned_rhythm_v1)')
    result.add_argument('--exhaustion-policy',choices=('auto','report','stop'),default='auto',
        help='auto: continuous checkpoints continue physical conflicts with a report; old window checkpoints stop')
    return result


def _sample(logits, allowed, generator, temperature, top_p):
    if not allowed:
        raise ValueError('Component grammar has no legal continuation')
    allowed = torch.as_tensor(sorted(set(allowed)), dtype=torch.long, device=logits.device)
    values = logits[allowed].float()
    if not torch.isfinite(values).all():
        raise ValueError('Nonfinite component logits')
    if temperature == 0:
        return int(allowed[values.argmax()])
    probabilities = torch.softmax(values / temperature, dim=-1)
    probabilities, order = torch.sort(probabilities, descending=True)
    keep = probabilities.cumsum(-1) - probabilities < top_p
    probabilities = probabilities * keep
    probabilities = probabilities / probabilities.sum()
    selected = torch.multinomial(probabilities, 1, generator=generator)
    return int(allowed[order[selected]].item())


def _device_batch(item, device):
    batch = collate_composer_v2([item])
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


@lru_cache(maxsize=8192)
def _atom_heads(text):
    from maichart.notes import parse_timing_point_notes
    from maichart.timing import RawTimingPoint
    parsed=parse_timing_point_notes(RawTimingPoint(index=0,raw=text,note_text=text,
        beat=Fraction(0),tick=0,time_sec=0.,division=4,bpm=120.))
    return tuple(str(note.position) for note in parsed.notes)


class WindowSampler:
    """One shared acoustic encoding, multiple causal component continuations."""
    def __init__(self, model, codec, item, device, generator, temperature=.9, top_p=.95):
        self.model, self.codec = model, codec
        self.device, self.generator = device, generator
        self.temperature, self.top_p = temperature, top_p
        self.batch = _device_batch(item, device)
        self.prefix = item['input_ids'].tolist()
        self.max_tokens = int(model.config.max_tokens)
        self.cache = model.encode_audio(self.batch) if hasattr(model, 'encode_audio') else None

    def next(self, target, allowed):
        ids = self.prefix + list(target)
        if len(ids) > self.max_tokens:
            raise ValueError('Generated complete window exceeds checkpoint max_tokens; no tokens truncated')
        batch = dict(self.batch)
        batch['input_ids'] = torch.tensor([ids], dtype=torch.long, device=self.device)
        batch['token_beats'] = torch.tensor([self.codec.token_clocks(ids)], dtype=torch.float32,
                                           device=self.device)
        batch['token_mask'] = torch.ones((1, len(ids)), dtype=torch.bool, device=self.device)
        if self.cache is not None and hasattr(self.model, 'decode'):
            logits = self.model.decode(batch, self.cache)
        else:
            logits = self.model(batch)
        if isinstance(logits, dict):
            logits = logits['logits']
        return _sample(logits[0, -1], allowed, self.generator, self.temperature, self.top_p)

    def _head_choices(self,allowed,atom,used):
        """Only exclude heads that the exact export syntax would reject twice.

        Endpoints of a Slide are not new heads. Compact Tap atoms can contain
        several heads; each of those participates in the same uniqueness check.
        """
        if not atom:return allowed
        text=[self.codec.tokens[i] for i in atom]
        positions=[t[2:] for t in text if t.startswith('P:')]
        if positions and text[0]!='N:tap':return allowed
        unavailable=set(used)|set(positions)
        return tuple(i for i in allowed if not (self.codec.tokens[i].startswith('P:')
            and self.codec.tokens[i][2:] in unavailable))

    def _rewind(self,target):
        if self.cache is None:return
        rewind=getattr(self.model,'rewind_decode_cache',None)
        if rewind is None:self.model.reset_decode_cache(self.cache)
        else:rewind(self.cache,len(self.prefix)+len(target))

    def joint(self, end_beat, *, guard=None, origin=0, event_retries=16):
        self.reset()
        target = [];used=set();atom_start=0;onset_start=None;retry=0
        trial=deepcopy(guard) if guard is not None else None
        self.repair_report=[];self.checked_guard=None
        while True:
            allowed=self.codec.allowed_next(target,end_beat=end_beat)
            allowed=self._head_choices(allowed,target[atom_start:],used)
            token = self.next(target,allowed)
            closes=token==self.codec.eos_id or token in self.codec.time_values
            if closes and onset_start is not None and trial is not None:
                point=self.codec.decode_points(target[onset_start:],origin=origin)[0]
                if not trial.accepts(point):
                    retry+=1
                    self.repair_report.append(dict(beat=str(point.beat),note_text=point.note_text,
                        reason=trial.last_reason,retry=retry,kept_tokens=onset_start))
                    if retry>=event_retries:
                        raise ValueError(f'Onset retry exhausted at beat {point.beat}: {trial.last_reason}')
                    # The completed prefix and its physical state remain committed.
                    # Retry this onset only; discard the speculative next-time token.
                    target=target[:onset_start+1];atom_start=len(target);used=set()
                    self._rewind(target)
                    continue
                trial.commit(point)
            target.append(token)
            if token == self.codec.eos_id:
                self.checked_guard=trial
                return target
            if token in self.codec.time_values:
                onset_start=len(target)-1;atom_start=len(target);used=set();retry=0
            elif token==self.codec.end_note_id:
                used.update(_atom_heads(render_atom([self.codec.tokens[i] for i in target[atom_start:]])))
                atom_start=len(target)

    def scaffold(self, points, origin, *, guard=None, event_retries=16):
        """Resample configuration while preserving every onset and atom count."""
        self.reset()
        target = []
        trial=deepcopy(guard) if guard is not None else None
        self.repair_report=[];self.checked_guard=None
        for point in points:
            target.extend(self.codec._time(point.beat - origin))
            prefix_length=len(target)
            for retry in range(event_retries):
                used=set();start=prefix_length
                for _ in _atoms(point.note_text):
                    atom=[]
                    while True:
                        token=self.next(target,self._head_choices(self.codec.allowed_atom_next(atom),atom,used))
                        target.append(token);atom.append(token)
                        if token==self.codec.end_note_id:
                            used.update(_atom_heads(render_atom([self.codec.tokens[i] for i in atom])))
                            break
                text='/'.join(p.note_text for p in self.codec.decode_points(
                    [self.codec.ids['T:0'],*target[start:]]))
                proposal=EventPoint(point.beat,text)
                if trial is None or trial.accepts(proposal):
                    if trial is not None:trial.commit(proposal)
                    break
                self.repair_report.append(dict(beat=str(point.beat),note_text=text,
                    reason=trial.last_reason,retry=retry+1,kept_tokens=prefix_length))
                target=target[:prefix_length];self._rewind(target)
            else:raise ValueError(f'Onset retry exhausted at beat {point.beat}: {trial.last_reason}')
        target.append(self.codec.eos_id)
        self.checked_guard=trial
        return target

    def reset(self):
        if self.cache is not None:
            # Keep acoustic memory on device and clear only mutable symbolic KV.
            self.model.reset_decode_cache(self.cache)


def _check_window(guard, points, failure_details=None):
    """Replay on a separate transaction; rejected trials never mutate committed state."""
    trial = deepcopy(guard)
    for point in points:
        if not trial.accepts(point):
            if failure_details is not None:
                failure_details.extend(dict(kind=issue.conflict_type, time_sec=issue.time,
                    note_beats=[str(note.beat) for note in issue.notes],
                    notes=[dict(beat=str(note.beat),raw=note.raw) for note in issue.notes])
                    for issue in getattr(trial,'last_conflicts',[]))
            return None, trial.last_reason
        trial.commit(point)
    return trial, None


def _new_guard(tempo, first, duration, *, guard_options=None):
    from .tail_guard import configure
    return configure(EventGuard(tempo, first, duration, min_same_head_sec=0.,
        touch_layout_policy='open', intercept_tap_on_slide=False, physical_policy='runtime_hard_v2'),guard_options)


def _rollback_committed(points, records, tempo, clock, boundary, duration, window_beats,
                        *, state_factory=None, guard_factory=None):
    """Revoke complete old windows only when a native action crosses the boundary."""
    if not points:
        return None
    if state_factory is None:
        from maichart.generation.sustain_state import SustainStateIndex
        state_factory = SustainStateIndex
    state = state_factory(points, tempo, clock.seconds[0])
    second = clock.at(boundary)
    active = [marker for marker in state._candidate_markers(second)
              if marker['source_beat'] < boundary and float(marker['end_second']) > second
              and marker.get('type') in ('hold', 'touch_hold', 'slide', 'wifi')]
    if not active:
        return None
    source = min(Fraction(marker['source_beat']) for marker in active)
    rewind = Fraction((source // window_beats) * window_beats)
    kept_points = [point for point in points if point.beat < rewind]
    kept_records = [record for record in records if Fraction(record['start_beat']) < rewind]
    factory = _new_guard if guard_factory is None else guard_factory
    guard = factory(tempo, clock.seconds[0], duration)
    for point in kept_points:
        if not guard.accepts(point):
            raise RuntimeError(f'Rollback replay rejected previously committed point at {point.beat}: '
                               f'{guard.last_reason}')
        guard.commit(point)
    report = dict(source_beat=str(source), rewind_start_beat=str(rewind),
                  failed_window_beat=str(boundary), revoked_points=len(points)-len(kept_points),
                  revoked_windows=len(records)-len(kept_records),
                  action_types=sorted({marker['type'] for marker in active}),
                  policy='revoke_whole_source_window_and_suffix_then_resample')
    return kept_points, kept_records, guard, rewind, report


def _comfort(prefix, candidate, tempo, clock, start, end, duration):
    from maichart.composer.stream_quality import _trajectory_comfort
    score, coverage = _trajectory_comfort([*prefix, *candidate], tempo, clock,
        clock.at(start), clock.at(end), duration)
    return float(np.clip(score, 0., 1.)), coverage


def _tempo_map(value, bpm):
    if not math.isfinite(bpm) or bpm <= 0:
        raise ValueError('bpm must be finite and positive')
    if value is None:
        return [EventTempo(Fraction(0), bpm)]
    if not isinstance(value, list) or not value:
        raise ValueError('tempo_map must be a nonempty JSON array')
    changes = [EventTempo(Fraction(str(item.get('beat', item.get('start_beat', 0)))),
                          float(item['bpm'])) for item in value]
    TempoClock(changes, 0.)
    return sorted(changes, key=lambda change: change.beat)


def _action_sampling(args):
    temperature = getattr(args, 'action_temperature', None)
    top_p = getattr(args, 'action_top_p', None)
    split = temperature is not None or top_p is not None
    return (args.temperature if temperature is None else temperature,
            getattr(args, 'top_p', .95) if top_p is None else top_p, split)


def _sampling_policy(args):
    action_temperature, action_top_p, split = _action_sampling(args)
    greedy = args.temperature == action_temperature == 0
    return dict(temperature=args.temperature, top_p=getattr(args,'top_p',.95),
        timing_temperature=args.temperature, timing_top_p=getattr(args,'top_p',.95),
        action_temperature=action_temperature, action_top_p=action_top_p,
        rng_policy='separate_timing_action_v1' if split else 'shared_v1',
        fully_greedy=greedy, requested_candidates=args.candidates,
        effective_candidates=1 if action_temperature==0 else args.candidates,
        requested_retries=getattr(args,'window_retries',16),
        effective_retries=1 if greedy else getattr(args,'window_retries',16),
        requested_rollback_budget=getattr(args,'rollback_windows',2),
        effective_rollback_budget=0 if greedy else getattr(args,'rollback_windows',2),
        recovery_budget=getattr(args,'rollback_windows',2),
        greedy_policy='single_deterministic_attempt_no_redundant_rollbacks')


@torch.inference_mode()
def generate(args, *, model_override=None, payload_override=None, audio_store_override=None, prepared_override=None):
    strength=getattr(args,'intensity_strength',1.)
    if not math.isfinite(strength) or not 0<=strength<=1:raise ValueError('intensity_strength must be in 0–1')
    from .tail_guard import options as tail_options
    tail_options(args)
    if args.device == 'cuda':
        from maichart.webui.startup_check import require_cuda
        require_cuda(torch)
    threads=int(getattr(args,'cpu_threads',2))
    if threads<1:raise ValueError('cpu_threads must be positive')
    torch.set_num_threads(threads)
    from .model import ComposerV2Config, ComposerV2
    if args.temperature < 0 or not 0 < args.top_p <= 1:
        raise ValueError('temperature must be nonnegative and top_p in (0,1]')
    if args.max_events < 0 or getattr(args, 'window_retries', 16) < 1 or getattr(args, 'rollback_windows', 2) < 0:
        raise ValueError('max_events/rollback_windows must be nonnegative and window_retries positive')
    if not all(math.isfinite(value) for value in (args.first, args.level, args.temperature) if value is not None):
        raise ValueError('first, level and temperature must be finite')
    sampling_policy = _sampling_policy(args)
    if not math.isfinite(sampling_policy['action_temperature']) or sampling_policy['action_temperature'] < 0:
        raise ValueError('action_temperature must be finite and nonnegative')
    if not 0 < sampling_policy['action_top_p'] <= 1:
        raise ValueError('action_top_p must be in (0,1]')
    effective_candidates = sampling_policy['effective_candidates']
    effective_retries = sampling_policy['effective_retries']
    effective_rollback_budget = sampling_policy['effective_rollback_budget']
    print('composer_v2_sampling '+json.dumps(sampling_policy,ensure_ascii=False),flush=True)
    payload = payload_override if payload_override is not None else read(args.checkpoint)
    inspected = inspect_payload(payload)
    if not inspected['can_generate']:
        raise ValueError(inspected['error'])
    checkpoint_info=dict(path=str(Path(args.checkpoint).resolve()),optimizer_steps=inspected['optimizer_steps'],
        epoch=inspected['epoch'],inference_only=inspected['inference_only'],family=inspected['family'])
    print('composer_v2_checkpoint_loaded '+json.dumps(checkpoint_info,ensure_ascii=False),flush=True)
    codec = ComponentCodec.from_dict(payload['codec'])
    config = ComposerV2Config.from_dict(inspected['model_config'])
    if getattr(args,'intensity_curve',None) and config.intensity_policy=='off':
        raise ValueError('This checkpoint has not learned intensity curves; use a new training stage')
    from .onset_readout import effective_strength
    onset_strength=effective_strength(getattr(args,'onset_strength',None),config.onset_readout_policy)
    print('composer_v2_onset_readout '+json.dumps(dict(policy=config.onset_readout_policy,
        strength=onset_strength,scope='conditional_phase_only')),flush=True)
    if sampling_policy['rng_policy']!='shared_v1' and config.sequence_policy!='continuous_events_v1':
        raise ValueError('Separate configuration sampling requires a continuous-event checkpoint')
    if getattr(args,'reference_selection','auto')=='aligned' and config.sequence_policy!='continuous_events_v1':
        raise ValueError('对齐参考检索仅适用于连续事件检查点')
    from .exhaustion import resolve_policy
    exhaustion_policy=resolve_policy(getattr(args,'exhaustion_policy','auto'),config.sequence_policy)
    print('composer_v2_exhaustion_policy '+json.dumps(dict(requested=getattr(args,'exhaustion_policy','auto'),
        effective=exhaustion_policy)),flush=True)
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    audio_path = Path(args.audio).resolve()
    if prepared_override is None:
        tempo,first,timing_report,audio = prepare_inputs(args,config,device,audio_store_override)
    else:
        tempo,first,timing_report,audio = prepared_override
    print('composer_v2_timing '+json.dumps(timing_report,ensure_ascii=False),flush=True)
    for warning in timing_report.get('warnings', ()):
        print('composer_v2_timing_warning '+warning,flush=True)
    bpm=float(tempo[0].bpm)
    clock = TempoClock(tempo, first)
    model = model_override if model_override is not None else ComposerV2(config, codec)
    if model_override is None:
        model.load_state_dict(payload['model_state_dict'], strict=True)
    model = model.to(device).eval()
    rng = torch.Generator(device=device).manual_seed(int(args.seed))
    end_value = float(_beats_at_seconds(clock, [audio.duration_sec])[0])
    if end_value <= 0:
        raise ValueError('Audio ends before chart beat zero; adjust first')
    end = Fraction(str(end_value))
    condition,condition_known,condition_report=resolve_generation_condition(
        args.level,getattr(args,'condition',None),config.condition_policy)
    if config.sequence_policy=='continuous_events_v1':
        from .stream_generate import generate_continuous
        return generate_continuous(model,audio,audio_path,tempo,first,timing_report,
            condition,condition_known,condition_report,args,rng,device,end,checkpoint_info=checkpoint_info)
    song = ComposerV2Song(audio_path.stem, audio_path, audio_path, 5, args.level,
        [], clock, end, condition, audio=audio,condition_known=condition_known)
    song.summary = dict(audio.summary(160))
    song.summary['beats'] = _beats_at_seconds(clock, song.summary['times_sec'])
    dataset = ComposerV2Dataset.__new__(ComposerV2Dataset)
    dataset.codec, dataset.max_tokens = codec, config.max_tokens
    dataset.window_beats = config.window_beats
    dataset.history_beats = config.history_beats
    dataset.future_beats = config.future_beats
    guard = _new_guard(tempo, first, audio.duration_sec,guard_options=tail_options(args))
    points, records, rollbacks = [], [], []
    index = 0
    while index < math.ceil(end_value / config.window_beats):
        start = Fraction(index * config.window_beats)
        stop = min(start + config.window_beats, end)
        item = dataset.build_window(song, points, start, training=False)
        sampler = WindowSampler(model, codec, item, device, rng, args.temperature, args.top_p)
        selected = None
        failures = []
        for attempt in range(effective_retries):
            try:
                baseline_ids = sampler.joint(stop - start,guard=guard,origin=start,event_retries=effective_retries)
                baseline = codec.decode_points(baseline_ids, origin=start)
            except ValueError as error:
                failures.append(str(error))
                continue
            alternatives = [(baseline,sampler.checked_guard,list(sampler.repair_report))]
            if baseline:
                for _ in range(effective_candidates - 1):
                    try:
                        ids=sampler.scaffold(baseline,start,guard=guard,event_retries=effective_retries)
                        alternatives.append((codec.decode_points(ids,origin=start),sampler.checked_guard,list(sampler.repair_report)))
                    except ValueError as error:
                        failures.append(str(error))
            feasible = []
            for candidate,trial,repairs in alternatives:
                reason=None
                if trial is None:trial,reason=_check_window(guard,candidate)
                if trial is None:
                    failures.append(reason)
                    continue
                if args.max_events and len(points) + len(candidate) > args.max_events:
                    raise ValueError('max_events would truncate a sampled window; increase the limit')
                score, coverage = (.5, {}) if len(alternatives) == 1 else _comfort(
                    points, candidate, tempo, clock, start, stop, audio.duration_sec)
                feasible.append((score, candidate, trial, coverage,repairs))
            if feasible:
                selected = max(feasible, key=lambda entry: entry[0])
                break
        if selected is None:
            if len(rollbacks) < effective_rollback_budget:
                rollback = _rollback_committed(points, records, tempo, clock, start,
                                               audio.duration_sec, config.window_beats,
                    guard_factory=lambda t,f,d:_new_guard(t,f,d,guard_options=tail_options(args)))
                if rollback is not None:
                    points, records, guard, rewind, report = rollback
                    report['number'] = len(rollbacks) + 1
                    report['failure_reasons'] = failures[-8:]
                    rollbacks.append(report)
                    print('composer_v2_rollback ' + json.dumps(report, ensure_ascii=False), flush=True)
                    index = int(rewind // config.window_beats)
                    continue
            diagnostic=Path(args.output).resolve().with_suffix('.failure.json')
            diagnostic.parent.mkdir(parents=True,exist_ok=True)
            diagnostic.write_text(json.dumps(dict(failed_window_beat=str(start),
                checkpoint=str(args.checkpoint),seed=args.seed,
                committed_points=[dict(beat=str(p.beat),note_text=p.note_text) for p in points],
                windows=records,onset_repairs=sampler.repair_report[-64:],reasons=failures,
                rollbacks=rollbacks,partial_chart_exported=False),ensure_ascii=False,indent=2),encoding='utf-8')
            raise RuntimeError(f'No native-feasible Composer v2 window at beat {start}; '
                               f'rollback_budget_used={len(rollbacks)}/'
                               f'{effective_rollback_budget}, effective_retries={effective_retries}, '
                               f'effective_candidates={effective_candidates}. Reasons: {failures[-8:]}; diagnostics={diagnostic}')
        score, candidate, guard, coverage,repairs = selected
        points.extend(candidate)
        record = dict(start_beat=str(start), end_beat=str(stop), events=len(candidate),
            attempts=attempt + 1, reference_start=str(item['reference_start']) if item['reference_start'] is not None else None,
            scaffold_candidates=len(alternatives), comfort=score, comfort_coverage=coverage,
            rejected_reasons=failures[-32:],onset_repairs=repairs[-32:],onset_repair_count=len(repairs),
            repair_policy='native_checked_onset_transactions_v1')
        records.append(record)
        print('composer_v2_window ' + json.dumps(record, ensure_ascii=False), flush=True)
        index += 1
    from .tail_guard import cleanup
    points,tail_report=cleanup(points,tempo,first,guard)
    # Remove unused future tempo directives outside the authored audio extent.
    output_tempo = [change for change in tempo if change.beat <= end]
    export_end = Fraction(math.ceil(end * config.ticks_per_beat), config.ticks_per_beat)
    text, _ = compile_event_points(points, tempo_changes=output_tempo, bpm=bpm,
        first=first, end_beat=export_end, title=audio_path.stem, level=str(args.level),
        designer='Composer v2', strict_notes=False)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding='utf-8')
    metadata = dict(family='chart_composer_v2', checkpoint=str(Path(args.checkpoint).resolve()),
        checkpoint_info=checkpoint_info,
        audio=str(audio_path), output=str(output), seed=args.seed, level=args.level,
        first=first,timing=timing_report,tempo_map=[dict(beat=str(t.beat), bpm=t.bpm) for t in tempo],
        meter_map=getattr(args, 'meter_map', None), meter_policy='metadata_only_no_fixed_4_4_condition',
        end_beat=str(export_end), audio_end_beat=str(end),
        end_policy='E_ceil_to_tick_only_notes_unchanged', duration_sec=audio.duration_sec, candidates=args.candidates,
        temperature=args.temperature, top_p=args.top_p, events=len(points), windows=records,
        eos_policy='window_end_only', hard_guard='runtime_hard_v2',
        rollbacks=rollbacks, rollback_budget=effective_rollback_budget, sampling_policy=sampling_policy,tap_on_slide=tail_report,
        window_beats=config.window_beats, history_beats=config.history_beats, future_beats=config.future_beats,
        candidate_policy='shared_onset_and_atom_count_scaffold',
        condition=condition_report,
        soft_score='native_trajectory_comfort_only_unknown_neutral_0.5',
        audio_identity=audio.identity)
    output.with_suffix('.composer_v2.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    print('composer_v2_done ' + json.dumps({'output': str(output), 'events': len(points),
                                           'windows': len(records)}, ensure_ascii=False), flush=True)
    return metadata


def prepare_inputs(args,config,device,audio_store_override=None):
    """Read once per batch; feature preparation is independent of chart seed."""
    from .audio import AudioFeatureStore
    from .tempo import resolve_timing
    audio_path=Path(args.audio).resolve()
    tempo,first,timing_report=resolve_timing(audio_path,args.cache_dir,bpm=args.bpm,
        first=args.first,tempo_map=getattr(args,'tempo_map',None),device='cpu',
        tempo_detection=getattr(args,'tempo_detection','auto'))
    store=audio_store_override if audio_store_override is not None else AudioFeatureStore(
        args.cache_dir,backend=config.audio_backend,model_path=config.mert_model,device=device)
    try:
        store.prepare(audio_path)
        audio=store.load(audio_path)
    finally:
        store.close()
    return tempo,first,timing_report,audio


def main(argv=None):
    args=parser().parse_args(argv)
    from .defaults import MAX_BATCH_COUNT
    if not 1<=args.count<=MAX_BATCH_COUNT:
        raise ValueError(f'count must be in 1..{MAX_BATCH_COUNT}')
    if args.count>1:
        from .batch_generate import generate_batch
        return generate_batch(args)
    return generate(args)


if __name__ == '__main__':
    main()
