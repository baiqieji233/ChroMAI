"""Generate exact NOTE/WAIT streams using the same causal training inputs."""
from __future__ import annotations
import argparse,json,math,time
from collections import Counter
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn import functional as F
from maichart.event_timing import EventPoint,EventTempo
from maichart.form.data import organize_song
from maichart.form.rational_audio import extract_rational_audio
from maichart.form.timing import normalize_meter_map
from maichart.generation.conditions import GenerationCondition,GENERATION_DEFAULTS
from maichart.generation.timeline import normalize_tempo_map
from maichart.generation.event_compile import compile_event_points
from maichart.generation.post_process import process_chart_points
from maichart.tempo_clock import TempoClock
from .stream_data import NOTE,WAIT,EOS,StreamStep,StreamInferenceContext,DeltaVocabulary,beat_at_second,second_at_beat
from .stream_model import to_device_batch
from .stream_checkpoint import load,PROFILE
from .guard import EventGuard
from .events import event_metadata
from .stream_sampling import kind_logits_for_audio,note_delta_logits_for_audio
from .stream_proposal import sample_timed_event
from .stream_timing import current_timing_deltas

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('audio','checkpoint','output'):p.add_argument('--'+key,required=True)
    for key in ('condition-file','tempo-map','meter-map','intensity-curve-file'):p.add_argument('--'+key)
    p.add_argument('--bpm',type=float);p.add_argument('--first',type=float,default=0.)
    p.add_argument('--level',type=float,default=13.3);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--device',default='auto');p.add_argument('--max-events',type=int,default=0)
    for key,default in [('temperature',.65),('pointer-temperature',None),('action-temperature',None),('min-same-head-ms',50.),('intensity-curve-strength',.65)]:p.add_argument('--'+key,type=float,default=default)
    p.add_argument('--touch-layout-policy',choices=('open','legacy_two_heads'),default='open')
    p.add_argument('--intensity-curve-mode',choices=('soft','follow','hard','strict'),default='follow')
    p.add_argument('--search-policy',choices=('legacy','phrase_v1'),default='phrase_v1')
    p.add_argument('--phrase-candidates',type=int,choices=(1,2,4),default=1)
    p.add_argument('--phrase-write-beats',type=float,default=8.)
    p.add_argument('--phrase-commit-beats',type=float,default=4.)
    p.add_argument('--phrase-step-budget',type=int,default=256)
    p.add_argument('--phrase-repair-budget',type=int,default=8)
    p.add_argument('--phrase-rollback-budget',type=int,default=4)
    p.add_argument('--phrase-time-budget-sec',type=float,default=30.)
    p.add_argument('--softcomfort-strength',type=float,default=1.)
    p.add_argument('--audio-tracking-strength',type=float,default=0.)
    p.add_argument('--physical-policy',choices=('legacy','runtime_hard_v2'),default='runtime_hard_v2')
    return p


def resolve_temperatures(args):
    """Explicit stage temperatures override the common sampling temperature."""
    temperature=float(getattr(args,'temperature',.65))
    for key in ('temperature','pointer_temperature','action_temperature'):
        raw=temperature if key=='temperature' else getattr(args,key,None)
        value=temperature if raw is None else float(raw)
        if not math.isfinite(value) or value<0:
            raise ValueError(f'{key} must be finite and nonnegative')
        setattr(args,key,value)
    return args

def _sample(logits,temperature,generator):
    if not bool(torch.isfinite(logits).any()):raise ValueError('No legal stream decision')
    if temperature<=0:return int(logits.argmax())
    return int(torch.multinomial((logits.float()/temperature).softmax(-1),1,generator=generator))


def record_sampling_trace(observation,trace):
    """Bounded CPU-only evidence for how legal sampling changes proposals."""
    if not trace:return
    counts=observation.setdefault('counts',Counter())
    counts['decisions']+=1
    counts['changed_time']+=bool(trace.get('changed_time'))
    counts['changed_configuration']+=bool(trace.get('configuration_changed'))
    first,last=trace.get('first_proposal'),trace.get('final_proposal')
    if first:observation.setdefault('first_family_counts',Counter())[str(first.get('family'))]+=1
    if last:observation.setdefault('accepted_family_counts',Counter())[str(last.get('family'))]+=1
    if first and last:counts['changed_family']+=first.get('family')!=last.get('family')
    counts['failed_decisions']+=last is None
    for onset in trace.get('onsets',()):
        counts['onsets_tried']+=1
        for trial in onset.get('trials',()):
            counts['proposals']+=1
            if not trial.get('accepted'):
                counts['rejected_proposals']+=1
                observation.setdefault('rejection_reasons',Counter())[trial.get('rejection_reason') or 'unknown']+=1
    examples=observation.setdefault('examples',[])
    if (trace.get('changed_time') or trace.get('configuration_changed') or last is None) and len(examples)<12:
        examples.append(trace)


def physical_policy_report(result,guard):
    """Keep native findings visible while reporting the actual execution policy."""
    report=result.to_dict()
    def details(issues):
        rows=[]
        for issue in issues:
            row=issue.to_dict()
            row['engine_action']=row.get('action')
            row['action']='enforce' if guard.enforced(issue) else 'report'
            row['source']='native_runtime' if issue.evidence.startswith('miacode_runtime:') else 'static_reference'
            rows.append(row)
        return rows
    report['details']=details(result.conflicts_detected)
    report['remaining_details']=details(result.remaining_conflicts)
    report['remaining_actionable_hard']=sum(guard.enforced(issue) for issue in result.remaining_conflicts)
    report['remaining_report_only_hard']=sum(issue.severity=='hard' and not guard.enforced(issue)
                                            for issue in result.remaining_conflicts)
    report['physical_policy']=guard.physical_policy
    report['conflict_kind_modes']=dict(guard.config.conflict_kind_modes or {})
    report['scope']='native_findings_preserved_actions_follow_actual_generation_guard'
    return report

def _condition(args,payload):
    if getattr(args,'condition_file',None):return GenerationCondition.from_json(args.condition_file)
    d=payload.get('condition_defaults')
    c=GenerationCondition.from_dict(d) if d else GenerationCondition(level=args.level,
        radar={k:GENERATION_DEFAULTS[k] for k in ('note_density','peak','sweep','slide_tricky','slide_sequence','jack','chord')},
        style={k:GENERATION_DEFAULTS[k] for k in ('tap_bias','slide_bias','hold_bias','touch_bias','chord_bias','melody_following','pattern_reuse','directionality')})
    c.level=float(args.level)
    return c

class _Curve:
    """User edits affect real event odds and completed head budgets, not labels."""
    def __init__(self,args,song,model,cache,condition):
        from .curve import apply_curve_to_composer
        self.seconds=cache['seconds'][0].detach().cpu().numpy();self.counts=Counter()
        control=condition.to_vector(device=cache['local'].device)[None]
        logits=model.plan_head(cache['local'])[0]+F.linear(model.condition(control),model.plan_head.weight)[0]
        values=F.softplus(logits[:,1:]).float()
        widths=np.diff(np.r_[self.seconds,float(song.audio.duration_sec)])
        exposure=torch.as_tensor(widths,device=values.device).clamp_min(1e-5)[:,None]
        rates=values if model.config.plan_target_units=='rates_per_second_v2' else values/exposure
        counts=rates*exposure
        baseline=float((counts[:,0].sum()/float(song.audio.duration_sec)).clamp_min(.1))
        self.controller=apply_curve_to_composer(getattr(args,'intensity_curve_file',None),
            duration_sec=float(song.audio.duration_sec),effective_start_sec=0.,effective_end_sec=float(song.audio.duration_sec),
            mode=getattr(args,'intensity_curve_mode','follow'),span_mode='full',
            strength=getattr(args,'intensity_curve_strength',.65),baseline_head_rate=baseline)
        self.active=self.controller.active and not self.controller._neutral()
        # Input order is event/head, while plan labels are blank/head/event.
        native=rates[:,[1,0]]
        adjusted=self.controller._adjust_rates(native,self.seconds) if self.active else native
        self.ratio=(adjusted[:,0].clamp_min(1e-6)/native[:,0].clamp_min(1e-6)).log().clamp(-4,4).cpu().numpy()
        self.head_rate=adjusted[:,1].detach().cpu().numpy()
        self.duration=float(song.audio.duration_sec)
        self.head_counts=counts[:,0].detach().cpu().numpy()
    def bias(self,seconds):return np.interp(seconds,self.seconds,self.ratio) if self.active else np.zeros_like(seconds,dtype=float)
    def allows(self,second,heads):
        if not self.active:return True
        if self.controller.is_in_rest_region(second):return False
        if self.controller.mode not in ('hard','strict'):return True
        region=int(second//2);a=region*2;b=min(self.duration,a+2)
        times=np.r_[a,self.seconds[(self.seconds>a)&(self.seconds<b)],b]
        rate=np.interp(times,self.seconds,self.head_rate)
        limit=int(math.ceil(float(np.trapezoid(rate,times))))
        return self.counts[region]+heads<=limit
    def commit(self,second,heads):self.counts[int(second//2)]+=heads


def replay_stream_prefix(prefix,context_factory,guard_factory,base_curve,clock):
    """Rebuild every prefix-dependent state without sharing branch mutations."""
    from copy import copy
    context=context_factory();guard=guard_factory();curve=copy(base_curve);curve.counts=Counter()
    points=[]
    for step in prefix:
        context.commit(step)
        if step.kind==NOTE:
            point=EventPoint(step.end,step.text);points.append(point);guard.commit(point)
            curve.commit(second_at_beat(clock,step.end),len(event_metadata(step.text).heads))
    return context,guard,curve,points

@torch.no_grad()
def continuation_observation(steps):
    """Describe committed runs without a musical-quality label or rejection."""
    from .stream_continuation import ContinuationState,typed_event
    state=ContinuationState('pattern_runs_v2')
    names=('same_configuration_run','button_fast_run','exact_period2_run','motion_pattern_run',
           'template_run','type_set_run','touch_only_run','same_gap_run')
    result={name:dict(count=0) for name in names}
    notes=[];buttons={}
    def record(name,count,first,last,**extra):
        if count<=result[name]['count']:return
        result[name]=dict(count=int(count),start_event=first[0],end_event=last[0],
            start_beat=str(first[1].end),end_beat=str(last[1].end),
            start_second=float(first[1].end_second),end_second=float(last[1].end_second),
            elapsed_seconds=float(last[1].end_second-first[1].end_second),**extra)
    for step in steps:
        state.append(step)
        if step.kind!=NOTE:
            buttons={};continue
        notes.append((len(notes)+1,step));last=notes[-1]
        for name,count in (('same_configuration_run',state.exact_count),('exact_period2_run',state.period2_count),
            ('template_run',state.template_count),('type_set_run',state.type_count),('touch_only_run',state.touch_count),
            ('same_gap_run',state.gap_count)):
            if count:
                extra={}
                if name=='same_configuration_run':extra['configuration']=step.text
                if name=='same_gap_run':extra.update(gap_beats=str(step.end-step.start),elapsed_beats=float(state.gap_elapsed))
                record(name,count,notes[-count],last,**extra)
        if state.motion_count:
            record('motion_pattern_run',state.motion_count,notes[-state.motion_count-1],last,
                   count_unit='transitions',rotation_steps=list(state.motion_relation))
        members=typed_event(step.text)[1]
        buttons={key:(buttons[key][0]+1,buttons[key][1])
                 if key in buttons and step.end-step.start<=Fraction(1,2) else (1,last) for key in members}
        for key,(count,first) in buttons.items():record('button_fast_run',count,first,last,button=key,max_gap_beats='1/2')
    return dict(schema='composer_continuation_observation_v2',event_count=len(notes),
        scope='descriptive_runs_not_bad_chart_labels; actual_committed_WAIT_resets_runs',**result)


@torch.no_grad()
def generate(args,*,song_override=None,model_override=None,payload_override=None):
    resolve_temperatures(args)
    started=time.perf_counter();device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    torch.set_num_threads(2)
    model,payload=load(args.checkpoint,device) if model_override is None else (model_override,payload_override)
    model.eval();condition=_condition(args,payload)
    tracking=float(getattr(args,'audio_tracking_strength',0.))
    if not math.isfinite(tracking) or not 0<=tracking<=1:raise ValueError('audio_tracking_strength must be in [0,1]')
    if tracking>0 and getattr(model.config,'phrase_context_policy','off')=='off':
        raise ValueError('This checkpoint has no trained phrase context; audio_tracking_strength must be 0')
    if hasattr(model,'set_audio_tracking_strength'):model.set_audio_tracking_strength(tracking)
    elif tracking:raise ValueError('model does not support audio tracking')
    vocab=DeltaVocabulary.from_dict(payload['delta_vocab'])
    checkpoint_epoch=payload.get('epoch');optimizer_steps=payload.get('optimizer_steps')
    # Do not retain mapped checkpoint tensors while a trainer replaces its file.
    if model_override is None:
        payload.pop('model_state_dict',None);payload.pop('optimizer_state_dict',None);payload.pop('training_state',None)
    if song_override is None:
        audio=extract_rational_audio(args.audio,cache_dir=Path(args.checkpoint).parent/'audio_cache',evidence_profile='balanced_evidence_v1')
        tempo=normalize_tempo_map(json.loads(Path(args.tempo_map).read_text(encoding='utf-8-sig'))) if getattr(args,'tempo_map',None) else [EventTempo(Fraction(0),args.bpm or audio.tempo_bpm)]
        meters=normalize_meter_map(json.loads(Path(args.meter_map).read_text(encoding='utf-8-sig'))) if getattr(args,'meter_map',None) else None
        song=organize_song(audio,tempo,args.first,meters=meters,max_seconds=999.,quality_profile='musical_evidence_v1')
    else:song=song_override
    clock=TempoClock(song.tempo,song.first);duration=float(song.audio.duration_sec);end=beat_at_second(clock,duration)
    context=StreamInferenceContext(song,model.codec,vocab,condition,lead_in=model.config.history_events,
        audio_grid_seconds=model.config.audio_grid_seconds,
        history_window_policy=model.config.history_window_policy,rollout_steps=model.config.rollout_steps,
        continuation_policy=model.config.continuation_policy,timing_output_policy=model.config.timing_output_policy,
        phrase_context_policy=getattr(model.config,'phrase_context_policy','off'))
    batch=to_device_batch(context.current_batch(),device)
    cache=model.encode_audio(batch['song_audio'],batch['song_mask'],batch['song_seconds'],batch['song_beats'])
    if model.config.plan_kind_policy!='off':
        cache=model.prepare_plan_cache(cache,batch['condition'],batch['song_inverse'],batch['audio_end_sec'])
    curve=_Curve(args,song,model,cache,condition)
    generator=torch.Generator(device=device).manual_seed(int(args.seed))
    guard=EventGuard(song.tempo,song.first,duration,min_same_head_sec=args.min_same_head_ms/1000.,
        touch_layout_policy=args.touch_layout_policy,intercept_tap_on_slide=True,
        physical_policy=getattr(args,'physical_policy','runtime_hard_v2'))
    points=[];cursor=Fraction(0);waits=0;retries=0;timing_retries=0;forced_waits=0;reason='budget'
    model_diagnostic_sums=None;model_diagnostic_count=0
    sampling_observation=dict(retry_policy='joint_event_rejection_v2',
        scope='all_trial_rollouts_including_rejected_candidates_not_only_exported_notes')
    diagnostic_fields=('phrase_residual_rms','phrase_ordered_anchor_residual_rms',
                       'phrase_ordered_reference_residual_rms','phrase_ordered_actor_residual_rms')
    def model_diagnostics():
        if model_diagnostic_sums is None:return None
        means=(model_diagnostic_sums/max(1,model_diagnostic_count)).cpu().tolist()
        return dict(forward_queries=model_diagnostic_count,mean_reference_weights=means[:5],
                    **dict(zip(diagnostic_fields,means[5:])))
    maximum=args.max_events or 100000
    deltas=[Fraction(x) for x in model.config.delta_beats]
    step_index=0
    def advance(stop_beat,step_budget,deadline=None):
        nonlocal points,cursor,waits,retries,timing_retries,forced_waits,reason,step_index,context,guard,curve,generator,model_diagnostic_sums,model_diagnostic_count
        for step_index in range(step_index,step_index+step_budget):
            if getattr(args,'search_policy','phrase_v1')=='legacy' and len(points)>=maximum:reason='max_events';break
            if cursor>=stop_beat:reason='audio_end' if getattr(args,'search_policy','phrase_v1')=='legacy' else 'span_complete';break
            if deadline is not None and time.perf_counter()>=deadline:reason='time_budget';break
            if len(points)>=maximum:reason='max_events';break
            if cursor>=end:reason='audio_end';break
            batch=to_device_batch({k:v for k,v in context.current_batch().items()
                if k not in ('song_audio','song_mask','song_seconds','song_beats')},device)
            result=model(batch,audio_cache=cache)
            if 'phrase_reference_weights' in result:
                reference=result['phrase_reference_weights'][0,-1].detach().float()
                values=torch.cat((reference,*[(result[field][0,-1].detach().float().reshape(1)
                    if field in result else reference.new_zeros(1)) for field in diagnostic_fields]))
                model_diagnostic_sums=(values.clone() if model_diagnostic_sums is None else model_diagnostic_sums+values)
                model_diagnostic_count+=1
            kinds=kind_logits_for_audio(result['kind_logits'][0,-1],cursor,end)
            kinds[NOTE]+=float(curve.bias(np.asarray([second_at_beat(clock,cursor)]))[0])
            kind=_sample(kinds,args.pointer_temperature,generator)
            if kind==EOS:
                reason='learned_eos'
                if getattr(args,'search_policy','phrase_v1')!='legacy':cursor=end
                break
            if kind==WAIT:
                nxt=cursor+2
                context.commit(StreamStep(WAIT,cursor,nxt,max(0.,second_at_beat(clock,cursor)),second_at_beat(clock,nxt)))
                cursor=nxt;waits+=1;continue
            deltas=current_timing_deltas(model,result,cursor)
            timing=note_delta_logits_for_audio(result['delta_logits'][0,-1],deltas,cursor,end,
                clock,duration,first_step=step_index==0)
            seconds=np.asarray([second_at_beat(clock,cursor+d) for d in deltas])
            if curve.active:
                timing+=torch.as_tensor(curve.bias(seconds),device=device)
                timing=timing.masked_fill(torch.as_tensor(curve.controller.rest_mask(seconds),device=device),-torch.inf)
            if not bool(torch.isfinite(timing).any()):
                if end-cursor<=2:
                    reason='tail_no_legal_onset'
                    if getattr(args,'search_policy','phrase_v1')!='legacy':cursor=end
                    break
                kind=WAIT;nxt=cursor+2;forced_waits+=1
            else:
                def at_time(delta_id,beat):
                    actual_seconds=torch.full_like(batch['step_second'],second_at_beat(clock,beat))
                    ids=torch.full_like(batch['history_delta'],delta_id)
                    if hasattr(model,'lane_load_projection'):
                        ctx,load=model.action_context_for_delta(result,batch,ids,cache,event_seconds=actual_seconds,return_button_load=True)
                        proposal=dict(context=ctx[0,-1],button_load=load[0,-1])
                    else:
                        proposal=dict(context=model.action_context_for_delta(result,batch,ids,cache,event_seconds=actual_seconds)[0,-1])
                    duration_context=(model.duration_context_at(result,batch,actual_seconds,cache)
                                      if hasattr(model,'duration_context_at') else None)
                    if duration_context is not None:
                        from .stream_duration import select_duration_context
                        proposal['duration_context']=select_duration_context(duration_context,
                            torch.tensor([batch['step_second'].numel()-1],device=device))
                    if model.config.event_family_policy in ('audio_semantic_v2','audio_prior_transition_v3','audio_state_phrase_v4'):
                        families=model.family_context_at(cache,actual_seconds,batch['song_inverse'],batch,**({'phrase_output':result} if model.config.event_family_policy=='audio_state_phrase_v4' else {}))
                        proposal['family_context']=families['full'][0,-1]
                        if 'full_logits' in families:proposal['family_logits']=families['full_logits'][0,-1]
                    if getattr(model.config,'transition_policy','off') in ('music_exit_v1','music_exit_v2','music_exit_v3'):
                        transition=model.transition_logits_at(cache,actual_seconds,batch['song_inverse'],batch)[0,-1]
                        proposal['motion_keep_logit']=transition[1]
                        proposal['motion_reference_shape']=result['motion_reference_shape'][0,-1]
                        proposal['template_keep_logit']=transition[2]
                        proposal['template_reference_shape']=torch.where(batch['history_kind'][0,-1].eq(NOTE),
                            batch['history_shape'][0,-1],batch['history_shape'].new_tensor(-1))
                    return proposal
                accepted,delta_id,failed,failed_times=sample_timed_event(model,timing,deltas,cursor,at_time,
                    int(result['reference_phase'][0,-1]),result.get('reference_phase_mask',None)[0,-1]
                        if 'reference_phase_mask' in result else None,guard,generator,
                    pointer_temperature=args.pointer_temperature,action_temperature=args.action_temperature,
                    timing_grid_ids=result.get('timing_grid_ids'),
                    timing_group_count=result.get('timing_group_count'),
                    timing_keep_mask=result.get('timing_keep_mask'),
                    repeat_delta_id=int(result['repeat_delta_id'][0,-1]) if 'repeat_delta_id' in result else -1,
                    motion_phase_distribution=result['motion_phase_distribution'][0,-1]
                        if 'motion_phase_distribution' in result else None,
                    accepts_extra=lambda pt:curve.allows(
                        second_at_beat(clock,pt.beat),len(event_metadata(pt.note_text).heads)))
                retries+=failed;timing_retries+=failed_times
                record_sampling_trace(sampling_observation,getattr(guard,'last_timed_sampling_trace',None))
                if accepted is not None:
                    nxt=accepted.beat
                    heads=len(event_metadata(accepted.note_text).heads)
                    points.append(accepted);curve.commit(second_at_beat(clock,nxt),heads)
                    context.commit(StreamStep(NOTE,cursor,nxt,max(0.,second_at_beat(clock,cursor)),second_at_beat(clock,nxt),accepted.note_text))
                    cursor=nxt
                    if len(points)%100==0:print('composer_generation_progress '+json.dumps(dict(events=len(points),beat=float(cursor),duration_sec=duration)),flush=True)
                    continue
                if end-cursor<=2 and getattr(args,'search_policy','phrase_v1')=='legacy':reason='tail_rejected_configurations';break
                if getattr(args,'search_policy','phrase_v1')!='legacy':
                    reason='physical_dead_end';break
                kind=WAIT;nxt=cursor+2;forced_waits+=1
            context.commit(StreamStep(WAIT,cursor,nxt,max(0.,second_at_beat(clock,cursor)),second_at_beat(clock,nxt)))
            cursor=nxt;waits+=1
    search_report={'policy':'legacy','candidate_count':1}
    if getattr(args,'search_policy','phrase_v1')=='legacy':
        advance(end,maximum+max(100,math.ceil(float(end)*2)))
    else:
        from copy import copy
        from .stream_search import search_stream
        base_curve=curve
        audio_bank=context.bank
        def rollout(prefix,seed,stop_beat,deadline=None):
            nonlocal points,cursor,waits,retries,timing_retries,forced_waits,reason,step_index,context,guard,curve,generator,model_diagnostic_sums,model_diagnostic_count
            def context_factory():
                return StreamInferenceContext(song,model.codec,vocab,condition,lead_in=model.config.history_events,
                    audio_bank=audio_bank,audio_grid_seconds=model.config.audio_grid_seconds,
                    history_window_policy=model.config.history_window_policy,rollout_steps=model.config.rollout_steps,
                    continuation_policy=model.config.continuation_policy,timing_output_policy=model.config.timing_output_policy,
                    phrase_context_policy=getattr(model.config,'phrase_context_policy','off'))
            def guard_factory():
                return EventGuard(song.tempo,song.first,duration,min_same_head_sec=args.min_same_head_ms/1000.,
                    touch_layout_policy=args.touch_layout_policy,intercept_tap_on_slide=True,
                    physical_policy=getattr(args,'physical_policy','runtime_hard_v2'))
            context,guard,curve,points=replay_stream_prefix(prefix,context_factory,guard_factory,base_curve,clock)
            cursor=prefix[-1].end if prefix else Fraction(0)
            anchor_second=second_at_beat(clock,cursor)
            waits=sum(s.kind==WAIT for s in prefix);retries=timing_retries=forced_waits=0
            step_index=len(prefix);reason='step_budget'
            model_diagnostic_sums=None;model_diagnostic_count=0
            generator=torch.Generator(device=device).manual_seed(seed)
            advance(stop_beat,int(getattr(args,'phrase_step_budget',256)),deadline)
            return dict(steps=list(context._committed),points=list(points),cursor=cursor,reason=reason,
                retries=retries,timing_retries=timing_retries,forced_waits=forced_waits,guard=guard,
                anchor_second=anchor_second,model_diagnostics=model_diagnostics())
        try:
            selected,search_report=search_stream(rollout,song,condition,args,end,clock,base_curve,audio_bank=audio_bank,codec=model.codec)
        except ValueError as exc:
            if getattr(exc,'diagnostics',None) is not None:
                exc.diagnostics.update(checkpoint_epoch=checkpoint_epoch,optimizer_steps=optimizer_steps)
            raise
        rollout(selected,int(args.seed),selected[-1].end if selected else Fraction(0))
        reason=search_report['stop_reason']
        totals=search_report['search_totals']
        retries=totals['configuration_retries'];timing_retries=totals['rejected_onset_attempts'];forced_waits=totals['forced_waits']
        guard.checks=totals['guard_checks'];guard.rejections=totals['guard_rejections'];guard.reason_counts=Counter(totals['guard_reasons'])
        search_report['retry_counter_scope']='all_searched_candidates'
    text,_=compile_event_points(points,tempo_changes=song.tempo,first=song.first,
        end_beat=Fraction(math.ceil(float(end)*4),4),level=str(condition.level),difficulty=condition.difficulty,strict_notes=True)
    physical=process_chart_points(points,song.tempo,song.first,guard.config)
    if getattr(args,'search_policy','phrase_v1')!='legacy':
        if physical.coverage.get('parse_errors') or any(guard.enforced(x) for x in physical.conflicts_detected):
            raise ValueError('final physical validation failed; chart not exported')
        search_report['final_physical_gate']='passed'
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(text,encoding='utf-8')
    from .configuration_observation import configuration_observation
    from .structure_metrics import structure_metrics
    from .stream_continuation import continuation_dim
    report=dict(schema='maichart-chart-composer-generation-v2',family='chart_composer_v1',architecture_profile=PROFILE,
        checkpoint=str(Path(args.checkpoint).resolve()),checkpoint_epoch=checkpoint_epoch,optimizer_steps=optimizer_steps,
        history_window_policy=model.config.history_window_policy,continuation_policy=model.config.continuation_policy,
        timing_output_policy=model.config.timing_output_policy,
        timing_group_policy=model.config.timing_group_policy,
        transition_policy=getattr(model.config,'transition_policy','off'),
        timing_phase_count=len(model.timing_grid.phase_catalog) if hasattr(model,'timing_grid') else 0,
        event_family_policy=model.config.event_family_policy,
        event_family_audio_weight=getattr(model.config,'event_family_audio_weight',0.),
        timing_grid_audio_policy=getattr(model.config,'timing_grid_audio_policy','off'),
        configuration_retry_semantics='joint_event_rejection_v2',sampling_observation=sampling_observation,
        training_objective_recipe=(payload.get('args') or {}).get('training_objective_recipe','legacy_joint_event_objective'),
        gap_continuation_policy=model.config.gap_continuation_policy,sensor_load_policy=model.config.sensor_load_policy,
        lane_load_policy=model.config.lane_load_policy,
        motion_continuation_policy=model.config.motion_continuation_policy,
        recovery_policy=getattr(model.config,'recovery_policy','persistent_v1'),
        prefix_augmentation_rate=getattr(model.config,'prefix_augmentation_rate',0.),
        training_prefix_input_policy=(payload.get('args') or {}).get('prefix_augmentation_input_policy','off'),
        continuation_feature_dim=continuation_dim(model.config.continuation_policy),
        transition_exit_weight=getattr(model.config,'transition_exit_weight',0.),
        history_events=model.config.history_events,rollout_steps=model.config.rollout_steps,
        event_count=len(points),seed=args.seed,condition=condition.to_dict(),stop_reason=reason,device=device,
        temperature=args.temperature,pointer_temperature=args.pointer_temperature,action_temperature=args.action_temperature,
        stream_steps=(step_index+1 if search_report['policy']=='legacy' else len(context._committed)),wait_steps=waits,forced_waits=forced_waits,configuration_retries=retries,
        rejected_onset_attempts=timing_retries,
        audio_duration_sec=duration,last_head_sec=second_at_beat(clock,points[-1].beat) if points else None,
        elapsed_sec=time.perf_counter()-started,codec_coverage=model.codec.coverage,
        guard=dict(checks=guard.checks,rejections=guard.rejections,reasons=dict(guard.reason_counts)),
        configuration_observation=configuration_observation(points),
        structure_metrics=structure_metrics(points),
        continuation_observation=continuation_observation(context._committed),
        phrase_search=search_report,audio_tracking_strength=tracking,
        phrase_context_policy=getattr(model.config,'phrase_context_policy','off'),
        duration_audio_policy=getattr(model.config,'duration_audio_policy','off'),
        duration_audio_coverage=(model.duration_audio.coverage() if hasattr(model,'duration_audio') else None),
        physical_policy=guard.physical_policy,
        curve=dict(active=curve.active,mode=args.intensity_curve_mode,strength=args.intensity_curve_strength),
        playability=physical_policy_report(physical,guard),
        quality_scope='generated_stream_requires_listening_and_playtest')
    output.with_suffix('.composer.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report

def evaluate_panel(checkpoint_path,dataset,directory,*,count=2,seeds=(0,1),device='cpu',temperature=.65,generation_settings=None):
    from .stream_evaluation_settings import validate
    settings=validate(generation_settings or {})
    directory=Path(directory);reports=[];model,payload=load(checkpoint_path,device)
    selected=[];seen=set()
    for i,record in enumerate(dataset.base.songs):
        audio=record[0].audio
        identity=str(getattr(audio,'path',None) or getattr(audio,'source',None) or ('object',id(audio)))
        if identity in seen:continue
        seen.add(identity);selected.append(i)
        if len(selected)>=count:break
    if count<=0:selected=[]
    for i in selected:
        record=dataset.base.songs[i];song=record[0]
        for seed in seeds:
            arguments=['--audio','unused','--checkpoint',str(checkpoint_path),
                '--output',str(directory/f'song_{i}_seed_{seed}'/'maidata.txt'),'--device',str(device),'--seed',str(seed),
                '--temperature',str(temperature)]
            for name,value in settings.items():
                if value is not None:arguments.extend(('--'+name.replace('_','-'),str(value)))
            args=resolve_temperatures(parser().parse_args(arguments))
            condition=record[5];args.level=float(condition.level)
            output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
            condfile=output.parent/'condition.json';condfile.write_text(json.dumps(condition.to_dict()),encoding='utf-8');args.condition_file=str(condfile)
            try:
                reports.append(generate(args,song_override=song,model_override=model,payload_override=payload))
            except ValueError as exc:
                # An invalid validation artifact is a failed quality check,
                # not a reason to discard the remaining training epochs.
                failure=dict(status='failed',song_index=i,seed=seed,error=str(exc),output=str(output))
                (output.parent/'generation_failure.json').write_text(
                    json.dumps(failure,ensure_ascii=False,indent=2),encoding='utf-8')
                reports.append(failure)
    directory.mkdir(parents=True,exist_ok=True);path=directory/'summary.json'
    failed=sum(run.get('status')=='failed' for run in reports)
    result=dict(status='completed_with_failures' if failed else 'completed',failed_runs=failed,
        report_path=str(path),runs=reports,condition_scope='reference-derived controls',
        temperature=float(temperature),
        pointer_temperature=float(temperature if settings['pointer_temperature'] is None else settings['pointer_temperature']),
        action_temperature=float(temperature if settings['action_temperature'] is None else settings['action_temperature']),
        generation_settings=settings,selected_song_indices=selected,
        selection_scope='first_distinct_audio_sources_with_reference_conditions_no_external_intensity_curve')
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result

def main(argv=None):
    args=parser().parse_args(argv)
    try:report=generate(args)
    except ValueError as exc:
        diagnostics=getattr(exc,'diagnostics',None)
        if diagnostics is not None:
            path=Path(args.output).with_suffix('.failure.json')
            write_error=None
            try:
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(json.dumps(dict(status='failed',error=str(exc),
                    checkpoint=str(args.checkpoint),seed=args.seed,settings=vars(args),
                    diagnostics=diagnostics),ensure_ascii=False,indent=2),encoding='utf-8')
            except OSError as error:write_error=str(error)
            print('composer_generation_failure '+json.dumps(dict(error=str(exc),report_path=str(path),
                report_write_error=write_error,
                beat=diagnostics.get('start_beat'),deadline_exhausted=diagnostics.get('deadline_exhausted'),
                repairs_used=diagnostics.get('repairs_used'),rollbacks_used=diagnostics.get('rollbacks_used')),
                ensure_ascii=False),flush=True)
        raise
    print(json.dumps(report,ensure_ascii=False));return 0

if __name__=='__main__':raise SystemExit(main())
