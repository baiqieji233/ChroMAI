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
from maichart.generation.post_process import PostProcessConfig,process_chart_points
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
        continuation_policy=model.config.continuation_policy,timing_output_policy=model.config.timing_output_policy)
    batch=to_device_batch(context.current_batch(),device)
    cache=model.encode_audio(batch['song_audio'],batch['song_mask'],batch['song_seconds'],batch['song_beats'])
    if model.config.plan_kind_policy!='off':
        cache=model.prepare_plan_cache(cache,batch['condition'],batch['song_inverse'],batch['audio_end_sec'])
    curve=_Curve(args,song,model,cache,condition)
    generator=torch.Generator(device=device).manual_seed(int(args.seed))
    guard=EventGuard(song.tempo,song.first,duration,min_same_head_sec=args.min_same_head_ms/1000.,
        touch_layout_policy=args.touch_layout_policy,intercept_tap_on_slide=True)
    points=[];cursor=Fraction(0);waits=0;retries=0;timing_retries=0;forced_waits=0;reason='budget'
    maximum=args.max_events or 100000
    deltas=[Fraction(x) for x in model.config.delta_beats]
    for step_index in range(maximum+max(100,math.ceil(float(end)*2))):
        if len(points)>=maximum:reason='max_events';break
        if cursor>=end:reason='audio_end';break
        batch=to_device_batch({k:v for k,v in context.current_batch().items()
            if k not in ('song_audio','song_mask','song_seconds','song_beats')},device)
        result=model(batch,audio_cache=cache)
        kinds=kind_logits_for_audio(result['kind_logits'][0,-1],cursor,end)
        kinds[NOTE]+=float(curve.bias(np.asarray([second_at_beat(clock,cursor)]))[0])
        kind=_sample(kinds,args.pointer_temperature,generator)
        if kind==EOS:reason='learned_eos';break
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
            if end-cursor<=2:reason='tail_no_legal_onset';break
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
                if model.config.event_family_policy in ('audio_semantic_v2','audio_prior_transition_v3'):
                    families=model.family_context_at(cache,actual_seconds,batch['song_inverse'],batch)
                    proposal['family_context']=families['full'][0,-1]
                    if 'full_logits' in families:proposal['family_logits']=families['full_logits'][0,-1]
                return proposal
            accepted,delta_id,failed,failed_times=sample_timed_event(model,timing,deltas,cursor,at_time,
                int(result['reference_phase'][0,-1]),result.get('reference_phase_mask',None)[0,-1]
                    if 'reference_phase_mask' in result else None,guard,generator,
                pointer_temperature=args.pointer_temperature,action_temperature=args.action_temperature,
                timing_grid_ids=result.get('timing_grid_ids'),
                timing_group_count=result.get('timing_group_count'),
                repeat_delta_id=int(result['repeat_delta_id'][0,-1]) if 'repeat_delta_id' in result else -1,
                motion_phase_distribution=result['motion_phase_distribution'][0,-1]
                    if 'motion_phase_distribution' in result else None,
                accepts_extra=lambda pt:curve.allows(
                    second_at_beat(clock,pt.beat),len(event_metadata(pt.note_text).heads)))
            retries+=failed;timing_retries+=failed_times
            if accepted is not None:
                nxt=accepted.beat
                heads=len(event_metadata(accepted.note_text).heads)
                points.append(accepted);curve.commit(second_at_beat(clock,nxt),heads)
                context.commit(StreamStep(NOTE,cursor,nxt,max(0.,second_at_beat(clock,cursor)),second_at_beat(clock,nxt),accepted.note_text))
                cursor=nxt
                if len(points)%100==0:print('composer_generation_progress '+json.dumps(dict(events=len(points),beat=float(cursor),duration_sec=duration)),flush=True)
                continue
            if end-cursor<=2:reason='tail_rejected_configurations';break
            kind=WAIT;nxt=cursor+2;forced_waits+=1
        context.commit(StreamStep(WAIT,cursor,nxt,max(0.,second_at_beat(clock,cursor)),second_at_beat(clock,nxt)))
        cursor=nxt;waits+=1
    text,_=compile_event_points(points,tempo_changes=song.tempo,first=song.first,
        end_beat=Fraction(math.ceil(float(end)*4),4),level=str(condition.level),difficulty=condition.difficulty,strict_notes=True)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(text,encoding='utf-8')
    physical=process_chart_points(points,song.tempo,song.first,PostProcessConfig(enable_conflict_detection=True,
        conflict_engine='miacode',conflict_repair_mode='none',detect_hard_conflicts=True,detect_soft_conflicts=False))
    from .configuration_observation import configuration_observation
    from .structure_metrics import structure_metrics
    from .stream_continuation import continuation_dim
    report=dict(schema='maichart-chart-composer-generation-v2',family='chart_composer_v1',architecture_profile=PROFILE,
        checkpoint=str(Path(args.checkpoint).resolve()),checkpoint_epoch=checkpoint_epoch,optimizer_steps=optimizer_steps,
        history_window_policy=model.config.history_window_policy,continuation_policy=model.config.continuation_policy,
        timing_output_policy=model.config.timing_output_policy,
        timing_group_policy=model.config.timing_group_policy,
        timing_phase_count=len(model.timing_grid.phase_catalog) if hasattr(model,'timing_grid') else 0,
        event_family_policy=model.config.event_family_policy,
        event_family_audio_weight=getattr(model.config,'event_family_audio_weight',0.),
        timing_grid_audio_policy=getattr(model.config,'timing_grid_audio_policy','off'),
        configuration_retry_semantics=('preserve_sampled_family' if model.config.event_family_policy in ('audio_semantic_v2','audio_prior_transition_v3') else 'resample_full_event'),
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
        stream_steps=step_index+1,wait_steps=waits,forced_waits=forced_waits,configuration_retries=retries,
        rejected_onset_attempts=timing_retries,
        audio_duration_sec=duration,last_head_sec=second_at_beat(clock,points[-1].beat) if points else None,
        elapsed_sec=time.perf_counter()-started,codec_coverage=model.codec.coverage,
        guard=dict(checks=guard.checks,rejections=guard.rejections,reasons=dict(guard.reason_counts)),
        configuration_observation=configuration_observation(points),
        structure_metrics=structure_metrics(points),
        continuation_observation=continuation_observation(context._committed),
        curve=dict(active=curve.active,mode=args.intensity_curve_mode,strength=args.intensity_curve_strength),
        playability=physical.to_dict(),
        quality_scope='generated_stream_requires_listening_and_playtest')
    output.with_suffix('.composer.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report

def evaluate_panel(checkpoint_path,dataset,directory,*,count=2,seeds=(0,1),device='cpu',temperature=.65):
    directory=Path(directory);reports=[];model,payload=load(checkpoint_path,device)
    for i in range(min(count,len(dataset.base.songs))):
        record=dataset.base.songs[i];song=record[0]
        for seed in seeds:
            args=parser().parse_args(['--audio','unused','--checkpoint',str(checkpoint_path),
                '--output',str(directory/f'song_{i}_seed_{seed}'/'maidata.txt'),'--device',str(device),'--seed',str(seed),
                '--temperature',str(temperature),'--pointer-temperature',str(temperature),'--action-temperature',str(temperature)])
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
        temperature=float(temperature),pointer_temperature=float(temperature),action_temperature=float(temperature))
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result

def main(argv=None):
    args=parser().parse_args(argv);print(json.dumps(generate(args),ensure_ascii=False));return 0

if __name__=='__main__':raise SystemExit(main())
