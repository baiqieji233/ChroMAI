"""Continuous complete-event commit, bounded context, no timed output blocks."""
from fractions import Fraction
from copy import deepcopy
from pathlib import Path
import json
import math
import numpy as np
import torch
from .codec import render_atom
from .stream_data import StreamDataset,collate_stream
from .retry_sampling import RejectedEventTree,RetrySpaceExhausted
from .generate import WindowSampler,_sample,_atom_heads,_new_guard,_check_window,_comfort
from maichart.event_timing import EventPoint
from maichart.generation.event_compile import compile_event_points
from maichart.composer.event_set_codec import _atoms


class RefreshContext(ValueError):pass


class RecoveryBudget:
    """Bound backtracking at one frontier, not successful repairs over a song."""
    def __init__(self, limit):
        self.limit=limit;self.used=0;self.frontier=Fraction(-1)
    @property
    def available(self):return self.used<self.limit
    def spend(self):self.used+=1
    def commit(self, beat):
        if beat<=self.frontier:return False
        self.frontier=beat;self.used=0
        return True
    def report(self):
        return dict(recovery_budget=self.limit,recovery_budget_scope='forward_progress_v2',
            recovery_used_at_frontier=self.used,furthest_committed_beat=str(self.frontier))


class GrammarDeadEnd(ValueError):
    def __init__(self,diagnostic):
        super().__init__('Component grammar has no legal continuation')
        self.diagnostic=diagnostic


class EventSampler(WindowSampler):
    def __init__(self,model,dataset,song,points,device,rng,args):
        self.model=model;self.codec=model.codec;self.dataset=dataset;self.song=song
        self.device=device;self.generator=rng;self.temperature=args.temperature;self.top_p=args.top_p
        self.cursor=points[-1].beat if points else Fraction(0)
        item=dataset.build_stream(song,points,self.cursor)
        self.prefix=item['input_ids'].tolist();self.anchors=list(item['anchors'])
        self.anchor_ids={a:i for i,a in enumerate(self.anchors)}
        self.batch={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in collate_stream([item]).items()}
        self.max_tokens=model.config.max_tokens;self.cache=model.encode_audio(self.batch)
        self.retry_stats={};self._retry_scope=None
    def next(self,target,allowed):
        if len(target)>=self.max_tokens//2:raise RefreshContext('Component budget reached inside next event')
        if not allowed:
            scope=getattr(self,'proposal_scope',{})
            begin=scope.get('target_prefix_length',0)
            tokens=list(target[begin:])
            raise GrammarDeadEnd(dict(**scope,token_count=len(tokens),
                proposal_token_ids=tokens[-64:],proposal_tokens=[self.codec.tokens[i] for i in tokens[-64:]],
                allowed_token_count=0,context_cursor=str(self.cursor)))
        tree=getattr(self,'_retry_tree',None)
        relative=target[len(self._retry_base):] if tree is not None else None
        logits=tree.cached(relative) if tree is not None else None
        if logits is None:
            if getattr(self,'_retry_cache_dirty',False):self._rewind(self._retry_base)
            logits=self._logits(target);self._retry_cache_dirty=False
            self.retry_stats['decoder_queries']=self.retry_stats.get('decoder_queries',0)+1
        else:
            self._retry_cache_dirty=True
            self.retry_stats['decoder_cache_hits']=self.retry_stats.get('decoder_cache_hits',0)+1
        if tree is not None:
            logits=tree.remember(relative,getattr(self,'_free_choices',allowed),logits)
            if self._retry_condition:
                logits,allowed,scaled=tree.condition(relative,allowed,self.temperature)
                if scaled:
                    self.retry_stats['conditioned_draws']=self.retry_stats.get('conditioned_draws',0)+1
                    return _sample(logits,allowed,self.generator,1.,self.top_p)
        return _sample(logits,allowed,self.generator,self.temperature,self.top_p)
    def _logits(self,target):
        ids=self.prefix+list(target)
        clocks=self.codec.token_clocks(ids,cursor=self.cursor)
        missing=sorted({math.floor(t) for t in clocks}-self.anchor_ids.keys())
        if missing:
            tiles=[self.dataset.tile(self.song,a) for a in missing]
            query={**self.batch,**{k:torch.stack([t[k] for t in tiles]).to(self.device) for k in tiles[0]},
                'tile_owner':torch.zeros(len(missing),dtype=torch.long,device=self.device)}
            for key in ('bank_audio','bank_semantic','bank_semantic_mask','bank_owner','tile_frames'):query.pop(key,None)
            encoded=self.model.encode_audio(query)
            for key in ('local','local_mask','local_beats'):self.cache[key]=torch.cat((self.cache[key],encoded[key]),0)
            for a in missing:self.anchor_ids[a]=len(self.anchors);self.anchors.append(a)
        query={**self.batch,'input_ids':torch.tensor([ids],device=self.device),
            'token_beats':torch.tensor([clocks],dtype=torch.float32,device=self.device),
            'token_mask':torch.ones((1,len(ids)),dtype=torch.bool,device=self.device),
            'tile_ids_cpu':[[self.anchor_ids[math.floor(t)] for t in clocks]]}
        return self.model.decode(query,self.cache)[0,-1]
    def reject(self,ids):return self._retry_tree.reject(ids)
    def propose(self,target,after,end,fixed=None):
        self.proposal_scope=dict(after=str(after),audio_end_beat=str(end),target_prefix_length=len(target),
            fixed_beat=str(fixed.beat) if fixed is not None else None)
        scope=(tuple(target),after,end)
        if scope!=getattr(self,'_retry_scope',None):
            self._retry_tree=RejectedEventTree();self._retry_scope=scope
        self._retry_base=list(target);self._retry_cache_dirty=False;self._retry_condition=fixed is None
        self._rewind(target)
        if fixed is None:
            time=[]
            while not time or time[-1] not in self.codec.phases:
                allowed=self.codec.time_choices(time,after,end)
                if not time and not getattr(self,'retry_allow_eos',True) and self.codec.at_id in allowed:
                    allowed=tuple(t for t in allowed if t!=self.codec.eos_id)
                self._free_choices=allowed
                token=self.next(target+time,allowed);time.append(token)
                if token==self.codec.eos_id:return time
        else:time=self.codec._time(fixed.beat)
        out=list(time);used=set();atom=[];count=0
        wanted=len(_atoms(fixed.note_text)) if fixed else None
        while True:
            if not atom:
                if wanted is not None and count==wanted:
                    out.append(self.codec.event_end_id);return out
                allowed=tuple(self.codec.kind_ids)+((self.codec.event_end_id,) if count and wanted is None else ())
                self._free_choices=tuple(self.codec.kind_ids)+((self.codec.event_end_id,) if count else ())
            else:
                allowed=self._head_choices(self.codec.allowed_atom_next(atom),atom,used);self._free_choices=allowed
            token=self.next(target+out,allowed);out.append(token)
            if token==self.codec.event_end_id:return out
            atom.append(token)
            if token==self.codec.end_note_id:
                used.update(_atom_heads(render_atom([self.codec.tokens[t] for t in atom])));atom=[];count+=1


def restore_prefix(points,tempo,clock,duration,failed_beat,conflict_sources=(),*,rejected=()):
    """Prefer the latest committed event explicitly involved in the dead end."""
    from maichart.generation.sustain_state import SustainStateIndex
    confirmed=set(map(Fraction,conflict_sources)) & {p.beat for p in points}
    if confirmed:source=max(confirmed);reason='latest_native_conflicting_event'
    else:
        state=SustainStateIndex(points,tempo,clock.seconds[0]);second=clock.at(failed_beat)
        sources=[Fraction(m['source_beat']) for m in state._candidate_markers(second)
            if m['source_beat']<=failed_beat and float(m['end_second'])>second
            and m.get('type') in ('hold','touch_hold','slide','wifi')]
        if not sources:return None
        source=min(sources);reason='active_sustained_event'
    # Native reports identify a judged note, not every contributing action.
    # Before revoking that old source, see whether removing a later event alone
    # resolves an actual failed continuation while retaining the other events.
    # The bounded suffix probes share one prefix replay, never re-run the model.
    lower=max(next(i for i,p in enumerate(points) if p.beat==source)+1,len(points)-8)
    witnesses=[];seen=set()
    for point,details in rejected:
        key=(point.beat,point.note_text)
        if details and point.beat>points[-1].beat and key not in seen:
            witnesses.append(point);seen.add(key)
        if len(witnesses)==4:break
    if lower<len(points) and witnesses:
        base=_new_guard(tempo,clock.seconds[0],duration);states={}
        for i,point in enumerate(points):
            if i>=lower:states[i]=deepcopy(base)
            if not base.accepts(point):raise RuntimeError('Previously accepted prefix failed rollback probe replay')
            base.commit(point)
        checks=0
        for i in range(len(points)-1,lower-1,-1):
            for witness in witnesses:
                checks+=1
                trial,_=_check_window(states[i],[*points[i+1:],witness])
                if trial is not None:
                    return points[:i],states[i],dict(source_beat=str(points[i].beat),
                        revoked_events=len(points)-i,reason='latest_verified_conflict_contributor',
                        reported_source_beats=[str(b) for b in sorted(confirmed)],
                        witness=dict(beat=str(witness.beat),note_text=witness.note_text),
                        native_counterfactual_checks=checks,
                        removed_points=[dict(beat=str(p.beat),note_text=p.note_text) for p in points[i:]])
    kept=[p for p in points if p.beat<source]
    if len(kept)==len(points):return None
    guard=_new_guard(tempo,clock.seconds[0],duration)
    for point in kept:
        if not guard.accepts(point):raise RuntimeError('Previously accepted prefix failed exact replay')
        guard.commit(point)
    return kept,guard,dict(source_beat=str(source),revoked_events=len(points)-len(kept),reason=reason,
        removed_points=[dict(beat=str(p.beat),note_text=p.note_text) for p in points[len(kept):]])


def generate_continuous(model,audio,audio_path,tempo,first,timing_report,condition,known,condition_report,args,rng,device,end,*,checkpoint_info=None):
    from .data import ComposerV2Song,_beats_at_seconds
    from maichart.tempo_clock import TempoClock
    config=model.config;clock=TempoClock(tempo,first)
    from .exhaustion import resolve_policy,admit_physical_conflict,conflict_assessment
    exhaustion_policy=resolve_policy(getattr(args,'exhaustion_policy','auto'),config.sequence_policy)
    song=ComposerV2Song(audio_path.stem,audio_path,audio_path,5,args.level,[],clock,end,condition,audio=audio,condition_known=known)
    song.summary=dict(audio.summary(160));song.summary['beats']=_beats_at_seconds(clock,song.summary['times_sec'])
    dataset=StreamDataset.__new__(StreamDataset)
    dataset.codec=model.codec;dataset.max_tokens=config.max_tokens;dataset.history_beats=config.history_beats
    dataset.window_beats=config.window_beats;dataset.future_beats=config.future_beats
    guard=_new_guard(tempo,first,audio.duration_sec)
    points=[];target=[];sampler=None;committed_in_context=0;refreshes=0;rollbacks=[];repairs=[];done=False
    duration_repairs=[];release_trials=set();bypassed_events=[]
    retry_stats={};native_memo={}
    recovery=RecoveryBudget(args.rollback_windows)
    hold_release_repair=args.rollback_windows>0
    attempts=1 if args.temperature==0 else args.window_retries
    candidate_count=1 if args.temperature==0 else args.candidates
    while not done:
        if sampler is None:
            sampler=EventSampler(model,dataset,song,points,device,rng,args);target=[];committed_in_context=0;refreshes+=1
            sampler.retry_stats=retry_stats
        after=points[-1].beat if points else Fraction(-1)
        selected=None;failed_beat=after;need_refresh=False;conflict_sources=set();rejected=[];history_edited=False
        prefix_key=tuple((p.beat,p.note_text) for p in points) if native_memo else None
        known_failed=native_memo.get(prefix_key,{}) if prefix_key is not None else {}
        failed_any=False
        for attempt in range(attempts):
            try:
                sampler.retry_allow_eos=not failed_any and not known_failed and recovery.used==0
                ids=sampler.propose(target,after,end)
                if ids==[model.codec.eos_id]:
                    if not sampler.retry_allow_eos and model.codec.at_id in model.codec.time_choices([],after,end):
                        repairs.append(dict(reason='eos_is_not_physical_recovery',after=str(after)));continue
                    done=True;break
                proposal=model.codec.decode_points(ids)[0];failed_beat=proposal.beat
                candidates=[(proposal,ids)]
                for _ in range(candidate_count-1):
                    alternative=sampler.propose(target,after,end,fixed=proposal)
                    candidates.append((model.codec.decode_points(alternative)[0],alternative))
                feasible=[]
                for candidate_index,(point,encoded) in enumerate(candidates):
                    event_key=(point.beat,point.note_text)
                    cached=event_key in known_failed
                    if cached:
                        trial=None;reason,details=known_failed[event_key]
                        retry_stats['native_cache_hits']=retry_stats.get('native_cache_hits',0)+1
                    else:
                        details=[];trial,reason=_check_window(guard,[point],details)
                        retry_stats['native_checks']=retry_stats.get('native_checks',0)+1
                    if trial is None:
                        failed_any=True
                        if prefix_key is None:prefix_key=tuple((p.beat,p.note_text) for p in points)
                        native_memo.setdefault(prefix_key,known_failed)
                        known_failed[event_key]=(reason,details)
                        if candidate_index==0 and hasattr(sampler,'reject'):sampler.reject(encoded)
                        conflict_sources.update(Fraction(b) for issue in details for b in issue['note_beats'])
                        if reason=='new_miacode_runtime_hard_conflict':rejected.append((point,details))
                        repairs.append(dict(beat=str(point.beat),note_text=point.note_text,reason=reason,
                            conflicts=details,cached_rejection=cached));continue
                    score=.5 if candidate_count==1 else _comfort(points,[point],tempo,clock,max(Fraction(0),after),min(end,point.beat+1),audio.duration_sec)[0]
                    feasible.append((score,point,encoded,trial))
                if feasible:selected=max(feasible,key=lambda row:row[0]);break
            except RetrySpaceExhausted:
                retry_stats['exhausted_subtrees']=retry_stats.get('exhausted_subtrees',0)+1;break
            except RefreshContext:
                if not committed_in_context:
                    repairs.append(dict(reason='event_component_budget_exhausted',after=str(after)))
                    continue
                need_refresh=True;break
            except ValueError as error:
                repairs.append(dict(reason=str(error),after=str(after),
                    **({'grammar':error.diagnostic} if isinstance(error,GrammarDeadEnd) else {})))
        if done:break
        if need_refresh:sampler=None;continue
        if selected is None:
            budget_left=recovery.available
            if points and rejected and budget_left and hold_release_repair:
                from .release_repair import repair_hold_conflict
                audit={}
                repaired=repair_hold_conflict(points,rejected,tempo,clock,audio.duration_sec,model.codec,
                    attempted=release_trials,audit=audit)
                if repaired:
                    points,witness,trial,report=repaired
                    recovery.spend()
                    report['recovery_state_before_commit']=recovery.report()
                    duration_repairs.append(report);history_edited=True
                    selected=(.5,witness,model.codec.encode_points([witness]),trial)
                    print('composer_v2_hold_release_repair '+json.dumps(report),flush=True)
                elif audit['native_checks'] or audit['duplicate_trials']:
                    repairs.append(dict(reason='hold_duration_repair_unresolved',after=str(after),**audit))
        if selected is None:
            replay=restore_prefix(points,tempo,clock,audio.duration_sec,failed_beat,conflict_sources,
                rejected=rejected) if points and budget_left and args.temperature!=0 else None
            if replay:
                recovery.spend()
                points,guard,report=replay;report['recovery_state']=recovery.report();rollbacks.append(report);sampler=None
                print('composer_v2_rollback '+json.dumps(report),flush=True);continue
            if exhaustion_policy=='report':
                admitted=admit_physical_conflict(guard,rejected,after)
                if admitted:
                    point,trial,details=admitted
                    warning=dict(beat=str(point.beat),note_text=point.note_text,after=str(after),
                        reason='recovery_budget_exhausted' if not recovery.available else 'recovery_unavailable',
                        conflicts=details,**recovery.report())
                    bypassed_events.append(warning)
                    selected=(.5,point,model.codec.encode_points([point]),trial)
                    print('composer_v2_conflict_admitted '+json.dumps(warning,ensure_ascii=False),flush=True)
        if selected is None:
            from .diagnostics import write_failure_snapshot
            diagnostic=write_failure_snapshot(args.output,dict(after=str(after),checkpoint=str(args.checkpoint),seed=args.seed,
                temperature=args.temperature,top_p=args.top_p,timing=timing_report,audio=str(audio_path),
                audio_end_beat=str(end),context_cursor=str(sampler.cursor),committed_in_context=committed_in_context,
                checkpoint_info=checkpoint_info,condition=condition_report,generation_settings=_generation_settings(args,device),
                rollbacks=rollbacks,duration_repairs=duration_repairs,**recovery.report(),
                hold_release_repair=hold_release_repair,repair_count=len(repairs),
                hold_repair_policy='native_hold_conflict_repair_v3',
                retry_policy='native_rejected_completion_conditioning_v1',retry_stats=retry_stats,
                exhaustion_policy=exhaustion_policy,bypassed_events=bypassed_events,
                repairs=repairs[-64:],points=[dict(beat=str(p.beat),note_text=p.note_text) for p in points]))
            detail='; no complete structurally valid physical-conflict candidate was available to admit' if exhaustion_policy=='report' else ''
            raise RuntimeError(f'Continuous event recovery exhausted after beat {after}{detail}; see {diagnostic}')
        _,point,encoded,guard=selected
        if args.max_events and len(points)>=args.max_events:raise ValueError('max_events would truncate the chart; increase the limit')
        # Alternatives may have changed decoder KV: retain only the unchanged prefix.
        points.append(point)
        if recovery.commit(point.beat):release_trials.clear();native_memo.clear()
        if history_edited:
            # Discard symbolic KV, REF/CARRY, and clock queries together. The
            # next context is rebuilt from the actual edited-and-committed chart.
            sampler=None
        else:
            sampler._rewind(target);target.extend(encoded);committed_in_context+=1
        if len(points)%16==0:
            print('composer_v2_event '+json.dumps(dict(events=len(points),beat=str(point.beat),context_refreshes=refreshes,repairs=len(repairs))),flush=True)
        if committed_in_context>=config.refresh_events:sampler=None
    export_end=Fraction(math.ceil(end*config.ticks_per_beat),config.ticks_per_beat)
    text,_=compile_event_points(points,tempo_changes=[t for t in tempo if t.beat<=end],bpm=tempo[0].bpm,
        first=first,end_beat=export_end,title=audio_path.stem,level=str(args.level),designer='Composer v2 continuous',strict_notes=False)
    output=Path(args.output).resolve();output.parent.mkdir(parents=True,exist_ok=True);output.write_text(text,encoding='utf-8')
    report=dict(family='chart_composer_v2',sequence_policy=config.sequence_policy,eos_policy='actual_chart_end_only',
        checkpoint=str(args.checkpoint),audio=str(audio_path),output=str(output),seed=args.seed,events=len(points),
        context_refreshes=refreshes,refresh_events=config.refresh_events,history_beats=config.history_beats,
        timing=timing_report,condition=condition_report,first=first,end_beat=str(export_end),duration_sec=audio.duration_sec,
        checkpoint_info=checkpoint_info,generation_settings=_generation_settings(args,device),
        repairs=repairs[-128:],repair_count=len(repairs),rollbacks=rollbacks,candidates=candidate_count,
        duration_repairs=duration_repairs,hold_release_repair=hold_release_repair,**recovery.report(),
        hold_repair_policy='native_hold_conflict_repair_v3',
        retry_policy='native_rejected_completion_conditioning_v1',retry_stats=retry_stats,
        exhaustion_policy=exhaustion_policy,bypassed_events=bypassed_events,
        accepted_with_conflicts_count=len(bypassed_events),
        physics_status='unresolved_conflicts' if bypassed_events else 'no_bypassed_conflicts',
        candidate_policy='next_complete_event_same_onset_atom_count',temperature=args.temperature,top_p=args.top_p,
        audio_identity=audio.identity,hard_guard='runtime_hard_v2')
    if bypassed_events:
        from .diagnostics import write_conflict_snapshot
        assessment=conflict_assessment(points,tempo,first,guard)
        conflict_path=write_conflict_snapshot(output,dict(checkpoint_info=checkpoint_info,
            checkpoint=str(args.checkpoint),audio=str(audio_path),output=str(output),
            timing=timing_report,first=first,tempo_map=[dict(beat=str(t.beat),bpm=t.bpm) for t in tempo],
            condition=condition_report,duration_sec=audio.duration_sec,
            points=[dict(beat=str(p.beat),note_text=p.note_text) for p in points],
            exhaustion_policy=exhaustion_policy,bypassed_events=bypassed_events,
            accepted_with_conflicts_count=len(bypassed_events),assessment=assessment,
            generation_settings=_generation_settings(args,device)))
        report.update(conflict_report=str(conflict_path),conflict_assessment_status=assessment['status'],
            final_enforced_conflict_count=assessment['enforced_conflict_count'])
    output.with_suffix('.composer_v2.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('composer_v2_done '+json.dumps(dict(output=str(output),events=len(points),context_refreshes=refreshes,
        accepted_with_conflicts_count=len(bypassed_events),conflict_report=report.get('conflict_report'))),flush=True)
    return report


def _generation_settings(args,device):
    keys=('level','seed','temperature','top_p','candidates','cpu_threads','device','cache_dir',
        'window_retries','rollback_windows','exhaustion_policy','max_events','condition','bpm','first','tempo_map','meter_map')
    return {**{key:getattr(args,key,None) for key in keys},'effective_device':str(device)}
