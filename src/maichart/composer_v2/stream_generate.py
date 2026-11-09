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
from .generate import WindowSampler,_sample,_atom_heads,_new_guard,_check_window,_comfort,_action_sampling,_sampling_policy
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
    def __init__(self,model,dataset,song,points,device,rng,args,*,action_generator=None):
        self.model=model;self.codec=model.codec;self.dataset=dataset;self.song=song
        self.device=device;self.generator=rng;self.temperature=args.temperature;self.top_p=args.top_p
        self.action_temperature,self.action_top_p,_=_action_sampling(args)
        self.action_generator=rng if action_generator is None else action_generator
        self.cursor=points[-1].beat if points else Fraction(0)
        item=dataset.build_stream(song,points,self.cursor)
        self.reference=dict(cursor=str(self.cursor),shift=item.get('reference_shift'),score=item.get('reference_score'),
            checked_start=item.get('reference_checked_start'),checked_end=item.get('reference_checked_end'))
        self.reference_refresh_beat=math.floor(self.cursor)+dataset.window_beats
        self.prefix=item['input_ids'].tolist();self.anchors=list(item['anchors'])
        self.anchor_ids={a:i for i,a in enumerate(self.anchors)}
        self.batch={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in collate_stream([item]).items()}
        from .onset_readout import effective_strength
        self.batch['onset_strength'] = effective_strength(getattr(args,'onset_strength',None),
            getattr(model.config,'onset_readout_policy','off'))
        self.batch['intensity_strength']=getattr(args,'intensity_strength',1.)
        self.max_tokens=model.config.max_tokens;self.cache=model.encode_audio(self.batch)
        if getattr(model.config,'intensity_policy','off')!='off' and not hasattr(dataset,'intensity_profile'):
            from .intensity import serialize,blend_profile
            dataset.intensity_profile=blend_profile(serialize(self.cache['curve_context'][0].expm1(),song),
                getattr(song,'intensity_input_profile',None),self.batch['intensity_strength'])
        self.retry_stats={};self._retry_scope=None
    def next(self,target,allowed,*,action=False):
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
        if action:
            generator,temperature,top_p=self.action_generator,self.action_temperature,self.action_top_p
        else:
            generator,temperature,top_p=self.generator,self.temperature,self.top_p
        if tree is not None:
            logits=tree.remember(relative,getattr(self,'_free_choices',allowed),logits,temperature=temperature)
            if self._retry_condition:
                logits,allowed,scaled=tree.condition(relative,allowed,temperature)
                if scaled:
                    self.retry_stats['conditioned_draws']=self.retry_stats.get('conditioned_draws',0)+1
                    temperature=1.
        return _sample(logits,allowed,generator,temperature,top_p)
    def _logits(self,target):
        ids=self.prefix+list(target)
        clocks=self.codec.token_clocks(ids,cursor=self.cursor)
        missing=sorted({math.floor(t) for t in clocks}-self.anchor_ids.keys())
        if missing:
            tiles=[self.dataset.tile(self.song,a) for a in missing]
            query={**self.batch,**{k:torch.stack([t[k] for t in tiles]).to(self.device) for k in tiles[0] if not k.startswith('onset_')},
                'tile_owner':torch.zeros(len(missing),dtype=torch.long,device=self.device)}
            if 'onset_audio' in tiles[0]:
                from .onset_readout import collate_tiles
                onset=collate_tiles([{k:v[None] for k,v in t.items() if k.startswith('onset_')} for t in tiles])
                query.update({k:v.to(self.device) for k,v in onset.items()})
            for key in ('bank_audio','bank_semantic','bank_semantic_mask','bank_semantic_right','bank_semantic_weight','bank_owner','tile_frames','bank_envelope'):query.pop(key,None)
            encoded=self.model.encode_audio(query)
            for key in ('local','local_mask','local_beats'):self.cache[key]=torch.cat((self.cache[key],encoded[key]),0)
            if 'onset_scores' in encoded:self.cache['onset_scores']=torch.cat((self.cache['onset_scores'],encoded['onset_scores']),0)
            for a in missing:self.anchor_ids[a]=len(self.anchors);self.anchors.append(a)
        query={**self.batch,'input_ids':torch.tensor([ids],device=self.device),
            'token_beats':torch.tensor([clocks],dtype=torch.float32,device=self.device),
            'token_mask':torch.ones((1,len(ids)),dtype=torch.bool,device=self.device),
            'tile_ids_cpu':[[self.anchor_ids[math.floor(t)] for t in clocks]]}
        if getattr(self.dataset,'intensity_policy','off')!='off':
            from .intensity import query_indices
            state=self.cache.get('decode_state')
            start=int(state['length']) if state and state['length']<len(ids) else 0
            query.update({k:v[None].to(self.device) for k,v in query_indices(self.song,clocks[start:]).items()})
            query['curve_query_start']=start
        if getattr(self.dataset,'onset_readout_policy','off') != 'off':
            from .onset_readout import query_rows
            query['phase_queries_cpu']=query_rows(ids,query['tile_ids_cpu'][0],self.codec.ids['[PHASE]'])
        if self.dataset._evidence_v2():
            from .rhythm_structure import local_evidence
            state=self.cache.get('decode_state') if not self.model.training and not torch.is_grad_enabled() else None
            start=int(state['length']) if state and state['length']<len(ids) else 0
            boundary=max(start,len(self.prefix))
            pieces=[]
            if start<len(self.prefix):pieces.append(self.batch['reference_evidence'][:,start:len(self.prefix)])
            if boundary<len(ids):
                values=local_evidence(self.dataset._profile(self.song),clocks[boundary:],self.reference['shift'],
                    start=math.floor(self.cursor)-4,end=self.reference_refresh_beat)
                pieces.append(torch.as_tensor(values,device=self.device)[None])
            query['reference_evidence']=torch.cat(pieces,1)
            query['reference_evidence_start']=start
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
            token=self.next(target+out,allowed,action=True);out.append(token)
            if token==self.codec.event_end_id:return out
            atom.append(token)
            if token==self.codec.end_note_id:
                used.update(_atom_heads(render_atom([self.codec.tokens[t] for t in atom])));atom=[];count+=1


def restore_prefix(points,tempo,clock,duration,failed_beat,conflict_sources=(),*,rejected=(),guard_options=None):
    """Restore committed state, checking any changed suffix with a fresh trial."""
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
        base=_new_guard(tempo,clock.seconds[0],duration,guard_options=guard_options);states={}
        for i,point in enumerate(points):
            if i>=lower:states[i]=deepcopy(base)
            # Report mode can have committed physical conflicts in this history.
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
    guard=_new_guard(tempo,clock.seconds[0],duration,guard_options=guard_options)
    for point in kept:
        guard.commit(point)
    return kept,guard,dict(source_beat=str(source),revoked_events=len(points)-len(kept),reason=reason,
        removed_points=[dict(beat=str(p.beat),note_text=p.note_text) for p in points[len(kept):]])


def generate_continuous(model,audio,audio_path,tempo,first,timing_report,condition,known,condition_report,args,rng,device,end,*,checkpoint_info=None):
    from .data import ComposerV2Song,_beats_at_seconds
    from maichart.tempo_clock import TempoClock
    config=model.config;clock=TempoClock(tempo,first)
    from .phrase_search import settings as phrase_settings,trial_search,write_previews
    phrase_mode,phrase_beats,phrase_commit_beats=phrase_settings(args)
    from .exhaustion import resolve_policy,admit_physical_conflict,conflict_assessment
    exhaustion_policy=resolve_policy(getattr(args,'exhaustion_policy','auto'),config.sequence_policy)
    song=ComposerV2Song(audio_path.stem,audio_path,audio_path,5,args.level,[],clock,end,condition,audio=audio,condition_known=known)
    song.summary=dict(audio.summary(160));song.summary['beats']=_beats_at_seconds(clock,song.summary['times_sec'])
    dataset=StreamDataset.__new__(StreamDataset)
    dataset.codec=model.codec;dataset.max_tokens=config.max_tokens;dataset.history_beats=config.history_beats
    dataset.window_beats=config.window_beats;dataset.future_beats=config.future_beats
    from .onset_readout import effective_strength,phase_values
    onset_strength=effective_strength(getattr(args,'onset_strength',None),getattr(config,'onset_readout_policy','off'))
    dataset.onset_readout_policy=getattr(config,'onset_readout_policy','off') if onset_strength else 'off'
    dataset.onset_phases=phase_values(model.codec) if onset_strength else None
    # Same aligned reference and phase inputs as training, never the training-only
    # reference augmentation.
    dataset.structure_policy=getattr(config,'structure_policy','off')
    dataset.intensity_policy=getattr(config,'intensity_policy','off')
    if getattr(args,'intensity_curve',None):
        from .intensity import profile_rates
        profile=json.loads(Path(args.intensity_curve).read_text(encoding='utf-8'))
        profile_rates(profile,song)  # Validate source identity and numeric contract once.
        song.intensity_input_profile=profile
    dataset.reference_augmentation=False;dataset.refresh_events=config.refresh_events
    requested_selection=getattr(args,'reference_selection','auto') or 'auto'
    # Older checkpoints keep their REF layout; 'aligned' only changes which
    # earlier passage fills it and omits it when nothing repeats.
    dataset.reference_selection='aligned' if requested_selection=='aligned' and dataset.structure_policy=='off' else 'model'
    references=[]
    from .tail_guard import options as tail_options
    tail_settings=tail_options(args)
    guard=_new_guard(tempo,first,audio.duration_sec,guard_options=tail_settings)
    points=[];target=[];sampler=None;committed_in_context=0;refreshes=0;rollbacks=[];repairs=[];done=False
    duration_repairs=[];release_trials=set();bypassed_events=[]
    retry_stats={};native_memo={};pending_phrase=[];phrase_trials=[];phrase_count=0
    recovery=RecoveryBudget(args.rollback_windows)
    hold_release_repair=args.rollback_windows>0
    sampling_policy=_sampling_policy(args)
    attempts=sampling_policy['effective_retries']
    candidate_count=sampling_policy['effective_candidates']
    sampler_options={}
    if sampling_policy['rng_policy']=='separate_timing_action_v1':
        # Keep both streams alive across context refreshes and conflict retries.
        action_seed=rng.initial_seed() ^ 0x9E3779B97F4A7C15
        sampler_options['action_generator']=torch.Generator(device=device).manual_seed(action_seed)
    while not done:
        if sampler is None:
            sampler=EventSampler(model,dataset,song,points,device,rng,args,**sampler_options);target=[];committed_in_context=0;refreshes+=1
            sampler.retry_stats=retry_stats
            if dataset.structure_policy!='off' or dataset.reference_selection=='aligned':references.append(sampler.reference)
        after=points[-1].beat if points else Fraction(-1)
        selected=None;failed_beat=after;need_refresh=False;conflict_sources=set();rejected=[];history_edited=False
        prefix_key=tuple((p.beat,p.note_text) for p in points) if native_memo else None
        known_failed=native_memo.get(prefix_key,{}) if prefix_key is not None else {}
        failed_any=False
        for attempt in range(attempts):
            try:
                sampler.retry_allow_eos=not failed_any and not known_failed and recovery.used==0
                queued=pending_phrase.pop(0) if pending_phrase and attempt==0 else None
                if attempt:pending_phrase.clear()
                ids=queued[1] if queued else sampler.propose(target,after,end)
                if ids==[model.codec.eos_id]:
                    if not sampler.retry_allow_eos and model.codec.at_id in model.codec.time_choices([],after,end):
                        repairs.append(dict(reason='eos_is_not_physical_recovery',after=str(after)));continue
                    done=True;break
                proposal=model.codec.decode_points(ids)[0];failed_beat=proposal.beat
                candidates=[(proposal,ids)]
                used_phrase=False
                if queued is None and phrase_mode!='event' and candidate_count>1:
                    def make_sampler(history):
                        value=EventSampler(model,dataset,song,history,device,rng,args,**sampler_options)
                        value.retry_stats=retry_stats
                        return value
                    planned,evidence,trials=trial_search(sampler,target,points,guard,(proposal,ids),
                        args=args,song=song,tempo=tempo,end=end,count=candidate_count,
                        make_sampler=make_sampler,check_window=_check_window)
                    write_previews(args.output,phrase_count,points,trials,evidence,song,tempo,args.level)
                    phrase_count+=1;phrase_trials.append(evidence);phrase_trials=phrase_trials[-16:]
                    if planned:
                        candidates=planned[:1];pending_phrase=planned[1:];used_phrase=True
                for _ in range(candidate_count-1 if not used_phrase and queued is None else 0):
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
                        pending_phrase.clear()
                        failed_any=True
                        if prefix_key is None:prefix_key=tuple((p.beat,p.note_text) for p in points)
                        native_memo.setdefault(prefix_key,known_failed)
                        known_failed[event_key]=(reason,details)
                        if candidate_index==0 and hasattr(sampler,'reject'):sampler.reject(encoded)
                        conflict_sources.update(Fraction(b) for issue in details for b in issue['note_beats'])
                        if reason=='new_miacode_runtime_hard_conflict':rejected.append((point,details))
                        repairs.append(dict(beat=str(point.beat),note_text=point.note_text,reason=reason,
                            conflicts=details,cached_rejection=cached));continue
                    score=.5 if candidate_count==1 or used_phrase or queued is not None else _comfort(points,[point],tempo,clock,max(Fraction(0),after),min(end,point.beat+1),audio.duration_sec)[0]
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
                    attempted=release_trials,audit=audit,guard_options=tail_settings)
                if repaired:
                    pending_phrase.clear()
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
                rejected=rejected,guard_options=tail_settings) if points and budget_left and not sampling_policy['fully_greedy'] else None
            if replay:
                pending_phrase.clear()
                recovery.spend()
                points,guard,report=replay;report['recovery_state']=recovery.report();rollbacks.append(report);sampler=None
                print('composer_v2_rollback '+json.dumps(report),flush=True);continue
            if exhaustion_policy=='report':
                admitted=admit_physical_conflict(guard,rejected,after)
                if admitted:
                    pending_phrase.clear()
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
                temperature=args.temperature,top_p=args.top_p,sampling_policy=sampling_policy,
                timing=timing_report,audio=str(audio_path),
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
        refresh_due=committed_in_context>=config.refresh_events
        if sampler is not None and dataset._evidence_v2():refresh_due|=point.beat>=sampler.reference_refresh_beat
        if refresh_due:sampler=None
    from .tail_guard import cleanup
    points,tail_report=cleanup(points,tempo,first,guard)
    export_end=Fraction(math.ceil(end*config.ticks_per_beat),config.ticks_per_beat)
    text,_=compile_event_points(points,tempo_changes=[t for t in tempo if t.beat<=end],bpm=tempo[0].bpm,
        first=first,end_beat=export_end,title=audio_path.stem,level=str(args.level),designer='Composer v2 continuous',strict_notes=False)
    output=Path(args.output).resolve();output.parent.mkdir(parents=True,exist_ok=True);output.write_text(text,encoding='utf-8')
    report=dict(family='chart_composer_v2',sequence_policy=config.sequence_policy,eos_policy='actual_chart_end_only',
        checkpoint=str(args.checkpoint),audio=str(audio_path),output=str(output),seed=args.seed,events=len(points),
        context_refreshes=refreshes,refresh_events=config.refresh_events,history_beats=config.history_beats,
        timing=timing_report,condition=condition_report,first=first,end_beat=str(export_end),duration_sec=audio.duration_sec,
        checkpoint_info=checkpoint_info,generation_settings=_generation_settings(args,device),
        sampling_policy=sampling_policy,tap_on_slide=tail_report,
        repairs=repairs[-128:],repair_count=len(repairs),rollbacks=rollbacks,candidates=candidate_count,
        duration_repairs=duration_repairs,hold_release_repair=hold_release_repair,**recovery.report(),
        hold_repair_policy='native_hold_conflict_repair_v3',
        retry_policy='native_rejected_completion_conditioning_v1',retry_stats=retry_stats,
        exhaustion_policy=exhaustion_policy,bypassed_events=bypassed_events,
        accepted_with_conflicts_count=len(bypassed_events),
        physics_status='unresolved_conflicts' if bypassed_events else 'no_bypassed_conflicts',
        candidate_policy='next_complete_event_same_onset_atom_count',temperature=args.temperature,top_p=args.top_p,
        audio_identity=audio.identity,hard_guard='runtime_hard_v2',
        onset_readout=dict(policy=getattr(config,'onset_readout_policy','off'),strength=onset_strength,
                           scope='conditional_phase_only',phase_mass_preserved=True),
        rhythm_structure=_reference_report(dataset.structure_policy,references,
                                           requested=requested_selection,selection=dataset.reference_selection))
    report['cpu_runtime']=getattr(args,'cpu_runtime',None)
    report['phrase_search']=dict(mode=phrase_mode,lookahead_beats=float(phrase_beats),
        commit_beats=float(phrase_commit_beats),decisions=phrase_count,trials=phrase_trials,
        scoring='native_trajectory_and_selected_attack_support_no_density_reward',
        speculative_suffix_is_committed=False)
    if phrase_mode!='event':report['candidate_policy']='continuous_phrase_'+phrase_mode
    report['intensity_curve']=dict(policy=dataset.intensity_policy,
        source='off' if dataset.intensity_policy=='off' else 'edited' if getattr(args,'intensity_curve',None) else 'model',
        strength=getattr(args,'intensity_strength',1.),profile=getattr(dataset,'intensity_profile',None))
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


def _reference_report(policy,references,*,requested='auto',selection='model'):
    if policy=='off' and selection!='aligned':
        return dict(policy='off',reference_selection=requested,reference='checkpoint_trained_mean_pooled_v1')
    from .rhythm_structure import REF_MIN_SCORE
    used=[r for r in references if r['shift'] is not None]
    scores=[r['score'] for r in used]
    layout=('per_beat_evidence_full_coverage_v2' if policy=='aligned_rhythm_v2' else 'aligned_integer_beat_shift_trust_v1') if policy!='off' else 'aligned_choice_legacy_layout_training_free_v1'
    return dict(policy=policy,reference_selection=requested,reference=layout,min_score=REF_MIN_SCORE,
        context_refreshes=len(references),refreshes_with_reference=len(used),
        mean_reference_score=float(np.mean(scores)) if scores else None,
        recent=references[-64:])


def _generation_settings(args,device):
    keys=('level','seed','temperature','top_p','action_temperature','action_top_p','onset_strength','reference_selection','intensity_curve','intensity_strength','intercept_tap_on_slide','tap_on_slide_threshold_ms','candidates','phrase_search','phrase_beats','phrase_commit_beats','cpu_threads','device','cache_dir',
        'window_retries','rollback_windows','exhaustion_policy','max_events','condition','bpm','first','tempo_map','meter_map','tempo_detection')
    return {**{key:getattr(args,key,None) for key in keys},'effective_device':str(device)}
