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


class GridDurationRejected(ValueError):
    def __init__(self,beat,body):
        super().__init__('grid_duration_outside_forward_bound')
        self.beat=beat
        self.body=body


class EventSampler(WindowSampler):
    def __init__(self,model,dataset,song,points,device,rng,args,*,action_generator=None):
        self.progress=getattr(args,'_generation_progress',None)
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
        self.local_anchor_ids = {a: i for i, a in enumerate(item['local_anchors'])}
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
        self._fixed_retry_scope=None;self._fixed_retry_tree=None;self._retry_offset=0
        need_probabilities=(bool(getattr(args,'record_candidate_probabilities',False))
            or getattr(args,'phrase_rank_policy','comfort_v1')!='comfort_v1')
        self.record_probabilities=bool(need_probabilities
            and getattr(args,'phrase_search','layout')!='event' and getattr(args,'candidates',2)>1
            and getattr(args,'_candidate_audit_active',True))
        self.last_probability_event=None;self._probability=None
        self._decoded_ids=();self._query_ids=()
        self._clock_states=[];self._clocks=[]
        self._tiles=[];self._local_tiles=[];self._token_buffers=None
        self.args=args;self._active_guard=None;self._forward_cache=None

    def _forward_guard(self):
        guard=getattr(self,'_active_guard',None)
        policy=getattr(self.args,'forward_feasibility','balanced_v1') if hasattr(self,'args') else 'off'
        if guard is None or policy=='off' or not hasattr(guard,'prefix') or not hasattr(guard,'_markers'):
            return None
        tos_ms=getattr(self.args,'forward_tap_on_slide_ms',None)
        key=(id(guard),len(guard.prefix),guard.prefix[-1][0] if guard.prefix else None,policy,tos_ms)
        cached=getattr(self,'_forward_cache',None)
        if cached is not None and cached[0]==key:
            return cached[1]
        from .forward_feasibility import ForwardFeasibilityGuard
        fg=ForwardFeasibilityGuard(guard,self.codec,song=self.song,policy=policy,forward_tap_on_slide_ms=tos_ms)
        self._forward_cache=(key,fg)
        return fg

    def _rewind(self,target):
        state=self.cache.get('decode_state') if self.cache is not None else None
        decoded=getattr(self,'_decoded_ids',())
        if state is None or not decoded:
            return super()._rewind(target)
        desired=(*self.prefix,*target)
        common=0
        for old,new in zip(decoded[:state['length']],desired):
            if old!=new:break
            common+=1
        # Retain all matching KV when new tokens follow. If the entire query
        # is already cached, replay its last token to obtain fresh logits.
        retained=min(common,max(0,len(desired)-1))
        self.model.rewind_decode_cache(self.cache,retained+1)
        self._decoded_ids=decoded[:self.cache['decode_state']['length']]
    def next(self,target,allowed,*,action=False):
        if getattr(self,'progress',None) is not None:self.progress.pulse()
        if len(target)>=self.max_tokens//2:raise RefreshContext('Component budget reached inside next event')
        if not allowed:
            scope=getattr(self,'proposal_scope',{})
            begin=scope.get('target_prefix_length',0)
            tokens=list(target[begin:])
            raise GrammarDeadEnd(dict(**scope,token_count=len(tokens),
                proposal_token_ids=tokens[-64:],proposal_tokens=[self.codec.tokens[i] for i in tokens[-64:]],
                allowed_token_count=0,context_cursor=str(self.cursor)))
        fixed_retry=bool(getattr(self,'_retry_offset',0))
        tree=self._fixed_retry_tree if fixed_retry else getattr(self,'_retry_tree',None)
        relative=target[len(self._retry_base):] if tree is not None else None
        logits=tree.cached(relative) if tree is not None else None
        if logits is None:
            if getattr(self,'_retry_cache_dirty',False):self._rewind(target)
            logits=self._logits(target);self._retry_cache_dirty=False
            self.retry_stats['decoder_queries']=self.retry_stats.get('decoder_queries',0)+1
        else:
            self._retry_cache_dirty=True
            self.retry_stats['decoder_cache_hits']=self.retry_stats.get('decoder_cache_hits',0)+1
        if action:
            generator,temperature,top_p=self.action_generator,self.action_temperature,self.action_top_p
        else:
            generator,temperature,top_p=self.generator,self.temperature,self.top_p
        raw_logits=logits
        grammar=getattr(self,'_audit_grammar',None) or getattr(self,'_free_choices',allowed)
        if tree is not None:
            choices=allowed if fixed_retry else getattr(self,'_free_choices',allowed)
            logits=tree.remember(relative,choices,logits,temperature=temperature)
            if self._retry_condition:
                logits,allowed,scaled=tree.condition(relative,allowed,temperature)
                if scaled:
                    self.retry_stats['conditioned_draws']=self.retry_stats.get('conditioned_draws',0)+1
                    temperature=1.
        token=_sample(logits,allowed,generator,temperature,top_p)
        if getattr(self,'_probability',None) is not None:
            self._probability.record(raw_logits,token,grammar,role='configuration' if action else 'timing',
                proposal_logits=logits,allowed=allowed,temperature=temperature,top_p=top_p)
        return token
    def _logits(self,target,*,all_new=False):
        ids=(*self.prefix,*target)
        retained=min(len(ids),len(self._query_ids))
        # Input buffers follow the most recent query. The KV cache may have
        # rewound independently to an earlier prefix.
        if ids[:retained]!=self._query_ids[:retained]:
            retained=next(i for i in range(retained) if ids[i]!=self._query_ids[i])
        initial=self._clock_states[retained-1] if retained else (0.,'',False)
        new_states=list(self.codec.clock_states(ids[retained:],self.cursor,initial=initial))
        new_clocks=[state[0] for state in new_states]
        beats=[math.floor(t) for t in new_clocks]
        missing=sorted(set(beats)-self.anchor_ids.keys())
        shared_audio=getattr(self.dataset,'local_audio_block_beats',1)!=1
        if missing and not shared_audio:
            self._append_single_beat_tiles(missing)
        elif missing:
            local_missing = sorted({self.dataset.local_anchor(a) for a in missing} - self.local_anchor_ids.keys())
            if local_missing:
                tiles = [self.dataset.tile(self.song, a, include_beat_features=False) for a in local_missing]
                local_batch = {k: torch.stack([t[k] for t in tiles]).to(self.device) for k in tiles[0]}
                local_batch['tile_owner'] = torch.zeros(len(tiles), dtype=torch.long, device=self.device)
                keep = torch.ones(1, dtype=torch.bool, device=self.device)
                for key, value in self.model.encode_local_audio(local_batch, keep).items():
                    self.cache[key] = torch.cat((self.cache[key], value), dim=0)
                for anchor in local_missing:
                    self.local_anchor_ids[anchor] = len(self.local_anchor_ids)
            beat_batch = {k: v.to(self.device) for k, v in self.dataset.beat_features(self.song, missing).items()}
            beat_batch['onset_strength'] = self.batch['onset_strength']
            for key, value in self.model.encode_beat_features(beat_batch).items():
                self.cache[key] = torch.cat((self.cache[key], value), dim=0)
        for a in missing:
            self.anchor_ids[a] = len(self.anchors)
            self.anchors.append(a)
        self._clock_states[retained:]=new_states
        self._clocks[retained:]=new_clocks
        self._tiles[retained:]=[self.anchor_ids[beat] for beat in beats]
        if shared_audio:
            self._local_tiles[retained:]=[self.local_anchor_ids[self.dataset.local_anchor(t)] for t in new_clocks]
            local_tiles=self._local_tiles
        else:
            local_tiles=self._tiles
        if self._token_buffers is None or self._token_buffers[0].size(1)<len(ids):
            shape=(1,max(self.max_tokens,len(ids)))
            self._token_buffers=(torch.empty(shape,dtype=torch.long,device=self.device),
                torch.empty(shape,dtype=torch.float32,device=self.device),
                torch.ones(shape,dtype=torch.bool,device=self.device))
            upload_start=0
        else:
            upload_start=retained
        token_ids,token_beats,token_mask=self._token_buffers
        clocks=self._clocks
        if upload_start<len(ids):
            token_ids[0,upload_start:len(ids)]=torch.tensor(ids[upload_start:],device=self.device)
            token_beats[0,upload_start:len(ids)]=torch.tensor(clocks[upload_start:],dtype=torch.float32,device=self.device)
        self._query_ids=ids
        query={**self.batch,'input_ids':token_ids[:,:len(ids)],'token_beats':token_beats[:,:len(ids)],
            'token_mask':token_mask[:,:len(ids)],'tile_ids_cpu':[self._tiles],'local_tile_ids_cpu':[local_tiles]}
        state=self.cache.get('decode_state') if not self.model.training and not torch.is_grad_enabled() else None
        start=int(state['length']) if state and state['length']<len(ids) else 0
        if getattr(self.dataset,'intensity_policy','off')!='off':
            from .intensity import query_indices
            query.update({k:v[None].to(self.device) for k,v in query_indices(self.song,clocks[start:]).items()})
            query['curve_query_start']=start
        if getattr(self.dataset,'onset_readout_policy','off') != 'off':
            phase_id=self.codec.ids['[PHASE]']
            query['phase_queries_cpu']=[(0,pos,self._tiles[pos]) for pos in range(start,len(ids)) if ids[pos]==phase_id]
        if self.dataset._evidence_v2():
            from .rhythm_structure import local_evidence
            boundary=max(start,len(self.prefix))
            pieces=[]
            if start<len(self.prefix):pieces.append(self.batch['reference_evidence'][:,start:len(self.prefix)])
            if boundary<len(ids):
                values=local_evidence(self.dataset._profile(self.song),clocks[boundary:],self.reference['shift'],
                    start=math.floor(self.cursor)-4,end=self.reference_refresh_beat)
                pieces.append(torch.as_tensor(values,device=self.device)[None])
            query['reference_evidence']=torch.cat(pieces,1)
            query['reference_evidence_start']=start
        if self.dataset._paired_music():
            from .music_evidence import paired_music_features
            boundary=max(start,len(self.prefix));pieces=[]
            if start<len(self.prefix):pieces.append(self.batch['paired_music_features'][:,start:len(self.prefix)])
            if boundary<len(ids):
                values=paired_music_features(self.dataset._profile(self.song),clocks[boundary:],self.reference['shift'],
                    start=math.floor(self.cursor)-4,end=self.reference_refresh_beat)
                pieces.append(torch.as_tensor(values,device=self.device)[None])
            query['paired_music_features']=torch.cat(pieces,1)
            query['paired_music_start']=start
        logits=self.model.decode(query,self.cache)[0]
        self._decoded_ids=ids
        return logits if all_new else logits[-1]

    def _append_single_beat_tiles(self, missing):
        tiles = [self.dataset.tile(self.song, anchor) for anchor in missing]
        query = {
            key: torch.stack([tile[key] for tile in tiles]).to(self.device)
            for key in tiles[0] if not key.startswith('onset_')
        }
        query['tile_owner']=torch.zeros(len(missing),dtype=torch.long,device=self.device)
        query['onset_strength']=self.batch['onset_strength']
        if 'onset_audio' in tiles[0]:
            from .onset_readout import collate_tiles
            onset = collate_tiles([{k: v[None] for k, v in tile.items() if k.startswith('onset_')} for tile in tiles])
            query.update({k: v.to(self.device) for k, v in onset.items()})
        # Only these beat-local values are appended. The song memory,
        # conditions and intensity plan remain valid across the context.
        keep=torch.ones(1,dtype=torch.bool,device=self.device)
        encoded=self.model.encode_local_audio(query,keep)
        encoded.update(self.model.encode_beat_features(query))
        for key,value in encoded.items():
            self.cache[key]=torch.cat((self.cache[key],value),dim=0)
    def _forced_time_probability(self,target,tokens,after,end):
        if self._probability is None:return
        # _rewind leaves the last prefix token live. A single incremental pass
        # obtains every forced-time conditional, including the phase readout.
        values=self._logits(target+tokens[:-1],all_new=True)[-len(tokens):]
        if len(values)!=len(tokens):raise RuntimeError('Forced-time probability prefix is incomplete')
        for index,(token,logits) in enumerate(zip(tokens,values)):
            self._probability.record(logits,token,self.codec.time_choices(tokens[:index],after,end),
                role='timing',forced=True)
    def _finish_probability(self):
        self.last_probability_event=self._probability
    def reject_free(self,ids):
        tree=getattr(self,'_retry_tree',None)
        return tree.reject(ids) if tree is not None else False
    def reject(self,ids):
        offset=getattr(self,'_retry_offset',0)
        if not offset:return self.reject_free(ids)
        if list(ids[:offset])!=self._retry_base[-offset:]:return False
        return self._fixed_retry_tree.reject(ids[offset:])
    def propose(self,target,after,end,fixed=None):
        if getattr(self,'record_probabilities',False):
            from .probability_audit import EventProbability
            self._probability=EventProbability()
        else:self._probability=None
        self.last_probability_event=None
        self._audit_grammar=None
        self.proposal_scope=dict(after=str(after),audio_end_beat=str(end),target_prefix_length=len(target),
            fixed_beat=str(fixed.beat) if fixed is not None else None)
        scope=(tuple(target),after,end)
        self._retry_offset=0;self._retry_base=list(target)
        self._retry_cache_dirty=False;self._retry_condition=True
        self._rewind(target)
        fg=self._forward_guard()
        # A compact Tap such as "12" is one text atom but two note heads.
        wanted=None if fixed is None or getattr(fixed,'beat_only',False) else len(_atom_heads(fixed.note_text))
        if fixed is None:
            if scope!=getattr(self,'_retry_scope',None):
                self._retry_tree=RejectedEventTree();self._retry_scope=scope
            time=[]
            while not time or time[-1] not in self.codec.phases:
                grammar_allowed=self.codec.time_choices(time,after,end)
                self._audit_grammar=grammar_allowed
                allowed=fg.filter_time_choices(grammar_allowed,time,after,end) if fg is not None else grammar_allowed
                if not time and not getattr(self,'retry_allow_eos',True) and self.codec.at_id in grammar_allowed:
                    allowed=tuple(t for t in allowed if t!=self.codec.eos_id)
                self._free_choices=allowed
                token=self.next(target+time,allowed);time.append(token)
                if token==self.codec.eos_id:self._finish_probability();return time
        else:
            time=self.codec._time(fixed.beat)
            fixed_scope=(*scope,fixed.beat,wanted)
            if fixed_scope!=getattr(self,'_fixed_retry_scope',None):
                self._fixed_retry_tree=RejectedEventTree();self._fixed_retry_scope=fixed_scope
            self._retry_offset=len(time);self._retry_base.extend(time)
            self._forced_time_probability(target,time,after,end)
        split=time.index(self.codec.phase_id)
        event_beat=fixed.beat if fixed is not None else (
            Fraction(int(''.join(self.codec.digits[i] for i in time[1:split])))+self.codec.phases[time[-1]])
        out=list(time);used=set();atom=[];count=0;completed_atoms=[]
        while True:
            rejected_closures=[]
            if not atom:
                if wanted is not None and count==wanted:
                    if self._probability is not None:
                        if getattr(self,'_retry_cache_dirty',False):self._rewind(target+out)
                        logits=self._logits(target+out)
                        self._retry_cache_dirty=False
                        self._probability.record(logits,self.codec.event_end_id,
                            (*self.codec.kind_ids,self.codec.event_end_id),role='configuration',forced=True)
                    self._finish_probability()
                    out.append(self.codec.event_end_id);return out
                uncond_free=tuple(self.codec.kind_ids)+((self.codec.event_end_id,) if count else ())
                uncond_allowed=tuple(self.codec.kind_ids)+((self.codec.event_end_id,) if count and wanted is None else ())
                self._audit_grammar=uncond_free
                if fg is not None:
                    atoms_tuple=tuple(completed_atoms)
                    free_filtered=fg.filter_atom_start_choices(uncond_free,event_beat,atoms_tuple)
                    allowed=fg.filter_atom_start_choices(uncond_allowed,event_beat,atoms_tuple)
                    if not allowed and wanted is None and count and self.codec.event_end_id in uncond_free:
                        allowed=(self.codec.event_end_id,)
                    self._free_choices=free_filtered if free_filtered else uncond_free
                else:
                    allowed=uncond_allowed
                    self._free_choices=uncond_free
            else:
                uncond_next=self._head_choices(self.codec.allowed_atom_next(atom),atom,used)
                self._audit_grammar=uncond_next
                free_next=fg.filter_atom_next_choices(
                    uncond_next,atom,event_beat,tuple(completed_atoms),
                    rejected_closures=rejected_closures,
                ) if fg is not None else uncond_next
                # Sample an out-of-bounds close as a rejection. Masking it here
                # would transfer that probability to additional digits.
                free_next=(*free_next,*rejected_closures)
                self._free_choices=free_next
                allowed=free_next
                if wanted is not None and self.codec.tokens[atom[0]]=='N:tap':
                    positions=sum(self.codec.tokens[t].startswith('P:') for t in atom)
                    if count+positions>=wanted:
                        allowed=tuple(t for t in allowed if not self.codec.tokens[t].startswith('P:'))
            token=self.next(target+out,allowed,action=True);out.append(token)
            if token in rejected_closures:
                self.reject(out)
                self.retry_stats['grid_duration_rejections']=self.retry_stats.get('grid_duration_rejections',0)+1
                body=''.join(self.codec.tokens[t][2:] for t in atom if self.codec.tokens[t].startswith('V:'))
                raise GridDurationRejected(event_beat,body)
            if token==self.codec.event_end_id:self._finish_probability();return out
            atom.append(token)
            if token==self.codec.end_note_id:
                rendered=render_atom([self.codec.tokens[t] for t in atom])
                completed_atoms.append(rendered)
                heads=_atom_heads(rendered)
                used.update(heads);atom=[];count+=len(heads)


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
    from .generation_progress import get_progress
    progress=get_progress(args)
    progress.update(force=True,stage='encoding_audio',audio_duration_sec=float(audio.duration_sec))
    from .phrase_search import settings as phrase_settings,trial_search,attach_candidate_content,checked_event_guard
    phrase_mode,phrase_beats,phrase_commit_beats=phrase_settings(args)
    if getattr(args,'record_candidate_probabilities',False):
        audit_path=Path(args.output).resolve().with_suffix('.candidate_audit.jsonl')
        audit_path.parent.mkdir(parents=True,exist_ok=True);audit_path.write_text('',encoding='utf-8')
    from .exhaustion import resolve_policy,admit_physical_conflict,conflict_assessment,downgrade_or_skip_conflict
    exhaustion_policy=resolve_policy(getattr(args,'exhaustion_policy','auto'),config.sequence_policy)
    song=ComposerV2Song(audio_path.stem,audio_path,audio_path,5,args.level,[],clock,end,condition,audio=audio,condition_known=known)
    song.summary=dict(audio.summary(160));song.summary['beats']=_beats_at_seconds(clock,song.summary['times_sec'])
    dataset=StreamDataset.__new__(StreamDataset)
    dataset.codec=model.codec;dataset.max_tokens=config.max_tokens;dataset.history_beats=config.history_beats
    dataset.window_beats=config.window_beats;dataset.future_beats=config.future_beats
    dataset.local_audio_block_beats=getattr(config,'local_audio_block_beats',1)
    from .onset_readout import effective_strength,phase_values
    onset_strength=effective_strength(getattr(args,'onset_strength',None),getattr(config,'onset_readout_policy','off'))
    dataset.onset_readout_policy=getattr(config,'onset_readout_policy','off') if onset_strength else 'off'
    dataset.onset_phases=phase_values(model.codec) if onset_strength else None
    # Same aligned reference and phase inputs as training, never the training-only
    # reference augmentation.
    dataset.structure_policy=getattr(config,'structure_policy','off')
    dataset.intensity_policy=getattr(config,'intensity_policy','off')
    dataset.music_context_policy=getattr(config,'music_context_policy','off')
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
    from .tail_guard import playability_options
    tail_settings=playability_options(args)
    guard=_new_guard(tempo,first,audio.duration_sec,guard_options=tail_settings)
    points=[];target=[];sampler=None;committed_in_context=0;refreshes=0;rollbacks=[];repairs=[];done=False
    duration_repairs=[];release_trials=set();bypassed_events=[];downgraded_or_skipped=[]
    after_floor=Fraction(-1)
    retry_stats={};native_memo={};pending_phrase=[];phrase_trials=[];phrase_count=0
    phrase_compute=dict(proposal_calls=0,native_window_checks=0,rank_calls=0,
        proposal_sec=0.,native_window_sec=0.,rank_sec=0.,speculative_events=0,
        incomplete_searches=0,skipped_candidates=0)
    recovery=RecoveryBudget(args.rollback_windows)
    hold_release_repair=args.rollback_windows>0
    sampling_policy=_sampling_policy(args)
    attempts=sampling_policy['effective_retries']
    candidate_count=sampling_policy['effective_candidates']
    args._candidate_audit_active=phrase_mode!='event' and candidate_count>1
    sampler_options={}
    if sampling_policy['rng_policy']=='separate_timing_action_v1':
        # Keep both streams alive across context refreshes and conflict retries.
        action_seed=rng.initial_seed() ^ 0x9E3779B97F4A7C15
        sampler_options['action_generator']=torch.Generator(device=device).manual_seed(action_seed)
    while not done:
        progress.position(clock,points[-1].beat if points else Fraction(0),audio.duration_sec,events=len(points))
        if sampler is None:
            sampler=EventSampler(model,dataset,song,points,device,rng,args,**sampler_options);target=[];committed_in_context=0;refreshes+=1
            sampler.retry_stats=retry_stats
            if dataset.structure_policy!='off' or dataset.reference_selection=='aligned':references.append(sampler.reference)
        after=max(points[-1].beat if points else Fraction(-1),after_floor)
        selected=None;failed_beat=after;need_refresh=False;conflict_sources=set();rejected=[];history_edited=False
        prefix_key=tuple((p.beat,p.note_text) for p in points) if native_memo else None
        known_failed=native_memo.get(prefix_key,{}) if prefix_key is not None else {}
        failed_any=False
        sampler._active_guard=guard
        for attempt in range(attempts):
            if attempt:progress.update(stage='retrying')
            try:
                sampler._active_guard=guard
                sampler.retry_allow_eos=not failed_any and not known_failed and recovery.used==0
                queued=pending_phrase.pop(0) if pending_phrase and attempt==0 else None
                if attempt:pending_phrase.clear()
                ids=queued[1] if queued else sampler.propose(target,after,end)
                if ids==[model.codec.eos_id]:
                    root_choices=model.codec.time_choices([],after,end)
                    if not sampler.retry_allow_eos and model.codec.at_id in root_choices:
                        repairs.append(dict(reason='eos_is_not_physical_recovery',after=str(after)));continue
                    done=True;break
                proposal=model.codec.decode_points(ids)[0];failed_beat=proposal.beat
                candidates=[queued if queued is not None else (proposal,ids)]
                used_phrase=False
                if queued is None and phrase_mode!='event' and candidate_count>1:
                    progress.update(stage='comparing')
                    def make_sampler(history):
                        value=EventSampler(model,dataset,song,history,device,rng,args,**sampler_options)
                        value.retry_stats=retry_stats
                        return value
                    planned,evidence,trials=trial_search(sampler,target,points,guard,(proposal,ids),
                        args=args,song=song,tempo=tempo,end=end,count=candidate_count,
                        make_sampler=make_sampler,check_window=_check_window)
                    phrase_count+=1;phrase_trials.append(evidence);phrase_trials=phrase_trials[-16:]
                    for key,value in evidence.get('compute',{}).items():
                        phrase_compute[key]=phrase_compute.get(key,0)+value
                    phrase_compute['incomplete_searches']+=not bool(planned)
                    phrase_compute['skipped_candidates']+=evidence.get('skipped_candidates',0)
                    if getattr(args,'record_candidate_probabilities',False):
                        attach_candidate_content(evidence,points,trials,tempo=tempo,first=first)
                        path=Path(args.output).resolve().with_suffix('.candidate_audit.jsonl')
                        path.parent.mkdir(parents=True,exist_ok=True)
                        evidence.update(pool_index=phrase_count,
                            audio_sha256=audio.identity.get('audio_sha256'),checkpoint_info=checkpoint_info,
                            generation_settings=_generation_settings(args,device),
                            selection_scope='trial_decision_not_final_chart_commit')
                        record=dict(schema='composer-v2-candidate-pool-v1',**evidence)
                        with path.open('w' if phrase_count==1 else 'a',encoding='utf-8') as handle:
                            handle.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
                    if planned:
                        candidates=planned[:1];pending_phrase=planned[1:];used_phrase=True
                for _ in range(candidate_count-1 if not used_phrase and queued is None else 0):
                    try:
                        alternative=sampler.propose(target,after,end,fixed=proposal)
                    except GridDurationRejected as error:
                        repairs.append(dict(reason=str(error),beat=str(error.beat),duration_beats=error.body))
                        continue
                    except RetrySpaceExhausted:
                        retry_stats['exhausted_subtrees']=retry_stats.get('exhausted_subtrees',0)+1
                        break
                    candidates.append((model.codec.decode_points(alternative)[0],alternative))
                feasible=[]
                for candidate_index,candidate in enumerate(candidates):
                    point,encoded=candidate
                    event_key=(point.beat,point.note_text)
                    cached=event_key in known_failed
                    if cached:
                        trial=None;reason,details=known_failed[event_key]
                        retry_stats['native_cache_hits']=retry_stats.get('native_cache_hits',0)+1
                    else:
                        trial=checked_event_guard(candidate,guard)
                        details=[];reason=None
                        if trial is None:
                            trial,reason=_check_window(guard,[point],details)
                            retry_stats['native_checks']=retry_stats.get('native_checks',0)+1
                        else:
                            retry_stats['native_trial_reuse']=retry_stats.get('native_trial_reuse',0)+1
                    if trial is None:
                        pending_phrase.clear()
                        failed_any=True
                        if prefix_key is None:prefix_key=tuple((p.beat,p.note_text) for p in points)
                        native_memo.setdefault(prefix_key,known_failed)
                        known_failed[event_key]=(reason,details)
                        if candidate_index==0 and hasattr(sampler,'reject'):
                            getattr(sampler,'reject_free',sampler.reject)(encoded)
                        conflict_sources.update(Fraction(b) for issue in details for b in issue['note_beats'])
                        if reason=='new_miacode_runtime_hard_conflict':rejected.append((point,details))
                        repairs.append(dict(beat=str(point.beat),note_text=point.note_text,reason=reason,
                            conflicts=details,cached_rejection=cached));continue
                    score=.5 if candidate_count==1 or used_phrase or queued is not None else _comfort(points,[point],tempo,clock,max(Fraction(0),after),min(end,point.beat+1),audio.duration_sec)[0]
                    feasible.append((score,point,encoded,trial))
                if feasible:selected=max(feasible,key=lambda row:row[0]);break
            except RetrySpaceExhausted:
                retry_stats['exhausted_subtrees']=retry_stats.get('exhausted_subtrees',0)+1;break
            except GridDurationRejected as error:
                failed_any=True
                failed_beat=error.beat
                repairs.append(dict(reason=str(error),beat=str(error.beat),duration_beats=error.body))
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
                after_floor=points[-1].beat if points else Fraction(-1)
                print('composer_v2_rollback '+json.dumps(report),flush=True);continue
            elif exhaustion_policy=='downgrade_or_skip':
                resolved=downgrade_or_skip_conflict(guard,rejected,after)
                if resolved is not None:
                    pending_phrase.clear()
                    if resolved['kind']=='downgraded':
                        point=resolved['point']
                        trial=resolved['guard']
                        entry=dict(kind='downgraded',beat=str(point.beat),original=resolved['original'],
                            replacement=resolved['replacement'],operation=resolved['operation'],after=str(after))
                        downgraded_or_skipped.append(entry)
                        selected=(.5,point,model.codec.encode_points([point]),trial)
                        print('composer_v2_conflict_downgraded '+json.dumps(entry,ensure_ascii=False),flush=True)
                    elif resolved['kind']=='skipped_beat':
                        skip_to=resolved['skip_until_beat']
                        entry=dict(kind='skipped_beat',skipped_until_beat=str(skip_to),after=str(after))
                        downgraded_or_skipped.append(entry)
                        repairs.append(dict(reason='exhaustion_skipped_locked_beat',after=str(after),skipped_until=str(skip_to)))
                        after_floor=max(after_floor,skip_to)
                        print('composer_v2_conflict_beat_skipped '+json.dumps(entry,ensure_ascii=False),flush=True)
                        continue
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
        progress.position(clock,point.beat,audio.duration_sec,events=len(points))
        if recovery.commit(point.beat):release_trials.clear();native_memo.clear()
        if history_edited:
            # Discard symbolic KV, REF/CARRY, and clock queries together. The
            # next context is rebuilt from the actual edited-and-committed chart.
            sampler=None
        else:
            target.extend(encoded);sampler._rewind(target);committed_in_context+=1
        if len(points)%16==0:
            print('composer_v2_event '+json.dumps(dict(events=len(points),beat=str(point.beat),context_refreshes=refreshes,repairs=len(repairs))),flush=True)
        refresh_due=committed_in_context>=config.refresh_events
        if sampler is not None and dataset._evidence_v2():refresh_due|=point.beat>=sampler.reference_refresh_beat
        if refresh_due:sampler=None
    from .self_refine import refine_worst_spans
    def _refine_sampler(history):
        value=EventSampler(model,dataset,song,history,device,rng,args,**sampler_options)
        value.retry_stats=retry_stats
        return value
    points,guard,self_refine_report=refine_worst_spans(points,guard,args=args,song=song,tempo=tempo,
        first=first,end=end,make_sampler=_refine_sampler,new_guard_fn=_new_guard,
        check_window=_check_window,guard_options=tail_settings,bypassed_events=bypassed_events)
    progress.update(force=True,stage='exporting',events=len(points))
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
        touch_reachability=dict(policy=tail_settings['touch_reachability_policy'],
                                cohold_radius=tail_settings['touch_cohold_radius'],
                                intercept_slide_head_tap=tail_settings['intercept_slide_head_tap']),
        forward_feasibility=dict(policy=getattr(args,'forward_feasibility','balanced_v1'),
                                 forward_tap_on_slide_ms=getattr(args,'forward_tap_on_slide_ms',None)),
        self_refine=self_refine_report,
        repairs=repairs[-128:],repair_count=len(repairs),rollbacks=rollbacks,candidates=candidate_count,
        duration_repairs=duration_repairs,hold_release_repair=hold_release_repair,**recovery.report(),
        hold_repair_policy='native_hold_conflict_repair_v3',
        retry_policy='native_rejected_completion_conditioning_v1',retry_stats=retry_stats,
        exhaustion_policy=exhaustion_policy,bypassed_events=bypassed_events,
        downgraded_or_skipped_events=downgraded_or_skipped,
        accepted_with_conflicts_count=len(bypassed_events),
        physics_status='unresolved_conflicts' if bypassed_events else 'no_bypassed_conflicts',
        candidate_policy='next_complete_event_same_onset_atom_count',temperature=args.temperature,top_p=args.top_p,
        audio_identity=audio.identity,hard_guard='runtime_hard_v2',
        onset_readout=dict(policy=getattr(config,'onset_readout_policy','off'),strength=onset_strength,
                           scope='conditional_phase_only',phase_mass_preserved=True),
        rhythm_structure=_reference_report(dataset.structure_policy,references,
                                           requested=requested_selection,selection=dataset.reference_selection))
    report['cpu_runtime']=getattr(args,'cpu_runtime',None)
    if report['cpu_runtime'] is not None:report['cpu_runtime']['actual_threads']=torch.get_num_threads()
    report['acceleration']=getattr(getattr(model,'_decoder_runtime',None),'report',None)
    report['candidate_probability_audit']=dict(requested=bool(getattr(args,'record_candidate_probabilities',False)),
        status='recorded' if getattr(args,'record_candidate_probabilities',False) and phrase_count else
            'no_trial_competition' if getattr(args,'record_candidate_probabilities',False) else 'disabled',
        pools=phrase_count if getattr(args,'record_candidate_probabilities',False) else 0,
        path=str(Path(args.output).resolve().with_suffix('.candidate_audit.jsonl'))
            if getattr(args,'record_candidate_probabilities',False) and phrase_count else None,
        scope='trial_search_only; last 16 pools embedded, complete pool history in JSONL',
        ranking_changed=False)
    from .phrase_search import rank_policy as resolve_rank_policy
    rank_mode = resolve_rank_policy(args)
    report['phrase_search']=dict(mode=phrase_mode,ranking_policy=rank_mode,lookahead_beats=float(phrase_beats),
        commit_beats=float(phrase_commit_beats),decisions=phrase_count,trials=phrase_trials,compute=phrase_compute,
        scoring=('grammar_likelihood_and_attack_support_with_comfort_veto'
                 if rank_mode=='likelihood_veto_v1'
                 else 'chirality_hand_flow_grammar_likelihood_and_attack_support'
                 if rank_mode=='comfort_v2'
                 else 'native_trajectory_and_selected_attack_support_no_density_reward'),
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
    progress.update(force=True,stage='completed',status='completed',progress=1.,events=len(points))
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
    keys=('level','seed','temperature','top_p','action_temperature','action_top_p','onset_strength','reference_selection','intensity_curve','intensity_strength','intercept_tap_on_slide','tap_on_slide_threshold_ms','intercept_slide_head_tap','touch_reachability_policy','touch_cohold_radius','forward_feasibility','forward_tap_on_slide_ms','self_refine','candidates','phrase_search','phrase_rank_policy','phrase_beats','phrase_commit_beats','cpu_threads','device','cache_dir',
        'window_retries','rollback_windows','exhaustion_policy','max_events','condition','bpm','first','tempo_map','meter_map','tempo_detection','tempo_octave_policy','downbeat_alignment','record_candidate_probabilities','audio_device','inference_backend')
    return {**{key:getattr(args,key,None) for key in keys},'effective_device':str(device)}
