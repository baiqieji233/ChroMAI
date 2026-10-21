"""Inference bridge sharing the training clock, probes and archive encoders."""
from __future__ import annotations

from collections import OrderedDict
from fractions import Fraction
import numpy as np
import torch

from maichart.event_timing import seconds_at_beat
from maichart.generation.sustain_state import SustainStateIndex,_TYPE_OFFSETS
from maichart.generation.muri.parser import parse_markers
from maichart.generation.muri.model import RELEASE
from .grounding import CausalClock,features_at_beats,detail_queries,history_clock,action_summaries


class IncrementalState(SustainStateIndex):
    """Append one complete cell; never recompile the whole authored prefix."""
    def __init__(self,tempo,first):
        self.points=[];self.tempo_changes=tuple(tempo);self.first=float(first)
        self.markers=[];self._references=None;self._release_index=None

    def append(self,point):
        second=float(seconds_at_beat(point.beat,self.tempo_changes,self.first))
        bpm=next(t.bpm for t in reversed(self.tempo_changes) if t.beat<=point.beat)
        markers,errors,_=parse_markers(f'({bpm}){{4}}{point.note_text},',first=second)
        if errors:raise ValueError(f'Cannot append causal action state: {errors[:1]}')
        # Queries never precede the last committed head. Expired actions can
        # be discarded without affecting retries of uncommitted events.
        self.markers=[m for m in self.markers if float(m.get('end_second',-1))+RELEASE>=second]
        added=[]
        for marker in markers:
            if marker['type'] in _TYPE_OFFSETS:
                marker['key']=f'stream:{len(self.points)}:{marker["key"]}'
                self.markers.append(marker);added.append(marker)
        self.points.append(point);self._release_index=None
        self._stream_query_summaries=OrderedDict()
        return added


class GroundedInference:
    def __init__(self,model,codec,song,encoded_audio,audio_times=None):
        self.model,self.codec,self.song=model,codec,song
        self.device=next(model.parameters()).device
        detail,times,_=detail_queries(song,model.config.detail_audio_nodes)
        self.bank=model.grounding.prepare_bank(detail[None].to(self.device),
            torch.as_tensor(times,dtype=torch.float64,device=self.device)[None],
            torch.as_tensor((times<0)|(times>=song.audio.duration_sec),device=self.device)[None])
        self.global_context=encoded_audio[0].sum(1)/(~encoded_audio[1]).sum(1).clamp_min(1)[:,None]
        # The planner curve and its clock, for the same phrase pooling used in training.
        self.curve=encoded_audio[2] if audio_times is not None else None
        self.curve_mask=encoded_audio[1]
        self.audio_times=(torch.as_tensor(np.asarray(audio_times,dtype=np.float64),device=self.device)[None]
                          if audio_times is not None else None)
        from .grounded_model import prepare_phrase_bank
        self.phrase_bank=(prepare_phrase_bank(self.curve.detach().sigmoid(),self.audio_times,self.curve_mask)
                          if self.curve is not None else None)
        self.lane_sets=[];self.point_seconds=[];self.trajectory_sets=[]
        from .hand_tracking import HandHistory
        self.hand_history=HandHistory(self.lane_sets,self.point_seconds,self.trajectory_sets)
        self.state=IncrementalState(song.tempo,song.first)
        self.points=[];self.archive_values=[];self.archive_audio=[]
        self.archive_event_indices=[]
        self.centers=OrderedDict();self.context_cache=OrderedDict()
        self.prefix=[];self.clock_position=0;self.clock=None;self.archive=None
        self._timing_source=self._timing_tensor=None
        self.replay_state=None
        self._position_bank=None
        self.probes=self.query_hits=self.archive_queries=0
        self.routes_sum=torch.zeros(5,device=self.device);self.route_count=0

    def __getattr__(self,name):return getattr(self.model,name)

    def _new_clock(self):
        count=self.context_events
        return CausalClock(self.codec,self.song.tempo,self.song.first,self.song.audio.duration_sec,
            meters=self.song.meters,previous=self.points[count-1].beat if count else Fraction(0),
            previous_previous=self.points[count-2].beat if count>1 else Fraction(0),committed=count)

    def center(self,beat):
        if beat not in self.centers:
            values,_=features_at_beats(self.song,[beat]);self.centers[beat]=values[0].to(self.device)
            self.probes+=1
            if len(self.centers)>8192:self.centers.popitem(last=False)
        self.centers.move_to_end(beat)
        return self.centers[beat]

    def begin_context(self,points,history):
        self.context_events=len(points);self.clock=self._new_clock();self.clock_position=0
        # Replay queries precede the most recent committed event. Incremental
        # state has pruned older actions, so use the complete causal prefix.
        self.replay_state=(SustainStateIndex(self.points,self.song.tempo,self.song.first)
                           if len(points)<len(self.points) else None)
        self.context_cache.clear()
        full=self.codec.encode(points)[:-1] if points else [self.codec.bos_id]
        if history:
            if full[-len(history):]!=history:raise ValueError('Grounded history differs from committed prefix')
            positions=[i for i,t in enumerate(full) if t==self.codec.at_id]
            geometry=history_clock(full,len(full)-len(history),len(full),positions,points,self.codec,self.song)
            self.history_clock=geometry[None].to(self.device)
        else:self.history_clock=None
        count=max(0,len(points)-history.count(self.codec.at_id))
        candidate_slots=[slot for slot,event_index in enumerate(self.archive_event_indices) if event_index < count]
        if len(candidate_slots)>self.config.archive_events:
            selected=[candidate_slots[index] for index in np.linspace(
                0,len(candidate_slots)-1,self.config.archive_events,dtype=int).tolist()]
        else:
            selected=candidate_slots
        selected_events=[self.archive_event_indices[index] for index in selected]
        self.archive={
            'embeddings':torch.stack([self.archive_values[i] for i in selected])[None] if selected else self.global_context.new_zeros(1,0,self.config.d_model),
            'audio':torch.stack([self.archive_audio[i] for i in selected])[None] if selected else self.bank['values'].new_zeros(1,0,120),
            'padding':torch.zeros(1,len(selected),dtype=torch.bool,device=self.device),
            'beats':torch.tensor([[float(points[i].beat) for i in selected_events]],device=self.device,dtype=torch.float64),
            'seconds':torch.tensor([[seconds_at_beat(points[i].beat,self.song.tempo,self.song.first) for i in selected_events]],device=self.device,dtype=torch.float64),
            'event_index':torch.tensor([selected_events],device=self.device,dtype=torch.long)}

    def set_prefix(self,ids):self.prefix=ids

    def position_evidence(self,boundary,condition,*,strength):
        """Use the same BOS/END boundary state as alignment supervision."""
        from .position_evidence import PositionEvidence
        module=self.grounding
        if self._position_bank is None:
            keys=module.alignment_key(self.bank['values'])
            control=condition[:,None].expand(-1,self.bank['values'].size(1),-1)
            selection=module.selection_head(torch.cat((self.bank['projected'],control.to(
                self.bank['projected'])),-1)).squeeze(-1).float().sigmoid()
            self._position_bank=keys,selection
        keys,selection=self._position_bank
        seconds=self.bank['times'].new_tensor([[self.clock.second]])
        bpm=seconds.new_tensor([[self.clock.timeline.bpm(self.clock.beat)]])
        scores,_,indices=module.alignment_window(boundary[:,-1:],seconds,bpm,
            self.bank['values'],self.bank['times'],self.bank['padding'],full_keys=keys)
        slots=indices[0,0]
        values=torch.stack((self.bank['times'][0,slots],scores[0,0],selection[0,slots])).cpu().numpy()
        return PositionEvidence(*values,strength=strength)

    @property
    def timing_features(self):
        return self._feature_tensor(self.clock._features)

    def _feature_tensor(self,features):
        # CausalClock reuses its numpy row until beat/delta/commit/pending
        # changes. Most position digits and note characters share this row.
        if features is not self._timing_source:
            self._timing_source=features
            self._timing_tensor=torch.as_tensor(features,device=self.device)[None,None]
        return self._timing_tensor

    def _context(self,features):
        key=(self.clock.beat,self.clock.committed,self.clock.pending)
        if key not in self.context_cache:
            center=self.center(self.clock.beat)[None,None]
            seconds=torch.tensor([[self.clock.second]],device=self.device,dtype=torch.float64)
            context=self.grounding.contexts(features[:1],seconds,center,self.bank,self.archive)
            if self.curve is not None:
                context['local']=context['local']+self.grounding.phrase_from_bank(
                    self.phrase_bank,seconds).to(context['local'].dtype)
            state=self.replay_state if self.replay_state is not None and self.clock.committed<len(self.points) else self.state
            actions=action_summaries(state,[self.clock.beat],[self.clock.pending],[self.clock.committed])[None].to(self.device)
            context['action']=self.grounding.action_value(actions)*actions[...,-1:].gt(0)
            from .symmetry import kinematic_rows
            kinematics=kinematic_rows(self.lane_sets,self.point_seconds,[self.clock.committed],[self.clock.second],
                                      trajectories=self.trajectory_sets,pending=[self.clock.pending],profile=self.config.motor_profile)
            context['action']=context['action']+self.grounding.kinematics(kinematics[None].to(self.device)).to(context['action'].dtype)
            if self.config.motor_profile=='causal_hands_v1':
                hands=self.hand_history.rows([self.clock.committed],[self.clock.second])[None].to(self.device)
                context['action']=context['action']+self.grounding.hands(hands).to(context['action'].dtype)
            context['global']=self.global_context
            self.context_cache[key]=context;self.archive_queries+=1
            if len(self.context_cache)>2048:self.context_cache.popitem(last=False)
        else:self.query_hits+=1
        context=self.context_cache[key]
        return {key:value.expand(features.size(0),*value.shape[1:]) for key,value in context.items() if key in {'local','archive','action','global'}}

    def decode_step(self,token,position,memory,mask,caches=None,*,timing_features=None):
        if timing_features is None:
            if token.size(0)!=1:raise ValueError('Batched speculative decoding requires a fixed causal clock')
            if position!=self.clock_position:
                self.clock=self._new_clock()
                for item in self.prefix[:position]:self.clock.consume(item)
            # Normal generation already owns this exact token on the CPU.
            # Reading it back from CUDA would synchronize every decoder step.
            item=self.prefix[position] if position<len(self.prefix) else int(token[0,0])
            features=self.clock.consume(item)
            self.clock_position=position+1
            timing_features=self._feature_tensor(features)
        context=self._context(timing_features)
        output,caches=self.model.decode_step(token,position,memory,mask,caches,
            timing_features=timing_features,grounding_context=context)
        if output.get('routes') is not None:
            self.routes_sum+=output['routes'].detach().sum((0,1));self.route_count+=token.numel()
        return output,caches

    def commit(self,point):
        self.points.append(point);added=self.state.append(point)
        from .symmetry import event_lanes,sustain_trajectories
        self.lane_sets.append(event_lanes(point.note_text))
        self.point_seconds.append(float(seconds_at_beat(point.beat,self.song.tempo,self.song.first)))
        self.trajectory_sets.extend(sustain_trajectories(added or (),[self.point_seconds[-1]]))
        cell=[self.codec.ids['N:'+c] for c in point.note_text];n=len(cell)
        batch={'archive_padding_mask':torch.zeros(1,1,dtype=torch.bool,device=self.device),
            'archive_flat_tokens':torch.tensor(cell,device=self.device),
            'archive_token_event':torch.zeros(n,dtype=torch.long,device=self.device),
            'archive_token_position':(torch.arange(n,device=self.device)+.5)/n,
            'archive_token_inverse_length':torch.full((n,),1/n,device=self.device)}
        self.archive_values.append(self.grounding.archive_embeddings(batch,self.token_embedding)[0,0])
        self.archive_audio.append(self.center(point.beat))
        self.archive_event_indices.append(len(self.points)-1)
        limit=max(1,int(self.config.archive_events))
        if len(self.archive_values)>2*limit:
            keep=np.linspace(0,len(self.archive_values)-1,limit,dtype=int).tolist()
            self.archive_values=[self.archive_values[index] for index in keep]
            self.archive_audio=[self.archive_audio[index] for index in keep]
            self.archive_event_indices=[self.archive_event_indices[index] for index in keep]
        # The same clock after END now includes this newly committed action.
        self.context_cache.clear()

    def report(self):
        return {'profile':self.config.grounding_profile,'detail_audio_nodes':self.bank['values'].size(1),
            'motor_profile':self.config.motor_profile,
            'hand_assignment':('bounded_latent_strike_tracks_no_fixed_alternation'
                if self.config.motor_profile=='causal_hands_v1' else 'legacy_features_only'),
            'native_center_probes':self.probes,'local_query_cache_hits':self.query_hits,
            'archive_events_committed':len(self.points),'archive_candidates':self.archive['embeddings'].size(1),
            'mean_routes':dict(zip(('local_audio','archive','global_audio','active_actions','innovation'),
                (self.routes_sum/max(1,self.route_count)).cpu().tolist())),
            'clock':'consumed_prefix_only','positions_quantized':False}
