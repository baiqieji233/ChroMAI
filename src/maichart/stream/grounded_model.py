"""Query-local acoustics and causal chart archives without quadratic bank encoding."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from .grounding import CLOCK_DIM
from .symmetry import KINEMATIC_DIM
from .hand_tracking import HAND_DIM

RELATIVE_DIM = 9
# Musical-load windows around a query, in seconds: the current beat region,
# the coming phrase, the coming section and the section just played.
PHRASE_WINDOWS = ((-1., 1.), (0., 4.), (0., 12.), (-12., 0.))
PHRASE_DIM = 4*len(PHRASE_WINDOWS)
ALIGNMENT_WINDOW = 128
ALIGNMENT_BACK = 2
# Parameters added after Prism_stream_v9. Each is zero-initialised, so an
# older checkpoint loads with exactly its previous function.
UPGRADE_PARAMETERS = ('grounding.local_distance_key.weight','grounding.local_distance_value.weight',
                      'grounding.phrase_projection.weight','grounding.kinematic_projection.weight',
                      'grounding.alignment_distance.weight','grounding.audio_boundary_projection.weight',
                      'grounding.hand_projection.weight','grounding.dynamic_projection.weight',
                      'grounding.local_condition_projection.weight')


def relative_features(delta_seconds, bpm, scale):
    """Translation-invariant offsets: seconds, beats and sub-beat phase."""
    beats = delta_seconds*bpm/60.
    turn = 2*math.pi*beats
    return torch.stack((delta_seconds/scale, delta_seconds.abs()/scale, (beats/16.).clamp(-1, 1),
                        torch.sin(turn), torch.cos(turn), torch.sin(turn/2), torch.cos(turn/2),
                        torch.sin(turn/4), torch.cos(turn/4)), -1)


def prepare_phrase_bank(curve, times, padding):
    """Immutable prefix integrals; reused by every query during generation."""
    padding = padding.bool()
    valid = ~padding & torch.isfinite(times)
    # Keep the caller's clock precision. Training already transfers seconds
    # as FP32; promoting them back to FP64 needlessly invokes slow laptop-GPU
    # FP64 arithmetic. Native FP64 callers keep their previous precision.
    times = times.contiguous() if times.dtype in (torch.float32,torch.float64) else times.double().contiguous()
    values = curve.float().masked_fill(~valid[..., None], 0.)
    batch, total, channels = values.shape
    if total == 0:
        return {'values':values,'times':times}
    counts = valid.sum(1).long()
    last_index = (counts - 1).clamp_min(0)
    last_time = times.gather(1, last_index[:, None]).squeeze(1)
    first_time = times[:, 0]
    has_values = counts > 0
    first_time = torch.where(has_values, first_time, first_time.new_zeros(()))
    last_time = torch.where(has_values, last_time, last_time.new_zeros(()))
    safe_times = torch.where(valid, times, last_time[:, None])
    pair_valid = valid[:, :-1] & valid[:, 1:]
    delta = (safe_times[:, 1:] - safe_times[:, :-1]).clamp_min(0.) * pair_valid
    area = .5 * (values[:, :-1] + values[:, 1:]) * delta[..., None]
    cumulative = torch.cat((values.new_zeros(batch, 1, channels), area.cumsum(1)), 1)
    return dict(values=values,times=safe_times,first_time=first_time,last_time=last_time,
                last_index=last_index,cumulative=cumulative)


def query_phrase_bank(bank, seconds):
    """Evaluate all eight window boundaries in one search/gather pipeline."""
    values,times=bank['values'],bank['times']
    batch,total,channels=values.shape
    queries=seconds.size(1)
    if total == 0:
        return values.new_zeros(batch,queries,PHRASE_DIM)
    if total == 1:
        base=values[:,:1].expand(-1,queries,-1)
        return torch.cat((base,base,base,base),-1)
    seconds=seconds.to(times)
    offsets=times.new_tensor(PHRASE_WINDOWS)
    points=(seconds[...,None,None]+offsets).reshape(batch,-1).contiguous()
    first,last=bank['first_time'][:,None],bank['last_time'][:,None]
    indices=torch.searchsorted(times,points).clamp(1,total-1)
    left=indices-1
    t0,t1=times.gather(1,left),times.gather(1,indices)
    c0=bank['cumulative'].gather(1,left[...,None].expand(-1,-1,channels))
    c1=bank['cumulative'].gather(1,indices[...,None].expand(-1,-1,channels))
    ratio=((points-t0)/(t1-t0).clamp_min(1e-12)).clamp(0.,1.)
    interpolated=c0+(c1-c0)*ratio[...,None]
    last_indices=bank['last_index'][:,None,None].expand(-1,1,channels)
    final=bank['cumulative'].gather(1,last_indices)
    final=final+(points-last).clamp_min(0.)[...,None]*values.gather(1,last_indices)
    integral=torch.where((points<=first)[...,None],torch.zeros_like(interpolated),interpolated)
    integral=torch.where((points>=last)[...,None],final,integral)
    integral=integral.reshape(batch,queries,len(PHRASE_WINDOWS),2,channels)
    duration=(offsets[:,1]-offsets[:,0])[None,None,:,None]
    return ((integral[:,:,:,1]-integral[:,:,:,0])/duration).reshape(batch,queries,PHRASE_DIM)


def phrase_features(curve, times, padding, seconds):
    """Time-weighted pooling on the native nonuniform audio clock."""
    return query_phrase_bank(prepare_phrase_bank(curve,times,padding),seconds)


class GroundedContext(nn.Module):
    def __init__(self,config):
        super().__init__();self.config=config
        width=config.d_model;key=min(64,width)
        self.clock_projection=nn.Linear(CLOCK_DIM,width,bias=False)
        self.local_query=nn.Linear(CLOCK_DIM,key,bias=False)
        self.local_key=nn.Linear(config.audio_dim,key,bias=False)
        self.local_value=nn.Sequential(nn.Linear(config.audio_dim,width),nn.LayerNorm(width),nn.GELU())
        self.local_input=nn.Linear(width,width,bias=False)
        self.action_value=nn.Sequential(nn.Linear(51,width),nn.LayerNorm(width),nn.GELU())
        self.archive_value=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width),nn.GELU())
        self.archive_distance=nn.Linear(3,width,bias=False)
        self.router=nn.Sequential(nn.Linear(width*2, max(32,width//2)),nn.GELU(),nn.Linear(max(32,width//2),5))
        self.fusion=nn.Linear(width,width,bias=False)
        self.intent_projection=nn.Linear(config.plan_dim,width,bias=False)
        self.alignment_query=nn.Linear(width,key,bias=False)
        self.alignment_key=nn.Linear(config.audio_dim,key,bias=False)
        self.selection_head=nn.Sequential(nn.Linear(width+config.condition_dim,key),nn.GELU(),nn.Linear(key,1))
        self.local_distance_key=nn.Linear(RELATIVE_DIM,key,bias=False)
        self.local_distance_value=nn.Linear(RELATIVE_DIM,width,bias=False)
        self.phrase_projection=nn.Linear(PHRASE_DIM,width,bias=False)
        self.kinematic_projection=nn.Linear(KINEMATIC_DIM,width,bias=False)
        self.hand_projection=nn.Linear(HAND_DIM,width,bias=False)
        self.alignment_distance=nn.Linear(RELATIVE_DIM,key,bias=False)
        # A direct event-boundary audio residual prevents the decoder from
        # dropping every acoustic route through the learned router. It starts
        # at zero so old checkpoints keep their exact function until finetuned.
        self.audio_boundary_projection=nn.Linear(width,width,bias=False)
        # Keep absolute dynamics available after the main audio LayerNorm.
        # RMS, relative RMS, slope and novelty are already measured channels;
        # this residual lets the decoder use them without undoing the stable
        # normalized acoustic representation.
        self.dynamic_projection=nn.Linear(4,width,bias=False)
        # Local difficulty/style controls are a weak residual.  Zero init keeps
        # old checkpoints behaviourally compatible until finetuned.
        self.local_condition_projection=nn.Linear(config.condition_dim,width,bias=False)
        # Explicit upgrades start from the old logits. New branches learn
        # through residual outputs, with unchanged legacy parameter shapes.
        for layer in (self.clock_projection,self.local_input,self.fusion,self.intent_projection,
                      self.local_distance_key,self.local_distance_value,self.phrase_projection,
                      self.kinematic_projection,self.alignment_distance,
                      self.audio_boundary_projection,self.hand_projection,
                      self.dynamic_projection,self.local_condition_projection):
            nn.init.zeros_(layer.weight)

    def prepare_bank(self,values,times,padding):
        mask=padding.bool()
        return {'values':values,'times':times,'padding':mask,
                'keys':self.local_key(values),'projected':self.local_value(values),
                'dynamic':self.dynamic(values),
                'audio_mean':(values[...,:98]*(~mask)[...,None]).sum(1)/(~mask).sum(1).clamp_min(1)[:,None]}

    def phrase(self,curve,times,padding,seconds):
        return self.phrase_projection(phrase_features(curve.detach().sigmoid(),times,padding.bool(),seconds).to(
            self.phrase_projection.weight.dtype))

    def phrase_from_bank(self,bank,seconds):
        return self.phrase_projection(query_phrase_bank(bank,seconds).to(self.phrase_projection.weight.dtype))

    def kinematics(self,values):
        return self.kinematic_projection(values.to(self.kinematic_projection.weight.dtype))

    def hands(self,values):
        return self.hand_projection(values.to(self.hand_projection.weight.dtype))

    def boundary_audio(self,local):
        return self.audio_boundary_projection(local.to(self.audio_boundary_projection.weight.dtype))

    def dynamic(self,values):
        if values.size(-1) < 98:
            raise ValueError('Dynamic audio bypass requires the 98-column acoustic block')
        dynamic=values[..., (93,95,96,97)]
        return self.dynamic_projection(dynamic.to(self.dynamic_projection.weight.dtype))

    def local_condition(self,condition):
        return self.local_condition_projection(condition.to(self.local_condition_projection.weight.dtype))

    def local(self,clock,seconds,center,bank):
        values,times,padding=bank['values'],bank['times'],bank['padding']
        batch,count=seconds.shape
        total=times.size(1);width=self.config.local_audio_neighbors
        # Leading pre-audio nodes must keep their sorted native times.
        searchable=times.contiguous()
        right=torch.searchsorted(searchable,seconds.contiguous())
        if width<=1:
            raw=right[...,None]
        else:
            # Span [-0.25*radius, +radius] so dense detail grids still observe
            # upcoming beats instead of collapsing to ~0.2s around `right`.
            radius=float(self.config.local_audio_seconds)
            back_count=max(1,width//4);fwd_count=width-back_count
            low=torch.searchsorted(searchable,(seconds-.25*radius).contiguous())
            high=(torch.searchsorted(searchable,(seconds+radius).contiguous(),right=True)-1).clamp_min(0)
            back_excess=((right-low)-back_count).clamp_min(0)[...,None]
            fwd_excess=((high-right)-max(1,fwd_count-1)).clamp_min(0)[...,None]
            b=torch.arange(back_count,0,-1,device=times.device)
            f=torch.arange(fwd_count,device=times.device)
            back_offsets=-b-torch.round((b/max(1,back_count))**1.6*back_excess).long()
            fwd_offsets=f+torch.round((f/max(1,fwd_count-1))**1.6*fwd_excess).long()
            raw=right[...,None]+torch.cat((back_offsets,fwd_offsets),-1)
        indices=raw.clamp(0,max(0,total-1))
        rows=torch.arange(batch,device=times.device)[:,None,None]
        sample_times=times[rows,indices]
        mask=(raw<0)|(raw>=total)|padding[rows,indices]
        mask|=(sample_times-seconds[...,None]).abs()>self.config.local_audio_seconds
        keys=bank['keys'][rows,indices];features=bank['projected'][rows,indices]
        dynamic=bank['dynamic'][rows,indices]
        center_keys=self.local_key(center);center_values=self.local_value(center)
        center_dynamic=self.dynamic(center)
        keys=torch.cat((keys,center_keys[:,:,None]),2)
        features=torch.cat((features,center_values[:,:,None]),2)
        dynamic=torch.cat((dynamic,center_dynamic[:,:,None]),2)
        # Signed offsets let the query tell the coming onset from the one a
        # beat earlier; beat phase alone repeats every beat inside the window.
        delta=torch.cat(((sample_times-seconds[...,None]).masked_fill(mask,0.),
                         sample_times.new_zeros(batch,count,1)),2).float()
        bpm=(clock[...,19].float()*300.).clamp_min(1.)[...,None]
        relative=relative_features(delta,bpm,self.config.local_audio_seconds).to(keys.dtype)
        keys=keys+self.local_distance_key(relative)
        features=features+self.local_distance_value(relative).to(features.dtype)
        center_mask=(clock[...,14]<0)|(clock[...,14]>=1)
        mask=torch.cat((mask,center_mask[...,None]),2)
        # Always offer an explicit null observation, including before audio.
        keys=torch.cat((keys,keys.new_zeros(batch,count,1,keys.size(-1))),2)
        features=torch.cat((features,features.new_zeros(batch,count,1,features.size(-1))),2)
        dynamic=torch.cat((dynamic,dynamic.new_zeros(batch,count,1,dynamic.size(-1))),2)
        mask=torch.cat((mask,mask.new_zeros(batch,count,1)),2)
        query=self.local_query(clock)
        scores=(query[:,:,None].float()*keys.float()).sum(-1)/math.sqrt(keys.size(-1))
        scores=scores.masked_fill(mask,-1e4)
        weights=scores.softmax(-1).to(features.dtype)
        return (weights[...,None]*features).sum(2)+(weights[...,None]*dynamic).sum(2)

    def archive_embeddings(self,batch,embedding):
        shape=batch['archive_padding_mask'].shape
        if not shape[1]:return embedding.weight.new_zeros((*shape,self.config.d_model))
        tokens=batch['archive_flat_tokens'];owner=batch['archive_token_event']
        position=batch['archive_token_position'].to(embedding.weight.dtype)
        values=embedding(tokens)
        # Ordered moments preserve direction (1-5 differs from 5-1) without
        # padding every archived event to the longest Slide in the song.
        frequencies=torch.arange(1,values.size(-1)+1,device=values.device,dtype=values.dtype)
        weights=1.+torch.sin(position[:,None]*frequencies[None]*math.pi)
        values=values*weights*batch['archive_token_inverse_length'].to(values)[:,None]
        pooled=values.new_zeros(shape[0]*shape[1],values.size(-1))
        pooled.index_add_(0,owner,values)
        return self.archive_value(pooled.reshape(*shape,-1))

    def retrieve(self,clock,seconds,center,archive,bank):
        batch,queries,_=center.shape
        values=archive['embeddings'];count=values.size(1)
        if not count:
            return center.new_zeros(batch,queries,self.config.d_model),center.new_zeros(batch,queries)
        mean=bank.get('audio_mean')
        if mean is None:
            valid=~bank['padding']
            mean=(bank['values'][...,:98]*valid[...,None]).sum(1)/valid.sum(1).clamp_min(1)[:,None]
        query=F.normalize((center[...,:98]-mean[:,None]).float(),dim=-1,eps=1e-6)
        keys=F.normalize((archive['audio'][...,:98]-mean[:,None]).float(),dim=-1,eps=1e-6)
        similarity=query@keys.transpose(1,2)
        causal=(archive['event_index'][:,None]<torch.round(clock[...,17]*1024).long()[...,None])
        mask=archive['padding'][:,None]|~causal
        similarity=similarity.masked_fill(mask,-1e4)
        k=min(self.config.archive_retrievals,count)
        scores,indices=similarity.topk(k,dim=-1)
        rows=torch.arange(batch,device=seconds.device)[:,None,None]
        retrieved=values[rows,indices]
        relative=torch.stack(((archive['beats'][rows,indices]-clock[...,0,None]*64)/64,
            (archive['seconds'][rows,indices]-seconds[...,None])/20,
            (archive['event_index'][rows,indices]-clock[...,17,None]*1024)/128),-1).to(retrieved)
        retrieved=retrieved+self.archive_distance(relative)
        eligible=scores>=self.config.archive_min_similarity
        # Null stays available even with good matches. No template is forced.
        scores=torch.cat((scores*4,scores.new_full((batch,queries,1),4*self.config.archive_min_similarity)),2)
        scores[:,:,:k]=scores[:,:,:k].masked_fill(~eligible,-1e4)
        retrieved=torch.cat((retrieved,retrieved.new_zeros(batch,queries,1,retrieved.size(-1))),2)
        weights=scores.softmax(-1).to(retrieved.dtype)
        return (weights[...,None]*retrieved).sum(2),eligible.any(-1).float()

    def contexts(self,clock,seconds,center,bank,archive,condition=None):
        local=self.local(clock,seconds,center,bank)
        recalled,available=self.retrieve(clock,seconds,center,archive,bank)
        if condition is not None:
            # Keep difficulty/style as a secondary control. Audio and its
            # authored pointer remain the dominant local evidence path.
            local=local+.25*self.local_condition(condition)[:,None]
        return {'local':local,'archive':recalled,'archive_available':available}

    def input(self,tokens,clock,local):
        return tokens+self.clock_projection(clock.to(tokens))+self.local_input(local.to(tokens))

    def fuse(self,hidden,clock,local,archive,global_context,action):
        query=self.clock_projection(clock.to(hidden))
        routes=self.router(torch.cat((hidden,query),-1)).float().softmax(-1).to(hidden)
        evidence=(routes[...,0:1]*local+routes[...,1:2]*archive
                  +routes[...,2:3]*global_context[:,None]+routes[...,3:4]*action)
        # Fifth route is innovation: no retrieved or local residual forced.
        return hidden+self.fusion(evidence),routes

    def alignment(self,hidden,batch,bank=None):
        """Next-onset pointer from the event boundary state.

        A boundary row has consumed the previous event only; the label is the
        detail node of the event about to be written. Candidates are the nodes
        just after the current clock, keyed by their relative offset, which is
        the decision the following position tokens actually make.
        """
        indices=batch['alignment_event_indices'].clamp_min(0)
        boundary=hidden.gather(1,indices[...,None].expand(-1,-1,hidden.size(-1)))
        rows=batch['decoder_query_index'].clamp_min(0).gather(1,indices)
        seconds=batch['query_seconds'].gather(1,rows)
        bpm=(batch['query_clock'][...,19].gather(1,rows).float()*300.).clamp_min(1.)
        scores,start,_=self.alignment_window(boundary,seconds,bpm,batch['detail_audio'],
            batch['detail_times_sec'],batch['detail_audio_padding_mask'].bool())
        width=scores.size(-1)
        labels=batch['alignment_audio_index']-start
        labels=torch.where((labels>=0)&(labels<width),labels,torch.full_like(labels,-1))
        control=batch['condition'][:,None].expand(-1,batch['detail_audio'].size(1),-1)
        if batch.get('condition_null') is not None:
            control=control.clone()
            control[...,1:]=torch.where(batch['condition_null'][:,None,None],.5,control[...,1:])
        # Share the acoustic value encoder used by the actual continuation.
        evidence=bank['projected'] if bank is not None else self.local_value(batch['detail_audio'])
        selection=self.selection_head(torch.cat((evidence,control.to(evidence)),-1)).squeeze(-1)
        return {'alignment_event_logits':scores,'alignment_event_labels':labels,
                'position_alignment_logits':selection}

    def alignment_window(self,boundary,seconds,bpm,values,times,padding,*,full_keys=None):
        """The same trained boundary pointer for supervision and inference.

        Inference must pass the previous END/BOS hidden row, before emitting AT;
        substituting a hidden row after the candidate NOTE would know its label.
        """
        query=self.alignment_query(boundary)
        total=times.size(1);width=min(ALIGNMENT_WINDOW,total)
        start=(torch.searchsorted(times.contiguous(),seconds.contiguous())-ALIGNMENT_BACK).clamp(0,max(0,total-width))
        candidates=start[...,None]+torch.arange(width,device=times.device)
        batch_rows=torch.arange(times.size(0),device=times.device)[:,None,None]
        if full_keys is None:full_keys=self.alignment_key(values)
        keys=full_keys[batch_rows,candidates]
        delta=(times[batch_rows,candidates]-seconds[...,None])
        invalid=padding[batch_rows,candidates]
        relative=relative_features(delta.masked_fill(invalid,0.).float(),bpm[...,None],4.).to(keys.dtype)
        keys=keys+self.alignment_distance(relative)
        scores=(query[:,:,None].float()*keys.float()).sum(-1)/math.sqrt(keys.size(-1))
        scores=scores.masked_fill(invalid,-1e4)
        return scores,start,candidates
