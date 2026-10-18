"""Query-local acoustics and causal chart archives without quadratic bank encoding."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from .grounding import CLOCK_DIM


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
        # Explicit upgrades start from the old logits. New branches learn
        # through residual outputs, with unchanged legacy parameter shapes.
        for layer in (self.clock_projection,self.local_input,self.fusion,self.intent_projection):
            nn.init.zeros_(layer.weight)

    def prepare_bank(self,values,times,padding):
        mask=padding.bool()
        return {'values':values,'times':times,'padding':mask,
                'keys':self.local_key(values),'projected':self.local_value(values)}

    def local(self,clock,seconds,center,bank):
        values,times,padding=bank['values'],bank['times'],bank['padding']
        batch,count=seconds.shape
        total=times.size(1);width=self.config.local_audio_neighbors
        # Leading pre-audio nodes must keep their sorted native times.
        searchable=times.contiguous()
        right=torch.searchsorted(searchable,seconds.contiguous())
        offsets=torch.arange(width,device=times.device)-width//2
        raw=right[...,None]+offsets
        indices=raw.clamp(0,max(0,total-1))
        rows=torch.arange(batch,device=times.device)[:,None,None]
        sample_times=times[rows,indices]
        mask=(raw<0)|(raw>=total)|padding[rows,indices]
        mask|=(sample_times-seconds[...,None]).abs()>self.config.local_audio_seconds
        keys=bank['keys'][rows,indices];features=bank['projected'][rows,indices]
        center_keys=self.local_key(center);center_values=self.local_value(center)
        keys=torch.cat((keys,center_keys[:,:,None]),2)
        features=torch.cat((features,center_values[:,:,None]),2)
        center_mask=(clock[...,14]<0)|(clock[...,14]>=1)
        mask=torch.cat((mask,center_mask[...,None]),2)
        # Always offer an explicit null observation, including before audio.
        keys=torch.cat((keys,keys.new_zeros(batch,count,1,keys.size(-1))),2)
        features=torch.cat((features,features.new_zeros(batch,count,1,features.size(-1))),2)
        mask=torch.cat((mask,mask.new_zeros(batch,count,1)),2)
        query=self.local_query(clock)
        scores=(query[:,:,None].float()*keys.float()).sum(-1)/math.sqrt(keys.size(-1))
        scores=scores.masked_fill(mask,-1e4)
        weights=scores.softmax(-1).to(features.dtype)
        return (weights[...,None]*features).sum(2)

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

    def contexts(self,clock,seconds,center,bank,archive):
        local=self.local(clock,seconds,center,bank)
        recalled,available=self.retrieve(clock,seconds,center,archive,bank)
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
        indices=batch['alignment_event_indices'].clamp_min(0)
        query=self.alignment_query(hidden.gather(1,indices[...,None].expand(-1,-1,hidden.size(-1))))
        keys=self.alignment_key(batch['detail_audio'])
        scores=(query@keys.transpose(1,2))/math.sqrt(keys.size(-1))
        scores=scores.masked_fill(batch['detail_audio_padding_mask'][:,None],-1e4)
        control=batch['condition'][:,None].expand(-1,batch['detail_audio'].size(1),-1)
        if batch.get('condition_null') is not None:
            control=control.clone()
            control[...,1:]=torch.where(batch['condition_null'][:,None,None],.5,control[...,1:])
        # Share the acoustic value encoder used by the actual continuation.
        # An isolated head on frozen input features would train only itself.
        evidence=bank['projected'] if bank is not None else self.local_value(batch['detail_audio'])
        selection=self.selection_head(torch.cat((evidence,control.to(evidence)),-1)).squeeze(-1)
        return {'alignment_event_logits':scores,'position_alignment_logits':selection}
