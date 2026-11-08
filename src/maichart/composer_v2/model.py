"""One causal component stream with local music and whole-song attention.

Attention-probability dropout is zero; residual/FF dropout supplies regularization
without the BF16 attention dropout cancellation encountered in the old family.
"""
from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ComposerV2Config:
    d_model: int = 384
    layers: int = 8
    nhead: int = 8
    dim_feedforward: int = 1536
    dropout: float = .1
    audio_layers: int = 2
    song_layers: int = 2
    window_beats: int = 16
    history_beats: int = 16
    future_beats: int = 4
    ticks_per_beat: int = 48
    max_tokens: int = 2048
    audio_backend: str = 'mert'
    mert_model: str | None = None
    condition_policy: str = 'radar_style_v1'
    sequence_policy: str = 'continuous_events_v1'
    refresh_events: int = 16
    onset_readout_policy: str = 'off'
    structure_policy: str = 'off'
    def __post_init__(self):
        if self.d_model%self.nhead or min(self.d_model,self.layers,self.nhead,self.dim_feedforward,self.window_beats,self.max_tokens)<1:
            raise ValueError('Invalid Composer v2 dimensions')
        if self.audio_backend not in ('mert','native') or not 0<=self.dropout<1:
            raise ValueError('Invalid Composer v2 audio/dropout contract')
        if self.condition_policy not in ('level_only_v1','radar_style_v1'):
            raise ValueError('Invalid Composer v2 condition policy')
        if self.sequence_policy not in ('window_v1','continuous_events_v1') or self.refresh_events<1:raise ValueError('Invalid stream policy')
        from .onset_policy import POLICY
        if self.onset_readout_policy not in ('off', POLICY) or (self.onset_readout_policy != 'off' and self.sequence_policy != 'continuous_events_v1'):
            raise ValueError('Phase audio readout requires continuous events')
        from .rhythm_structure import POLICIES
        if self.structure_policy not in POLICIES or (self.structure_policy != 'off' and self.sequence_policy != 'continuous_events_v1'):
            raise ValueError('Rhythm structure requires continuous events')
    def to_dict(self):return asdict(self)
    @classmethod
    def from_dict(cls,value):return cls(**{'condition_policy':'level_only_v1','sequence_policy':'window_v1',**value})


def positions(values,width):
    scale=torch.exp(torch.arange(0,width,2,device=values.device,dtype=torch.float32)*(-math.log(10000.)/width))
    angle=values.float()[...,None]*scale
    return torch.stack((angle.sin(),angle.cos()),-1).flatten(-2)[...,:width]


class Attention(nn.Module):
    def __init__(self,width,heads,*,local=False):
        super().__init__();self.heads=heads;self.dim=width//heads;self.local=local
        self.q=nn.Linear(width,width);self.k=nn.Linear(width,width);self.v=nn.Linear(width,width);self.out=nn.Linear(width,width)
        if local:
            self.past=nn.Parameter(torch.full((heads,),-1.5));self.future=nn.Parameter(torch.full((heads,),-1.5))
    def split(self,value):return value.reshape(value.size(0),value.size(1),self.heads,self.dim).transpose(1,2)
    def memory(self,value):return self.split(self.k(value)),self.split(self.v(value))
    def forward(self,query,memory=None,*,kv=None,key_mask=None,causal=False,past_length=0,query_beats=None,key_beats=None):
        q=self.split(self.q(query));k,v=self.memory(memory) if kv is None else kv
        mask=None
        if key_mask is not None:mask=key_mask[:,None,None,:]
        if causal:
            allowed=torch.arange(k.size(-2),device=q.device)[None,:] <= (torch.arange(q.size(-2),device=q.device)[:,None]+past_length)
            mask=allowed[None,None] if mask is None else mask&allowed[None,None]
        if self.local:
            gap=key_beats[:,None,:]-query_beats[:,:,None]
            bias=-(F.softplus(self.past)[None,:,None,None]*(-gap[:,None]).clamp_min(0)
                   +F.softplus(self.future)[None,:,None,None]*gap[:,None].clamp_min(0))
            mask=bias.masked_fill(~mask,-torch.inf) if mask is not None else bias
            mask=mask.to(q.dtype)
        attended=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0.)
        return self.out(attended.transpose(1,2).reshape(query.shape))


class EncoderBlock(nn.Module):
    def __init__(self,c):
        super().__init__();self.norm1=nn.LayerNorm(c.d_model);self.norm2=nn.LayerNorm(c.d_model)
        self.attention=Attention(c.d_model,c.nhead)
        self.ff=nn.Sequential(nn.Linear(c.d_model,c.dim_feedforward),nn.GELU(),nn.Dropout(c.dropout),nn.Linear(c.dim_feedforward,c.d_model))
        self.drop=nn.Dropout(c.dropout)
    def forward(self,x,mask):
        value=self.norm1(x);x=x+self.drop(self.attention(value,value,key_mask=mask))
        return (x+self.drop(self.ff(self.norm2(x))))*mask[...,None]


def query_groups(tile_ids,start,device):
    width=len(tile_ids[0])-start;groups={}
    for row,ids in enumerate(tile_ids):
        for column,tile in enumerate(ids[start:]):
            if tile>=0:groups.setdefault(tile,[]).append(row*width+column)
    pieces=[(tile,indices[i:i+32]) for tile,indices in groups.items() for i in range(0,len(indices),32)]
    count=max(len(indices) for _,indices in pieces);positions=[];valid=[]
    for _,indices in pieces:
        positions.append(indices+[0]*(count-len(indices)));valid.append([True]*len(indices)+[False]*(count-len(indices)))
    return (torch.tensor([tile for tile,_ in pieces],device=device),torch.tensor(positions,device=device),torch.tensor(valid,device=device))


class DecoderBlock(nn.Module):
    def __init__(self,c,*,song=True):
        super().__init__();self.norms=nn.ModuleList([nn.LayerNorm(c.d_model) for _ in range(4)])
        self.self_attention=Attention(c.d_model,c.nhead);self.local_attention=Attention(c.d_model,c.nhead,local=True)
        self.song_attention=Attention(c.d_model,c.nhead) if song else None
        self.ff=nn.Sequential(nn.Linear(c.d_model,c.dim_feedforward),nn.GELU(),nn.Dropout(c.dropout),nn.Linear(c.dim_feedforward,c.d_model))
        self.drop=nn.Dropout(c.dropout)
    def forward(self,x,condition,token_mask,beats,audio,*,state=None,groups=None):
        value=self.norms[0](x+condition[:,None]);kv=self.self_attention.memory(value)
        old_length=0
        if state is not None:
            if 'self_kv' in state:
                old_length=state['self_kv'][0].size(-2)
                kv=tuple(torch.cat((old,new),dim=-2) for old,new in zip(state['self_kv'],kv))
            state['self_kv']=kv
        x=x+self.drop(self.self_attention(value,kv=kv,key_mask=token_mask,causal=True,past_length=old_length))
        local_kv=state.get('local_kv') if state is not None else None
        if local_kv is not None and local_kv[0].size(0)!=audio['local'].size(0):local_kv=None
        if local_kv is None:
            local_kv=self.local_attention.memory(audio['local'])
            if state is not None:state['local_kv']=local_kv
        query=self.norms[1](x+condition[:,None])
        if groups is None:
            attended=self.local_attention(query,kv=local_kv,key_mask=audio['local_mask'],query_beats=beats,key_beats=audio['local_beats'])
        else:
            tiles,indices,valid=groups
            packed=query.flatten(0,1).index_select(0,indices.flatten()).reshape(*indices.shape,query.size(-1))
            attended=self.local_attention(packed,kv=tuple(v.index_select(0,tiles) for v in local_kv),
                key_mask=audio['local_mask'][tiles],query_beats=beats.flatten()[indices],key_beats=audio['local_beats'][tiles])
            attended=(attended*valid[...,None]).flatten(0,1)
            attended=attended.new_zeros(query.numel()//query.size(-1),query.size(-1)).index_add(0,indices.flatten(),attended).reshape_as(query)
        x=x+self.drop(attended)
        if self.song_attention is not None:
            song_kv=state.get('song_kv') if state is not None else None
            if song_kv is None:
                song_kv=self.song_attention.memory(audio['song'])
                if state is not None:state['song_kv']=song_kv
            x=x+self.drop(self.song_attention(self.norms[2](x+condition[:,None]),kv=song_kv,key_mask=audio['song_mask']))
        return x+self.drop(self.ff(self.norms[3](x)))


class ComposerV2(nn.Module):
    def __init__(self,config,codec):
        super().__init__();self.config=config;self.codec=codec;c=config;d=c.d_model
        self.embedding=nn.Embedding(codec.vocab_size,d,padding_idx=codec.pad_id)
        nn.init.normal_(self.embedding.weight,std=.02)
        with torch.no_grad():self.embedding.weight[codec.pad_id].zero_()
        self.condition=nn.Sequential(nn.Linear(16,d),nn.SiLU(),nn.Linear(d,d))
        if c.condition_policy=='radar_style_v1':
            self.condition_missing=nn.Linear(16,d,bias=False)
            nn.init.zeros_(self.condition_missing.weight)
        self.mel=nn.Sequential(nn.LayerNorm(98),nn.Linear(98,d))
        self.semantic=nn.Sequential(nn.LayerNorm(1024),nn.Linear(1024,d,bias=False))
        self.local_conv=nn.Conv1d(d,d,kernel_size=3,stride=2,padding=1)
        self.audio_blocks=nn.ModuleList([EncoderBlock(c) for _ in range(c.audio_layers)])
        self.song_blocks=nn.ModuleList([EncoderBlock(c) for _ in range(c.song_layers)])
        self.decoder=nn.ModuleList([DecoderBlock(c,song=(i%2==1)) for i in range(c.layers)])
        self.norm=nn.LayerNorm(d);self.dropout=nn.Dropout(c.dropout)
        self.output_bias=nn.Parameter(torch.zeros(codec.vocab_size))
        self._song_cache={}
        self._structured_weight_cache={}
        self.register_buffer('phase_targets',torch.tensor([t.startswith('F:') for t in codec.tokens]),persistent=False)
        self.register_buffer('time_tokens',torch.tensor([t.startswith(('T:','F:')) or t in ('[AT]','[PHASE]','[TIME_EXT]','[TIME_END]') or t.startswith('NUM:') for t in codec.tokens]),persistent=False)
        if c.onset_readout_policy != 'off':
            from .onset_readout import PhaseAudioReadout
            with torch.random.fork_rng(devices=[]):
                self.onset_readout = PhaseAudioReadout()
            self.register_buffer('phase_token_ids', torch.tensor(sorted(codec.phases)), persistent=False)
        if c.structure_policy != 'off':
            # Zero-initialized residual inputs: an upgraded parent starts from
            # exactly its original function and learns only what finetune adds.
            from .rhythm_structure import PHASE_FEATURES,TRUST_FEATURES,ENVELOPE_FEATURES,phase_token_table
            with torch.random.fork_rng(devices=[]):
                self.phase_position=nn.Linear(PHASE_FEATURES,d,bias=False)
                self.audio_phase_position=nn.Linear(PHASE_FEATURES,d,bias=False)
                self.phase_token=nn.Linear(PHASE_FEATURES,d,bias=False)
                self.audio_envelope=nn.Linear(ENVELOPE_FEATURES,d,bias=False)
                self.reference_trust=nn.Linear(TRUST_FEATURES,d,bias=False)
                self.segment=nn.Embedding(5,d)
            for module in (self.phase_position,self.audio_phase_position,self.phase_token,
                           self.audio_envelope,self.reference_trust,self.segment):
                nn.init.zeros_(module.weight)
            from .rhythm_structure import HARMONICS, EVIDENCE_FEATURES, POLICY as LATEST
            self.register_buffer('phase_harmonics',torch.tensor(HARMONICS,dtype=torch.float32),persistent=False)
            if c.structure_policy==LATEST:
                with torch.random.fork_rng(devices=[]):
                    self.reference_evidence=nn.Linear(EVIDENCE_FEATURES,d,bias=False)
                nn.init.zeros_(self.reference_evidence.weight)
            self.register_buffer('phase_token_features',phase_token_table(codec),persistent=False)
            code=torch.zeros(codec.vocab_size,dtype=torch.long)
            for value,name in ((1,'carry_id'),(2,'ctx_id'),(3,'ref_id'),(4,'tgt_id')):code[getattr(codec,name)]=value
            self.register_buffer('segment_code',code,persistent=False)
    def token_weight(self):
        """Tied input/output table; phase tokens share subdivision-family structure."""
        device_type=self.embedding.weight.device.type
        key=(device_type,torch.get_autocast_dtype(device_type) if torch.is_autocast_enabled(device_type) else None)
        cacheable=not self.training and not torch.is_grad_enabled()
        if cacheable and key in self._structured_weight_cache:return self._structured_weight_cache[key]
        weight=self.embedding.weight
        if hasattr(self,'phase_token'):
            weight=weight+self.phase_token(self.phase_token_features).to(weight.dtype)
        if cacheable:self._structured_weight_cache[key]=weight
        return weight
    def train(self,mode=True):
        self._song_cache.clear();self._structured_weight_cache.clear()
        return super().train(mode)
    def _apply(self,fn,recurse=True):
        self._song_cache.clear();self._structured_weight_cache.clear()
        return super()._apply(fn,recurse=recurse)
    def load_state_dict(self,*args,**kwargs):
        self._song_cache.clear();self._structured_weight_cache.clear()
        return super().load_state_dict(*args,**kwargs)
    def _audio_input(self,mel,semantic,valid,keep):
        return self.mel(mel.float())+self.semantic(semantic.float())*(valid&keep[:,None])[...,None]
    def encode_audio(self,batch):
        # Training encodes the batch once. Inference shares immutable song
        # memory and each window's local encoding across all candidate branches.
        mel=batch.get('bank_audio',batch['local_audio']);b=batch['input_ids'].size(0)
        keep=torch.ones(b,device=mel.device,dtype=torch.bool)
        if self.training:keep=torch.rand(b,device=mel.device)>=.15
        owners=batch.get('tile_owner')
        if 'bank_audio' in batch:
            semantic=batch['bank_semantic'].float()
            if 'bank_semantic_right' in batch:
                fraction=batch['bank_semantic_weight'][:,None]
                semantic=(semantic.double()*(1.-fraction)
                          +batch['bank_semantic_right'].float().double()*fraction).float()
            bank=self.mel(mel.float())+self.semantic(semantic)*(batch['bank_semantic_mask']&keep[batch['bank_owner']])[:,None]
            if hasattr(self,'audio_envelope'):bank=bank+self.audio_envelope(batch['bank_envelope'].float())
            frames=batch['tile_frames']
            local=bank.index_select(0,frames.flatten()).reshape(*frames.shape,bank.size(-1))
        else:
            local=self._audio_input(mel,batch['local_semantic'],batch['local_semantic_mask'],keep if owners is None else keep[owners])
            if hasattr(self,'audio_envelope'):local=local+self.audio_envelope(batch['local_envelope'].float())
        local=local+positions(batch['local_beats'],self.config.d_model)
        if hasattr(self,'audio_phase_position'):
            from .rhythm_structure import phase_features
            local=local+self.audio_phase_position(phase_features(batch['local_beats'],self.phase_harmonics))
        local=local*batch['local_mask'][...,None]
        local=self.local_conv(local.transpose(1,2)).transpose(1,2)
        mask=F.max_pool1d(batch['local_mask'].float()[:,None],3,2,1).squeeze(1).bool()
        beats=batch['local_beats'][:,::2]
        # An all-padding edge query has one neutral key, avoiding undefined attention.
        mask=mask.clone();mask[:,0]|=~mask.any(-1)
        for layer in self.audio_blocks:local=layer(local,mask)
        ids=batch.get('song_cache_key',[])
        key=ids[0] if not self.training and not torch.is_grad_enabled() and len(ids)==1 else None
        cached=self._song_cache.get(key) if key is not None else None
        if cached is None:
            song=self._audio_input(batch['song_audio'],batch['song_semantic'],batch['song_semantic_mask'],keep)
            starts=torch.tensor([float(s) for s in batch.get('start_beat',[0.]*b)],device=mel.device)
            song=song+positions(batch['song_beats']+starts[:,None],self.config.d_model)
            song_mask=batch['song_mask'].clone();song_mask[:,0]|=~song_mask.any(-1)
            for layer in self.song_blocks:song=layer(song,song_mask)
            if key is not None:
                if len(self._song_cache)>=4:self._song_cache.clear()
                self._song_cache[key]=(song.detach(),song_mask)
        else:song,song_mask=cached
        result = dict(local=local,local_mask=mask,local_beats=beats,song=song,song_mask=song_mask)
        if hasattr(self, 'onset_readout') and batch.get('onset_strength', 1.) != 0:
            result['onset_scores'] = self.onset_readout(batch)
        return result
    @staticmethod
    def reset_decode_cache(cache):cache.pop('decode_state',None)
    @staticmethod
    def rewind_decode_cache(cache,prefix_length):
        state=cache.get('decode_state')
        if state is None:return
        # Recompute the final retained token to obtain its next-token logits.
        length=min(state['length'],max(0,int(prefix_length)-1))
        for layer in state['layers']:
            if 'self_kv' in layer:
                layer['self_kv']=tuple(value[:,:,:length] for value in layer['self_kv'])
        state['length']=length
        state.pop('last_segment',None)
    def condition_context(self,batch):
        values=batch['condition'].float()
        if not hasattr(self,'condition_missing'):return self.condition(values)
        known=batch.get('condition_known')
        if known is None:
            known=torch.zeros_like(values,dtype=torch.bool);known[:,0]=True
        else:known=known.bool().clone()
        if self.training:
            # Train the same unspecified/partial controls used by generation.
            # They request music-led choices, rather than pretending zero means automatic.
            keep=(torch.rand(values.size(0),1,device=values.device)>=.25)
            keep=keep&(torch.rand(values.shape,device=values.device)>=.15)
            keep[:,0]=True;known=known&keep
        values=values*known
        return self.condition(values)+self.condition_missing((~known).float())
    def decode(self,batch,cache,*,incremental=True):
        ids=batch['input_ids'];length=ids.size(1)
        use_cache=incremental and not self.training and not torch.is_grad_enabled()
        state=cache.get('decode_state') if use_cache else None
        if state is None or length<=state['length']:
            state={'length':0,'layers':[{} for _ in self.decoder]}
        start=state['length'] if use_cache else 0
        value=ids[:,start:];beats=batch['token_beats'][:,start:]
        index=torch.arange(start,length,device=ids.device)[None].expand_as(value)
        starts=torch.tensor([float(s) for s in batch.get('start_beat',[0.]*ids.size(0))],device=ids.device)
        structured=hasattr(self,'phase_position')
        weight=self.token_weight() if structured else self.embedding.weight
        x=(F.embedding(value,weight,padding_idx=self.codec.pad_id) if structured else self.embedding(value))*math.sqrt(self.config.d_model)
        x=x+positions(index,self.config.d_model)+positions(beats,self.config.d_model)
        if self.config.sequence_policy=='window_v1':x=x+positions(beats+starts[:,None],self.config.d_model)
        if structured:
            from .rhythm_structure import phase_features
            # Sub-beat phase of each token's clock, its stream region (carry,
            # history, aligned reference, target) and the reference trust.
            x=x+self.phase_position(phase_features(beats,self.phase_harmonics))
            codes=self.segment_code[value]
            if start and 'last_segment' in state:
                segment=torch.maximum(codes.cummax(1).values,state['last_segment'][:,None])
            elif start:segment=torch.maximum(codes.cummax(1).values,self.segment_code[ids[:,:start]].amax(1)[:,None])
            else:segment=codes.cummax(1).values
            if use_cache:state['last_segment']=segment[:,-1]
            x=x+self.segment(segment)
            trust=self.reference_trust(batch['reference_trust'].float())
            x=x+trust[:,None,:]*(segment==3)[...,None].to(trust.dtype)
        if hasattr(self,'reference_evidence'):
            evidence=batch.get('reference_evidence')
            if evidence is not None:
                offset=int(batch.get('reference_evidence_start',0))
                x=x+self.reference_evidence(evidence[:,start-offset:].float())
        x=self.dropout(x);condition=self.condition_context(batch)
        groups=query_groups(batch['tile_ids_cpu'],start,ids.device) if 'tile_ids_cpu' in batch else None
        for i,layer in enumerate(self.decoder):
            x=layer(x,condition,batch['token_mask'],beats,cache,state=state['layers'][i] if use_cache else None,groups=groups)
        logits=F.linear(self.norm(x),weight,self.output_bias)
        if hasattr(self, 'onset_readout') and batch.get('onset_strength', 1.) != 0:
            from .onset_readout import apply_phase_evidence
            logits = apply_phase_evidence(logits, cache['onset_scores'], batch['phase_queries_cpu'],
                self.phase_token_ids, start=start, strength=batch.get('onset_strength', 1.))
        if use_cache:
            state['length']=length;cache['decode_state']=state
        return logits
    def forward(self,batch):
        return {'logits':self.decode(batch,self.encode_audio(batch),incremental=False)}
    def loss(self,output,batch):
        logits=output['logits'].float();labels=batch['labels'];active=labels.ne(-100)
        weights=batch.get('token_weights',active.float()).float()*active
        weights=weights*batch.get('sample_weight',weights.new_ones(weights.size(0)))[:,None]
        costs=F.cross_entropy(logits.flatten(0,1),labels.flatten(),ignore_index=-100,reduction='none').reshape_as(weights)
        mass=weights.sum().clamp_min(1);loss=(costs*weights).sum()/mass
        timing=self.time_tokens[labels.clamp_min(0)]&active
        def mean(mask):
            w=weights*mask;return (costs*w).sum()/w.sum().clamp_min(1)
        return loss,dict(loss=loss.detach(),time_loss=mean(timing).detach(),
            component_loss=mean(active&~timing).detach(),
            phase_loss=mean(self.phase_targets[labels.clamp_min(0)]&active).detach(),token_count=active.sum().detach())
