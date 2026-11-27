"""One causal component stream with local music and whole-song attention.

Attention-probability dropout is zero; residual/FF dropout supplies regularization
without the BF16 attention dropout cancellation encountered in the old family.
"""
from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
from torch.nn import functional as F
from .defaults import AUDIO_WRITING_POLICIES


@dataclass
class ComposerV2Config:
    d_model: int = 384
    layers: int = 8
    nhead: int = 8
    dim_feedforward: int = 1536
    dropout: float = .1
    audio_layers: int = 4
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
    intensity_policy: str = 'off'
    music_context_policy: str = 'off'
    local_audio_block_beats: int = 1
    attention_normalization: str = 'qk_rms_v1'
    future_configuration_policy: str = 'off'
    future_configuration_weight: float = .05
    spatial_policy: str = 'off'
    intent_policy: str = 'off'
    hand_plan_policy: str = 'off'
    def __post_init__(self):
        from .hand_plan import CHOICES as HAND_PLANS
        if self.hand_plan_policy not in HAND_PLANS[1:] or (self.hand_plan_policy!='off' and self.sequence_policy!='continuous_events_v1'):
            raise ValueError('手顺预判需要连续事件模型及有效策略')
        from .intent_policy import POLICY as INTENT_POLICY
        if self.intent_policy not in ('off', INTENT_POLICY):
            raise ValueError('Unknown phrase intent policy')
        if self.intent_policy != 'off' and self.sequence_policy != 'continuous_events_v1':
            raise ValueError('Phrase intent requires continuous events')
        from . import spatial_policy
        if self.spatial_policy == 'auto':
            raise ValueError('Model configuration requires a resolved spatial policy')
        spatial_policy.resolve(self.spatial_policy, saved=None, resume=False,
                               sequence=self.sequence_policy, writing=self.future_configuration_policy)
        if self.future_configuration_policy not in ('off', 'ordered_future_v1', *AUDIO_WRITING_POLICIES):
            raise ValueError('Unknown ordered future configuration policy')
        if (not isinstance(self.future_configuration_weight, (int, float))
                or isinstance(self.future_configuration_weight, bool)
                or not math.isfinite(self.future_configuration_weight)
                or not 0 <= self.future_configuration_weight <= 1):
            raise ValueError('Future configuration weight must be finite and between 0 and 1')
        if self.future_configuration_policy != 'off' and self.sequence_policy != 'continuous_events_v1':
            raise ValueError('Ordered future configuration requires continuous events')
        if self.attention_normalization not in ('off', 'qk_rms_v1'):
            raise ValueError('Unknown attention normalization policy')
        from .music_policy import POLICIES as MUSIC_POLICIES
        if self.music_context_policy not in MUSIC_POLICIES or min(self.audio_layers,self.song_layers)<1:
            raise ValueError('Invalid music context/encoder depth')
        if self.music_context_policy == 'song_paired_v2' and self.sequence_policy != 'continuous_events_v1':
            raise ValueError('Paired music context requires continuous events')
        if (type(self.local_audio_block_beats) is not int or self.local_audio_block_beats not in (1, 2)
                or (self.local_audio_block_beats != 1 and self.sequence_policy != 'continuous_events_v1')):
            raise ValueError('Two-beat audio sharing requires continuous events')
        from .intensity import POLICY as CURVE_POLICY
        if self.intensity_policy not in ('off',CURVE_POLICY) or (self.intensity_policy!='off' and self.sequence_policy!='continuous_events_v1'):
            raise ValueError('Intensity curves require continuous events')
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
    def from_dict(cls,value):return cls(**{'condition_policy':'level_only_v1','sequence_policy':'window_v1','audio_layers':2,
                                         'attention_normalization':'off',**value})


def positions(values,width):
    scale=torch.exp(torch.arange(0,width,2,device=values.device,dtype=torch.float32)*(-math.log(10000.)/width))
    angle=values.float()[...,None]*scale
    return torch.stack((angle.sin(),angle.cos()),-1).flatten(-2)[...,:width]


def _stable_biased_attention(query, key, value, mask, causal):
    from torch.nn.attention import SDPBackend, sdpa_kernel

    with torch.autocast('cuda', enabled=False), sdpa_kernel(SDPBackend.MATH):
        return F.scaled_dot_product_attention(
            query.float(), key.float(), value.float(), attn_mask=mask.float(),
            dropout_p=0., is_causal=causal).to(query.dtype)


def _normalized_qk(query, key):
    q32, k32 = query.float(), key.float()
    return ((q32 * torch.rsqrt(q32.square().mean(-1, keepdim=True) + 1e-6)).to(query.dtype),
            (k32 * torch.rsqrt(k32.square().mean(-1, keepdim=True) + 1e-6)).to(key.dtype))


def _normalized_attention(query, key, value, mask, causal, stable):
    query, key = _normalized_qk(query, key)
    if stable:
        return _stable_biased_attention(query, key, value, mask, causal)
    return F.scaled_dot_product_attention(query, key, value, attn_mask=mask, dropout_p=0., is_causal=causal)


class Attention(nn.Module):
    pack_projections = False
    _inference_qkv = None

    def __init__(self,width,heads,*,local=False,relative=False,normalization='off'):
        super().__init__();self.heads=heads;self.dim=width//heads;self.local=local
        if normalization not in ('off', 'qk_rms_v1'):
            raise ValueError('Unknown attention normalization policy')
        self.normalization = normalization
        self.q=nn.Linear(width,width);self.k=nn.Linear(width,width);self.v=nn.Linear(width,width);self.out=nn.Linear(width,width)
        if local:
            self.past=nn.Parameter(torch.full((heads,),-1.5));self.future=nn.Parameter(torch.full((heads,),-1.5))
        if relative:
            self.relative_beat=nn.Parameter(torch.zeros(heads,2))
    def split(self,value):return value.reshape(value.size(0),value.size(1),self.heads,self.dim).transpose(1,2)
    def _packed_projection(self, value, *layers):
        # Keep the separate q/k/v parameters for checkpoint and optimizer compatibility.
        weight = torch.cat([layer.weight for layer in layers])
        bias = torch.cat([layer.bias for layer in layers])
        parts = F.linear(value, weight, bias).chunk(len(layers), dim=-1)
        return tuple(self.split(part) for part in parts)

    def memory(self, value):
        if self.training and self.pack_projections:
            return self._packed_projection(value, self.k, self.v)
        return self.split(self.k(value)), self.split(self.v(value))

    def incremental_qkv(self,value):
        # Generation reuses immutable FP32 weights across contexts and retries.
        if self._inference_qkv is None:
            self._inference_qkv=tuple(torch.cat((getattr(self.q,name),getattr(self.k,name),getattr(self.v,name)))
                for name in ('weight','bias'))
        return tuple(self.split(part) for part in F.linear(value,*self._inference_qkv).chunk(3,dim=-1))

    def forward(self,query,memory=None,*,kv=None,key_mask=None,causal=False,past_length=0,query_beats=None,key_beats=None,
                projected_query=None):
        if projected_query is not None:
            q=projected_query
            k,v=kv
        elif self.training and self.pack_projections and kv is None and query is memory:
            q, k, v = self._packed_projection(query, self.q, self.k, self.v)
        else:
            q = self.split(self.q(query))
            k, v = self.memory(memory) if kv is None else kv
        implicit_causal = causal and key_mask is None and past_length == 0 and q.size(-2) == k.size(-2)
        mask=None
        if key_mask is not None:mask=key_mask[:,None,None,:]
        if causal and not implicit_causal and not (q.size(-2)==1 and past_length+1==k.size(-2)):
            allowed=torch.arange(k.size(-2),device=q.device)[None,:] <= (torch.arange(q.size(-2),device=q.device)[:,None]+past_length)
            mask=allowed[None,None] if mask is None else mask&allowed[None,None]
        if hasattr(self,'relative_beat'):
            gap=key_beats[:,None,:]-query_beats[:,:,None]
            features=torch.stack((torch.log1p((-gap).clamp_min(0)),torch.log1p(gap.clamp_min(0))),-1)
            bias=torch.einsum('bqkf,hf->bhqk',features,self.relative_beat)
            mask=bias.masked_fill(~mask,-torch.inf) if mask is not None else bias
            mask=mask.to(q.dtype)
        if self.local:
            gap=key_beats[:,None,:]-query_beats[:,:,None]
            bias=-(F.softplus(self.past)[None,:,None,None]*(-gap[:,None]).clamp_min(0)
                   +F.softplus(self.future)[None,:,None,None]*gap[:,None].clamp_min(0))
            mask=bias.masked_fill(~mask,-torch.inf) if mask is not None else bias
            mask=mask.to(q.dtype)
        # Fused low-precision backward loses softmax's zero-sum score gradient
        # when audio keys acquire a large common offset. The resulting error
        # is amplified by those keys, even with finite losses and gradient clipping.
        biased_training = (torch.is_grad_enabled() and q.is_cuda
                           and q.dtype in (torch.float16, torch.bfloat16)
                           and (self.local or hasattr(self, 'relative_beat')))
        if self.normalization == 'qk_rms_v1':
            # Projected Q/K magnitudes are not bounded by pre-LayerNorm.
            # Fixed RMS normalization bounds content logits by sqrt(dim).
            # Recompute normalization and attention together so long batches
            # do not retain both raw and normalized Q/K buffers per layer.
            if torch.is_grad_enabled() and q.is_cuda:
                from torch.utils.checkpoint import checkpoint
                attended = checkpoint(_normalized_attention, q, k, v, mask, implicit_causal, biased_training,
                                      use_reentrant=False, preserve_rng_state=False)
            else:
                attended = _normalized_attention(q, k, v, mask, implicit_causal, False)
        elif biased_training:
            from torch.utils.checkpoint import checkpoint
            # Recompute only attention probabilities rather than retaining an
            # FP32 query-by-key matrix for every audio/decoder layer.
            attended = checkpoint(_stable_biased_attention, q, k, v, mask, implicit_causal,
                                  use_reentrant=False, preserve_rng_state=False)
        else:
            attended=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0.,is_causal=implicit_causal)
        return self.out(attended.transpose(1,2).reshape(query.shape))


class EncoderBlock(nn.Module):
    def __init__(self,c):
        super().__init__();self.norm1=nn.LayerNorm(c.d_model);self.norm2=nn.LayerNorm(c.d_model)
        self.attention=Attention(c.d_model,c.nhead,normalization=c.attention_normalization)
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
    flash_self_attention = False

    def __init__(self,c,*,song=True):
        super().__init__();self.norms=nn.ModuleList([nn.LayerNorm(c.d_model) for _ in range(4)])
        self.self_attention=Attention(c.d_model,c.nhead,normalization=c.attention_normalization)
        self.local_attention=Attention(c.d_model,c.nhead,local=True,normalization=c.attention_normalization)
        self.song_attention=Attention(c.d_model,c.nhead,relative=c.music_context_policy!='off',
                                      normalization=c.attention_normalization) if song else None
        self.ff=nn.Sequential(nn.Linear(c.d_model,c.dim_feedforward),nn.GELU(),nn.Dropout(c.dropout),nn.Linear(c.dim_feedforward,c.d_model))
        self.drop=nn.Dropout(c.dropout)
        self.cache_capacity=c.max_tokens
    def _append_cached_kv(self,kv,state):
        """Append into reusable storage; rewinding only changes the live views."""
        old=state.get('self_kv');old_length=old[0].size(-2) if old is not None else 0
        length=old_length+kv[0].size(-2)
        storage=state.get('self_kv_storage')
        if storage is None or storage[0].size(-2)<length:
            capacity=max(length,min(self.cache_capacity,max(64,2*storage[0].size(-2) if storage else length+64)))
            storage=tuple(value.new_empty((*value.shape[:-2],capacity,value.size(-1))) for value in kv)
            if old is not None:
                for buffer,value in zip(storage,old):buffer[...,:old_length,:].copy_(value)
            state['self_kv_storage']=storage
        for buffer,value in zip(storage,kv):buffer[...,old_length:length,:].copy_(value)
        return tuple(buffer[...,:length,:] for buffer in storage),old_length
    _append_cpu_kv=_append_cached_kv  # OpenVINO commits into the same host-owned cache.
    def forward(self,x,condition,token_mask,beats,audio,*,state=None,groups=None,right_padded=False):
        value=self.norms[0](x+condition[:,None])
        if state is not None:
            q=None
            if (not self.training and not torch.is_grad_enabled() and value.is_cuda
                    and value.dtype==torch.float32 and value.size(1)==1 and not torch.is_autocast_enabled('cuda')):
                q,k,v=self.self_attention.incremental_qkv(value)
                kv=(k,v)
            else:
                kv=self.self_attention.memory(value)
            old_length=0
            if kv[0].device.type=='cpu' or (kv[0].device.type=='cuda' and not torch.is_grad_enabled()):
                kv,old_length=self._append_cached_kv(kv,state)
            elif 'self_kv' in state:
                old_length=state['self_kv'][0].size(-2)
                kv=tuple(torch.cat((old,new),dim=-2) for old,new in zip(state['self_kv'],kv))
            state['self_kv']=kv
            attended=self.self_attention(value,kv=kv,key_mask=token_mask,causal=True,past_length=old_length,projected_query=q)
        else:
            # With right padding, active causal queries cannot see the padded suffix.
            mask = None if self.training and self.flash_self_attention and right_padded else token_mask
            attended=self.self_attention(value,value,key_mask=mask,causal=True)
        x=x+self.drop(attended)
        local_kv=state.get('local_kv') if state is not None else None
        if local_kv is not None and local_kv[0].size(0)<audio['local'].size(0):
            # Within one context, newly discovered audio tiles only append.
            added=self.local_attention.memory(audio['local'][local_kv[0].size(0):])
            local_kv=tuple(torch.cat((old,new),0) for old,new in zip(local_kv,added))
            state['local_kv']=local_kv
        elif local_kv is not None and local_kv[0].size(0)!=audio['local'].size(0):local_kv=None
        if local_kv is None:
            local_kv=self.local_attention.memory(audio['local'])
            if state is not None:state['local_kv']=local_kv
        query=self.norms[1](x+condition[:,None])
        if groups is None:
            attended=self.local_attention(query,kv=local_kv,key_mask=audio['local_mask'],query_beats=beats,key_beats=audio['local_beats'])
        elif isinstance(groups,int):
            # One incremental token reads one tile. Slices share cached memory;
            # packing, copying the whole K/V tile, and scattering are redundant.
            tile=slice(groups,groups+1)
            attended=self.local_attention(query,kv=tuple(v[tile] for v in local_kv),
                key_mask=audio['local_mask'][tile],query_beats=beats,key_beats=audio['local_beats'][tile])
        else:
            tiles,indices,valid=groups
            packed=query.flatten(0,1).index_select(0,indices.flatten()).reshape(*indices.shape,query.size(-1))
            attended=self.local_attention(packed,kv=tuple(v.index_select(0,tiles) for v in local_kv),
                key_mask=audio['local_mask'].index_select(0,tiles),
                query_beats=beats.flatten().index_select(0,indices.flatten()).reshape_as(indices),
                key_beats=audio['local_beats'].index_select(0,tiles))
            attended=(attended*valid[...,None]).flatten(0,1)
            attended=attended.new_zeros(query.numel()//query.size(-1),query.size(-1)).index_add(0,indices.flatten(),attended).reshape_as(query)
        x=x+self.drop(attended)
        if self.song_attention is not None:
            song_kv=state.get('song_kv') if state is not None else None
            if song_kv is None:
                song_kv=self.song_attention.memory(audio['song'])
                if state is not None:state['song_kv']=song_kv
            x=x+self.drop(self.song_attention(self.norms[2](x+condition[:,None]),kv=song_kv,key_mask=audio['song_mask'],query_beats=beats,key_beats=audio.get('song_beats')))
        return x+self.drop(self.ff(self.norms[3](x)))


class ComposerV2(nn.Module):
    def _training_block(self, layer, *args, **kwargs):
        if (self.training and torch.is_grad_enabled() and getattr(self, 'activation_checkpointing', False)
                and (getattr(self, 'activation_checkpointing_scope', 'all') != 'decoder' or isinstance(layer, DecoderBlock))):
            from torch.utils.checkpoint import checkpoint
            return checkpoint(layer, *args, use_reentrant=False, preserve_rng_state=True, **kwargs)
        return layer(*args, **kwargs)

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
        if c.future_configuration_policy == 'ordered_future_v1':
            from .ordered_configuration import HORIZONS, TARGET_DIM
            # Training-only readout. Preserve all old initialization RNG and
            # never feed targets or predictions back into token decoding.
            with torch.random.fork_rng(devices=[]):
                self.future_configuration_head=nn.Sequential(
                    nn.LayerNorm(d),nn.Linear(d,min(128,d)),nn.SiLU(),
                    nn.Linear(min(128,d),len(HORIZONS)*TARGET_DIM))
        self.register_buffer('rim_button_ids',torch.tensor([codec.ids[f'P:{i}'] for i in range(1,9)
                                                            if f'P:{i}' in codec.ids],dtype=torch.long),
                             persistent=False)
        if c.spatial_policy == 'disk_geometry_v1':
            from .spatial_history import DiskSpatialHistory
            with torch.random.fork_rng(devices=[]):
                self.spatial_history = DiskSpatialHistory(codec, d)
        if c.future_configuration_policy in AUDIO_WRITING_POLICIES:
            from .local_writing import LocalWriting
            with torch.random.fork_rng(devices=[]):
                if c.spatial_policy == 'off':
                    self.local_writing = LocalWriting(d, c.future_configuration_policy)
                else:
                    self.local_writing = LocalWriting(d, c.future_configuration_policy, spatial_policy=c.spatial_policy)
            self.register_buffer('writing_time_tokens', torch.tensor([
                token == '[AT]' or token.startswith('NUM:') for token in codec.tokens]), persistent=False)
            excluded = {'[PAD]', '[BOS]', '[CARRY]', '[CTX]', '[REF]', '[TGT]',
                        '[AT]', '[PHASE]', '[EOS]', '[END_EVENT]'}
            self.register_buffer('writing_query_tokens', torch.tensor([
                token not in excluded and not token.startswith(('NUM:', 'T:'))
                for token in codec.tokens]), persistent=False)
        if c.music_context_policy!='off':
            with torch.random.fork_rng(devices=[]):self.song_detail=nn.Linear(16,d,bias=False)
            nn.init.zeros_(self.song_detail.weight)
        if c.music_context_policy == 'song_paired_v2':
            from .music_evidence import PAIR_FEATURES, ENTRY_FEATURES
            with torch.random.fork_rng(devices=[]):
                self.paired_music=nn.Linear(PAIR_FEATURES,d,bias=False)
                self.entry_preview=nn.Linear(ENTRY_FEATURES,d,bias=False)
            nn.init.zeros_(self.paired_music.weight)
            nn.init.zeros_(self.entry_preview.weight)
            self.register_buffer('entry_query_tokens',torch.tensor([
                t in ('[AT]','[END_EVENT]','[TGT]') or t.startswith('NUM:') for t in codec.tokens]),persistent=False)
        if c.intensity_policy!='off':
            from .intensity import IntensityContext
            with torch.random.fork_rng(devices=[]):self.intensity=IntensityContext(d)
        self.register_buffer('phase_targets',torch.tensor([t.startswith('F:') for t in codec.tokens]),persistent=False)
        self.register_buffer('time_tokens',torch.tensor([t.startswith(('T:','F:')) or t in ('[AT]','[PHASE]','[TIME_EXT]','[TIME_END]','[EOS]') or t.startswith('NUM:') for t in codec.tokens]),persistent=False)
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
        if c.hand_plan_policy != 'off':
            from .hand_plan import HandPlan
            with torch.random.fork_rng(devices=[]):
                self.hand_plan = HandPlan(codec,d)
                from . import hand_memory
                if c.hand_plan_policy in hand_memory.POLICIES:
                    self.hand_memory = hand_memory.HandMemory(d,self.hand_plan.execute.out_features)
        if c.intent_policy != 'off':
            from .intent import PhraseIntent
            with torch.random.fork_rng(devices=[]):
                self.intent = PhraseIntent(codec, d)
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
    def _clear_inference_caches(self):
        self._song_cache.clear();self._structured_weight_cache.clear()
        for layer in self.decoder:
            layer.self_attention._inference_qkv=None
    def train(self,mode=True):
        self._clear_inference_caches()
        return super().train(mode)
    def _apply(self,fn,recurse=True):
        self._clear_inference_caches()
        return super()._apply(fn,recurse=recurse)
    def load_state_dict(self,*args,**kwargs):
        self._clear_inference_caches()
        return super().load_state_dict(*args,**kwargs)
    def _audio_input(self,mel,semantic,valid,keep):
        return self.mel(mel.float())+self.semantic(semantic.float())*(valid&keep[:,None])[...,None]
    def encode_local_audio(self, batch, keep):
        mel = batch.get('bank_audio', batch['local_audio'])
        owners=batch.get('tile_owner')
        if 'bank_audio' in batch:
            semantic=batch['bank_semantic'].float()
            if 'bank_semantic_right' in batch:
                fraction=batch['bank_semantic_weight'][:,None]
                semantic=(semantic.double()*(1.-fraction)
                          +batch['bank_semantic_right'].float().double()*fraction).float()
            bank=self.mel(mel.float())+self.semantic(semantic)*(batch['bank_semantic_mask']&keep.index_select(0,batch['bank_owner']))[:,None]
            if hasattr(self,'audio_envelope'):bank=bank+self.audio_envelope(batch['bank_envelope'].float())
            frames=batch['tile_frames']
            local=bank.index_select(0,frames.flatten()).reshape(*frames.shape,bank.size(-1))
        else:
            local=self._audio_input(mel,batch['local_semantic'],batch['local_semantic_mask'],keep if owners is None else keep.index_select(0,owners))
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
        for layer in self.audio_blocks:local=self._training_block(layer,local,mask)
        return dict(local=local, local_mask=mask, local_beats=beats)

    def encode_beat_features(self, batch):
        features = {}
        if hasattr(self,'entry_preview'):
            features['entry_context'] = self.entry_preview(batch['entry_preview_features'].float())
        if hasattr(self, 'onset_readout') and batch.get('onset_strength', 1.) != 0:
            features['onset_scores'] = self.onset_readout(batch)
        return features

    def encode_audio(self,batch):
        # Local audio and beat-specific evidence can grow independently during decoding.
        b = batch['input_ids'].size(0)
        device = batch['input_ids'].device
        keep = torch.rand(b, device=device) >= .15 if self.training else torch.ones(b, device=device, dtype=torch.bool)
        result = self.encode_local_audio(batch, keep)
        ids=batch.get('song_cache_key',[])
        key=ids[0] if not self.training and not torch.is_grad_enabled() and len(ids)==1 else None
        cached=self._song_cache.get(key) if key is not None else None
        if cached is None:
            song=self._audio_input(batch['song_audio'],batch['song_semantic'],batch['song_semantic_mask'],keep)
            if hasattr(self,'song_detail'):
                song=song+self.song_detail(batch['song_detail'].float())
            starts=torch.tensor([float(s) for s in batch.get('start_beat',[0.]*b)],device=device)
            song=song+positions(batch['song_beats']+starts[:,None],self.config.d_model)
            song_mask=batch['song_mask'].clone();song_mask[:,0]|=~song_mask.any(-1)
            for layer in self.song_blocks:song=self._training_block(layer,song,song_mask)
            if key is not None:
                if len(self._song_cache)>=4:self._song_cache.clear()
                self._song_cache[key]=(song.detach(),song_mask)
        else:song,song_mask=cached
        result.update(song=song,song_mask=song_mask,song_beats=batch['song_beats'])
        if hasattr(self,'local_writing'):
            result.update(self.local_writing.plan(song,batch))
        # Preserve the legacy order around the stochastic curve planner.
        if hasattr(self,'entry_preview'):
            result['entry_context'] = self.entry_preview(batch['entry_preview_features'].float())
        if hasattr(self,'intensity'):
            result['condition']=self.condition_context(batch)
            result['curve_rates'],result['curve_context']=self.intensity.plan(song,batch,training=self.training)
        if hasattr(self, 'onset_readout') and batch.get('onset_strength', 1.) != 0:
            result['onset_scores'] = self.onset_readout(batch)
        if hasattr(self, 'intent') and batch.get('intent_mode', 'soft' if self.training else 'off') == 'soft':
            strength = float(batch.get('intent_strength', 1.))
            if not math.isfinite(strength) or not 0. <= strength <= 1.:
                raise ValueError('Phrase intent strength must be finite and between 0 and 1')
            if strength > 0.:
                plan = batch.get('intent_plan')
                if plan is None:
                    plan = self.intent.plan(song, batch, strength=strength, training=self.training)
                result['intent_plan'] = plan
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
    def decode(self,batch,cache,*,incremental=True,return_future_hidden=False,decision_positions=None,button_queries=None,
               use_spatial_history=True):
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
        writing_tiles=None
        if hasattr(self,'paired_music'):
            offset=int(batch.get('paired_music_start',0))
            x=x+self.paired_music(batch['paired_music_features'][:,start-offset:].float())
            tiles=batch['tile_ids_cpu']
            indices=torch.tensor([row[start:] for row in tiles],device=x.device,dtype=torch.long)
            writing_tiles=indices
            preview=cache['entry_context'][indices.clamp_min(0)]
            x=x+preview*(self.entry_query_tokens[value] & indices.ge(0))[...,None]
        if self.config.sequence_policy=='window_v1':x=x+positions(beats+starts[:,None],self.config.d_model)
        if structured:
            from .rhythm_structure import phase_features
            # Sub-beat phase of each token's clock, its stream region (carry,
            # history, aligned reference, target) and the reference trust.
            x=x+self.phase_position(phase_features(beats,self.phase_harmonics))
            codes=self.segment_code.index_select(0,value.flatten()).reshape_as(value)
            if start and 'last_segment' in state:
                segment=torch.maximum(codes.cummax(1).values,state['last_segment'][:,None])
            elif start:segment=torch.maximum(codes.cummax(1).values,
                self.segment_code.index_select(0,ids[:,:start].flatten()).reshape(ids.size(0),start).amax(1)[:,None])
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
        if hasattr(self,'intensity'):
            x=x+self.intensity(cache['curve_context'],batch,start=start)
        rim_features=None
        if use_spatial_history and hasattr(self, 'spatial_history'):
            spatial_state=state.setdefault('spatial', {}) if use_cache else None
            spatial_frame=self.spatial_history.scan(value,beats,batch['token_mask'][:,start:],
                                                    start=start,state=spatial_state)
            single_cpu=use_cache and value.device.type=='cpu' and value.numel()==1
            spatial_context,rim_features=self.spatial_history.features(spatial_frame,beats,single_cpu=single_cpu)
            if spatial_context is not None:
                x=x+self.spatial_history.context(spatial_context.to(x.dtype))
        x=self.dropout(x);condition=cache['condition'] if 'condition' in cache else self.condition_context(batch)
        groups=None
        if 'tile_ids_cpu' in batch:
            tiles=batch.get('local_tile_ids_cpu', batch['tile_ids_cpu'])
            groups=(int(tiles[0][start]) if use_cache and ids.size(0)==1
                    and length-start==1 and tiles[0][start]>=0 else query_groups(tiles,start,ids.device))
        runtime=getattr(self,'_decoder_runtime',None)
        accelerated=(runtime.step(x,condition,beats,batch['token_mask'],cache,state,groups)
                     if runtime is not None and use_cache else None)
        if accelerated is not None:
            x=accelerated
        else:
            for i,layer in enumerate(self.decoder):
                x=self._training_block(layer,x,condition,batch['token_mask'],beats,cache,
                    state=state['layers'][i] if use_cache else None,groups=groups,
                    right_padded=batch.get('right_padded_tokens',False))
        if hasattr(self,'local_writing'):
            x=x+self.local_writing.read(cache['writing_context'],batch,start,value,self.writing_query_tokens,
                time_mask=self.writing_time_tokens,indices=writing_tiles)
        normalized=self.norm(x)
        if hasattr(getattr(self,'local_writing',None),'successor'):
            # Re-read the causal state at the native timing/layout decision.
            # Frontier tokens may teach a gap but never convert that gap into an EOS bias.
            forecast=self.local_writing.successor(normalized)
            probabilities=forecast.float().sigmoid()-.5
            role_values=torch.cat((probabilities[:,:,:3]*self.writing_time_tokens[value][...,None],
                                   probabilities[:,:,3:]*self.writing_query_tokens[value][...,None]),dim=-1)
            normalized=normalized+self.local_writing.execute_successor(role_values.to(normalized.dtype))
        hand_bias=None
        if hasattr(self,'hand_plan'):
            hand_state=state.setdefault('hand',{}) if use_cache else None
            hand_bias,planned,observed=self.hand_plan(normalized,value,start,hand_state,batch.get('mechanics_rows'),
                                                        batch['token_mask'][:,start:])
            if 'hand_positions' in batch and not use_cache:
                cache['hand_prediction']=(planned,observed)
            if hasattr(self,'hand_memory'):
                from .hand_memory import FEEDBACK
                timeline=batch.get('hand_timeline')
                offset=int(batch.get('hand_timeline_start',0))
                rows=timeline[:,start-offset:] if timeline is not None else None
                memory_state=state.setdefault('hand_memory',{}) if use_cache else None
                post,extra=self.hand_memory(normalized,rows,start,memory_state,batch.get('mechanics_rows'),
                    self.config.hand_plan_policy==FEEDBACK and batch.get('use_hand_feedback',True))
                if extra is not None:
                    # Query and vocabulary restriction match the existing N:/P: channel.
                    target=(hand_state['region'][:,start:length].eq(3) if use_cache
                            else ids.eq(self.codec.tgt_id).cumsum(1)>0)
                    hand_bias=hand_bias+extra*(target & self.hand_plan.layout_query[value])[...,None]
                if 'memory_positions' in batch and not use_cache:
                    cache['memory_readout']=post
        intent_plan = cache.get('intent_plan')
        if batch.get('intent_mode') == 'off' or batch.get('intent_strength', 1.) == 0.:
            intent_plan = None
        if (intent_plan is not None and use_cache and value.device.type == 'cpu' and value.numel() == 1
                and not self.intent.layout_inputs[value.item()]):
            intent_plan = None
        if button_queries is not None:
            queried=normalized[button_queries]
            logits=F.linear(queried,weight.index_select(0,self.rim_button_ids),
                            self.output_bias.index_select(0,self.rim_button_ids))
            if hand_bias is not None:
                logits=logits+hand_bias[button_queries].index_select(-1,self.hand_plan.buttons).to(logits.dtype)
            if rim_features is not None:
                logits=logits+self.spatial_history.rim_bias(queried,rim_features[button_queries]).to(logits.dtype)
            if intent_plan is not None and intent_plan.strength > 0.:
                residual, strength = self.intent.residual(normalized, value, beats,
                    weight.index_select(0, self.rim_button_ids), intent_plan)
                adjusted = residual[button_queries].to(logits.dtype)
                if adjusted.numel():
                    intent_plan.record_execution(adjusted, strength[button_queries])
                logits = logits + adjusted
            return logits
        logits=F.linear(normalized,weight,self.output_bias)
        if hand_bias is not None:
            logits.index_add_(-1,self.hand_plan.ids,hand_bias.index_select(-1,self.hand_plan.columns).to(logits.dtype))
        if rim_features is not None:
            rim_bias=self.spatial_history.rim_bias(normalized,rim_features).to(logits.dtype)
            logits.index_add_(-1,self.rim_button_ids,rim_bias)
        if intent_plan is not None and intent_plan.strength > 0.:
            candidates = self.intent.component_ids
            residual, strength = self.intent.residual(normalized, value, beats,
                weight.index_select(0, candidates), intent_plan)
            residual = residual.to(logits.dtype)
            logits.index_add_(-1, candidates, residual)
            intent_plan.record_execution(residual[:, -1], strength[:, -1])
            if not use_cache:
                cache['intent_execution'] = (strength.detach(), residual.detach().float().abs().mean(-1))
        if hasattr(self, 'onset_readout') and batch.get('onset_strength', 1.) != 0:
            from .onset_readout import apply_phase_evidence
            logits = apply_phase_evidence(logits, cache['onset_scores'], batch['phase_queries_cpu'],
                self.phase_token_ids, start=start, strength=batch.get('onset_strength', 1.))
        if use_cache:
            state['length']=length;cache['decode_state']=state
        if decision_positions is not None:
            rows=decision_positions
            return logits,forecast[rows[:,0],rows[:,1]-start]
        if return_future_hidden:
            rows=batch['future_configuration_positions']
            return logits,normalized[rows[:,0],rows[:,1]-start]
        return logits
    def forward(self,batch,*,audio_cache=None):
        cache=self.encode_audio(batch) if audio_cache is None else audio_cache
        if 'intent_plan' in cache:
            cache.pop('intent_execution', None)
        if self.training and 'layout_replay' in batch:
            from .phrase_replay import replace_keys
            if 'condition' not in cache:
                cache['condition']=self.condition_context(batch)
            batch=replace_keys(self,batch,cache)
        positions=batch.get('future_configuration_positions')
        use_future=(hasattr(self,'future_configuration_head') and
                    self.config.future_configuration_weight>0 and positions is not None and positions.numel()>0)
        future={}
        if 'layout_replay_changed' in batch:
            future['layout_replay_changed']=batch['layout_replay_changed']
        if use_future:
            from .ordered_configuration import HORIZONS,TARGET_DIM
            logits,hidden=self.decode(batch,cache,incremental=False,return_future_hidden=True)
            future['future_configuration_logits']=self.future_configuration_head(hidden).reshape(-1,len(HORIZONS),TARGET_DIM)
        elif 'decision_positions' in batch and self.config.future_configuration_weight>0:
            logits,future['decision_logits']=self.decode(batch,cache,incremental=False,
                                                        decision_positions=batch['decision_positions'])
        else:logits=self.decode(batch,cache,incremental=False)
        if 'hand_prediction' in cache:
            future['hand_prediction']=cache.pop('hand_prediction')
        if 'memory_readout' in cache:
            future['memory_readout']=cache.pop('memory_readout')
        if hasattr(self,'local_writing'):
            future['future_configuration_logits']=cache['writing_logits']
            if 'slide_geometry_logits' in cache:
                future['slide_geometry_logits'] = cache['slide_geometry_logits']
        if ('intent_plan' in cache and batch.get('intent_mode') != 'off'
                and batch.get('intent_strength', 1.) != 0.):
            execution = cache.get('intent_execution')
            if execution is None:
                zero = logits.new_zeros(logits.shape[:2], dtype=torch.float32)
                execution = (zero, zero)
            future['intent_execution'] = execution
            future['intent_plan'] = cache['intent_plan']
        return {'logits':logits,**future,
                **({'curve_rates':cache['curve_rates']} if 'curve_rates' in cache else {})}
    def loss(self,output,batch):
        logits=output['logits'].float();labels=batch['labels'];active=labels.ne(-100)
        weights=batch.get('token_weights',active.float()).float()*active
        weights=weights*batch.get('sample_weight',weights.new_ones(weights.size(0)))[:,None]
        costs=F.cross_entropy(logits.flatten(0,1),labels.flatten(),ignore_index=-100,reduction='none').reshape_as(weights)
        mass=weights.sum().clamp_min(1);loss=(costs*weights).sum()/mass
        indices=labels.clamp_min(0)
        # index_select avoids the Intel XPU boolean advanced-index kernel;
        # flatten/reshape preserves exactly the original multidimensional lookup.
        timing=self.time_tokens.index_select(0,indices.flatten()).reshape_as(labels)&active
        def mean(mask):
            w=weights*mask;return (costs*w).sum()/w.sum().clamp_min(1)
        auxiliary=loss.new_zeros(())
        if 'curve_rates' in output:
            from .intensity import curve_loss,AUX_WEIGHT
            auxiliary=curve_loss(output['curve_rates'],batch)
            objective=loss+AUX_WEIGHT*auxiliary
        else:objective=loss
        future_loss=loss.new_zeros(())
        if 'future_configuration_logits' in output and 'future_configuration_targets' in batch:
            prediction=output['future_configuration_logits'].float()
            target=batch['future_configuration_targets'].float()
            future_weights=batch['future_configuration_weights'].float()
            # Labels/quality are loss-only; credit is normalized to the same
            # supervised mass as the main loss for exact gradient accumulation.
            future_loss=(F.binary_cross_entropy_with_logits(prediction,target,reduction='none')*future_weights).sum()/mass
            if 'decision_logits' in output:
                error=F.binary_cross_entropy_with_logits(output['decision_logits'].float(),
                    batch['decision_targets'].float(),reduction='none')
                future_loss=future_loss+(error*batch['decision_weights']).sum()/mass
            objective=objective+self.config.future_configuration_weight*future_loss
        if 'gesture_positions' in batch:
            from . import gesture_pairs
            objective=objective+gesture_pairs.loss(logits,batch)/mass
        geometry_loss = None
        if 'slide_geometry_logits' in output and 'slide_geometry_weights' in batch:
            from . import slide_geometry, spatial_policy
            geometry_loss = slide_geometry.loss(output['slide_geometry_logits'], batch, mass)
            objective = objective + spatial_policy.GEOMETRY_LOSS_WEIGHT * geometry_loss
        intent_metrics = {}
        if 'intent_plan' in output:
            from .intent import AUX_WEIGHT
            intent_loss, intent_metrics = self.intent.auxiliary(output['intent_plan'], batch, weights.sum(1), mass)
            objective = objective + AUX_WEIGHT * intent_loss
            strength, shift = output['intent_execution']
            intent_metrics['intent_execution_strength'] = (strength * weights).sum() / mass
            intent_metrics['intent_gate_coverage'] = ((strength > 0.) * weights).sum() / mass
            intent_metrics['intent_execution_coverage'] = ((shift > 0.) * weights).sum() / mass
            intent_metrics['intent_logit_shift'] = (shift * weights).sum() / mass
        metrics=dict(loss=objective.detach(),token_loss=loss.detach(),curve_loss=auxiliary.detach(),
            future_configuration_loss=future_loss.detach(),time_loss=mean(timing).detach(),
            component_loss=mean(active&~timing).detach(),
            phase_loss=mean(self.phase_targets.index_select(0,indices.flatten()).reshape_as(labels)&active).detach(),token_count=active.sum().detach())
        if 'layout_replay_changed' in output:
            metrics['layout_replay_changed']=output['layout_replay_changed']
        if geometry_loss is not None:
            metrics['slide_geometry_loss'] = geometry_loss.detach()
        metrics.update({name: value.detach() for name, value in intent_metrics.items()})
        if 'hand_prediction' in output:
            from .hand_plan import supervision,AUX_WEIGHT
            hand_sum,hand_mass=supervision(*output['hand_prediction'],batch)
            if 'memory_readout' in output:
                from .hand_memory import supervision as memory_supervision
                memory_sum,memory_mass,stats=memory_supervision(output['memory_readout'],batch,by_recording=not self.training)
                hand_sum=hand_sum+memory_sum
                hand_mass=hand_mass+memory_mass
                metrics.update(hand_memory_sum=memory_sum.detach(),hand_memory_mass=memory_mass.detach())
                if stats is not None:
                    metrics['hand_memory_evidence']=stats
            main_mass=weights.sum()
            total=(main_mass+batch.get('mechanics_mass',0.)).clamp_min(1.)
            objective=(objective*main_mass+AUX_WEIGHT*hand_sum)/total
            metrics.update(loss=objective.detach(),hand_plan_sum=hand_sum.detach(),hand_plan_mass=hand_mass.detach())
        return objective,metrics
