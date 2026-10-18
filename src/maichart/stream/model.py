"""Continuous event-stream transformer for whole-song chart drafting.

The model consumes a bounded recent event prefix and a persistent summary. A
chunk is only a memory/compute boundary: it has no musical EOS. The final EOS
is supervised only at the end of a complete song stream.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass(slots=True)
class StreamConfig:
    vocab_size: int
    d_model: int = 256
    nhead: int = 8
    layers: int = 4
    dim_feedforward: int = 768
    dropout: float = .1
    audio_dim: int = 120
    condition_dim: int = 16
    max_audio_tokens: int = 1024
    max_history_tokens: int = 256
    max_stream_memory: int = 32
    max_tokens: int = 1024
    plan_dim: int = 8
    action_dim: int = 50
    grounding_profile: str = 'none'
    detail_audio_nodes: int = 8192
    local_audio_neighbors: int = 32
    local_audio_seconds: float = 2.
    archive_events: int = 4096
    archive_retrievals: int = 8
    archive_min_similarity: float = .5
    alignment_tolerance_ms: float = 15.

    def __post_init__(self):
        for name in ('vocab_size','d_model','nhead','layers','dim_feedforward',
                     'audio_dim','condition_dim','max_audio_tokens',
                     'max_history_tokens','max_stream_memory','max_tokens','plan_dim','action_dim',
                     'detail_audio_nodes','local_audio_neighbors','archive_events','archive_retrievals'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.d_model % self.nhead:
            raise ValueError('d_model must be divisible by nhead')
        if not 0 <= self.dropout < 1:
            raise ValueError('dropout must be in [0, 1)')
        if self.grounding_profile not in {'none','causal_audio_v1'}:
            raise ValueError('Unknown stream grounding_profile')
        if not math.isfinite(self.local_audio_seconds) or not .05<=self.local_audio_seconds<=30:
            raise ValueError('local_audio_seconds must be in [.05,30]')
        if not math.isfinite(self.archive_min_similarity) or not -1<=self.archive_min_similarity<=1:
            raise ValueError('archive_min_similarity must be in [-1,1]')
        if not math.isfinite(self.alignment_tolerance_ms) or not 1<=self.alignment_tolerance_ms<=100:
            raise ValueError('alignment_tolerance_ms must be in [1,100]')


class ChartStreamTransformer(nn.Module):
    architecture = 'chart_stream_v1'

    def __init__(self, config: StreamConfig):
        super().__init__()
        self.config = config
        w = config.d_model
        self.token_embedding = nn.Embedding(config.vocab_size, w, padding_idx=0)
        self.position_embedding = nn.Embedding(max(config.max_tokens, config.max_history_tokens) + 4, w)
        self.audio_projection = nn.Linear(config.audio_dim, w)
        self.condition_projection = nn.Linear(config.condition_dim, w)
        self.action_projection = nn.Linear(config.action_dim, w)
        self.context_router = nn.Sequential(nn.Linear(w, max(16, w // 2)), nn.GELU(), nn.Linear(max(16, w // 2), 3))
        nn.init.zeros_(self.context_router[-1].weight)
        nn.init.zeros_(self.context_router[-1].bias)
        self.null_condition = nn.Parameter(torch.zeros(w))
        self.audio_norm = nn.LayerNorm(w)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=w, nhead=config.nhead, dim_feedforward=config.dim_feedforward,
            dropout=config.dropout, activation='gelu', batch_first=True, norm_first=True)
        self.audio_encoder = nn.TransformerEncoder(encoder_layer, config.layers)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=w, nhead=config.nhead, dim_feedforward=config.dim_feedforward,
            dropout=config.dropout, activation='gelu', batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, config.layers)
        self.decoder_norm = nn.LayerNorm(w)
        self.output = nn.Linear(w, config.vocab_size)
        self.plan_head = nn.Linear(w, config.plan_dim)
        self.curve_head = nn.Linear(w, 4)
        if config.grounding_profile!='none':
            from .grounded_model import GroundedContext
            self.grounding=GroundedContext(config)

    def encode(self, audio, condition, *, audio_padding_mask=None,
               condition_null=None, stream_memory=None, stream_memory_padding_mask=None, action_state=None,
               action_state_padding_mask=None, return_curve=False, encoded_audio=None):
        values, mask, curve_prediction, route = (self.encode_audio(audio, condition,
            audio_padding_mask=audio_padding_mask, condition_null=condition_null)
            if encoded_audio is None else encoded_audio)
        batch = values.size(0)
        if action_state is not None:
            if action_state.ndim != 3 or action_state.shape[0] != batch or action_state.size(-1) != self.config.action_dim:
                raise ValueError('stream action_state shape mismatch')
            action_mask = (action_state_padding_mask.bool() if action_state_padding_mask is not None
                           else torch.zeros(action_state.shape[:2], dtype=torch.bool, device=values.device))
            if action_mask.shape != action_state.shape[:2]:
                raise ValueError('stream action_state mask shape mismatch')
            action = self.action_projection(action_state.to(values)).masked_fill(action_mask[..., None], 0.)
            action = action * route[:, None, 1:2]
            values = torch.cat((action, values), dim=1)
            memory_mask = torch.cat((action_mask, mask), dim=1)
        else:
            memory_mask = mask
        if stream_memory is not None:
            if stream_memory.ndim != 3 or stream_memory.shape[0] != batch or stream_memory.size(-1) != self.config.d_model:
                raise ValueError('stream memory shape mismatch')
            summary_mask = (torch.zeros(stream_memory.shape[:2], dtype=torch.bool, device=values.device)
                            if stream_memory_padding_mask is None else stream_memory_padding_mask.to(values.device).bool())
            if summary_mask.shape != stream_memory.shape[:2]:
                raise ValueError('stream summary padding shape mismatch')
            summary = stream_memory.to(values).masked_fill(summary_mask[..., None], 0.)
            values = torch.cat((summary * route[:, None, 2:3], values), dim=1)
            memory_mask = torch.cat((summary_mask, memory_mask), dim=1)
        return (values, memory_mask, curve_prediction) if return_curve else (values, memory_mask)

    def encode_audio(self, audio, condition, *, audio_padding_mask=None, condition_null=None):
        """Encode immutable song evidence once, retaining the checkpoint math."""
        if audio.ndim != 3 or audio.size(-1) != self.config.audio_dim:
            raise ValueError('stream audio shape mismatch')
        batch = audio.size(0)
        if condition.shape != (batch, self.config.condition_dim):
            raise ValueError('stream condition shape mismatch')
        mask = (torch.zeros(audio.shape[:2], dtype=torch.bool, device=audio.device)
                if audio_padding_mask is None else audio_padding_mask.bool().to(audio.device))
        if mask.shape != audio.shape[:2]:
            raise ValueError('stream audio padding shape mismatch')
        null = (torch.zeros(batch, dtype=torch.bool, device=audio.device)
                if condition_null is None else condition_null.reshape(-1).bool().to(audio.device))
        if null.shape != (batch,):
            raise ValueError('stream condition_null shape mismatch')
        c = condition.to(audio.dtype).clone()
        if c.size(1) > 1:
            c[:, 1:] = torch.where(null[:, None], .5, c[:, 1:])
        global_token = self.condition_projection(c) + null[:, None].to(audio.dtype) * self.null_condition
        values = self.audio_projection(audio) + global_token[:, None]
        values = self.audio_norm(values)
        values = self.audio_encoder(values, src_key_padding_mask=mask)
        values = values.masked_fill(mask[..., None], 0.)
        audio_summary = values.masked_fill(mask[..., None], 0.).sum(1) / (~mask).sum(1).clamp_min(1)[:, None]
        route = torch.softmax(self.context_router(global_token + audio_summary), dim=-1)
        values = values * route[:, None, 0:1]
        curve_prediction = self.curve_head(values)
        return values, mask, curve_prediction, route

    def prepare_decode_memory(self, memory, memory_mask, *, history=None, history_padding_mask=None,history_clock=None):
        """Use identical history embeddings in full and cached generation."""
        if history is None or history.size(1) == 0:
            return memory, memory_mask
        if history.ndim != 2 or history.size(0) != memory.size(0) or history.size(1) > self.config.max_history_tokens:
            raise ValueError('stream history shape mismatch')
        positions = torch.arange(history.size(1), device=history.device)[None]
        values = self.token_embedding(history) + self.position_embedding(positions)
        if hasattr(self,'grounding'):
            if history_clock is None or history_clock.shape!=(*history.shape,20):
                raise ValueError('Grounded history requires causal clock features')
            values=values+self.grounding.clock_projection(history_clock.to(values))
        padding = history == 0 if history_padding_mask is None else history_padding_mask.bool()
        if padding.shape != history.shape:
            raise ValueError('stream history mask shape mismatch')
        return torch.cat((values.to(memory), memory), 1), torch.cat((padding, memory_mask), 1)

    def decode(self, decoder_input, memory, memory_mask, *, history=None, history_padding_mask=None,
               history_clock=None,timing_features=None,grounding_context=None):
        if decoder_input.ndim != 2 or decoder_input.size(1) == 0:
            raise ValueError('stream decoder_input must be nonempty [B,L]')
        batch, length = decoder_input.shape
        positions = torch.arange(length, device=decoder_input.device)[None]
        target = self.token_embedding(decoder_input) + self.position_embedding(positions)
        if hasattr(self,'grounding'):
            if timing_features is None or grounding_context is None:
                raise ValueError('Grounded decoding requires causal clock and acoustic queries')
            target=self.grounding.input(target,timing_features,grounding_context['local']+grounding_context['action'])
        memory, memory_mask = self.prepare_decode_memory(memory, memory_mask,
            history=history, history_padding_mask=history_padding_mask,history_clock=history_clock)
        causal = torch.triu(torch.ones((length, length), dtype=torch.bool, device=target.device), diagonal=1)
        hidden = self.decoder(target, memory, tgt_mask=causal,
                              memory_key_padding_mask=memory_mask)
        hidden = self.decoder_norm(hidden)
        routes=None
        if hasattr(self,'grounding'):
            hidden,routes=self.grounding.fuse(hidden,timing_features,grounding_context['local'],
                grounding_context['archive'],grounding_context['global'],grounding_context['action'])
        plan=self.plan_head(hidden)
        if hasattr(self,'grounding'):
            positions=torch.arange(length,device=decoder_input.device)[None].expand(batch,-1)
            boundaries=(decoder_input==1)|(decoder_input==5)
            active=positions.masked_fill(~boundaries,0).cummax(1).values
            intent=plan.gather(1,active[...,None].expand(-1,-1,plan.size(-1)))
            hidden=hidden+self.grounding.intent_projection(intent)
        return {'logits': self.output(hidden), 'plan':plan,'hidden':hidden,'routes':routes}

    def decode_step(self, token, position, memory, memory_mask, caches=None,*,timing_features=None,grounding_context=None):
        """Incremental one-token decode with cached self/cross attention."""
        if self.training:
            raise ValueError('Cached decoding requires eval mode')
        if token.ndim != 2 or token.size(1) != 1 or not 0 <= position < self.position_embedding.num_embeddings:
            raise ValueError('decode_step expects [B,1] and nonnegative position')
        from maichart.generation.event_model import _cached_decoder_layer_with_cross_cache
        hidden = self.token_embedding(token) + self.position_embedding(
            torch.tensor([position], device=token.device)).unsqueeze(0)
        if hasattr(self,'grounding'):
            if timing_features is None or grounding_context is None:
                raise ValueError('Grounded cached decoding requires causal queries')
            hidden=self.grounding.input(hidden,timing_features,grounding_context['local']+grounding_context['action'])
        caches = caches or []
        updated=[]
        for index, layer in enumerate(self.decoder.layers):
            previous = caches[index] if index < len(caches) else None
            if previous is None:
                values = (None, None, None, None)
            elif len(previous) == 2:
                values = (*previous, None, None)
            else:
                values = previous
            hidden, present = _cached_decoder_layer_with_cross_cache(
                layer, hidden, memory, *values, memory_mask)
            updated.append(present)
        hidden = self.decoder_norm(hidden)
        routes=None
        if hasattr(self,'grounding'):
            hidden,routes=self.grounding.fuse(hidden,timing_features,grounding_context['local'],
                grounding_context['archive'],grounding_context['global'],grounding_context['action'])
        plan=self.plan_head(hidden)
        if hasattr(self,'grounding'):
            previous=(caches[len(self.decoder.layers)][0] if len(caches)>len(self.decoder.layers) else None)
            active=plan if previous is None or previous.size(1)==0 else torch.where(
                ((token==1)|(token==5))[...,None],plan,previous[:,-1:])
            hidden=hidden+self.grounding.intent_projection(active)
            history=active if previous is None else torch.cat((previous,active),1)
            updated.append((history,history,None,None))
        return {'logits': self.output(hidden), 'plan':plan,'hidden':hidden,'routes':routes},updated

    def forward(self, batch):
        stream_memory = batch.get('stream_memory')
        summary_mask = batch.get('stream_memory_padding_mask')
        if stream_memory is None and batch.get('history') is not None:
            stream_memory, summary_mask = self.summarize_tokens(batch['history'],
                batch.get('history_padding_mask'), return_padding_mask=True)
        encoded=self.encode_audio(batch['audio'],batch['condition'],
            audio_padding_mask=batch.get('audio_padding_mask'),condition_null=batch.get('condition_null'))
        memory, memory_mask, curve = self.encode(batch['audio'], batch['condition'],
            audio_padding_mask=batch.get('audio_padding_mask'),
            condition_null=batch.get('condition_null'),
            stream_memory=stream_memory, stream_memory_padding_mask=summary_mask,
            action_state=batch.get('action_state'),
            action_state_padding_mask=batch.get('action_state_padding_mask'), return_curve=True,encoded_audio=encoded)
        context=None
        if hasattr(self,'grounding'):
            bank=self.grounding.prepare_bank(batch['detail_audio'],batch['detail_times_sec'],batch['detail_audio_padding_mask'])
            archive={'embeddings':self.grounding.archive_embeddings(batch,self.token_embedding),
                'audio':batch['archive_audio'],'padding':batch['archive_padding_mask'],
                'beats':batch['archive_beats'],'seconds':batch['archive_seconds'],'event_index':batch['archive_event_index']}
            queries=self.grounding.contexts(batch['query_clock'],batch['query_seconds'],batch['query_audio'],bank,archive)
            queries['action']=self.grounding.action_value(batch['query_actions'])*batch['query_actions'][...,-1:].gt(0)
            indices=batch['decoder_query_index'].clamp_min(0)
            context={key:queries[key].gather(1,indices[...,None].expand(-1,-1,self.config.d_model))
                     for key in ('local','archive','action')}
            padding=encoded[1]
            context['global']=encoded[0].sum(1)/(~padding).sum(1).clamp_min(1)[:,None]
        result = self.decode(batch['decoder_input'], memory, memory_mask,
                           history=batch.get('history'), history_padding_mask=batch.get('history_padding_mask'),
                           history_clock=batch.get('history_clock'),timing_features=batch.get('decoder_clock'),
                           grounding_context=context)
        if hasattr(self,'grounding') and 'alignment_event_indices' in batch:
            result.update(self.grounding.alignment(result['hidden'],batch,bank))
        result['curve'] = curve
        return result

    @torch.no_grad()
    def summarize(self, hidden, *, slots=None):
        slots = self.config.max_stream_memory if slots is None else int(slots)
        if hidden.ndim != 3:
            raise ValueError('hidden must be [B,L,D]')
        if hidden.size(1) <= slots:
            return hidden
        stride = math.ceil(hidden.size(1) / slots)
        rows = [hidden[:, i:i+stride].mean(1) for i in range(0, hidden.size(1), stride)]
        return torch.stack(rows, 1)[:, :slots]

    def summarize_tokens(self, tokens, padding_mask=None, *, slots=None, return_padding_mask=False):
        """Deterministic train/generate-compatible long-history summary."""
        if tokens.ndim != 2:
            raise ValueError('tokens must be [B,L]')
        slots = self.config.max_stream_memory if slots is None else int(slots)
        if slots < 1:
            raise ValueError('summary slots must be positive')
        values = self.token_embedding(tokens)
        mask = (tokens == 0 if padding_mask is None else padding_mask.bool())
        if mask.shape != tokens.shape:
            raise ValueError('summary mask shape mismatch')
        values = values.masked_fill(mask[..., None], 0.)
        # Each row uses its own true history length, independent of the other
        # songs in the batch. Scatter pooling avoids a per-row CUDA sync.
        count = min(tokens.size(1), slots)
        lengths = (~mask).sum(1)
        stride = ((lengths + slots - 1) // slots).clamp_min(1)
        group = ((~mask).long().cumsum(1)-1).clamp_min(0) // stride[:, None]
        group = group.clamp_max(max(0, count-1))
        sums = values.new_zeros((tokens.size(0), count, values.size(-1)))
        counts = values.new_zeros((tokens.size(0), count))
        sums.scatter_add_(1, group[..., None].expand_as(values), values)
        counts.scatter_add_(1, group, (~mask).to(values.dtype))
        result = sums / counts.clamp_min(1)[..., None]
        padding = counts == 0
        return (result, padding) if return_padding_mask else result

