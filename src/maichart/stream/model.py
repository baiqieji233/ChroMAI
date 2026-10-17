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

    def __post_init__(self):
        for name in ('vocab_size','d_model','nhead','layers','dim_feedforward',
                     'audio_dim','condition_dim','max_audio_tokens',
                     'max_history_tokens','max_stream_memory','max_tokens','plan_dim'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.d_model % self.nhead:
            raise ValueError('d_model must be divisible by nhead')
        if not 0 <= self.dropout < 1:
            raise ValueError('dropout must be in [0, 1)')


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
        self.memory_projection = nn.Linear(w, w)
        self.plan_projection = nn.Linear(config.plan_dim, w)
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
        self.event_gate = nn.Linear(w, 1)
        self.curve_head = nn.Linear(w, 4)

    def encode(self, audio, condition, *, audio_padding_mask=None,
               condition_null=None, stream_memory=None, action_state=None,
               action_state_padding_mask=None, return_curve=False):
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
        if action_state is not None:
            if action_state.ndim != 3 or action_state.shape[0] != batch or action_state.size(-1) != self.config.action_dim:
                raise ValueError('stream action_state shape mismatch')
            action_mask = (action_state_padding_mask.bool() if action_state_padding_mask is not None
                           else torch.zeros(action_state.shape[:2], dtype=torch.bool, device=audio.device))
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
            values = torch.cat((stream_memory.to(values) * route[:, None, 2:3], values), dim=1)
            memory_mask = torch.cat((torch.zeros(stream_memory.shape[:2], dtype=torch.bool, device=values.device), memory_mask), dim=1)
        return (values, memory_mask, curve_prediction) if return_curve else (values, memory_mask)

    def decode(self, decoder_input, memory, memory_mask, *, history=None, history_padding_mask=None):
        if decoder_input.ndim != 2 or decoder_input.size(1) == 0:
            raise ValueError('stream decoder_input must be nonempty [B,L]')
        batch, length = decoder_input.shape
        positions = torch.arange(length, device=decoder_input.device)[None]
        target = self.token_embedding(decoder_input) + self.position_embedding(positions)
        if history is not None:
            if history.ndim != 2 or history.shape[0] != batch or history.size(1) > self.config.max_history_tokens:
                raise ValueError('stream history shape mismatch')
            hist_pos = torch.arange(history.size(1), device=history.device)[None]
            hist = self.token_embedding(history) + self.position_embedding(hist_pos)
            hmask = (history_padding_mask.bool() if history_padding_mask is not None
                     else history == 0)
            if hmask.shape != history.shape:
                raise ValueError('stream history mask shape mismatch')
            memory = torch.cat((hist, memory), dim=1)
            memory_mask = torch.cat((hmask, memory_mask), dim=1)
        causal = torch.triu(torch.ones((length, length), dtype=torch.bool, device=target.device), diagonal=1)
        hidden = self.decoder(target, memory, tgt_mask=causal,
                              memory_key_padding_mask=memory_mask)
        hidden = self.decoder_norm(hidden)
        return {'logits': self.output(hidden), 'plan': self.plan_head(hidden),
                'event_gate': self.event_gate(hidden).squeeze(-1), 'hidden': hidden}

    def decode_step(self, token, position, memory, memory_mask, caches=None):
        """Incremental one-token decode with cached self/cross attention."""
        if token.ndim != 2 or token.size(1) != 1 or position < 0:
            raise ValueError('decode_step expects [B,1] and nonnegative position')
        from maichart.generation.event_model import _cached_decoder_layer_with_cross_cache
        hidden = self.token_embedding(token) + self.position_embedding(
            torch.tensor([position], device=token.device)).unsqueeze(0)
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
        return {'logits': self.output(hidden), 'plan': self.plan_head(hidden),
                'event_gate': self.event_gate(hidden).squeeze(-1), 'hidden': hidden}, updated

    def forward(self, batch):
        stream_memory = batch.get('stream_memory')
        if stream_memory is None and batch.get('history') is not None:
            stream_memory = self.summarize_tokens(batch['history'], batch.get('history_padding_mask'))
        memory, memory_mask, curve = self.encode(batch['audio'], batch['condition'],
            audio_padding_mask=batch.get('audio_padding_mask'),
            condition_null=batch.get('condition_null'),
            stream_memory=stream_memory, action_state=batch.get('action_state'),
            action_state_padding_mask=batch.get('action_state_padding_mask'), return_curve=True)
        result = self.decode(batch['decoder_input'], memory, memory_mask,
                           history=batch.get('history'), history_padding_mask=batch.get('history_padding_mask'))
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

    def summarize_tokens(self, tokens, padding_mask=None, *, slots=None):
        """Deterministic train/generate-compatible long-history summary."""
        if tokens.ndim != 2:
            raise ValueError('tokens must be [B,L]')
        slots = self.config.max_stream_memory if slots is None else int(slots)
        values = self.token_embedding(tokens)
        mask = (tokens == 0 if padding_mask is None else padding_mask.bool())
        values = values.masked_fill(mask[..., None], 0.)
        if values.size(1) <= slots:
            return values
        stride = math.ceil(values.size(1) / slots)
        rows=[]
        for i in range(0, values.size(1), stride):
            local = values[:, i:i+stride]
            local_mask = mask[:, i:i+stride]
            rows.append(local.sum(1) / (~local_mask).sum(1).clamp_min(1)[:, None])
        return torch.stack(rows, 1)[:, :slots]

