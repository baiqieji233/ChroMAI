"""Compact event-level composer model with a candidate pointer and action head."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import torch
from torch import nn

from . import INPUT_CONTRACT


@dataclass(slots=True)
class ComposerConfig:
    atom_vocab_size: int
    d_model: int = 192
    nhead: int = 6
    layers: int = 3
    dim_feedforward: int = 576
    dropout: float = .1
    audio_dim: int = 120
    condition_dim: int = 16
    action_dim: int = 50
    history_events: int = 128
    candidate_count: int = 64
    action_slots: int = 8
    candidate_feature_dim: int = 129
    future_dim: int = 8

    def to_dict(self): return asdict(self)


class ChartComposer(nn.Module):
    architecture = 'chart_composer_v1'

    def __init__(self, config: ComposerConfig):
        super().__init__(); self.config=config
        width=config.d_model
        self.audio_projection=nn.Linear(config.audio_dim,width)
        self.condition_projection=nn.Linear(config.condition_dim,width)
        self.action_projection=nn.Linear(config.action_dim,width)
        self.future_projection=nn.Linear(config.future_dim,width)
        self.time_projection=nn.Linear(config.candidate_feature_dim-config.audio_dim,width)
        self.event_embedding=nn.Embedding(config.atom_vocab_size,width)
        self.position_embedding=nn.Embedding(config.history_events+4,width)
        layer=nn.TransformerEncoderLayer(width,config.nhead,config.dim_feedforward,
                                         dropout=config.dropout,activation='gelu',batch_first=True,norm_first=True)
        self.audio_encoder=nn.TransformerEncoder(layer,config.layers)
        self.history_encoder=nn.TransformerEncoder(layer,config.layers)
        self.norm=nn.LayerNorm(width)
        self.pointer=nn.Sequential(nn.Linear(width,width),nn.GELU(),nn.Linear(width,1))
        self.write_head=nn.Linear(width,1)
        self.action_head=nn.Linear(width,config.action_slots*config.atom_vocab_size)
        self.future_head=nn.Linear(width,8)

    def forward(self, candidate_features, history, condition, action_state=None, future_state=None,
                candidate_mask=None, history_padding_mask=None):
        batch=candidate_features.size(0)
        acoustic=candidate_features[...,:self.config.audio_dim]
        candidate_hidden=self.audio_projection(acoustic)+self.time_projection(candidate_features[...,self.config.audio_dim:])
        candidate_hidden=self.audio_encoder(candidate_hidden,
            src_key_padding_mask=candidate_mask)
        valid=~candidate_mask if candidate_mask is not None else torch.ones(candidate_hidden.shape[:2],dtype=torch.bool,device=candidate_hidden.device)
        audio_summary=(candidate_hidden*valid[...,None]).sum(1)/valid.sum(1).clamp_min(1)[:,None]
        if history is None or history.size(1)==0:
            history_summary=audio_summary
        else:
            positions=torch.arange(history.size(1),device=history.device)[None]
            history_hidden=self.history_encoder(self.event_embedding(history)+self.position_embedding(positions),
                src_key_padding_mask=history_padding_mask)
            history_summary=history_hidden[:,-1]
        context=self.norm(history_summary+audio_summary+self.condition_projection(condition))
        if action_state is not None:
            context=context+self.action_projection(action_state)
        if future_state is not None:
            context=context+self.future_projection(future_state)
        pointer=self.pointer(context[:,None,:]+candidate_hidden).squeeze(-1)
        if candidate_mask is not None:pointer=pointer.masked_fill(candidate_mask,-1e4)
        return {'candidate_logits':pointer,'write_logit':self.write_head(context).squeeze(-1),
                'action_logits':self.action_head(context).reshape(batch,self.config.action_slots,self.config.atom_vocab_size),
                'future':self.future_head(context)}
