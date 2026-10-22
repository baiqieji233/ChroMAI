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
    candidate_count: int = 96
    action_slots: int = 8
    candidate_feature_dim: int = 125
    history_feature_dim: int = 8
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
        self.time_projection=nn.Linear(config.candidate_feature_dim-config.audio_dim,width)
        self.event_embedding=nn.Embedding(config.atom_vocab_size,width)
        self.position_embedding=nn.Embedding(config.history_events+4,width)
        self.history_feature_projection=nn.Linear(config.history_feature_dim,width)
        layer=nn.TransformerEncoderLayer(width,config.nhead,config.dim_feedforward,
                                         dropout=config.dropout,activation='gelu',batch_first=True,norm_first=True)
        self.audio_encoder=nn.TransformerEncoder(layer,config.layers,enable_nested_tensor=False)
        self.history_encoder=nn.TransformerEncoder(layer,config.layers,enable_nested_tensor=False)
        self.norm=nn.LayerNorm(width)
        self.pointer=nn.Sequential(nn.Linear(width,width),nn.GELU(),nn.Linear(width,1))
        self.action_head=nn.Linear(width,config.action_slots*config.atom_vocab_size)
        self.future_head=nn.Linear(width,8)
        self.eos_head=nn.Linear(width,1)

    def forward(self, candidate_features, history, condition, action_state=None, future_state=None,
                history_features=None, candidate_mask=None, history_padding_mask=None,
                candidate_indices=None, decode_actions=True):
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
            if history.ndim == 2:
                history = history.unsqueeze(-1)
            positions=torch.arange(history.size(1),device=history.device)[None]
            if history_padding_mask is None:
                lengths=torch.full((history.size(0),),history.size(1),device=history.device,dtype=torch.long)
            else:
                lengths=(~history_padding_mask).sum(1)
            positions=(lengths[:,None]-1-positions).clamp_min(0).clamp_max(self.config.history_events+3)
            event_embeddings=self.event_embedding(history)
            event_embeddings=event_embeddings.masked_fill(history.eq(0).unsqueeze(-1),0.)
            history_hidden=event_embeddings.sum(dim=2)+self.position_embedding(positions)
            if history_features is not None:
                history_hidden=history_hidden+self.history_feature_projection(history_features)
            safe_padding=history_padding_mask
            if safe_padding is not None:
                empty=(~safe_padding).sum(1)==0
                safe_padding=safe_padding.clone()
                history_hidden=history_hidden.clone()
                safe_padding[:,0]=safe_padding[:,0] & ~empty
                history_hidden[:,0]=history_hidden[:,0].masked_fill(empty[:,None],0.)
            history_hidden=self.history_encoder(history_hidden,
                src_key_padding_mask=safe_padding)
            if safe_padding is None:
                lengths=torch.full((history.size(0),),history.size(1)-1,device=history.device,dtype=torch.long)
            else:
                lengths=(~safe_padding).sum(1).clamp_min(1)-1
            gathered=history_hidden[torch.arange(history.size(0),device=history.device),lengths]
            if safe_padding is not None:
                valid_rows=(~history_padding_mask).any(1)
                history_summary=torch.where(valid_rows[:,None],gathered,audio_summary)
            else:
                history_summary=gathered
        context=self.norm(history_summary+audio_summary+self.condition_projection(condition))
        if action_state is not None:
            context=context+self.action_projection(action_state)
        # Future intent is an auxiliary prediction.  It is deliberately not
        # fed back into the current decision: the old teacher-forced path used
        # a target-derived previous state that included the current event,
        # while free generation used the model's own prediction.
        candidate_context=context[:,None,:]+candidate_hidden
        pointer=self.pointer(candidate_context).squeeze(-1)
        if candidate_mask is not None:pointer=pointer.masked_fill(candidate_mask,-1e4)
        result={'candidate_logits':pointer,'candidate_context':candidate_context}
        if decode_actions:
            result.update(self.decode_candidate(candidate_context,candidate_indices))
        return result

    def decode_candidate(self, candidate_context, candidate_indices=None):
        """Project only supervised/chosen candidates without changing the heads.

        Pointer scores still use every candidate. The large action vocabulary
        needs one projection per example during training and free generation,
        rather than one projection for each of the 96 proposed positions.
        """
        if candidate_indices is not None:
            rows=torch.arange(candidate_context.size(0),device=candidate_context.device)
            candidate_context=candidate_context[rows,candidate_indices]
        return {'action_logits':self.action_head(candidate_context).reshape(
                    *candidate_context.shape[:-1],self.config.action_slots,self.config.atom_vocab_size),
                'future':self.future_head(candidate_context),
                'eos_logits':self.eos_head(candidate_context).squeeze(-1)}
