"""Causal phrase rate envelopes from music and actual NOTE evidence.

There is no WAIT token/count, decoder hidden, or future chart input here.
The small intent readout reuses the existing phrase music bank. Historical
evidence fades with its age; a silent archive contributes exactly zero.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class PhraseArrangementRate(nn.Module):
    def __init__(self,phrase_width,condition_dim,width=64):
        super().__init__()
        # Type counts, physical button counts, head amount and NOTE amount.
        # Gap bins, WAIT counts and WAIT-reset motion statistics are excluded.
        self.register_buffer('note_columns',torch.tensor([*range(13),22,23]),persistent=False)
        self.music=nn.Sequential(nn.Linear(phrase_width+condition_dim,width),nn.GELU(),nn.LayerNorm(width))
        self.history=nn.Sequential(nn.Linear(60,width),nn.GELU(),nn.LayerNorm(width))
        self.envelope=nn.Linear(width,4)
        self.progress=nn.Sequential(nn.Linear(60,width),nn.GELU(),nn.LayerNorm(width))
        self.closing=nn.Sequential(nn.Linear(2*width+4,width),nn.GELU(),nn.Linear(width,1))
        for layer in (self.envelope,self.closing[-1]):
            nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)

    def forward(self,phrase,batch,condition=None):
        anchor=phrase['phrase_music_anchor'].float()
        condition=batch['condition'] if condition is None else condition
        controls=condition.float()[:,None].expand(*anchor.shape[:-1],condition.size(-1))
        memory=torch.nan_to_num(batch['phrase_memory_features'].float(),nan=0.,posinf=0.,neginf=0.)
        memory=memory.reshape(*memory.shape[:-1],4,24)
        visible=(batch['phrase_memory_mask'].bool()
            & (batch['phrase_memory_end_beat']<=batch['phrase_start_beat'][...,None])
            & (batch['phrase_memory_end_beat']>batch['phrase_memory_start_beat'])
            & memory[...,23].gt(0).any(-1))
        span=(batch['phrase_end_beat']-batch['phrase_start_beat']).clamp_min(1e-8)
        age=(batch['phrase_start_beat'][...,None]-batch['phrase_memory_end_beat']).clamp_min(0)
        evidence=(-age/span[...,None]).exp().float()*visible
        features=memory.index_select(-1,self.note_columns).flatten(-2)
        mass=evidence.sum(-1)
        pooled=(features*evidence[...,None]).sum(-2)/mass.clamp_min(1e-8)[...,None]
        support=mass/(1+mass)
        # A phrase-entry intent: completed references and music stay fixed
        # within a phrase, regardless of how many WAITs have been sampled.
        intent=self.music(torch.cat((anchor,controls),-1))+support[...,None]*self.history(pooled)
        end=torch.minimum(batch['step_beat']+2,batch['phrase_end_beat'])
        end=torch.minimum(end,batch['audio_end_beat'][:,None])
        position=((.5*(batch['step_beat']+end)-batch['phrase_start_beat'])/span).float().clamp(0,1)
        left=1-position
        basis=torch.stack((left**3,3*position*left**2,3*position**2*left,position**3),-1)
        raw=(self.envelope(intent)*basis).sum(-1)
        progress=torch.nan_to_num(batch['phrase_progress'].float(),nan=0.,posinf=0.,neginf=0.)
        progress=progress.reshape(*progress.shape[:-1],4,24)
        has_notes=progress[...,23].gt(0).any(-1)
        written=self.progress(progress.index_select(-1,self.note_columns).flatten(-2))
        closing=self.closing(torch.cat((intent,written,basis),-1)).squeeze(-1)*has_notes
        # Smooth, positive, identity at zero; no upper rate cap and no exp
        # amplification for large positive readouts.
        gain=F.softplus((raw+closing).float())/math.log(2.)
        return dict(gain=gain,history_support=support,progress_has_notes=has_notes,
                    intent=intent,phrase_end_beat=batch['phrase_end_beat'])
