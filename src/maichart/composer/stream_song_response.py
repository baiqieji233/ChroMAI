"""Audio-only whole-song response choices; latent modes alter phrase priors only."""
import torch
from torch import nn


class SongResponse(nn.Module):
    def __init__(self,d_model,phrase_width,condition_dim,modes=4,phrase_modes=4,*,unified=False):
        super().__init__()
        self.unified=bool(unified)
        self.modes=modes
        self.phrase_modes=phrase_modes
        self.global_audio=nn.Sequential(nn.LayerNorm(d_model),nn.Linear(d_model,128),nn.GELU())
        if not self.unified:
            self.song=nn.Sequential(nn.Linear(128+condition_dim,128),nn.GELU(),nn.Linear(128,modes))
        self.local_audio=nn.Linear(phrase_width,128)
        self.phrase=nn.Sequential(nn.Linear(128*3+condition_dim+1,128),nn.GELU(),nn.Linear(128,modes*phrase_modes))
        nn.init.normal_(self.phrase[-1].weight,std=.005)
        nn.init.zeros_(self.phrase[-1].bias)

    def pool_audio(self,audio,audio_mask):
        valid=audio_mask.to(audio.dtype)[...,None]
        pooled=(audio*valid).sum(-2)/valid.sum(-2).clamp_min(1)
        return self.global_audio(pooled)

    def song_prior(self,audio,audio_mask,condition,global_audio=None,*,prior_logits=None):
        if prior_logits is not None:return prior_logits.float()
        if self.unified:raise ValueError('Unified response requires the shared profile prior')
        global_audio=self.pool_audio(audio,audio_mask) if global_audio is None else global_audio
        return self.song(torch.cat((global_audio,condition),-1)).float()

    def forward(self,audio,audio_mask,condition,anchor,position=None,base_prior=None,global_audio=None,*,prior_logits=None,causal_condition=None):
        # Caller may cache this projection with the immutable encoded audio bank.
        global_audio=self.pool_audio(audio,audio_mask) if global_audio is None else global_audio
        if self.unified and causal_condition is None:raise ValueError('Unified response requires masked causal controls')
        condition=condition if causal_condition is None else causal_condition
        song_logits=self.song_prior(audio,audio_mask,condition,global_audio=global_audio,prior_logits=prior_logits)
        local=self.local_audio(anchor)
        global_rows=global_audio[:,None,:].expand_as(local)
        conditions=condition[:,None,:].expand(*local.shape[:-1],condition.size(-1))
        if position is None:position=local.new_zeros(local.shape[:-1])
        features=torch.cat((local,global_rows,local-global_rows,conditions,position[...,None].to(local)),-1)
        phrase_logits=self.phrase(features).reshape(*local.shape[:-1],self.modes,self.phrase_modes).float()
        if base_prior is not None:phrase_logits=phrase_logits+base_prior[...,None,:]
        return dict(song_response_logits=song_logits,song_response_phrase_logits=phrase_logits)
