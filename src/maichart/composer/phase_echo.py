"""Causal phase-echo evidence shared by diagnostics and the learned prior."""
from __future__ import annotations

import torch
from torch import nn

ECHO_LAGS = (2, 3, 4, 6, 8, 12, 16)
ECHO_TRANSFORMS = ('identity', 'mirror_lr', 'rotate_180', 'mirror_ud')
ECHO_PERMUTATIONS = (tuple(range(8)), tuple(range(7, -1, -1)),
                     (4, 5, 6, 7, 0, 1, 2, 3), (3, 2, 1, 0, 7, 6, 5, 4))
ECHO_TOLERANCE = 1e-3
ECHO_PARAM_SCALE = 8.


def phase_echo_features(lanes, ages, valid, cand_gap):
    """Return evidence and exact source sets; never read future chart events.

    Histories are chronological, ages measured from the last committed event.
    ``valid`` describes actual history padding, never stochastic dropout.
    """
    lanes, valid = lanes.bool(), valid.bool()
    ages, cand_gap = ages.float(), cand_gap.float()
    b, h, _ = lanes.shape
    c = cand_gap.size(1)
    lag = ages.new_tensor(ECHO_LAGS)
    k = len(ECHO_LAGS)
    permutation = torch.tensor(ECHO_PERMUTATIONS, device=lanes.device)
    button = lanes.any(-1) & valid
    counts = button.sum(-1)
    if not h:
        return dict(evidence=torch.zeros(b,k,4,device=lanes.device,dtype=torch.long),
                    rhythm_evidence=torch.zeros(b,k,device=lanes.device,dtype=torch.long),
                    candidate_source=torch.zeros(b,c,k,8,device=lanes.device,dtype=torch.bool),
                    has_candidate_source=torch.zeros(b,c,k,device=lanes.device,dtype=torch.bool),
                    rhythm_prediction=torch.zeros(b,c,k,device=lanes.device,dtype=torch.bool),
                    has_rhythm_source=torch.zeros(b,k,device=lanes.device,dtype=torch.bool),
                    rhythm_gap=ages.new_zeros(b,k),button_count=counts,event_count=valid.sum(-1))
    # Only the last four button events contribute to E; Touch-only events
    # still contribute to the independent last-four-event rhythm window.
    # Select bounded queries before matching against the full history, so the
    # matching tensor is [B,K,2*min(4,H),H], never [B,K,H,H].
    positions = torch.arange(h, device=lanes.device)
    q = min(4, h)
    button_index = positions[None].expand(b,h).masked_fill(~button, -1).topk(q, dim=1).values
    event_index = positions[None].expand(b,h).masked_fill(~valid, -1).topk(q, dim=1).values
    query_index = torch.cat((button_index, event_index), dim=1)
    query_valid = query_index >= 0
    query_age = ages.gather(1, query_index.clamp_min(0))
    pair = ((ages[:,None,None,:] - query_age[:,None,:,None]
             - lag[None,:,None,None]).abs() < ECHO_TOLERANCE)
    pair = pair & query_valid[:,None,:,None] & valid[:,None,None,:]
    has_pair = pair.any(-1)
    source = torch.einsum('bkji,bil->bkjl', pair[:,:,:q].float(), lanes.float()) > 0
    transformed = source[..., permutation].permute(0,1,3,2,4)
    query_lanes = lanes.gather(1, button_index.clamp_min(0)[...,None].expand(b,q,8))
    copied = (transformed == query_lanes[:,None,None]).all(-1) & has_pair[:,:,:q][:,:,None]
    evidence = copied.sum(-1)
    rhythm_evidence = has_pair[:,:,q:].sum(-1)
    wanted_age = lag[None,None] - cand_gap[...,None]
    matches = ((wanted_age[...,None] - ages[:,None,None]).abs() < ECHO_TOLERANCE)
    matches = matches & button[:,None,None] & (wanted_age >= 0)[...,None]
    candidate_source = torch.einsum('bcki,bil->bckl', matches.float(), lanes.float()) > 0
    has_source = matches.any(-1)
    anchors = ((ages[:,None] - lag[None,:,None]).abs() < ECHO_TOLERANCE) & valid[:,None]
    anchor_index = positions[None,None].expand(b,k,h).masked_fill(~anchors, -1).max(-1).values
    following = valid[:,None] & (positions[None,None] > anchor_index[...,None])
    next_index = positions[None,None].expand(b,k,h).masked_fill(~following, h).min(-1).values
    rhythm_source = (anchor_index >= 0) & (next_index < h)
    next_age = ages.gather(1, next_index.clamp_max(h-1))
    rhythm_gap = lag[None] - next_age
    rhythm_prediction = ((cand_gap[...,None] - rhythm_gap[:,None]).abs() < ECHO_TOLERANCE) & rhythm_source[:,None]
    return dict(evidence=evidence, rhythm_evidence=rhythm_evidence,
                candidate_source=candidate_source,has_candidate_source=has_source,
                rhythm_prediction=rhythm_prediction,has_rhythm_source=rhythm_source,
                rhythm_gap=rhythm_gap,button_count=counts,event_count=valid.sum(-1))


class PhaseEchoPrior(nn.Module):
    """Linear zero-initialized evidence calibration, shared across all lags."""
    def __init__(self, width):
        super().__init__()
        self.table_E=nn.Parameter(torch.zeros(5))
        self.lag_bias=nn.Parameter(torch.zeros(len(ECHO_LAGS)))
        self.tf_bias=nn.Parameter(torch.zeros(4))
        self.ctx=nn.Linear(width,len(ECHO_LAGS)*4)
        self.rtable=nn.Parameter(torch.zeros(5))
        self.rlag=nn.Parameter(torch.zeros(len(ECHO_LAGS)))
        self.rctx=nn.Linear(width,len(ECHO_LAGS))
        for layer in (self.ctx,self.rctx):
            nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
        self.register_buffer('permutations',torch.tensor(ECHO_PERMUTATIONS),persistent=False)

    def forward(self, lanes, ages, valid, cand_gap, audio_context, symbolic_keep):
        f=phase_echo_features(lanes,ages,valid,cand_gap)
        b,c=audio_context.shape[:2]
        weight=(self.table_E[f['evidence']][:,None]+self.lag_bias[None,None,:,None]
                +self.tf_bias[None,None,None]+self.ctx(audio_context).float().reshape(b,c,len(ECHO_LAGS),4))
        weight=ECHO_PARAM_SCALE*weight*f['has_candidate_source'][...,None]
        source=f['candidate_source'][...,self.permutations].float()
        lane=(weight[...,None]*source).sum((2,3))*symbolic_keep[:,None].float()
        rhythm=(self.rtable[f['rhythm_evidence']][:,None]+self.rlag[None,None]
                +self.rctx(audio_context).float())
        rhythm=ECHO_PARAM_SCALE*(rhythm*f['rhythm_prediction']).sum(-1)*symbolic_keep.float()
        return lane,rhythm
