"""Causal per-button timing and load, with learned soft output scores."""
from __future__ import annotations

import torch
from torch import nn

BUTTON_LOAD_FEATURE_DIM = 10
BUTTON_LOAD_FEATURE_DIM_V2 = 23
BUTTON_LOAD_TAUS = (.5, 2., 8.)


def button_load_features(history_heads, ages_beats, valid, head_timing, candidate_gap_beats,
                         history_second_ages=None, candidate_gap_seconds=None, *, extended=False):
    """Return [B,C,8,10]; chords contribute one activation to every button.

    Historical ages are measured at the committed-history origin; candidate
    gaps advance that clock. Timing already contains candidate-relative ages.
    The 99-pad compatibility layout merges buttons and Touch A pads, while
    123 separates them. In both layouts buttons occupy the first eight pads.
    """
    if head_timing.ndim != 3 or head_timing.shape[-1] not in (99, 123):
        raise ValueError('head_timing requires [B,C,99/123]')
    b, c, dim = head_timing.shape
    if history_heads.ndim != 3 or history_heads.shape[0] != b or history_heads.shape[-1] != 8:
        raise ValueError('history_heads requires [B,H,8]')
    if ages_beats.shape != history_heads.shape[:2] or valid.shape != ages_beats.shape:
        raise ValueError('History ages and visibility must match [B,H]')
    if candidate_gap_beats.shape != (b, c):
        raise ValueError('candidate_gap_beats requires [B,C]')
    raw = head_timing.float()
    pads = dim // 3
    seen = raw[..., :8].clamp(0., 1.)
    log_sec = raw[..., pads:pads + 8].clamp_min(0.)
    log_beat = raw[..., 2*pads:2*pads + 8].clamp_min(0.)
    # Bounded, smooth encodings preserve both short and longer intervals.
    sec = torch.expm1(log_sec.clamp_max(20.))
    beat = torch.expm1(log_beat.clamp_max(20.))
    intervals = [seen, log_sec / (1. + log_sec) * seen,
                 log_beat / (1. + log_beat) * seen,
                 torch.exp(-sec / .08) * seen,
                 torch.exp(-sec / .25) * seen,
                 torch.exp(-beat / .25) * seen,
                 torch.exp(-beat) * seen]
    heads = history_heads.to(raw) * valid.to(raw).unsqueeze(-1)
    safe_ages = ages_beats.to(raw).masked_fill(~valid.bool(), 0.)
    age = (None if extended else
           (safe_ages[:, None, :] + candidate_gap_beats.to(raw)[:, :, None]).clamp_min(0.))
    for tau in BUTTON_LOAD_TAUS:
        if extended:
            decay = torch.exp(-safe_ages.clamp_min(0.) / tau)
            advance = torch.exp(-candidate_gap_beats.to(raw).clamp_min(0.) / tau).unsqueeze(-1)
            load = torch.einsum('bh,bhl->bl', decay, heads)[:,None] * advance
        else:
            load = torch.einsum('bch,bhl->bcl', torch.exp(-age / tau), heads)
        logged = torch.log1p(load)
        intervals.append(logged / (1. + logged))
    if extended:
        # Do not infer seconds from a beat gap: tempo may change inside it.
        second_loads = []
        chord_loads = []
        adjacent_loads = []
        if history_second_ages is not None and candidate_gap_seconds is not None:
            if history_second_ages.shape != ages_beats.shape or candidate_gap_seconds.shape != (b, c):
                raise ValueError('Second ages/gaps must match [B,H] and [B,C]')
            seconds_age = history_second_ages.to(raw).masked_fill(~valid.bool(), 0.).clamp_min(0.)
            gaps = candidate_gap_seconds.to(raw).clamp_min(0.)
            chord_heads = heads * (heads.sum(-1, keepdim=True) - 1.).clamp_min(0.)
            for tau in (.08, .18, .45, 1.2):
                # Separable exponential avoids a [B,C,H] temporary.
                decay = torch.exp(-seconds_age / tau)
                advance = torch.exp(-gaps / tau).unsqueeze(-1)
                load = torch.einsum('bh,bhl->bl', decay, heads)[:, None] * advance
                chord = torch.einsum('bh,bhl->bl', decay, chord_heads)[:, None] * advance
                adjacent = load.roll(1, -1) + load.roll(-1, -1)
                for output, value in ((second_loads, load), (chord_loads, chord), (adjacent_loads, adjacent)):
                    logged = torch.log1p(value)
                    output.append(logged / (1. + logged))
        else:
            zero = raw.new_zeros((b,c,8))
            second_loads = [zero] * 4
            chord_loads = [zero] * 4
            adjacent_loads = [zero] * 4
        # Consecutive committed events containing each key, ignoring padding.
        positions=torch.arange(heads.shape[1],device=heads.device)[None,:,None]
        missing=valid[...,None].bool() & ~heads.bool()
        last_missing=positions.expand_as(heads).masked_fill(~missing,-1).amax(1) if heads.size(1) else heads.new_full((b,8),-1)
        run=(heads * (positions>last_missing[:,None])).sum(1)
        run = (run / (1. + run))[:,None].expand(-1,c,-1)
        intervals.extend(second_loads + chord_loads + [run] + adjacent_loads)
    return torch.stack(intervals, dim=-1)


class PerButtonLoadV2(nn.Module):
    """Shared feature/context interaction; no sign restriction or hand labels."""
    def __init__(self, width, *, feature_dim=BUTTON_LOAD_FEATURE_DIM_V2, kinds=3):
        super().__init__()
        hidden = min(64, int(width))
        self.feature_dim = int(feature_dim)
        self.kinds = int(kinds)
        self.features = nn.Linear(self.feature_dim, hidden)
        self.context = nn.Linear(width, hidden, bias=False)
        self.output = nn.Linear(hidden, self.kinds, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, features, context):
        if features.shape[-2:] != (8, self.feature_dim):
            raise ValueError('Button features have incompatible dimensions')
        # Candidate features can be shared across a subsequent slot dimension.
        while features.ndim < context.ndim + 1:
            features = features.unsqueeze(-3)
        hidden = torch.tanh(self.features(features.to(context)) + self.context(context).unsqueeze(-2))
        if self.kinds == 1:
            return 8. * self.output(hidden).squeeze(-1) * features[...,0]
        scores = 8. * self.output(hidden).transpose(-1,-2)
        visible = features[...,0].ne(0)
        scores = scores * visible.unsqueeze(-2)
        # Center over visible keys and keep invisible keys exactly zero.
        count = visible.sum(-1,keepdim=True).clamp_min(1).unsqueeze(-2)
        mean = scores.sum(-1,keepdim=True) / count
        return (scores - mean) * visible.unsqueeze(-2)


class PerButtonLoad(PerButtonLoadV2):
    """Original ten-feature single-score head, preserving checkpoint shapes."""
    def __init__(self, width):
        super().__init__(width, feature_dim=BUTTON_LOAD_FEATURE_DIM, kinds=1)
