"""Shared, audio-only whole-song representation for paired song context."""
import torch
from torch import nn
from torch.nn import functional as F


class SongContext(nn.Module):
    def __init__(self, condition_dim, width=96):
        super().__init__()
        self.width = width
        self.input = nn.Linear(104, width)
        self.input_norm = nn.LayerNorm(width)
        self.attention_norm = nn.LayerNorm(width)
        self.clock = nn.Sequential(nn.Linear(2, width), nn.GELU(), nn.Linear(width, width))
        self.local = nn.ModuleList(nn.Conv1d(width, width, 3, padding=d, dilation=d, groups=width) for d in (1, 2, 4))
        self.attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.fusion = nn.Sequential(nn.Linear(width * 3, width), nn.GELU(), nn.LayerNorm(width))
        self.condition = nn.Linear(condition_dim, width, bias=False)
        self.plan = nn.Linear(width, 6)
        self.phrase = nn.Linear(width, 9)

    def encode(self, audio, padding, clock):
        valid = ~padding.bool()
        if audio.size(1) == 0:
            return audio.new_zeros((*audio.shape[:2], self.width))
        x = self.input_norm(self.input(audio) + self.clock(torch.sign(clock) * torch.log1p(clock.abs())))
        x = x * valid[..., None]
        for layer in self.local:
            x = (x + F.gelu(layer(x.transpose(1, 2)).transpose(1, 2))) * valid[..., None]
        safe = padding.bool().clone()
        empty = safe.all(1)
        safe[empty, 0] = False
        normalized = self.attention_norm(x)
        if x.size(1) > 256:
            # Pool all positions into at most 256 mask-aware summary tokens.
            # No tail or middle of a long song is discarded.
            # Each song's own valid length defines its resolution. A longer
            # padded neighbour must not change this song's music representation.
            ranks = (valid.long().cumsum(1) - 1).clamp_min(0)
            lengths = valid.sum(1, keepdim=True).clamp_min(1)
            bins = torch.div(ranks * 256, lengths, rounding_mode='floor').clamp_max(255)
            assignment = F.one_hot(bins, 256).to(x.dtype) * valid[..., None]
            counts = assignment.sum(1)
            keys = torch.bmm(assignment.transpose(1, 2), normalized) / counts.clamp_min(1)[..., None]
            key_padding = counts.eq(0)
            key_padding[empty, 0] = False
        else:
            keys, key_padding = normalized, safe
        attended, _ = self.attention(normalized, keys, keys, key_padding_mask=key_padding, need_weights=False)
        attended = attended * valid[..., None]
        return self.fusion(torch.cat((x, attended, x - attended), -1)) * valid[..., None]

    def heads(self, state, condition, *, return_rest_logits=False):
        x = state + self.condition(condition)[:, None]
        raw = self.plan(x).float()
        result = (torch.cat((F.softplus(raw[..., :2]), raw[..., 2:].sigmoid()), -1), self.phrase(x).sigmoid())
        return (*result, raw[..., 5]) if return_rest_logits else result
