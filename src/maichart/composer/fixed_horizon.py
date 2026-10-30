"""Linear-cost, fixed eight-beat audio probes for configuration decoding."""
from __future__ import annotations

import math

import torch
from torch import nn


class FixedFutureAudio(nn.Module):
    """Aggregate ten actual audio probes per candidate into a zero-init residual.

    Inputs are batched quality-context tensors. ``quality_future_mask`` uses
    True for valid audio; masked probes contribute exactly zero, including
    candidates entirely beyond the audio end. No candidate-to-candidate
    attention or learned-state cache is used.
    """

    def __init__(self, width: int):
        super().__init__()
        if width <= 0:
            raise ValueError('width must be positive')
        self.width = int(width)
        hidden = min(self.width, 64)
        self.norm = nn.LayerNorm(self.width)
        self.query = nn.Linear(self.width, hidden, bias=False)
        self.audio = nn.Linear(104, hidden)
        self.offset = nn.Linear(4, hidden, bias=False)
        self.output = nn.Linear(hidden, self.width, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, current_context: torch.Tensor, inputs: dict) -> torch.Tensor:
        if current_context.ndim != 3 or current_context.size(-1) != self.width:
            raise ValueError('current_context must have shape [B, C, width]')
        audio = inputs['quality_future_audio']
        valid = inputs['quality_future_mask'].bool()
        offsets = inputs['quality_future_offsets']
        batch, count, _ = current_context.shape
        if audio.shape != (batch, count, 10, 104) or valid.shape != (batch, count, 10):
            raise ValueError('future audio/mask must have shape [B, C, 10, 104] / [B, C, 10]')
        if offsets.shape == (10,):
            offsets = offsets.unsqueeze(0).expand(batch, -1)
        if offsets.shape != (batch, 10):
            raise ValueError('future offsets must have shape [10] or [B, 10]')
        offsets = offsets.to(device=current_context.device, dtype=current_context.dtype)
        features = torch.stack((offsets / 8., torch.log1p(offsets) / math.log(9.),
                                torch.sin(math.pi * offsets), torch.cos(math.pi * offsets)), dim=-1)
        # Remove masked values before projection as well, so even NaN padding
        # cannot contaminate a valid residual through a zero multiplication.
        clean = audio.to(dtype=current_context.dtype).masked_fill(~valid.unsqueeze(-1), 0.)
        values = torch.tanh(self.audio(clean) + self.offset(features)[:, None, :, :])
        query = self.query(self.norm(current_context))
        scores = (query.unsqueeze(-2) * values).sum(-1) / math.sqrt(values.size(-1))
        weights = scores.float().masked_fill(~valid, -1e4).softmax(-1) * valid
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        pooled = (weights.to(values.dtype).unsqueeze(-1) * values).sum(-2)
        return self.output(pooled)
