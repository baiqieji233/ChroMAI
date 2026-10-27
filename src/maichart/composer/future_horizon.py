"""Future candidate audio horizon cross-attention for configuration decoding.

In ``audio_grounded_v3``, the timing pointer inspects all ``C`` candidate beats,
but configuration decoding previously read only the single vector
``audio_context[b, chosen]`` at the selected onset.  Choosing between a Tap, a
Hold of a given length, a Slide trajectory/duration, or a multi-note chord
depends on how the music unfolds over the *next* several beats after the onset.

This module lets each candidate's configuration context attend over valid
future candidate audio states within ``horizon_beats`` (default 8.0 beats, two
4/4 bars), using a factorized relative beat/second offset encoding that avoids
allocating a 4D ``[B, C, C, D]`` tensor.  Output projection is zero-initialized
so newly migrated checkpoints preserve their exact baseline outputs at step 0.
"""
from __future__ import annotations

import math
import torch
from torch import nn

FUTURE_HORIZON_POLICIES = frozenset({'off', 'candidate_horizon_v1'})
HORIZON_REL_DIM = 10
DEFAULT_HORIZON_BEATS = 8.0


def relative_horizon_features(delta_beats: torch.Tensor, delta_seconds: torch.Tensor,
                              horizon_beats: float = DEFAULT_HORIZON_BEATS) -> torch.Tensor:
    """Return ``[B, Q, K, 10]`` relative features from query beat/sec to key beat/sec.

    ``delta_beats`` has shape ``[B, C]`` (or ``[B, Q]`` and ``[B, K]`` when
    passed as a tuple). Beat-distance and phase values are bounded by one;
    the seconds feature is capped at ``log(31) / log(9)`` (about 1.56).
    """
    if isinstance(delta_beats, tuple):
        q_beats, k_beats = delta_beats
        q_secs, k_secs = delta_seconds
    else:
        q_beats = k_beats = delta_beats
        q_secs = k_secs = delta_seconds
    db = (k_beats[:, None, :] - q_beats[:, :, None]).float().clamp(0.0, float(horizon_beats))
    ds = (k_secs[:, None, :] - q_secs[:, :, None]).float().clamp(0.0, 30.0)
    scale = max(float(horizon_beats), 1e-6)
    norm_b = db / scale
    log_b = torch.log1p(db) / math.log1p(scale)
    log_s = torch.log1p(ds) / math.log1p(8.0)
    ang1 = (2.0 * math.pi) * db
    ang2 = math.pi * db
    ang4 = (0.5 * math.pi) * db
    self_flag = (db <= 1e-6).to(dtype= torch.float32)
    return torch.stack((
        norm_b, log_b, log_s,
        torch.sin(ang1), torch.cos(ang1),
        torch.sin(ang2), torch.cos(ang2),
        torch.sin(ang4), torch.cos(ang4),
        self_flag,
    ), dim=-1)


class FutureAudioHorizon(nn.Module):
    """Multi-head cross-attention over future candidate audio states in ``[0, horizon_beats]``."""

    def __init__(self, width: int, nhead: int = 4, *, horizon_beats: float = DEFAULT_HORIZON_BEATS):
        super().__init__()
        if width <= 0 or nhead <= 0 or width % nhead != 0:
            raise ValueError(f'FutureAudioHorizon width ({width}) must be positive and divisible by nhead ({nhead})')
        if not math.isfinite(float(horizon_beats)) or float(horizon_beats) <= 0.0:
            raise ValueError('horizon_beats must be finite and positive')
        self.width = int(width)
        self.nhead = int(nhead)
        self.head_dim = self.width // self.nhead
        self.horizon_beats = float(horizon_beats)

        self.norm = nn.LayerNorm(self.width)
        self.q_proj = nn.Linear(self.width, self.width, bias=False)
        self.k_proj = nn.Linear(self.width, self.width, bias=False)
        self.v_proj = nn.Linear(self.width, self.width, bias=False)
        self.rel_key = nn.Linear(self.width, self.nhead * HORIZON_REL_DIM, bias=False)
        self.rel_value = nn.Linear(self.nhead * HORIZON_REL_DIM, self.width, bias=False)
        self.output = nn.Linear(self.width, self.width, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, audio_context: torch.Tensor, candidate_features: torch.Tensor,
                valid_mask: torch.Tensor, *, audio_dim: int = 120) -> torch.Tensor:
        """Compute future-horizon residual ``[B, C, width]`` over valid candidates."""
        if audio_context.ndim != 3 or audio_context.size(-1) != self.width:
            raise ValueError('audio_context must have shape [B, C, width]')
        batch, count, _ = audio_context.shape
        if candidate_features.shape[:2] != (batch, count) or candidate_features.size(-1) < audio_dim + 2:
            raise ValueError('candidate_features must match [B, C, >=audio_dim+2]')
        if valid_mask.shape != (batch, count):
            raise ValueError('valid_mask must have shape [B, C]')

        delta_beats = candidate_features[..., audio_dim].float()
        delta_secs = candidate_features[..., audio_dim + 1].float()
        normed = self.norm(audio_context)

        q = self.q_proj(normed).view(batch, count, self.nhead, self.head_dim).transpose(1, 2).float()
        k = self.k_proj(normed).view(batch, count, self.nhead, self.head_dim).transpose(1, 2).float()
        v = self.v_proj(normed).view(batch, count, self.nhead, self.head_dim).transpose(1, 2).float()

        # Content scores: [B, H, Q, K]
        scale = 1.0 / math.sqrt(self.head_dim)
        content_scores = torch.matmul(q, k.transpose(-1, -2)) * scale

        # Factorized relative position scores: [B, H, Q, K] without [B, Q, K, width]
        phi = relative_horizon_features(delta_beats, delta_secs, self.horizon_beats)
        q_rel = self.rel_key(normed).view(batch, count, self.nhead, HORIZON_REL_DIM).float()
        rel_scores = torch.einsum('bihp,bijp->bhij', q_rel, phi) * scale

        # Future window mask: key is valid AND 0 <= beat_k - beat_q <= horizon_beats.
        # Diagonal (j == i) is always enabled so even padded/isolated rows never NaN.
        db_raw = delta_beats[:, None, :] - delta_beats[:, :, None]
        eye = torch.eye(count, dtype=torch.bool, device=audio_context.device).unsqueeze(0)
        in_window = (db_raw >= -1e-6) & (db_raw <= self.horizon_beats) & valid_mask[:, None, :]
        allow = in_window | eye

        scores = (content_scores + rel_scores).masked_fill(~allow[:, None, :, :], -1e4)
        weights = scores.softmax(dim=-1)

        # Aggregate content values [B, H, Q, head_dim] and relative features [B, H, Q, 10]
        content_out = torch.matmul(weights, v).transpose(1, 2).reshape(batch, count, self.width)
        rel_expected = torch.einsum('bhij,bijp->bhip', weights, phi).transpose(1, 2).reshape(
            batch, count, self.nhead * HORIZON_REL_DIM)
        combined = content_out.to(normed.dtype) + self.rel_value(rel_expected.to(normed.dtype))
        out = self.output(combined)
        return out * valid_mask.unsqueeze(-1).to(out.dtype)
