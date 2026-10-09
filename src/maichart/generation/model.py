"""Conditioned frame Transformer used by the first generation tool."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from maichart.generation.conditions import CONDITION_KEYS
from maichart.models.positional_encoding import LearnedPositionalEncoding, SinusoidalPositionalEncoding

AUDIO_FEATURE_KEYS = (
    "onset_strength",
    "rms",
    "percussive_rms",
    "harmonic_rms",
    "spectral_centroid",
    "spectral_bandwidth",
    "zero_crossing_rate",
)


@dataclass(slots=True)
class ConditionedChartTransformerConfig:
    input_dim: int = len(AUDIO_FEATURE_KEYS)
    condition_dim: int = len(CONDITION_KEYS)
    num_note_types: int = 7
    d_model: int = 256
    nhead: int = 4
    num_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    max_len: int = 4096
    positional_encoding: str = "sinusoidal"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ConditionedChartTransformer(nn.Module):
    """Predict chart frame structure from audio and user controls."""

    def __init__(self, config: ConditionedChartTransformerConfig | None = None, **kwargs: Any) -> None:
        super().__init__()
        self.config = config or ConditionedChartTransformerConfig(**kwargs)
        if self.config.input_dim <= 0 or self.config.condition_dim <= 0:
            raise ValueError("input_dim and condition_dim must be positive.")
        if self.config.d_model % self.config.nhead:
            raise ValueError("d_model must be divisible by nhead.")
        self.input_projection = nn.Linear(self.config.input_dim, self.config.d_model)
        self.condition_projection = nn.Sequential(
            nn.Linear(self.config.condition_dim, self.config.d_model),
            nn.GELU(),
            nn.Linear(self.config.d_model, self.config.d_model),
        )
        self.positional_encoding = _make_positional_encoding(
            self.config.positional_encoding,
            d_model=self.config.d_model,
            max_len=self.config.max_len,
            dropout=self.config.dropout,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.nhead,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=self.config.num_layers)
        self.final_norm = nn.LayerNorm(self.config.d_model)
        self.note_presence_head = nn.Linear(self.config.d_model, 1)
        self.button_head = nn.Linear(self.config.d_model, 8)
        self.type_head = nn.Linear(self.config.d_model, self.config.num_note_types)
        self.density_head = nn.Linear(self.config.d_model, 1)

    def forward(self, x: torch.Tensor, condition: torch.Tensor, padding_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"x must have shape [B, T, F], got {tuple(x.shape)}.")
        if condition.ndim != 2 or condition.size(0) != x.size(0):
            raise ValueError("condition must have shape [B, C] matching x batch size.")
        if x.size(-1) != self.config.input_dim:
            raise ValueError(f"Expected {self.config.input_dim} audio features, got {x.size(-1)}.")
        if condition.size(-1) != self.config.condition_dim:
            raise ValueError(f"Expected {self.config.condition_dim} condition values, got {condition.size(-1)}.")
        if padding_mask is not None:
            padding_mask = padding_mask.to(device=x.device, dtype=torch.bool)
        hidden = self.input_projection(x)
        hidden = self.positional_encoding(hidden + self.condition_projection(condition).unsqueeze(1))
        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
        hidden = self.final_norm(hidden)
        return {
            "note_presence_logits": self.note_presence_head(hidden),
            "button_logits": self.button_head(hidden),
            "type_logits": self.type_head(hidden),
            "density_pred": self.density_head(hidden),
        }


def _make_positional_encoding(kind: str, *, d_model: int, max_len: int, dropout: float) -> nn.Module:
    normalized = kind.strip().lower()
    if normalized in {"sin", "sinusoidal", "fixed"}:
        return SinusoidalPositionalEncoding(d_model, max_len=max_len, dropout=dropout)
    if normalized in {"learned", "embedding"}:
        return LearnedPositionalEncoding(d_model, max_len=max_len, dropout=dropout)
    raise ValueError(f"Unknown positional encoding: {kind!r}.")
