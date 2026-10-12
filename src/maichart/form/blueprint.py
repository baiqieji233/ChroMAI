"""Full-song rhythm and note-form blueprint model and chart-derived targets.

The model predicts chart onsets and broad note-form cues from complete-song
audio. Acoustic onset detections are not training targets: targets are built
only from trusted chart events mapped through their tempo clock.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Iterable, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from maichart.event_timing import EventPoint, seconds_at_beat
from maichart.generation.event_quality import parse_event_notes


BLUEPRINT_NOTE_TYPES = ("tap", "hold", "slide", "touch")
FORM_EXTRA_DIM = 3
MAX_FORM_STEPS = 128
GAUSSIAN_SUPPORT_SIGMA = 6.1


@dataclass(frozen=True, slots=True)
class BlueprintConfig:
    audio_dim: int = 120
    descriptor_dim: int = 504
    condition_dim: int = 16
    width: int = 128
    form_heads: int = 4
    form_layers: int = 2
    temporal_layers: int = 4
    dropout: float = 0.1
    max_form_steps: int = MAX_FORM_STEPS
    load_curve_channels: int = 4

    def __post_init__(self) -> None:
        if min(self.audio_dim, self.descriptor_dim, self.condition_dim, self.width) <= 0:
            raise ValueError("Blueprint dimensions must be positive")
        if self.width % self.form_heads:
            raise ValueError("width must be divisible by form_heads")
        if self.form_layers < 1 or self.temporal_layers < 1:
            raise ValueError("Blueprint encoders require at least one layer")
        if not 1 <= self.max_form_steps <= MAX_FORM_STEPS:
            raise ValueError(f"max_form_steps must be in [1, {MAX_FORM_STEPS}]")
        if self.load_curve_channels != 4:
            raise ValueError("load_curve_channels must be 4")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class _TemporalResidualBlock(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv = nn.Conv1d(width, width, kernel_size=3, padding=dilation,
                              dilation=dilation)
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
        residual = x
        hidden = self.conv(x.transpose(1, 2)).transpose(1, 2)
        hidden = self.norm(F.gelu(hidden))
        hidden = residual + self.dropout(hidden)
        if padding_mask is not None:
            hidden = hidden.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return hidden


class SongBlueprint(nn.Module):
    """Predict whole-song chart onsets and four broad note classes.

    Audio time is processed by finite-receptive-field dilated convolutions,
    never full-song self-attention. Self-attention is limited to at most 128
    form descriptors. Form and condition summaries are added at every audio
    frame so the output remains a single song-wide plan.
    """

    def __init__(self, config: BlueprintConfig = BlueprintConfig()) -> None:
        super().__init__()
        self.config = config
        width = config.width
        self.audio_projection = nn.Linear(config.audio_dim, width)
        self.form_projection = nn.Linear(config.descriptor_dim + FORM_EXTRA_DIM, width)
        self.form_position = nn.Embedding(config.max_form_steps, width)
        form_layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=config.form_heads, dim_feedforward=width * 4,
            dropout=config.dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.form_encoder = nn.TransformerEncoder(form_layer, num_layers=config.form_layers)
        self.form_pool_score = nn.Linear(width, 1)
        self.audio_pool_score = nn.Linear(width, 1)
        self.condition_projection = nn.Linear(config.condition_dim, width)
        dilations = (1, 2, 4, 8)
        self.temporal = nn.ModuleList(
            _TemporalResidualBlock(width, dilations[index % len(dilations)], config.dropout)
            for index in range(config.temporal_layers)
        )
        self.final_norm = nn.LayerNorm(width)
        self.onset_head = nn.Linear(width, 1)
        self.type_head = nn.Linear(width, len(BLUEPRINT_NOTE_TYPES))
        self.duration_head = nn.Linear(width, 1)
        self.load_curve_head = nn.Linear(width, config.load_curve_channels)

    def forward(
        self,
        audio: torch.Tensor,
        form: torch.Tensor,
        condition: torch.Tensor,
        audio_padding_mask: torch.Tensor | None = None,
        form_padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if audio.ndim != 3:
            raise ValueError("audio must have shape [B, T, audio_dim]")
        if form.ndim != 3:
            raise ValueError("form must have shape [B, S, descriptor_dim + 3]")
        batch, frames, audio_dim = audio.shape
        form_batch, steps, form_dim = form.shape
        if frames < 1 or steps < 1:
            raise ValueError("audio and form sequences must be nonempty")
        if audio_dim != self.config.audio_dim:
            raise ValueError(f"Expected audio_dim={self.config.audio_dim}, got {audio_dim}")
        if form_dim != self.config.descriptor_dim + FORM_EXTRA_DIM:
            raise ValueError(
                f"Expected form feature width {self.config.descriptor_dim + FORM_EXTRA_DIM}, "
                f"got {form_dim}"
            )
        if form_batch != batch or condition.shape != (batch, self.config.condition_dim):
            raise ValueError("form and condition batch/dimension shapes do not match audio")
        if steps > self.config.max_form_steps:
            raise ValueError(f"form sequence has {steps} steps; max is {self.config.max_form_steps}")
        if audio_padding_mask is not None and audio_padding_mask.shape != (batch, frames):
            raise ValueError("audio_padding_mask must have shape [B, T]")
        if form_padding_mask is not None and form_padding_mask.shape != (batch, steps):
            raise ValueError("form_padding_mask must have shape [B, S]")
        if not torch.isfinite(audio).all():
            raise ValueError("audio features must be finite")
        if not torch.isfinite(form).all() or not torch.isfinite(condition).all():
            raise ValueError("form and condition features must be finite")

        audio_padding_mask = (None if audio_padding_mask is None else
                              audio_padding_mask.to(device=audio.device, dtype=torch.bool))
        form_padding_mask = (None if form_padding_mask is None else
                             form_padding_mask.to(device=form.device, dtype=torch.bool))
        if form_padding_mask is not None and torch.any(form_padding_mask.all(dim=1)):
            raise ValueError("Each song needs at least one unmasked form descriptor")

        positions = torch.arange(steps, device=form.device).unsqueeze(0)
        form_hidden = self.form_projection(form) + self.form_position(positions)
        form_hidden = self.form_encoder(form_hidden, src_key_padding_mask=form_padding_mask)
        scores = self.form_pool_score(form_hidden).squeeze(-1)
        if form_padding_mask is not None:
            scores = scores.masked_fill(form_padding_mask, torch.finfo(scores.dtype).min)
        pooled_form = torch.sum(torch.softmax(scores, dim=1).unsqueeze(-1) * form_hidden, dim=1)

        hidden = self.audio_projection(audio)
        if audio_padding_mask is not None:
            hidden = hidden.masked_fill(audio_padding_mask.unsqueeze(-1), 0.0)
        audio_scores = self.audio_pool_score(hidden).squeeze(-1)
        if audio_padding_mask is not None:
            audio_scores = audio_scores.masked_fill(
                audio_padding_mask, torch.finfo(audio_scores.dtype).min)
        pooled_audio = torch.sum(torch.softmax(audio_scores, dim=1).unsqueeze(-1) * hidden, dim=1)
        global_signal = pooled_form + pooled_audio + self.condition_projection(condition)
        hidden = hidden + global_signal.unsqueeze(1)
        if audio_padding_mask is not None:
            hidden = hidden.masked_fill(audio_padding_mask.unsqueeze(-1), 0.0)
        for block in self.temporal:
            hidden = block(hidden, audio_padding_mask)
        hidden = self.final_norm(hidden)
        return {
            "onset_logits": self.onset_head(hidden).squeeze(-1),
            "type_logits": self.type_head(hidden),
            "sustain_seconds": F.softplus(self.duration_head(hidden).squeeze(-1)),
            "load_curve": self.load_curve_head(hidden),
        }


def build_blueprint_targets(
    points: Iterable[EventPoint],
    tempo_changes: Iterable,
    first: float,
    frame_times: Sequence[float] | torch.Tensor,
    *,
    sigma_seconds: float = 0.02,
) -> dict[str, torch.Tensor]:
    """Map trusted authored notes onto a full-song frame clock.

    Onsets receive a Gaussian target at their exact tempo-integrated time.
    Gaussian values are computed only within 6.1 sigma of each event; omitted
    tails are below 1e-8. This keeps work bounded
    by the local frame support instead of allocating an event-by-song matrix.
    Simultaneous heads share one frame label: Slide, Hold, Touch, then Tap
    priority. Touch Holds use the Touch class. Empty charts yield all-zero
    onset/duration targets and ignored type labels.
    """
    sigma = float(sigma_seconds)
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma_seconds must be finite and positive")
    times = torch.as_tensor(frame_times, dtype=torch.float64).flatten()
    if times.numel() < 1 or not torch.isfinite(times).all():
        raise ValueError("frame_times must be nonempty and finite")
    if torch.any(times[1:] <= times[:-1]):
        raise ValueError("frame_times must be strictly increasing")
    chart_points = tuple(EventPoint(p.beat, p.note_text) for p in points)
    tempo = tuple(tempo_changes)
    parsed = parse_event_notes(chart_points, tempo, first)
    if not parsed.syntax_valid:
        errors = parsed.fatal_errors[:3]
        raise ValueError(f"Blueprint targets require valid chart syntax: {errors}")

    grouped: dict[Fraction, list] = {}
    for note in parsed.notes:
        grouped.setdefault(Fraction(note.beat), []).append(note)
    onset_times = [seconds_at_beat(beat, tempo, first) for beat in sorted(grouped)]
    in_range = [(beat, second) for beat, second in zip(sorted(grouped), onset_times)
                if float(times[0]) <= second <= float(times[-1])]
    in_range_beats = [item[0] for item in in_range]
    onset_times = [item[1] for item in in_range]
    onset_target = torch.zeros(times.shape, dtype=torch.float32)
    type_target = torch.full(times.shape, -100, dtype=torch.long)
    duration_target = torch.zeros(times.shape, dtype=torch.float32)
    duration_mask = torch.zeros(times.shape, dtype=torch.bool)
    class_index = {name: index for index, name in enumerate(BLUEPRINT_NOTE_TYPES)}
    priorities = {"tap": 0, "touch": 1, "touch_hold": 1, "hold": 2, "slide": 3}

    frame_notes: dict[int, list] = {}
    if onset_times:
        support = GAUSSIAN_SUPPORT_SIGMA * sigma
        for onset_sec in onset_times:
            center = times.new_tensor(onset_sec)
            low = int(torch.searchsorted(times, center - support, right=False).item())
            high = int(torch.searchsorted(times, center + support, right=True).item())
            if low < high:
                distances = (times[low:high] - center) / sigma
                values = torch.exp(-0.5 * distances.square()).float()
                onset_target[low:high] = torch.maximum(onset_target[low:high], values)
    for beat, onset_sec in zip(in_range_beats, onset_times):
        right = int(torch.searchsorted(times, times.new_tensor(onset_sec)).item())
        if right <= 0:
            frame = 0
        elif right >= times.numel():
            frame = times.numel() - 1
        else:
            left_delta = onset_sec - float(times[right - 1])
            right_delta = float(times[right]) - onset_sec
            frame = right - 1 if left_delta <= right_delta else right
        frame_notes.setdefault(frame, []).extend(grouped[beat])
    for frame, notes in frame_notes.items():
        selected = max(notes, key=lambda note: priorities.get(note.note_type, -1))
        normalized_type = "touch" if selected.note_type == "touch_hold" else selected.note_type
        if normalized_type not in class_index:
            continue
        type_target[frame] = class_index[normalized_type]
        durations = [float(note.duration_sec or 0.0) for note in notes
                     if note.note_type in {"hold", "touch_hold", "slide"}]
        if durations:
            duration_target[frame] = max(durations)
            duration_mask[frame] = True
    return {
        "onset_target": onset_target,
        "type_target": type_target,
        "duration_target_seconds": duration_target,
        "duration_mask": duration_mask,
    }


def compute_blueprint_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    audio_padding_mask: torch.Tensor | None = None,
    onset_pos_weight: float | str = "auto",
    onset_weight: float = 1.0,
    type_weight: float = 1.0,
    duration_weight: float = 0.25,
    load_curve_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Balanced onset BCE, onset-masked type CE, and duration regression."""
    logits = outputs["onset_logits"]
    device = logits.device
    onset_target = targets["onset_target"].to(device=device, dtype=logits.dtype)
    type_target = targets["type_target"].to(device=device, dtype=torch.long)
    duration_target = targets["duration_target_seconds"].to(device=device, dtype=logits.dtype)
    duration_mask = targets["duration_mask"].to(device=device, dtype=torch.bool)
    if onset_target.shape != logits.shape or type_target.shape != logits.shape:
        raise ValueError("onset and type targets must have shape [B, T]")
    if duration_target.shape != logits.shape or duration_mask.shape != logits.shape:
        raise ValueError("duration targets and mask must have shape [B, T]")
    valid = torch.ones_like(logits, dtype=torch.bool)
    if audio_padding_mask is not None:
        if audio_padding_mask.shape != logits.shape:
            raise ValueError("audio_padding_mask must have shape [B, T]")
        valid &= ~audio_padding_mask.to(device=device, dtype=torch.bool)
    if isinstance(onset_pos_weight, str):
        if onset_pos_weight.lower() != "auto":
            raise ValueError("onset_pos_weight must be numeric or 'auto'")
        positive = (onset_target * valid).sum()
        negative = valid.sum().to(logits.dtype) - positive
        pos_weight = (negative / positive.clamp_min(1.0)).clamp_min(1.0)
    else:
        if not math.isfinite(float(onset_pos_weight)) or float(onset_pos_weight) <= 0:
            raise ValueError("onset_pos_weight must be finite and positive")
        pos_weight = logits.new_tensor(float(onset_pos_weight))
    loss_onset = F.binary_cross_entropy_with_logits(
        logits, onset_target, reduction="none", pos_weight=pos_weight,
    )
    loss_onset = (loss_onset * valid).sum() / valid.sum().clamp_min(1)

    type_valid = valid & (type_target >= 0) & (onset_target >= 0.5)
    type_safe = type_target.masked_fill(~type_valid, -100)
    loss_type_all = F.cross_entropy(outputs["type_logits"].transpose(1, 2), type_safe,
                                    ignore_index=-100, reduction="none")
    loss_type = (loss_type_all * type_valid).sum() / type_valid.sum().clamp_min(1)

    duration_valid = valid & duration_mask
    predicted_log = torch.log1p(outputs["sustain_seconds"].clamp_min(0.0))
    target_log = torch.log1p(duration_target.clamp_min(0.0))
    loss_duration_all = F.smooth_l1_loss(predicted_log, target_log, reduction="none")
    loss_duration = (loss_duration_all * duration_valid).sum() / duration_valid.sum().clamp_min(1)
    total = onset_weight * loss_onset + type_weight * loss_type + duration_weight * loss_duration
    loss_curve = logits.new_zeros(())
    if "load_curve" in outputs and "load_curve_target" in targets:
        target_curve = targets["load_curve_target"].to(device=device, dtype=outputs["load_curve"].dtype)
        curve_mask = targets.get("load_curve_mask")
        curve_error = F.smooth_l1_loss(outputs["load_curve"], target_curve, reduction="none")
        if curve_mask is not None:
            curve_error = curve_error * curve_mask.to(device=device, dtype=curve_error.dtype)
            denom = curve_mask.to(device=device, dtype=curve_error.dtype).sum().clamp_min(1.0)
        else:
            denom = torch.tensor(curve_error.numel(), device=device, dtype=curve_error.dtype).clamp_min(1.0)
        loss_curve = curve_error.sum() / denom
        total = total + load_curve_weight * loss_curve
    return {"loss": total, "loss_onset": loss_onset,
            "loss_type": loss_type, "loss_duration": loss_duration,
            "loss_load_curve": loss_curve}
