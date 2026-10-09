"""Losses for the V2.5 multi-task frame predictor."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_v25_losses(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    loss_mask: torch.Tensor,
    *,
    note_weight: float = 1.0,
    buttons_weight: float = 1.0,
    type_weight: float = 1.0,
    density_weight: float = 1.0,
    note_pos_weight: float = 1.0,
    button_pos_weight: float | str = 1.0,
) -> dict[str, torch.Tensor]:
    mask = loss_mask.to(device=outputs["note_presence_logits"].device, dtype=torch.float32)
    if mask.ndim != 2:
        raise ValueError(f"loss_mask must have shape [B, T], got {tuple(mask.shape)}.")

    note_target = targets["note_presence"].to(outputs["note_presence_logits"].device)
    buttons_target = targets["buttons"].to(outputs["button_logits"].device)
    type_target = targets["note_type"].to(outputs["type_logits"].device)
    density_target = targets["density"].to(outputs["density_pred"].device)

    note_weight_tensor = torch.tensor(
        float(note_pos_weight),
        dtype=outputs["note_presence_logits"].dtype,
        device=outputs["note_presence_logits"].device,
    )
    button_weight_tensor = _button_pos_weight(buttons_target, button_pos_weight)
    loss_note = _masked_mean(
        F.binary_cross_entropy_with_logits(
            outputs["note_presence_logits"],
            note_target,
            reduction="none",
            pos_weight=note_weight_tensor,
        ),
        mask.unsqueeze(-1),
    )
    loss_buttons = _masked_mean(
        F.binary_cross_entropy_with_logits(
            outputs["button_logits"],
            buttons_target,
            reduction="none",
            pos_weight=button_weight_tensor,
        ),
        mask.unsqueeze(-1),
    )
    loss_type = _masked_mean(
        F.cross_entropy(
            outputs["type_logits"].transpose(1, 2),
            type_target,
            reduction="none",
        ),
        mask,
    )
    loss_density = _masked_mean(
        F.mse_loss(outputs["density_pred"], density_target, reduction="none"),
        mask.unsqueeze(-1),
    )
    total = (
        note_weight * loss_note
        + buttons_weight * loss_buttons
        + type_weight * loss_type
        + density_weight * loss_density
    )
    return {
        "loss": total,
        "loss_note": loss_note,
        "loss_buttons": loss_buttons,
        "loss_type": loss_type,
        "loss_density": loss_density,
    }


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = values * mask
    denom = mask.expand_as(values).sum().clamp_min(1.0)
    return masked.sum() / denom


def _button_pos_weight(target: torch.Tensor, value: float | str) -> torch.Tensor:
    if isinstance(value, str):
        if value.lower() != "auto":
            raise ValueError("button_pos_weight must be a number or 'auto'.")
        positives = target.sum(dim=(0, 1))
        negatives = target.numel() / target.size(-1) - positives
        return (negatives / positives.clamp_min(1.0)).clamp_min(1.0).to(target)
    return torch.full(
        (target.size(-1),),
        float(value),
        dtype=target.dtype,
        device=target.device,
    )
