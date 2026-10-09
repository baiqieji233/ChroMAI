"""Thresholded metrics for the V2.5 frame predictor."""

from __future__ import annotations

import torch


def compute_v25_metrics(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    loss_mask: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Return masked note, button, type, and density metrics."""

    mask = loss_mask.to(device=outputs["note_presence_logits"].device, dtype=torch.bool)
    note_prob = torch.sigmoid(outputs["note_presence_logits"])[..., 0][mask]
    note_target = targets["note_presence"].to(mask.device)[..., 0][mask]
    button_prob = torch.sigmoid(outputs["button_logits"])[mask]
    button_target = targets["buttons"].to(mask.device)[mask]
    type_pred = outputs["type_logits"].argmax(dim=-1)[mask]
    type_target = targets["note_type"].to(mask.device)[mask]
    density_pred = outputs["density_pred"][mask]
    density_target = targets["density"].to(mask.device)[mask]

    note_precision, note_recall, note_f1 = _binary_metrics(note_prob >= threshold, note_target >= 0.5)
    button_precision, button_recall, button_f1 = _binary_metrics(
        button_prob >= threshold,
        button_target >= 0.5,
    )
    button_macro_f1 = _macro_f1(button_prob >= threshold, button_target >= 0.5)
    best_threshold, best_f1 = _best_threshold(button_prob, button_target)
    note_best_threshold, note_best_f1 = _best_threshold(note_prob, note_target)
    return {
        "note_presence_precision": note_precision,
        "note_presence_recall": note_recall,
        "note_presence_f1": note_f1,
        "button_micro_precision": button_precision,
        "button_micro_recall": button_recall,
        "button_micro_f1": button_f1,
        "button_micro_f1@0.05": _binary_metrics(button_prob >= 0.05, button_target >= 0.5)[2],
        "button_micro_f1@0.10": _binary_metrics(button_prob >= 0.10, button_target >= 0.5)[2],
        "button_micro_f1@0.20": _binary_metrics(button_prob >= 0.20, button_target >= 0.5)[2],
        "button_macro_f1": button_macro_f1,
        "button_best_f1": best_f1,
        "button_best_threshold": best_threshold,
        "note_type_accuracy": _accuracy(type_pred, type_target),
        "note_best_f1": note_best_f1,
        "note_best_threshold": note_best_threshold,
        "density_mae": float(torch.abs(density_pred - density_target).mean().item()) if density_pred.numel() else 0.0,
    }


def _binary_metrics(predicted: torch.Tensor, target: torch.Tensor) -> tuple[float, float, float]:
    predicted = predicted.bool().reshape(-1)
    target = target.bool().reshape(-1)
    true_positive = float((predicted & target).sum().item())
    false_positive = float((predicted & ~target).sum().item())
    false_negative = float((~predicted & target).sum().item())
    precision = true_positive / max(1.0, true_positive + false_positive)
    recall = true_positive / max(1.0, true_positive + false_negative)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return float(precision), float(recall), float(f1)


def _macro_f1(predicted: torch.Tensor, target: torch.Tensor) -> float:
    if predicted.ndim == 1:
        return _binary_metrics(predicted, target)[2]
    values = [_binary_metrics(predicted[:, column], target[:, column])[2] for column in range(predicted.size(1))]
    return float(sum(values) / max(1, len(values)))


def _best_threshold(probability: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90):
        f1 = _binary_metrics(probability >= threshold, target >= 0.5)[2]
        if f1 > best_f1:
            best_threshold, best_f1 = threshold, f1
    return float(best_threshold), float(best_f1)


def _accuracy(predicted: torch.Tensor, target: torch.Tensor) -> float:
    return float((predicted == target).float().mean().item()) if predicted.numel() else 0.0
