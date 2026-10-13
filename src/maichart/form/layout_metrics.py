"""Diagnostic token metrics for song-form layout training."""
from __future__ import annotations

import torch

from maichart.form.codec import RationalCodec


_CODEC = RationalCodec()


def layout_diagnostics(logits, targets, decoder_input, tasks, token_ce, codec=_CODEC):
    """Return task-split sampled-configuration and forced-header metrics.

    Configuration decisions are note characters plus event termination. Position
    headers are reported separately because generation supplies them directly.
    EOS is structural and is not included in either group.
    """
    predictions = logits.detach().argmax(dim=-1)
    target = targets
    if (predictions.shape != target.shape or decoder_input.shape != target.shape
            or token_ce.shape != target.shape):
        raise ValueError("logits, targets, decoder_input, and token_ce must share [B, L]")
    if tasks.shape != (target.shape[0],):
        raise ValueError("tasks must have shape [B]")

    note_ids = [index for index, token in enumerate(codec.tokens) if token.startswith("N:")]
    position_ids = [index for index, token in enumerate(codec.tokens)
                    if token == "<at>" or token == "<note>" or token.startswith("P:")]
    device = target.device
    note_ids = torch.tensor(note_ids, dtype=target.dtype, device=device)
    position_ids = torch.tensor(position_ids, dtype=target.dtype, device=device)
    valid = target != codec.pad_id
    note_token = (target.unsqueeze(-1) == note_ids).any(dim=-1) & valid
    configuration = (note_token | (target == codec.end_id)) & valid
    position_header = (target.unsqueeze(-1) == position_ids).any(dim=-1) & valid
    first_head = note_token & (decoder_input == codec.note_id)

    values = {}
    counts = {}
    for task_id, prefix in ((0, "draft"), (1, "revision")):
        task_mask = (tasks.to(device=device).unsqueeze(1) == task_id)
        for name, selected in (("note_configuration", configuration & task_mask),
                               ("position_header", position_header & task_mask),
                               ("first_head", first_head & task_mask)):
            count = selected.sum()
            correct = ((predictions == target) & selected).sum()
            count_float = count.to(dtype=torch.float32).clamp_min(1.0)
            key = f"{prefix}_{name}_accuracy"
            values[key] = correct.to(dtype=torch.float32) / count_float
            counts[key] = count
            if name == "note_configuration":
                values[f"{prefix}_{name}_loss"] = (
                    (token_ce.detach() * selected).sum()
                    / count.to(dtype=token_ce.dtype).clamp_min(1.0)
                )
                counts[f"{prefix}_{name}_loss"] = count

    return values, counts
