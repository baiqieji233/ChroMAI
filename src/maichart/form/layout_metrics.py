"""Diagnostic token metrics for song-form layout training."""
from __future__ import annotations

import torch

from maichart.form.codec import RationalCodec


_CODEC = RationalCodec()


def layout_diagnostics(logits, targets, decoder_input, tasks, token_ce, codec=_CODEC):
    """Return task-split token and event diagnostics under teacher forcing.

    Configuration decisions are note characters plus event termination. Position
    headers are retained as a token metric for compatibility.  The rational
    Fraction metric compares the complete target digit string for each event;
    it is a teacher-forced sequence diagnostic, not a musical beat-position
    accuracy metric.  Event-boundary metrics measure the AT/EOS decision after
    BOS or ``<end>``.  Direct generation does not necessarily supply position
    headers, so these metrics must not be read as generation-time guarantees.
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

    # Each target <at> starts one event.  Cumulative AT/NOTE counts identify
    # its position-token group without synchronizing to Python per token.
    at = (target == codec.at_id) & valid
    note = (target == codec.note_id) & valid
    event_ids = at.to(torch.long).cumsum(dim=1) - 1
    p_token = torch.zeros_like(target, dtype=torch.bool)
    for token_id, token in enumerate(codec.tokens):
        if token.startswith("P:"):
            p_token |= target == token_id
    p_token &= valid
    p_mismatch = p_token & (predictions != target)
    event_slots = torch.zeros_like(target, dtype=torch.float32)
    event_slots.scatter_reduce_(1, event_ids.clamp_min(0),
                                p_mismatch.to(torch.float32),
                                reduce="amax", include_self=True)
    fraction_exact_at = at & (event_slots.gather(1, event_ids.clamp_min(0)) == 0)
    # AT/EOS is only a decision at the start of a new event (BOS) or after a
    # committed cell (<end>).  Keep the masks explicit so accidental EOS in a
    # note body is not counted as a valid termination decision.
    boundary = ((decoder_input == codec.bos_id) | (decoder_input == codec.end_id)) & valid
    continue_decision = boundary & (target == codec.at_id)
    terminate_decision = boundary & (target == codec.eos_id)
    event_decision = continue_decision | terminate_decision

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

        for name, selected in (("position_fraction_exact", at & task_mask),
                               ("event_decision", event_decision & task_mask),
                               ("event_continue", continue_decision & task_mask),
                               ("event_terminate", terminate_decision & task_mask)):
            count = selected.sum()
            correct = ((fraction_exact_at if name == "position_fraction_exact"
                        else (predictions == target)) & selected).sum()
            count_float = count.to(dtype=torch.float32).clamp_min(1.0)
            values[f"{prefix}_{name}_accuracy"] = correct.to(dtype=torch.float32) / count_float
            counts[f"{prefix}_{name}_accuracy"] = count
            if name != "position_fraction_exact":
                values[f"{prefix}_{name}_loss"] = (
                    (token_ce.detach() * selected).sum()
                    / count.to(dtype=token_ce.dtype).clamp_min(1.0)
                )
                counts[f"{prefix}_{name}_loss"] = count

    return values, counts
