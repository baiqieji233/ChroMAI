"""Diagnostic token metrics for song-form layout training."""
from __future__ import annotations

import torch

from maichart.form.codec import RationalCodec


_CODEC = RationalCodec()
_TOKEN_TABLES = {}


def _token_tables(codec, device):
    """Vocabulary flags built once per device.

    Creating index tensors from Python lists for every batch is a blocking
    host-to-device copy, which waits for the whole queued forward pass.
    """
    key = (codec.tokens, device)
    tables = _TOKEN_TABLES.get(key)
    if tables is None:
        flags = (
            [token.startswith("N:") for token in codec.tokens],
            [token in ("<at>", "<note>") or token.startswith("P:") for token in codec.tokens],
            [token.startswith("P:") for token in codec.tokens],
        )
        tables = _TOKEN_TABLES[key] = tuple(
            torch.tensor(values, dtype=torch.bool).to(device) for values in flags)
    return tables


@torch.no_grad()
def layout_diagnostics(logits, targets, decoder_input, tasks, token_ce, codec=_CODEC, *, predictions=None):
    """Return task-split token and event diagnostics under teacher forcing.

    Configuration decisions are note characters plus event termination. Position
    headers are retained as a token metric for compatibility.  The rational
    Fraction metric compares the complete target digit string for each event;
    it is a teacher-forced sequence diagnostic, not a musical beat-position
    accuracy metric.  Event-boundary metrics measure the AT/EOS decision after
    BOS or ``<end>``.  Direct generation does not necessarily supply position
    headers, so these metrics must not be read as generation-time guarantees.
    """
    predictions = logits.detach().argmax(dim=-1) if predictions is None else predictions
    target = targets
    if (predictions.shape != target.shape or decoder_input.shape != target.shape
            or token_ce.shape != target.shape):
        raise ValueError("logits, targets, decoder_input, and token_ce must share [B, L]")
    if tasks.shape != (target.shape[0],):
        raise ValueError("tasks must have shape [B]")

    device = target.device
    note_table, position_table, fraction_table = _token_tables(codec, device)
    valid = target != codec.pad_id
    note_token = note_table[target] & valid
    configuration = (note_token | (target == codec.end_id)) & valid
    position_header = position_table[target] & valid
    first_head = note_token & (decoder_input == codec.note_id)

    # Each target <at> starts one event.  Cumulative AT/NOTE counts identify
    # its position-token group without synchronizing to Python per token.
    at = (target == codec.at_id) & valid
    note = (target == codec.note_id) & valid
    event_ids = at.to(torch.long).cumsum(dim=1) - 1
    p_token = fraction_table[target] & valid
    p_mismatch = p_token & (predictions != target)
    event_slots = torch.zeros_like(target, dtype=torch.float32)
    event_slots.scatter_reduce_(1, event_ids.clamp_min(0),
                                p_mismatch.to(torch.float32),
                                reduce="amax", include_self=True)
    fraction_exact_at = at & (event_slots.gather(1, event_ids.clamp_min(0)) == 0)
    # A high character accuracy can hide a wrong button/path in almost every
    # cell. Count a whole configuration only when every N character and END
    # is correct. This remains teacher-forced, not free-generation quality.
    configuration_slots = torch.zeros_like(target, dtype=torch.float32)
    configuration_slots.scatter_reduce_(1, event_ids.clamp_min(0),
        (configuration & (predictions != target)).to(torch.float32),
        reduce='amax', include_self=True)
    configuration_exact_at = at & (configuration_slots.gather(1, event_ids.clamp_min(0)) == 0)
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
                               ("configuration_exact", at & task_mask),
                               ("event_decision", event_decision & task_mask),
                               ("event_continue", continue_decision & task_mask),
                               ("event_terminate", terminate_decision & task_mask)):
            count = selected.sum()
            correctness = (fraction_exact_at if name == 'position_fraction_exact'
                           else configuration_exact_at if name == 'configuration_exact'
                           else predictions == target)
            correct = (correctness & selected).sum()
            count_float = count.to(dtype=torch.float32).clamp_min(1.0)
            values[f"{prefix}_{name}_accuracy"] = correct.to(dtype=torch.float32) / count_float
            counts[f"{prefix}_{name}_accuracy"] = count
            if name not in ('position_fraction_exact', 'configuration_exact'):
                values[f"{prefix}_{name}_loss"] = (
                    (token_ce.detach() * selected).sum()
                    / count.to(dtype=token_ce.dtype).clamp_min(1.0)
                )
                counts[f"{prefix}_{name}_loss"] = count

    return values, counts
