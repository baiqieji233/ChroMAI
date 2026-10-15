"""Optional objectives for rational song-form layout training.

The main layout cross-entropy remains token-normalized.  This module exposes a
small per-example auxiliary objective for the only decision that is otherwise
strongly diluted by long Fraction and note strings: after ``<bos>`` or a
committed ``<end>``, continue with ``<at>`` or terminate with ``<eos>``.
It is intentionally independent of the trainer so old checkpoints and the
existing WebUI contract are unaffected until a caller opts in.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from maichart.form.codec import RationalCodec


_CODEC = RationalCodec()


def event_decision_loss(
    logits: torch.Tensor,
    batch: dict[str, torch.Tensor],
    *,
    codec: RationalCodec = _CODEC,
) -> torch.Tensor:
    """Return one AT/EOS cross-entropy mean for each batch row.

    A decision is supervised only when the teacher-forced decoder input is
    ``<bos>`` or ``<end>`` and its target is ``<at>`` or ``<eos>``. Padding and
    malformed/non-boundary positions are ignored. Rows without a decision
    return an exact finite zero, allowing callers to add the vector with a
    configurable weight without special casing empty batches.
    """
    decoder_input = batch.get("decoder_input")
    target = batch.get("target")
    if decoder_input is None or target is None:
        raise ValueError("batch must include decoder_input and target")
    if logits.ndim != 3 or decoder_input.ndim != 2 or target.ndim != 2:
        raise ValueError("logits must be [B,L,V] and decoder_input/target [B,L]")
    if decoder_input.shape != target.shape or logits.shape[:2] != target.shape:
        raise ValueError("logits, decoder_input, and target must share [B,L]")
    if logits.size(-1) <= max(codec.at_id, codec.eos_id):
        raise ValueError("logits vocabulary is missing AT/EOS tokens")

    # Keep the objective in fp32 even when the model uses BF16/FP16. This is a
    # tiny binary CE and otherwise can lose the AT/EOS margin near zero.
    # Integer indexing avoids building a host index tensor for every batch.
    values = torch.stack((logits[..., codec.at_id], logits[..., codec.eos_id]), dim=-1).float()
    boundary = (decoder_input == codec.bos_id) | (decoder_input == codec.end_id)
    valid = boundary & (decoder_input != codec.pad_id) & (target != codec.pad_id)
    decision_target = torch.where(
        target == codec.at_id,
        torch.zeros_like(target, dtype=torch.long),
        torch.ones_like(target, dtype=torch.long),
    )
    valid &= (target == codec.at_id) | (target == codec.eos_id)
    per_token = F.cross_entropy(
        values.transpose(1, 2), decision_target, reduction="none"
    )
    per_token = per_token * valid.to(dtype=per_token.dtype)
    counts = valid.sum(dim=1).to(dtype=per_token.dtype)
    return per_token.sum(dim=1) / counts.clamp_min(1.0)


__all__ = ["event_decision_loss"]
