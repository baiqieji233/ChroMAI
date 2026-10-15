"""Causal numeric geometry helpers for rational event decoding.

The optional ``exact_fraction_v1`` checkpoint contract uses these helpers to
turn exact :class:`fractions.Fraction` positions into floating point features
and auxiliary targets. They never quantize or choose a generated position.
"""
from __future__ import annotations

from fractions import Fraction
import math
from typing import Iterable, Sequence

import torch


GEOMETRY_DIM = 8


def _coordinate_row(beat, origin):
    delta = float(Fraction(beat) - Fraction(origin))
    absolute = abs(delta)
    frac = delta - math.floor(delta)
    scale = math.log1p(64.0)
    return (
        delta / 64.0,
        math.copysign(math.log1p(absolute) / scale, delta) if delta else 0.0,
        absolute / 64.0,
        frac,
        math.sin(2.0 * math.pi * frac),
        math.cos(2.0 * math.pi * frac),
        # A fixed four-beat harmonic is a smooth positional basis only; it is
        # not a meter/4-4 assertion and must not drive snapping.
        math.sin(0.5 * math.pi * delta),
        math.cos(0.5 * math.pi * delta),
    )


def coordinate_features(beats: Iterable[Fraction | int | float], *, origin=Fraction(0)) -> torch.Tensor:
    """Return deterministic 8-D features for exact beat coordinates.

    The conversion to float is confined to this feature boundary.  The first
    three channels preserve signed/absolute local scale, while the remaining
    channels encode beat phase at quarter-note and whole-beat periods.
    """
    origin = Fraction(origin)
    rows = [_coordinate_row(value, origin) for value in beats]
    return torch.tensor(rows, dtype=torch.float32).reshape(-1, GEOMETRY_DIM)


class CausalGeometry:
    """One-pass exact-Fraction geometry state for cached decoder inference.

    Call :meth:`consume` once for each token supplied to the decoder.  It
    returns the causal geometry attached to that same token.  A caller that
    rewinds a decoder cache can recreate this tiny state and replay the kept
    token prefix; no target stream or future spelling is required.
    """
    def __init__(self, codec, *, origin=Fraction(0), coordinate_origin=None):
        self.codec = codec
        self.origin = Fraction(origin)
        self.coordinate_origin = (self.origin if coordinate_origin is None
                                  else Fraction(coordinate_origin))
        self.current = self.origin
        self.state = 'start'
        self.digits = ''
        self.pending = None
        # The row depends only on the committed coordinate, which changes once
        # per event; recomputing its Fraction arithmetic for every token of a
        # long prefix dominated data preparation.
        self._row_beat = None
        self._row = None

    def consume(self, token_id: int) -> tuple[float, ...]:
        token = self.codec.tokens[int(token_id)]
        if token == '<pad>':
            return (0.0,) * GEOMETRY_DIM
        if token == '<at>' and self.state == 'start':
            self.state, self.digits = 'position', ''
        elif token.startswith('P:') and self.state == 'position':
            self.digits += token[2:]
        elif token == '<note>' and self.state == 'position':
            try:
                self.pending = self.origin + Fraction(self.digits)
                self.current = self.pending
                self.state = 'note'
            except (ValueError, ZeroDivisionError):
                self.state, self.digits, self.pending = 'start', '', None
        elif token.startswith('N:') and self.state == 'note':
            pass
        elif token == '<end>' and self.state == 'note' and self.pending is not None:
            self.current = self.pending
            self.state, self.digits, self.pending = 'start', '', None
        if self._row is None or self.current != self._row_beat:
            self._row_beat = self.current
            self._row = _coordinate_row(self.current, self.coordinate_origin)
        return self._row


def causal_prefix_geometry(ids: Sequence[int] | torch.Tensor, codec, *, origin=Fraction(0), coordinate_origin=None) -> torch.Tensor:
    """Build one geometry row per consumed token without reading future tokens.

    A position becomes visible only after all of its ``P:`` digits and the
    ``<note>`` marker have been consumed.  ``<end>`` commits that position;
    malformed or truncated prefixes retain the last committed coordinate.
    """
    values = ids.detach().cpu().tolist() if isinstance(ids, torch.Tensor) else list(ids)
    state = CausalGeometry(codec, origin=origin, coordinate_origin=coordinate_origin)
    rows = [state.consume(token_id) for token_id in values]
    return torch.tensor(rows, dtype=torch.float32).reshape(-1, GEOMETRY_DIM)


def event_timing_targets(decoder_input: torch.Tensor | Sequence[int], target: torch.Tensor | Sequence[int], codec, *, origin=Fraction(0)) -> dict[str, torch.Tensor]:
    """Derive next-event geometry from the target token stream only.

    The decoder prefix supplies the last committed position; the target
    supplies the complete next ``<at> P:* <note>`` spelling.  No target token
    is fed back into the prefix state, so this is safe for teacher forcing.
    """
    inputs = decoder_input.detach().cpu().tolist() if isinstance(decoder_input, torch.Tensor) else list(decoder_input)
    targets = target.detach().cpu().tolist() if isinstance(target, torch.Tensor) else list(target)
    n = min(len(inputs), len(targets))
    origin = Fraction(origin)
    current = origin
    state, digits = "start", ""
    rows, mask = [], []
    for i in range(n):
        token = codec.tokens[int(inputs[i])]
        if token == "<at>" and state == "start": state, digits = "position", ""
        elif token.startswith("P:") and state == "position": digits += token[2:]
        elif token == "<note>" and state == "position":
            try: current = origin + Fraction(digits)
            except (ValueError, ZeroDivisionError): pass
            state = "note"
        elif token == "<end>" and state == "note": state, digits = "start", ""
        boundary = inputs[i] in (codec.bos_id, codec.end_id)
        target_token = codec.tokens[int(targets[i])]
        valid = boundary and target_token == "<at>"
        beat = None
        if valid:
            j = i + 1; parsed = ""
            if j < n and codec.tokens[int(targets[j])] == "<at>": j += 1
            while j < n and codec.tokens[int(targets[j])].startswith("P:"):
                parsed += codec.tokens[int(targets[j])][2:]; j += 1
            try: beat = origin + Fraction(parsed)
            except (ValueError, ZeroDivisionError): valid = False
        if valid and beat is not None:
            gap = max(beat - current, Fraction(0)); local = float(beat - origin)
            phase = 2.0 * math.pi * (local - math.floor(local))
            rows.append((math.log1p(float(gap)), math.sin(phase), math.cos(phase))); mask.append(True)
        else:
            rows.append((0.0, 0.0, 0.0)); mask.append(False)
    labels = torch.tensor(rows, dtype=torch.float32).reshape(-1, 3)
    return {"mask": torch.tensor(mask, dtype=torch.bool), "labels": labels,
            "gap_log1p": labels[:, 0], "phase_sin": labels[:, 1], "phase_cos": labels[:, 2]}


def timing_targets(points: Sequence, decoder_input: torch.Tensor | Sequence[int], target: torch.Tensor | Sequence[int], codec, *, origin=Fraction(0)) -> dict[str, torch.Tensor]:
    """Create masked next-event gap/phase labels at BOS and committed END.

    ``points`` must be the exact ordered events for the span.  A row is valid
    only when the target token starts the next event (``<at>``); EOS and all
    interior spelling tokens receive a false mask and zero labels.
    """
    inputs = decoder_input.detach().cpu().tolist() if isinstance(decoder_input, torch.Tensor) else list(decoder_input)
    targets = target.detach().cpu().tolist() if isinstance(target, torch.Tensor) else list(target)
    n = min(len(inputs), len(targets))
    origin = Fraction(origin)
    events = list(points)
    mask, gaps, sins, coss = [], [], [], []
    event_index = 0
    for i in range(n):
        boundary = inputs[i] in (codec.bos_id, codec.end_id)
        is_at = targets[i] == codec.at_id
        valid = boundary and is_at and event_index < len(events)
        if valid:
            previous = origin if event_index == 0 else Fraction(events[event_index - 1].beat)
            beat = Fraction(events[event_index].beat)
            gap = max(beat - previous, Fraction(0))
            local = float(beat - origin)
            phase = 2.0 * math.pi * (local - math.floor(local))
            gaps.append(math.log1p(float(gap)))
            sins.append(math.sin(phase))
            coss.append(math.cos(phase))
            event_index += 1
        else:
            gaps.append(0.0); sins.append(0.0); coss.append(0.0)
        mask.append(valid)
    return {
        "mask": torch.tensor(mask, dtype=torch.bool),
        "gap_log1p": torch.tensor(gaps, dtype=torch.float32),
        "phase_sin": torch.tensor(sins, dtype=torch.float32),
        "phase_cos": torch.tensor(coss, dtype=torch.float32),
    }


def position_event_loss(logits: torch.Tensor, target: torch.Tensor, decoder_input: torch.Tensor, codec, *, reduction="mean") -> torch.Tensor:
    """Cross entropy averaged per Fraction spelling, then per event.

    This prevents long note strings or multi-digit positions from determining
    the objective solely by character count.  It is an optional helper and
    does not alter the existing token head or checkpoint contract.
    """
    ce = torch.nn.functional.cross_entropy(logits.float().transpose(1, 2), target, reduction="none")
    valid = target != codec.pad_id
    is_position = torch.zeros_like(valid)
    event_ids = torch.full_like(target, -1)
    state = torch.zeros((target.size(0),), dtype=torch.long, device=target.device)
    event = torch.zeros((target.size(0),), dtype=torch.long, device=target.device)
    for i in range(target.size(1)):
        inp = decoder_input[:, i]
        boundary = ((inp == codec.bos_id) | (inp == codec.end_id)) & (target == codec.at_id)
        event = event + boundary.to(event.dtype)
        starts = inp == codec.at_id
        state = torch.where(starts, torch.ones_like(state), state)
        state = torch.where((inp == codec.note_id) | (inp == codec.end_id), torch.zeros_like(state), state)
        is_position[:, i] = (state == 1) & valid
        event_ids[:, i] = torch.where(is_position[:, i], event, torch.full_like(event, -1))
    # Sum each event's position spelling, then average events.  A long
    # denominator such as 123/128 therefore contributes the same example
    # weight as a one-character denominator.
    event_losses = []
    for row in range(target.size(0)):
        ids = event_ids[row]
        unique = torch.unique(ids[ids >= 1])
        if unique.numel():
            values = []
            for event_id in unique:
                selected = ids == event_id
                values.append((ce[row] * selected).sum() / selected.sum().clamp_min(1))
            event_losses.append(torch.stack(values).mean())
        else:
            event_losses.append(ce[row].new_zeros(()))
    per_row = torch.stack(event_losses)
    return per_row.mean() if reduction == "mean" else per_row


def numeric_timing_loss(prediction: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, *, reduction="mean") -> torch.Tensor:
    """Masked Smooth-L1 loss for ``[log1p(gap), sin(phase), cos(phase)]``."""
    values = torch.nn.functional.smooth_l1_loss(prediction.float(), labels.float(), reduction="none").mean(-1)
    per_row = (values * mask.to(values.dtype)).sum(-1) / mask.sum(-1).clamp_min(1)
    return per_row.mean() if reduction == "mean" else per_row


__all__ = ["GEOMETRY_DIM", "coordinate_features", "CausalGeometry", "causal_prefix_geometry", "event_timing_targets", "timing_targets", "position_event_loss", "numeric_timing_loss"]
