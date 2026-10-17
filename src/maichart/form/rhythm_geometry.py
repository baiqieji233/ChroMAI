"""Causal numeric geometry helpers for rational event decoding.

The optional geometry contracts turn exact :class:`fractions.Fraction`
positions into floating point features and auxiliary targets. They never
quantize or choose a generated position.

Two contracts exist.

``exact_fraction_v1``
    Eight channels. Rhythm identity is the offset from the *span start*: a
    sub-beat phase plus a fixed four-beat harmonic anchored at ``delta = 0``.
    A span start comes from a seconds-based resource cap, so the same musical
    note receives a different coordinate depending on which span it lands in,
    and the fixed harmonic asserts 4/4 at an arbitrary origin.

``exact_fraction_v2``
    Thirteen channels. Rhythm identity is *meter phase*: sub-beat phase from
    the absolute beat plus bar and group phase measured from the active meter
    change, whose declared beat is a bar line by construction. The same
    musical position therefore receives the same coordinate anywhere in the
    song, and a 24th on beat 1 is distinguished from a 24th on beat 2 by bar
    phase rather than by distance from a span edge.
"""
from __future__ import annotations

from fractions import Fraction
import math
from typing import Iterable, Sequence

import torch

from maichart.form.timing import MeterPhaseIndex


GEOMETRY_DIM = 8
GEOMETRY_DIM_V2 = 13
GEOMETRY_PROFILES = ("none", "exact_fraction_v1", "exact_fraction_v2")


def geometry_dim(profile: str) -> int:
    """Channel count for a geometry profile; zero when geometry is disabled."""
    if profile == "none":
        return 0
    if profile == "exact_fraction_v2":
        return GEOMETRY_DIM_V2
    if profile == "exact_fraction_v1":
        return GEOMETRY_DIM
    raise ValueError("Unknown rhythm geometry profile")


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


def _coordinate_row_v2(beat, origin, phases):
    """Meter-anchored coordinate row.

    Only the first two channels describe distance from the span start, and
    they exist so the decoder can gauge local scale; they are not the rhythm
    identity. Channels 2-10 are bar-relative and therefore identical for the
    same musical position in every span and every repetition of a bar.
    """
    value = Fraction(beat)
    delta = float(value - Fraction(origin))
    absolute = abs(delta)
    scale = math.log1p(64.0)
    # Sub-beat phase from the absolute beat. Deriving it from ``delta`` would
    # let a span that begins mid-beat shift the whole grid.
    sub = float(value - math.floor(value))
    bar_phase, group_phase, _bar_length, meter_known, grouping_known = phases(value)
    return (
        delta / 64.0,
        math.copysign(math.log1p(absolute) / scale, delta) if delta else 0.0,
        sub,
        math.sin(2.0 * math.pi * sub),
        math.cos(2.0 * math.pi * sub),
        bar_phase,
        math.sin(2.0 * math.pi * bar_phase),
        math.cos(2.0 * math.pi * bar_phase),
        group_phase,
        math.sin(2.0 * math.pi * group_phase),
        math.cos(2.0 * math.pi * group_phase),
        meter_known,
        grouping_known,
    )


def coordinate_features(beats: Iterable[Fraction | int | float], *, origin=Fraction(0),
                        profile: str = "exact_fraction_v1", meters=None) -> torch.Tensor:
    """Return deterministic features for exact beat coordinates.

    The conversion to float is confined to this feature boundary. Under
    ``exact_fraction_v1`` the first three channels preserve signed/absolute
    local scale and the remainder encode beat phase at quarter-note and
    whole-beat periods. Under ``exact_fraction_v2`` the remainder encode
    meter phase anchored at real bar lines.
    """
    origin = Fraction(origin)
    if profile == "exact_fraction_v2":
        index = meters if isinstance(meters, MeterPhaseIndex) else MeterPhaseIndex(meters)
        rows = [_coordinate_row_v2(value, origin, index.phase) for value in beats]
        return torch.tensor(rows, dtype=torch.float32).reshape(-1, GEOMETRY_DIM_V2)
    rows = [_coordinate_row(value, origin) for value in beats]
    return torch.tensor(rows, dtype=torch.float32).reshape(-1, GEOMETRY_DIM)


class CausalGeometry:
    """One-pass exact-Fraction geometry state for cached decoder inference.

    Call :meth:`consume` once for each token supplied to the decoder.  It
    returns the causal geometry attached to that same token.  A caller that
    rewinds a decoder cache can recreate this tiny state and replay the kept
    token prefix; no target stream or future spelling is required.
    """
    def __init__(self, codec, *, origin=Fraction(0), coordinate_origin=None,
                 profile: str = "exact_fraction_v1", meters=None):
        self.codec = codec
        self.origin = Fraction(origin)
        self.coordinate_origin = (self.origin if coordinate_origin is None
                                  else Fraction(coordinate_origin))
        self.profile = profile
        self.dim = geometry_dim(profile)
        self.phases = (meters if isinstance(meters, MeterPhaseIndex)
                       else MeterPhaseIndex(meters)) if profile == "exact_fraction_v2" else None
        self.current = self.origin
        self.state = 'start'
        self.digits = ''
        self.pending = None
        # The row depends only on the committed coordinate, which changes once
        # per event; recomputing its Fraction arithmetic for every token of a
        # long prefix dominated data preparation.
        self._row_beat = None
        self._row = None

    def _row_for(self, beat):
        if self.profile == "exact_fraction_v2":
            return _coordinate_row_v2(beat, self.coordinate_origin, self.phases.phase)
        return _coordinate_row(beat, self.coordinate_origin)

    def consume(self, token_id: int) -> tuple[float, ...]:
        token = self.codec.tokens[int(token_id)]
        if token == '<pad>':
            return (0.0,) * self.dim
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
            self._row = self._row_for(self.current)
        return self._row


def causal_prefix_geometry(ids: Sequence[int] | torch.Tensor, codec, *, origin=Fraction(0),
                           coordinate_origin=None, profile: str = "exact_fraction_v1",
                           meters=None) -> torch.Tensor:
    """Build one geometry row per consumed token without reading future tokens.

    A position becomes visible only after all of its ``P:`` digits and the
    ``<note>`` marker have been consumed.  ``<end>`` commits that position;
    malformed or truncated prefixes retain the last committed coordinate.
    """
    values = ids.detach().cpu().tolist() if isinstance(ids, torch.Tensor) else list(ids)
    state = CausalGeometry(codec, origin=origin, coordinate_origin=coordinate_origin,
                           profile=profile, meters=meters)
    rows = [state.consume(token_id) for token_id in values]
    return torch.tensor(rows, dtype=torch.float32).reshape(-1, state.dim)


def event_timing_targets(decoder_input: torch.Tensor | Sequence[int], target: torch.Tensor | Sequence[int], codec, *,
                         origin=Fraction(0), profile: str = "exact_fraction_v1", meters=None) -> dict[str, torch.Tensor]:
    """Derive next-event geometry from the target token stream only.

    The decoder prefix supplies the last committed position; the target
    supplies the complete next ``<at> P:* <note>`` spelling.  No target token
    is fed back into the prefix state, so this is safe for teacher forcing.
    Under ``exact_fraction_v2`` the phase channels carry bar phase, matching
    the meter-anchored decoder geometry rather than a span-relative phase.
    """
    inputs = decoder_input.detach().cpu().tolist() if isinstance(decoder_input, torch.Tensor) else list(decoder_input)
    targets = target.detach().cpu().tolist() if isinstance(target, torch.Tensor) else list(target)
    n = min(len(inputs), len(targets))
    origin = Fraction(origin)
    index = (meters if isinstance(meters, MeterPhaseIndex)
             else MeterPhaseIndex(meters)) if profile == "exact_fraction_v2" else None
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
            gap = max(beat - current, Fraction(0))
            if index is not None:
                phase = 2.0 * math.pi * index.phase(beat)[0]
            else:
                local = float(beat - origin)
                phase = 2.0 * math.pi * (local - math.floor(local))
            rows.append((math.log1p(float(gap)), math.sin(phase), math.cos(phase))); mask.append(True)
        else:
            rows.append((0.0, 0.0, 0.0)); mask.append(False)
    labels = torch.tensor(rows, dtype=torch.float32).reshape(-1, 3)
    return {"mask": torch.tensor(mask, dtype=torch.bool), "labels": labels,
            "gap_log1p": labels[:, 0], "phase_sin": labels[:, 1], "phase_cos": labels[:, 2]}


def timing_targets(points: Sequence, decoder_input: torch.Tensor | Sequence[int], target: torch.Tensor | Sequence[int], codec, *,
                   origin=Fraction(0), profile: str = "exact_fraction_v1", meters=None) -> dict[str, torch.Tensor]:
    """Create masked next-event gap/phase labels at BOS and committed END.

    ``points`` must be the exact ordered events for the span.  A row is valid
    only when the target token starts the next event (``<at>``); EOS and all
    interior spelling tokens receive a false mask and zero labels.
    """
    inputs = decoder_input.detach().cpu().tolist() if isinstance(decoder_input, torch.Tensor) else list(decoder_input)
    targets = target.detach().cpu().tolist() if isinstance(target, torch.Tensor) else list(target)
    n = min(len(inputs), len(targets))
    origin = Fraction(origin)
    index = (meters if isinstance(meters, MeterPhaseIndex)
             else MeterPhaseIndex(meters)) if profile == "exact_fraction_v2" else None
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
            if index is not None:
                phase = 2.0 * math.pi * index.phase(beat)[0]
            else:
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


def position_event_loss(logits: torch.Tensor, target: torch.Tensor, decoder_input: torch.Tensor, codec, *,
                        reduction="mean", token_ce: torch.Tensor | None = None) -> torch.Tensor:
    """Cross entropy averaged per Fraction spelling, then per event.

    A token-averaged objective weights an event by its spelling length.  A
    Fraction needs one digit per character, so a denominator like ``123/128``
    costs several times the supervision of a one-digit integer, and the
    objective drifts toward whichever events happen to have long spellings.
    Averaging each event's position spelling first makes one event one unit of
    supervision, which matches how a chart is actually read.

    Only the position spelling is regrouped.  Note text keeps its ordinary
    token supervision; this helper is additive and leaves the token head and
    checkpoint contract untouched.
    """
    ce = (torch.nn.functional.cross_entropy(logits.float().transpose(1, 2), target, reduction="none")
          if token_ce is None else token_ce)
    if ce.shape != target.shape or decoder_input.shape != target.shape:
        raise ValueError('Position loss requires matching token loss, target and prefix shapes')
    valid = target != codec.pad_id
    boundary = ((decoder_input == codec.bos_id) | (decoder_input == codec.end_id)) & (target == codec.at_id)
    event_ids = boundary.long().cumsum(dim=1)
    positions = torch.arange(1, target.size(1) + 1, device=target.device)[None, :]
    # A token belongs to a spelling iff the latest AT occurs after the latest
    # NOTE/END. Prefix maxima implement the original state machine without
    # launching tiny CUDA operations for each token or synchronizing unique().
    starts = torch.where(decoder_input == codec.at_id, positions, 0).cummax(dim=1).values
    stops = torch.where((decoder_input == codec.note_id) | (decoder_input == codec.end_id),
                        positions, 0).cummax(dim=1).values
    selected = (starts > stops) & valid & (event_ids > 0)
    weights = selected.to(ce.dtype)
    shape = (target.size(0), target.size(1) + 1)
    totals = ce.new_zeros(shape).scatter_add(1, event_ids, ce * weights)
    counts = ce.new_zeros(shape).scatter_add(1, event_ids, weights)
    means = totals / counts.clamp_min(1)
    per_row = means.sum(dim=1) / (counts > 0).sum(dim=1).clamp_min(1)
    return per_row.mean() if reduction == "mean" else per_row


def numeric_timing_loss(prediction: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, *, reduction="mean") -> torch.Tensor:
    """Masked Smooth-L1 loss for ``[log1p(gap), sin(phase), cos(phase)]``."""
    if prediction.shape != labels.shape or mask.shape != prediction.shape[:-1]:
        raise ValueError('Timing prediction, labels and mask shapes differ')
    if reduction not in ('mean', 'none'):
        raise ValueError('Timing reduction must be mean or none')
    mask = mask.bool()
    # Ignore unsupervised spellings/padding *before* the nonlinear objective.
    # NaN * 0 is still NaN, and masking afterwards can poison valid gradients.
    values = torch.nn.functional.smooth_l1_loss(
        prediction.float().masked_fill(~mask[..., None], 0.),
        labels.float().masked_fill(~mask[..., None], 0.), reduction="none").mean(-1)
    per_row = (values * mask.to(values.dtype)).sum(-1) / mask.sum(-1).clamp_min(1)
    return per_row.mean() if reduction == "mean" else per_row


__all__ = ["GEOMETRY_DIM", "GEOMETRY_DIM_V2", "GEOMETRY_PROFILES", "geometry_dim",
           "coordinate_features", "CausalGeometry", "causal_prefix_geometry",
           "event_timing_targets", "timing_targets", "position_event_loss", "numeric_timing_loss"]
