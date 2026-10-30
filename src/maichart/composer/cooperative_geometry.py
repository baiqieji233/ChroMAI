"""Soft coordination for later chord slots, without assigning hands.

The old first-slot movement branch stays independent.  These zero-initialized
tables learn how later button notes relate to the last committed event and to
the heads and slide destinations already selected at this event. Both Tap-first
and Slide-first chords can therefore coordinate with the other note's endpoint.
A slide destination is a geometric hint, not current pad occupancy.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .events import event_metadata


class CooperativeGeometry(nn.Module):
    def __init__(self, action_factors, width: int):
        super().__init__()
        self.later_cross_factors = nn.Embedding(8, width)
        self.end_to_head_factors = nn.Embedding(8, width)
        self.end_to_end_factors = nn.Embedding(8, width)
        self.head_to_end_factors = nn.Embedding(8, width)
        for table in (self.later_cross_factors, self.end_to_head_factors,
                      self.end_to_end_factors, self.head_to_end_factors):
            nn.init.zeros_(table.weight)

        kinds = [None]
        slides = [False]
        for atom in action_factors.atom_catalog:
            meta = event_metadata(atom)
            # Touch-containing atoms retain their own probability unchanged.
            kinds.append(None if meta.touch else 2 if meta.slide else 1 if meta.hold else 0)
            slides.append(meta.slide and not meta.touch)
        slide_mask = torch.tensor(slides, dtype=torch.bool)
        self.register_buffer('prefix_end_lanes',
            action_factors.atom_end_lanes.clone() & slide_mask[:, None], persistent=False)
        self.register_buffer('target_heads',
            action_factors._normalized_head_lanes.clone(), persistent=False)
        self.register_buffer('target_ends',
            action_factors._normalized_end_lanes.clone() * slide_mask[:, None], persistent=False)
        lanes = torch.arange(8)
        self.register_buffer('relative_offsets', (lanes[None, :] - lanes[:, None]) % 8,
                             persistent=False)
        for kind in range(3):
            self.register_buffer(f'kind_{kind}', torch.tensor(
                [index for index, value in enumerate(kinds) if value == kind], dtype=torch.long),
                persistent=False)

    def _lane_scores(self, context, source, table):
        source = source.to(device=context.device, dtype=context.dtype)
        while source.ndim < context.ndim:
            source = source.unsqueeze(-2)
        source = source / source.sum(-1, keepdim=True).clamp_min(1)
        # Compute eight offset scores first; the vocabulary projection below
        # avoids materializing an extra full-vocabulary tensor for every relation.
        offsets = F.linear(context, table.weight)
        return (offsets[..., self.relative_offsets] * source.unsqueeze(-1)).sum(-2)

    def forward(self, logits, context, *, previous_heads=None, prefix_heads=None, prefix_ends=None,
                slots=0, history_strength=1., active=None):
        head_scores = context.new_zeros((*context.shape[:-1], 8))
        end_scores = None
        if previous_heads is not None:
            later = (slots > 0) if isinstance(slots, int) else slots.gt(0)
            if isinstance(later, torch.Tensor):
                later = later.unsqueeze(-1)
            head_scores = head_scores + self._lane_scores(
                context, previous_heads, self.later_cross_factors) * later * float(history_strength)
        if prefix_ends is not None:
            head_scores = head_scores + self._lane_scores(
                context, prefix_ends, self.end_to_head_factors)
            end_scores = self._lane_scores(context, prefix_ends, self.end_to_end_factors)
        if prefix_heads is not None:
            from_heads = self._lane_scores(context, prefix_heads, self.head_to_end_factors)
            end_scores = from_heads if end_scores is None else end_scores + from_heads
        if active is not None:
            head_scores = head_scores * active.unsqueeze(-1)
            if end_scores is not None:
                end_scores = end_scores * active.unsqueeze(-1)

        result = logits
        for kind in range(3):
            ids = getattr(self, f'kind_{kind}')
            if not ids.numel():
                continue
            original = logits.index_select(-1, ids).float()
            delta = F.linear(head_scores, self.target_heads.index_select(0, ids).to(head_scores))
            if end_scores is not None and kind == 2:
                delta = delta + F.linear(end_scores, self.target_ends.index_select(0, ids).to(end_scores))
            adjusted = original + delta.float()
            # Redistribute position/shape choices within each button type.
            # Neither Touch/STOP logits nor total Tap/Hold/Slide mass changes.
            adjusted = adjusted + (original.logsumexp(-1, keepdim=True)
                                   - adjusted.logsumexp(-1, keepdim=True))
            result = result.index_copy(-1, ids, adjusted.to(logits.dtype))
        return result
