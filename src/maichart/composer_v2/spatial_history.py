"""Causal observations of completed symbolic contacts on the DX disk."""

from dataclasses import dataclass

import torch
from torch import nn

from maichart.generation.judgment_system import PADS, pad_center


POLICY = 'disk_geometry_v1'
_CONTACTS = (*range(1, 9), *PADS)
_RIM_BITS = (1 << 8) - 1
_HISTORY_SIZE = 3
_CONTEXT_WIDTH = 27
_RIM_FEATURES = (
    'nearest_distance', 'repeat_contact', 'step', 'step_size', 'opposite',
    'repeat_step', 'same_direction', 'turn_back', 'return_to_previous', 'repeat_step_at_matching_gap',
)


def _latest(when, positions, before):
    return torch.maximum(_prefix_max(when * positions), before[:, None])


def _prefix_max(value):
    return value if value.size(1) == 1 else value.cummax(1).values


def _previous(value, before):
    return torch.cat((before[:, None], value[:, :-1]), dim=1)


def _rim_step(before, after, known):
    """There is no unique signed direction for an antipodal move."""
    wrapped = (after - before) % 8
    signed = (wrapped + 4) % 8 - 4
    signed = torch.where(known & wrapped.ne(4), signed, 0)
    size = torch.where(known, wrapped.minimum(8 - wrapped), 0)
    return signed, size


@dataclass
class SpatialFrame:
    contacts: torch.Tensor
    completed_beats: torch.Tensor
    layout: torch.Tensor
    rim_query: torch.Tensor


class DiskSpatialHistory(nn.Module):
    """Observe CTX and TGT, with one bounded parser state per cached token.

    Working storage and computation are linear in the supplied suffix. All
    prefix scans stay on the tensor device. Rewinds just move the decoder's
    existing length pointer, so the same suffix scan handles branch retries.
    """

    def __init__(self, codec, width):
        super().__init__()
        self.codec = codec
        self.tap_id = codec.ids.get('N:tap', -1)
        self.timing_only_ids = frozenset(i for i,t in enumerate(codec.tokens)
            if t.startswith('NUM:') or t in ('[AT]', '[PHASE]', '[END_EVENT]'))
        self.rim_kind_ids = frozenset(i for i,t in enumerate(codec.tokens) if t in ('N:hold','N:slide','N:tap'))
        self.context = nn.Linear(_CONTEXT_WIDTH, width, bias=False)
        self.rim_readout = nn.Linear(width, len(_RIM_FEATURES))
        nn.init.zeros_(self.context.weight)
        nn.init.zeros_(self.rim_readout.weight)
        nn.init.zeros_(self.rim_readout.bias)

        # Kind roles separate actual rim and Touch contacts from Slide knots.
        region = {'[CTX]': 1, '[REF]': 2, '[TGT]': 3}
        kind = {'N:tap': 1, 'N:hold': 1, 'N:slide': 1, 'N:touch': 2, 'N:touch_hold': 2}
        contacts = {f'P:{p}': i for i, p in enumerate(_CONTACTS)}
        self.register_buffer('token_fields', torch.tensor([
            (region.get(t, 0), kind.get(t, 0), contacts.get(t, -1), t.startswith('F:'))
            for t in codec.tokens
        ], dtype=torch.long), persistent=False)
        xy = torch.tensor([pad_center(p) for p in _CONTACTS], dtype=torch.float32)
        bits = 1 << torch.arange(len(_CONTACTS))
        self.register_buffer('contact_bits', bits, persistent=False)
        self.register_buffer('contact_xy', xy, persistent=False)
        self.register_buffer('contact_indices', torch.arange(len(_CONTACTS)), persistent=False)
        distances = torch.cdist(xy, xy[:8], compute_mode='donot_use_mm_for_euclid_dist')
        # Each group has at most eight contacts. Its 256 bit masks let one
        # gather replace a per-query expansion over every contact and key.
        padding = (-len(_CONTACTS)) % 8
        grouped = torch.cat((distances, distances.new_full((padding, 8), 3.))).reshape(-1, 8, 8)
        group_bits = (torch.arange(256)[:, None] & bits[:8]).ne(0)
        nearest = torch.where(group_bits[None, :, :, None], grouped[:, None], 3.).amin(-2)
        groups = torch.arange(len(nearest))
        self.register_buffer('nearest_rim', nearest.flatten(0, 1), persistent=False)
        self.register_buffer('nearest_shifts', 8 * groups, persistent=False)
        self.register_buffer('nearest_offsets', 256 * groups, persistent=False)
        self.register_buffer('rim_lanes', torch.arange(8), persistent=False)
        self.register_buffer('rim_columns', torch.tensor([i for i in range(8) if f'P:{i + 1}' in codec.ids],
                                                         dtype=torch.long), persistent=False)

    @staticmethod
    def _remember(state, name, value, start):
        length = start + value.size(1)
        saved = state.get(name)
        if saved is None or saved.size(1) < length:
            capacity = max(length, 128 if saved is None else 2 * saved.size(1))
            storage = value.new_empty((value.size(0), capacity, *value.shape[2:]))
            if start:
                storage[:, :start] = saved[:, :start]
            saved = state[name] = storage
        saved[:, start:length] = value

    def scan(self, ids, beats, visible, *, start=0, state=None):
        """Return observations after each input token. Labels never enter here."""
        batch, length = ids.shape
        if start:
            prior, seen_before, prior_contacts, prior_beats = (
                state[name][:, start - 1] for name in ('parser', 'seen', 'contacts', 'beats'))
        else:
            prior = ids.new_zeros(batch, 8)
            prior_contacts = ids.new_zeros(batch, _HISTORY_SIZE)
            prior_beats = beats.new_zeros(batch, _HISTORY_SIZE)
            if state is not None:
                seen_before = ids.new_zeros(batch, len(_CONTACTS), dtype=torch.int32)

        (region_before, open_before, fence_before, hidden_before, tap_before, other_before,
         phase_before, previous_kind) = prior.unbind(-1)
        region_code, kind, pad, phase = self.token_fields[ids].unbind(-1)
        position = torch.arange(start + 1, start + length + 1, device=ids.device)
        region = torch.maximum(_prefix_max(region_code), region_before[:, None])
        observing = region.eq(1) | region.eq(3)
        opened = _latest(ids.eq(self.codec.at_id), position, open_before)
        ended = ids.eq(self.codec.event_end_id)
        delimiter = ended | region_code.ne(0) | ids.eq(self.codec.carry_id) | ids.eq(self.codec.bos_id)
        fence = _latest(delimiter, position, fence_before)
        inside = opened > _previous(fence, fence_before)
        hidden = _latest(~visible, position, hidden_before)
        phase_seen = _latest(phase.ne(0), position, phase_before)
        layout = observing & inside & phase_seen.gt(opened) & ~ended & visible

        # Legacy Tap atoms may contain several consecutive button tokens.
        # Other note kinds have exactly one head, followed by their payload.
        previous = _previous(kind, previous_kind)
        tap = ids.eq(self.tap_id)
        tap_begin = _latest(tap, position, tap_before)
        other = _latest((kind.ne(0) & ~tap) | ids.eq(self.codec.end_note_id) | delimiter, position, other_before)
        first_head = (previous > 0) & (pad.lt(8) == previous.eq(1))
        head = inside & observing & pad.ge(0) & (first_head | (tap_begin.gt(other) & pad.lt(8)))
        if state is None:
            # Full sequences can place each completed atom in its event and
            # pad slot once. Equal writes of True collapse duplicate contacts.
            slots = torch.where(head, opened * len(_CONTACTS) + pad, 0)
            observed = ids.new_zeros(batch, (length + 1) * len(_CONTACTS), dtype=torch.bool)
            observed.scatter_(1, slots, True)
            masks = (observed.reshape(batch, length + 1, len(_CONTACTS)) * self.contact_bits).sum(-1)
            contact_mask = masks.gather(1, opened)
        else:
            # Cached prefixes also preserve partial events for mid-atom rewind.
            matches = head[..., None] & pad[..., None].eq(self.contact_indices)
            seen = torch.maximum(_prefix_max(matches * position[:, None].int()), seen_before[:, None])
            contact_mask = ((seen > opened[..., None]) * self.contact_bits).sum(-1)
        complete = ended & observing & inside
        observable = complete & hidden.lt(opened) & phase_seen.gt(opened)
        committed = torch.where(observable, contact_mask, 0)

        # Three synthetic indices hold the previous suffix's complete events.
        # A hidden completed event remains in sequence as an unknown observation.
        initial = ids.new_full((batch,), _HISTORY_SIZE)
        indices = position + _HISTORY_SIZE - start
        last = _prefix_max(torch.where(complete, indices, initial[:, None]))
        links = torch.cat((ids.new_tensor((0, 0, 1, 2)).expand(batch, -1), _previous(last, initial)), dim=1)
        older = links.gather(1, last)
        selected = torch.stack((last, older, links.gather(1, older)), dim=-1).flatten(1)

        def recent(source, before):
            rows = torch.cat((source.new_zeros((batch, 1)), before.flip(1), source), dim=1)
            return rows.gather(1, selected).reshape(batch, length, _HISTORY_SIZE)

        contacts = recent(committed, prior_contacts)
        completed_beats = recent(beats, prior_beats)
        if state is not None:
            parser = torch.stack((region, opened, fence, hidden, tap_begin, other, phase_seen, kind), dim=-1)
            self._remember(state, 'parser', parser, start)
            self._remember(state, 'seen', seen, start)
            self._remember(state, 'contacts', contacts, start)
            self._remember(state, 'beats', completed_beats, start)
        return SpatialFrame(contacts, completed_beats, layout, layout & kind.eq(1))

    def _motion(self, frame, beats):
        known = frame.contacts.ne(0)
        rim = frame.contacts & _RIM_BITS
        single = rim.ne(0) & (rim & (rim - 1)).eq(0) & rim.eq(frame.contacts)
        lanes = ((rim[..., None] & self.contact_bits[:8]).ne(0) * self.rim_lanes).sum(-1)
        consecutive = single[..., :-1] & single[..., 1:]
        steps, sizes = _rim_step(lanes[..., 1:], lanes[..., :-1], consecutive)
        intervals = (frame.completed_beats[..., :-1] - frame.completed_beats[..., 1:]).clamp_min(0)
        intervals = intervals * (known[..., :-1] & known[..., 1:])
        age = (beats - frame.completed_beats[..., 0]).clamp_min(0)
        return known, single, lanes, consecutive, steps, sizes, intervals, age

    def features(self, frame, beats, *, single_cpu=False, context_only=False):
        if single_cpu and not bool(frame.layout & frame.contacts[..., 0].ne(0)):
            return None, None
        known, single, lanes, consecutive, steps, sizes, intervals, age = self._motion(frame, beats)
        touches = frame.contacts[..., None].bitwise_and(self.contact_bits).ne(0)
        count = touches.sum(-1).clamp_min(1)
        xy = touches.to(beats.dtype) @ self.contact_xy.to(beats.dtype)
        xy = xy / count[..., None]
        chord = count.gt(1)
        touch = touches[..., 8:].any(-1)
        ring_direction = steps.ne(0).all(-1)
        same_direction = ring_direction & steps[..., 0].sign().eq(steps[..., 1].sign())
        direction_change = ring_direction & ~same_direction
        previous_rim = lanes[..., 0].eq(lanes[..., 2]) & single.all(-1) & steps[..., 0].ne(0)
        context = torch.cat((xy.flatten(-2), known, chord, touch, steps / 4, sizes / 4,
                             consecutive, intervals.log1p(), age.log1p()[..., None],
                             same_direction[..., None], direction_change[..., None], previous_rim[..., None]), dim=-1)
        # The published query clock may be far ahead of the last completed note.
        fresh = known[..., 0] * frame.layout / (1 + age)
        context.mul_(fresh[..., None])
        if context_only or (single_cpu and not bool(frame.rim_query)):
            return context, None

        latest = (frame.contacts[..., 0, None] >> self.nearest_shifts) & 255
        distance = self.nearest_rim[latest + self.nearest_offsets].amin(-2)
        repeat = (frame.contacts[..., 0, None] & self.contact_bits[:8]).ne(0)
        step, step_size = _rim_step(lanes[..., 0, None], self.rim_lanes, single[..., 0, None])
        prior_step = steps[..., 0, None]
        directed = step.ne(0) & prior_step.ne(0)
        continuing = directed & step.sign().eq(prior_step.sign())
        previous_gap = intervals[..., 0]
        gap_match = (1 + (age - previous_gap).abs()).reciprocal() * consecutive[..., 0]
        returns = consecutive[..., 0] & lanes[..., 0].ne(lanes[..., 1])
        repeat_step = directed & step.eq(prior_step)
        candidate = torch.stack((distance / 2, repeat, step / 4, step_size / 4, step_size.eq(4),
                                 repeat_step, continuing, directed & ~continuing,
                                 lanes[..., 1, None].eq(self.rim_lanes) & returns[..., None],
                                 gap_match[..., None] * repeat_step), dim=-1)
        candidate.mul_((fresh * frame.rim_query)[..., None, None])
        return context, candidate

    def rim_bias(self, hidden, features):
        weights = self.rim_readout(hidden)
        bias = (weights[..., None, :] * features.to(weights.dtype)).sum(-1)
        return bias.index_select(-1, self.rim_columns)
