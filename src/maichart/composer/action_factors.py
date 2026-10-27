"""Parser-backed residual factors over the existing legal action catalog.

The catalog excludes STOP: entry ``i`` keeps action ID ``i + 1``.  Add this
module's logits to the original action head, and its embeddings to the original
event embeddings.  Factor weights are shared between both paths and start at
zero, preserving the existing model without a zero gain that blocks gradients.
"""
from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from maichart.duration import DurationExpr, parse_duration_expr
from maichart.notes import ParsedNote, parse_timing_point_notes
from maichart.timing import RawTimingPoint


FACTOR_CONTRACT = 'parsed_legal_action_factors_v1'
JOINT_FACTOR_CONTRACT = 'parsed_legal_action_factors_v2'
SLIDE_FACTOR_POLICIES = {'marginal_v1', 'branch_joint_v2'}
KIND_BITS = {name: 1 << index for index, name in enumerate(
    ('tap', 'hold', 'slide', 'touch', 'touch_hold'))}
TOUCH_RING_BITS = {name: 1 << index for index, name in enumerate('ABCDE')}
FACTOR_FIELDS = (
    'kind_mask', 'head_lane_mask', 'hold_duration', 'slide_duration',
    'trajectory_shape', 'relative_endpoint', 'break_ex', 'branch_count',
    'touch_ring', 'firework',
)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(',', ':'), sort_keys=True)


def _duration_key(duration: DurationExpr | None) -> tuple:
    if duration is None:
        raise ValueError('A parsed sustained note is missing its duration')
    # Reparse without chart BPM so seconds are never conflated with grid beats.
    expr = parse_duration_expr(duration.raw)
    if expr.kind == 'grid_fraction':
        return expr.kind, str(expr.beats)
    if expr.kind == 'seconds':
        return expr.kind, str(Fraction(expr.raw[1:-1].strip()[1:]))
    if expr.kind == 'timing_pair':
        return expr.kind, tuple(str(Fraction(item.strip()))
                               for item in expr.raw[1:-1].split('##'))
    raise ValueError(f'Unsupported factor duration: {duration.raw!r}')


def _sorted(values: list) -> tuple:
    return tuple(sorted(values, key=_canonical))


def _slide_keys(note: ParsedNote) -> tuple:
    shapes, endpoints, durations = [], [], []
    for segment in note.slide_segments:
        start = int(segment['start_position'])
        parts = segment.get('path_parts') or [segment]
        branch_shapes, branch_endpoints = [], []
        current = start
        for part in parts:
            branch_shapes.append(str(part['pattern']))
            # Preserve every intermediate argument (including a V turn), and
            # advance only within a chain, never between same-head branches.
            branch_endpoints.append(tuple((int(arg) - current) % 8
                                          for arg in part['path_args']))
            current = int(part['end_position'])
        shapes.append(tuple(branch_shapes))
        endpoints.append(tuple(branch_endpoints))
        durations.append(_duration_key(segment['duration']))
    if not shapes:
        raise ValueError(f'A parsed slide has no trajectory: {note.raw!r}')
    return _sorted(shapes), _sorted(endpoints), _sorted(durations)


def _slide_joint_keys(note: ParsedNote) -> tuple:
    branches = []
    for segment in note.slide_segments:
        current = int(segment['start_position'])
        path = []
        for part in segment.get('path_parts') or [segment]:
            path.append((str(part['pattern']),
                         tuple((int(arg)-current) % 8 for arg in part['path_args'])))
            current = int(part['end_position'])
        # Keep path, duration, and segment modifiers associated with one branch.
        # Head modifiers are note-level and already represented by break_ex;
        # assigning them to the first textual branch would break permutation.
        branches.append((tuple(path), _duration_key(segment['duration']),
                         segment.get('modifiers', {})))
    return _sorted(branches)


def _parse_factors(atom: str, *, joint=False) -> dict[str, Any]:
    parsed = parse_timing_point_notes(RawTimingPoint(
        index=0, raw=atom, note_text=atom, beat=Fraction(0), tick=0,
        time_sec=0., division=4, bpm=120.,
    ))
    if parsed.unknown_tokens or not parsed.notes:
        reasons = ', '.join(f'{token.raw!r}: {token.reason}'
                            for token in parsed.unknown_tokens)
        raise ValueError(f'Invalid Composer factor atom {atom!r}: {reasons or "empty event"}')
    result = dict(kind_mask=0, head_lane_mask=0, hold_duration=[],
                  slide_duration=[], trajectory_shape=[], relative_endpoint=[],
                  break_ex=0, branch_count=0, touch_ring=0, firework=0)
    if joint:
        result['slide_branch_joint'] = []
    for note in parsed.notes:
        if note.note_type not in KIND_BITS:
            raise ValueError(f'Unsupported parsed note kind: {note.note_type!r}')
        result['kind_mask'] |= KIND_BITS[note.note_type]
        if isinstance(note.position, int):
            result['head_lane_mask'] |= 1 << (note.position - 1)
        else:
            ring = note.modifiers.get('touch_area')
            if ring not in TOUCH_RING_BITS:
                raise ValueError(f'Unsupported parsed touch area: {note.position!r}')
            result['touch_ring'] |= TOUCH_RING_BITS[ring]
        result['break_ex'] |= int(note.is_break) | (2 * int(bool(note.modifiers.get('ex'))))
        result['firework'] |= int(bool(note.modifiers.get('firework')))
        if note.note_type in {'hold', 'touch_hold'}:
            result['hold_duration'].append(_duration_key(note.duration))
        if note.note_type == 'slide':
            shapes, endpoints, durations = _slide_keys(note)
            result['trajectory_shape'].append(shapes)
            result['relative_endpoint'].append(endpoints)
            result['slide_duration'].append(durations)
            result['branch_count'] += len(note.slide_segments)
            if joint:
                result['slide_branch_joint'].append(_slide_joint_keys(note))
    for name in ('hold_duration', 'slide_duration', 'trajectory_shape', 'relative_endpoint'):
        result[name] = _sorted(result[name])
    if joint:
        result['slide_branch_joint'] = _sorted(result['slide_branch_joint'])
    return result


class CompositionalActions(nn.Module):
    """Shared additive legal-action logits and history embeddings.

    ``atom_<field>`` buffers store categorical IDs; ``factor_values`` describes
    those stable categories. ``atom_head_lanes`` stores all eight button heads
    as a boolean mask, so compact chords never lose a head.  Touch pads use the
    separate ring factor and do not become button heads.
    """

    def __init__(self, atom_catalog: list[str], width: int, anchor_policy='legacy_digit_v1',
                 slide_factor_policy='marginal_v1'):
        super().__init__()
        if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
            raise ValueError('Factor width must be a positive integer')
        atoms = tuple(atom_catalog)
        if any(not isinstance(atom, str) or not atom.strip() for atom in atoms):
            raise ValueError('Factor catalog must contain nonempty note strings, excluding STOP')
        if len(set(atoms)) != len(atoms):
            raise ValueError('Factor catalog contains duplicate atoms')
        self.width = width
        if slide_factor_policy not in SLIDE_FACTOR_POLICIES:
            raise ValueError('Unknown slide_factor_policy')
        self.slide_factor_policy = slide_factor_policy
        self.contract = FACTOR_CONTRACT if slide_factor_policy == 'marginal_v1' else JOINT_FACTOR_CONTRACT
        self.factor_fields = FACTOR_FIELDS + (('slide_branch_joint',) if slide_factor_policy == 'branch_joint_v2' else ())
        self.atom_catalog = atoms
        self.vocab_size = len(atoms) + 1
        rows = [_parse_factors(atom, joint=slide_factor_policy == 'branch_joint_v2') for atom in atoms]
        self.factor_values: dict[str, tuple[str, ...]] = {}
        self.factors = nn.ModuleDict()
        for name in self.factor_fields:
            keys = [_canonical(row[name]) for row in rows]
            # STOP always has category zero, including when a note has no value
            # for this field.  Its output remains learnable and independent.
            categories = ('STOP', *sorted(set(keys)))
            ids = {key: index for index, key in enumerate(categories)}
            self.factor_values[name] = categories
            self.register_buffer(f'atom_{name}', torch.tensor(
                [0, *(ids[key] for key in keys)], dtype=torch.long))
            embedding = nn.Embedding(len(categories), width)
            nn.init.zeros_(embedding.weight)
            self.factors[name] = embedding
        head_masks = [0, *(row['head_lane_mask'] for row in rows)]
        self.register_buffer('atom_head_lanes', torch.tensor(
            [[bool(mask & (1 << lane)) for lane in range(8)] for mask in head_masks],
            dtype=torch.bool))
        heads=self.atom_head_lanes.float()
        self.register_buffer('_normalized_head_lanes',heads/heads.sum(-1,keepdim=True).clamp_min(1),persistent=False)
        atom_to_id={atom:i+1 for i,atom in enumerate(atoms)}
        # Retain the historical mapping for old checkpoints. Parsed-head
        # history never reads these duration-sensitive compatibility IDs.
        anchors=[0]
        for i,atom in enumerate(atoms,start=1):
            digit=next((c for c in atom if c in '12345678'),None)
            anchors.append(atom_to_id.get(digit,atom_to_id.get('4',i)) if digit is not None else atom_to_id.get('4',i))
        self.register_buffer('_lane_anchor_ids',torch.tensor(anchors,dtype=torch.long),persistent=False)
        self.register_buffer('_parsed_anchor_weights', torch.empty(0), persistent=False)
        self.register_buffer('_anchor_tap_ids', torch.tensor(
            [atom_to_id.get(str(lane), 0) for lane in range(1, 9)], dtype=torch.long), persistent=False)
        self.set_history_anchor_policy(anchor_policy)
        self.relative_factors = nn.Embedding(8, width)
        nn.init.zeros_(self.relative_factors.weight)
        fingerprint = hashlib.sha256(_canonical({
            'contract': self.contract, 'atoms': atoms, 'fields': self.factor_values,
        }).encode('ascii')).digest()
        self.register_buffer('catalog_fingerprint', torch.tensor(list(fingerprint), dtype=torch.uint8))

    def set_history_anchor_policy(self, policy):
        """Select a derived history representation without changing learned factors."""
        if policy not in {'legacy_digit_v1', 'parsed_heads_v2', 'parsed_geometry_v3'}:
            raise ValueError('Unknown history_anchor_policy')
        self.history_anchor_policy = policy
        weights = torch.zeros((self.vocab_size, 8), device=self._lane_anchor_ids.device)
        if policy in {'parsed_heads_v2', 'parsed_geometry_v3'}:
            for index, atom in enumerate(self.atom_catalog, 1):
                parsed = parse_timing_point_notes(RawTimingPoint(
                    index=0, raw=atom, note_text=atom, beat=Fraction(0), tick=0,
                    time_sec=0., division=4, bpm=120.))
                lanes = set()
                for note in parsed.notes:
                    position = note.position
                    if policy == 'parsed_geometry_v3':
                        if isinstance(position, int):
                            weights[index, position-1] += 1.
                        elif position != 'C' and len(position) == 2 and position[1] in '12345678':
                            lane = int(position[1])-1
                            if position[0] in 'DE':
                                weights[index, (lane-1) % 8] += .5
                                weights[index, lane] += .5
                            else:
                                weights[index, lane] += 1.
                        continue
                    if isinstance(position, int):
                        lanes.add(position)
                    elif position != 'C' and len(position) == 2 and position[1] in '12345678':
                        lanes.add(int(position[1]))
                for lane in lanes:
                    weights[index, lane - 1] = 1. / len(lanes)
                if policy == 'parsed_geometry_v3':
                    weights[index] /= weights[index].sum().clamp_min(1)
        self._parsed_anchor_weights = weights

    def history_anchor_embedding(self, ids, event_embedding):
        if self.history_anchor_policy == 'legacy_digit_v1':
            return event_embedding(self._lane_anchor_ids[ids])
        tap_ids = self._anchor_tap_ids
        taps = event_embedding(tap_ids) * tap_ids.ne(0)[:, None]
        return self._parsed_anchor_weights[ids].to(taps) @ taps

    def _clear_composed_cache(self):
        self._composed_cache_key=None
        self._composed_cache=None

    def train(self,mode=True):
        # Fused optimizers can update parameters without advancing _version.
        # An eval cache must never survive a training/evaluation boundary.
        self._clear_composed_cache()
        return super().train(mode)

    def composed_weight(self) -> torch.Tensor:
        """Return [legal IDs including STOP, width] without an ID-specific head."""
        if self.training or torch.is_grad_enabled():
            # Also cover dropout-off optimization that intentionally stays in
            # eval mode: its grad-enabled forward precedes an optimizer step.
            self._clear_composed_cache()
        cache_key=None
        if (not self.training and not torch.is_grad_enabled()
                and not any(torch.is_inference(value.weight) for value in self.factors.values())):
            cache_key=tuple((value.weight._version,value.weight.device,value.weight.dtype)
                            for value in self.factors.values())
            if getattr(self,'_composed_cache_key',None)==cache_key:return self._composed_cache
        weight = self.factors[FACTOR_FIELDS[0]](getattr(self, f'atom_{FACTOR_FIELDS[0]}'))
        for name in self.factor_fields[1:]:
            weight = weight + self.factors[name](getattr(self, f'atom_{name}'))
        if cache_key is not None:
            self._composed_cache=weight
            self._composed_cache_key=cache_key
        return weight

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        if context.ndim < 1 or context.shape[-1] != self.width:
            raise ValueError(f'Action factor context must end in width {self.width}')
        return F.linear(context, self.composed_weight())

    def embed(self, ids: torch.Tensor, *, composed_weight=None) -> torch.Tensor:
        """Additive history embeddings using exactly the output factor weights."""
        if composed_weight is not None:
            if composed_weight.shape != (self.vocab_size,self.width):
                raise ValueError('Composed factor table must match the atom catalog and width')
            return F.embedding(ids,composed_weight)
        result = None
        for name in self.factor_fields:
            categories = getattr(self, f'atom_{name}').index_select(0, ids.reshape(-1)).reshape(ids.shape)
            contribution = self.factors[name](categories)
            result = contribution if result is None else result + contribution
        return result

    def relative_logits(self, context: torch.Tensor, last_head_mask: torch.Tensor, *, factor_embedding=None,
                        target_geometry=None) -> torch.Tensor:
        """Mean over every previous/current head pair, with no hand assignment.

        ``last_head_mask`` is [batch, 8] or follows context's leading dimensions.
        Empty history and actions without button heads contribute zero.  This is
        a learned soft relation; it applies no movement rule to slot indices.
        """
        if context.ndim < 1 or context.shape[-1] != self.width:
            raise ValueError(f'Relative factor context must end in width {self.width}')
        previous = torch.as_tensor(last_head_mask, device=context.device)
        if previous.ndim < 1 or previous.shape[-1] != 8:
            raise ValueError('last_head_mask must end in eight button lanes')
        if previous.ndim > context.ndim:
            raise ValueError('last_head_mask has too many dimensions')
        previous = previous.to(dtype=context.dtype)
        while previous.ndim < context.ndim:
            previous = previous.unsqueeze(-2)
        if any(size not in (1, expected) for size, expected in
               zip(previous.shape[:-1], context.shape[:-1])):
            raise ValueError('last_head_mask cannot broadcast to the action context')
        previous = previous / previous.sum(-1, keepdim=True).clamp_min(1)
        heads = (self._normalized_head_lanes if target_geometry is None else target_geometry).to(dtype=context.dtype)
        lanes = torch.arange(8, device=context.device)
        delta = (lanes[None, :] - lanes[:, None]) % 8
        relation = torch.einsum('...d,ijd->...ij', context,
            (self.relative_factors if factor_embedding is None else factor_embedding)(delta))
        current = torch.einsum('...ij,...i->...j',relation,previous)
        return F.linear(current,heads)

    def get_extra_state(self) -> dict[str, Any]:
        return {'contract': self.contract, 'atom_catalog': list(self.atom_catalog),
                'factor_values': dict(self.factor_values)}

    def set_extra_state(self, state: dict[str, Any]) -> None:
        if state != self.get_extra_state():
            raise ValueError('Compositional action catalog identity or factor mapping mismatch')

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Validate before PyTorch copies buffers, even under strict=False.  A
        # reordered same-size catalog must never reinterpret learned rows.
        state = state_dict.get(prefix + '_extra_state')
        if state != self.get_extra_state():
            raise ValueError('Compositional action catalog identity or factor mapping mismatch')
        for name, expected in self.named_buffers(recurse=False):
            if name in self._non_persistent_buffers_set:continue
            stored = state_dict.get(prefix + name)
            if stored is None or stored.shape != expected.shape or stored.dtype != expected.dtype \
                    or not torch.equal(stored.detach().cpu(), expected.detach().cpu()):
                raise ValueError(f'Compositional action factor buffer mismatch: {name}')
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)
