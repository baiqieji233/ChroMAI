"""Shared compositional representations of exact, catalog-constrained details.

This changes parameter sharing, not the codec or the autoregressive joint
distribution. Duration spellings remain exact categories; no quantization or
independent factor losses are introduced. Candidate IDs retain catalog order.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from maichart.notes import BASIC_TAP_RE, BASIC_HOLD_RE, BASIC_SLIDE_RE


def detail_components(kind, head, suffix):
    """Extract exact syntax components, with slide destinations head-relative."""
    regex = {'tap': BASIC_TAP_RE, 'hold': BASIC_HOLD_RE,
             'slide': BASIC_SLIDE_RE}.get(kind)
    match = regex.fullmatch(str(head + 1) + suffix) if regex is not None else None
    if match is None:
        raise ValueError(f'Unsupported regular detail: {(kind, head, suffix)!r}')
    fields = match.groupdict()
    # Keep modifier placement and ordering: equivalent spellings may coexist in
    # the exact catalog, and must not be accidentally forced to identical scores.
    modifiers = (
        fields.get('modifiers', fields.get('before_modifiers', fields.get('head_modifiers', ''))),
        fields.get('after_modifiers', fields.get('before_duration_modifiers', '')),
        fields.get('tail_modifiers', ''),
    )
    endpoints = fields.get('end_position', '')
    relative = tuple((int(digit) - 1 - head) % 8 for digit in endpoints)
    return (kind, modifiers, fields.get('duration') or '',
            fields.get('pattern', ''), relative)


class SharedDetailEmbedding(nn.Module):
    """Encode each distinct structure once, then gather legal candidate rows.

    ``bank = encode_catalog()`` may be shared within a single forward/loss call.
    Never retain this differentiable bank on the module across optimizer steps.
    ``candidate_weights(shape, slot, bank=None)`` returns [catalog_size, width].
    The same rows serve the output classifier and teacher/generated feedback.
    """
    def __init__(self, codec, width=32):
        super().__init__()
        structures = []
        structure_ids = {}
        self._indices = {}
        self._shape_indices = {}
        self._shape_slot_indices = {}
        self._shape_slots = {}
        self._slot_unique = {}
        self._slot_inverse = {}
        for shape, slots in enumerate(codec.shape_catalog):
            if not shape:
                continue
            self._shape_slots[shape] = len(slots)
            for slot, (kind, head) in enumerate(slots):
                ids = []
                for suffix in codec.detail_catalog[shape][slot]:
                    parts = detail_components(kind, head, suffix)
                    if parts not in structure_ids:
                        structure_ids[parts] = len(structures)
                        structures.append(parts)
                    ids.append(structure_ids[parts])
                name = f'candidate_index_{shape}_{slot}'
                self.register_buffer(name, torch.tensor(ids, dtype=torch.long), persistent=False)
                self._indices[shape, slot] = name
                unique_ids = sorted(set(ids))
                local_ids = {value: i for i, value in enumerate(unique_ids)}
                name = f'slot_unique_{shape}_{slot}'
                self.register_buffer(name, torch.tensor(unique_ids, dtype=torch.long), persistent=False)
                self._slot_unique[shape, slot] = name
                name = f'slot_inverse_{shape}_{slot}'
                self.register_buffer(name, torch.tensor([local_ids[value] for value in ids], dtype=torch.long), persistent=False)
                self._slot_inverse[shape, slot] = name
            joined = [int(index) for slot in range(len(slots))
                      for index in getattr(self, self._indices[shape, slot]).tolist()]
            unique = sorted(set(joined))
            local = {index: i for i, index in enumerate(unique)}
            name = f'shape_index_{shape}'
            self.register_buffer(name, torch.tensor(unique, dtype=torch.long), persistent=False)
            self._shape_indices[shape] = name
            for slot in range(len(slots)):
                rows = getattr(self, self._indices[shape, slot]).tolist()
                name = f'shape_slot_index_{shape}_{slot}'
                self.register_buffer(name, torch.tensor([local[index] for index in rows], dtype=torch.long), persistent=False)
                self._shape_slot_indices[shape, slot] = name
        self.component_embeddings = nn.ModuleList()
        component_ids = []
        for coordinate in range(5):
            vocabulary = {}
            ids = []
            for parts in structures:
                value = parts[coordinate]
                if value not in vocabulary:
                    vocabulary[value] = len(vocabulary)
                ids.append(vocabulary[value])
            self.component_embeddings.append(nn.Embedding(max(1, len(vocabulary)), width))
            component_ids.append(ids)
        self.register_buffer('component_ids', torch.tensor(component_ids, dtype=torch.long).reshape(5, -1),
                             persistent=False)
        self.compose = nn.Sequential(nn.Linear(width * 5, width * 2), nn.SiLU(),
                                     nn.Linear(width * 2, width), nn.LayerNorm(width))
        self.width = int(width)
        self.num_structures = len(structures)

    def encode_catalog(self):
        parts = [embedding(self.component_ids[i])
                 for i, embedding in enumerate(self.component_embeddings)]
        return self.compose(torch.cat(parts, dim=-1))

    def encode_shape(self, shape):
        """Compose only this shape's unique structures, once per decode call.

        Returns a tuple of slot candidate matrices in original catalog order.
        No model cache is retained, so this also remains safe with gradients.
        """
        ids = getattr(self, self._shape_indices[shape])
        selected = self.component_ids.index_select(-1, ids)
        bank = self.compose(torch.cat([embedding(selected[i])
                    for i, embedding in enumerate(self.component_embeddings)], dim=-1))
        return tuple(bank.index_select(0, getattr(self, self._shape_slot_indices[shape, slot]))
                     for slot in range(self._shape_slots[shape]))

    def candidate_weights(self, shape, slot, bank=None):
        if bank is None:
            ids = getattr(self, self._slot_unique[shape, slot])
            selected = self.component_ids.index_select(-1, ids)
            encoded = self.compose(torch.cat([embedding(selected[i])
                    for i, embedding in enumerate(self.component_embeddings)], dim=-1))
            return encoded.index_select(0, getattr(self, self._slot_inverse[shape, slot]))
        return bank.index_select(0, getattr(self, self._indices[shape, slot]))

    def embedding_for(self, shape, slot, indices, bank=None):
        return F.embedding(indices, self.candidate_weights(shape, slot, bank))
