"""Small audio-linked archive of strictly committed Composer events.

Retrieval shares the same causal boundary in teacher forcing and generation.
Audio relations may describe the complete song; chart values beyond the committed
prefix are never read. The archive is optional and does not enforce repetition.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from fractions import Fraction
import math

import numpy as np
import torch
from torch import nn

MOTIF_POLICY = 'audio_recurrence_archive_v1'
MOTIF_POLICIES = {'off', MOTIF_POLICY}


@dataclass(frozen=True, slots=True)
class MotifArchive:
    atom_ids: np.ndarray
    relative_beats: np.ndarray
    mask: np.ndarray


class ComposerMotifIndex:
    """Index a sorted event sequence once, then inspect O(budget) labels/query.

    ``history_start_index`` is the first index retained by the ordinary rolling
    history. Only older archive entries can be retrieved. ``committed_count``
    is exclusive and must be the same committed prefix used by that history.
    """

    def __init__(self, relations, beats, atom_ids):
        self.relations = relations
        self.beats = tuple(Fraction(beat) for beat in beats)
        if any(a > b for a, b in zip(self.beats, self.beats[1:])):
            raise ValueError('Motif beats must be sorted')
        self.atom_ids = np.asarray(atom_ids, dtype=np.int64)
        if self.atom_ids.ndim != 2 or len(self.atom_ids) != len(self.beats):
            raise ValueError('Motif atom_ids must have shape [events, action_slots]')
        self.spans = {span.index: span for span in relations.spans} if relations is not None else {}
        self.targets = sorted(self.spans.values(), key=lambda span: span.start)
        self.starts = [span.start for span in self.targets]
        self.links = {}
        for span in self.targets:
            self.links[span.index] = tuple(relations.prior_links(span.index)[:4])

    def append(self, beat, atom_ids):
        """Commit one generated event; retrieval never owns proposed events."""
        beat = Fraction(beat)
        row = np.asarray(atom_ids, dtype=np.int64)
        if row.shape != (self.atom_ids.shape[1],) or (self.beats and beat <= self.beats[-1]):
            raise ValueError('Invalid motif commit')
        self.beats = (*self.beats, beat)
        self.atom_ids = np.concatenate((self.atom_ids, row[None]), axis=0)

    def retrieve(self, query_beat, *, committed_count, history_start_index, max_events=16):
        if not 0 <= max_events <= 128:
            raise ValueError('Motif max_events must be in [0,128]')
        if not 0 <= history_start_index <= committed_count <= len(self.beats):
            raise ValueError('Invalid committed motif prefix')
        result = MotifArchive(np.zeros((max_events, self.atom_ids.shape[1]), np.int64),
                              np.zeros(max_events, np.float32), np.zeros(max_events, bool))
        query = Fraction(query_beat)
        target_pos = bisect_right(self.starts, query) - 1
        if not max_events or target_pos < 0:
            return result
        target = self.targets[target_pos]
        if query >= target.end:
            return result
        # Never inspect atom values in the future or in the ordinary history.
        stop = min(history_start_index, committed_count, bisect_left(self.beats, query))
        candidates = {}
        for rank, relation in enumerate(self.links.get(target.index, ())):
            source = self.spans[relation.source]
            if source.end <= source.start or relation.phase_bins <= 0:
                continue
            scale = (target.end - target.start) / (source.end - source.start)
            phase = Fraction(relation.phase_shift_bins, relation.phase_bins) * (target.end-target.start)
            source_query = source.start + (query-target.start+phase)/scale
            lo = bisect_left(self.beats, source.start, 0, stop)
            hi = bisect_left(self.beats, source.end, lo, stop)
            center = bisect_left(self.beats, source_query, lo, hi)
            # Only 2*budget nearby indices per link, irrespective of song size.
            for index in range(max(lo, center-max_events), min(hi, center+max_events)):
                projected = target.start + (self.beats[index]-source.start)*scale-phase
                if not target.start <= projected < target.end:
                    continue
                key = (abs(projected-query), rank, index)
                if index not in candidates or key < candidates[index][0]:
                    candidates[index] = (key, projected)
        selected = sorted(candidates, key=lambda i: candidates[i][0])[:max_events]
        selected.sort(key=lambda i: (candidates[i][1], i))
        for slot, index in enumerate(selected):
            result.atom_ids[slot] = self.atom_ids[index]
            result.relative_beats[slot] = float(candidates[index][1]-query)
            result.mask[slot] = bool(np.any(result.atom_ids[slot] != 0))
        return result

    def retrieve_many(self, query_beats, *, committed_count, history_start_index, max_events=16):
        rows = [self.retrieve(beat, committed_count=committed_count,
                             history_start_index=history_start_index, max_events=max_events)
                for beat in query_beats]
        if not rows:
            return MotifArchive(np.zeros((0,max_events,self.atom_ids.shape[1]),np.int64),
                                np.zeros((0,max_events),np.float32),np.zeros((0,max_events),bool))
        return MotifArchive(np.stack([r.atom_ids for r in rows]),
                            np.stack([r.relative_beats for r in rows]),
                            np.stack([r.mask for r in rows]))


class MotifMemoryAdapter(nn.Module):
    """Single bounded attention read; zero initialization preserves the parent.

    Returns a residual with ``query`` shape [B,D] or [B,Q,D]. ``mask`` uses True
    for present archive events. Shared atom embedding may be a callable, so a
    Composer can retain its existing compositional atom representation.
    """

    def __init__(self, d_model):
        super().__init__()
        self.time = nn.Linear(3, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, query, atom_embedding, atom_ids, relative_beats, mask):
        return self.read(query, atom_embedding(atom_ids), atom_ids, relative_beats, mask)

    def read(self, query, embedded_atoms, atom_ids, relative_beats, mask):
        """Read shared [B,M,S] or candidate-specific [B,C,M,S] archive IDs."""
        if atom_ids.shape[-2] == 0:
            return torch.zeros_like(query)
        if atom_ids.ndim == 4:
            if query.ndim != 3 or query.shape[:2] != atom_ids.shape[:2]:
                raise ValueError('Candidate motif memory must match query [B,C,D]')
            batch, count = query.shape[:2]
            return self.read(query.reshape(batch*count,-1),
                embedded_atoms.flatten(0,1), atom_ids.flatten(0,1),
                relative_beats.flatten(0,1), mask.flatten(0,1)).reshape_as(query)
        valid_atoms = atom_ids.ne(0)
        memory = (embedded_atoms * valid_atoms[..., None]).sum(-2)
        memory = memory / valid_atoms.sum(-1).clamp_min(1)[..., None]
        relative = relative_beats.to(memory)
        timing = torch.stack((torch.tanh(relative/16), torch.sin(relative*math.pi/2),
                              torch.cos(relative*math.pi/2)), -1)
        memory = memory + self.time(timing)
        valid = mask.bool() & valid_atoms.any(-1)
        expanded = query[:, None] if query.ndim == 2 else query
        logits = torch.matmul(expanded.float(), self.key(memory).float().transpose(-1,-2)) / math.sqrt(query.shape[-1])
        logits = logits.masked_fill(~valid[:, None], -1e9)
        weights = logits.softmax(-1) * valid[:, None]
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-9)
        context = torch.matmul(weights.to(memory.dtype), self.value(memory))
        result = torch.tanh(self.output(context)).to(query)
        return result[:, 0] if query.ndim == 2 else result
