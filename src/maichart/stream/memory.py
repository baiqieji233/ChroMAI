"""Bounded recent/archive memory for continuous chart composition."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import torch


@dataclass(frozen=True)
class MemoryEntry:
    beat: Fraction
    note_text: str
    vector: torch.Tensor


class StreamMemoryBank:
    """Acoustic retrieval evidence; it never forces copying a motif."""
    def __init__(self, *, max_entries=256, max_retrieval=8):
        if isinstance(max_entries, bool) or not 1 <= int(max_entries) <= 4096:
            raise ValueError('max_entries must be in [1, 4096]')
        if isinstance(max_retrieval, bool) or not 1 <= int(max_retrieval) <= 4096:
            raise ValueError('max_retrieval must be in [1, 4096]')
        self.max_entries, self.max_retrieval = int(max_entries), int(max_retrieval)
        self.entries: list[MemoryEntry] = []

    def append(self, beat, note_text, vector):
        value = torch.as_tensor(vector).detach().float().reshape(-1).cpu()
        if value.numel() < 1 or not torch.isfinite(value).all():
            raise ValueError('memory vector must be finite and nonempty')
        self.entries.append(MemoryEntry(Fraction(beat), str(note_text), value))
        if len(self.entries) > self.max_entries:
            del self.entries[:len(self.entries) - self.max_entries]

    def retrieve(self, query, *, top_k=None, current_beat=None, min_similarity=0.):
        if not self.entries:
            return []
        query = torch.as_tensor(query).detach().float().reshape(-1).cpu()
        if query.numel() != self.entries[0].vector.numel() or not torch.isfinite(query).all():
            raise ValueError('memory query shape or values differ')
        top_k = self.max_retrieval if top_k is None else int(top_k)
        if not 1 <= top_k <= self.max_retrieval:
            raise ValueError('top_k exceeds memory retrieval budget')
        q = query / query.norm().clamp_min(1e-8)
        scored=[]
        for index, entry in enumerate(self.entries):
            score=float(torch.dot(q, entry.vector/entry.vector.norm().clamp_min(1e-8)))
            if current_beat is not None:
                score += 1e-5/(1.+abs(float(Fraction(current_beat)-entry.beat)))
            if score >= float(min_similarity): scored.append((score,index))
        scored.sort(reverse=True)
        return [self.entries[index] for _,index in scored[:top_k]]

    def tensor(self, query, *, top_k=None, current_beat=None, min_similarity=0.):
        rows=self.retrieve(query,top_k=top_k,current_beat=current_beat,min_similarity=min_similarity)
        return torch.stack([entry.vector for entry in rows],0) if rows else None
