"""Finite alphabet for exact rational positions; independent of legacy AT vocab."""
from __future__ import annotations

from fractions import Fraction
from typing import Iterable

from maichart.event_timing import EventPoint
from maichart.generation.event_model import NOTE_CHARACTERS

SCHEMA = "rational-events-v1"
SPECIALS = ("<pad>", "<bos>", "<eos>", "<at>", "<note>", "<end>", "<mask>")


class RationalCodec:
    def __init__(self):
        self.tokens = (*SPECIALS, *("P:" + c for c in "0123456789/"),
                       *("N:" + c for c in dict.fromkeys(NOTE_CHARACTERS)))
        self.ids = {token: i for i, token in enumerate(self.tokens)}
        self.pad_id, self.bos_id, self.eos_id = 0, 1, 2
        self.at_id, self.note_id, self.end_id, self.mask_id = 3, 4, 5, 6

    def __len__(self):
        return len(self.tokens)

    def position(self, beat):
        beat = Fraction(beat)
        if beat < 0:
            raise ValueError("Relative beat must be nonnegative")
        return [self.at_id, *(self.ids['P:' + c] for c in str(beat)), self.note_id]

    def cell(self, point, *, origin=Fraction(0)):
        text = point.note_text
        body = [self.mask_id] if text is None else [self.ids['N:' + c] for c in text]
        if not body:
            raise ValueError("Empty cell")
        return self.position(point.beat - origin) + body + [self.end_id]

    def encode(self, points: Iterable, *, origin=Fraction(0)):
        values = list(points)
        if any(b.beat <= a.beat for a, b in zip(values, values[1:])):
            raise ValueError("Event positions must be strictly increasing")
        return [self.bos_id] + [t for p in values for t in self.cell(p, origin=origin)] + [self.eos_id]

    def bounded_memory(self, points, *, origin=Fraction(0), max_tokens=512, newest=True):
        kept, used = [], 2
        values = (reversed(points) if hasattr(points, '__reversed__') else reversed(list(points))) if newest else iter(points)
        for point in values:
            cell = self.cell(point, origin=origin)
            if used + len(cell) > max_tokens:
                break
            kept.append(cell); used += len(cell)
        if newest:
            kept.reverse()
        return [self.bos_id] + [t for c in kept for t in c] + [self.eos_id] if kept else [0]

    def decode(self, tokens, *, origin=Fraction(0)):
        points, digits, chars, state = [], "", "", "start"
        closed = False
        for token_id in tokens:
            token = self.tokens[int(token_id)]
            if token in ("<bos>", "<pad>") and state == "start":
                continue
            if token == "<eos>" and state == "start":
                closed = True; break
            if token == "<at>" and state == "start":
                state, digits = "position", ""; continue
            if token.startswith('P:') and state == 'position':
                digits += token[2:]; continue
            if token == '<note>' and state == 'position':
                beat = Fraction(digits)
                if str(beat) != digits or beat < 0:
                    raise ValueError("Noncanonical rational position")
                state, chars = 'note', ''; continue
            if token.startswith('N:') and state == 'note':
                chars += token[2:]; continue
            if token == '<end>' and state == 'note' and chars:
                if points and points[-1].beat >= origin + beat:
                    raise ValueError("Non-monotone positions")
                points.append(EventPoint(origin + beat, chars)); state = 'start'; continue
            raise ValueError(f"Invalid rational event token {token} in {state}")
        if state != 'start' or not closed:
            raise ValueError("Incomplete event sequence")
        return points

    def to_dict(self):
        return {"schema": SCHEMA, "tokens": list(self.tokens)}

    @classmethod
    def from_dict(cls, data):
        codec = cls()
        if data != codec.to_dict():
            raise ValueError("Unsupported rational codec contract")
        return codec
