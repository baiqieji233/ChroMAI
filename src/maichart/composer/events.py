"""Structured event records and deterministic Simai configuration helpers."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable

from maichart.event_timing import EventPoint
from maichart.form.type_guidance import _counts
from maichart.generation.event_cli import _NoteGrammar


@dataclass(frozen=True, slots=True)
class ComposerEvent:
    """One committed event. ``note_text`` remains a full simultaneous group."""
    beat: Fraction
    note_text: str

    def point(self) -> EventPoint:
        return EventPoint(self.beat, self.note_text)

    @property
    def structure(self):
        counts = _counts(self.note_text)
        if counts is None:
            return (0, 0, 0, 0, 0)
        return tuple(int(value) for value in counts)


def atom_catalog(points: Iterable[EventPoint]) -> list[str]:
    values = set()
    for point in points:
        values.update(_NoteGrammar._split_slashes(str(point.note_text)))
    return sorted(values)


def event_delta_seconds(previous: ComposerEvent | None, current: ComposerEvent, tempo, first: float) -> float:
    from maichart.event_timing import seconds_at_beat
    before = Fraction(0) if previous is None else previous.beat
    return float(seconds_at_beat(current.beat, tempo, first) - seconds_at_beat(before, tempo, first))


def structure_signature(note_text: str) -> tuple:
    counts = _counts(str(note_text))
    if counts is None:
        return ("unknown",)
    text = str(note_text)
    return (tuple(int(value) for value in counts), bool('/' in text),
            bool('b' in text), bool('x' in text),
            bool(any(char in text for char in '-<>^pqszvwV')),
            len(text))
