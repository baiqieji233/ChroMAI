"""Structured event records and deterministic Simai configuration helpers."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from fractions import Fraction
from typing import Iterable

from maichart.event_timing import EventPoint
from maichart.form.type_guidance import _counts
from maichart.generation.event_cli import _NoteGrammar
from maichart.notes import parse_timing_point_notes
from maichart.timing import RawTimingPoint


EVENT_SEMANTICS = 'parsed_heads_and_types_v2'
HEAD_PADS = tuple(f'{group}{lane}' for group in 'ABDE' for lane in range(1,9)) + ('C',)
HEAD_TIMING_DIM = 3 * len(HEAD_PADS)
_PAD_INDEX = {pad:i for i,pad in enumerate(HEAD_PADS)}


@dataclass(frozen=True, slots=True)
class EventMetadata:
    atoms: tuple[str, ...]
    heads: tuple[str, ...]
    head_pad_indices: tuple[int, ...]
    hold: bool
    slide: bool
    touch: bool
    break_ex: bool
    type_counts: tuple[int, ...]

    @property
    def chord(self):
        return len(self.heads) > 1

    @property
    def structure(self):
        return self.type_counts, self.chord, self.hold, self.slide, self.touch, self.break_ex


@lru_cache(maxsize=65536)
def event_metadata(note_text: str) -> EventMetadata:
    """One parser-backed definition for heads, types and symbolic history."""
    parsed = parse_timing_point_notes(RawTimingPoint(index=0, raw=note_text, note_text=note_text,
        beat=Fraction(0), tick=0, time_sec=0., division=4, bpm=120.))
    if parsed.unknown_tokens or not parsed.notes:
        raise ValueError(f'Unsupported or empty Composer event: {note_text!r}')
    heads=tuple(sorted({str(n.position) for n in parsed.notes}))
    kinds={n.note_type for n in parsed.notes}
    counts=tuple(sum(n.note_type==kind for n in parsed.notes)
                 for kind in ('tap','slide','hold','touch','touch_hold'))
    pads=tuple(sorted({_PAD_INDEX[f'A{h}' if h.isdigit() else h] for h in heads
                       if (f'A{h}' if h.isdigit() else h) in _PAD_INDEX}))
    return EventMetadata(tuple(_NoteGrammar._split_slashes(note_text)),heads,pads,
        bool(kinds & {'hold','touch_hold'}),'slide' in kinds,
        bool(kinds & {'touch','touch_hold'}),
        any(n.is_break or n.modifiers.get('ex',False) for n in parsed.notes),counts)


def history_event_features(note_text, delta_beats, delta_seconds, *, parsed=True):
    if parsed:
        meta=event_metadata(str(note_text))
        content=(min(1.,len(meta.heads)/4.),float(meta.chord),float(meta.hold),
                 float(meta.slide),float(meta.touch),float(meta.break_ex))
    else:
        atoms=_NoteGrammar._split_slashes(str(note_text))
        content=(min(1.,len(atoms)/4.),float('/' in note_text),float('h' in note_text),
            float(any(c in note_text for c in '-<>^pqszvwV')),
            float(any(c in note_text for c in 'ABCDE')),float(any(c in note_text for c in 'bx')))
    return (min(1.,max(0.,float(delta_beats)/4.)), min(1.,max(0.,float(delta_seconds)/2.)), *content)


def head_timing_features(last_seconds, last_beats, seconds, beats):
    """Causal per-pad ages; no universal jack-speed limit is encoded here."""
    import numpy as np
    last_seconds=np.asarray(last_seconds,dtype=np.float64)
    last_beats=np.asarray(last_beats,dtype=np.float64)
    seconds=np.asarray(seconds,dtype=np.float64).reshape(-1)
    beats=np.asarray(beats,dtype=np.float64).reshape(-1)
    seen=np.isfinite(last_seconds)
    ages_sec=np.maximum(0.,seconds[:,None]-np.where(seen,last_seconds,0.))
    ages_beat=np.maximum(0.,beats[:,None]-np.where(seen,last_beats,0.))
    return np.concatenate((np.broadcast_to(seen,(len(seconds),len(seen))),
        np.log1p(ages_sec)*seen,np.log1p(ages_beat)*seen),axis=-1).astype(np.float32)


def enrich_head_timing(head_timing):
    """Map [seen, log1p(age_sec), log1p(age_beat)] to bounded short/beat/long recency in [0, 1]."""
    import torch
    raw = head_timing.float()
    pads = len(HEAD_PADS)
    seen = raw[..., :pads].clamp(0., 1.)
    log_sec = raw[..., pads:2*pads].clamp(0., 10.)
    log_beat = raw[..., 2*pads:3*pads].clamp(0., 10.)
    age_sec = torch.expm1(log_sec)
    age_beat = torch.expm1(log_beat)
    fast_sec = torch.exp(-age_sec / .08) * seen
    fast_beat = torch.exp(-age_beat / .35) * seen
    bounded_age = (log_sec / 2.2).clamp(0., 1.) * seen
    return torch.cat((fast_sec, fast_beat, bounded_age), dim=-1).to(dtype=head_timing.dtype)


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
