"""Exact event timing without the legacy integer-tick restriction."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping

from maichart.duration import duration_to_ticks_or_none
from maichart.generation.simai_compile import TempoChange
from maichart.maidata import (RawMaidataChart, parse_maidata_file,
                              parse_maidata_metadata, resolve_chart_first)
from maichart.timing import TimingDirective, _parse_point_directives, _split_top_level_commas, normalize_note_text

# Keep the tempo contract shared with the frame compiler.
EventTempo = TempoChange


def _exact_beat(value: Fraction | int | str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (Fraction, int, str)):
        raise TypeError("Beats must be Fraction, integer, or rational string, not float.")
    return Fraction(value)


def _finite_number(value: float, name: str, *, positive: bool = False) -> float:
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0):
        condition = "finite and positive" if positive else "finite"
        raise ValueError(f"{name} must be {condition}.")
    return number


@dataclass(slots=True)
class EventPoint:
    """One nonempty timing point; '/' and '*' remain opaque and ordered."""

    beat: Fraction
    note_text: str

    def __post_init__(self) -> None:
        self.beat = _exact_beat(self.beat)
        self.note_text = "".join(self.note_text.split())

    def to_dict(self) -> dict[str, Any]:
        return {"beat": str(self.beat), "note_text": self.note_text}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EventPoint:
        return cls(_exact_beat(data["beat"]), data["note_text"])


@dataclass(slots=True)
class EventChart:
    """A tick-independent event document with an exact exclusive end beat."""

    points: list[EventPoint] = field(default_factory=list)
    tempo_changes: list[EventTempo] = field(default_factory=list)
    first: float = 0.0
    level: str = ""
    title: str = ""
    end_beat: Fraction = Fraction(0)
    artist: str = ""
    difficulty: int = 5
    # Optional explicit chart-wide offset in seconds. Parsed note-local waits
    # do not populate this field; callers may set it when constructing an IR.
    time_offset: float = 0.0

    @property
    def audio_first(self) -> float:
        """Audio timestamp of beat 0 including a negative chart offset."""

        return float(self.first) + float(self.time_offset)

    def to_dict(self) -> dict[str, Any]:
        """Serialize exact beats as rational strings, never floats."""

        return {
            "points": [point.to_dict() for point in self.points],
            "tempo_changes": [
                {"beat": str(change.beat), "bpm": change.bpm}
                for change in self.tempo_changes
            ],
            "first": self.first,
            "time_offset": self.time_offset,
            "level": self.level,
            "title": self.title,
            "end_beat": str(self.end_beat),
            "artist": self.artist,
            "difficulty": self.difficulty,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EventChart:
        return cls(
            points=[EventPoint.from_dict(point) for point in data.get("points", [])],
            tempo_changes=[
                EventTempo(_exact_beat(change["beat"]), float(change["bpm"]))
                for change in data.get("tempo_changes", [])
            ],
            first=float(data.get("first", 0)),
            time_offset=float(data.get("time_offset", 0)),
            level=str(data.get("level", "")),
            title=str(data.get("title", "")),
            end_beat=_exact_beat(data.get("end_beat", "0")),
            artist=str(data.get("artist", "")),
            difficulty=int(data.get("difficulty", 5)),
        )


EVENT_END_TOKEN = "<event_end>"


def _format_fraction(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def event_points_to_joint_tokens(
    points: Iterable[EventPoint],
    *,
    local_start: Fraction | int | str = Fraction(0),
) -> list[str]:
    """Encode events as ``AT`` plus one ``CHAR`` token per source character.

    AT is relative to the local training window start, not to the preceding
    event. This keeps uncommon rational positions exact and makes every note
    combination compositional instead of a single vocabulary item.
    """

    origin = _exact_beat(local_start)
    tokens: list[str] = []
    for point in points:
        event = EventPoint(point.beat, point.note_text)
        relative = event.beat - origin
        if relative < 0:
            raise ValueError("Event beat precedes local_start.")
        tokens.append(f"AT:{_format_fraction(relative)}")
        tokens.extend(f"CHAR:{char}" for char in event.note_text)
        tokens.append(EVENT_END_TOKEN)
    return tokens


def joint_tokens_to_event_points(
    tokens: Iterable[str],
    *,
    local_start: Fraction | int | str = Fraction(0),
) -> list[EventPoint]:
    """Decode the compositional joint event stream without note semantics."""

    origin = _exact_beat(local_start)
    decoded: list[EventPoint] = []
    current_beat: Fraction | None = None
    chars: list[str] = []
    for token in tokens:
        if token.startswith("AT:"):
            if current_beat is not None:
                raise ValueError("AT token encountered before <event_end>.")
            current_beat = origin + _exact_beat(token[3:])
            if current_beat < origin:
                raise ValueError("AT token precedes local_start.")
        elif token.startswith("CHAR:"):
            if current_beat is None:
                raise ValueError("CHAR token must follow AT token.")
            char = token[5:]
            if len(char) != 1:
                raise ValueError("Each CHAR token must contain exactly one character.")
            chars.append(char)
        elif token == EVENT_END_TOKEN:
            if current_beat is None or not chars:
                raise ValueError("<event_end> must close a nonempty event.")
            decoded.append(EventPoint(current_beat, "".join(chars)))
            current_beat = None
            chars = []
        else:
            raise ValueError(f"Unknown event training token {token!r}.")
    if current_beat is not None:
        raise ValueError("Event token stream is missing <event_end>.")
    return decoded


# Short aliases make this contract convenient for dataset code.
encode_event_points = event_points_to_joint_tokens
decode_event_tokens = joint_tokens_to_event_points


@dataclass(slots=True)
class EventTimingPoint:
    """Exact tokenizer output, including empty and terminal timing points."""

    index: int
    raw: str
    note_text: str
    beat: Fraction
    tick: int | None
    time_sec: float
    division: int
    bpm: float
    directives: list[TimingDirective] = field(default_factory=list)


# EventTiming is the public short name; the longer alias documents that one
# item still includes raw directives and the terminal E marker.
EventTiming = EventTimingPoint


def seconds_at_beat(
    beat: Fraction | int | str,
    tempo_changes: Iterable[EventTempo],
    first: float = 0.0,
) -> float:
    """Integrate an exact beat through a tempo map, including ``first``."""

    target = _exact_beat(beat)
    if target < 0:
        raise ValueError("beat must be nonnegative.")
    changes = sorted(
        (EventTempo(_exact_beat(change.beat), _finite_number(change.bpm, "BPM", positive=True))
         for change in tempo_changes),
        key=lambda change: change.beat,
    )
    if not changes or changes[0].beat != 0:
        raise ValueError("tempo_changes must explicitly include a BPM at beat 0.")
    if any(change.beat < 0 for change in changes):
        raise ValueError("tempo beats must be nonnegative.")
    elapsed = _finite_number(first, "first")
    previous = Fraction(0)
    current_bpm = changes[0].bpm
    for change in changes[1:]:
        if change.beat >= target:
            break
        elapsed += float(change.beat - previous) * 60.0 / current_bpm
        previous = change.beat
        current_bpm = change.bpm
    elapsed += float(target - previous) * 60.0 / current_bpm
    return elapsed


def tokenize_event_timing(
    raw_inote: str,
    initial_bpm: float = 120.0,
    initial_division: int = 4,
    *,
    first: float = 0.0,
) -> list[EventTimingPoint]:
    """Reuse raw timing syntax, advancing only with exact Fraction steps.

    The first E stops the document. Ticks are optional derived values and
    never constrain the grid. Seconds include first and integrate each BPM.
    """

    bpm = _finite_number(initial_bpm, "initial_bpm", positive=True)
    if isinstance(initial_division, bool) or not isinstance(initial_division, int) or initial_division <= 0:
        raise ValueError("initial_division must be a positive integer.")
    elapsed = Fraction(str(_finite_number(first, "first")))
    if not raw_inote.strip():
        return []

    beat = Fraction(0)
    division = initial_division
    points: list[EventTimingPoint] = []
    for index, raw_point in enumerate(_split_top_level_commas(raw_inote)):
        directives, note_text = _parse_point_directives(raw_point)
        for directive in directives:
            if directive.directive_type == "bpm":
                bpm = _finite_number(directive.value, "BPM", positive=True)
            elif directive.directive_type == "grid":
                division = int(directive.value)
        points.append(
            EventTimingPoint(
                index=index,
                raw=raw_point,
                note_text=normalize_note_text("".join(note_text.split())),
                beat=beat,
                tick=duration_to_ticks_or_none(beat),
                time_sec=float(elapsed),
                division=division,
                bpm=bpm,
                directives=directives,
            )
        )
        if any(directive.directive_type == "end" for directive in directives):
            break
        step = Fraction(4, division)
        beat += step
        elapsed += step * 60 / Fraction(str(bpm))
    return points


parse_event_timing = tokenize_event_timing


def parse_event_chart(
    maidata_path: str | Path,
    difficulty_index: int,
    *,
    strict_notes: bool = False,
) -> EventChart:
    """Read exact events, BPM directives, and the first E from Maidata.

    Whitespace is removed from raw note text, but note syntax is not rewritten.
    strict_notes rejects the entire document on any unsupported note token or
    repeated button head.  It defaults to False so that a few dubious cells do
    not discard an otherwise good chart; callers that need a hard guarantee (the
    training manifest) can still opt in.
    """

    return _parse_event_metadata(
        parse_maidata_file(maidata_path), difficulty_index, strict_notes=strict_notes
    )


def parse_event_chart_text(
    maidata_text: str,
    difficulty_index: int,
    *,
    strict_notes: bool = False,
) -> EventChart:
    """In-memory counterpart used for exact compiler roundtrip verification."""

    return _parse_event_metadata(
        parse_maidata_metadata(maidata_text), difficulty_index, strict_notes=strict_notes
    )


def _parse_event_metadata(
    chart: RawMaidataChart,
    difficulty_index: int,
    *,
    strict_notes: bool,
) -> EventChart:
    difficulty = next(
        (item for item in chart.difficulties if item.index == difficulty_index), None
    )
    if difficulty is None or difficulty.inote is None:
        raise ValueError(f"No inote for difficulty {difficulty_index} found.")
    try:
        initial_bpm = _finite_number(chart.wholebpm or 120, "wholebpm", positive=True)
    except ValueError:
        initial_bpm = 120.0
    first_raw = resolve_chart_first(chart, difficulty_index)
    first = _finite_number(first_raw or 0, "first")
    timing = tokenize_event_timing(difficulty.inote, initial_bpm, first=first)

    if strict_notes:
        # No event_quality dependency: callers may apply stricter quality rules.
        from maichart.notes import parse_timing_point_notes

        unknown: list[tuple[Fraction, str]] = []
        duplicates: list[tuple[Fraction, str, int]] = []
        for point in timing:
            parsed = parse_timing_point_notes(point)
            unknown.extend((point.beat, token.raw) for token in parsed.unknown_tokens)
            # Touch notes report a string position ("A4", "B7", "C") while taps
            # and slide heads report 1..8.  Both must be counted, otherwise a
            # repeated touch button is invisible here.
            heads = Counter(str(note.position) for note in parsed.notes)
            duplicates.extend(
                (point.beat, button, count)
                for button, count in heads.items()
                if count > 1
            )
        if unknown or duplicates:
            details = ", ".join(f"{beat}: {raw!r}" for beat, raw in unknown)
            details += "; " if details and duplicates else ""
            details += ", ".join(
                f"{beat}: duplicate head {button} x{count}"
                for beat, button, count in duplicates
            )
            raise ValueError(f"Unsupported or duplicate notes; rejecting entire chart: {details}")

    changes = [
        EventTempo(point.beat, float(directive.value))
        for point in timing
        for directive in point.directives
        if directive.directive_type == "bpm"
    ]
    if not changes or changes[0].beat != 0:
        changes.insert(0, EventTempo(Fraction(0), initial_bpm))

    end_beat = Fraction(0)
    if timing:
        last = timing[-1]
        end_beat = last.beat
        if not any(directive.directive_type == "end" for directive in last.directives):
            end_beat += Fraction(4, last.division)
    return EventChart(
        points=[EventPoint(point.beat, point.note_text) for point in timing if point.note_text],
        tempo_changes=changes,
        first=first,
        level=difficulty.level or "",
        title=chart.title or "",
        end_beat=end_beat,
        artist=chart.artist or "",
        difficulty=difficulty_index,
    )
