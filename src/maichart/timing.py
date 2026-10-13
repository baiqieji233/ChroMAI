"""Timing-only tokenizer for raw Maidata ``inote_x`` blocks."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from maichart.maidata import RawDifficultyBlock, RawMaidataChart, resolve_chart_first

TICKS_PER_BEAT = 1920


@dataclass(slots=True)
class TimingDirective:
    """A timing directive found at the start of a raw timing point."""

    directive_type: str
    raw: str
    value: float | int | str | None = None


@dataclass(slots=True)
class RawTimingPoint:
    """One comma-delimited timing point with opaque note text."""

    index: int
    raw: str
    note_text: str
    beat: Fraction
    tick: int
    time_sec: float | None
    division: int
    bpm: float
    directives: list[TimingDirective] = field(default_factory=list)


_PSEUDO_CHORD_SEPARATOR = "`"
_PSEUDO_CHORD_SEPARATOR_NORMALIZED = "/"
# Touch note areas (A-E) as defined by the note grammar.
TOUCH_AREAS = "ABCDE"


def _collapse_repeated_touch_area(text: str) -> str:
    """``EE`` is a typo for one ``E`` touch: collapse repeated identical areas.

    Only a whole ``/``-separated token made of one repeated area letter is
    touched, so ordinary chords (``36``), positioned touches (``E8``) and
    multi-area groups (``E8/B8``) are left alone.
    """

    if not text:
        return text
    parts = text.split(_PSEUDO_CHORD_SEPARATOR_NORMALIZED)
    collapsed: list[str] = []
    for part in parts:
        if len(part) > 1 and part[0] in TOUCH_AREAS and all(char == part[0] for char in part):
            collapsed.append(part[0])
        else:
            collapsed.append(part)
    return "/".join(collapsed)


def normalize_note_text(text: str) -> str:
    """Canonicalise the shorthand forms real hand-made packs use.

    Applied once, where raw cell text becomes note text, so event charts, V1
    notes, the quality checker and the training vocabulary all see one spelling:

    * ``1`2`` -- the "pseudo double press": the charter writes a backtick where
      a simultaneity separator belongs, and ``1/2`` is exactly equivalent, so
      the backtick is rewritten to ``/``;
    * ``EE`` -- a repeated identical touch area is a typo for a single ``E``.

    Anything else (``?``, ``!``, ``$``, a mid-cell ``{division}``, an unknown
    timing pair) is deliberately left untouched and stays rejected.
    """

    return _collapse_repeated_touch_area(text.replace(_PSEUDO_CHORD_SEPARATOR, "/"))


def tokenize_inote_timing(
    raw_inote: str,
    initial_bpm: float = 120.0,
    initial_division: int = 4,
    first: float = 0.0,
) -> list[RawTimingPoint]:
    """Tokenize raw ``inote_x`` text into timing points.

    This recognizes only timing structure: comma-separated points, grid
    directives, BPM directives, and the ``E`` end marker. Note text remains
    opaque.
    """
    if initial_division <= 0:
        raise ValueError("initial_division must be a positive integer.")
    if initial_bpm <= 0:
        raise ValueError("initial_bpm must be positive.")
    first = float(first)
    if not math.isfinite(first):
        raise ValueError("first must be finite.")

    points: list[RawTimingPoint] = []
    beat = Fraction(0, 1)
    time_sec = first
    bpm = float(initial_bpm)
    division = int(initial_division)

    for index, raw_point in enumerate(_split_top_level_commas(raw_inote)):
        directives, note_text = _parse_point_directives(raw_point)
        point_bpm = bpm
        point_division = division

        for directive in directives:
            if directive.directive_type == "bpm":
                point_bpm = float(directive.value)
            elif directive.directive_type == "grid":
                point_division = int(directive.value)

        points.append(
            RawTimingPoint(
                index=index,
                raw=raw_point,
                note_text=normalize_note_text(note_text),
                beat=beat,
                tick=_beat_to_tick(beat),
                time_sec=time_sec,
                division=point_division,
                bpm=point_bpm,
                directives=directives,
            )
        )

        step_beats = Fraction(4, point_division)
        time_sec += float(step_beats) * 60.0 / point_bpm
        beat += step_beats
        bpm = point_bpm
        division = point_division

    return points


def tokenize_difficulty_timing(
    difficulty: RawDifficultyBlock,
    initial_bpm: float = 120.0,
    initial_division: int = 4,
    first: float = 0.0,
) -> list[RawTimingPoint]:
    """Tokenize the raw ``inote`` field from a difficulty block."""

    if difficulty.inote is None:
        return []
    return tokenize_inote_timing(
        difficulty.inote,
        initial_bpm=initial_bpm,
        initial_division=initial_division,
        first=first,
    )


def tokenize_chart_difficulty_timing(
    chart: RawMaidataChart,
    difficulty_index: int,
    initial_bpm: float = 120.0,
    initial_division: int = 4,
) -> list[RawTimingPoint]:
    """Tokenize one chart slot with its resolved audio-time ``first`` anchor."""
    difficulty = next((item for item in chart.difficulties
                       if item.index == int(difficulty_index)), None)
    if difficulty is None:
        raise ValueError(f"No difficulty {difficulty_index} found.")
    raw_first = resolve_chart_first(chart, difficulty_index)
    try:
        first = float(raw_first) if raw_first is not None else 0.0
    except (TypeError, ValueError) as exc:
        raise ValueError("first must be finite.") from exc
    if not math.isfinite(first):
        raise ValueError("first must be finite.")
    return tokenize_difficulty_timing(
        difficulty, initial_bpm=initial_bpm,
        initial_division=initial_division, first=first,
    )


def timing_point_to_dict(point: RawTimingPoint) -> dict[str, Any]:
    """Convert a raw timing point to JSON-compatible primitives."""

    return {
        "index": point.index,
        "raw": point.raw,
        "note_text": point.note_text,
        "beat": _format_fraction(point.beat),
        "tick": point.tick,
        "time_sec": point.time_sec,
        "division": point.division,
        "bpm": point.bpm,
        "directives": [
            {
                "directive_type": directive.directive_type,
                "raw": directive.raw,
                "value": directive.value,
            }
            for directive in point.directives
        ],
    }


def timing_points_to_json(points: list[RawTimingPoint], *, indent: int = 2) -> str:
    """Serialize raw timing points to JSON."""

    return json.dumps(
        [timing_point_to_dict(point) for point in points],
        ensure_ascii=False,
        indent=indent,
    )


def _split_top_level_commas(raw_inote: str) -> list[str]:
    points: list[str] = []
    current: list[str] = []
    square_depth = 0

    for char in raw_inote:
        if char == "[":
            square_depth += 1
        elif char == "]" and square_depth > 0:
            square_depth -= 1

        if char == "," and square_depth == 0:
            points.append("".join(current))
            current = []
        else:
            current.append(char)

    points.append("".join(current))
    return points


def _parse_point_directives(raw_point: str) -> tuple[list[TimingDirective], str]:
    text = raw_point.strip()
    directives: list[TimingDirective] = []
    cursor = 0

    while cursor < len(text):
        cursor = _skip_whitespace(text, cursor)
        if cursor >= len(text):
            break

        char = text[cursor]
        if char == "{":
            end = text.find("}", cursor + 1)
            if end == -1:
                raise ValueError(f"Malformed grid directive in timing point: {raw_point!r}")
            raw = text[cursor : end + 1]
            value_text = text[cursor + 1 : end].strip()
            try:
                value = int(value_text)
            except ValueError as exc:
                raise ValueError(f"Invalid grid division {raw!r}.") from exc
            if value <= 0:
                raise ValueError(f"Grid division must be positive in {raw!r}.")
            directives.append(TimingDirective("grid", raw, value))
            cursor = end + 1
            continue

        if char == "(":
            end = text.find(")", cursor + 1)
            if end == -1:
                raise ValueError(f"Malformed BPM directive in timing point: {raw_point!r}")
            raw = text[cursor : end + 1]
            value_text = text[cursor + 1 : end].strip()
            try:
                value = float(value_text)
            except ValueError as exc:
                raise ValueError(f"Invalid BPM value {raw!r}.") from exc
            if value <= 0:
                raise ValueError(f"BPM must be positive in {raw!r}.")
            directives.append(TimingDirective("bpm", raw, value))
            cursor = end + 1
            continue

        if text.startswith("||", cursor):
            match = re.match(r"\|\|\s*(\d+)\s*/\s*(\d+)", text[cursor:])
            if match is None:
                raise ValueError(f"Malformed meter directive in timing point: {raw_point!r}")
            raw = match.group(0)
            directives.append(TimingDirective("meter", raw, f"{match.group(1)}/{match.group(2)}"))
            cursor += len(raw)
            continue

        if char == "E" and text[cursor + 1 :].strip() == "":
            directives.append(TimingDirective("end", "E", "E"))
            cursor = len(text)
            break

        break

    note_text = text[cursor:].strip()
    return directives, note_text


def _skip_whitespace(text: str, cursor: int) -> int:
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    return cursor


def _beat_to_tick(beat: Fraction) -> int:
    tick = beat * TICKS_PER_BEAT
    if tick.denominator != 1:
        raise ValueError(f"Beat {beat} does not map to an integer tick.")
    return tick.numerator


def _format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"
