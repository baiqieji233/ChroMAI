"""Basic tap, break, hold, and slide parser for raw timing points."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from maichart.duration import (
    DurationExpr,
    duration_expr_to_dict,
    duration_to_ticks_or_none,
    resolve_duration_expr,
    parse_duration_expr,
)
from maichart.ir import Note
from maichart.slides import get_slide_pattern_definition, is_supported_slide_pattern
from maichart.timing import RawTimingPoint

BASIC_TAP_RE = re.compile(r"^(?P<position>[1-8])(?P<modifiers>[bx]*)$")
# A hold's duration block is optional: hand-made packs write `7h` for a hold that
# starts and ends at the same instant, which is exactly the zero-length form
# `7h[1:0]` (see ZERO_HOLD_DURATION below).  Modifiers are order-free: `3xh`,
# `2hx` and `4h[4:1]x` are the same note (slides have always allowed a trailing
# modifier, so holds accept one too).
BASIC_HOLD_RE = re.compile(
    r"^(?P<position>[1-8])(?P<before_modifiers>[bx]*)h(?P<after_modifiers>[bx]*)"
    r"(?P<duration>\[[^\]]+\])?(?P<tail_modifiers>[bx]*)$"
)
BASIC_SLIDE_RE = re.compile(
    r"^(?P<position>[1-8])(?P<head_modifiers>[bx]*)(?P<pattern>pp|qq|[-<>^pqszvwV])(?P<end_position>[1-8]{1,2})"
    r"(?P<before_duration_modifiers>[bx]*)(?P<duration>\[[^\]]+\])(?P<tail_modifiers>[bx]*)$"
)
CONTINUED_SLIDE_RE = re.compile(
    r"^(?P<pattern>pp|qq|[-<>^pqszvwV])(?P<end_position>[1-8]{1,2})"
    r"(?P<before_duration_modifiers>[bx]*)(?P<duration>\[[^\]]+\])(?P<tail_modifiers>[bx]*)$"
)
COMPOUND_SLIDE_RE = re.compile(
    r"^(?P<position>[1-8])(?P<head_modifiers>[bx]*)(?P<path_text>[1-8<>^pqszvwV-]+)"
    r"(?P<before_duration_modifiers>[bx]*)(?P<duration>\[[^\]]+\])(?P<tail_modifiers>[bx]*)$"
)
CONTINUED_COMPOUND_SLIDE_RE = re.compile(
    r"^(?P<path_text>[1-8<>^pqszvwV-]+)(?P<before_duration_modifiers>[bx]*)(?P<duration>\[[^\]]+\])(?P<tail_modifiers>[bx]*)$"
)
BASIC_COMPACT_CHORD_RE = re.compile(r"^(?P<positions>[1-8]{2,8})(?P<modifiers>[bx]*)$")
BASIC_TOUCH_RE = re.compile(
    r"^(?P<area>[ABCDE])(?P<position>[1-8]?)(?P<modifiers>[bf]*)$"
)
# Touch holds: the firework (`f`) and break (`b`) markers are order-free and may
# sit before `h`, after it, or after the duration block (`Cfh[..]` == `Chf[..]`
# == `Ch[..]f`), and the duration block itself is optional.
BASIC_TOUCH_HOLD_RE = re.compile(
    r"^(?P<area>[ABCDE])(?P<position>[1-8]?)(?P<before_modifiers>[bf]*)h"
    r"(?P<after_modifiers>[bf]*)(?P<duration>\[[^\]]+\])?(?P<tail_modifiers>[bf]*)$"
)
# The canonical spelling of a hold without a duration block: `[1:0]` resolves to
# zero beats, so `7h` and `7h[1:0]` are the same note.
ZERO_HOLD_DURATION = ZERO_HOLD_DURATION_RAW = "[1:0]"


@dataclass(slots=True)
class ParsedNote:
    """A V1 parsed note."""

    note_type: str
    position: int | str
    is_break: bool
    raw: str
    time_sec: float | None = None
    beat: Fraction = Fraction(0, 1)
    tick: int = 0
    group_id: str | None = None
    duration_beats: Fraction | None = None
    duration_ticks: int | None = None
    duration_sec: float | None = None
    end_position: int | str | None = None
    slide_pattern: str | None = None
    slide_segments: list[dict[str, Any]] = field(default_factory=list)
    duration: DurationExpr | None = None
    segments: list[dict[str, Any]] = field(default_factory=list)
    modifiers: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class UnknownNoteToken:
    """A raw note token intentionally left unparsed."""

    raw: str
    reason: str


@dataclass(slots=True)
class ParsedTimingPointNotes:
    """Parsed and unknown note tokens for one raw timing point."""

    timing_index: int
    raw: str
    note_text: str
    beat: Fraction
    tick: int
    time_sec: float | None
    notes: list[ParsedNote] = field(default_factory=list)
    unknown_tokens: list[UnknownNoteToken] = field(default_factory=list)
    recognized_token_count: int = 0


def parse_basic_tap_break_token(token: str) -> ParsedNote | None:
    """Parse one complete basic tap or break token.

    The token must fully match ``^[1-8]b?$``. Complex note syntax is not
    partially parsed.
    """

    candidate = re.sub(r"\s+", "", token.strip())
    match = BASIC_TAP_RE.fullmatch(candidate)
    if match is None:
        return None

    return ParsedNote(
        note_type="tap",
        position=int(match.group("position")),
        is_break="b" in match.group("modifiers"),
        raw=token,
        modifiers=_parse_note_modifier_text(match.group("modifiers")),
    )


def parse_basic_hold_token(token: str) -> ParsedNote | None:
    """Parse one complete basic hold token.

    Supported hold syntax is ``^[1-8][bx]*h[bx]*(\\[..\\])?$``: the duration
    block may be omitted, in which case the hold is the zero-length ``[1:0]``
    form (a hold that starts and ends at the same instant).  Duration is
    measured with the Maidata-style measure fraction where ``[4:1]`` is one beat
    and ``[8:3]`` is one and a half beats.
    """

    candidate = re.sub(r"\s+", "", token.strip())
    match = BASIC_HOLD_RE.fullmatch(candidate)
    if match is None:
        return None

    duration = parse_duration_expr(match.group("duration") or ZERO_HOLD_DURATION)
    if duration.kind == "unknown":
        return None
    modifiers = _parse_note_modifier_text(
        match.group("before_modifiers") + match.group("after_modifiers")
        + match.group("tail_modifiers")
    )
    return ParsedNote(
        note_type="hold",
        position=int(match.group("position")),
        is_break=bool(modifiers["break"]),
        raw=token,
        duration_beats=duration.beats,
        duration_ticks=duration.ticks,
        duration_sec=duration.seconds,
        duration=duration,
        modifiers=modifiers,
    )


def parse_basic_slide_token(token: str) -> ParsedNote | None:
    """Parse one slide head and its trajectory branches.

    Supported slide syntax is one start button, optional break marker, one or
    more path segments, and duration blocks, such as ``1-4[8:1]``,
    ``3b-1[8:1]``, ``7pp1[8:1]``, or ``1-4[8:1]*-6[8:1]``.
    """

    candidate = re.sub(r"\s+", "", token.strip())
    raw_segments = _split_top_level_stars(candidate)
    if not raw_segments:
        return None

    parsed_segments: list[dict[str, Any]] = []
    for index, raw_segment in enumerate(raw_segments):
        regex = BASIC_SLIDE_RE if index == 0 else CONTINUED_SLIDE_RE
        match = regex.fullmatch(raw_segment)
        if match is None:
            compound_regex = COMPOUND_SLIDE_RE if index == 0 else CONTINUED_COMPOUND_SLIDE_RE
            compound_match = compound_regex.fullmatch(raw_segment)
            if compound_match is None:
                return None
            try:
                parsed_segments.append(
                    _compound_slide_segment_from_match(compound_match, raw_segment)
                )
            except ValueError:
                return None
            continue
        try:
            parsed_segments.append(_slide_segment_from_match(match, raw_segment))
        except ValueError:
            return None

    _attach_slide_trajectory_refs(parsed_segments)
    first_segment = parsed_segments[0]
    duration = _combine_segment_durations(parsed_segments, token)
    return ParsedNote(
        note_type="slide",
        position=int(first_segment["position"]),
        is_break=bool(_slide_note_modifiers(parsed_segments)["break"]),
        raw=token,
        duration_beats=duration.beats,
        duration_ticks=duration.ticks,
        duration_sec=duration.seconds,
        duration=duration,
        end_position=first_segment["end_position"],
        slide_pattern=str(first_segment.get("pattern", "compound")),
        slide_segments=parsed_segments,
        segments=parsed_segments,
        modifiers=_slide_note_modifiers(parsed_segments),
    )


def parse_basic_touch_token(token: str) -> ParsedNote | None:
    """Parse one complete touch tap or touch hold token."""

    candidate = re.sub(r"\s+", "", token.strip())

    hold_match = BASIC_TOUCH_HOLD_RE.fullmatch(candidate)
    if hold_match is not None:
        duration = parse_duration_expr(hold_match.group("duration") or ZERO_HOLD_DURATION)
        if duration.kind == "unknown":
            return None
        position = _touch_position(
            hold_match.group("area"),
            hold_match.group("position"),
        )
        marker_text = (hold_match.group("before_modifiers")
                       + hold_match.group("after_modifiers")
                       + hold_match.group("tail_modifiers"))
        return ParsedNote(
            note_type="touch_hold",
            position=position,
            is_break="b" in marker_text,
            raw=token,
            duration_beats=duration.beats,
            duration_ticks=duration.ticks,
            duration_sec=duration.seconds,
            duration=duration,
            modifiers={
                "touch_area": hold_match.group("area"),
                "firework": "f" in marker_text,
            },
        )

    tap_match = BASIC_TOUCH_RE.fullmatch(candidate)
    if tap_match is None:
        return None
    position = _touch_position(tap_match.group("area"), tap_match.group("position"))
    return ParsedNote(
        note_type="touch",
        position=position,
        is_break="b" in tap_match.group("modifiers"),
        raw=token,
        modifiers={
            "touch_area": tap_match.group("area"),
            "firework": "f" in tap_match.group("modifiers"),
        },
    )


def parse_compact_chord_token(token: str) -> list[ParsedNote] | None:
    """Parse compact simultaneous button syntax such as ``34`` or ``57``."""

    candidate = re.sub(r"\s+", "", token.strip())
    match = BASIC_COMPACT_CHORD_RE.fullmatch(candidate)
    if match is None:
        return None
    positions = match.group("positions")
    if len(set(positions)) != len(positions):
        return None
    modifiers = _parse_note_modifier_text(match.group("modifiers"))
    return [
        ParsedNote(
            note_type="tap",
            position=int(position),
            is_break=bool(modifiers["break"]),
            raw=token,
            modifiers=dict(modifiers),
        )
        for position in positions
    ]


def parse_timing_point_notes(point: RawTimingPoint) -> ParsedTimingPointNotes:
    """Parse basic tap, break, hold, and slide notes from one timing point."""

    parsed = ParsedTimingPointNotes(
        timing_index=point.index,
        raw=point.raw,
        note_text=point.note_text,
        beat=point.beat,
        tick=point.tick,
        time_sec=point.time_sec,
    )

    if point.note_text == "":
        return parsed

    tokens = _split_top_level_slashes(point.note_text)
    for token in tokens:
        if token.strip() == "":
            parsed.unknown_tokens.append(UnknownNoteToken(token, "empty token"))
            continue

        parsed_notes: list[ParsedNote] = []
        note = parse_basic_tap_break_token(token)
        if note is not None:
            parsed_notes = [note]
        if not parsed_notes:
            note = parse_basic_hold_token(token)
            if note is not None:
                parsed_notes = [note]
        if not parsed_notes:
            note = parse_basic_slide_token(token)
            if note is not None:
                parsed_notes = [note]
        if not parsed_notes:
            note = parse_basic_touch_token(token)
            if note is not None:
                parsed_notes = [note]
        if not parsed_notes:
            parsed_notes = parse_compact_chord_token(token) or []
        if not parsed_notes:
            parsed.unknown_tokens.append(
                UnknownNoteToken(token, "unsupported note syntax")
            )
            continue

        parsed.recognized_token_count += 1
        for parsed_note in parsed_notes:
            parsed_note.beat = point.beat
            parsed_note.tick = point.tick
            parsed_note.time_sec = point.time_sec
            _apply_duration_context(parsed_note, point.bpm)
            parsed.notes.append(parsed_note)

    if len(parsed.notes) > 1:
        group_id = f"timing-{point.index}"
        for note in parsed.notes:
            note.group_id = group_id

    return parsed


def parse_timing_points_notes(points: list[RawTimingPoint]) -> list[ParsedTimingPointNotes]:
    """Parse basic tap, break, hold, and slide notes from timing points."""

    return [parse_timing_point_notes(point) for point in points]


def parsed_note_to_chart_note(note: ParsedNote) -> Note:
    """Convert a parsed tap, break, or hold note to the current ChartIR shape."""

    modifiers: dict[str, Any] = {"break": note.is_break}
    modifiers.update(note.modifiers)
    if note.group_id is not None:
        modifiers["group_id"] = note.group_id
    if note.slide_pattern is not None:
        modifiers["slide_pattern"] = note.slide_pattern
    if note.slide_segments:
        legacy_segments = [_json_safe_slide_segment(segment) for segment in note.slide_segments]
        modifiers["slide_segments"] = legacy_segments

    return Note(
        note_type=note.note_type,
        position=str(note.position),
        end_position=(
            str(note.end_position) if note.end_position is not None else None
        ),
        time_sec=note.time_sec,
        beat=float(note.beat),
        tick=note.tick,
        raw=note.raw,
        duration_sec=note.duration_sec,
        duration_beats=(
            float(note.duration_beats) if note.duration_beats is not None else None
        ),
        duration_ticks=note.duration_ticks,
        duration=duration_expr_to_dict(note.duration),
        path=(
            _slide_path(note)
            if note.slide_pattern is not None and note.end_position is not None
            else []
        ),
        segments=[_json_safe_slide_segment(segment) for segment in note.segments],
        modifiers=modifiers,
    )


def parsed_timing_notes_to_chart_notes(
    parsed_points: list[ParsedTimingPointNotes],
) -> list[Note]:
    """Flatten parsed timing-point notes into ChartIR Note objects."""

    return [
        parsed_note_to_chart_note(note)
        for point in parsed_points
        for note in point.notes
    ]


def parsed_timing_point_notes_to_dict(point: ParsedTimingPointNotes) -> dict[str, Any]:
    """Convert parsed timing-point notes to JSON-compatible primitives."""

    return {
        "timing_index": point.timing_index,
        "raw": point.raw,
        "note_text": point.note_text,
        "beat": _format_fraction(point.beat),
        "tick": point.tick,
        "time_sec": point.time_sec,
        "notes": [
            {
                "note_type": note.note_type,
                "position": note.position,
                "is_break": note.is_break,
                "raw": note.raw,
                "time_sec": note.time_sec,
                "beat": _format_fraction(note.beat),
                "tick": note.tick,
                "group_id": note.group_id,
                "duration_beats": (
                    _format_fraction(note.duration_beats)
                    if note.duration_beats is not None
                    else None
                ),
                "duration_ticks": note.duration_ticks,
                "duration_sec": note.duration_sec,
                "duration": duration_expr_to_dict(note.duration),
                "end_position": note.end_position,
                "slide_pattern": note.slide_pattern,
                "slide_segments": [
                    _json_safe_slide_segment(segment)
                    for segment in note.slide_segments
                ],
                "segments": [
                    _json_safe_slide_segment(segment)
                    for segment in note.segments
                ],
                "modifiers": note.modifiers,
            }
            for note in point.notes
        ],
        "unknown_tokens": [
            {
                "raw": token.raw,
                "reason": token.reason,
            }
            for token in point.unknown_tokens
        ],
        "recognized_token_count": point.recognized_token_count,
    }


def parsed_timing_points_notes_to_json(
    parsed_points: list[ParsedTimingPointNotes],
    *,
    indent: int = 2,
) -> str:
    """Serialize parsed timing-point notes to JSON."""

    return json.dumps(
        [parsed_timing_point_notes_to_dict(point) for point in parsed_points],
        ensure_ascii=False,
        indent=indent,
    )


def _split_top_level_slashes(note_text: str) -> list[str]:
    tokens: list[str] = []
    current: list[str] = []
    square_depth = 0

    for char in note_text:
        if char == "[":
            square_depth += 1
        elif char == "]" and square_depth > 0:
            square_depth -= 1

        if char == "/" and square_depth == 0:
            tokens.append("".join(current))
            current = []
        else:
            current.append(char)

    tokens.append("".join(current))
    return tokens


def _split_top_level_stars(note_text: str) -> list[str]:
    tokens: list[str] = []
    current: list[str] = []
    square_depth = 0

    for char in note_text:
        if char == "[":
            square_depth += 1
        elif char == "]" and square_depth > 0:
            square_depth -= 1

        if char == "*" and square_depth == 0:
            tokens.append("".join(current))
            current = []
        else:
            current.append(char)

    tokens.append("".join(current))
    return tokens


def _slide_segment_from_match(match: re.Match[str], raw: str) -> dict[str, Any]:
    duration = parse_duration_expr(match.group("duration"))
    if duration.kind == "unknown":
        raise ValueError("Unsupported slide duration.")

    pattern = match.group("pattern")
    if not is_supported_slide_pattern(pattern):
        raise ValueError(f"Unsupported slide pattern {pattern!r}.")

    end_position = match.group("end_position")
    path_args = _slide_path_args(pattern, end_position)
    segment: dict[str, Any] = {
        "pattern": pattern,
        "end_position": _slide_end_position(pattern, path_args),
        "path_args": path_args,
        "duration_beats": duration.beats,
        "duration_ticks": duration.ticks,
        "duration_sec": duration.seconds,
        "duration": duration,
        "modifiers": _parse_note_modifier_text(
            match.group("before_duration_modifiers") + match.group("tail_modifiers")
        ),
        "raw": raw,
    }
    if "position" in match.groupdict():
        segment["position"] = int(match.group("position"))
        head_modifiers = _parse_note_modifier_text(match.group("head_modifiers"))
        tail_modifiers = segment["modifiers"]
        segment["head_modifiers"] = head_modifiers
        before_duration_modifiers = _parse_note_modifier_text(
            match.group("before_duration_modifiers")
        )
        segment["is_break"] = bool(
            head_modifiers["break"]
            or before_duration_modifiers["break"]
            or tail_modifiers["break"]
        )
    return segment


def _compound_slide_segment_from_match(match: re.Match[str], raw: str) -> dict[str, Any]:
    duration = parse_duration_expr(match.group("duration"))
    if duration.kind == "unknown":
        raise ValueError("Unsupported slide duration.")

    path_text = match.group("path_text")
    path_parts = _parse_compound_slide_path(path_text)
    if not path_parts:
        raise ValueError("Unsupported compound slide path.")
    segment: dict[str, Any] = {
        "pattern": "compound",
        "path_text": path_text,
        "path_parts": path_parts,
        "end_position": _compound_slide_end_position(path_parts),
        "path_args": [
            arg
            for part in path_parts
            for arg in part["path_args"]
        ],
        "duration_beats": duration.beats,
        "duration_ticks": duration.ticks,
        "duration_sec": duration.seconds,
        "duration": duration,
        "modifiers": _parse_note_modifier_text(
            match.group("before_duration_modifiers") + match.group("tail_modifiers")
        ),
        "raw": raw,
    }
    if "position" in match.groupdict():
        head_modifiers = _parse_note_modifier_text(match.group("head_modifiers"))
        tail_modifiers = segment["modifiers"]
        segment["position"] = int(match.group("position"))
        segment["head_modifiers"] = head_modifiers
        before_duration_modifiers = _parse_note_modifier_text(
            match.group("before_duration_modifiers")
        )
        segment["is_break"] = bool(
            head_modifiers["break"]
            or before_duration_modifiers["break"]
            or tail_modifiers["break"]
        )
    return segment


def _json_safe_slide_segment(segment: dict[str, Any]) -> dict[str, Any]:
    safe = dict(segment)
    duration = safe.get("duration")
    if isinstance(duration, DurationExpr):
        safe["duration"] = duration_expr_to_dict(duration)
    duration_beats = safe.get("duration_beats")
    if isinstance(duration_beats, Fraction):
        safe["duration_beats"] = _format_fraction(duration_beats)
    trajectory = safe.get("trajectory")
    if isinstance(trajectory, dict):
        safe["trajectory"] = dict(trajectory)
    return safe


def _slide_path(note: ParsedNote) -> list[str]:
    if not note.slide_segments:
        return [str(note.position), str(note.slide_pattern), str(note.end_position)]

    path = [str(note.position)]
    for index, segment in enumerate(note.slide_segments):
        if index > 0:
            path.append("*")
        if "path_text" in segment:
            path.append(str(segment["path_text"]))
        else:
            path.append(str(segment["pattern"]))
            path.append(str(segment["end_position"]))
    return path


def _format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _note_modifiers(*, is_break: bool = False, is_ex: bool = False) -> dict[str, Any]:
    return {
        "break": is_break,
        "ex": is_ex,
    }


def _parse_note_modifier_text(text: str | None) -> dict[str, Any]:
    text = text or ""
    return _note_modifiers(
        is_break="b" in text,
        is_ex="x" in text,
    )


def _slide_note_modifiers(segments: list[dict[str, Any]]) -> dict[str, Any]:
    is_break = any(bool(segment.get("is_break")) for segment in segments)
    is_ex = any(
        bool(segment.get("head_modifiers", {}).get("ex"))
        or bool(segment.get("modifiers", {}).get("ex"))
        for segment in segments
    )
    return _note_modifiers(is_break=is_break, is_ex=is_ex)


def _touch_position(area: str, position: str) -> str:
    return f"{area}{position}" if position else area


def _slide_path_args(pattern: str, end_position: str) -> list[int]:
    expected_count = get_slide_pattern_definition(pattern).path_arg_count if get_slide_pattern_definition(pattern) else 1
    args = [int(char) for char in end_position]
    if isinstance(expected_count, int) and len(args) != expected_count:
        raise ValueError(
            f"Slide pattern {pattern!r} expects {expected_count} path arg(s), got {len(args)}."
        )
    return args


def _slide_end_position(pattern: str, path_args: list[int]) -> int | None:
    if pattern == "V":
        return path_args[-1] if path_args else None
    return path_args[0] if len(path_args) == 1 else None


def _attach_slide_trajectory_refs(segments: list[dict[str, Any]]) -> None:
    if not segments:
        return
    head_position = int(segments[0]["position"])
    for segment in segments:
        # A top-level '*' creates another slide from the same star head.  The
        # endpoint of the previous part is not the start of the next part.
        _resolve_shortest_arc_in_segment(segment, head_position)
        segment["start_position"] = head_position
        segment["connection"] = "same_head" if len(segments) > 1 else "single"
        segment["trajectory"] = _slide_trajectory_ref(segment, head_position)


def _slide_trajectory_ref(segment: dict[str, Any], start_position: int) -> dict[str, Any]:
    pattern = str(segment["pattern"])
    path_args = [int(value) for value in segment.get("path_args", [])]
    raw_path = _raw_slide_path(segment, start_position)
    return {
        "trajectory_id": raw_path,
        "pattern": pattern,
        "start_position": start_position,
        "end_position": segment.get("end_position"),
        "path_args": path_args,
        "raw": raw_path,
    }


def _resolve_shortest_arc_in_segment(segment: dict[str, Any], start_position: int) -> None:
    """Resolve ``^`` to the shorter clockwise/counter-clockwise arc.

    Simai's ``start^end`` is a compact spelling for whichever of
    ``start<end`` and ``start>end`` travels fewer button intervals.  The
    canonical pattern is stored in ChartIR so downstream labels see the same
    trajectory vocabulary as explicit ``<`` and ``>`` slides.  Opposite
    buttons have no unique shortest direction and retain ``^`` for validation.
    """

    if segment.get("pattern") == "^":
        end_position = _segment_terminal_position(segment)
        if end_position is not None:
            resolved = _shortest_arc_pattern(start_position, end_position)
            segment["requested_pattern"] = "^"
            segment["resolved_pattern"] = resolved
            segment["pattern"] = resolved

    path_parts = segment.get("path_parts")
    if not isinstance(path_parts, list):
        return
    current = start_position
    for part in path_parts:
        if not isinstance(part, dict):
            continue
        if part.get("pattern") == "^":
            end_position = _segment_terminal_position(part)
            if end_position is not None:
                resolved = _shortest_arc_pattern(current, end_position)
                part["requested_pattern"] = "^"
                part["resolved_pattern"] = resolved
                part["pattern"] = resolved
        terminal = _segment_terminal_position(part)
        if terminal is not None:
            current = terminal


def _shortest_arc_pattern(start_position: int, end_position: int) -> str:
    """Resolve ``^`` using Simai's button-orientation convention."""

    clockwise = (end_position - start_position) % 8
    counter_clockwise = (start_position - end_position) % 8
    if clockwise == counter_clockwise:
        return "^"
    is_clockwise = clockwise < counter_clockwise
    start_on_top_half = start_position in {1, 2, 7, 8}
    return ">" if is_clockwise == start_on_top_half else "<"


def _raw_slide_path(segment: dict[str, Any], start_position: int) -> str:
    if "path_text" in segment:
        return f"{start_position}{segment['path_text']}"
    return f"{start_position}{segment['pattern']}{''.join(str(arg) for arg in segment.get('path_args', []))}"


def _segment_terminal_position(segment: dict[str, Any]) -> int | None:
    end_position = segment.get("end_position")
    if isinstance(end_position, int):
        return end_position
    path_args = segment.get("path_args")
    if path_args:
        return int(path_args[-1])
    return None


def _parse_compound_slide_path(path_text: str) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    cursor = 0
    patterns = ("pp", "qq", "-", "<", ">", "^", "p", "q", "s", "z", "v", "V", "w")

    while cursor < len(path_text):
        pattern = next(
            (candidate for candidate in patterns if path_text.startswith(candidate, cursor)),
            None,
        )
        if pattern is None:
            return []
        cursor += len(pattern)

        definition = get_slide_pattern_definition(pattern)
        if definition is None or not definition.supported:
            return []

        if definition.path_arg_count == "variable":
            digits = []
            while cursor < len(path_text) and path_text[cursor].isdigit():
                digits.append(int(path_text[cursor]))
                cursor += 1
            if not digits:
                return []
        else:
            end = cursor + definition.path_arg_count
            digit_text = path_text[cursor:end]
            if len(digit_text) != definition.path_arg_count or not digit_text.isdigit():
                return []
            digits = [int(char) for char in digit_text]
            cursor = end

        parts.append(
            {
                "pattern": pattern,
                "path_args": digits,
                "end_position": _slide_end_position(pattern, digits),
            }
        )

    return parts


def _compound_slide_end_position(parts: list[dict[str, Any]]) -> int | None:
    if not parts:
        return None
    return _segment_terminal_position(parts[-1])


def _combine_segment_durations(
    segments: list[dict[str, Any]],
    raw: str,
) -> DurationExpr:
    """Summarize parallel '*' branches without adding their durations."""
    durations = [
        segment["duration"]
        for segment in segments
        if isinstance(segment.get("duration"), DurationExpr)
    ]
    beats = (
        max(duration.beats for duration in durations)
        if durations and all(duration.beats is not None for duration in durations)
        else None
    )
    seconds = (
        max(duration.seconds for duration in durations)
        if durations and all(duration.seconds is not None for duration in durations)
        else None
    )
    kind = durations[0].kind if len(durations) == 1 else "compound"
    values = (
        list(durations[0].values)
        if len(durations) == 1 and durations[0].values is not None
        else None
    )
    return DurationExpr(
        raw=durations[0].raw if len(durations) == 1 else raw,
        kind=kind,
        beats=beats,
        seconds=seconds,
        ticks=duration_to_ticks_or_none(beats) if beats is not None else None,
        values=values,
    )


def _apply_duration_context(note: ParsedNote, bpm: float) -> None:
    if note.slide_segments:
        for segment in note.slide_segments:
            duration = segment.get("duration")
            if isinstance(duration, DurationExpr):
                resolved = resolve_duration_expr(duration, bpm)
                segment["duration"] = resolved
                segment["duration_beats"] = resolved.beats
                segment["duration_ticks"] = resolved.ticks
                segment["duration_sec"] = resolved.seconds
        note.duration = _combine_segment_durations(note.slide_segments, note.raw)
    elif note.duration is not None:
        note.duration = resolve_duration_expr(note.duration, bpm)

    if note.duration is not None:
        note.duration_beats = note.duration.beats
        note.duration_ticks = note.duration.ticks
        note.duration_sec = note.duration.seconds
