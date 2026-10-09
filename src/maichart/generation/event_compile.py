"""Adaptive, lossless compilation of exact Simai event points."""

from __future__ import annotations

import json
from collections import defaultdict
from fractions import Fraction
from math import gcd
from typing import Iterable

from maichart.event_timing import (
    EventChart,
    EventPoint,
    EventTempo,
    _exact_beat,
    _finite_number,
    parse_event_chart_text,
    tokenize_event_timing,
)
from maichart.ir import (
    BpmEvent,
    ChartIR,
    ChartMetadata,
    DifficultyMetadata,
    GridEvent,
    TimingData,
    UnknownChartToken,
)
from maichart.maidata import parse_maidata_metadata
from maichart.notes import parse_timing_point_notes, parsed_note_to_chart_note


def compile_event_points(
    points: Iterable[EventPoint],
    *,
    tempo_changes: Iterable[EventTempo] | None = None,
    bpm: float = 120.0,
    first: float = 0.0,
    end_beat: Fraction | None = None,
    title: str = "",
    level: str = "13",
    difficulty: int = 5,
    artist: str = "",
    designer: str = "",
    tail_step: Fraction = Fraction(1, 4),
    strict_notes: bool = False,
) -> tuple[str, ChartIR]:
    """Compile a full Maidata document and verify its exact event roundtrip.

    Each EventPoint is one whole comma-delimited note group. Duplicate heads
    inside that group are preserved; two EventPoints at the same beat are
    rejected because their individual raw strings cannot roundtrip separately.
    Arbitrary positive integer grids are allowed. An omitted end leaves one
    tail_step after the last note or tempo anchor, without inferring durations.
    strict_notes rejects the whole chart on any unsupported note token.
    """

    bpm = _finite_number(bpm, "bpm", positive=True)
    first = _finite_number(first, "first")
    if isinstance(difficulty, bool) or not isinstance(difficulty, int) or difficulty not in range(1, 7):
        raise ValueError("difficulty must be an integer from 1 to 6.")
    for name, value in (("title", title), ("artist", artist), ("level", level), ("designer", designer)):
        if "\n" in value or "\r" in value:
            raise ValueError(f"{name} must be a single-line Maidata field.")

    events = sorted(
        (EventPoint(point.beat, point.note_text) for point in points), key=lambda point: point.beat
    )
    # ``E`` is the chart terminator in Simai syntax, not a playable note.
    # A model may emit it as a note atom; retaining it in the event list causes
    # the parser to terminate the timing stream early and makes an otherwise
    # valid generated chart impossible to round-trip.  Keep the event for
    # diagnostics upstream, but omit it from the compiler's playable map.
    events = [point for point in events if point.note_text and point.note_text != "E"]
    by_beat: dict[Fraction, str] = {}
    for point in events:
        if point.beat < 0:
            raise ValueError("Event beats must be nonnegative.")
        if point.beat in by_beat:
            raise ValueError("Use one EventPoint per beat, keeping all heads in its raw note group.")
        by_beat[point.beat] = point.note_text

    changes = sorted(
        (
            EventTempo(
                _exact_beat(change.beat), _finite_number(change.bpm, "BPM", positive=True)
            )
            for change in (tempo_changes if tempo_changes is not None else [])
        ),
        key=lambda change: change.beat,
    )
    if any(change.beat < 0 for change in changes):
        raise ValueError("Tempo beats must be nonnegative.")
    if not changes or changes[0].beat != 0:
        changes.insert(0, EventTempo(Fraction(0), bpm))
    last_anchor = max([Fraction(0), *(point.beat for point in events), *(change.beat for change in changes)])
    if end_beat is None:
        tail_step = _exact_beat(tail_step)
        if tail_step <= 0:
            raise ValueError("tail_step must be positive.")
        end = last_anchor + tail_step
    else:
        end = _exact_beat(end_beat)
    if end < 0 or any(point.beat >= end for point in events) or any(change.beat > end for change in changes):
        raise ValueError("end_beat must be after every note and not before any tempo change.")

    tempos: dict[Fraction, list[EventTempo]] = defaultdict(list)
    for change in changes:
        tempos[change.beat].append(change)
    anchors = {Fraction(0), end, *by_beat, *tempos}
    boundary = Fraction(4)
    while boundary < end:
        anchors.add(boundary)
        boundary += 4
    anchors = sorted(anchors)

    rendered: list[tuple[Fraction, str]] = []
    current_division: int | None = None
    for start, target in zip(anchors, anchors[1:]):
        beat = start
        while beat < target:
            remaining = target - beat
            # Choose the smallest exact reciprocal grid for this local
            # remainder. For example 4/3 beats becomes {3}; 1/7 becomes
            # {28}. No floating-point rounding is involved.
            division = 4 * remaining.denominator // gcd(remaining.numerator, 4)
            prefix = "".join(f"({_format_number(change.bpm)})" for change in tempos.get(beat, []))
            if division != current_division:
                prefix += f"{{{division}}}"
                current_division = division
            rendered.append((beat, prefix + by_beat.get(beat, "")))
            beat += Fraction(4, division)
    end_prefix = "".join(f"({_format_number(change.bpm)})" for change in tempos.get(end, []))
    rendered.append((end, end_prefix + "E"))

    parts: list[str] = []
    for index, (beat, text) in enumerate(rendered):
        if index:
            parts.append(",\n" if beat > 0 and beat.denominator == 1 and beat.numerator % 4 == 0 else ",")
        parts.append(text)
    inote = "".join(parts)
    maidata_text = "\n".join(
        [
            f"&title={title}",
            f"&artist={artist}",
            f"&first={_format_number(first)}",
            f"&wholebpm={_format_number(changes[0].bpm)}",
            f"&lv_{difficulty}={level}",
            f"&des_{difficulty}={designer}",
            f"&inote_{difficulty}={inote}",
        ]
    ) + "\n"

    # The rendered document always ends with an explicit ``E`` at ``end``.
    # The parser therefore treats that beat as the chart end.  Comparing against
    # an implicit tail-derived end here rejects otherwise exact generated charts
    # whenever the final event is earlier than the tail anchor.
    expected = EventChart(events, changes, first, level, title, end, artist, difficulty)
    reparsed = parse_event_chart_text(maidata_text, difficulty, strict_notes=strict_notes)
    if reparsed != expected:
        differences = []
        for field in ("points", "tempo_changes", "first", "level", "title", "end_beat", "artist", "difficulty", "time_offset"):
            left = getattr(expected, field)
            right = getattr(reparsed, field)
            if left != right:
                differences.append(f"{field}: expected={left!r} reparsed={right!r}")
        detail = "; ".join(differences[:4])
        raise ValueError(f"Compiled Maidata does not preserve exact events, tempos, or metadata. {detail}")
    return maidata_text, _build_ir(reparsed, inote, designer)


def build_event_chart_ir(
    maidata_text: str,
    difficulty: int,
    *,
    strict_notes: bool = False,
) -> ChartIR:
    """Build regular ChartIR without forcing exact event beats to integer ticks."""

    raw_chart = parse_maidata_metadata(maidata_text)
    difficulty_block = next(
        (item for item in raw_chart.difficulties if item.index == difficulty), None
    )
    if difficulty_block is None or difficulty_block.inote is None:
        raise ValueError(f"No inote for difficulty {difficulty} found.")
    events = parse_event_chart_text(maidata_text, difficulty, strict_notes=strict_notes)
    return _build_ir(events, difficulty_block.inote, difficulty_block.designer or "")


def _build_ir(events: EventChart, inote: str, designer: str) -> ChartIR:
    chart = ChartIR(
        metadata=ChartMetadata(
            title=events.title,
            artist=events.artist,
            offset=events.first,
            raw=json.dumps(events.to_dict(), ensure_ascii=True),
        ),
        difficulty=DifficultyMetadata(index=events.difficulty, level=events.level, designer=designer),
        timing=TimingData(),
        raw=f"&inote_{events.difficulty}={inote}\n",
    )
    timing_points = tokenize_event_timing(inote, events.tempo_changes[0].bpm, first=events.first)
    for point in timing_points:
        for directive in point.directives:
            fields = dict(beat=float(point.beat), tick=point.tick, time_sec=point.time_sec, raw=directive.raw)
            if directive.directive_type == "bpm":
                chart.timing.bpms.append(BpmEvent(bpm=float(directive.value), **fields))
            elif directive.directive_type == "grid":
                chart.timing.grids.append(GridEvent(division=int(directive.value), **fields))
        parsed = parse_timing_point_notes(point)
        chart.notes.extend(parsed_note_to_chart_note(note) for note in parsed.notes)
        chart.unknown_tokens.extend(
            UnknownChartToken(
                time_sec=point.time_sec,
                beat=float(point.beat),
                tick=point.tick,
                raw=token.raw,
                timing_index=point.index,
                raw_timing_point=point.raw,
                raw_token=token.raw,
                reason=token.reason,
            )
            for token in parsed.unknown_tokens
        )
    return chart


def _format_number(value: float) -> str:
    """Render a BPM for a ``(N)`` or ``&wholebpm=`` field.

    A beat tracker returns lag bins, so writing ``repr()`` of the estimate put
    values such as ``136.36363636363637`` straight into the chart.  This function
    only removes float representation noise -- it never rounds a tempo onto a
    musical grid, because the caller has already chosen the clock and an
    explicit ``--bpm 174.63`` must survive verbatim.  Values that need more than
    three decimals to distinguish are rounded to three, which is finer than any
    ``{division}`` can resolve in practice.
    """

    number = float(value)
    if number.is_integer():
        return str(int(number))
    text = f"{number:.3f}".rstrip("0").rstrip(".")
    return text or "0"
