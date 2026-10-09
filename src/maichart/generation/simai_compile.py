"""Deterministic compilation of frame predictions to Simai text."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from maichart.ir import BpmEvent, ChartIR, ChartMetadata, DifficultyMetadata, GridEvent, Note, TimingData
from maichart.serialization import save_chart_json
from maichart.timing import TICKS_PER_BEAT


@dataclass(slots=True)
class PredictedNote:
    note_type: str
    position: int
    break_note: bool = False
    end_position: int | None = None


@dataclass(slots=True)
class TempoChange:
    beat: Fraction
    bpm: float


def compile_predictions(
    frame_notes: dict[int, list[PredictedNote]],
    *,
    frame_count: int,
    bpm: float,
    division: int = 16,
    first: float = 0.0,
    tempo_changes: list[TempoChange] | None = None,
    title: str = "Generated maimai chart",
    artist: str = "",
    level: str = "13+",
    designer: str = "AI",
) -> tuple[str, ChartIR]:
    """Compile frame-indexed notes using exact integer ticks."""

    if bpm <= 0 or division <= 0:
        raise ValueError("bpm and division must be positive.")
    changes = sorted(tempo_changes or [TempoChange(Fraction(0), bpm)], key=lambda item: item.beat)
    if not changes or changes[0].beat > 0:
        changes.insert(0, TempoChange(Fraction(0), bpm))
    groups: list[str] = []
    chart_notes: list[Note] = []
    current_bpm = changes[0].bpm
    change_index = 0
    for frame_index in range(max(1, frame_count)):
        beat = Fraction(frame_index * 4, division)
        while change_index + 1 < len(changes) and changes[change_index + 1].beat <= beat:
            change_index += 1
            current_bpm = changes[change_index].bpm
        prefix = ""
        if frame_index == 0:
            prefix += f"({_fmt(current_bpm)}){{{division}}}"
        elif change_index > 0 and changes[change_index].beat == beat:
            prefix += f"({_fmt(current_bpm)})"
        tokens: list[str] = []
        for note in sorted(frame_notes.get(frame_index, []), key=lambda item: (item.position, item.note_type)):
            tokens.append(_render_note(note))
            chart_notes.append(
                _to_ir_note(
                    note,
                    beat=beat,
                    time_sec=_seconds_at_beat(beat, changes, first),
                )
            )
        groups.append(prefix + "/".join(tokens))

    simai = ",".join(groups + ["E"])
    chart = ChartIR(
        metadata=ChartMetadata(title=title, artist=artist, offset=first),
        difficulty=DifficultyMetadata(index=5, level=level, designer=designer),
        timing=TimingData(
            bpms=[
                BpmEvent(
                    bpm=change.bpm,
                    beat=float(change.beat),
                    tick=int(change.beat * TICKS_PER_BEAT),
                    time_sec=None,
                )
                for change in changes
            ],
            grids=[GridEvent(division=division, beat=0.0, tick=0, time_sec=None)],
        ),
        notes=chart_notes,
    )
    return simai, chart


def parse_tempo_changes(path: str | Path) -> list[TempoChange]:
    import json

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_changes = payload.get("tempo_changes", payload) if isinstance(payload, dict) else payload
    if not isinstance(raw_changes, list):
        raise ValueError("Tempo map must be a list or an object with 'tempo_changes'.")
    result: list[TempoChange] = []
    for item in raw_changes:
        if not isinstance(item, dict):
            raise ValueError("Each tempo change must be an object.")
        beat = Fraction(str(item.get("beat", 0)))
        value = float(item["bpm"])
        if value <= 0:
            raise ValueError("Tempo values must be positive.")
        result.append(TempoChange(beat, value))
    return result


def save_generated_ir(chart: ChartIR, path: str | Path) -> None:
    save_chart_json(chart, path)


def _render_note(note: PredictedNote) -> str:
    position = str(note.position)
    suffix = "b" if note.break_note else ""
    if note.note_type in {"tap", "break", "none"}:
        return position + suffix
    if note.note_type == "hold":
        return f"{position}{suffix}h[4:1]"
    if note.note_type == "slide":
        end = note.end_position or (note.position % 8 + 1)
        if end == note.position:
            end = note.position % 8 + 1
        return f"{position}{suffix}-{end}[4:1]"
    if note.note_type == "touch_hold":
        return f"A{position}h[4:1]"
    return f"A{position}"


def _to_ir_note(note: PredictedNote, *, beat: Fraction, time_sec: float) -> Note:
    duration_beats = Fraction(1, 1) if note.note_type in {"hold", "slide", "touch_hold"} else None
    output_type = "touch" if note.note_type in {"touch", "touch_hold"} else note.note_type
    if output_type == "break":
        output_type = "tap"
    return Note(
        time_sec=time_sec,
        beat=float(beat),
        tick=int(beat * TICKS_PER_BEAT),
        note_type=output_type,
        position=str(note.position),
        end_position=str(note.end_position) if note.end_position else None,
        duration_beats=float(duration_beats) if duration_beats is not None else None,
        duration_ticks=int(duration_beats * TICKS_PER_BEAT) if duration_beats is not None else None,
        modifiers={"break": note.break_note, "generated": True},
        segments=[{"pattern": "-", "end_position": note.end_position}] if note.note_type == "slide" else [],
    )


def _seconds_at_beat(beat: Fraction, changes: list[TempoChange], first: float) -> float:
    """Integrate the tempo map instead of multiplying by one BPM."""

    elapsed = float(first)
    previous = Fraction(0)
    bpm = changes[0].bpm
    for change in changes[1:]:
        if change.beat >= beat:
            break
        elapsed += float(change.beat - previous) * 60.0 / bpm
        previous = change.beat
        bpm = change.bpm
    elapsed += float(beat - previous) * 60.0 / bpm
    return elapsed


def _fmt(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.6f}".rstrip("0").rstrip(".")
