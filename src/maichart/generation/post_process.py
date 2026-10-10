"""Inspect and optionally repair the exact full-song event stream before export."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from fractions import Fraction
from pathlib import Path

from maichart.event_timing import EventPoint, seconds_at_beat
from maichart.generation.conflict_detector import ConflictDetector, ConflictReport, NoteEvent, SlideTrajectory
from maichart.generation.event_quality import parse_event_notes
from maichart.generation.judgment_system import REFERENCE_VERSION
from maichart.notes import BASIC_COMPACT_CHORD_RE


@dataclass
class PostProcessConfig:
    enable_conflict_detection: bool = False
    detect_hard_conflicts: bool = True
    detect_soft_conflicts: bool = False
    conflict_repair_mode: str = "none"
    conflict_retries: int = 0

    def __post_init__(self):
        if self.conflict_repair_mode not in {"none", "local", "reject"}:
            raise ValueError("conflict-repair-mode must be none, local or reject")
        if isinstance(self.conflict_retries, bool) or not isinstance(self.conflict_retries, int) or not 0 <= self.conflict_retries <= 5:
            raise ValueError("conflict-retries must be an integer in 0..5")
        if not self.enable_conflict_detection and (self.conflict_repair_mode != "none" or self.conflict_retries):
            raise ValueError("Enable conflict detection before requesting repair/retries")
        if self.conflict_repair_mode == "none" and self.conflict_retries:
            raise ValueError("Detect-only mode does not regenerate; set retries to 0")

    def to_dict(self):
        return asdict(self)


@dataclass
class PostProcessResult:
    original_notes_count: int
    processed_notes_count: int
    conflicts_detected: list[ConflictReport]
    repairs_applied: list[dict]
    success: bool
    message: str = ""
    points: list[EventPoint] = field(default_factory=list)
    coverage: dict = field(default_factory=dict)
    remaining_conflicts: list[ConflictReport] = field(default_factory=list)

    def to_dict(self):
        return {"schema": "maichart-playability-static-v2", "reference": REFERENCE_VERSION,
                "status": self.coverage.get("status", "disabled"),
                "notes": self.processed_notes_count, "original_notes": self.original_notes_count,
                "conflicts": len(self.conflicts_detected),
                "hard": sum(c.severity == "hard" for c in self.conflicts_detected),
                "details": [c.to_dict() for c in self.conflicts_detected],
                "remaining_hard": sum(c.severity == "hard" for c in self.remaining_conflicts),
                "repairs": self.repairs_applied, "success": self.success,
                "coverage": self.coverage, "message": self.message}


def _note_text(note):
    # A compact chord gives every child the same raw text. Rebuilding a changed
    # cell must not resurrect deleted children or expand every child twice.
    if note.note_type == "tap" and BASIC_COMPACT_CHORD_RE.fullmatch(note.raw):
        suffix = ("b" if note.is_break else "") + ("x" if note.modifiers.get("ex") else "")
        return str(note.position) + suffix
    return note.raw


def _extract(points, tempo_changes, first):
    parsed = parse_event_notes(points, tempo_changes, first)
    events, serializable, unsupported = [], {}, list(parsed.unsupported)
    for point in parsed.points:
        for i, note in enumerate(point.notes):
            source_id = (point.timing_index, i)
            serializable[source_id] = _note_text(note)
            event = NoteEvent(note.note_type, note.position, float(note.time_sec),
                              float(note.duration_sec or 0), raw=note.raw,
                              source_id=source_id, beat=Fraction(note.beat),
                              protected=bool(note.modifiers.get("ex")))
            if note.note_type == "slide":
                event.duration = 0.0  # head is a press; trace is a separate action
                for segment in note.slide_segments:
                    duration = segment.get("duration")
                    parts = segment.get("path_parts") or []
                    if duration is None or parts:
                        unsupported.append({"source": list(source_id), "raw": note.raw,
                                            "reason": "compound_slide_path_not_simulated"})
                        continue
                    if duration.kind == "timing_pair":
                        delay, length = duration.values
                        begin = event.time + float(delay)
                    else:
                        # Default Simai head wait is one beat, not zero.
                        trace_beat = event.beat + 1
                        begin = float(seconds_at_beat(trace_beat, tempo_changes, first))
                        if duration.kind == "grid_fraction":
                            length = float(seconds_at_beat(trace_beat + duration.beats, tempo_changes, first)) - begin
                        else:
                            length = float(duration.seconds or 0)
                    end = segment.get("end_position")
                    pattern = segment.get("pattern")
                    if not isinstance(end, int) or not math.isfinite(length) or length <= 0 or pattern not in {"-", "<", ">", "^"}:
                        unsupported.append({"source": list(source_id), "raw": note.raw,
                                            "reason": "slide_path_or_duration_not_simulated"})
                        continue
                    event.trajectories.append(SlideTrajectory(int(note.position), pattern, end,
                                                              begin, length, source_id))
            events.append(event)
    coverage = {"status": "partial" if unsupported or parsed.fatal_errors else "static_only",
                "unsupported": unsupported, "parse_errors": parsed.fatal_errors,
                "reference_paths": sum(len(e.trajectories) for e in events),
                "limitations": ["静态判定区参考不等同于实机 Perfect 可行性证明。",
                                "复杂 Slide/Wifi/连续转折的有序判定区状态机尚未完整模拟；不自动删除此类配置。",
                                "人体可达性未用玩家动作验证；Touch 多押不按音符个数判无理。"]}
    return events, serializable, coverage


def detect_chart_conflicts(points, tempo_changes, first, config):
    points = list(points)
    if not config.enable_conflict_detection:
        return PostProcessResult(0, 0, [], [], True, points=points, coverage={"status": "disabled"})
    events, _, coverage = _extract(points, tempo_changes, first)
    issues = ConflictDetector(config.detect_hard_conflicts, config.detect_soft_conflicts).detect_conflicts(events)
    hard = any(c.severity == "hard" for c in issues)
    success = not (config.conflict_repair_mode == "reject" and hard)
    # An incomplete parser result cannot be presented as a strict acceptance.
    if config.conflict_repair_mode == "reject" and (
            coverage["parse_errors"] or coverage["unsupported"]):
        success = False
    return PostProcessResult(len(events), len(events), issues, [], success,
                             message=f"{len(issues)} static references", points=points,
                             coverage=coverage, remaining_conflicts=issues)


def process_chart_points(points, tempo_changes, first, config, *, repair_after_beat=None):
    """Return repaired EventPoints; never change the caller's list in place.

    Repair only standalone tap/touch children whose exact source is known.
    Mixed chords retain all unaffected children and their raw modifiers.
    repair_after_beat protects already committed generation context.
    """
    points = list(points)
    initial = detect_chart_conflicts(points, tempo_changes, first, config)
    if config.conflict_repair_mode != "local" or not config.enable_conflict_detection:
        return initial
    # Local repair is only safe when the complete chart was interpreted.  An
    # unsupported Slide/Wifi path can hide additional conflicts; repairing the
    # subset we happened to understand would falsely present a partial result
    # as a repaired chart.  Keep the diagnostics and leave points unchanged.
    if initial.coverage.get("unsupported") or any(
            e.get("code") != "duplicate_button_head"
            for e in initial.coverage.get("parse_errors", [])):
        initial.message = "Syntax diagnostics present: local repair skipped"
        return initial
    events, text_by_source, _ = _extract(points, tempo_changes, first)
    by_source = {e.source_id: e for e in events}
    removed = {}
    for issue in initial.conflicts_detected:
        target = issue.repair_target
        if issue.severity != "hard" or target not in by_source:
            continue
        event = by_source[target]
        if repair_after_beat is not None and event.beat < repair_after_beat:
            continue
        removed[target] = {"point_index": target[0], "note_index": target[1],
                           "beat": str(event.beat), "time_sec": event.time,
                           "raw": text_by_source[target], "reason": issue.conflict_type}
    changed = {index for index, _ in removed}
    repaired = []
    for index, point in enumerate(points):
        if index not in changed:
            repaired.append(point)
            continue
        texts = [text for key, text in text_by_source.items() if key[0] == index and key not in removed]
        if texts:
            repaired.append(EventPoint(point.beat, "/".join(texts)))
    final = detect_chart_conflicts(repaired, tempo_changes, first, config)
    initial.points = repaired
    initial.repairs_applied = list(removed.values())
    initial.processed_notes_count = final.processed_notes_count
    initial.remaining_conflicts = final.conflicts_detected
    initial.message = f"Removed {len(removed)} conflicting press children; rechecked full chart"
    return initial


def save_post_process_report(result, output_path):
    Path(output_path).write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def run_post_process(points, config=None, verbose=True, *, tempo_changes=None, first=0.0):
    if tempo_changes is None:
        raise ValueError("Post-processing requires exact EventPoints and a tempo map")
    return process_chart_points(points, tempo_changes, first, config or PostProcessConfig())
