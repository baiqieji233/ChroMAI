"""Inspect and optionally repair the exact full-song event stream before export."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict, replace
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
    conflict_engine: str = "static"
    conflict_kind_modes: dict[str, str] | None = None
    tap_on_slide_threshold_ms: float = 200.0
    conflict_backtrack_windows: int = 0
    conflict_backtrack_budget: int = 4
    sustain_repair_mode: str = "off"
    note_repair_mode: str = "delete"

    def __post_init__(self):
        if self.note_repair_mode not in {"delete", "edit", "edit_then_delete"}:
            raise ValueError("单音符修复模式须为 delete/edit/edit_then_delete")
        kinds = {"MultiTouch", "Overlap", "TapOnSlide", "SlideHeadTap", "SlideTooFast"}
        if self.conflict_kind_modes is not None:
            if not isinstance(self.conflict_kind_modes, dict) or any(
                key not in kinds or mode not in {"off", "report", "enforce"}
                for key, mode in self.conflict_kind_modes.items()):
                raise ValueError("无理类别策略须为五类 MiaCode 名称到 off/report/enforce 的映射")
        if (not math.isfinite(float(self.tap_on_slide_threshold_ms))
                or not 150 <= float(self.tap_on_slide_threshold_ms) <= 250):
            raise ValueError("撞尾静态参考范围必须为 150–250 ms")
        if (isinstance(self.conflict_backtrack_windows, bool)
                or self.conflict_backtrack_windows not in {-1, 0, 1, 2, 4}):
            raise ValueError("回溯窗口数须为 -1（自动）、0、1、2 或 4")
        if (isinstance(self.conflict_backtrack_budget, bool)
                or not isinstance(self.conflict_backtrack_budget, int)
                or not 0 <= self.conflict_backtrack_budget <= 8):
            raise ValueError("全曲回溯预算必须为 0–8 次")
        if self.sustain_repair_mode not in {"off", "hold", "hold_slide"}:
            raise ValueError("前置长音符修复模式须为 off/hold/hold_slide")
        if self.conflict_engine not in {"static", "miacode"}:
            raise ValueError("conflict-engine must be static or miacode")
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
        return {"schema": "maichart-playability-static-v2", "reference": self.coverage.get("reference", REFERENCE_VERSION),
                "status": self.coverage.get("status", "disabled"),
                "notes": self.processed_notes_count, "original_notes": self.original_notes_count,
                "conflicts": len(self.conflicts_detected),
                "hard": sum(c.severity == "hard" for c in self.conflicts_detected),
                "details": [c.to_dict() for c in self.conflicts_detected],
                "remaining_hard": sum(c.severity == "hard" for c in self.remaining_conflicts),
                "remaining_actionable_hard": sum(c.severity == "hard" and actionable(c) for c in self.remaining_conflicts),
                "remaining_details": [c.to_dict() for c in self.remaining_conflicts],
                "repairs": self.repairs_applied, "success": self.success,
                "coverage": self.coverage, "message": self.message}


def _note_text(note):
    # A compact chord gives every child the same raw text. Rebuilding a changed
    # cell must not resurrect deleted children or expand every child twice.
    if note.note_type == "tap" and BASIC_COMPACT_CHORD_RE.fullmatch(note.raw):
        suffix = ("b" if note.is_break else "") + ("x" if note.modifiers.get("ex") else "")
        return str(note.position) + suffix
    return note.raw


def actionable(issue):
    """Keep physical findings separate from the user's generation policy."""
    return getattr(issue, "action", "enforce") == "enforce"


def actionable_hard(result):
    return sum(c.severity == "hard" and actionable(c) for c in result.conflicts_detected)


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
    if config.conflict_engine == "miacode":
        from maichart.generation.miacode_conflicts import detect_native_conflicts
        issues, coverage = detect_native_conflicts(points, tempo_changes, first, events, config)
    else:
        detector = ConflictDetector(config.detect_hard_conflicts, config.detect_soft_conflicts,
                                    conflict_kind_modes=config.conflict_kind_modes)
        issues = detector.detect_conflicts(events)
        coverage["ignored"] = [item.to_dict() for item in getattr(detector, "ignored", [])]
    hard = any(c.severity == "hard" and actionable(c) for c in issues)
    success = not (config.conflict_repair_mode == "reject" and hard)
    # An incomplete parser result cannot be presented as a strict acceptance.
    if config.conflict_repair_mode == "reject" and coverage["parse_errors"]:
        success = False
    return PostProcessResult(len(events), len(events), issues, [], success,
                             message=f"{len(issues)} {config.conflict_engine} references", points=points,
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
    if config.conflict_engine == "miacode":
        return _repair_native(points, tempo_changes, first, config, initial, repair_after_beat)
    if any(e.get("code") != "duplicate_button_head" for e in initial.coverage.get("parse_errors", [])):
        initial.message = "Syntax diagnostics present: local repair skipped"
        return initial
    events, text_by_source, _ = _extract(points, tempo_changes, first)
    by_source = {e.source_id: e for e in events}
    removed = {}
    for issue in initial.conflicts_detected:
        target = issue.repair_target
        if issue.severity != "hard" or not actionable(issue) or target not in by_source:
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
    for repair in initial.repairs_applied:
        print("playability_repair " + json.dumps(repair, ensure_ascii=False), flush=True)
    initial.processed_notes_count = final.processed_notes_count
    initial.remaining_conflicts = final.conflicts_detected
    initial.message = f"Removed {len(removed)} conflicting press children; rechecked full chart"
    return initial


def _repair_native(points, tempo_changes, first, config, initial, repair_after_beat):
    """Remove only exactly mapped affected children and re-run the real judge.

    A deletion is committed only if it reduces hard findings. Never modify
    prior-window context, EX children, parser failures or ambiguous MultiTouch.
    """
    current = initial
    repairs = []
    for _ in range(16):
        if current.coverage.get("parse_errors"):
            break
        hard_before = actionable_hard(current)
        if not hard_before:
            break
        events, texts, _ = _extract(points, tempo_changes, first)
        by_source = {e.source_id: e for e in events}
        accepted = False
        tried = set()
        repair_issues = []
        for issue in current.conflicts_detected:
            repair_issues.append((issue, False))
            if (issue.repair_target is None and issue.severity == "hard" and actionable(issue)
                    and config.note_repair_mode in {"edit", "edit_then_delete"}):
                # MultiTouch may have no unique victim. Propose a few nearby
                # single-note edits, but never delete these inferred candidates.
                nearby = [e for e in events if e.note_type in {"tap", "touch"}
                          and abs(e.time - issue.time) <= .18 and not e.protected]
                for event in nearby[:4]:
                    repair_issues.append((replace(issue, repair_target=event.source_id), True))
        for issue, inferred_target in repair_issues:
            target = issue.repair_target
            if issue.severity != "hard" or not actionable(issue) or target not in by_source or target in tried:
                continue
            tried.add(target)
            event = by_source[target]
            if event.protected or repair_after_beat is not None and event.beat < repair_after_beat:
                continue
            if config.note_repair_mode in {"edit", "edit_then_delete"}:
                from maichart.generation.note_repair import candidate_note_edits
                normalized_event = replace(event, raw=texts[target])
                for candidate in candidate_note_edits(normalized_event, issue, events, tempo_changes, first):
                    new_atom = candidate["raw"]
                    new_text = "/".join(new_atom if key == target else text
                                       for key, text in texts.items() if key[0] == target[0])
                    trial = [EventPoint(p.beat, new_text) if i == target[0] else p
                             for i, p in enumerate(points)]
                    try:
                        after = detect_chart_conflicts(trial, tempo_changes, first, config)
                    except ValueError:
                        continue  # candidate cannot be parsed losslessly
                    hard_after = actionable_hard(after)
                    # Same source identities/onsets/count: comparing each
                    # (kind, affected source) prevents exchanging one hard
                    # problem for a different hard problem, even report-only.
                    before_keys = _hard_keys(current)
                    after_keys = _hard_keys(after)
                    if (after.coverage.get("parse_errors") or hard_after >= hard_before
                            or not after_keys <= before_keys):
                        continue
                    repair = {"point_index": target[0], "note_index": target[1],
                              "beat": str(event.beat), "time_sec": event.time,
                              "raw": texts[target], "replacement": new_atom,
                              "operation": candidate["operation"], "reason": issue.conflict_type,
                              "hard_before": hard_before, "hard_after": hard_after}
                    repairs.append(repair)
                    print("playability_note_edit " + json.dumps(repair, ensure_ascii=False), flush=True)
                    points, current, accepted = trial, after, True
                    break
                if accepted:
                    break
                if config.note_repair_mode == "edit":
                    continue
            if inferred_target:
                continue
            changed = "/".join(text for key, text in texts.items() if key[0] == target[0] and key != target)
            trial = [EventPoint(p.beat, changed) if i == target[0] else p
                     for i, p in enumerate(points) if i != target[0] or changed]
            # Avoid turning a bad single-note chart into an empty success.
            if not trial:
                continue
            after = detect_chart_conflicts(trial, tempo_changes, first, config)
            hard_after = actionable_hard(after)
            if after.coverage.get("parse_errors") or hard_after >= hard_before:
                continue
            repair = {"point_index": target[0], "note_index": target[1], "operation": "delete",
                      "beat": str(event.beat), "time_sec": event.time, "raw": texts[target],
                      "reason": issue.conflict_type, "hard_before": hard_before, "hard_after": hard_after}
            repairs.append(repair)
            print("playability_repair " + json.dumps(repair, ensure_ascii=False), flush=True)
            points, current, accepted = trial, after, True
            break
        if not accepted:
            break
    initial.points = points
    initial.repairs_applied = repairs
    initial.processed_notes_count = current.processed_notes_count
    initial.remaining_conflicts = current.conflicts_detected
    initial.coverage = current.coverage
    deleted = sum(r.get("operation") == "delete" for r in repairs)
    initial.message = f"MiaCode: edited {len(repairs)-deleted}, removed {deleted}; rechecked; remaining_hard={sum(c.severity == 'hard' for c in current.conflicts_detected)}"
    if initial.conflicts_detected:
        print(f"playability_repair_done engine=miacode edited={len(repairs)-deleted} removed={deleted} "
              f"remaining_hard={sum(c.severity == 'hard' for c in current.conflicts_detected)} "
              f"unmapped={len(current.coverage.get('unsupported', []))} "
              f"policy={config.note_repair_mode}", flush=True)
    return initial


def _hard_keys(result):
    return {(c.conflict_type, tuple(n.source_id for n in c.notes[:1]))
            for c in result.conflicts_detected if c.severity == "hard"}


def save_post_process_report(result, output_path):
    Path(output_path).write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def run_post_process(points, config=None, verbose=True, *, tempo_changes=None, first=0.0):
    if tempo_changes is None:
        raise ValueError("Post-processing requires exact EventPoints and a tempo map")
    return process_chart_points(points, tempo_changes, first, config or PostProcessConfig())
