"""Bounded whole-chart causal repair using existing conflict evidence.

This is a deterministic repair runner, not a learned model or a playability
certificate. Every proposed single-note edit is reparsed and redetected over
the complete chart before it can replace the current best chart.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from fractions import Fraction
import hashlib
import json
import math
import time
from typing import Iterable, Sequence

from maichart.event_timing import EventPoint, EventTempo
from maichart.generation.note_repair import candidate_note_edits
from maichart.generation.post_process import (
    PostProcessConfig,
    _extract,
    actionable,
    actionable_hard,
    detect_chart_conflicts,
)
from maichart.form.revision import (
    EditPatch,
    RevisionAssessment,
    RevisionBudget,
    apply_patch_candidate,
    _fingerprint,
)


@dataclass(frozen=True, slots=True)
class RepairBudget:
    max_checks: int = 24
    max_edits: int = 4
    deadline_secs: float = 10.0

    def __post_init__(self) -> None:
        if self.max_checks < 1 or self.max_edits < 0:
            raise ValueError("Repair budgets require at least one check and nonnegative edits")
        if not math.isfinite(float(self.deadline_secs)) or self.deadline_secs < 0:
            raise ValueError("deadline_secs must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class RepairAttempt:
    source_id: tuple[int, int]
    conflict_type: str
    operation: str
    replacement: str
    accepted: bool
    reason: str


@dataclass(frozen=True, slots=True)
class FormRepairResult:
    points: tuple[EventPoint, ...]
    repairs: tuple[dict, ...]
    attempts: tuple[RepairAttempt, ...]
    remaining_actionable_hard: int | None
    checks: int
    cache_hits: int
    elapsed_secs: float
    status: str
    message: str = ""
    original_coverage: dict | None = None
    final_coverage: dict | None = None

    def to_dict(self) -> dict:
        return {
            "schema": "maichart-form-repair-v1",
            "status": self.status,
            "repairs": list(self.repairs),
            "attempts": [asdict(a) for a in self.attempts],
            "remaining_actionable_hard": self.remaining_actionable_hard,
            "checks": self.checks,
            "cache_hits": self.cache_hits,
            "elapsed_secs": self.elapsed_secs,
            "message": self.message,
            "coverage_status": _coverage_status(self.final_coverage),
            "original_coverage": self.original_coverage,
            "final_coverage": self.final_coverage,
        }


def _policy_key(config, tempo_changes, first) -> str:
    payload = {
        "config": asdict(config),
        "tempo": [(str(t.beat), float(t.bpm)) for t in tempo_changes],
        "first": float(first),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _coverage_score(result) -> float:
    status = result.coverage.get("status", "partial")
    if status in {"engine_reference", "static_only"}:
        return 1.0
    if status == "disabled":
        return 0.0
    return 0.5


def _coverage_status(coverage: dict | None) -> str:
    if not coverage:
        return "unverified"
    status = coverage.get("status")
    if status in {"engine_reference", "static_only", "partial"}:
        return status
    return "unverified"


def _assessment(result) -> RevisionAssessment:
    hard_keys = frozenset(
        f"{issue.conflict_type}:{tuple(_event_source_id(note) for note in issue.notes)}"
        for issue in result.conflicts_detected
        if issue.severity == "hard" and actionable(issue)
    )
    return RevisionAssessment(
        actionable_hard=actionable_hard(result),
        syntax_errors=len(result.coverage.get("parse_errors", [])),
        coverage=_coverage_score(result),
        hard_keys=hard_keys,
    )


def _event_source_id(event):
    return tuple(getattr(event, "source_id", (-1, -1)))


def _causal_targets(result, events):
    """Order sustained roots that span the finding before nearby victims."""
    by_source = {_event_source_id(event): event for event in events}
    proposals = []
    seen = set()
    for issue in result.conflicts_detected:
        if issue.severity != "hard" or not actionable(issue):
            continue
        direct = {_event_source_id(note) for note in issue.notes}
        if issue.repair_target is not None:
            direct.add(tuple(issue.repair_target))
        roots = set(direct)
        for event in events:
            source = _event_source_id(event)
            end = float(event.time) + float(event.duration or 0.0)
            spans = (event.note_type in {"hold", "touch_hold"}
                     and event.duration > 0
                     and event.time - 1e-9 <= issue.time <= end + 1e-9)
            if event.note_type == "slide":
                spans = spans or any(
                    trajectory.start_time - 1e-9 <= issue.time
                    <= trajectory.start_time + trajectory.duration + 1e-9
                    for trajectory in event.trajectories
                )
            if spans:
                roots.add(source)
        ordered = sorted(
            (source for source in roots if source in by_source),
            key=lambda source: (
                0 if (by_source[source].note_type in {"hold", "touch_hold", "slide"}
                      and by_source[source].time < issue.time) else 1,
                by_source[source].time,
                source,
            ),
        )
        for source in ordered:
            marker = (source, issue.conflict_type, round(float(issue.time), 6))
            if marker not in seen:
                seen.add(marker)
                proposals.append((by_source[source], issue))
    return proposals


def _point_end(points: Sequence[EventPoint], beat: Fraction) -> Fraction:
    later = [point.beat for point in points if point.beat > beat]
    return min(later) if later else beat + 1


def _replace_note_atom(points, text_by_source, target, replacement):
    point_index = target[0]
    new_text = "/".join(
        replacement if source == target else text
        for source, text in text_by_source.items()
        if source[0] == point_index
    )
    return tuple(EventPoint(point.beat, new_text) if index == point_index else
                 EventPoint(point.beat, point.note_text)
                 for index, point in enumerate(points))


def repair_chart(
    points: Sequence[EventPoint],
    tempo_changes: Iterable[EventTempo],
    first: float = 0.0,
    *,
    config: PostProcessConfig | None = None,
    budget: RepairBudget = RepairBudget(),
) -> FormRepairResult:
    """Try bounded note-preserving edits against real whole-chart evidence."""
    original = tuple(EventPoint(p.beat, p.note_text) for p in points)
    tempo = tuple(EventTempo(Fraction(t.beat), float(t.bpm)) for t in tempo_changes)
    if not tempo or tempo[0].beat != 0:
        raise ValueError("tempo_changes must explicitly include a BPM at beat 0")
    started = time.monotonic()
    base_config = config or PostProcessConfig()
    detector_config = replace(base_config, enable_conflict_detection=True,
                              conflict_repair_mode="none")
    policy_key = _policy_key(detector_config, tempo, first)
    assessment_cache = {}
    result_cache = {}
    cache_hits = checks = 0
    expired = False

    class _BudgetStop(Exception):
        pass

    def assess(chart):
        nonlocal checks, cache_hits, expired
        key = (_fingerprint(chart), policy_key)
        if key in assessment_cache:
            cache_hits += 1
            return assessment_cache[key]
        if checks >= budget.max_checks or time.monotonic() - started >= budget.deadline_secs:
            expired = True
            raise _BudgetStop("repair budget exhausted")
        checks += 1
        report = detect_chart_conflicts(chart, tempo, first, detector_config)
        result_cache[key] = report
        value = _assessment(report)
        assessment_cache[key] = value
        return value

    current = original
    try:
        before = assess(current)
    except _BudgetStop:
        return FormRepairResult(original, (), (), None, checks, cache_hits,
                                time.monotonic() - started, "budget_exhausted")
    except Exception as exc:
        return FormRepairResult(original, (), (), None, checks, cache_hits,
                                time.monotonic() - started, "baseline_validation_failed",
                                f"{type(exc).__name__}: {exc}")
    original_report = result_cache[(_fingerprint(current), policy_key)]
    failed, seen = set(), set()
    repairs = []
    attempts: list[RepairAttempt] = []
    while len(repairs) < budget.max_edits:
        if before.actionable_hard == 0:
            break
        if time.monotonic() - started >= budget.deadline_secs or checks >= budget.max_checks:
            expired = True
            break
        report = result_cache[(_fingerprint(current), policy_key)]
        events, text_by_source, _ = _extract(current, tempo, first)
        proposals = _causal_targets(report, events)
        accepted_any = False
        for event, issue in proposals:
            target = _event_source_id(event)
            for candidate in candidate_note_edits(event, issue, events, tempo, first):
                if checks >= budget.max_checks or time.monotonic() - started >= budget.deadline_secs:
                    expired = True
                    break
                edited = _replace_note_atom(current, text_by_source, target, candidate["raw"])
                patch = EditPatch(event.beat, _point_end(current, event.beat),
                                  tuple(p for p in edited if event.beat <= p.beat < _point_end(current, event.beat)),
                                  reason=issue.conflict_type, source_id=str(target))
                revision = apply_patch_candidate(
                    current, patch, assess,
                    budget=RevisionBudget(max_checks=2, max_edits=1,
                                          deadline_secs=max(0.0, budget.deadline_secs -
                                                           (time.monotonic() - started))),
                    tempo_changes=tempo, first=first,
                    failed_candidate_cache=failed, seen_candidates=seen,
                )
                if revision.accepted:
                    current = revision.points
                    before = revision.after
                    repairs.append({
                        "source_id": list(target), "beat": str(event.beat),
                        "time_sec": float(event.time), "original": event.raw,
                        "replacement": candidate["raw"], "operation": candidate["operation"],
                        "reason": issue.conflict_type,
                        "hard_before": revision.before.actionable_hard,
                        "hard_after": revision.after.actionable_hard,
                    })
                    accepted_any = True
                    break
                if revision.reason == "deadline_exceeded":
                    expired = True
                attempts.append(RepairAttempt(
                    source_id=target, conflict_type=issue.conflict_type,
                    operation=candidate["operation"], replacement=candidate["raw"],
                    accepted=False, reason=("budget_exhausted" if expired else revision.reason),
                ))
                if expired:
                    break
            if accepted_any or expired:
                break
        if expired or not accepted_any:
            break
    final_report = result_cache[(_fingerprint(current), policy_key)]
    final_coverage = final_report.coverage
    coverage_status = _coverage_status(final_coverage)
    if coverage_status == "partial":
        status = "partial_coverage"
    elif coverage_status == "unverified":
        status = "unverified"
    elif expired:
        status = "budget_exhausted" if not repairs else "partial_budget_exhausted"
    elif before.actionable_hard == 0:
        status = ("resolved_under_engine_reference" if coverage_status == "engine_reference"
                  else "resolved_under_static_only")
    elif repairs:
        status = ("partially_resolved_under_engine_reference" if coverage_status == "engine_reference"
                  else "partially_resolved_under_static_only")
    else:
        status = ("unresolved_under_engine_reference" if coverage_status == "engine_reference"
                  else "unresolved_under_static_only")
    return FormRepairResult(
        points=tuple(EventPoint(p.beat, p.note_text) for p in current),
        repairs=tuple(repairs), attempts=tuple(attempts),
        remaining_actionable_hard=before.actionable_hard,
        checks=checks, cache_hits=cache_hits,
        elapsed_secs=time.monotonic() - started, status=status,
        original_coverage=original_report.coverage,
        final_coverage=final_coverage,
    )
