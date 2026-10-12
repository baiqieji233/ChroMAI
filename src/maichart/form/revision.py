"""Contracts for bounded, transactional chart revision experiments.

These utilities provide data and acceptance rules; they do not claim to learn
musical preference or certify playability. A caller supplies the validator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import hashlib
import math
import random
import time
from typing import Callable, Iterable, Mapping, Sequence

from maichart.event_timing import EventPoint


@dataclass(frozen=True, slots=True)
class EditPatch:
    """Replace the points in the half-open beat interval with ``points``."""

    start: Fraction
    end: Fraction
    points: tuple[EventPoint, ...]
    reason: str
    source_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", Fraction(self.start))
        object.__setattr__(self, "end", Fraction(self.end))
        object.__setattr__(self, "points", tuple(EventPoint(p.beat, p.note_text) for p in self.points))
        if self.end <= self.start:
            raise ValueError("Patch end must be after start")
        if any(not self.start <= p.beat < self.end for p in self.points):
            raise ValueError("Every replacement point must be inside the patch interval")
        if tuple(sorted(p.beat for p in self.points)) != tuple(p.beat for p in self.points):
            raise ValueError("Patch points must be sorted by beat")
        if len({p.beat for p in self.points}) != len(self.points):
            raise ValueError("A patch may contain only one point per beat")


@dataclass(frozen=True, slots=True)
class RevisionAssessment:
    """Comparable validator evidence; coverage is a fraction in [0, 1]."""

    actionable_hard: int
    syntax_errors: int = 0
    coverage: float = 1.0
    quality_score: float | None = None
    hard_keys: frozenset[str] | None = None

    def __post_init__(self) -> None:
        if self.actionable_hard < 0 or self.syntax_errors < 0:
            raise ValueError("Issue counts must be nonnegative")
        if not 0.0 <= self.coverage <= 1.0:
            raise ValueError("coverage must be in [0, 1]")
        if self.quality_score is not None and not math.isfinite(float(self.quality_score)):
            raise ValueError("quality_score must be finite or None")


@dataclass(frozen=True, slots=True)
class RevisionBudget:
    max_checks: int = 8
    max_edits: int = 8
    deadline_secs: float | None = 5.0

    def __post_init__(self) -> None:
        if self.max_checks < 0 or self.max_edits < 0:
            raise ValueError("Budgets must be nonnegative")
        if self.deadline_secs is not None and (
                not math.isfinite(float(self.deadline_secs)) or self.deadline_secs < 0):
            raise ValueError("deadline_secs must be nonnegative or None")


@dataclass(frozen=True, slots=True)
class RevisionResult:
    points: tuple[EventPoint, ...]
    accepted: bool
    reason: str
    checks: int
    edits: int
    before: RevisionAssessment | None = None
    after: RevisionAssessment | None = None


def _assessment(value: RevisionAssessment | Mapping) -> RevisionAssessment:
    if isinstance(value, RevisionAssessment):
        return value
    return RevisionAssessment(
        actionable_hard=int(value["actionable_hard"]),
        syntax_errors=int(value.get("syntax_errors", 0)),
        coverage=float(value.get("coverage", 1.0)),
        quality_score=(None if value.get("quality_score") is None
                       else float(value["quality_score"])),
        hard_keys=(None if value.get("hard_keys") is None
                   else frozenset(str(item) for item in value["hard_keys"])),
    )


def _fingerprint(points: Sequence[EventPoint]) -> str:
    payload = "\n".join(f"{p.beat}:{p.note_text}" for p in points)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def apply_patch_candidate(
    original: Sequence[EventPoint],
    patch: EditPatch,
    validator: Callable[[Sequence[EventPoint]], RevisionAssessment | Mapping],
    *,
    budget: RevisionBudget = RevisionBudget(),
    preserve_onsets: bool = True,
    preserve_note_heads: bool = True,
    tempo_changes: Iterable = (),
    first: float = 0.0,
    failed_candidate_cache: set[str] | None = None,
    seen_candidates: set[str] | None = None,
) -> RevisionResult:
    """Validate a whole-chart candidate and return a new immutable result.

    Onsets are preserved exactly by default. A change that does not reduce the
    actionable hard count needs an externally supplied, strictly improved
    quality score. No note deletion receives implicit credit.
    """
    source = tuple(EventPoint(p.beat, p.note_text) for p in original)
    baseline_key = _fingerprint(source)
    failed = failed_candidate_cache if failed_candidate_cache is not None else set()
    seen = seen_candidates if seen_candidates is not None else set()
    if budget.max_checks == 0:
        return RevisionResult(source, False, "check_budget_exhausted", 0, 0)
    started = time.monotonic()
    if budget.deadline_secs == 0:
        return RevisionResult(source, False, "deadline_exceeded", 0, 0)
    try:
        before = _assessment(validator(source))
    except Exception:
        return RevisionResult(source, False, "baseline_validation_failed", 1, 0)
    checks = 1
    if budget.deadline_secs is not None and time.monotonic() - started >= budget.deadline_secs:
        return RevisionResult(source, False, "deadline_exceeded", checks, 0, before)

    in_patch = tuple(p for p in source if patch.start <= p.beat < patch.end)
    edits = sum(1 for p in in_patch if p.note_text != next(
        (q.note_text for q in patch.points if q.beat == p.beat), ""))
    edits += sum(1 for p in patch.points if not any(q.beat == p.beat for q in in_patch))
    if edits == 0:
        return RevisionResult(source, False, "no_change", checks, 0, before)
    if edits > budget.max_edits:
        return RevisionResult(source, False, "edit_budget_exceeded", checks, edits, before)

    candidate = tuple(sorted(
        tuple(p for p in source if not patch.start <= p.beat < patch.end) + patch.points,
        key=lambda p: p.beat,
    ))
    key = _fingerprint(candidate)
    scoped_key = f"{baseline_key}:{patch.start}:{patch.end}:{patch.source_id}:{key}"
    if key == baseline_key or scoped_key in failed or scoped_key in seen:
        failed.add(scoped_key)
        return RevisionResult(source, False, "cached_or_cyclic_candidate", checks, edits, before)
    if preserve_onsets and {p.beat for p in candidate} != {p.beat for p in source}:
        failed.add(scoped_key)
        return RevisionResult(source, False, "onset_set_changed", checks, edits, before)
    if preserve_note_heads and _note_head_counts(candidate, tempo_changes, first) != _note_head_counts(
            source, tempo_changes, first):
        failed.add(scoped_key)
        return RevisionResult(source, False, "note_head_count_changed", checks, edits, before)
    if checks >= budget.max_checks:
        return RevisionResult(source, False, "check_budget_exhausted", checks, edits, before)
    if budget.deadline_secs is not None and time.monotonic() - started >= budget.deadline_secs:
        return RevisionResult(source, False, "deadline_exceeded", checks, edits, before)

    checks += 1
    try:
        after = _assessment(validator(candidate))
    except Exception:
        failed.add(scoped_key)
        return RevisionResult(source, False, "candidate_validation_failed", checks, edits, before)
    if budget.deadline_secs is not None and time.monotonic() - started >= budget.deadline_secs:
        failed.add(scoped_key)
        return RevisionResult(source, False, "deadline_exceeded", checks, edits, before, after)
    no_regression = after.syntax_errors <= before.syntax_errors and after.coverage >= before.coverage
    hard_nonworse = after.actionable_hard <= before.actionable_hard
    if before.hard_keys is not None and after.hard_keys is not None:
        hard_nonworse = hard_nonworse and after.hard_keys <= before.hard_keys
    hard_improved = after.actionable_hard < before.actionable_hard
    quality_improved = (before.quality_score is not None and after.quality_score is not None
                        and after.quality_score > before.quality_score)
    accepted = no_regression and hard_nonworse and (hard_improved or quality_improved)
    if accepted:
        seen.add(scoped_key)
        return RevisionResult(candidate, True, "accepted", checks, edits, before, after)
    failed.add(scoped_key)
    reason = ("validation_regression" if not no_regression or not hard_nonworse
              else "no_objective_improvement")
    return RevisionResult(source, False, reason, checks, edits, before, after)


def _note_head_counts(points: Sequence[EventPoint], tempo_changes: Iterable, first: float):
    from maichart.generation.event_quality import parse_event_notes
    from maichart.event_timing import EventTempo

    tempo = tuple(tempo_changes)
    if not tempo:
        tempo = (EventTempo(Fraction(0), 120.0),)
    parsed = parse_event_notes(points, tempo, first)
    return tuple((point.beat, len(point.notes)) for point in parsed.points)


@dataclass(frozen=True, slots=True)
class CausalSpan:
    """Exact-beat span of a parsed note/action, with its original point index."""

    start: Fraction
    end: Fraction
    point_index: int
    source_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", Fraction(self.start))
        object.__setattr__(self, "end", Fraction(self.end))
        if self.end < self.start:
            raise ValueError("Causal span end precedes start")


def causal_affected_range(
    defect_start: Fraction,
    defect_end: Fraction,
    spans: Iterable[CausalSpan],
) -> tuple[int, ...]:
    """Return source point indices whose parsed spans overlap a defect range.

    Callers provide parsed sustain/slide durations; earlier roots are included
    whenever their exact span reaches the defect, with no fixed lookback cap.
    """
    start, end = Fraction(defect_start), Fraction(defect_end)
    if end < start:
        raise ValueError("Defect end precedes start")
    return tuple(sorted({s.point_index for s in spans
                         if s.start <= end and s.end >= start}))


@dataclass(frozen=True, slots=True)
class DraftPoint:
    beat: Fraction
    note_text: str | None
    masked: bool = False


@dataclass(frozen=True, slots=True)
class RevisionExample:
    left_context: tuple[EventPoint, ...]
    draft_region: tuple[DraftPoint, ...]
    right_context: tuple[EventPoint, ...]
    target_region: tuple[EventPoint, ...]
    editable_start: Fraction
    editable_end: Fraction
    masked_beats: tuple[Fraction, ...] = ()
    supervision: str = "synthetic_reconstruction"
    clean: bool = False


def build_denoising_examples(
    points: Sequence[EventPoint],
    *,
    count: int,
    seed: int,
    mask_probability: float = 0.25,
    max_masks: int = 3,
    clean_probability: float = 0.1,
) -> tuple[RevisionExample, ...]:
    """Make deterministic trusted-target reconstruction examples.

    Masking is metadata plus ``note_text=None``; it never emits malformed
    Simai text. These examples teach recovery of trusted authored content,
    not aesthetic preference or playability.
    """
    if count < 0 or max_masks < 1:
        raise ValueError("count must be nonnegative and max_masks positive")
    if not 0 <= mask_probability <= 1 or not 0 <= clean_probability <= 1:
        raise ValueError("probabilities must be in [0, 1]")
    source = tuple(EventPoint(p.beat, p.note_text) for p in points)
    rng = random.Random(seed)
    examples = []
    for _ in range(count):
        clean = rng.random() < clean_probability or not source
        masked_indices: set[int] = set()
        if not clean:
            for i in range(len(source)):
                if rng.random() < mask_probability and len(masked_indices) < max_masks:
                    masked_indices.add(i)
            if not masked_indices:
                masked_indices.add(rng.randrange(len(source)))
        if not masked_indices:
            examples.append(RevisionExample((), tuple(DraftPoint(p.beat, p.note_text)
                for p in source), (), source, Fraction(0),
                (source[-1].beat + 1 if source else Fraction(0)), (), clean=True))
            continue
        first, last = min(masked_indices), max(masked_indices)
        region = source[first:last + 1]
        draft = tuple(DraftPoint(p.beat, None if first + j in masked_indices else p.note_text,
                                 first + j in masked_indices)
                      for j, p in enumerate(region))
        editable_end = (source[last + 1].beat if last + 1 < len(source)
                        else region[-1].beat + 1)
        examples.append(RevisionExample(
            left_context=source[:first], draft_region=draft, right_context=source[last + 1:],
            target_region=region, editable_start=region[0].beat,
            editable_end=editable_end,
            masked_beats=tuple(source[i].beat for i in sorted(masked_indices)),
        ))
    return tuple(examples)
