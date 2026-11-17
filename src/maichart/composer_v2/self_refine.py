"""Experimental post-generation worst-span self-refinement (``worst_spans_v1``).

Off by default. When explicitly enabled, inspects 8-beat windows across the
generated chart, identifies up to ``max_spans`` lowest-scoring (or conflict-
bypassed) spans, and resamples replacements from the exact committed prefix.
A replacement is accepted **only** if replaying the replacement **and** the
entire subsequent chart suffix through ``EventGuard`` introduces zero hard
conflicts and improves ergonomic + acoustic score over the boundary span.
"""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import math
from typing import Any

from .hand_flow import chirality_comfort
from .phrase_search import timing_evidence

SELF_REFINE_POLICIES = ("off", "worst_spans_v1")
SPAN_BEATS = Fraction(8, 1)
STRIDE_BEATS = Fraction(4, 1)
MIN_IMPROVEMENT = 0.03
COMFORT_REWRITE_THRESHOLD = 0.58


def validate_self_refine(policy: str = "off") -> str:
    normalized = str(policy or "off").strip()
    if normalized not in SELF_REFINE_POLICIES:
        raise ValueError(f"self_refine must be one of {', '.join(SELF_REFINE_POLICIES)}")
    return normalized


def _span_score(
    points: list[Any],
    song: Any,
    tempo: list[Any],
    b_start: Fraction,
    b_stop: Fraction,
    *,
    motion_cache: Any = None,
) -> tuple[float, float, float]:
    comfort, _ = chirality_comfort(
        points,
        tempo,
        song.clock,
        float(song.clock.at(b_start)),
        float(song.clock.at(b_stop)),
        float(song.audio.duration_sec),
        motion_cache=motion_cache,
    )
    timing = timing_evidence(points, song, b_start, b_stop)
    return 0.75 * float(comfort) + 0.25 * float(timing), float(comfort), float(timing)


def refine_worst_spans(
    points: list[Any],
    guard: Any,
    *,
    args: Any,
    song: Any,
    tempo: list[Any],
    first: float,
    end: Fraction,
    make_sampler: Any,
    new_guard_fn: Any,
    check_window: Any,
    guard_options: dict[str, Any] | None = None,
    bypassed_events: list[dict[str, Any]] | None = None,
    max_spans: int = 2,
) -> tuple[list[Any], Any, dict[str, Any]]:
    policy = validate_self_refine(getattr(args, "self_refine", "off"))
    if policy == "off":
        return points, guard, {"policy": "off", "status": "disabled", "replaced_spans": 0, "spans": []}
    if len(points) < 4 or end <= Fraction(4):
        return points, guard, {
            "policy": policy,
            "status": "skipped_short_chart",
            "replaced_spans": 0,
            "spans": [],
        }

    from collections import OrderedDict

    motion_cache: OrderedDict[Any, Any] = OrderedDict()
    bypassed_beats = {
        Fraction(item["beat"])
        for item in (bypassed_events or ())
        if isinstance(item, dict) and "beat" in item
    }

    candidates_to_rewrite: list[tuple[float, Fraction, Fraction, float, bool]] = []
    total_steps = max(1, int(math.ceil(float(end) / float(STRIDE_BEATS))))
    for step in range(total_steps):
        b0 = Fraction(step) * STRIDE_BEATS
        b1 = min(end, b0 + SPAN_BEATS)
        if b1 - b0 < Fraction(2):
            break
        in_span = [p for p in points if b0 <= p.beat < b1]
        has_bypass = any(b0 <= b < b1 for b in bypassed_beats)
        if len(in_span) < 2 and not has_bypass:
            continue
        blended, comfort, _ = _span_score(points, song, tempo, b0, b1, motion_cache=motion_cache)
        if comfort < COMFORT_REWRITE_THRESHOLD or has_bypass:
            candidates_to_rewrite.append((blended, b0, b1, comfort, has_bypass))

    # Each repair replays the entire suffix. Spend the budget on downstream
    # bypasses first, then select ordinary refinements by score.
    candidates_to_rewrite.sort(key=lambda item: (not item[4], -item[1] if item[4] else item[0]))
    chosen_spans: list[tuple[Fraction, Fraction, float, bool]] = []
    for _, b0, b1, comfort, has_bypass in candidates_to_rewrite:
        if any(not (b1 <= prev_b0 or b0 >= prev_b1) for prev_b0, prev_b1, _, _ in chosen_spans):
            continue
        chosen_spans.append((b0, b1, comfort, has_bypass))
        if len(chosen_spans) >= max_spans:
            break

    if not chosen_spans:
        return points, guard, {
            "policy": policy,
            "status": "all_spans_healthy",
            "replaced_spans": 0,
            "spans": [],
        }

    current_points = list(points)
    current_guard = guard
    from .stream_generate import GridDurationRejected, RefreshContext

    trials_per_span = max(2, int(getattr(args, "candidates", 2)))
    attempts = min(4, max(1, int(getattr(args, "window_retries", 8))))
    span_reports: list[dict[str, Any]] = []
    replaced_count = 0

    for b0, b1, orig_comfort, had_bypass in chosen_spans:
        prefix_pts = [p for p in current_points if p.beat < b0]
        orig_mid_pts = [p for p in current_points if b0 <= p.beat < b1]
        suffix_pts = [p for p in current_points if p.beat >= b1]

        prefix_guard = new_guard_fn(tempo, first, song.audio.duration_sec, guard_options=guard_options)
        prefix_valid = True
        for pt in prefix_pts:
            if not prefix_guard.accepts(pt) and not (had_bypass and pt.beat in bypassed_beats):
                prefix_valid = False
                break
            prefix_guard.commit(pt)
        if not prefix_valid:
            span_reports.append(
                {"start_beat": str(b0), "end_beat": str(b1), "accepted": False, "reason": "prefix_replay_failed"}
            )
            continue

        eval_lo = max(Fraction(0), b0 - Fraction(2))
        eval_hi = min(end, b1 + Fraction(2))
        orig_eval_score, _, _ = _span_score(
            current_points, song, tempo, eval_lo, eval_hi, motion_cache=motion_cache
        )

        best_choice: tuple[float, list[Any], Any] | None = None
        for _ in range(trials_per_span):
            sampler = make_sampler(prefix_pts)
            trial_guard = deepcopy(prefix_guard)
            target: list[int] = []
            cursor = prefix_pts[-1].beat if prefix_pts else Fraction(-1)
            mid_pts: list[Any] = []
            ok = True
            complete = False
            for _step in range(96):
                accepted_pt = None
                accepted_ids = None
                for _retry in range(attempts):
                    try:
                        sampler._active_guard = trial_guard
                        ids = sampler.propose(target, max(cursor, b0 - Fraction(1, 192)), end)
                    except GridDurationRejected:
                        continue
                    except RefreshContext:
                        if not target:
                            break
                        sampler = make_sampler([*prefix_pts, *mid_pts])
                        target = []
                        continue
                    except ValueError:
                        break
                    finally:
                        sampler._active_guard = None
                    if ids == [sampler.codec.eos_id]:
                        accepted_ids = ids
                        break
                    pt = sampler.codec.decode_points(ids)[0]
                    if pt.beat >= b1:
                        accepted_ids = ids
                        break
                    checked, _ = check_window(trial_guard, [pt], [])
                    if checked is not None:
                        trial_guard = checked
                        accepted_pt = pt
                        accepted_ids = ids
                        break
                    if hasattr(sampler, "reject"):
                        sampler.reject(ids)
                if accepted_ids == [sampler.codec.eos_id]:
                    complete = not suffix_pts
                    break
                if accepted_ids is not None and accepted_pt is None:
                    complete = True
                    break
                if accepted_pt is None:
                    ok = False
                    break
                mid_pts.append(accepted_pt)
                target.extend(accepted_ids)
                cursor = accepted_pt.beat
                if (cursor >= getattr(sampler, 'reference_refresh_beat', end)
                        or len(target) >= sampler.max_tokens // 2 - 96):
                    sampler = make_sampler([*prefix_pts, *mid_pts])
                    target = []

            if not ok or not complete or (not mid_pts and orig_mid_pts):
                continue

            # Transactional verification across the entire remaining suffix!
            full_guard, _ = check_window(trial_guard, suffix_pts, [])
            if full_guard is None:
                continue

            spliced = [*prefix_pts, *mid_pts, *suffix_pts]
            cand_score, _, _ = _span_score(
                spliced, song, tempo, eval_lo, eval_hi, motion_cache=motion_cache
            )
            improved = had_bypass or (cand_score >= orig_eval_score + MIN_IMPROVEMENT)
            if improved and (best_choice is None or cand_score > best_choice[0]):
                best_choice = (cand_score, spliced, full_guard)

        if best_choice is not None:
            new_score, current_points, current_guard = best_choice
            replaced_count += 1
            bypassed_beats = {beat for beat in bypassed_beats if not b0 <= beat < b1}
            if bypassed_events is not None and had_bypass:
                bypassed_events[:] = [
                    item
                    for item in bypassed_events
                    if not (b0 <= Fraction(item["beat"]) < b1)
                ]
            span_reports.append(
                {
                    "start_beat": str(b0),
                    "end_beat": str(b1),
                    "accepted": True,
                    "original_comfort": round(orig_comfort, 4),
                    "original_score": round(orig_eval_score, 4),
                    "refined_score": round(new_score, 4),
                    "cleared_bypassed_conflict": bool(had_bypass),
                }
            )
        else:
            span_reports.append(
                {
                    "start_beat": str(b0),
                    "end_beat": str(b1),
                    "accepted": False,
                    "original_comfort": round(orig_comfort, 4),
                    "original_score": round(orig_eval_score, 4),
                    "reason": "no_strictly_better_conflict_free_candidate",
                }
            )

    return current_points, current_guard, {
        "policy": policy,
        "status": "completed",
        "evaluated_spans": len(chosen_spans),
        "replaced_spans": replaced_count,
        "spans": span_reports,
    }
