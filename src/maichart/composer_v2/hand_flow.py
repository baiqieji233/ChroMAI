"""Experimental chirality-aware (Left/Right) 2-hand flow scorer (``comfort_v2``).

Strictly a soft reranking and corpus-audit heuristic — never a token/guard legality
constraint. Reuses MiaCode's real path samples from ``_motion_actions``, replacing
unlabeled 2-trajectory assignment with labeled ``(LeftHand, RightHand)`` beam Viterbi
so unintentional arm crossings, deep opposite-hemisphere reaches, and rapid same-hand
cross-button jumps receive gentle ergonomic penalties while intentional slow Slide
crossovers and center-axis ``1/8``/``4/5`` alternations remain unrestricted.

Includes ``PrecomputedMotionStream`` for O(log N) window slicing when auditing whole
charts across hundreds of sliding windows.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PrecomputedMotionStream:
    """Whole-chart pre-parsed and time-indexed motion stream for fast window slicing."""

    jobs: tuple[dict[str, Any], ...]
    job_starts: np.ndarray
    job_ends: np.ndarray
    pulses: tuple[tuple[float, tuple[float, float], str, float], ...]
    pulse_seconds: np.ndarray
    coverage: dict[str, Any]


def build_motion_stream(
    points: list[Any],
    tempo: list[Any],
    clock: Any,
    *,
    motion_cache: Any = None,
) -> PrecomputedMotionStream:
    """Parse and index a chart's motion actions once for O(log N) window queries."""
    from maichart.composer.stream_quality import _motion_actions

    raw_jobs, raw_pulses, coverage = _motion_actions(points, tempo, clock, cache=motion_cache)
    jobs_t = tuple(raw_jobs)
    if jobs_t:
        job_starts = np.fromiter((float(j["start"]) for j in jobs_t), dtype=np.float64, count=len(jobs_t))
        job_ends = np.fromiter((float(j["end"]) for j in jobs_t), dtype=np.float64, count=len(jobs_t))
    else:
        job_starts = np.empty(0, dtype=np.float64)
        job_ends = np.empty(0, dtype=np.float64)

    sorted_pulses = tuple(sorted(raw_pulses, key=lambda x: x[0]))
    if sorted_pulses:
        pulse_seconds = np.fromiter((float(p[0]) for p in sorted_pulses), dtype=np.float64, count=len(sorted_pulses))
    else:
        pulse_seconds = np.empty(0, dtype=np.float64)

    return PrecomputedMotionStream(
        jobs=jobs_t,
        job_starts=job_starts,
        job_ends=job_ends,
        pulses=sorted_pulses,
        pulse_seconds=pulse_seconds,
        coverage=coverage,
    )


def chirality_comfort(
    points: list[Any],
    tempo: list[Any],
    clock: Any,
    start: float,
    end: float,
    observed_end: float,
    *,
    motion_cache: Any = None,
    motion_stream: PrecomputedMotionStream | None = None,
    beam_width: int = 16,
) -> tuple[float, dict[str, Any]]:
    """Return ``(score_in_0_1, coverage_dict)`` using labeled (L, R) Viterbi."""
    from maichart.composer.stream_quality import _job_position, _motion_actions

    warm = start - 0.6
    horizon = max(end, min(end + 0.35, observed_end))

    if motion_stream is not None:
        coverage = motion_stream.coverage
        if len(motion_stream.jobs):
            mask = (motion_stream.job_ends >= warm) & (motion_stream.job_starts <= horizon)
            jobs = [motion_stream.jobs[i] for i in np.flatnonzero(mask)]
        else:
            jobs = []
        lo = int(np.searchsorted(motion_stream.pulse_seconds, warm, side="left"))
        hi = int(np.searchsorted(motion_stream.pulse_seconds, horizon, side="right"))
        window_pulses = motion_stream.pulses[lo:hi]
    else:
        raw_jobs, raw_pulses, coverage = _motion_actions(points, tempo, clock, cache=motion_cache)
        jobs = [job for job in raw_jobs if job["end"] >= warm and job["start"] <= horizon]
        window_pulses = [
            p for p in sorted(raw_pulses, key=lambda x: x[0]) if warm <= p[0] <= horizon
        ]

    positions: dict[tuple[int, float], tuple[float, float]] = {}

    def position_at(index: int, second: float) -> tuple[float, float]:
        key = (index, second)
        if key not in positions:
            positions[key] = _job_position(jobs[index], second)
        return positions[key]

    events: list[tuple[float, int, tuple[float, float], float, bool, str]] = []
    for index, job in enumerate(jobs):
        begin = max(warm, job["start"])
        finish = min(horizon, job["end"])
        count = min(96, max(1, int(math.ceil((finish - begin) / 0.05))))
        clocks = sorted(
            set(
                np.linspace(begin, finish, count + 1).tolist()
                + [x for seg in job["segments"] for x in seg[:2] if begin <= x <= finish]
            )
        )
        previous = begin
        for second in clocks:
            events.append(
                (second, index, position_at(index, second), max(0.025, second - previous), False, job["kind"])
            )
            previous = second

    groups: list[dict[str, Any]] = []
    for second, position, kind, radius in window_pulses:
        group = next(
            (
                g
                for g in groups
                if kind == "touch"
                and g["kind"] == "touch"
                and abs(g["second"] - second) < 1e-7
                and all(math.dist(position, p) <= min(0.5, radius + r + 0.15) for p, r in g["members"])
            ),
            None,
        )
        if group is None:
            groups.append({"second": second, "kind": kind, "members": [(position, radius)]})
        else:
            group["members"].append((position, radius))

    for group in groups:
        members = group["members"]
        if len(members) == 1:
            pos = members[0][0]
        else:
            pos = (
                sum(p[0] for p, _ in members) / len(members),
                sum(p[1] for p, _ in members) / len(members),
            )
        events.append((group["second"], -1, pos, 1.0, True, group["kind"]))

    events.sort(key=lambda ev: (ev[0], ev[1]))

    # hand 0 = Left (natural x <= 0), hand 1 = Right (natural x >= 0).
    # State entry:
    # (cost, ((posL, tL, kindL), (posR, tR, kindR)), owners_tuple, hemi_cost,
    #  max_step_pen, last_btn_hand, last_btn_sec, alt_hits, alt_total)
    init_hands = (((-0.55, 0.0), warm - 1.0, "idle"), ((0.55, 0.0), warm - 1.0, "idle"))
    beam: list[tuple[float, Any, tuple[int, ...], float, float, int, float, int, int]] = [
        (0.0, init_hands, tuple([-1] * len(jobs)), 0.0, 0.0, -1, warm - 1.0, 0, 0)
    ]
    scored_events = 0

    for second, job_id, target, weight, is_pulse, ev_kind in events:
        updated: dict[tuple[Any, ...], tuple[float, Any, tuple[int, ...], float, float, int, float, int, int]] = {}
        in_window = second >= start
        for cost, hands, owners, hemi_acc, max_step, last_btn_h, last_btn_s, alt_hits, alt_tot in beam:
            assigned = owners[job_id] if job_id >= 0 else -1
            choices = (
                (assigned, 1 - assigned)
                if assigned >= 0 and jobs[job_id]["kind"] in ("slide", "wifi")
                else (assigned,)
                if assigned >= 0
                else (0, 1)
            )
            for hand in choices:
                old_pos, previous_sec, prev_kind = hands[hand]
                other_pos, other_sec, _ = hands[1 - hand]
                occupied = [
                    idx
                    for idx, owner in enumerate(owners)
                    if owner == hand and idx != job_id and jobs[idx]["start"] <= second <= jobs[idx]["end"]
                ]
                origins = [position_at(idx, second) for idx in occupied]
                origin = origins[0] if origins else old_pos
                distance = 0.0 if origin is None else math.dist(origin, target)
                dt = second - previous_sec
                elapsed = max(0.04, dt)
                spread = 0.45 if is_pulse and abs(dt) < 1e-7 else 0.0
                moving = min(3.0, max(0.0, max(0.0, distance - spread) / elapsed - 3.0) / 6.0)
                busy = sum(max(0.0, math.dist(pos, target) - 0.28) / 0.7 for pos in origins)
                transfer = 0.08 if assigned >= 0 and hand != assigned else 0.0

                # 1) Rapid same-hand cross-button jump penalty (favors L/R alternation).
                fast_jump = 0.0
                if is_pulse and ev_kind == "button" and prev_kind == "button" and 1e-5 < dt < 0.16 and distance > 0.26:
                    fast_jump = min(1.5, 0.35 * max(0.0, distance / max(0.045, dt) - 2.2))

                # 2) Gentle hemisphere preference outside center buttons 1/8/4/5 (|x| <= 0.39).
                hemisphere = 0.0
                if ev_kind not in ("slide", "wifi"):
                    if hand == 0 and target[0] > 0.39:
                        hemisphere = 0.12 * (target[0] - 0.39)
                    elif hand == 1 and target[0] < -0.39:
                        hemisphere = 0.12 * (-0.39 - target[0])

                # 3) Soft arm-cross penalty when both hands are simultaneously/recently active.
                arm_cross = 0.0
                if other_pos is not None and (second - other_sec) <= 0.35:
                    x_left = target[0] if hand == 0 else other_pos[0]
                    x_right = other_pos[0] if hand == 0 else target[0]
                    if x_left > x_right + 0.25:
                        scale = 1.0 if is_pulse else min(1.0, weight / 0.1)
                        arm_cross = 0.30 * (x_left - x_right - 0.25) * scale

                hemi_term = hemisphere if is_pulse else hemisphere * (weight / 0.1)
                base_penalty = (moving + busy) * (1.0 if is_pulse else weight / 0.1)
                penalty = base_penalty + transfer + fast_jump + hemi_term + arm_cross

                new_hands = list(hands)
                new_hands[hand] = (target, second, ev_kind)
                new_hands_t = (new_hands[0], new_hands[1])
                allocation = list(owners)
                if job_id >= 0:
                    allocation[job_id] = hand
                alloc_t = tuple(allocation)
                new_cost = cost + (penalty if in_window else 0.0)
                new_hemi = hemi_acc + (hemi_term if in_window else 0.0)
                new_max_step = max(max_step, penalty) if in_window else max_step

                new_btn_h, new_btn_s = last_btn_h, last_btn_s
                new_alt_hits, new_alt_tot = alt_hits, alt_tot
                if is_pulse and ev_kind == "button":
                    if in_window and last_btn_h >= 0 and 0.02 < (second - last_btn_s) <= 0.45:
                        new_alt_tot += 1
                        if hand != last_btn_h:
                            new_alt_hits += 1
                    new_btn_h = hand
                    new_btn_s = second

                # Deduplicate identical ownership + coarse hand coordinates to keep beam diverse.
                sig = (
                    round(new_hands_t[0][0][0], 2),
                    round(new_hands_t[0][0][1], 2),
                    round(new_hands_t[1][0][0], 2),
                    round(new_hands_t[1][0][1], 2),
                    alloc_t,
                )
                prev_entry = updated.get(sig)
                if prev_entry is None or new_cost < prev_entry[0]:
                    updated[sig] = (
                        new_cost,
                        new_hands_t,
                        alloc_t,
                        new_hemi,
                        new_max_step,
                        new_btn_h,
                        new_btn_s,
                        new_alt_hits,
                        new_alt_tot,
                    )

        beam = sorted(updated.values(), key=lambda entry: entry[0])[:beam_width]
        if in_window:
            scored_events += 1

    if beam:
        best_cost, _, _, best_hemi, best_max_step, _, _, best_alt_hits, best_alt_tot = beam[0]
    else:
        best_cost = best_hemi = best_max_step = 0.0
        best_alt_hits = best_alt_tot = 0

    unknown_count = len(coverage["unknown"])
    relevant_actions = len(jobs) + len(groups)
    known = (relevant_actions / max(1, relevant_actions + unknown_count)) if unknown_count else 1.0
    score = math.exp(-best_cost / max(1.0, horizon - start))
    out_coverage = dict(coverage)
    out_coverage.update(
        known_fraction=float(known),
        unknown_count=unknown_count,
        complete=not coverage["unknown"],
        sample_step_sec=0.05,
        max_samples_per_action=96,
        active_jobs=len(jobs),
        touch_contact_groups=sum(g["kind"] == "touch" for g in groups),
        hand_assignment=f"beam{beam_width}_chirality_lr_2d_soft_v2",
        lookahead_sec=max(0.0, horizon - end),
        events=scored_events,
        warmup_sec=0.6,
        best_cost=float(best_cost),
        max_step_penalty=float(best_max_step),
        alternation_pairs=int(best_alt_tot),
        alternation_ratio=float(best_alt_hits / best_alt_tot) if best_alt_tot > 0 else 1.0,
        mean_hemi_penalty=float(best_hemi / max(1, scored_events)),
    )
    out_coverage["unknown"] = out_coverage["unknown"][:32]
    return float(known * score + (1.0 - known) * 0.5), out_coverage
