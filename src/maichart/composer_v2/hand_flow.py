"""Two-hand phrase recognition and optional comfort scoring over native DX paths."""

from __future__ import annotations

from dataclasses import dataclass
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
    star_heads: frozenset = frozenset()


def build_motion_stream(
    points: list[Any],
    tempo: list[Any],
    clock: Any,
    *,
    motion_cache: Any = None,
) -> PrecomputedMotionStream:
    """Index transient contacts and sustain times once for repeated window queries."""
    from maichart.composer.stream_quality import _motion_actions

    raw_jobs, raw_pulses, coverage = _motion_actions(points, tempo, clock, cache=motion_cache)
    jobs_t = tuple(raw_jobs)
    if jobs_t:
        job_starts = np.fromiter((float(j["trace"]) for j in jobs_t), dtype=np.float64, count=len(jobs_t))
        job_ends = np.fromiter((float(j["end"]) for j in jobs_t), dtype=np.float64, count=len(jobs_t))
    else:
        job_starts = np.empty(0, dtype=np.float64)
        job_ends = np.empty(0, dtype=np.float64)

    sorted_pulses = tuple(sorted(raw_pulses, key=lambda x: x[0]))
    if sorted_pulses:
        pulse_seconds = np.fromiter(
            (float(p[0]) for p in sorted_pulses), dtype=np.float64, count=len(sorted_pulses)
        )
    else:
        pulse_seconds = np.empty(0, dtype=np.float64)

    return PrecomputedMotionStream(
        jobs=jobs_t,
        job_starts=job_starts,
        job_ends=job_ends,
        pulses=sorted_pulses,
        pulse_seconds=pulse_seconds,
        coverage=coverage,
        star_heads=frozenset((round(j["onset"], 7), j["position"]) for j in jobs_t if j.get("head_contact")),
    )


def recognize_hands(
    points,
    tempo,
    clock,
    start,
    end,
    observed_end,
    *,
    motion_cache=None,
    motion_stream=None,
    beam_width=16,
    max_solutions=3,
):
    """Return whole-phrase alternatives, with assignments in absolute audio seconds.

    Each frame contains simultaneous left/right contact lists. alternate_cost_gap
    measures the cheapest different assignment through the retained search graph.
    None indicates lack of a surviving rival, not certainty about human technique.
    """
    from .hand_assignment import analyze

    stream = (
        motion_stream
        if motion_stream is not None
        else build_motion_stream(points, tempo, clock, motion_cache=motion_cache)
    )
    return analyze(stream, start, end, observed_end, beam_width, max_solutions, True)


def chirality_comfort(
    points, tempo, clock, start, end, observed_end, *, motion_cache=None, motion_stream=None, beam_width=16
):
    """Score the same simultaneous-contact paths used for phrase diagnostics."""
    from .hand_assignment import analyze

    stream = (
        motion_stream
        if motion_stream is not None
        else build_motion_stream(points, tempo, clock, motion_cache=motion_cache)
    )
    result = analyze(stream, start, end, observed_end, beam_width, 1, False)
    return result["score"], result["coverage"]
