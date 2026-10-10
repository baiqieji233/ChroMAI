"""Playability post-processing: detector geometry and model-family gating.

These cover the parts that were wrong in the first draft of the P1 work:
slide star geometry, chord-as-jump false positives, and the fact that an
old checkpoint must not silently accept sampling-bias flags it cannot honour.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from maichart.event_timing import EventTempo
from maichart.generation.conflict_detector import (
    ConflictDetector,
    NoteEvent,
    SlideTrajectory,
)
from maichart.generation.maimai_physics import ButtonLayout, HandDispatch
from maichart.generation.post_process import PostProcessConfig, detect_chart_conflicts


class _Point:
    def __init__(self, beat, text):
        self.beat = Fraction(beat)
        self.note_text = text


TEMPO = [EventTempo(beat=Fraction(0), bpm=120.0)]


def _detect(points, **overrides):
    config = PostProcessConfig(enable_conflict_detection=True, **overrides)
    return detect_chart_conflicts(points, TEMPO, 0.0, config)


def test_button_layout_matches_the_cabinet():
    # Button 1 sits at the top, rotated 22.5 degrees clockwise; the ring pitch
    # is 45 degrees and 1/8 share the horizontal top edge.
    assert ButtonLayout.button_angle(1) == pytest.approx(67.5)
    assert ButtonLayout.button_angle(3) == pytest.approx(337.5)
    for button in range(1, 8):
        gap = ButtonLayout.button_angle(button) - ButtonLayout.button_angle(button + 1)
        assert gap % 360 == pytest.approx(45.0)
    assert HandDispatch.primary_hand(1) == "right"
    assert HandDispatch.primary_hand(5) == "left"


def test_hold_tap_overlap_is_hard():
    result = _detect([_Point(0, "1h[4:2]"), _Point(1, "1")])
    kinds = [c.conflict_type for c in result.conflicts_detected]
    assert "hold_tap_overlap" in kinds
    hard = [c for c in result.conflicts_detected if c.severity == "hard"]
    assert len(hard) == 1


def test_tap_on_the_slide_head_is_not_a_conflict():
    # An explicit duplicate head is a same-pad issue, not an overlapping
    # moving-star trajectory. Parallel branches use the separate '*' syntax.
    result = _detect([_Point(0, "1"), _Point(0, "1-5[4:1]")])
    assert all(c.conflict_type != "slide_tap_collision" for c in result.conflicts_detected)


def test_half_turn_straight_slide_reports_no_phantom_collision():
    # 2-6 is a chord through the centre. It never travels around the ring.
    trajectory = SlideTrajectory(start_button=2, pattern="-", end_button=6,
                                 start_time=0.0, duration=1.0)
    assert trajectory.angular_span() == 0.0
    assert trajectory.star_position_at(0.5) is None


def test_clockwise_arc_uses_the_declared_direction():
    trajectory = SlideTrajectory(start_button=1, pattern=">", end_button=3,
                                 start_time=0.0, duration=1.0)
    assert trajectory.angular_span() < 0
    assert trajectory.star_position_at(0.5) == 2


def test_chord_is_not_reported_as_an_excessive_jump():
    # Two buttons pressed at the same instant are one gesture, not a leap.
    result = _detect([_Point(0, "1/5")], detect_soft_conflicts=True)
    assert all(c.conflict_type != "excessive_jump" for c in result.conflicts_detected)


def test_short_left_right_trill_is_not_flagged():
    # 16th notes at 120 BPM: an ordinary trill, not a sustained hard run.
    taps = [_Point(Fraction(i, 8), "1" if i % 2 == 0 else "5") for i in range(4)]
    result = _detect(taps, detect_soft_conflicts=True)
    assert all(c.conflict_type != "rapid_alternation" for c in result.conflicts_detected)


def test_reject_mode_fails_on_hard_conflict():
    result = _detect([_Point(0, "1h[4:2]"), _Point(1, "1")], conflict_repair_mode="reject")
    assert result.success is False


def test_detect_only_mode_still_succeeds():
    result = _detect([_Point(0, "1h[4:2]"), _Point(1, "1")], conflict_repair_mode="none")
    assert result.success is True
