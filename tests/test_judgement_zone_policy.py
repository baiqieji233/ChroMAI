from fractions import Fraction as F

import pytest

from maichart.event_timing import EventPoint as P, EventTempo as T
from maichart.generation.conflict_detector import SlideTrajectory
from maichart.generation.event_cli import _DecodedWindow, _select_playable_window
from maichart.generation.event_model import EventSequence
from maichart.generation.judgment_system import pad_center, touch_groups, touch_group_threshold
from maichart.generation.post_process import PostProcessConfig, detect_chart_conflicts, process_chart_points, _extract


TEMPO = [T(F(0), 120)]


def config(mode="none", retries=0, **kw):
    return PostProcessConfig(True, conflict_repair_mode=mode, conflict_retries=retries, **kw)


def test_line_passes_centre_not_outer_ring():
    path = SlideTrajectory(2, "-", 6, 1.0, 1.0)
    assert path.position_at(1.5) == pytest.approx((0, 0), abs=1e-9)
    pads = {p for p, _, _ in path.pad_intervals()}
    assert "C" in pads
    assert not pads & {"A1", "A8", "A7"}
    assert path.position_at(.9) is None


def test_touch_ring_layout_and_group_judgement():
    assert pad_center("C") == (0, 0)
    assert pad_center("A1")[1] == pytest.approx(pad_center("A8")[1])
    assert pad_center("A1")[0] == pytest.approx(-pad_center("A8")[0])
    assert touch_group_threshold(5) == 3
    assert touch_groups(["B1", "B8", "C"]) == [[0, 1, 2]]
    dense_touch = [P(F(0), "1/5/C/B1/B2/B3/B4/B5/B6/B7/B8")]
    result = detect_chart_conflicts(dense_touch, TEMPO, 0, config())
    assert not [c for c in result.conflicts_detected if c.severity == "hard"]


def test_touch_lane_is_not_discarded():
    result = detect_chart_conflicts([P(F(0), "A4/1")], TEMPO, 0, config())
    assert not result.conflicts_detected
    result = detect_chart_conflicts([P(F(0), "A4/4")], TEMPO, 0, config())
    assert result.conflicts_detected[0].severity == "hard"
    assert result.conflicts_detected[0].notes[0].pad == "A4"


def test_hold_head_overlap_uses_soft_six_tick_tolerance():
    after_head = [P(F(0), "1h[4:2]"), P(F(1, 25), "1")]  # 20 ms after the hold head
    assert not detect_chart_conflicts(after_head, TEMPO, 0, config()).conflicts_detected

    enabled = detect_chart_conflicts(
        after_head, TEMPO, 0, config(detect_soft_conflicts=True))
    assert [(issue.conflict_type, issue.severity, issue.repair_target)
            for issue in enabled.conflicts_detected] == [("hold_tap_overlap", "soft", None)]

    before_head = [P(F(2), "1"), P(F(2) + F(1, 25), "1h[4:2]")]
    before = detect_chart_conflicts(before_head, TEMPO, 0, config())
    assert not before.conflicts_detected
    before_enabled = detect_chart_conflicts(
        before_head, TEMPO, 0, config(detect_soft_conflicts=True))
    assert len(before_enabled.conflicts_detected) == 1
    assert before_enabled.conflicts_detected[0].evidence == "static_hold_head_tolerance"
    assert before_enabled.conflicts_detected[0].severity == "soft"

    inside = [P(F(0), "1h[4:2]"), P(F(2, 25), "1")]  # 40 ms inside the hold
    interior = detect_chart_conflicts(inside, TEMPO, 0, config())
    assert len(interior.conflicts_detected) == 1
    assert interior.conflicts_detected[0].severity == "hard"

    same_beat = [P(F(0), "1h[4:2]/1")]
    exact = detect_chart_conflicts(same_beat, TEMPO, 0, config())
    assert any(issue.severity == "hard" for issue in exact.conflicts_detected)


def test_hold_hold_near_head_is_soft_and_local_repair_ignores_it():
    points = [P(F(0), "1h[4:2]"), P(F(1, 25), "1h[4:2]")]
    result = detect_chart_conflicts(points, TEMPO, 0, config(detect_soft_conflicts=True))
    assert len(result.conflicts_detected) == 1
    assert result.conflicts_detected[0].conflict_type == "same_pad_overlap"
    assert result.conflicts_detected[0].severity == "soft"

    repaired = process_chart_points(points, TEMPO, 0, config("local", detect_soft_conflicts=True))
    assert repaired.repairs_applied == []
    assert repaired.points == points
    assert all(issue.severity == "soft" for issue in repaired.remaining_conflicts)

    interior = detect_chart_conflicts(
        [P(F(0), "1h[4:2]"), P(F(2, 25), "1h[4:2]")], TEMPO, 0, config())
    assert len(interior.conflicts_detected) == 1
    assert interior.conflicts_detected[0].conflict_type == "same_pad_overlap"
    assert interior.conflicts_detected[0].severity == "hard"


def test_connected_touch_group_may_clear_conflicting_child():
    result = detect_chart_conflicts([P(F(0), "A1/B1/B2/C/1")], TEMPO, 0, config())
    assert result.conflicts_detected
    assert all(c.severity == "soft" for c in result.conflicts_detected)


def test_slide_wait_explicit_delay_and_bpm_boundary():
    tempo = [T(F(0), 120), T(F(1), 60)]
    events, _, _ = _extract([P(F(0), "1-5[4:1]")], tempo, -.25)
    trace = events[0].trajectories[0]
    assert trace.start_time == pytest.approx(.25)
    assert trace.duration == pytest.approx(1.0)
    explicit, _, _ = _extract([P(F(0), "1-5[0.2##0.8]")], tempo, -.25)
    assert explicit[0].trajectories[0].start_time == pytest.approx(-.05)
    assert explicit[0].trajectories[0].duration == pytest.approx(.8)


def test_simultaneous_slide_landing_is_not_automatically_hard():
    # Head at 0, trace starts at .5s and ends at 1.0s. A reference
    # early-input warning may remain; it is not proof of impossible Perfect.
    result = detect_chart_conflicts([P(F(0), "1-5[4:1]"), P(F(2), "5")], TEMPO, 0, config("local"))
    assert all(c.severity != "hard" for c in result.conflicts_detected)


def test_unknown_path_is_explicitly_partial_not_passed():
    result = detect_chart_conflicts([P(F(0), "1p5[4:1]")], TEMPO, 0, config())
    assert result.coverage["status"] == "partial"
    assert result.coverage["unsupported"]


def test_reject_mode_fails_on_unsupported_partial_path():
    result = detect_chart_conflicts(
        [P(F(0), "1p5[4:1]")], TEMPO, 0, config("reject"))
    assert result.success is False
    assert result.coverage["status"] == "partial"
    assert result.coverage["unsupported"]


def test_local_repair_skips_unsupported_partial_path():
    points = [P(F(0), "1p5[4:1]"), P(F(1), "1h[4:2]"), P(F(2), "1")]
    result = process_chart_points(points, TEMPO, 0, config("local"))
    assert result.points == points
    assert result.repairs_applied == []
    assert result.coverage["status"] == "partial"
    assert result.coverage["unsupported"]


def test_same_star_branches_count_one_head_but_explicit_duplicate_is_reported():
    branch = detect_chart_conflicts([P(F(0), "1-5[4:1]*-4[4:1]")], TEMPO, 0, config())
    assert not any(c.severity == "hard" for c in branch.conflicts_detected)
    duplicate = detect_chart_conflicts([P(F(0), "1/1-5[4:1]")], TEMPO, 0, config())
    assert any(c.conflict_type == "same_pad_overlap" for c in duplicate.conflicts_detected)


def test_local_repair_writes_mixed_chord_children_and_rechecks():
    points = [P(F(15), "1h[4:4]"), P(F(17), "12b/C/3-7[4:1]*-6[4:1]")]
    original = [p.to_dict() for p in points]
    result = process_chart_points(points, TEMPO, .1, config("local"))
    assert result.points[-1].note_text == "2b/C/3-7[4:1]*-6[4:1]"
    assert result.repairs_applied[0]["time_sec"] == pytest.approx(8.6)
    assert not any(c.severity == "hard" for c in result.remaining_conflicts)
    assert [p.to_dict() for p in points] == original


def test_no_repair_for_extected_partial_parse_or_protected_notes():
    points = [P(F(0), "1h[4:4]"), P(F(1), "1x")]
    assert process_chart_points(points, TEMPO, 0, config("local")).points == points
    points.append(P(F(2), "unknown"))
    assert process_chart_points(points, TEMPO, 0, config("local")).points == points


def test_cross_window_retry_keeps_history_and_preceding_hold():
    prior = [_DecodedWindow(EventSequence([F(15)], ["1h[4:4]"]), F(0), F(16), True)]
    calls = []
    def decode(attempt):
        calls.append(attempt)
        return EventSequence([F(1)], ["1" if attempt == 0 else "2"]), True, None, 0
    result = _select_playable_window(decode, prior, start=F(16), end=F(32), tempo_changes=TEMPO,
                                    first=0, config=config("reject", 2), seed=42)
    assert calls == [0, 1]
    assert result[0].note_texts == ["2"]
    assert result[4]["attempts"][1]["seed"] == 1000045
    assert prior[0].sequence.note_texts == ["1h[4:4]"]


def test_retry_off_and_detect_only_preserve_generated_events():
    calls = []
    def decode(attempt):
        calls.append(attempt)
        return EventSequence([F(0), F(1)], ["1h[4:4]", "1"]), True, None, 0
    result = _select_playable_window(decode, [], start=F(0), end=F(16), tempo_changes=TEMPO,
                                    first=0, config=config(), seed=42)
    assert calls == [0]
    assert result[0].note_texts == ["1h[4:4]", "1"]
    with pytest.raises(ValueError):
        config("none", 2)
