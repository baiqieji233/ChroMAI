from fractions import Fraction

import pytest

from maichart.event_timing import EventPoint, EventTempo
from maichart.generation.post_process import PostProcessConfig, PostProcessResult
from maichart.form.repair import RepairBudget, repair_chart


TEMPO = (EventTempo(Fraction(0), 120.0),)


def test_whole_chart_repair_shortens_earlier_causal_hold_and_preserves_heads():
    original = [EventPoint(Fraction(0), "1h[4:4]"), EventPoint(Fraction(2), "1")]
    saved = tuple((point.beat, point.note_text) for point in original)
    result = repair_chart(
        original, TEMPO,
        config=PostProcessConfig(enable_conflict_detection=True, conflict_engine="static"),
        budget=RepairBudget(max_checks=8, max_edits=2, deadline_secs=4),
    )
    assert result.repairs
    assert result.repairs[0]["operation"] == "shorten_hold"
    assert result.repairs[0]["source_id"] == [0, 0]
    assert result.remaining_actionable_hard == 0
    assert result.status == "resolved_under_static_only"
    assert result.to_dict()["coverage_status"] == "static_only"
    assert result.to_dict()["original_coverage"]["status"] == "static_only"
    assert result.to_dict()["final_coverage"]["status"] == "static_only"
    assert tuple((point.beat, point.note_text) for point in original) == saved
    assert tuple(point.beat for point in result.points) == tuple(point.beat for point in original)
    assert result.checks <= 8


def test_clean_chart_returns_without_attempting_edits():
    points = [EventPoint(Fraction(0), "1"), EventPoint(Fraction(1), "5")]
    result = repair_chart(points, TEMPO,
                          config=PostProcessConfig(enable_conflict_detection=True))
    assert result.status == "resolved_under_static_only"
    assert result.repairs == ()
    assert result.checks == 1


def test_repair_budget_exhaustion_keeps_original_chart():
    points = [EventPoint(Fraction(0), "1h[4:4]"), EventPoint(Fraction(2), "1")]
    result = repair_chart(points, TEMPO,
                          config=PostProcessConfig(enable_conflict_detection=True),
                          budget=RepairBudget(max_checks=1, max_edits=1, deadline_secs=5))
    assert result.status == "budget_exhausted"
    assert result.repairs == ()
    assert result.points == tuple(points)


@pytest.mark.parametrize(("coverage", "expected", "expected_coverage"), [
    ({"status": "partial", "unsupported": [{"reason": "unmapped"}],
      "parse_errors": []}, "partial_coverage", "partial"),
    ({"status": "missing_engine", "unsupported": [], "parse_errors": []},
     "unverified", "unverified"),
])
def test_zero_findings_keep_partial_or_unknown_coverage_unresolved(
        monkeypatch, coverage, expected, expected_coverage):
    from maichart.form import repair as repair_module

    def detector(points, _tempo, _first, _config):
        return PostProcessResult(0, 0, [], [], True, points=list(points),
                                 coverage=coverage, remaining_conflicts=[])

    monkeypatch.setattr(repair_module, "detect_chart_conflicts", detector)
    result = repair_module.repair_chart(
        [EventPoint(Fraction(0), "1")], TEMPO,
        config=PostProcessConfig(enable_conflict_detection=True),
    )
    report = result.to_dict()
    assert result.remaining_actionable_hard == 0
    assert result.status == expected
    assert report["coverage_status"] == expected_coverage
    assert report["original_coverage"] == coverage
    assert report["final_coverage"] == coverage
