from maichart.generation import event_cli


def _quality(*, syntax_valid=True, coverage="engine_reference", actionable_hard=0,
             details=None, score_marker=0.0):
    quality = {
        "syntax_valid": syntax_valid,
        "note_warning_count": 0,
        "audio_follow": {"rms_density_correlation": score_marker},
    }
    if coverage is not None:
        playability = {"coverage": {"status": coverage, "parse_errors": [], "unsupported": []}}
        if actionable_hard is not None:
            playability["remaining_actionable_hard"] = actionable_hard
        if details is not None:
            playability["remaining_details"] = details
        quality["playability"] = playability
    return quality


def test_no_actionable_hard_ranks_above_many_even_if_musical_score_is_higher(monkeypatch):
    monkeypatch.setattr(event_cli, "_candidate_quality_score", lambda _quality: 1_000_000.0)
    clean_key, _ = event_cli._candidate_quality_rank(_quality(actionable_hard=0))
    hard_key, _ = event_cli._candidate_quality_rank(_quality(actionable_hard=10))

    assert clean_key > hard_key


def test_complete_coverage_precedes_partial_coverage_and_missing_detector():
    complete_key, _ = event_cli._candidate_quality_rank(
        _quality(coverage="engine_reference", actionable_hard=4))
    partial_key, _ = event_cli._candidate_quality_rank(
        _quality(coverage="partial", actionable_hard=0, score_marker=1.0))
    legacy_key, legacy_breakdown = event_cli._candidate_quality_rank(
        _quality(coverage=None, actionable_hard=None, score_marker=1.0))

    assert complete_key > partial_key > legacy_key
    assert legacy_breakdown["remaining_actionable_hard"] is None
    assert legacy_breakdown["actionable_hard_count_known"] is False


def test_missing_coverage_evidence_is_not_a_complete_pass():
    complete = _quality(coverage="engine_reference", actionable_hard=0)
    incomplete = _quality(coverage="engine_reference", actionable_hard=0)
    incomplete["playability"]["coverage"].pop("unsupported")

    complete_key, _ = event_cli._candidate_quality_rank(complete)
    incomplete_key, breakdown = event_cli._candidate_quality_rank(incomplete)

    assert complete_key > incomplete_key
    assert breakdown["detection_coverage"] == "partial"


def test_report_only_muri_does_not_reduce_actionable_rank():
    report_only = _quality(actionable_hard=0)
    report_only["playability"]["remaining_hard"] = 1
    report_only["playability"]["remaining_details"] = [
        {"severity": "hard", "action": "report"},
    ]
    clean = _quality(actionable_hard=0)

    report_key, report_breakdown = event_cli._candidate_quality_rank(report_only)
    clean_key, _ = event_cli._candidate_quality_rank(clean)

    assert report_breakdown["remaining_actionable_hard"] == 0
    assert report_key == clean_key


def test_actionable_count_can_be_recovered_only_from_policy_labeled_details():
    quality = _quality(actionable_hard=None, details=[
        {"severity": "hard", "action": "enforce"},
        {"severity": "hard", "action": "report"},
        {"severity": "warning", "action": "enforce"},
        {"severity": "hard"},
    ])

    _, breakdown = event_cli._candidate_quality_rank(quality)

    assert breakdown["actionable_hard_count_known"] is False
    assert breakdown["remaining_actionable_hard"] is None


def test_parse_failure_ranks_below_valid_chart():
    valid_key, _ = event_cli._candidate_quality_rank(_quality(syntax_valid=True))
    invalid_key, invalid_breakdown = event_cli._candidate_quality_rank(
        _quality(syntax_valid=False))

    assert valid_key > invalid_key
    assert invalid_breakdown["parser_coverage"] == "invalid"


def test_candidate_quality_float_remains_compatible_with_existing_score():
    quality = _quality(actionable_hard=7, score_marker=0.25)

    key, breakdown = event_cli._candidate_quality_rank(quality)

    assert event_cli._candidate_quality_score(quality) == 0.25
    assert breakdown["musical_quality_score"] == 0.25
    assert key[-1] == 0.25
