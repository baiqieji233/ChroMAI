from fractions import Fraction

from maichart.event_timing import EventPoint
from maichart.form.revision import (
    CausalSpan,
    EditPatch,
    RevisionAssessment,
    RevisionBudget,
    apply_patch_candidate,
    build_denoising_examples,
    causal_affected_range,
)


def _points():
    return [EventPoint(Fraction(i), str(i % 8 + 1)) for i in range(5)]


def test_patch_accepts_hard_conflict_reduction_and_keeps_original_unmodified():
    original = _points()
    patch = EditPatch(Fraction(2), Fraction(3), (EventPoint(Fraction(2), "3h[4:2]"),),
                      "shorten causal sustain", source_id="root-2")
    result = apply_patch_candidate(original, patch, lambda points: {
        "actionable_hard": 1 if points[2].note_text == "3h[4:2]" else 2,
        "syntax_errors": 0, "coverage": 1.0,
    })
    assert result.accepted
    assert result.points[2].note_text == "3h[4:2]"
    assert original[2].note_text == "3"
    assert result.points[0] == original[0]


def test_equal_hard_count_requires_external_quality_improvement():
    original = _points()
    patch = EditPatch(Fraction(1), Fraction(2), (EventPoint(Fraction(1), "2b"),), "candidate")
    rejected = apply_patch_candidate(original, patch, lambda _p: RevisionAssessment(2, quality_score=0.4))
    assert not rejected.accepted
    accepted = apply_patch_candidate(original, patch, lambda p: RevisionAssessment(
        2, quality_score=0.5 if p[1].note_text == "2b" else 0.4))
    assert accepted.accepted


def test_quality_gain_cannot_accept_a_hard_conflict_regression():
    original = _points()
    patch = EditPatch(Fraction(1), Fraction(2), (EventPoint(Fraction(1), "2b"),), "candidate")
    result = apply_patch_candidate(original, patch, lambda p: RevisionAssessment(
        3 if p[1].note_text == "2b" else 2,
        quality_score=0.9 if p[1].note_text == "2b" else 0.1))
    assert not result.accepted
    assert result.reason == "validation_regression"


def test_default_onset_preservation_rejects_added_point():
    original = _points()
    patch = EditPatch(Fraction(1), Fraction(3),
                      (EventPoint(Fraction(1), "2"), EventPoint(Fraction(3, 2), "5")), "add")
    result = apply_patch_candidate(original, patch, lambda _p: RevisionAssessment(0))
    assert not result.accepted
    assert result.reason == "onset_set_changed"


def test_patch_copies_points_rejects_duplicate_beats_and_preserves_chord_heads():
    supplied = EventPoint(Fraction(1), "2/5")
    patch = EditPatch(Fraction(1), Fraction(2), (supplied,), "copy")
    supplied.note_text = "8"
    assert patch.points[0].note_text == "2/5"
    try:
        EditPatch(Fraction(1), Fraction(2),
                  (EventPoint(Fraction(1), "2"), EventPoint(Fraction(1), "5")), "duplicate")
        assert False, "duplicate patch beats must fail"
    except ValueError:
        pass
    original = _points()
    chord = EventPoint(Fraction(1), "2/5")
    chart = [original[0], chord, *original[2:]]
    drop_head = EditPatch(Fraction(1), Fraction(2),
                          (EventPoint(Fraction(1), "2"),), "drop chord voice")
    result = apply_patch_candidate(chart, drop_head, lambda _p: RevisionAssessment(0))
    assert not result.accepted
    assert result.reason == "note_head_count_changed"


def test_budget_and_failed_candidate_cache_bound_validation_work():
    original = _points()
    patch = EditPatch(Fraction(1), Fraction(2), (EventPoint(Fraction(1), "2b"),), "candidate")
    calls = []
    result = apply_patch_candidate(original, patch, lambda _p: calls.append(1) or RevisionAssessment(1),
                                   budget=RevisionBudget(max_checks=1))
    assert not result.accepted and result.reason == "check_budget_exhausted"
    assert len(calls) == 1
    failed = set()
    first = apply_patch_candidate(original, patch, lambda _p: RevisionAssessment(2),
                                  failed_candidate_cache=failed)
    calls = []
    second = apply_patch_candidate(original, patch,
                                   lambda _p: calls.append(1) or RevisionAssessment(2),
                                   failed_candidate_cache=failed)
    assert not first.accepted
    assert second.reason == "cached_or_cyclic_candidate"
    assert len(calls) == 1  # Baseline is checked; the cached candidate is not.


def test_causal_range_finds_earlier_long_span_without_fixed_lookback():
    spans = [CausalSpan(Fraction(0), Fraction(96), 0, "old-root"),
             CausalSpan(Fraction(94), Fraction(96), 12, "nearby")]
    assert causal_affected_range(Fraction(95), Fraction(95), spans) == (0, 12)


def test_denoising_examples_are_seeded_and_keep_trusted_target_and_context():
    points = _points()
    a = build_denoising_examples(points, count=10, seed=41, clean_probability=0)
    b = build_denoising_examples(points, count=10, seed=41, clean_probability=0)
    assert a == b
    sample = a[0]
    assert sample.supervision == "synthetic_reconstruction"
    assert sample.masked_beats
    assert sample.target_region == tuple(points[len(sample.left_context):
        len(sample.left_context) + len(sample.target_region)])
    assert all(p.masked == (p.note_text is None) for p in sample.draft_region)


def test_denoising_builder_can_emit_explicit_clean_no_change_examples():
    points = _points()
    sample, = build_denoising_examples(points, count=1, seed=3, clean_probability=1)
    assert sample.clean
    assert not sample.masked_beats
    assert tuple(EventPoint(p.beat, p.note_text) for p in sample.target_region) == tuple(points)
    assert all(not p.masked for p in sample.draft_region)


def test_denoising_patch_boundary_ends_at_next_triplet_onset():
    points = [EventPoint(Fraction(0), "1"), EventPoint(Fraction(1, 3), "2"),
              EventPoint(Fraction(2, 3), "3"), EventPoint(Fraction(1), "4")]
    sample, = build_denoising_examples(points, count=1, seed=0,
                                       mask_probability=1, max_masks=1, clean_probability=0)
    assert sample.editable_end == points[len(sample.left_context) + len(sample.target_region)].beat
