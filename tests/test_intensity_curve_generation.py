from fractions import Fraction

from maichart.event_timing import EventTempo
from maichart.generation.event_cli import _curve_filter_sequence
from maichart.generation.event_model import EventSequence


def _profile(values, *, duration=8.0):
    points = [{"time_sec": float(time), "intensity": float(value)} for time, value in values]
    return {
        "schema": "maichart-intensity-curve-v1",
        "duration_sec": float(duration),
        "points": points,
    }


def _filter(sequence, profile, *, base_density=0.5, start=0.0, end=8.0, bpm=120):
    return _curve_filter_sequence(
        sequence,
        start_beat=Fraction(0),
        tempo_changes=[EventTempo(Fraction(0), bpm)],
        first=0.0,
        curve_profile=profile,
        base_density=base_density,
        window_start_sec=start,
        window_end_sec=end,
    )


def test_zero_intensity_is_an_explicit_rest():
    sequence = EventSequence([Fraction(i) for i in range(8)], ["1"] * 8)
    filtered, stats = _filter(sequence, _profile([(0, 0), (8, 0)]))

    assert filtered.beats == []
    assert filtered.note_texts == []
    assert len(stats["local_bins"]) == 4
    assert all(item["actual_heads"] == 0 and item["desired_heads"] == 0
               for item in stats["local_bins"])


def test_zero_regions_inside_mixed_bin_are_removed_at_their_exact_times():
    sequence = EventSequence([Fraction(i) for i in range(8)], ["1"] * 8)
    profile = _profile([(0, 0), (1, 0), (2, 0.5), (8, 0.5)])
    filtered, stats = _filter(sequence, profile)

    # At 120 BPM the first three event cells are at 0, 0.5 and 1.0 seconds.
    assert all(float(beat) * 0.5 >= 1.0 for beat in filtered.beats)
    assert stats["local_bins"][0]["removed"] >= 3
    assert stats["local_bins"][0]["raw_heads"] > stats["local_bins"][0]["actual_heads"]


def test_partial_last_bin_uses_its_real_duration_and_empty_bins_are_reported():
    sequence = EventSequence([Fraction(0), Fraction(2)], ["1", "2"])
    profile = _profile([(0, 0.5), (5, 0.5)], duration=5.0)
    _, stats = _filter(sequence, profile, start=0.5, end=5.0)

    assert [(item["start_sec"], item["end_sec"]) for item in stats["local_bins"]] == [
        (0.5, 2.0), (2.0, 4.0), (4.0, 5.0),
    ]
    assert stats["local_bins"][1]["actual_heads"] == 0
    assert stats["local_bins"][2]["desired_heads"] == 8.0


def test_density_guard_keeps_whole_chord_cells():
    sequence = EventSequence(
        [Fraction(i, 2) for i in range(8)],
        ["1/2", "3/4", "5/6", "7/8", "12", "34", "56", "78"],
    )
    filtered, stats = _filter(
        sequence, _profile([(0, 0.5), (8, 0.5)]), base_density=0.1, end=2.0,
    )

    assert all("/" in text for text in filtered.note_texts)
    assert all(stats["local_bins"][0]["actual_heads"] <=
               stats["local_bins"][0]["desired_heads"] * 1.15 for _ in [0])


def test_final_curve_counts_use_earlier_window_ownership_and_real_bin_boundaries():
    from maichart.event_timing import EventPoint
    from maichart.generation.event_cli import _DecodedWindow, _final_curve_bins

    # First window ends at 3s, second overlaps 2..3s and extends to 5s.
    # A different chord prediction in the overlap belongs to the earlier one.
    first = _DecodedWindow(EventSequence([], []), Fraction(0), Fraction(6), True,
                           {"raw_cells": [{"time_sec": 2.5, "heads": 1}]})
    second = _DecodedWindow(EventSequence([], []), Fraction(4), Fraction(10), True,
                            {"raw_cells": [{"time_sec": 2.5, "heads": 3},
                                           {"time_sec": 3.5, "heads": 2},
                                           {"time_sec": 3.9999998, "heads": 1},
                                           {"time_sec": 4.0, "heads": 1}]})
    bins = _final_curve_bins(
        [EventPoint(Fraction(5), "1"), EventPoint(Fraction(7), "1/2")],
        _profile([(0, 0.5), (5, 0.5)], duration=5),
        tempo_changes=[EventTempo(Fraction(0), 120)], first=0, base_density=0.5,
        windows=[first, second],
    )
    assert bins[1]["raw_heads"] == 4
    assert bins[1]["actual_heads"] == 3
    assert bins[1]["removed"] == 1
    assert bins[2]["raw_heads"] == 1


def test_curve_planner_controls_preserve_configuration_logits():
    import torch
    from maichart.generation.event_model import EventConditionedTransformer, EventTransformerConfig

    model = EventConditionedTransformer(EventTransformerConfig(
        input_dim=8, condition_dim=16, vocab_size=16, d_model=8, nhead=2,
        num_layers=1, dim_feedforward=16, dropout=0,
    ))
    memory = torch.randn(1, 5, 8)
    _, _, predicted = model.append_plan_memory(memory, None)
    _, _, controlled = model.append_plan_memory(memory, None, plan_controls={"count": 0.2, "density": 0.3})
    assert torch.allclose(controlled[:, 2:], predicted[:, 2:])
    assert torch.allclose(controlled[:, :2].sigmoid(), torch.tensor([[0.2, 0.3]]))


def test_soft_curve_strength_uses_one_effective_profile_for_all_consumers():
    import pytest
    from maichart.generation.event_cli import _effective_curve_profile
    profile = _profile([(0, 0), (8, 1)])
    soft = _effective_curve_profile(profile, strength=.5, mode="soft", span=.5)
    assert soft["points"][0]["intensity"] == pytest.approx(.375)
    assert soft["points"][1]["intensity"] == pytest.approx(.625)
    bypass = _effective_curve_profile(profile, strength=0, mode="strict")
    assert all(point["intensity"] == .5 for point in bypass["points"])


def test_soft_curve_records_density_but_preserves_every_generated_cell():
    sequence = EventSequence([Fraction(i, 2) for i in range(8)], ["1/2"] * 8)
    filtered, stats = _curve_filter_sequence(
        sequence, start_beat=Fraction(0), tempo_changes=[EventTempo(Fraction(0), 120)],
        first=0, curve_profile=_profile([(0, 0), (8, 0)]), base_density=.1,
        window_start_sec=0, window_end_sec=2, apply_guard=False)
    assert filtered == sequence
    assert stats["removed"] == 0
    assert stats["local_bins"][0]["actual_heads"] > stats["local_bins"][0]["desired_heads"]
