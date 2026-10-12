from fractions import Fraction

import numpy as np
import pytest

from maichart.event_timing import EventTempo, seconds_at_beat
from maichart.form.timing import (
    MeterChange,
    build_spans,
    metrical_features,
    normalize_meter_map,
)


def test_quarter_note_clock_preserves_triplets_and_uncommon_subdivisions():
    meters = [MeterChange(Fraction(0), 3, 4)]
    spans = build_spans(24, meters)
    positions = [Fraction(2, 3), Fraction(1, 6), Fraction(1, 12), Fraction(1, 24)]

    assert [(span.start, span.end) for span in spans] == [(Fraction(0), Fraction(12)), (Fraction(12), Fraction(24))]
    assert all(any(span.start <= beat < span.end for span in spans) for beat in positions)
    assert [span.beats_per_bar for span in spans] == [Fraction(3), Fraction(3)]
    assert type(spans[0]).from_dict(spans[0].to_dict()) == spans[0]


@pytest.mark.parametrize(
    ("numerator", "denominator", "grouping", "expected_bar"),
    [(3, 4, (3,), Fraction(3)), (6, 8, (3, 3), Fraction(3)), (7, 8, (2, 2, 3), Fraction(7, 2))],
)
def test_simple_compound_and_odd_meter_lengths(numerator, denominator, grouping, expected_bar):
    spans = build_spans(24, [MeterChange(Fraction(0), numerator, denominator, grouping)])

    assert spans[0].beats_per_bar == expected_bar
    assert spans[0].grouping == grouping


def test_meter_changes_split_midbar_and_restart_the_new_bar_phase():
    meters = [
        MeterChange(Fraction(0), 4, 4),
        MeterChange(Fraction(6), 3, 4, (3,)),
    ]
    spans = build_spans(18, meters)

    assert spans[0].start == 0 and spans[0].end == 6 and spans[0].partial
    assert spans[1].start == 6 and spans[1].end == 18
    assert spans[1].bar_boundaries == (Fraction(6), Fraction(9), Fraction(12), Fraction(15), Fraction(18))
    assert not spans[1].partial


def test_phrase_boundaries_split_without_changing_bar_phase():
    spans = build_spans(
        24,
        [MeterChange(Fraction(0), 4, 4)],
        phrase_boundaries=[Fraction(6)],
    )

    assert [(span.start, span.end, span.bar_boundaries, span.partial) for span in spans] == [
        (Fraction(0), Fraction(6), (Fraction(0), Fraction(4)), True),
        (Fraction(6), Fraction(16), (Fraction(8), Fraction(12), Fraction(16)), True),
        (Fraction(16), Fraction(24), (Fraction(16), Fraction(20), Fraction(24)), False),
    ]


def test_missing_meter_uses_unknown_fallback_without_bar_phase():
    spans = build_spans(35)
    features = metrical_features([Fraction(2, 3), Fraction(1, 3)], None)

    assert [(span.start, span.end) for span in spans] == [
        (Fraction(0), Fraction(16)),
        (Fraction(16), Fraction(32)),
        (Fraction(32), Fraction(35)),
    ]
    assert all(not span.known and not span.bar_boundaries for span in spans)
    assert spans[-1].partial and spans[-1].confidence == 0
    assert features.shape == (2, 8)
    assert np.array_equal(features[:, 2:8], np.zeros((2, 6), dtype=np.float32))


def test_tempo_changes_do_not_change_meter_spans_or_features():
    meters = [MeterChange(Fraction(0), 6, 8, (3, 3))]
    spans_before = build_spans(18, meters)
    features_before = metrical_features([Fraction(2, 3), Fraction(3, 2), Fraction(3)], meters)
    seconds = seconds_at_beat(
        Fraction(3), [EventTempo(Fraction(0), 120), EventTempo(Fraction(1), 60)]
    )
    spans_after = build_spans(18, meters)
    features_after = metrical_features([Fraction(2, 3), Fraction(3, 2), Fraction(3)], meters)

    assert seconds == 2.5
    assert spans_after == spans_before
    assert np.array_equal(features_after, features_before)


@pytest.mark.parametrize(
    "meters",
    [
        [MeterChange(Fraction(1), 4, 4)],
        [MeterChange(Fraction(0), 4, 4), MeterChange(Fraction(2), 3, 4), MeterChange(Fraction(2), 7, 8)],
    ],
)
def test_meter_map_requires_zero_origin_and_strictly_increasing_changes(meters):
    with pytest.raises(ValueError):
        normalize_meter_map(meters)


def test_meter_validation_and_rational_json_roundtrip():
    with pytest.raises(ValueError, match="sum"):
        MeterChange(Fraction(0), 7, 8, (3, 3))
    original = MeterChange(Fraction(1, 3), 7, 8, (2, 2, 3), 0.75, "analysis")
    assert MeterChange.from_dict(original.to_dict()) == original
    with pytest.raises(ValueError, match="begin at beat 0"):
        normalize_meter_map([original])


def test_features_encode_bar_group_and_unknown_flags():
    meter = [MeterChange(Fraction(0), 6, 8, (3, 3))]
    features = metrical_features([Fraction(0), Fraction(3, 2), Fraction(3)], meter)

    assert features.shape == (3, 8)
    assert np.all(features[:, 6:] == 1)
    assert features[0, 3] == pytest.approx(1)
    assert features[1, 3] == pytest.approx(-1)
    assert features[2, 3] == pytest.approx(1)
