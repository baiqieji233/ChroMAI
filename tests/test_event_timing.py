from fractions import Fraction

import pytest

from maichart.event_timing import (
    EventChart,
    EventPoint,
    EventTempo,
    encode_event_points,
    joint_tokens_to_event_points,
    parse_event_chart_text,
    seconds_at_beat,
)


def test_event_timing_keeps_triplet_and_uncommon_fraction_without_tick_quantization():
    chart = parse_event_chart_text(
        "&title=T\n&wholebpm=120\n&lv_5=13+\n"
        "&inote_5=(120){12}1,,2,{28}3,E\n",
        5,
    )
    assert [point.beat for point in chart.points] == [Fraction(0), Fraction(2, 3), Fraction(1)]
    assert chart.points[-1].note_text == "3"
    assert chart.points[-1].beat * 1920 == Fraction(1920)
    assert chart.end_beat == Fraction(8, 7)


def test_event_timing_retains_compound_slides_and_tempo_at_empty_point():
    chart = parse_event_chart_text(
        "&wholebpm=120\n&lv_5=13\n"
        "&inote_5=(120){12}1-4[8:1]*-6[4:1],,,(150){16},2,E\n",
        5,
    )
    assert chart.points[0].note_text == "1-4[8:1]*-6[4:1]"
    assert chart.points[1].note_text == "2"
    assert [(change.beat, change.bpm) for change in chart.tempo_changes] == [(Fraction(0), 120.0), (Fraction(1), 150.0)]


def test_seconds_at_beat_integrates_exact_tempo_boundaries():
    assert seconds_at_beat(
        Fraction(3, 2),
        [EventTempo(Fraction(0), 120), EventTempo(Fraction(1), 60)],
        first=-0.25,
    ) == 0.75


def test_event_chart_serialization_and_joint_char_tokens_keep_exact_data():
    original = EventChart(
        points=[EventPoint(Fraction(1, 7), "3-7[8:1]*-5[4:1]/A1")],
        tempo_changes=[EventTempo(Fraction(0), 120), EventTempo(Fraction(1, 7), 150)],
        first=-0.125,
        level="13+",
        title="Exact",
        end_beat=Fraction(8, 7),
        artist="Composer",
        difficulty=5,
    )
    assert EventChart.from_dict(original.to_dict()) == original
    tokens = encode_event_points(original.points, local_start=Fraction(0))
    assert joint_tokens_to_event_points(tokens) == original.points


def test_strict_event_parser_rejects_duplicate_head_as_whole_document():
    with pytest.raises(ValueError, match="rejecting entire chart"):
        parse_event_chart_text("&lv_5=13\n&inote_5=1/1,E\n", 5, strict_notes=True)
