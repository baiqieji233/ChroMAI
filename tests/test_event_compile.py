from fractions import Fraction

import pytest

from maichart.event_timing import EventPoint, EventTempo, parse_event_chart_text
from maichart.generation.event_compile import compile_event_points


def test_compiler_selects_local_exact_grids_and_roundtrips_raw_note_groups():
    points = [
        EventPoint(Fraction(0), "1"),
        EventPoint(Fraction(1, 3), "3-7[8:1]*-5[4:1]/46"),
        EventPoint(Fraction(3, 4), "A1"),
        EventPoint(Fraction(1, 7), "2"),
    ]
    text, chart = compile_event_points(
        points,
        bpm=120,
        tempo_changes=[EventTempo(Fraction(0), 120), EventTempo(Fraction(1, 2), 150)],
        end_beat=Fraction(2),
        strict_notes=True,
    )
    reparsed = parse_event_chart_text(text, 5, strict_notes=True)
    assert [(point.beat, point.note_text) for point in reparsed.points] == [
        (point.beat, point.note_text) for point in sorted(points, key=lambda point: point.beat)
    ]
    assert "{28}" in text
    assert "{21}" in text
    assert "(150)" in text
    assert chart.notes


def test_compiler_rejects_two_event_points_at_one_beat_instead_of_creating_duplicate_heads():
    with pytest.raises(ValueError, match="one EventPoint per beat"):
        compile_event_points([EventPoint(Fraction(0), "1"), EventPoint(Fraction(0), "2")])


def test_compiler_writes_non_note_tempo_anchor_and_formats_at_bars():
    text, _ = compile_event_points(
        [EventPoint(Fraction(8), "1")],
        bpm=130,
        tempo_changes=[EventTempo(Fraction(0), 130), EventTempo(Fraction(4), 140)],
        end_beat=Fraction(9),
    )
    assert "(140)" in text
    assert "\n" in text
    reparsed = parse_event_chart_text(text, 5)
    assert [(point.beat, point.note_text) for point in reparsed.points] == [(Fraction(8), "1")]


def test_compiler_roundtrip_accepts_explicit_end_after_last_event():
    text, chart = compile_event_points(
        [EventPoint(Fraction(1), "1")],
        bpm=120,
        tempo_changes=[EventTempo(Fraction(0), 120)],
        end_beat=Fraction(4),
    )
    reparsed = parse_event_chart_text(text, 5)
    assert reparsed.end_beat == Fraction(4)
    assert chart.metadata is not None


@pytest.mark.parametrize("bpm", [92.54999999999788, 136.36363636363637, 174.63127, 0.00001])
@pytest.mark.parametrize("first", [0.0, 0.123456789, -0.123456789, 1e-7, -1e-7])
def test_compiler_roundtrips_full_bpm_and_offset_precision(bpm, first):
    points = [EventPoint(Fraction(0), "1"), EventPoint(Fraction(1, 3), "2h[4:1]")]
    tempos = [EventTempo(Fraction(0), bpm), EventTempo(Fraction(2, 3), 123.456789)]
    text, chart = compile_event_points(points, tempo_changes=tempos, first=first,
                                      end_beat=Fraction(4), strict_notes=True)
    parsed = parse_event_chart_text(text, 5, strict_notes=True)
    assert parsed.tempo_changes == tempos
    assert parsed.first == first
    assert parsed.points == points
    assert parsed.end_beat == Fraction(4)
    assert chart.metadata.offset == first
    assert [event.bpm for event in chart.timing.bpms] == [t.bpm for t in tempos]
    if first in (1e-7, -1e-7):
        assert f"&first={'-' if first < 0 else ''}0.0000001\n" in text
    if bpm == 0.00001:
        assert "(0.00001)" in text


def test_empty_sustain_prefix_compiles_with_unrounded_detected_bpm():
    tempos = [EventTempo(Fraction(0), 92.54999999999788)]
    text, _ = compile_event_points([], tempo_changes=tempos)
    assert parse_event_chart_text(text, 5).tempo_changes == tempos


def test_roundtrip_still_rejects_changed_tempo(monkeypatch):
    from maichart.generation import event_compile
    real_parse = event_compile.parse_event_chart_text

    def corrupt_tempo(*args, **kwargs):
        result = real_parse(*args, **kwargs)
        result.tempo_changes[0].bpm = 93.0
        return result

    monkeypatch.setattr(event_compile, "parse_event_chart_text", corrupt_tempo)
    with pytest.raises(ValueError, match="tempo_changes"):
        event_compile.compile_event_points([EventPoint(Fraction(0), "1")], bpm=92.55)
