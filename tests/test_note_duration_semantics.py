"""Cross-check public quality duration profiles against the reference parser."""
from fractions import Fraction as F

import pytest

from maichart.event_timing import EventPoint, EventTempo
from maichart.generation.event_quality import (
    LEGACY_DURATION_SEMANTICS, MIACODE_DURATION_SEMANTICS, assess_event_chart,
    condition_from_events, parse_event_notes,
)
from maichart.generation.muri.parser import parse_markers


@pytest.mark.parametrize('bpm,text', [
    (60, '1h[#1]'), (240, '1h[#1]'), (120, '1h[2:1]'),
    (120, '1h[#.0000001]'),
    (60, '1h[240#4:1]'), (120, 'Ch[240#4:1]'),
    (120, '1-5[4:1]'), (120, '1-5[240#4:1]'),
    (120, '1-5[0.2##0.7]'), (120, '1-5[0.2##240#4:1]'),
    (120, '1-5[4:1]*-7[8:1]'),
])
def test_head_bpm_duration_profile_agrees_with_reference(bpm, text):
    tempo = [EventTempo(F(0), bpm), EventTempo(F(1), bpm * 2)]
    parsed = parse_event_notes([EventPoint(F(0), text)], tempo, first=2,
                               duration_semantics=MIACODE_DURATION_SEMANTICS)
    assert parsed.syntax_valid, parsed.fatal_errors
    markers, errors, _ = parse_markers(f'({bpm}){{4}}{text},', first=2)
    assert not errors
    note = parsed.notes[0]
    assert note.raw == text
    trace = [m['end_second'] - m['slide_trace_second']
             if m['type'] in {'slide', 'wifi'} else m['end_second'] - m['second']
             for m in markers]
    assert note.duration_sec == pytest.approx(max(trace))
    if note.slide_segments:
        assert [part['duration_sec'] for part in note.slide_segments] == pytest.approx(trace)


def test_legacy_and_new_duration_profiles_are_explicitly_different():
    tempo = [EventTempo(F(0), 120), EventTempo(F(1), 240)]
    points = [EventPoint(F(0), '1h[2:1]/2-6[2:1]')]
    legacy = parse_event_notes(points, tempo)
    reference = parse_event_notes(points, tempo, duration_semantics=MIACODE_DURATION_SEMANTICS)
    assert [note.duration_sec for note in legacy.notes] == pytest.approx([0.75, 0.75])
    assert [note.duration_sec for note in reference.notes] == pytest.approx([1, 1])
    assert reference.notes[0].duration.raw == '[2:1]'
    assert reference.notes[1].slide_segments[0]['duration'].raw == '[2:1]'


def test_slide_extent_includes_wait_only_in_reference_profile():
    points = [EventPoint(F(0), '1-5[4:1]')]
    tempo = [EventTempo(F(0), 120)]
    legacy = assess_event_chart(points, tempo)
    reference = assess_event_chart(points, tempo, duration_semantics=MIACODE_DURATION_SEMANTICS)
    assert legacy['duration_sec'] == pytest.approx(0.5)
    assert reference['duration_sec'] == pytest.approx(1.0)
    assert reference['slides']['branch_durations_sec'] == pytest.approx([0.5])
    assert reference['control']['actual']['metadata']['duration_semantics_profile'] == MIACODE_DURATION_SEMANTICS


def test_condition_reports_duration_semantics_and_preserves_legacy_default():
    points = [EventPoint(F(0), '1-5[4:1]')]
    tempo = [EventTempo(F(0), 120)]
    legacy = condition_from_events(points, tempo, level=13, duration_sec=10)
    current = condition_from_events(points, tempo, level=13, duration_sec=10,
                                    duration_semantics=MIACODE_DURATION_SEMANTICS)
    assert legacy.metadata['duration_semantics_profile'] == LEGACY_DURATION_SEMANTICS
    assert current.metadata['duration_semantics_profile'] == MIACODE_DURATION_SEMANTICS


def test_parallel_slide_raw_metadata_is_not_replaced_by_canonical_seconds():
    text = '1-5[240#4:1]*-7[0.2##0.7]/2h[90#8:1]'
    parsed = parse_event_notes([EventPoint(F(0), text)], [EventTempo(F(0), 120)],
                               duration_semantics=MIACODE_DURATION_SEMANTICS)
    assert parsed.syntax_valid
    assert parsed.points[0].note_text == text
    slide, hold = parsed.notes
    assert slide.raw == text.split('/')[0]
    assert [part['duration'].raw for part in slide.slide_segments] == ['[240#4:1]', '[0.2##0.7]']
    assert [part['raw'] for part in slide.slide_segments] == ['1-5[240#4:1]', '-7[0.2##0.7]']
    assert hold.raw == '2h[90#8:1]'


def test_unknown_duration_profile_is_rejected():
    with pytest.raises(ValueError, match='duration_semantics'):
        parse_event_notes([], [EventTempo(F(0), 120)], duration_semantics='guess')
