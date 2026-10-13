from fractions import Fraction as F

import pytest

from maichart.event_timing import EventTempo, EventPoint, seconds_at_beat, parse_event_chart_text
from maichart.form.rhythm_snap import approximate_onsets, normalize_divisions
from maichart.generation.event_compile import compile_event_points


TEMPO = [EventTempo(F(0),120)]


def candidates(beats,tempo=TEMPO,first=0):
    return [{'beat':str(beat),'second':seconds_at_beat(beat,tempo,first),'score':.9} for beat in beats]


def test_snap_near_binary_and_triplet_grids_without_rounding_far_odd_rhythm():
    beats = [F(251,1000),F(667,2000),F(21,50),F(753,1000)]
    source = candidates(beats)
    result, report = approximate_onsets(source,TEMPO,0,tolerance_ms=15)
    assert [F(item['beat']) for item in result] == [F(1,4),F(1,3),F(21,50),F(3,4)]
    assert report['changed'] == 3 and report['outside_tolerance'] == 1
    assert report['maximum_shift_ms'] <= 15
    assert [F(item['beat']) for item in source] == beats


def test_common_grids_are_stable_and_zero_tolerance_does_not_move_notes():
    source = candidates([F(0),F(1,8),F(1,6),F(1,4),F(1,3),F(1,2)])
    result, report = approximate_onsets(source,TEMPO,0)
    assert result == source and report['already_common'] == len(source)
    source = candidates([F(251,1000)])
    result, report = approximate_onsets(source,TEMPO,0,tolerance_ms=0)
    assert result == source and report['changed'] == 0


def test_snap_does_not_collapse_two_separate_events_or_violate_minimum_spacing():
    source = candidates([F(249,1000),F(251,1000)])
    result, _ = approximate_onsets(source,TEMPO,0,tolerance_ms=5)
    assert len(result) == 2 and F(result[0]['beat']) < F(result[1]['beat'])
    source = candidates([F(251,1000),F(331,1000)])
    result, report = approximate_onsets(source,TEMPO,0,tolerance_ms=15,min_interval_ms=40)
    assert result[1]['second']-result[0]['second'] >= .04-1e-9


def test_snap_uses_piecewise_tempo_and_preserves_boundary_and_rest_regions():
    tempo = [EventTempo(F(0),120),EventTempo(F(1),240)]
    source = candidates([F(999,1000),F(1),F(5,4)+F(1,100)],tempo,first=.2)
    result, report = approximate_onsets(source,tempo,.2,tolerance_ms=10)
    assert result[0]['beat'] == source[0]['beat']
    assert result[1]['beat'] == '1'
    assert result[2]['beat'] == '5/4'
    assert report['maximum_shift_ms'] == pytest.approx(2.5)
    source = candidates([F(251,1000)])
    result, report = approximate_onsets(source,TEMPO,0,rest_regions=[{'start_sec':.124,'end_sec':.1254}])
    assert result == source and report['blocked'] == 1


def test_snap_checks_raw_peak_tolerance_and_does_not_move_past_audio_end():
    source = candidates([F(251,1000)])
    source[0]['source_second'] = .145
    result, report = approximate_onsets(source,TEMPO,0,tolerance_ms=15)
    assert result == source and report['blocked'] == 1
    result, _ = approximate_onsets(candidates([F(499,1000)]),TEMPO,0,tolerance_ms=5,end_beat=F(1,2))
    assert result[0]['beat'] == '499/1000'
    for bad in (-1,51,float('inf')):
        with pytest.raises(ValueError,match='rhythm_snap_ms'):
            approximate_onsets([],TEMPO,0,tolerance_ms=bad)


def test_example_is_approximated_by_event_timing_not_text_replacement():
    text = ('&wholebpm=180\n&lv_5=13\n&inote_5=(180)'
            '{192}2/4,{96},{48},{12},{320}4/5,{160},{8},{160}2,{8},{160}4/5,{80},{10},4/5,E\n')
    original = parse_event_chart_text(text,5)
    selected, report = approximate_onsets(candidates([p.beat for p in original.points],original.tempo_changes),
        original.tempo_changes,0,tolerance_ms=15,end_beat=original.end_beat)
    snapped = [EventPoint(F(item['beat']),point.note_text) for item,point in zip(selected,original.points)]
    assert len(snapped) == len(original.points)
    assert [p.note_text for p in snapped] == [p.note_text for p in original.points]
    assert all(abs(seconds_at_beat(p.beat,original.tempo_changes,0)-
                   seconds_at_beat(q.beat,original.tempo_changes,0)) <= .015+1e-9
               for p,q in zip(snapped,original.points))
    compiled,_ = compile_event_points(snapped,tempo_changes=original.tempo_changes,end_beat=original.end_beat)
    assert parse_event_chart_text(compiled,5).points == snapped
    assert report['changed'] > 0


def test_low_bpm_can_use_custom_48_64_and_explicit_unusual_divisions():
    tempo = [EventTempo(F(0),60)]
    beats = [F(1,12)+F(1,1000),F(3,16)+F(1,1000)]
    source = candidates(beats,tempo)
    unchanged, _ = approximate_onsets(source,tempo,0,tolerance_ms=10)
    assert unchanged == source
    kept, report = approximate_onsets(source,tempo,0,tolerance_ms=10,divisions=[48,64])
    assert [F(item['beat']) for item in kept] == [F(1,12),F(3,16)]
    assert report['divisions'] == [48,64]
    custom, _ = approximate_onsets(candidates([F(201,1000)],tempo),tempo,0,tolerance_ms=5,divisions=[20])
    assert custom[0]['beat'] == '1/5'
    assert normalize_divisions('64,48,64') == (48,64)
    assert normalize_divisions([64]*40) == (64,)
    for bad in ([],[True],[16.5],'12.0,16','-4', [769]):
        with pytest.raises(ValueError):
            normalize_divisions(bad)
