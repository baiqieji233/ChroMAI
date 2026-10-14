from fractions import Fraction as F

import pytest

from maichart.event_timing import EventTempo,seconds_at_beat
from maichart.form.generate import exact_candidate
from maichart.form.rhythm_snap import approximate_onsets


TEMPO=[EventTempo(F(0),185)]


def frame_records(times,tempo=TEMPO):
    return [dict(beat=str(exact_candidate(time,tempo,0)),
                 second=seconds_at_beat(exact_candidate(time,tempo,0),tempo,0),
                 source_second=time,score=.9) for time in times]


def test_uniform_sixteenths_recover_from_wrong_already_common_triplet_positions():
    original=frame_records([0,.1,.16,.26,.32])
    point,_=approximate_onsets(original,TEMPO,0,tolerance_ms=32,
        divisions=[4,8,12,16,24],mode='point')
    assert [F(item['beat']) for item in point] == [F(0),F(1,3),F(1,2),F(5,6),F(1)]
    fixed,report=approximate_onsets(original,TEMPO,0,tolerance_ms=32,
        divisions=[4,8,12,16,24],mode='sequence')
    assert [F(item['beat']) for item in fixed] == [F(i,4) for i in range(5)]
    assert report['common_reconsidered'] == 1
    assert report['uniform_runs'][0]['division'] == 16
    assert report['maximum_shift_ms'] <= 32
    assert len(fixed) == len(original)


@pytest.mark.parametrize('beats',[
    [F(i,3) for i in range(7)],
    [F(i,6) for i in range(7)],
    [F(0),F(1,3),F(1,2),F(5,6),F(1)],
    [F(0),F(3,8),F(1,2),F(7,8),F(1)],
])
def test_true_triplets_mixed_rhythms_and_dotted_patterns_are_not_forced_to_sixteenths(beats):
    records=[dict(beat=str(beat),second=seconds_at_beat(beat,TEMPO,0),
                  source_second=seconds_at_beat(beat,TEMPO,0),score=.9) for beat in beats]
    fixed,report=approximate_onsets(records,TEMPO,0,tolerance_ms=32,mode='sequence')
    assert [F(item['beat']) for item in fixed] == beats
    assert report['changed'] == 0


def test_sequence_respects_user_tolerance_and_rest_regions():
    records=frame_records([0,.1,.16,.26,.32])
    bounded,report=approximate_onsets(records,TEMPO,0,tolerance_ms=10,mode='sequence')
    assert all(abs(float(item['second'])-float(source['second'])) <= .010000001
               for item,source in zip(bounded,records))
    assert [F(item['beat']) for item in bounded] != [F(i,4) for i in range(5)]
    rested,_=approximate_onsets(records,TEMPO,0,tolerance_ms=32,mode='sequence',
                              rest_regions=[{'start_sec':.080,'end_sec':.090}])
    assert all(not .080 <= float(item['second']) < .090 for item in rested)


def test_sequence_never_moves_across_tempo_anchors_or_exceeds_minimum_spacing_budget():
    tempo=[EventTempo(F(0),185),EventTempo(F(1,2),240)]
    beats=[F(0),F(1,3),F(1,2),F(79,100),F(1)]
    records=[dict(beat=str(beat),second=seconds_at_beat(beat,tempo,0),score=.9) for beat in beats]
    fixed,_=approximate_onsets(records,tempo,0,tolerance_ms=32,mode='sequence')
    assert fixed[2]['beat'] == '1/2'
    assert all((F(item['beat']) < F(1,2)) == (beat < F(1,2))
               for item,beat in zip(fixed,beats))
    records=frame_records([0,.1,.16,.26,.32])
    fixed,report=approximate_onsets(records,TEMPO,0,tolerance_ms=32,
                                  min_interval_ms=90,mode='sequence')
    assert fixed == records
    assert report['uniform_runs_count'] == 0


def test_missing_onset_grid_run_recovers_two_shifted_points_from_20ms_frames():
    tempo=[EventTempo(F(0),220)]
    targets=[F(0),F(1,3),F(2,3),F(1),F(5,3),F(7,3),F(8,3),F(3)]
    current=[F(0),F(1,3),F(2,3),F(1),F(13,8),F(19,8),F(8,3),F(3)]
    raw=[round(float(seconds_at_beat(beat,tempo,0))/.02)*.02 for beat in targets]
    records=[dict(beat=str(beat),second=seconds_at_beat(beat,tempo,0),
                  source_second=source,score=.9)
             for beat,source in zip(current,raw)]
    fixed,report=approximate_onsets(records,tempo,0,tolerance_ms=40,mode='sequence')
    assert [F(item['beat']) for item in fixed] == targets
    assert report['changed'] == 2
    assert len(fixed) == len(records)
    assert all(any((F(item['beat'])*division/4).denominator == 1
                   for division in (4,8,12,16,24,32)) for item in records)
    assert report['uniform_runs'][0]['missed_grid_steps'] == 2


def test_safe_uniform_grid_repair_still_allows_a_single_interior_correction():
    tempo=[EventTempo(F(0),190)]
    targets=[F(0),F(1,4),F(1,2),F(3,4)]
    current=[F(0),F(1,4),F(11,24),F(3,4)]
    raw=[round(float(seconds_at_beat(beat,tempo,0))/.02)*.02 for beat in targets]
    records=[dict(beat=str(beat),second=seconds_at_beat(beat,tempo,0),
                  source_second=source,score=.9)
             for beat,source in zip(current,raw)]
    fixed,report=approximate_onsets(records,tempo,0,tolerance_ms=20,mode='sequence')
    assert [F(item['beat']) for item in fixed] == targets
    assert report['changed'] == 1


def test_half_note_spacing_keeps_uniform_two_step_repair_behavior():
    tempo=[EventTempo(F(0),300)]
    targets=[F(0),F(2),F(4),F(6),F(8)]
    current=[F(0),F(2),F(33,8),F(6),F(8)]
    raw=[round(float(seconds_at_beat(beat,tempo,0))/.02)*.02 for beat in targets]
    records=[dict(beat=str(beat),second=seconds_at_beat(beat,tempo,0),
                  source_second=source,score=.9)
             for beat,source in zip(current,raw)]
    fixed,report=approximate_onsets(records,tempo,0,tolerance_ms=40,mode='sequence')
    assert [F(item['beat']) for item in fixed] == targets
    assert report['changed'] == 1
    assert report['gapped_runs_count'] == 0


def test_uniform_shifted_eighth_phase_is_reported_and_left_unchanged():
    tempo=[EventTempo(F(0),190)]
    beats=[F(1,8)+i*F(1,2) for i in range(6)]
    records=[dict(beat=str(beat),second=seconds_at_beat(beat,tempo,0),
                  source_second=seconds_at_beat(beat,tempo,0),score=.9) for beat in beats]
    fixed,report=approximate_onsets(records,tempo,0,tolerance_ms=40,mode='sequence')
    assert [F(item['beat']) for item in fixed] == beats
    assert report['changed'] == 0
    assert report['ambiguous_phase_runs']
    assert report['ambiguous_phase_runs'][0]['division'] == 8
    assert report['ambiguous_phase_runs'][0]['phase_offset_beats'] == pytest.approx(.125)
    unchanged,disabled=approximate_onsets(records,tempo,0,tolerance_ms=0,mode='sequence')
    assert unchanged == records
    assert disabled['ambiguous_phase_runs'] == []


@pytest.mark.parametrize('beats',[
    [F(0),F(1,3),F(1,2),F(5,6),F(1),F(4,3),F(3,2)],
    [F(0),F(3,8),F(1,2),F(7,8),F(1),F(11,8),F(3,2),F(15,8),F(2)],
])
def test_exact_mixed_triplet_and_alternating_dotted_eighth_runs_are_preserved(beats):
    records=[dict(beat=str(beat),second=seconds_at_beat(beat,TEMPO,0),
                  source_second=seconds_at_beat(beat,TEMPO,0),score=.9) for beat in beats]
    fixed,report=approximate_onsets(records,TEMPO,0,tolerance_ms=32,mode='sequence')
    assert [F(item['beat']) for item in fixed] == beats
    assert report['changed'] == 0


def test_custom_fine_grids_and_zero_tolerance_remain_available():
    tempo=[EventTempo(F(0),60)]
    beats=[F(i,12) for i in range(6)]
    records=[dict(beat=str(beat),second=float(beat),source_second=float(beat),score=.9) for beat in beats]
    fixed,_=approximate_onsets(records,tempo,0,tolerance_ms=10,
                              divisions=[48,64],mode='sequence')
    assert [F(item['beat']) for item in fixed] == beats
    jitter=frame_records([0,.1,.16,.26,.32])
    fixed,report=approximate_onsets(jitter,TEMPO,0,tolerance_ms=0,mode='sequence')
    assert fixed == jitter and report['changed'] == 0
    with pytest.raises(ValueError,match='rhythm_snap_mode'):
        approximate_onsets([],TEMPO,0,mode='unknown')
