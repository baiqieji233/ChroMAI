from fractions import Fraction as F
from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch

from maichart.event_timing import EventPoint, EventTempo, parse_event_chart
from maichart.tempo_clock import TempoClock
from maichart.composer_v2.stream_codec import StreamCodec
from maichart.composer_v2.generate import _new_guard, _check_window
from maichart.composer_v2.release_repair import repair_hold_release, repair_hold_conflict, _elapsed_at_head_bpm


def fixture():
    tempo=[EventTempo(F(0),157.)]
    points=[EventPoint(F(134),'3h[16:3]/7h[16:3]'),EventPoint(F(541,4),'1h[16:3]/8h[16:3]')]
    witness=EventPoint(F(136),'2h[16:3]/6h[16:3]')
    codec=StreamCodec.fit([p.note_text for p in [*points,witness]])
    return points,witness,tempo,TempoClock(tempo),codec


def rejected(points,witness,tempo,clock):
    guard=_new_guard(tempo,clock.seconds[0],100.)
    for point in points:
        assert guard.accepts(point)
        guard.commit(point)
    details=[]
    assert _check_window(guard,[witness],details)[0] is None
    return [(witness,details)],guard


def test_reported_double_hold_repaired_atomically_with_minimal_tested_tail_change():
    points,witness,tempo,clock,codec=fixture()
    failures,original_guard=rejected(points,witness,tempo,clock)
    original_prefix=list(original_guard.prefix)
    fixed,continued,guard,report=repair_hold_release(points,failures,tempo,clock,100.,codec)
    assert fixed[-1]==EventPoint(F(541,4),'1h[96:17]/8h[96:17]')
    assert fixed[:-1]==points[:-1] and continued==witness
    assert points[-1].note_text=='1h[16:3]/8h[16:3]'
    assert original_guard.prefix==original_prefix
    assert report['duration_reduction_beats']=='1/24'
    assert report['shortened_atom_indices']==[0,1]
    assert codec.decode_points(codec.encode_points(fixed+[continued]))==fixed+[continued]
    replay=_new_guard(tempo,0.,100.)
    for point in fixed+[continued]:
        assert replay.accepts(point)
        replay.commit(point)
    assert replay.prefix==guard.prefix
    assert replay.last_head_seconds==guard.last_head_seconds


def test_tempo_change_uses_duration_at_head_bpm_not_chart_beat_subtraction():
    tempo=[EventTempo(F(0),120.),EventTempo(F(2),240.)]
    clock=TempoClock(tempo,.123)
    points=[EventPoint(F(1),'1h[4:2]/8h[4:2]')]
    witness=EventPoint(F(4),'2h[4:1]/6h[4:1]')
    codec=StreamCodec.fit([p.note_text for p in [*points,witness]])
    assert _elapsed_at_head_bpm(clock,F(1),F(4))==2
    failures,_=rejected(points,witness,tempo,clock)
    fixed,continued,guard,report=repair_hold_release(points,failures,tempo,clock,100.,codec)
    assert continued==witness and fixed[0].beat==1
    assert F(report['duration_reduction_beats'])<=F(1,16)


def test_native_anchor_is_not_the_full_list_of_contributing_holds():
    tempo=[EventTempo(F(0),157.)];clock=TempoClock(tempo)
    points=[EventPoint(F(185),'8h[4:11]'),EventPoint(F(187),'1h[4:1]')]
    witness=EventPoint(F(188),'4')
    codec=StreamCodec.fit([p.note_text for p in points]+['4'])
    failures,_=rejected(points,witness,tempo,clock)
    assert {b for item in failures[0][1] for b in item['note_beats']}=={'185'}
    fixed,continued,guard,report=repair_hold_release(points,failures,tempo,clock,100.,codec)
    assert fixed[0]==points[0]  # Do not delete/shorten the unrelated long tail.
    assert fixed[1]==EventPoint(F(187),'1h[96:23]')
    assert report['source_beat']=='187' and report['reported_source_beats']==['185']
    assert continued==witness


def test_joint_release_repair_can_span_holds_with_different_start_events():
    tempo=[EventTempo(F(0),157.)];clock=TempoClock(tempo)
    points=[EventPoint(F(1),'1h[4:2]'),EventPoint(F(2),'8h[4:1]')]
    witness=EventPoint(F(3),'2h[4:1]/6h[4:1]')
    codec=StreamCodec.fit([p.note_text for p in [*points,witness]])
    failures,_=rejected(points,witness,tempo,clock)
    fixed,continued,guard,report=repair_hold_release(points,failures,tempo,clock,100.,codec)
    assert len(report['changes'])==2
    assert [p.beat for p in fixed]==[F(1),F(2)] and continued==witness
    assert all(F(c['duration_reduction_beats'])<=F(1,16) for c in report['changes'])
    replay=_new_guard(tempo,0.,100.)
    for point in fixed+[continued]:
        assert replay.accepts(point);replay.commit(point)
    assert guard.prefix==replay.prefix


def test_same_pad_release_needs_an_inactive_native_frame_at_reported_phase():
    tempo=[EventTempo(F(0),190.)];clock=TempoClock(tempo)
    points=[EventPoint(F(252),'8hb[4:8]')]
    witness=EventPoint(F(260),'3b/8b')
    codec=StreamCodec.fit([p.note_text for p in [*points,witness]])
    failures,_=rejected(points,witness,tempo,clock)
    fixed,_,guard,report=repair_hold_conflict(points,failures,tempo,clock,160.,codec)
    assert fixed[0].note_text=='8hb[48:95]'
    assert report['duration_reduction_beats']=='1/12'
    assert report['mode']=='release_tail'
    # Leaving only the release duration can put release and press on one tick.
    too_close=[EventPoint(F(252),'8hb[64:127]')]
    rejected(too_close,witness,tempo,clock)


@pytest.mark.parametrize('text',['8hb[4:12]','8hbx[8:111]'])
def test_ongoing_same_pad_hold_is_revised_from_actual_rearticulation_not_capped(text):
    tempo=[EventTempo(F(0),190.)];clock=TempoClock(tempo)
    points=[EventPoint(F(252),text),EventPoint(F(258),'2'),EventPoint(F(259),'2')]
    witness=EventPoint(F(260),'3b/8b')
    codec=StreamCodec.fit([p.note_text for p in [*points,witness]])
    failures,_=rejected(points,witness,tempo,clock)
    fixed,continued,guard,report=repair_hold_conflict(points,failures,tempo,clock,160.,codec)
    assert fixed[0].note_text==text.split('[')[0]+'[48:95]'
    assert fixed[1:]==points[1:] and continued==witness
    assert report['mode']=='same_pad_rearticulation' and report['native_checks']<=8
    assert points[0].note_text==text
    replay=_new_guard(tempo,0.,160.)
    for p in fixed+[continued]:
        assert replay.accepts(p);replay.commit(p)
    assert guard.prefix==replay.prefix and guard.last_head_seconds==replay.last_head_seconds


def test_same_pad_rearticulation_respects_tempo_change_and_leaves_other_hold_intact():
    tempo=[EventTempo(F(0),120.),EventTempo(F(2),240.)];clock=TempoClock(tempo)
    points=[EventPoint(F(1),'1h[4:3]/8h[4:3]')]
    witness=EventPoint(F(5),'1')
    codec=StreamCodec.fit([p.note_text for p in [*points,witness]])
    failures,_=rejected(points,witness,tempo,clock)
    fixed,_,guard,report=repair_hold_conflict(points,failures,tempo,clock,100.,codec)
    assert fixed[0].note_text.split('/8h')[-1]=='[4:3]'
    assert report['shortened_atom_indices']==[0] and report['mode']=='same_pad_rearticulation'
    assert codec.decode_points(codec.encode_points(fixed))==fixed
    assert report['duration_reduction_seconds']<1.


def test_long_hold_without_same_pad_overlap_is_not_shortened_for_extra_chord():
    tempo=[EventTempo(F(0),190.)];clock=TempoClock(tempo)
    points=[EventPoint(F(252),'8hb[8:111]')]
    witness=EventPoint(F(260),'3b/4b')
    codec=StreamCodec.fit([p.note_text for p in [*points,witness]])
    failures,_=rejected(points,witness,tempo,clock)
    audit={}
    assert repair_hold_conflict(points,failures,tempo,clock,160.,codec,audit=audit) is None
    assert points[0].note_text=='8hb[8:111]' and audit['native_checks']==0


def test_long_hold_and_slide_are_not_shortened_to_force_a_continuation():
    points,witness,tempo,clock,codec=fixture()
    points=[EventPoint(F(130),'1h[1:4]/8h[1:4]')]
    codec=StreamCodec.fit(['1h[1:4]','1-5[4:4]',witness.note_text])
    failures=[(witness,[dict(note_beats=['130'])])]
    audit={}
    assert repair_hold_release(points,failures,tempo,clock,100.,codec,audit=audit) is None
    assert audit['native_checks']==0
    points=[EventPoint(F(134),'1-5[4:1]')]
    assert repair_hold_release(points,[(witness,[dict(note_beats=['134'])])],tempo,clock,100.,codec) is None


def test_failed_repair_preserves_source_and_deduplicates_only_identical_transactions():
    points,witness,tempo,clock,codec=fixture()
    failures,_=rejected(points,witness,tempo,clock)
    attempts=set();audit={}
    assert repair_hold_release(points,failures,tempo,clock,100.,codec,
        attempted=attempts,max_checks=1,audit=audit) is None
    assert audit['native_checks']==1 and points[-1].note_text=='1h[16:3]/8h[16:3]'
    assert repair_hold_release(points,failures,tempo,clock,100.,codec,attempted=attempts) is not None
    # A different continuation is not blacklisted by those same source edits.
    another=EventPoint(witness.beat,'3h[16:3]/5h[16:3]')
    assert repair_hold_release(points,[(another,failures[0][1])],tempo,clock,100.,codec,attempted=attempts) is not None


def test_repair_checks_the_retained_suffix_not_just_two_endpoints():
    points,witness,tempo,clock,codec=fixture()
    # Even when the witness would become valid, an invalid retained suffix
    # prevents a transaction from being committed.
    points.append(EventPoint(F(271,2),'4/4'))
    assert repair_hold_release(points,[(witness,[dict(note_beats=['541/4'])])],tempo,clock,100.,codec) is None


@pytest.mark.parametrize('budget',[0,1])
def test_generation_rebuilds_context_after_edit_and_preserves_later_rest(tmp_path,monkeypatch,budget):
    import maichart.composer_v2.stream_generate as generation
    points,witness,tempo,clock,codec=fixture()
    rest_end=EventPoint(F(190),'4')
    codec=StreamCodec.fit([p.note_text for p in [*points,witness,rest_end]])
    contexts=[]
    class Audio:
        duration_sec=100.
        identity={'fixture':True}
        def summary(self,n):return {'times_sec':np.asarray([0.,100.])}
    class Sampler:
        def __init__(self,model,dataset,song,history,device,rng,args):
            self.cursor=history[-1].beat if history else F(0)
            contexts.append(list(history))
            self.history=list(history)
        def propose(self,target,after,end,fixed=None):
            pending=next((p for p in [*points,witness,rest_end] if p.beat>after),None)
            if after>=witness.beat:
                # A new model context must see the edited tail plus the actual
                # validated continuation, never the stale source or KV.
                assert self.history[-1]==witness
                assert self.history[-2].note_text=='1h[96:17]/8h[96:17]'
            return codec.encode_points([pending]) if pending else [codec.eos_id]
        def _rewind(self,target):pass
    monkeypatch.setattr(generation,'EventSampler',Sampler)
    model=SimpleNamespace(codec=codec,config=SimpleNamespace(max_tokens=2048,history_beats=16,
        window_beats=16,future_beats=4,refresh_events=16,ticks_per_beat=48,sequence_policy='continuous_events_v1'))
    args=SimpleNamespace(temperature=.9,top_p=.95,window_retries=2,candidates=1,rollback_windows=budget,
        level=13.,seed=0,max_events=0,exhaustion_policy='stop',checkpoint='untouched.pt',output=str(tmp_path/'maidata.txt'))
    call=lambda:generation.generate_continuous(model,Audio(),tmp_path/'song.mp3',tempo,0.,{},
        np.zeros(16,'float32'),np.zeros(16,bool),{},args,torch.Generator(),'cpu',F(200))
    if not budget:
        with pytest.raises(RuntimeError,match='recovery exhausted'):call()
        failure=json.loads((tmp_path/'maidata.failure.json').read_text(encoding='utf-8'))
        assert failure['rollbacks']==failure['duration_repairs']==[]
        assert failure['points'][-1]['note_text']==points[-1].note_text
        assert failure['generation_settings']['level']==13.
        assert failure['generation_settings']['temperature']==.9
        assert failure['generation_settings']['effective_device']=='cpu'
    else:
        report=call()
        assert len(report['duration_repairs'])==1 and report['rollbacks']==[]
        exported=parse_event_chart(tmp_path/'maidata.txt',5)
        assert exported.points[-2:]==[witness,rest_end]
        assert exported.points[-3].note_text=='1h[96:17]/8h[96:17]'
        assert len(contexts)==2


def test_recovery_budget_only_refills_past_the_previously_furthest_commit():
    from maichart.composer_v2.stream_generate import RecoveryBudget
    budget=RecoveryBudget(2)
    assert budget.commit(F(187))
    budget.spend()
    assert not budget.commit(F(185)) and not budget.commit(F(187))
    assert budget.used==1
    budget.spend()
    assert not budget.available
    assert budget.commit(F(188)) and budget.available and budget.used==0
    off=RecoveryBudget(0);off.commit(F(100))
    assert not off.available


def test_rollback_preserves_older_hold_when_later_hold_is_verified_contributor():
    from maichart.composer_v2.stream_generate import restore_prefix
    tempo=[EventTempo(F(0),157.)];clock=TempoClock(tempo)
    points=[EventPoint(F(179),'6h[4:4]'),EventPoint(F(180),'4'),EventPoint(F(181),'4h[4:7]')]
    witness=EventPoint(F(182),'3')
    failures,_=rejected(points,witness,tempo,clock)
    assert {b for item in failures[0][1] for b in item['note_beats']}=={'179'}
    kept,guard,report=restore_prefix(points,tempo,clock,100.,witness.beat,['179'],rejected=failures)
    assert kept==points[:2] and report['source_beat']=='181'
    assert report['reason']=='latest_verified_conflict_contributor'
    fresh=_new_guard(tempo,0.,100.)
    for p in kept:
        assert fresh.accepts(p);fresh.commit(p)
    assert guard.prefix==fresh.prefix and guard.last_head_seconds==fresh.last_head_seconds


def test_rollback_does_not_blame_a_later_unrelated_event():
    from maichart.composer_v2.stream_generate import restore_prefix
    tempo=[EventTempo(F(0),157.)];clock=TempoClock(tempo)
    points=[EventPoint(F(179),'6h[4:4]'),EventPoint(F(180),'4')]
    witness=EventPoint(F(181),'2/8')
    failures,_=rejected(points,witness,tempo,clock)
    kept,guard,report=restore_prefix(points,tempo,clock,100.,witness.beat,['179'],rejected=failures)
    assert kept==[] and report['source_beat']=='179'


@pytest.mark.parametrize('stalled',[False,True])
def test_separate_dead_ends_recover_but_regenerating_old_range_still_exhausts(tmp_path,monkeypatch,stalled):
    import maichart.composer_v2.stream_generate as generation
    tempo=[EventTempo(F(0),157.)]
    if stalled:
        authored=[EventPoint(F(185),'8h[4:11]'),EventPoint(F(187),'1h[4:3]'),EventPoint(F(188),'4')]
    else:
        authored=[p for start in (10,20,30) for p in (
            EventPoint(F(start),'1h[4:1]/8h[4:1]'),EventPoint(F(start+1),'2h[4:1]/6h[4:1]'))]
    codec=StreamCodec.fit([p.note_text for p in authored])
    calls=[]
    class Audio:
        duration_sec=100.
        identity={'fixture':True}
        def summary(self,n):return {'times_sec':np.asarray([0.,100.])}
    class Sampler:
        def __init__(self,model,dataset,song,history,device,rng,args):
            self.cursor=history[-1].beat if history else F(0)
        def propose(self,target,after,end,fixed=None):
            calls.append(after)
            assert len(calls)<20  # A stale frontier must not permit an endless replay.
            pending=next((p for p in authored if p.beat>after),None)
            return codec.encode_points([pending]) if pending else [codec.eos_id]
        def _rewind(self,target):pass
    monkeypatch.setattr(generation,'EventSampler',Sampler)
    model=SimpleNamespace(codec=codec,config=SimpleNamespace(max_tokens=2048,history_beats=16,
        window_beats=16,future_beats=4,refresh_events=16,ticks_per_beat=48,sequence_policy='continuous_events_v1'))
    args=SimpleNamespace(temperature=.9,top_p=.95,window_retries=1,candidates=1,rollback_windows=2,
        level=13.,seed=0,max_events=0,exhaustion_policy='stop',checkpoint='untouched.pt',output=str(tmp_path/'maidata.txt'))
    call=lambda:generation.generate_continuous(model,Audio(),tmp_path/'song.mp3',tempo,0.,{},
        np.zeros(16,'float32'),np.zeros(16,bool),{},args,torch.Generator(),'cpu',F(200))
    if stalled:
        with pytest.raises(RuntimeError,match='recovery exhausted'):call()
        failure=json.loads((tmp_path/'maidata.failure.json').read_text(encoding='utf-8'))
        assert len(failure['rollbacks'])==2 and failure['duration_repairs']==[]
        assert failure['recovery_used_at_frontier']==2 and failure['furthest_committed_beat']=='187'
    else:
        report=call()
        assert len(report['duration_repairs'])==3 and report['rollbacks']==[]
        assert report['recovery_budget']==2 and report['recovery_used_at_frontier']==0
        assert report['recovery_budget_scope']=='forward_progress_v2'
        assert len(parse_event_chart(tmp_path/'maidata.txt',5,strict_notes=True).points)==6
