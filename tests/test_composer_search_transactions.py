from collections import Counter
from fractions import Fraction as F
from types import SimpleNamespace

import pytest
import torch

from maichart.composer.event_set_codec import EventSetCodec
from maichart.composer.guard import EventGuard
from maichart.composer.stream_data import StreamInferenceContext, StreamStep, DeltaVocabulary, NOTE, second_at_beat
from maichart.composer.stream_generate import replay_stream_prefix, _Curve
from maichart.composer.stream_search import _repair_prefix, _advances, _committed_history_blocker, search_stream
from maichart.event_timing import EventPoint, EventTempo
from maichart.tempo_clock import TempoClock


class Curve(_Curve):
    def __init__(self):self.counts=Counter()


def fixture():
    torch.set_num_threads(2)
    song=SimpleNamespace(tempo=[EventTempo(F(0),120.),EventTempo(F(4),180.)],first=0.,audio=SimpleNamespace(duration_sec=12.))
    clock=TempoClock(song.tempo,0.)
    beats=torch.arange(0.,30.,.25,dtype=torch.float64)
    bank=(torch.zeros(len(beats),238),torch.tensor([second_at_beat(clock,F(str(float(b)))) for b in beats]),beats)
    codec=EventSetCodec.fit(['1h[4:8]/5x','1h[4:1]/5x','1xh[4:8]','5x','1','2','3','4','5','6',
        '1-5[4:8]','1-5[4:1]','1-3-5[4:8]'])
    def context():return StreamInferenceContext(song,codec,DeltaVocabulary([0,F(1,16),F(1,2),1,2]),torch.zeros(16),
        audio_bank=bank,lead_in=3,continuation_policy='pattern_runs_v2',phrase_context_policy='causal_phrase_v1')
    def guard():return EventGuard(song.tempo,0.,12.,touch_layout_policy='open',physical_policy='runtime_hard_v2')
    def step(start,end,text):return StreamStep(NOTE,F(start),F(end),second_at_beat(clock,F(start)),second_at_beat(clock,F(end)),text)
    return song,clock,bank,codec,context,guard,step


def test_shorten_and_withdraw_keep_protected_same_onset_child():
    song,clock,_,_,context,guard,step=fixture()
    old=[step(0,0,'1h[4:8]/5x'),step(0,1,'2')]
    g=guard()
    for s in old:g.commit(EventPoint(s.end,s.text))
    assert not g.accepts(EventPoint(F(2),'1'))
    result=dict(steps=old,points=[EventPoint(s.end,s.text) for s in old],guard=g,anchor_second=.5)
    metrics=dict(native_calls=0,native_sec=0.)
    edits=list(_repair_prefix(result,song,clock,metrics,{}))
    assert edits and all('5x' in prefix[-1].text for prefix,_ in edits)
    withdrawals=list(_repair_prefix(result,song,clock,metrics,{},withdraw=True))
    assert withdrawals and withdrawals[0][0][-1].text=='5x'
    assert old[0].text=='1h[4:8]/5x'


def test_rewritten_history_progress_uses_beat_not_step_count():
    _,_,_,_,_,_,step=fixture()
    old=[step(0,1,'1'),step(1,2,'2'),step(2,3,'3')]
    rewritten=[step(0,2,'4'),step(2,4,'5')]
    assert len(rewritten)<len(old)
    assert _advances(rewritten,F(3))
    assert not _advances(rewritten[:1],F(3))


def test_search_accepts_shorter_rewritten_history_that_advances(monkeypatch):
    from maichart.composer import stream_quality
    monkeypatch.setattr(stream_quality,'score_phrase',lambda *a,**kw:dict(total=1.,components={}))
    song,clock,bank,codec,_,gf,step=fixture()
    calls=[]
    def rollout(prefix,seed,target,deadline):
        calls.append(target)
        steps=([step(0,1,'1'),step(1,2,'2'),step(2,3,'3')] if len(calls)==1
               else [step(0,3,'4'),step(3,6,'5')])
        return dict(steps=steps,points=[EventPoint(item.end,item.text) for item in steps],
            cursor=steps[-1].end,reason='span_complete',guard=gf(),anchor_second=0.,
            retries=0,timing_retries=0,forced_waits=0,
            _physical_gate=(SimpleNamespace(coverage={'parse_errors':[]}),[]))
    args=SimpleNamespace(seed=1,phrase_candidates=1,phrase_write_beats=3.,phrase_commit_beats=3.,
        phrase_repair_budget=0,phrase_rollback_budget=0,phrase_time_budget_sec=30.,softcomfort_strength=1.)
    selected,report=search_stream(rollout,song,SimpleNamespace(),args,F(6),clock,Curve(),audio_bank=bank,codec=codec)
    assert calls==[F(3),F(6)]
    assert len(selected)==2 and selected[-1].end==6
    assert report['phrases'][1]['candidate_logs'][0]['eligible']


def test_committed_sustain_conflict_is_history_priority(monkeypatch):
    from maichart.composer import stream_search
    song,_,_,_,_,_,step=fixture()
    committed=[step(0,1,'1b>8[4:13]')]
    old=SimpleNamespace(beat=F(1),raw='1b>8[4:13]')
    new=SimpleNamespace(beat=F(2),raw='2')
    issue=SimpleNamespace(notes=[old],conflict_type='miacode_MultiTouch',evidence={'native':'fixture'})
    guard=SimpleNamespace(failure_conflicts=[issue],failure_trials=[],enforced=lambda item:True)
    failed=[dict(guard=guard)]
    event=SimpleNamespace(beat=F(1),raw=old.raw,note_type='slide',protected=False)
    monkeypatch.setattr(stream_search,'_extract',lambda *args:([event],{},None))
    evidence=_committed_history_blocker(failed,committed,song)
    assert evidence['conflict_type']=='miacode_MultiTouch'
    assert evidence['named_committed_atoms']==[dict(beat='1',raw=old.raw,note_type='slide')]
    issue.notes=[old,new]
    assert not _committed_history_blocker(failed,committed,song)


@pytest.mark.parametrize('candidates',[1,2,4])
def test_multiround_withdrawal_replays_all_state_and_invalidates_old_tail(monkeypatch,candidates):
    from maichart.composer import stream_quality
    monkeypatch.setattr(stream_quality,'score_phrase',lambda *a,**kw:dict(total=1.,components={}))
    song,clock,bank,codec,cf,gf,step=fixture()
    snapshots=[];base=Curve()
    def rollout(prefix,seed,target,deadline):
        ctx,g,curve,points=replay_stream_prefix(prefix,cf,gf,base,clock)
        generator=torch.Generator().manual_seed(seed)
        signature=torch.rand(3,generator=generator)
        assert torch.equal(signature,torch.rand(3,generator=torch.Generator().manual_seed(seed)))
        cursor=prefix[-1].end if prefix else F(0)
        anchor=second_at_beat(clock,cursor)
        if not prefix:
            initial=step(0,0,'1h[4:8]/5x');ctx.commit(initial);g.commit(EventPoint(F(0),initial.text));curve.commit(0.,2);points.append(EventPoint(F(0),initial.text))
        reason='span_complete'
        while cursor<target:
            beat=cursor+1
            has_hold=any('h[' in p.note_text for p in points)
            text='1' if has_hold and beat>=3 else str(int(beat)%4+2)
            point=EventPoint(beat,text)
            if not g.accepts(point):reason='physical_dead_end';break
            g.commit(point);ctx.commit(step(cursor,beat,text));curve.commit(second_at_beat(clock,beat),1);points.append(point);cursor=beat
        # A fresh replay must match every tensor, including sustained occupancy,
        # clock, continuation and the causal phrase archive.
        fresh,fg,fc,fp=replay_stream_prefix(ctx._committed,cf,gf,base,clock)
        a,b=ctx.current_batch(),fresh.current_batch()
        for key in a:assert torch.equal(a[key],b[key]),key
        assert g.last_head_seconds==fg.last_head_seconds and curve.counts==fc.counts
        snapshots.append((list(prefix),seed,list(ctx._committed),dict(curve.counts),g))
        return dict(steps=list(ctx._committed),points=points,cursor=cursor,reason=reason,guard=g,
                    anchor_second=anchor,retries=0,timing_retries=0,forced_waits=0,
                    model_diagnostics=dict(seed_checksum=seed,new_interval_only=True))
    args=SimpleNamespace(seed=7,phrase_candidates=candidates,phrase_write_beats=2.,phrase_commit_beats=1.,
        phrase_repair_budget=0,phrase_rollback_budget=2,phrase_time_budget_sec=30.,softcomfort_strength=1.)
    global_rng=torch.get_rng_state().clone()
    selected,report=search_stream(rollout,song,SimpleNamespace(),args,F(5),clock,base,audio_bank=bank,codec=codec)
    assert torch.equal(global_rng,torch.get_rng_state())
    assert selected[0].text=='5x'
    assert not any('h[' in s.text for s in selected)
    assert report['rollbacks_used']==1
    assert any(p.get('invalidated_by_suffix_regeneration') for p in report['phrases'])
    assert len({snapshot[1] for snapshot in snapshots})==len(snapshots)
    assert len({id(snapshot[4]) for snapshot in snapshots})==len(snapshots)
    assert base.counts=={}
    assert selected[-1].end==5
    for phrase in report['phrases']:
        assert len(phrase['candidate_logs'])==phrase['attempted_rollouts']
        winner=next(log for log in phrase['candidate_logs'] if log['candidate_id']==phrase['winner_candidate_id'])
        assert winner['eligible'] and winner['score']==phrase['score']
        assert winner['seed']==phrase['winner_seed']
        assert winner['model_diagnostics']==phrase['model_diagnostics']
        assert all(isinstance(log['event_count'],int) and isinstance(log['head_count'],int) for log in phrase['candidate_logs'])
        assert phrase['score_effect']==('diagnostic_only_single_candidate' if candidates==1 else 'rank_physically_feasible_candidates')


@pytest.mark.parametrize('old,new', [('1h[4:8]/5x','1h[4:1]/5x'),
                                    ('1-5[4:8]','1-5[4:1]'),('1-3-5[4:8]','2')])
def test_edited_hold_slide_and_compound_slide_replace_all_occupancy(old,new):
    song,clock,_,_,cf,gf,step=fixture()
    original=[step(0,0,old)]+[step(i-1,i,'6') for i in range(1,7)]
    edited=[step(0,0,new)]+original[1:]
    before,bg,bc,_=replay_stream_prefix(original,cf,gf,Curve(),clock)
    original_batch={k:v.clone() for k,v in before.current_batch().items()}
    after,ag,ac,_=replay_stream_prefix(edited,cf,gf,bc,clock)
    assert torch.equal(before.current_batch()['action_state'],original_batch['action_state'])
    assert before._committed[0].text==old and after._committed[0].text==new
    assert before.clock.at(F(6))==after.clock.at(F(6))
    assert bg._lifetime(EventPoint(F(0),old))[1]>ag._lifetime(EventPoint(F(0),new))[1]
    assert not torch.equal(before.current_batch()['action_state'],after.current_batch()['action_state'])
    assert after._phrase is not before._phrase
    # The descriptor intentionally ignores duration, and no phrase has closed
    # at beat six. Archive equality is checked against a fresh edited replay.
    fresh,_,_,_=replay_stream_prefix(edited,cf,gf,Curve(),clock)
    assert torch.equal(after.current_batch()['phrase_memory_features'],fresh.current_batch()['phrase_memory_features'])
    assert ag.prefix is not bg.prefix and ag.last_head_seconds is not bg.last_head_seconds
    assert ac.counts is not bc.counts


def test_protected_hold_is_not_repaired_and_retry_budget_stops(monkeypatch):
    song,clock,bank,codec,cf,gf,step=fixture()
    calls=[]
    def rollout(prefix,seed,target,deadline):
        accepted=[step(0,0,'1xh[4:8]')]
        ctx,g,curve,points=replay_stream_prefix(accepted,cf,gf,Curve(),clock)
        assert not g.accepts(EventPoint(F(1),'1'))
        calls.append((list(prefix),seed))
        return dict(steps=accepted,points=points,cursor=F(0),reason='physical_dead_end',guard=g,
                    anchor_second=0.,retries=0,timing_retries=0,forced_waits=0)
    args=SimpleNamespace(seed=9,phrase_candidates=1,phrase_write_beats=2.,phrase_commit_beats=1.,
        phrase_repair_budget=1,phrase_rollback_budget=2,phrase_time_budget_sec=30.,softcomfort_strength=1.)
    with pytest.raises(ValueError,match='no complete physically feasible candidate') as caught:
        search_stream(rollout,song,SimpleNamespace(),args,F(4),clock,Curve(),audio_bank=bank,codec=codec)
    assert len(calls)==5 and len({seed for _,seed in calls})==5
    import json
    evidence=caught.value.diagnostics
    assert evidence['repair_budget']==1 and evidence['repairs_used']==0
    assert evidence['current_event_retries_used']==2
    assert evidence['private_tail_retries_used']==2
    assert evidence['recent_failures'][-1]['reason']=='physical_dead_end'
    assert evidence['recent_failures'][-1]['steps'][0]['text']=='1xh[4:8]'
    assert evidence['recent_failures'][-1]['seed']==calls[-1][1]
    json.dumps(evidence,allow_nan=False)


def test_current_retry_keeps_private_progress_without_spending_history_budget(monkeypatch):
    from maichart.composer import stream_quality
    monkeypatch.setattr(stream_quality,'score_phrase',lambda *a,**kw:dict(total=1.,components={}))
    song,clock,bank,codec,cf,gf,step=fixture()
    anchors=[]
    def rollout(prefix,seed,target,deadline):
        ctx,g,curve,points=replay_stream_prefix(prefix,cf,gf,Curve(),clock)
        cursor=prefix[-1].end if prefix else F(0)
        anchors.append(cursor)
        beat=cursor+1;point=EventPoint(beat,str(int(beat)+1))
        assert g.accepts(point)
        g.commit(point);ctx.commit(step(cursor,beat,point.note_text));points.append(point)
        return dict(steps=list(ctx._committed),points=points,cursor=beat,guard=g,
            reason='span_complete' if beat>=target else 'physical_dead_end',anchor_second=second_at_beat(clock,cursor),
            retries=0,timing_retries=0,forced_waits=0)
    args=SimpleNamespace(seed=11,phrase_candidates=1,phrase_write_beats=3.,phrase_commit_beats=1.,
        phrase_repair_budget=0,phrase_rollback_budget=0,phrase_time_budget_sec=30.,softcomfort_strength=1.)
    selected,report=search_stream(rollout,song,SimpleNamespace(),args,F(3),clock,Curve(),audio_bank=bank,codec=codec)
    assert anchors==[F(0),F(1),F(2)]
    assert selected[-1].end==3
    assert report['repairs_used']==0 and report['rollbacks_used']==0
    assert report['current_event_retries_used']==2


def test_short_write_eos_terminates_without_repeating_empty_commit(monkeypatch):
    from maichart.composer import stream_quality
    monkeypatch.setattr(stream_quality,'score_phrase',lambda *a,**kw:dict(total=1.,components={}))
    song,clock,bank,codec,_,gf,_=fixture()
    calls=[]
    def rollout(prefix,seed,target,deadline):
        calls.append(target)
        return dict(steps=list(prefix),points=[],cursor=F(1),reason='learned_eos',
            guard=gf(),anchor_second=0.,retries=0,timing_retries=0,forced_waits=0,
            _physical_gate=(SimpleNamespace(coverage={'parse_errors':[]}),[]))
    args=SimpleNamespace(seed=1,phrase_candidates=1,phrase_write_beats=.25,phrase_commit_beats=.125,
        phrase_repair_budget=0,phrase_rollback_budget=0,phrase_time_budget_sec=30.,softcomfort_strength=1.)
    selected,report=search_stream(rollout,song,SimpleNamespace(),args,F(1),clock,Curve(),audio_bank=bank,codec=codec)
    assert selected==[] and calls==[F(1,4)]
    assert report['stop_reason']=='audio_end'


def test_default_search_scores_full_write_tail_before_selecting(monkeypatch):
    from maichart.composer import stream_quality
    observed=[]
    def score(points,prefix,song,condition,start,finish,**kwargs):
        observed.append((tuple(point.beat for point in points),start,finish,kwargs['observed_end_second']))
        return dict(total=1.,components={})
    monkeypatch.setattr(stream_quality,'score_phrase',score)
    song,clock,bank,codec,_,gf,step=fixture()
    steps=[step(0,1,'1'),step(1,2,'2')]
    def rollout(prefix,seed,target,deadline):
        return dict(steps=steps,points=[EventPoint(F(1),'1'),EventPoint(F(2),'2')],
            cursor=F(2),reason='span_complete',guard=gf(),anchor_second=0.,
            retries=0,timing_retries=0,forced_waits=0,
            _physical_gate=(SimpleNamespace(coverage={'parse_errors':[]}),[]))
    args=SimpleNamespace(seed=1,phrase_candidates=1,phrase_write_beats=2.,phrase_commit_beats=1.,
        phrase_repair_budget=0,phrase_rollback_budget=0,phrase_time_budget_sec=30.,softcomfort_strength=1.)
    selected,report=search_stream(rollout,song,SimpleNamespace(),args,F(2),clock,Curve(),audio_bank=bank,codec=codec)
    assert selected==steps
    assert observed[0][:3]==((F(1),F(2)),F(0),F(2))
    assert observed[0][3]==second_at_beat(clock,F(2))
    assert report['phrases'][0]['score_end_beat']=='2'


def test_overshooting_commit_requires_common_observed_score_horizon(monkeypatch):
    from maichart.composer import stream_quality
    monkeypatch.setattr(stream_quality,'score_phrase',lambda *a,**kw:dict(total=1.,components={}))
    song,clock,bank,codec,_,gf,step=fixture()
    calls=[]
    def rollout(prefix,seed,target,deadline):
        calls.append(seed)
        steps=[step(0,3,'1')] if len(calls)==1 else [step(0,1,'1'),step(1,2,'2')]
        return dict(steps=steps,points=[EventPoint(item.end,item.text) for item in steps],
            cursor=steps[-1].end,reason='span_complete',guard=gf(),anchor_second=0.,
            retries=0,timing_retries=0,forced_waits=0,
            _physical_gate=(SimpleNamespace(coverage={'parse_errors':[]}),[]))
    args=SimpleNamespace(seed=1,phrase_candidates=2,phrase_write_beats=2.,phrase_commit_beats=1.,
        phrase_repair_budget=0,phrase_rollback_budget=0,phrase_time_budget_sec=30.,softcomfort_strength=1.)
    selected,report=search_stream(rollout,song,SimpleNamespace(),args,F(3),clock,Curve(),audio_bank=bank,codec=codec)
    assert selected[-1].end==3
    first,shorter=report['phrases'][0]['candidate_logs']
    assert first['score_end_beat']=='3' and first['eligible']
    assert not shorter['eligible']
    assert shorter['failure_reason']=='insufficient_observed_horizon'
    assert shorter['required_score_end_beat']=='3'


def test_uncommitted_sustain_tail_can_restart_without_history_budget(monkeypatch):
    from maichart.composer import stream_quality
    monkeypatch.setattr(stream_quality,'score_phrase',lambda *a,**kw:dict(total=1.,components={}))
    song,clock,bank,codec,cf,gf,step=fixture()
    calls=[]
    def rollout(prefix,seed,target,deadline):
        calls.append(list(prefix))
        ctx,g,curve,points=replay_stream_prefix(prefix,cf,gf,Curve(),clock)
        cursor=prefix[-1].end if prefix else F(0)
        if not prefix:
            raw='1h[4:8]/5x' if len(calls)==1 else '5x'
            ctx.commit(step(0,0,raw));g.commit(EventPoint(F(0),raw));points.append(EventPoint(F(0),raw))
        if any('h[' in point.note_text for point in points):
            if cursor==0:
                ctx.commit(step(0,1,'2'));g.commit(EventPoint(F(1),'2'));points.append(EventPoint(F(1),'2'));cursor=F(1)
            assert not g.accepts(EventPoint(F(2),'1'))
            reason='physical_dead_end'
        else:
            while cursor<target:
                beat=cursor+1;raw=str(int(beat)+1);point=EventPoint(beat,raw)
                assert g.accepts(point)
                g.commit(point);ctx.commit(step(cursor,beat,raw));points.append(point);cursor=beat
            reason='span_complete'
        return dict(steps=list(ctx._committed),points=points,cursor=cursor,guard=g,reason=reason,
            anchor_second=0.,retries=0,timing_retries=0,forced_waits=0)
    args=SimpleNamespace(seed=13,phrase_candidates=1,phrase_write_beats=3.,phrase_commit_beats=1.,
        phrase_repair_budget=0,phrase_rollback_budget=0,phrase_time_budget_sec=30.,softcomfort_strength=1.)
    selected,report=search_stream(rollout,song,SimpleNamespace(),args,F(3),clock,Curve(),audio_bank=bank,codec=codec)
    assert len(calls)==4 and calls[3]==[]
    assert not any('h[' in s.text for s in selected)
    assert selected[0].text=='5x'
    assert report['repairs_used']==report['rollbacks_used']==0
    assert report['private_tail_retries_used']==1 and report['current_event_retries_used']==2


@pytest.mark.parametrize('switch_slide_path',[False,True])
def test_duration_counterfactual_keeps_old_identity_but_rejects_new_slide_path(monkeypatch,switch_slide_path):
    """A diagnostic fingerprint may ignore duration, never trajectory identity."""
    from maichart.composer import stream_search
    song,clock,_,_,_,gf,step=fixture()
    old=[step(0,0,'1h[4:8]/5x'),step(0,1,'2')]
    guard=gf()
    for s in old:guard.commit(EventPoint(s.end,s.text))
    assert not guard.accepts(EventPoint(F(2),'1'))
    result=dict(steps=old,points=[EventPoint(s.end,s.text) for s in old],guard=guard,anchor_second=.5)
    def diagnostic(points,*args):
        hold=next(event for event in stream_search._extract(points,song.tempo,song.first)[0] if event.note_type=='hold')
        unchanged=hold.raw=='1h[4:8]'
        branch='4-8[4:1]' if switch_slide_path and not unchanged else '4-6[4:1]'
        notes=[SimpleNamespace(beat=hold.beat,raw=hold.raw,note_type='hold',button=1),
               SimpleNamespace(beat=F(1),raw=branch,note_type='slide',button=4)]
        def issue(kind):return SimpleNamespace(conflict_type=kind,time=1.,severity='hard',action='enforce',
            evidence='miacode_runtime:fingerprint_fixture',notes=notes)
        return SimpleNamespace(coverage={},conflicts_detected=[issue('miacode_Overlap')]
            +([issue('miacode_MultiTouch')] if unchanged else []))
    monkeypatch.setattr(stream_search,'_judge',diagnostic)
    edits=list(_repair_prefix(result,song,clock,dict(native_calls=0,native_sec=0.),{}))
    assert bool(edits)==(not switch_slide_path)


def test_completed_candidate_rejected_by_native_gate_can_backtrack_private_tail(monkeypatch):
    """A completed rollout is not feasible until the phrase gate agrees."""
    from maichart.composer import stream_quality
    monkeypatch.setattr(stream_quality,'score_phrase',lambda *a,**kw:dict(total=1.,components={}))
    song,clock,bank,codec,cf,gf,step=fixture()
    calls=[]
    def rollout(prefix,seed,target,deadline):
        calls.append(list(prefix))
        bad=len(calls)==1
        ctx,g,curve,points=replay_stream_prefix(prefix,cf,gf,Curve(),clock)
        # Reproduce the callback contract of the real v36 failure: it reports
        # span_complete with no last guard rejection, while full native judging
        # finds a conflict in the still-private completed suffix.
        source=[step(0,0,'1h[4:8]/5x' if bad else '5x')]
        source += [step(i-1,i,'1' if bad and i==2 else str(i+1)) for i in range(1,4)]
        for record in source:
            ctx.commit(record);g.commit(EventPoint(record.end,record.text));points.append(EventPoint(record.end,record.text))
        assert g.last_reason is None and not g.failure_trials
        return dict(steps=list(ctx._committed),points=points,cursor=F(3),reason='span_complete',guard=g,
            anchor_second=0.,retries=0,timing_retries=0,forced_waits=0)
    args=SimpleNamespace(seed=17,phrase_candidates=1,phrase_write_beats=3.,phrase_commit_beats=1.,
        phrase_repair_budget=0,phrase_rollback_budget=0,phrase_time_budget_sec=30.,softcomfort_strength=1.)
    selected,report=search_stream(rollout,song,SimpleNamespace(),args,F(3),clock,Curve(),audio_bank=bank,codec=codec)
    assert calls==[[],[]]
    assert selected[0].text=='5x'
    assert report['current_event_retries_used']==0
    assert report['private_tail_retries_used']==1
    assert report['repairs_used']==report['rollbacks_used']==0
    rejected=report['phrases'][0]['candidate_logs'][0]
    assert not rejected['eligible'] and rejected['score'] is None
    assert rejected['failure_reason']=='enforced_native_hard_conflict'
    assert rejected['physical_gate']['enforced_conflicts']


def test_phrase_gate_retains_intermediate_slide_during_old_active_slide():
    from maichart.composer.stream_search import _gate
    song=SimpleNamespace(tempo=[EventTempo(F(0),205.)],first=0.)
    guard=EventGuard(song.tempo,0.,120.,touch_layout_policy='open',intercept_tap_on_slide=True,
        physical_policy='runtime_hard_v2')
    points=[EventPoint(F(63623,192),'1bx-5[16.2406##0.2256]'),
            EventPoint(F(346),'3/8-4[8:1]'),EventPoint(F(699,2),'5')]
    result=dict(points=points,guard=guard,anchor_second=101.26829268292683)
    physical,hard=_gate(result,song,dict(native_calls=0,native_sec=0.),{})
    assert not physical.coverage.get('parse_errors')
    assert any(issue.conflict_type=='miacode_SlideTooFast' for issue in hard)
