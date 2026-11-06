from fractions import Fraction as F
from types import SimpleNamespace
from pathlib import Path
import json

import numpy as np
import pytest
import torch

from maichart.event_timing import EventPoint,EventTempo,parse_event_chart
from maichart.composer_v2.stream_codec import StreamCodec
from maichart.composer_v2.generate import _new_guard,parser
from maichart.composer_v2.exhaustion import admit_physical_conflict,resolve_policy


def test_public_default_resolves_to_report_for_continuous_and_stop_for_old_windows():
    args=parser().parse_args(['--checkpoint','x','--audio','a','--output','o'])
    assert args.exhaustion_policy=='auto'
    assert resolve_policy('auto','continuous_events_v1')=='report'
    assert resolve_policy('auto','window_v1')=='stop'
    assert resolve_policy('stop','continuous_events_v1')=='stop'
    with pytest.raises(ValueError):resolve_policy('report','window_v1')


def generate_fixture(tmp_path,monkeypatch,policy,points):
    import maichart.composer_v2.stream_generate as generation
    codec=StreamCodec.fit([p.note_text for p in points]);seen=[]
    class Audio:
        duration_sec=100.
        identity={'fixture':True}
        def summary(self,n):return {'times_sec':np.asarray([0.,100.])}
    class Sampler:
        def __init__(self,model,dataset,song,history,device,rng,args):
            self.cursor=history[-1].beat if history else F(0);seen.append(list(history))
        def propose(self,target,after,end,fixed=None):
            pending=next((p for p in points if p.beat>after),None)
            return codec.encode_points([pending]) if pending else [codec.eos_id]
        def _rewind(self,target):pass
    monkeypatch.setattr(generation,'EventSampler',Sampler)
    model=SimpleNamespace(codec=codec,config=SimpleNamespace(max_tokens=2048,history_beats=16,
        window_beats=16,future_beats=4,refresh_events=1,ticks_per_beat=48,sequence_policy='continuous_events_v1'))
    args=SimpleNamespace(temperature=.3,top_p=.95,window_retries=1,candidates=1,rollback_windows=0,
        level=14.5,seed=0,max_events=0,exhaustion_policy=policy,checkpoint='untouched.pt',output=str(tmp_path/'maidata.txt'))
    report=generation.generate_continuous(model,Audio(),tmp_path/'song.mp3',[EventTempo(F(0),120.)],0.,{},
        np.zeros(16,'float32'),np.zeros(16,bool),{},args,torch.Generator(),'cpu',F(200))
    return report,seen


def test_report_mode_commits_conflicting_events_keeps_real_history_and_exports_report(tmp_path,monkeypatch):
    points=[EventPoint(F(10),'8h[4:12]'),EventPoint(F(11),'8'),EventPoint(F(12),'8'),EventPoint(F(24),'1')]
    report,seen=generate_fixture(tmp_path,monkeypatch,'auto',points)
    assert report['events']==4 and report['accepted_with_conflicts_count']==2
    assert report['exhaustion_policy']=='report' and report['physics_status']=='unresolved_conflicts'
    assert parse_event_chart(tmp_path/'maidata.txt',5,strict_notes=True).points==points
    assert points[:3] in seen  # The bypassed notes and ongoing Hold reach the next model context.
    path=Path(report['conflict_report']);assert path.exists()
    data=json.loads(path.read_text(encoding='utf-8'))
    assert data['accepted_with_conflicts_count']==2
    assert data['points']==[p.to_dict() for p in points]
    assert data['assessment']['status']=='complete' and data['assessment']['enforced_conflict_count']>0
    assert data['bypassed_events'][0]['reason']=='recovery_budget_exhausted'
    assert (tmp_path/'maidata.conflicts.json').exists()
    assert not (tmp_path/'maidata.failure.json').exists()


def test_stop_mode_still_aborts_at_the_same_physical_conflict(tmp_path,monkeypatch):
    points=[EventPoint(F(10),'8h[4:12]'),EventPoint(F(11),'8'),EventPoint(F(24),'1')]
    with pytest.raises(RuntimeError,match='recovery exhausted'):
        generate_fixture(tmp_path,monkeypatch,'stop',points)
    assert not (tmp_path/'maidata.txt').exists()
    failure=json.loads((tmp_path/'maidata.failure.json').read_text(encoding='utf-8'))
    assert failure['exhaustion_policy']=='stop' and failure['bypassed_events']==[]


@pytest.mark.parametrize('text',['1/1','8h[4:400]'])
def test_report_mode_never_exports_duplicate_heads_or_actions_beyond_audio(tmp_path,monkeypatch,text):
    with pytest.raises(RuntimeError,match='recovery exhausted'):
        generate_fixture(tmp_path,monkeypatch,'report',[EventPoint(F(10),text)])
    assert not (tmp_path/'maidata.txt').exists()


def test_bypass_requires_forward_progress_and_does_not_mutate_the_original_guard():
    guard=_new_guard([EventTempo(F(0),120.)],0.,100.)
    source=EventPoint(F(10),'8h[4:12]');assert guard.accepts(source);guard.commit(source)
    candidate=EventPoint(F(11),'8');details=[dict(kind='miacode_Overlap',note_beats=['11'])]
    assert admit_physical_conflict(guard,[(candidate,details)],F(11)) is None
    original=list(guard.prefix)
    _,trial,_=admit_physical_conflict(guard,[(candidate,details)],F(10))
    assert guard.prefix==original and trial.prefix[-1][0]==candidate
