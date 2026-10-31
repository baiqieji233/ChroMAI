from collections import Counter
from dataclasses import replace
from fractions import Fraction as F
import random
from types import SimpleNamespace

import pytest
import torch

from maichart.composer import stream_augmentation as aug, stream_recovery as recovery
from maichart.composer.stream_data import StreamDataset, NOTE, WAIT, EOS
from maichart.event_timing import EventPoint, EventTempo


def dense_dataset(monkeypatch):
    from test_stream_augmentation import dataset
    dataset(monkeypatch)
    song=SimpleNamespace(audio=SimpleNamespace(duration_sec=100.),tempo=[EventTempo(F(0),120.)],first=0.)
    points=[EventPoint(F(i,4),str(i%8+1)) for i in range(700)]
    base=SimpleNamespace(songs=[(song,points,12.,1.,None,torch.zeros(16),(),None)])
    return StreamDataset(base=base,lead_in=128,targets=4,audio_grid_seconds=.05,
        history_window_policy='block_aligned_v2',continuation_policy='pattern_runs_v2')


def allow_synthetic(monkeypatch,groups):
    observed=[]
    monkeypatch.setattr(recovery,'_synthetic_loop_targets',lambda ds:groups)
    monkeypatch.setattr(recovery,'_guard',lambda *args:object())
    monkeypatch.setattr(aug,'_validate_window',lambda guard,steps:observed.append(steps))
    return observed


def test_v6_rotation_gap_accepts_more_than_eighths_without_changing_v5():
    from maichart.tempo_clock import TempoClock
    from test_stream_musical_loop_recovery import _steps_at
    source=_steps_at([2,4,6,8,9])
    ds=SimpleNamespace(delta_vocab=SimpleNamespace(values=list(map(F,('0','1/4','1/2','1','2')))),
        clocks=[TempoClock([EventTempo(F(0),120.)],0.)])
    with pytest.raises(ValueError,match='exact boundaries'):
        aug._same_gap_steps(ds,0,source,0,4,5,random.Random(0),rotation_gap=True)
    changed=aug._same_gap_steps(ds,0,source,0,4,5,random.Random(0),rotation_gap=True,
        rotation_gaps=('1/4','1/2','1','2'))
    assert all(step.end-step.start==2 for step in changed[:4])
    assert changed[4:]==source[4:]
    assert aug._augmentation_modes(aug.MUSICAL_RESIDUAL_INPUT_POLICY)==aug._augmentation_modes(aug.MUSICAL_INPUT_POLICY)


def test_synthetic_one_beat_loop_changes_history_count_without_moving_reference(monkeypatch):
    from test_stream_musical_loop_recovery import LongestRotation
    ds=dense_dataset(monkeypatch);index=600;target=ds.streams[0][index]
    observed=allow_synthetic(monkeypatch,{NOTE:((0,index),),WAIT:(),EOS:()})
    diagnostics=Counter()
    row=recovery._synthetic_long_exit_row(ds,None,LongestRotation(0),10,diagnostics)
    assert row is not None and diagnostics['synthetic_loop_events']==128
    loop=observed[0][:-1]
    assert len(loop)==128 and all(step.end-step.start==1 for step in loop[1:])
    assert loop[-1].end==target.start and observed[0][-1] is target
    # The old dense span contains >500 events; the new history uses 128.
    assert sum(loop[0].end<=step.end<=loop[-1].end for step in ds.streams[0])>500
    assert row['target_kind'][-1]==NOTE
    assert row['step_beat'][-1]==float(target.start) and row['target_beat'][-1]==float(target.end)
    assert row['continuation_features'][-1,0].expm1().item()==pytest.approx(127.,abs=.001)
    assert row['transition_exit_labels'][-1]


@pytest.mark.parametrize('kind,turn',[(WAIT,1),(EOS,2)])
def test_long_synthetic_note_loop_can_teach_real_rest_and_end(monkeypatch,kind,turn):
    from test_stream_musical_loop_recovery import LongestRotation
    ds=dense_dataset(monkeypatch)
    index=next(i for i,step in enumerate(ds.streams[0]) if step.kind==kind and step.start>100)
    target=ds.streams[0][index]
    observed=allow_synthetic(monkeypatch,{NOTE:(),WAIT:((0,index),) if kind==WAIT else (),EOS:((0,index),) if kind==EOS else ()})
    row=recovery._synthetic_long_exit_row(ds,None,LongestRotation(0),turn,Counter())
    assert row is not None
    assert observed[0][-1] is target and all(step.kind==NOTE for step in observed[0][:-1])
    assert observed[0][-2].end==target.start
    assert row['target_kind'][-1]==kind and row['target_second'][-1]==target.end_second
    assert row['transition_exit_labels'][-1]


def test_synthetic_history_does_not_receive_the_future_target_atom(monkeypatch):
    from test_stream_musical_loop_recovery import LongestRotation
    ds=dense_dataset(monkeypatch);index=600
    allow_synthetic(monkeypatch,{NOTE:((0,index),),WAIT:(),EOS:()})
    before=recovery._synthetic_long_exit_row(ds,None,LongestRotation(0),10,Counter())
    ds.streams[0][index]=replace(ds.streams[0][index],text='8')
    after=recovery._synthetic_long_exit_row(ds,None,LongestRotation(0),10,Counter())
    for key in ('history_atoms','history_kind','history_shape','history_delta','step_beat','continuation_features'):
        torch.testing.assert_close(before[key],after[key])
    assert not torch.equal(before['target_phase_mask'],after['target_phase_mask'])


def test_v4_long_sources_alternate_using_only_restored_cursors(monkeypatch):
    calls=[]
    model=torch.nn.Linear(1,1);model.config=SimpleNamespace(recovery_policy='greedy_mixed_v4')
    monkeypatch.setattr(recovery,'_short_exit_row',lambda *args:calls.append(('short',args[5])))
    def persistent(*args,**kwargs):calls.append(('persistent',kwargs['steps']));return None,{}
    monkeypatch.setattr(recovery,'_persistent_recovery_batch',persistent)
    monkeypatch.setattr(recovery,'_synthetic_long_exit_row',lambda *args:calls.append(('synthetic',args[3])))
    recovery.recovery_batch(None,model,'cpu',25,samples=2,steps=2)
    recovery.recovery_batch(None,model,'cpu',50,samples=2,steps=2)
    state=recovery.recovery_state(model)
    other=torch.nn.Linear(1,1);other.config=model.config;recovery.restore_recovery_state(other,state)
    recovery.recovery_batch(None,other,'cpu',75,samples=2,steps=2)
    _,diag=recovery.recovery_batch(None,other,'cpu',100,samples=2,steps=2)
    assert [row for row in calls if row[0]!='short']==[('persistent',8),('synthetic',0)]
    assert diag['synthetic_long_slots']==1
    assert recovery.recovery_state(other)['mix_cursor']==8


def test_synthetic_target_index_includes_trusted_wait_and_eos(monkeypatch):
    ds=dense_dataset(monkeypatch)
    monkeypatch.setattr(recovery,'_risk_exits',lambda _: {'gap':[(0,8)]})
    groups=recovery._synthetic_loop_targets(ds)
    assert groups[NOTE]==((0,8),)
    assert groups[WAIT] and groups[EOS]
    assert recovery._synthetic_loop_targets(ds) is groups
