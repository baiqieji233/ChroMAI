from collections import Counter
from dataclasses import replace
from fractions import Fraction as F
import random
from types import SimpleNamespace

import pytest
import torch

from maichart.composer import stream_augmentation as aug, stream_recovery as recovery
from maichart.composer.stream_data import StreamDataset, StreamStep, WAIT, NOTE
from maichart.event_timing import EventPoint, EventTempo


class LongestRotation(random.Random):
    def choices(self,population,weights=None,*,cum_weights=None,k=1):return [population[-1]]*k
    def randrange(self,*args):return 0


class ShortestRotation(LongestRotation):
    def choices(self,population,weights=None,*,cum_weights=None,k=1):return [population[0]]*k


def test_rotation_training_modes_are_versioned_and_do_not_change_old_recipes():
    assert aug._augmentation_modes(aug.LEGACY_INPUT_POLICY)==aug._LOOP_MODES
    assert aug._augmentation_modes(aug.PHASE_INPUT_POLICY)==(*aug._LOOP_MODES,'phase_shift')
    assert aug._augmentation_modes(aug.MUSICAL_INPUT_POLICY)[-2:]==('rotation','rotation_gap')
    for direction in (0,1):
        class Fixed(random.Random):
            def randrange(self,n):return direction if n==2 else 0
        texts=aug._loop_texts(None,[],0,12,'rotation',Fixed())
        expected=7 if direction==0 else 1
        assert {(int(b)-int(a))%8 for a,b in zip(texts,texts[1:])}=={expected}


def long_dataset(monkeypatch):
    from test_stream_augmentation import dataset
    dataset(monkeypatch)
    song=SimpleNamespace(audio=SimpleNamespace(duration_sec=60.),tempo=[EventTempo(F(0),120.)],first=0.)
    points=[EventPoint(F(i,2),str(i%8+1)) for i in range(200)]
    base=SimpleNamespace(songs=[(song,points,12.,1.,None,torch.zeros(16),(),None)])
    return StreamDataset(base=base,lead_in=128,targets=4,audio_grid_seconds=.05,
        history_window_policy='block_aligned_v2',continuation_policy='pattern_runs_v2')


@pytest.mark.parametrize('mode',['rotation','rotation_gap'])
def test_long_rotation_rows_keep_authored_targets_clock_and_training_budget(monkeypatch,mode):
    ds=long_dataset(monkeypatch);original=ds[32]
    changed,actual,count=aug.augment_stream_row(ds,32,rng=LongestRotation(0),mode=mode,
        input_policy=aug.MUSICAL_INPUT_POLICY,row=original)
    assert actual==mode and count==128
    for key in ('target_kind','target_delta','target_shape','target_phase_mask','target_details',
                'target_second','target_beat','step_beat','step_second','loss_mask'):
        torch.testing.assert_close(changed[key],original[key])
    torch.testing.assert_close(changed['sample_weight'].sum(),original['sample_weight'].sum(),rtol=0,atol=0)
    first_target=int(changed['loss_mask'].nonzero()[0])
    assert changed['continuation_features'][first_target,14].expm1().item()==pytest.approx(127.,abs=.001)
    assert not torch.equal(changed['history_atoms'],original['history_atoms'])


def test_joint_rotation_gap_never_inserts_wait_or_moves_the_target_to_fit():
    steps=[StreamStep(NOTE,F(i,4),F(i+1,4),i/8,(i+1)/8,str(i%8+1)) for i in range(8)]
    ds=SimpleNamespace(delta_vocab=SimpleNamespace(values=[F(0),F(1,4),F(1,2),F(1),F(2)]))
    with pytest.raises(ValueError,match='exact boundaries'):
        aug._same_gap_steps(ds,0,steps,0,4,8,random.Random(0),rotation_gap=True)
    assert len(steps)==8 and all(step.kind==NOTE for step in steps)


def _steps_at(beats):
    previous=F(0);result=[]
    for index,beat in enumerate(beats):
        beat=F(beat)
        result.append(StreamStep(NOTE,previous,beat,float(previous)/2,float(beat)/2,str(index%8+1)))
        previous=beat
    return result


def test_halfbeat_rotation_preserves_the_authored_bos_zero_note():
    from maichart.tempo_clock import TempoClock
    source=_steps_at([0,F(1,2),1,F(3,2),2,F(5,2)])
    ds=SimpleNamespace(delta_vocab=SimpleNamespace(values=[F(0),F(1,2),F(1),F(2)]),
        clocks=[TempoClock([EventTempo(F(0),120.)],0.)])
    changed=aug._same_gap_steps(ds,0,source,0,4,6,random.Random(0),rotation_gap=True)
    assert changed[0].start==changed[0].end==0
    assert all(step.end-step.start==F(1,2) for step in changed[1:4])
    assert changed[4:]==source[4:]


@pytest.mark.parametrize('beats,begin,start',[
    ([0,F(1,2),1,F(5,4),F(3,2),2],2,5),
    ([F(1,4),F(1,2),1,F(3,2),2,F(5,2)],0,4),
])
def test_halfbeat_rotation_cannot_introduce_an_interior_or_new_initial_zero(beats,begin,start):
    source=_steps_at(beats)
    ds=SimpleNamespace(delta_vocab=SimpleNamespace(values=[F(0),F(1,4),F(1,2),F(1),F(2)]))
    with pytest.raises(ValueError,match='exact boundaries'):
        aug._same_gap_steps(ds,0,source,begin,start,len(source),random.Random(0),rotation_gap=True)


def test_retimed_rotation_local_rebuild_matches_full_causal_replay(monkeypatch):
    from maichart.generation.sustain_state import SustainStateIndex
    bank=(torch.zeros(500,238),torch.arange(500,dtype=torch.float64)/20,
          torch.arange(500,dtype=torch.float64)/10)
    monkeypatch.setattr(StreamDataset,'_bank',lambda self,song:bank)
    song=SimpleNamespace(audio=SimpleNamespace(duration_sec=25.),tempo=[EventTempo(F(0),120.)],first=0.)
    beats=[F(i,2) for i in range(40)];beats[22]-=F(1,4)
    points=[EventPoint(beat,str(i%8+1)) for i,beat in enumerate(beats)]
    native=SustainStateIndex(points,song.tempo,song.first)
    base=SimpleNamespace(songs=[(song,points,12.,1.,native,torch.zeros(16),(),None)])
    ds=StreamDataset(base=base,lead_in=16,targets=4,audio_grid_seconds=.05,
        history_window_policy='block_aligned_v2',continuation_policy='pattern_runs_v2')
    row=ds[6]
    fast,_,_=aug.augment_stream_row(ds,6,rng=ShortestRotation(0),mode='rotation_gap',
        input_policy=aug.MUSICAL_INPUT_POLICY,row=row)
    assert fast['_augmentation_local_rebuild']
    monkeypatch.setattr(aug,'_native_prefix',lambda *args,**kwargs:None)
    slow,_,_=aug.augment_stream_row(ds,6,rng=ShortestRotation(0),mode='rotation_gap',
        input_policy=aug.MUSICAL_INPUT_POLICY,row=row)
    for key in ('history_atoms','history_delta','history_gap_beats','step_beat','step_second',
                'target_delta','target_second','target_beat','continuation_features',
                'transition_exit_mask','transition_exit_labels','action_state','action_state_mask'):
        torch.testing.assert_close(fast[key],slow[key])


def test_new_recovery_keeps_three_short_slots_per_eight_step_long_slot(monkeypatch):
    model=torch.nn.Linear(1,1);model.config=SimpleNamespace(recovery_policy='greedy_mixed_v3')
    calls=[]
    monkeypatch.setattr(recovery,'_short_exit_row',lambda *args:calls.append(('short',args[5])))
    def long(*args,**kwargs):calls.append(('long',kwargs['steps']));return None,{}
    monkeypatch.setattr(recovery,'_persistent_recovery_batch',long)
    recovery.recovery_batch(None,model,'cpu',25,samples=2,steps=2)
    _,diag=recovery.recovery_batch(None,model,'cpu',50,samples=2,steps=2)
    assert calls==[('short',0),('short',1),('short',2),('long',8)]
    assert diag['short_slots']==diag['long_slots']==1


def test_new_short_recovery_forms_four_event_motion_history_and_alternates_modes(monkeypatch):
    from test_composer_stream_recovery import _dataset
    ds=_dataset(monkeypatch);ds.continuation_policy='pattern_runs_v2'
    monkeypatch.setattr(recovery,'_risk_exits',lambda _: {'motion':[(0,5)]})
    monkeypatch.setattr(recovery,'_prepared_cache',lambda *args:None)
    calls=[]
    def fixed(dataset,model,context,source,guard,generator,device,cache,**options):
        calls.append(options['temperature']);return replace(source,text='8'),0
    monkeypatch.setattr(recovery,'_propose_fixed',fixed)
    model=torch.nn.Linear(1,1);model.config=SimpleNamespace(recovery_policy='greedy_mixed_v3')
    for cursor in (0,1):
        row=recovery._short_exit_row(ds,model,torch.device('cpu'),random.Random(0),
            torch.Generator().manual_seed(0),cursor,2,{},Counter())
        assert row is not None and row['step_beat'][-1]==float(ds.streams[0][5].start)
        assert row['target_second'][-1]==ds.streams[0][5].end_second
    assert calls==[0.]*4+[.65]*4


def test_persistent_greedy_choice_survives_resume_and_legacy_cap_is_retained(monkeypatch):
    from test_composer_stream_recovery import _dataset
    ds=_dataset(monkeypatch)
    monkeypatch.setattr(recovery,'_new_episode',lambda *args,**kwargs:dict(song=0,anchor=1,generated=[]))
    monkeypatch.setattr(recovery,'_guard',lambda *args:SimpleNamespace(accepts=lambda point:True))
    monkeypatch.setattr(recovery,'_encode_song',lambda *args:None)
    seen=[]
    def propose(dataset,model,song,prefix,guard,generator,device,**options):
        seen.append(options['temperature']);cursor=prefix[-1].end
        return StreamStep(WAIT,cursor,cursor+F(1,8),float(cursor)/2,float(cursor+F(1,8))/2),'ok'
    monkeypatch.setattr(recovery,'_propose',propose)
    model=torch.nn.Linear(1,1);model.config=SimpleNamespace(recovery_policy='greedy_mixed_v3')
    recovery._persistent_recovery_batch(ds,model,'cpu',0,samples=2,steps=1)
    state=recovery.recovery_state(model)
    other=torch.nn.Linear(1,1);other.config=model.config;recovery.restore_recovery_state(other,state)
    assert recovery.recovery_state(other)==state
    recovery._persistent_recovery_batch(ds,other,'cpu',25,samples=2,steps=1)
    assert seen==[0.,.65,0.,.65]
    episode=state['episodes'][0]
    episode['generated']=episode['generated']*128
    recovery.restore_recovery_state(other,state)
    state['policy']='mixed_exit_v2';other.config=SimpleNamespace(recovery_policy='mixed_exit_v2')
    with pytest.raises(ValueError,match='Invalid stream recovery episode'):
        recovery.restore_recovery_state(other,state)
