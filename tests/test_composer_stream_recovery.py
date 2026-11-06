from fractions import Fraction as F
from types import SimpleNamespace

import pytest
import torch

from maichart.event_timing import EventPoint, EventTempo
from maichart.composer import stream_data, stream_recovery
from maichart.composer.stream_data import NOTE, WAIT, StreamDataset, StreamStep


def test_mixed_recovery_rotates_three_short_slots_per_persistent_slot(monkeypatch):
    model=torch.nn.Linear(1,1).train()
    model.config=SimpleNamespace(recovery_policy='mixed_exit_v2')
    calls=[]
    def short(dataset,model,device,rng,generator,cursor,steps,caches,diagnostics):
        calls.append(('short',cursor));diagnostics['new_episode']+=1
    def long(dataset,model,device,step,**kwargs):
        calls.append(('long',kwargs['samples']))
        return None,{'new_episode':1}
    monkeypatch.setattr(stream_recovery,'_short_exit_row',short)
    monkeypatch.setattr(stream_recovery,'_persistent_recovery_batch',long)
    stream_recovery.recovery_batch(None,model,'cpu',25,samples=2,steps=2)
    _,diag=stream_recovery.recovery_batch(None,model,'cpu',50,samples=2,steps=2)
    assert calls==[('short',0),('short',1),('short',2),('long',1)]
    assert diag['short_slots']==diag['long_slots']==1
    assert model.training
    saved=stream_recovery.recovery_state(model)
    other=torch.nn.Linear(1,1);other.config=model.config
    stream_recovery.restore_recovery_state(other,saved)
    assert stream_recovery.recovery_state(other)==saved
    assert saved['mix_cursor']==4 and saved['short_cursor']==3


def test_old_persistent_state_is_not_inherited_by_mixed_policy():
    model=torch.nn.Linear(1,1)
    model.config=SimpleNamespace(recovery_policy='mixed_exit_v2')
    old=dict(schema=stream_recovery.SCHEMA,episodes=[None],anchor_cursor=100)
    stream_recovery.restore_recovery_state(model,old)
    state=stream_recovery.recovery_state(model)
    assert state['policy']=='mixed_exit_v2'
    assert state['episodes']==[] and state['anchor_cursor']==0 and state['mix_cursor']==0


def test_short_recovery_supervises_original_exit_without_crossing_its_clock(monkeypatch):
    ds=_dataset(monkeypatch);ds.continuation_policy='pattern_runs_v2'
    monkeypatch.setattr(stream_recovery,'_risk_exits',lambda dataset:{'template':[(0,3)]})
    monkeypatch.setattr(stream_recovery,'_prepared_cache',lambda *args:None)
    seen=[]
    def fixed(dataset,model,context,source_step,guard,generator,device,cache):
        seen.append((source_step.start,source_step.end))
        # Disturb configuration while preserving every authored timestamp.
        from dataclasses import replace
        return replace(source_step,text='8'),0
    monkeypatch.setattr(stream_recovery,'_propose_fixed',fixed)
    from collections import Counter
    import random
    diag=Counter();model=torch.nn.Linear(1,1)
    row=stream_recovery._short_exit_row(ds,model,torch.device('cpu'),random.Random(0),
        torch.Generator().manual_seed(0),0,2,{},diag)
    assert len(seen)==2
    assert row['target_kind'][-1]==NOTE
    assert row['step_beat'][-1]==float(ds.streams[0][3].start)
    assert row['target_second'][-1]==ds.streams[0][3].end_second
    assert row['target_delta'][-1]==ds.delta_vocab.encode(ds.streams[0][3].end-ds.streams[0][3].start)
    assert diag['authored_exit_supervised']==diag['new_episode']==1
    assert not diag['overshot']


def _dataset(monkeypatch):
    class EmptyState:
        def __init__(self,*args): self.points=[]
        def append(self,point): self.points.append(point)
        def at(self,*args,**kwargs): return torch.zeros(1,8),torch.ones(1,dtype=torch.bool),{}
    monkeypatch.setattr(stream_data,'STATE_DIM',8)
    monkeypatch.setattr(stream_data,'SustainStateIndex',EmptyState)
    monkeypatch.setattr(stream_data,'IncrementalState',EmptyState)
    bank=(torch.zeros(120,238),torch.arange(120,dtype=torch.float64)/20,torch.arange(120,dtype=torch.float64)/10)
    monkeypatch.setattr(StreamDataset,'_bank',lambda self,song:bank)
    song=SimpleNamespace(audio=SimpleNamespace(duration_sec=6.),tempo=[EventTempo(F(0),120.)],first=0.)
    points=[EventPoint(F(i),str(i%8+1)) for i in range(6)]
    base=SimpleNamespace(songs=[(song,points,12.,1.,None,torch.zeros(16),(),None)])
    return StreamDataset(base=base,lead_in=4,targets=4,audio_grid_seconds=.05)


def test_recovery_uses_actual_prefix_and_restores_persistent_state(monkeypatch):
    ds=_dataset(monkeypatch)
    monkeypatch.setattr(stream_recovery,'_anchors',lambda ds: ([(0,1)],[],[]))
    class Guard:
        def accepts(self,point): return True
    monkeypatch.setattr(stream_recovery,'_guard',lambda song,prefix,anchor:Guard())
    calls=[]
    monkeypatch.setattr(stream_recovery,'_encode_song',lambda *args: calls.append(1))
    def propose(dataset,model,song,prefix,guard,generator,device,**kwargs):
        cursor=prefix[-1].end
        return StreamStep(WAIT,cursor,cursor+F(1,2),float(cursor)/2,float(cursor+F(1,2))/2),'ok'
    monkeypatch.setattr(stream_recovery,'_propose',propose)
    model=torch.nn.Linear(1,1).train()
    batch,diagnostics=stream_recovery.recovery_batch(ds,model,'cpu',8,samples=2,steps=1)
    assert model.training and diagnostics['supervised'] == 2 and len(calls) == 1
    last=batch['loss_mask'][0].nonzero().item()
    assert batch['history_kind'][0,last] == WAIT
    assert batch['target_kind'][0,last] == NOTE
    assert batch['target_delta'][0,last] == ds.delta_vocab.encode(F(1,2))
    assert batch['audio_plan_weight'].sum() == 0
    state=stream_recovery.recovery_state(model)
    restored=torch.nn.Linear(1,1)
    stream_recovery.restore_recovery_state(restored,state)
    assert stream_recovery.recovery_state(restored) == state
    stream_recovery.restore_recovery_state(restored,{'schema':stream_recovery.SCHEMA,'episodes':[None]})


def test_illegal_reference_is_not_supervised(monkeypatch):
    ds=_dataset(monkeypatch)
    monkeypatch.setattr(stream_recovery,'_anchors',lambda ds: ([(0,1)],[],[]))
    class Reject:
        def accepts(self,point): return False
    monkeypatch.setattr(stream_recovery,'_guard',lambda song,prefix,anchor:Reject())
    monkeypatch.setattr(stream_recovery,'_encode_song',lambda *args:None)
    monkeypatch.setattr(stream_recovery,'_propose',lambda dataset,model,song,prefix,guard,generator,device,**kwargs:
        (StreamStep(WAIT,F(0),F(1,2),0.,.25),'ok'))
    model=torch.nn.Linear(1,1).train()
    batch,diag=stream_recovery.recovery_batch(ds,model,'cpu',9,samples=1,steps=1)
    assert batch is None and diag['reference_illegal'] == 1
    assert model.training
    with pytest.raises(ValueError):
        stream_recovery.restore_recovery_state(model,{'schema':stream_recovery.SCHEMA,'episodes':[{'song':0}]})


def test_causal_recovery_keeps_time_when_authored_configuration_is_illegal(monkeypatch):
    ds=_dataset(monkeypatch)
    ds.recovery_target_policy='causal_correspondence_v1'
    prefix=[ds.streams[0][0],StreamStep(WAIT,F(0),F(1,2),0.,.25)]
    class Reject:
        def accepts(self,point):return False
    target,reason=stream_recovery._next_reference(ds,0,prefix,Reject(),
        target_policy='causal_correspondence_v1')
    assert reason=='ok' and target.kind==NOTE
    target,trusted,reason=stream_recovery._qualified_continuation(ds,0,prefix,target,
        Reject(),1,1,'causal_correspondence_v1')
    assert not trusted and reason=='time_only'
    row=stream_data.recovery_sample(ds,0,prefix,target,configuration_trusted=trusted)
    packed=stream_data.collate_stream([row])
    assert packed['configuration_mask'][0,-1].item() is False
    assert packed['target_kind'][0,-1].item()==NOTE
    assert not packed['_detail_groups'] and packed['_literal_indices'].numel()==0


def test_rotation_requires_clear_old_boundary_even_with_matching_generated_notes(monkeypatch):
    ds=_dataset(monkeypatch)
    source=ds.streams[0]
    from dataclasses import replace
    from maichart.composer.stream_target_correspondence import rotated_text
    anchor=1;target_index=4
    prefix=list(source[:anchor])+[replace(step,text=rotated_text(step.text,2))
        for step in source[anchor:target_index]]
    class Guard:
        def accepts(self,point):return True
    monkeypatch.setattr(stream_recovery,'rotation_boundary_clear',lambda *args:False)
    target,trusted,reason=stream_recovery._qualified_continuation(ds,0,prefix,
        source[target_index],Guard(),anchor,target_index,'causal_correspondence_v1')
    assert not trusted and reason=='time_only' and target.text==source[target_index].text


def test_causal_recovery_rotates_target_after_proved_generated_suffix(monkeypatch):
    ds=_dataset(monkeypatch)
    source=ds.streams[0]
    from dataclasses import replace
    from maichart.composer.stream_target_correspondence import rotated_text
    anchor=1;target_index=4
    prefix=list(source[:anchor])+[replace(step,text=rotated_text(step.text,2))
        for step in source[anchor:target_index]]
    checked=[]
    class Guard:
        def accepts(self,point):checked.append(point.note_text);return True
    monkeypatch.setattr(stream_recovery,'rotation_boundary_clear',lambda *args:True)
    target,trusted,reason=stream_recovery._qualified_continuation(ds,0,prefix,
        source[target_index],Guard(),anchor,target_index,'causal_correspondence_v1')
    expected=rotated_text(source[target_index].text,2)
    assert trusted and reason=='rotation_correspondence'
    assert target.text==expected and checked==[expected]
    row=stream_data.recovery_sample(ds,0,prefix,target,configuration_trusted=trusted,
        correspondence=reason)
    assert row['configuration_mask'][-1].item()
    assert row['target_shape'][-1].item()==ds.codec.encode(expected)['shape']


def test_actual_sampled_note_is_guarded_before_commit(monkeypatch):
    ds=_dataset(monkeypatch)
    class Context(stream_data.StreamInferenceContext):
        def __init__(self,song,codec,vocab,condition,**kwargs):
            super().__init__(song,codec,vocab,condition,**kwargs)
        def batch(self,prefix,cursor):
            super().batch(prefix,cursor)
            return dict(song_audio=torch.zeros(1,2,238),song_mask=torch.ones(1,2,dtype=torch.bool),
                song_seconds=torch.zeros(1,2),song_beats=torch.zeros(1,2),
                history_delta=torch.zeros(1,1,dtype=torch.long),step_second=torch.zeros(1,1))
    monkeypatch.setattr(stream_recovery,'StreamInferenceContext',Context)
    delta=ds.delta_vocab.encode(F(1,2))
    class Model:
        config=SimpleNamespace(draft_policy='off')
        def encode_audio(self,*args): return None
        def __call__(self,batch,audio_cache=None):
            kind=torch.full((1,1,5),-100.);kind[0,0,NOTE]=100.
            values=torch.full((1,1,len(ds.delta_vocab.values)+1),-100.);values[0,0,delta]=100.
            return dict(kind_logits=kind,delta_logits=values,reference_phase=torch.zeros(1,1,dtype=torch.long))
        def action_context_for_delta(self,*args,**kwargs): return torch.zeros(1,1,8)
        def decode_event(self,*args,**kwargs): return {'text':'2'}
    class Guard:
        def __init__(self,accepted):self.accepted=accepted;self.committed=[];self.calls=0
        def accepts(self,point):self.calls+=1;return self.accepted
        def commit(self,point):self.committed.append(point)
    prefix=ds.streams[0][:1]
    generator=torch.Generator().manual_seed(1)
    denied=Guard(False)
    proposal,reason=stream_recovery._propose(ds,Model(),0,prefix,denied,generator,torch.device('cpu'))
    assert proposal.kind == WAIT and proposal.end == F(2)
    assert reason == 'rejected_configurations' and not denied.committed and denied.calls == 24
    accepted=Guard(True)
    proposal,reason=stream_recovery._propose(ds,Model(),0,prefix,accepted,generator,torch.device('cpu'))
    assert reason == 'ok' and proposal.end == F(1,2) and len(accepted.committed) == 1


def test_sampling_can_cross_reference_and_wait_eos_follow_audio_boundary(monkeypatch):
    ds=_dataset(monkeypatch)
    class Context(stream_data.StreamInferenceContext):
        def __init__(self,song,codec,vocab,condition,**kwargs):super().__init__(song,codec,vocab,condition,**kwargs)
        def batch(self,prefix,cursor):
            super().batch(prefix,cursor)
            return dict(song_audio=torch.zeros(1,2,238),song_mask=torch.ones(1,2,dtype=torch.bool),
                song_seconds=torch.zeros(1,2),song_beats=torch.zeros(1,2),
                history_delta=torch.zeros(1,1,dtype=torch.long),step_second=torch.zeros(1,1))
    monkeypatch.setattr(stream_recovery,'StreamInferenceContext',Context)
    class Model:
        config=SimpleNamespace(draft_policy='off')
        kind=NOTE
        delta=ds.delta_vocab.encode(F(2))
        def encode_audio(self,*args):return None
        def __call__(self,batch,audio_cache=None):
            kind=torch.full((1,1,5),-100.);kind[0,0,self.kind]=100.
            # A high WAIT score is used to show EOS is blocked before the tail.
            if self.kind == stream_recovery.EOS:kind[0,0,WAIT]=50.
            values=torch.full((1,1,len(ds.delta_vocab.values)+1),-100.);values[0,0,self.delta]=100.
            return dict(kind_logits=kind,delta_logits=values,reference_phase=torch.zeros(1,1,dtype=torch.long))
        def action_context_for_delta(self,*args,**kwargs):return torch.zeros(1,1,8)
        def decode_event(self,*args,**kwargs):return {'text':'2'}
    class Guard:
        def accepts(self,point):return True
        def commit(self,point):pass
    model=Model();generator=torch.Generator().manual_seed(2);prefix=ds.streams[0][:1]
    # The next authored event is at beat 1. A real sampled beat 2 must survive.
    proposal,reason=stream_recovery._propose(ds,model,0,prefix,Guard(),generator,torch.device('cpu'))
    assert reason == 'ok' and proposal.kind == NOTE and proposal.end == F(2)
    target,reason=stream_recovery._next_reference(ds,0,prefix+[proposal],Guard())
    assert reason == 'ok' and target.kind == NOTE and target.end == F(3)
    model.kind=WAIT;model.delta=ds.delta_vocab.encode(F(1,2))
    proposal,_=stream_recovery._propose(ds,model,0,prefix,Guard(),generator,torch.device('cpu'))
    assert proposal.kind == WAIT and proposal.end-proposal.start == 2
    model.kind=stream_recovery.EOS
    proposal,_=stream_recovery._propose(ds,model,0,prefix,Guard(),generator,torch.device('cpu'))
    assert proposal.kind == WAIT  # EOS masked while >2 beats remain.
    tail=[StreamStep(WAIT,F(i),F(i+2),i/2,(i+2)/2) for i in range(0,10,2)]
    proposal,reason=stream_recovery._propose(ds,model,0,tail,Guard(),generator,torch.device('cpu'))
    assert proposal is None and reason == 'sampled_eos'
