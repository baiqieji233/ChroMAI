from fractions import Fraction

import torch
from types import SimpleNamespace

from maichart.event_timing import EventPoint
from maichart.composer import stream_proposal as proposal


def test_rejected_time_is_not_resampled_and_context_uses_new_time(monkeypatch):
    calls=[]
    marker=torch.ones(3,8)
    def guarded(model,context,reference,beat,guard,generator,**kwargs):
        calls.append((context,beat))
        assert kwargs['motion_phase_distribution'] is marker
        return (None,8) if beat==Fraction(1,4) else (EventPoint(beat,'2'),0)
    monkeypatch.setattr(proposal,'sample_guarded_event',guarded)
    logits=torch.tensor([-torch.inf,10.,8.,7.])
    result=proposal.sample_timed_event(None,logits,[Fraction(0),Fraction(1,4),Fraction(1,2),Fraction(1)],
        Fraction(0),lambda index,beat:(index,beat),0,None,None,None,
        pointer_temperature=0,motion_phase_distribution=marker)
    assert result==(EventPoint(Fraction(1,2),'2'),2,8,1)
    assert calls==[((1,Fraction(1,4)),Fraction(1,4)),((2,Fraction(1,2)),Fraction(1,2))]
    assert logits.tolist()==[-torch.inf,10.,8.,7.]


def test_retry_budget_is_bounded_and_empty_support_never_calls_action(monkeypatch):
    calls=[]
    def denied(*args,**kwargs):calls.append(args[3]);return None,8
    monkeypatch.setattr(proposal,'sample_guarded_event',denied)
    deltas=[Fraction(i) for i in range(5)]
    result=proposal.sample_timed_event(None,torch.tensor([-torch.inf,4.,3.,2.,1.]),
        deltas,Fraction(0),lambda *_:None,0,None,None,None,pointer_temperature=0)
    assert result==(None,None,24,3) and calls==[Fraction(1),Fraction(2),Fraction(3)]
    calls.clear()
    assert proposal.sample_timed_event(None,torch.full((5,),-torch.inf),deltas,
        Fraction(0),lambda *_:None,0,None,None,None)==(None,None,0,0)
    assert not calls


def test_joint_retry_trace_keeps_time_until_action_budget_is_exhausted():
    draws=[];committed=[]
    def family(*args,**kwargs):
        draws.append(kwargs['temperature']);return 2 if len(draws)<=8 else 1
    def decode(context,**kwargs):
        return dict(shape=0,phase=0,family=kwargs['family_override'],details=[0],
            text='1h[4:1]' if kwargs['family_override']==2 else '2')
    model=SimpleNamespace(config=SimpleNamespace(event_family_policy='audio_semantic_v2'),
        sample_event_family=family,decode_event=decode)
    guard=SimpleNamespace(accepts=lambda p:p.note_text=='2',commit=committed.append,
        last_reason='test_occupied')
    contexts=[]
    def at_time(index,beat):
        contexts.append(beat);return dict(context=None,family_logits=torch.zeros(3))
    result=proposal.sample_timed_event(model,torch.tensor([-torch.inf,4.,3.]),
        [Fraction(0),Fraction(1,4),Fraction(1,2)],Fraction(0),at_time,
        0,None,guard,None,pointer_temperature=0,action_temperature=.65)
    assert result==(EventPoint(Fraction(1,2),'2'),2,8,1)
    assert contexts==[Fraction(1,4),Fraction(1,2)] and draws==[.65]*9
    trace=guard.last_timed_sampling_trace
    assert trace['retry_policy']=='joint_event_rejection_v2'
    assert [len(row['trials']) for row in trace['onsets']]==[8,1]
    assert trace['first_proposal']['family']==2 and trace['final_proposal']['family']==1
    assert trace['changed_time'] and trace['configuration_changed']
    assert trace['first_proposal']['rejection_reason']=='test_occupied'
    assert len(committed)==1
