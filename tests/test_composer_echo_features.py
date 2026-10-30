from fractions import Fraction
from types import SimpleNamespace
import torch
import pytest
from maichart.composer.phase_echo import phase_echo_features,ECHO_LAGS,ECHO_PERMUTATIONS
from maichart.composer.echo_metrics import EchoMetrics


@pytest.mark.parametrize('lag',[2,3,4,6,8,12,16])
@pytest.mark.parametrize('transform',range(4))
def test_evidence_and_exact_source(lag,transform):
    beats=[Fraction(i,4) for i in range(5)]
    events=[(b,i) for i,b in enumerate(beats)]
    events += [(b+lag,ECHO_PERMUTATIONS[transform][i]) for i,b in enumerate(beats[:4])]
    events.sort()
    lanes=torch.zeros(1,len(events)+2,8,dtype=torch.bool)
    ages=torch.zeros(1,len(events)+2)
    for n,(b,lane) in enumerate(events):
        lanes[0,n,lane]=True;ages[0,n]=float(events[-1][0]-b)
    valid=torch.arange(len(events)+2)[None]<len(events)
    f=phase_echo_features(lanes,ages,valid,torch.tensor([[.25,float(lag)+.1]]))
    k=ECHO_LAGS.index(lag)
    assert f['evidence'][0,k,transform]==4
    assert f['rhythm_evidence'][0,k]==4
    assert f['candidate_source'][0,0,k,4]
    assert f['rhythm_prediction'][0,0,k]
    assert not f['has_candidate_source'][0,1,k]


def test_button_history_skips_touch_but_rhythm_uses_touch():
    beats=[0,.25,.5,.75,4,4.25,4.5,4.75,4.875]
    lanes=torch.zeros(1,len(beats),8,dtype=torch.bool)
    for i in range(8):lanes[0,i,i%4]=True
    ages=torch.tensor([[beats[-1]-b for b in beats]])
    f=phase_echo_features(lanes,ages,torch.ones_like(ages,dtype=torch.bool),torch.tensor([[.125]]))
    assert f['evidence'][0,ECHO_LAGS.index(4),0]==4
    assert f['rhythm_evidence'][0,ECHO_LAGS.index(4)]==3


def test_empty_history_and_zero_age_padding():
    f=phase_echo_features(torch.zeros(2,0,8,dtype=torch.bool),torch.zeros(2,0),
                          torch.zeros(2,0,dtype=torch.bool),torch.ones(2,3))
    assert not f['candidate_source'].any()
    f=phase_echo_features(torch.ones(1,2,8,dtype=torch.bool),torch.zeros(1,2),
                          torch.zeros(1,2,dtype=torch.bool),torch.tensor([[2.,3.,4.]]))
    assert not f['has_candidate_source'].any()


def test_metrics_strongest_tie_break_and_probability_mass():
    heads=torch.zeros(9,8,dtype=torch.bool)
    heads[1:]=torch.eye(8,dtype=torch.bool)
    model=SimpleNamespace(action_factors=SimpleNamespace(atom_head_lanes=heads),config=SimpleNamespace(audio_dim=0))
    # Sixteen quarter-note events use a four-lane cycle; lag2 and lag3 have
    # transformed matches, tie order must still be deterministic.
    history=torch.tensor([[[i%4+1] for i in range(20)]])
    ages=torch.arange(19,-1,-1).float()[None]*.25
    batch=dict(history=history,history_beat_ages=ages,history_padding_mask=torch.zeros(1,20,dtype=torch.bool),
               candidate_features=torch.tensor([[[.25],[.5]]]),candidate_target=torch.tensor([0]),
               candidate_padding_mask=torch.zeros(1,2,dtype=torch.bool),action_target=torch.tensor([[1,0]]))
    output=dict(action_logits=torch.zeros(1,2,9),candidate_logits=torch.zeros(1,2))
    metric=EchoMetrics();metric.update(model,batch,output)
    r=metric.report()['echo_copy_mass_by_E']['4']
    assert r['count']==1
    assert r['model_mass']==pytest.approx(1/9)
    assert r['authored_copy_rate']==1
