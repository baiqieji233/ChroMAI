from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from maichart.composer.stream_actions import refresh_family_log_probs
from maichart.composer.stream_model import StreamComposer, NOTE, WAIT


def test_shared_refresh_kernel_has_the_audio_stationary_marginal():
    p=torch.tensor([.7,.2,.1],dtype=torch.float32)
    rho=.65
    kernel=refresh_family_log_probs(p.log()[None].expand(3,-1),
        torch.full((3,),float(torch.logit(torch.tensor(rho)))),torch.arange(1,4),torch.ones(3,dtype=torch.bool)).exp()
    torch.testing.assert_close(kernel.sum(-1),torch.ones(3))
    torch.testing.assert_close(p@kernel,p)
    torch.testing.assert_close(kernel.diagonal(),rho+(1-rho)*p)


def test_music_changes_produce_a_lagged_distribution_not_an_instantaneous_quota():
    previous=torch.tensor([.8,.1,.1])
    audio=torch.tensor([.1,.2,.7])
    rho=.6
    kernel=refresh_family_log_probs(audio.log()[None].expand(3,-1),
        torch.full((3,),float(torch.logit(torch.tensor(rho)))),torch.arange(1,4),torch.ones(3,dtype=torch.bool)).exp()
    actual=previous@kernel
    torch.testing.assert_close(actual,rho*previous+(1-rho)*audio)
    assert not torch.allclose(actual,audio)


def test_fresh_draw_contains_previous_type_and_gate_uses_mixture_nll():
    gate=torch.tensor([0.],requires_grad=True)
    lp=refresh_family_log_probs(torch.tensor([[.25,.75]]).log(),gate,
        torch.tensor([1]),torch.tensor([True]))
    torch.testing.assert_close(lp[0,0].exp(),torch.tensor(.625))
    (-lp[0,0]).backward()
    torch.testing.assert_close(gate.grad,torch.tensor([-.3]))
    # BCE(same=1) would give -0.5 and incorrectly double-count fresh copies.


def test_audio_budget_trains_proposal_while_mixture_trains_continuity():
    logits=torch.tensor([[.2,-.4,.6]],requires_grad=True)
    gate=torch.tensor([.3],requires_grad=True)
    previous=torch.tensor([1]);valid=torch.tensor([True]);target=torch.tensor([2])
    lp=refresh_family_log_probs(logits,gate,previous,valid,detach_audio=True)
    mixture=-lp[0,2]
    proposal=F.cross_entropy(logits,target)
    expected=torch.autograd.grad(.25*proposal,logits,retain_graph=True)[0]
    (.75*mixture+.25*proposal).backward()
    torch.testing.assert_close(logits.grad,expected)
    assert gate.grad.abs().sum()>0
    joined=torch.tensor([[.2,-.4,.6]],requires_grad=True)
    (-refresh_family_log_probs(joined,gate.detach(),previous,valid,detach_audio=False)[0,2]).backward()
    assert joined.grad.abs().sum()>0


def test_bos_uses_only_audio_and_extreme_gates_remain_finite_fp32():
    logits=torch.tensor([[0.,1.,-1.]],dtype=torch.bfloat16).expand(3,-1)
    gate=torch.tensor([-1000.,0.,1000.],dtype=torch.bfloat16)
    previous=torch.tensor([1,2,3]);none=torch.zeros(3,dtype=torch.bool)
    pure=refresh_family_log_probs(logits,gate,previous,none)
    torch.testing.assert_close(pure,logits.float().log_softmax(-1))
    mixed=refresh_family_log_probs(logits,gate,previous,~none)
    assert mixed.dtype==torch.float32 and torch.isfinite(mixed).all()
    torch.testing.assert_close(mixed.exp().sum(-1),torch.ones(3))


def model_fixture():
    from test_stream_audio_semantics import fixture
    old,batch,cache,seconds=fixture()
    model=StreamComposer(replace(old.config,event_family_policy='audio_prior_transition_v3'),old.codec).eval()
    return model,batch,cache,seconds


def test_continuity_evidence_cannot_read_the_previous_family_identity():
    model,batch,cache,seconds=model_fixture()
    with torch.no_grad():model.family_continuity_head.weight.normal_(0,.2)
    changed=dict(batch,history_atoms=batch['history_atoms'].clone())
    changed['history_atoms'][0,1:,0]=model.codec._atoms['C']
    a=model.family_context_at(cache,seconds,torch.tensor([0]),batch)
    b=model.family_context_at(cache,seconds,torch.tensor([0]),changed)
    torch.testing.assert_close(a['audio_logits'],b['audio_logits'],atol=0,rtol=0)
    torch.testing.assert_close(a['continuity_logits'],b['continuity_logits'],atol=0,rtol=0)
    assert not torch.equal(a['previous_family'],b['previous_family'])
    assert not torch.equal(a['full_logits'],b['full_logits'])
    assert not hasattr(model,'family_history_projection')


def test_previous_note_selection_is_causal_and_wait_retains_elapsed_origin():
    model,batch,_,_=model_fixture()
    changed=dict(batch,history_kind=batch['history_kind'].clone())
    changed['history_kind'][0,2]=WAIT
    previous,seconds,valid=model._family_previous_note(changed)
    assert previous[0,:3].tolist()==[0,1,1]
    torch.testing.assert_close(seconds[0,:3],torch.tensor([0.,.25,.25],dtype=torch.float64))
    assert valid[0,:3].tolist()==[False,True,True]
    later=dict(changed,history_atoms=changed['history_atoms'].clone())
    later['history_atoms'][0,3,0]=model.codec._atoms['C']
    p2,s2,v2=model._family_previous_note(later)
    torch.testing.assert_close(previous[:,:3],p2[:,:3])
    torch.testing.assert_close(seconds[:,:3],s2[:,:3])


def test_v3_cannot_silently_skip_the_transition_by_passing_only_a_vector():
    model,_,_,_=model_fixture()
    with pytest.raises(ValueError,match='requires mixed family_logits'):
        model.sample_event_family(torch.zeros(16))
    with pytest.raises(ValueError,match='requires mixed family_logits'):
        model.decode_event(torch.zeros(16),family_context=torch.zeros(16),temperature=0.)


def test_model_uses_audio_budget_to_choose_gradient_contract():
    model,batch,cache,seconds=model_fixture()
    fc=model.family_context_at(cache,seconds,torch.tensor([0]),batch)
    (-fc['full_logits'][0,1,7]).backward(retain_graph=True)
    assert model.audio_event_family_head.weight.grad is None
    assert model.family_continuity_head.weight.grad.abs().sum()>0
    model.zero_grad(set_to_none=True)
    model.config=replace(model.config,event_family_audio_weight=0.)
    fc=model.family_context_at(cache,seconds,torch.tensor([0]),batch)
    (-fc['full_logits'][0,1,7]).backward()
    assert model.audio_event_family_head.weight.grad.abs().sum()>0


def test_candidate_gap_clock_and_plan_rates_reach_the_audio_family_context():
    model,batch,cache,seconds=model_fixture()
    with torch.no_grad():
        model.plan_head.weight.zero_()
        model.family_rhythm_projection.weight.normal_(0,.1)
        model.audio_event_family_head.weight.normal_(0,.1)
    values=torch.tensor([.2,torch.log(torch.expm1(torch.tensor(2.))),torch.log(torch.expm1(torch.tensor(3.)))])
    cache={**cache,'plan_base_logits':values[None,None].expand(1,cache['local'].size(1),-1)}
    actual=model.family_context_at(cache,seconds,torch.tensor([0]),batch)
    features=actual['rhythm_features']
    assert features.shape==(1,4,19)
    torch.testing.assert_close(features[...,14],torch.log1p(seconds.float()*2-batch['step_beat'].float()))
    torch.testing.assert_close(features[...,15],torch.log1p(seconds.float()-batch['step_second'].float()))
    torch.testing.assert_close(features[...,16],torch.full((1,4),float(torch.sigmoid(torch.tensor(.2)))))
    torch.testing.assert_close(features[...,17],torch.full((1,4),float(torch.log1p(torch.tensor(2.)))))
    torch.testing.assert_close(features[...,18],torch.full((1,4),float(torch.log1p(torch.tensor(3.)))))
    changed=dict(batch,step_beat=batch['step_beat']-.25,step_second=batch['step_second']-.125)
    other=model.family_context_at(cache,seconds,torch.tensor([0]),changed)
    assert not torch.equal(actual['audio_logits'],other['audio_logits'])
    assert not features[...,16:].requires_grad
