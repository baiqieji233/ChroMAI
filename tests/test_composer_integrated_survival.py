import math
import pytest
import torch

from maichart.composer.event_survival import (
    integrated_survival, redistribute_survival, fuse_survival_rates,
)


def test_fixed_grid_integral_conserves_tail_across_candidate_refinement_and_restarts():
    rates=torch.tensor([[0.,0.,0.,0.,8.,8.]])
    edges=torch.arange(7,dtype=torch.float64)[None]*.1
    def query(seconds,start=0.):
        stamps=torch.tensor([seconds],dtype=torch.float64)
        return integrated_survival(rates,edges,stamps,torch.ones_like(stamps,dtype=torch.bool),torch.tensor([start]))
    first,tail,_=query([.2,.4,.6])
    refined,refined_tail,_=query([.1,.2,.3,.4,.5,.6])
    torch.testing.assert_close(tail,refined_tail)
    assert float(first[0,0].exp())<1e-20 and float(first[0,1].exp())<1e-20
    torch.testing.assert_close(first.exp().sum()+tail.exp().sum(),torch.tensor(1.))
    _,left,_=query([.45])
    _,right,_=query([.6],.45)
    torch.testing.assert_close(left+right,tail,atol=1e-6,rtol=1e-6)


def test_shape_and_history_cannot_spend_waiting_mass():
    rates=torch.full((1,4),.1)
    seconds=torch.tensor([[.1,.2,.4]])
    base,tail,_=integrated_survival(rates,torch.tensor([[0.,.1,.2,.3,.4]]),
        seconds,torch.ones_like(seconds,dtype=torch.bool),torch.zeros(1))
    delta=torch.tensor([[20.,-20.,5.]],requires_grad=True)
    actual,rest=redistribute_survival(base,tail,delta,torch.ones_like(seconds,dtype=torch.bool))
    torch.testing.assert_close(rest,tail,atol=0,rtol=0)
    torch.testing.assert_close(actual.exp().sum(),1.-tail.exp().sum(),atol=1e-7,rtol=1e-5)
    (-actual[:,0].mean()).backward()
    assert torch.isfinite(delta.grad).all()


def test_padding_and_exact_cursor_exposure_are_respected():
    rate=torch.ones(1,4,requires_grad=True)
    edges=torch.tensor([[1.,1.05,1.1,1.1,1.1]],dtype=torch.float64)
    valid=torch.tensor([[True,True,False]])
    first,tail,exposure=integrated_survival(rate,edges,torch.tensor([[1.075,1.1,1.1]]),valid,torch.tensor([1.025]))
    assert float(tail)==pytest.approx(-.075,abs=1e-6)
    assert float(exposure.sum())==pytest.approx(.075,abs=1e-6)
    assert float(first[0,2].exp())==0.
    (-tail.mean()).backward()
    assert not rate.grad[0,2:].any()


def test_existing_rest_plan_softly_reduces_rate_without_hard_silencing():
    lograte=torch.full((1,2),math.log1p(5.))
    plan=torch.zeros(1,2,6)
    plan[...,0]=math.log1p(5.)
    busy=fuse_survival_rates(lograte,plan)
    plan[...,5]=.9
    rest=fuse_survival_rates(lograte,plan)
    torch.testing.assert_close(busy,torch.full_like(busy,5.))
    assert torch.all(rest>0) and torch.all(rest<busy*.03)


def test_dense_poisson_gradients_use_the_same_rate_and_exclude_unknown_cells():
    from test_composer_survival_training import _loss_inputs
    from maichart.composer.train import _loss
    output,batch=_loss_inputs()
    rates=torch.tensor([[2.,2.,2.]],requires_grad=True)
    output['survival_grid_rates']=rates
    batch.update(survival_widths=torch.full((1,3),.05),
                 survival_count_target=torch.tensor([[0.,1.,0.]]),
                 survival_count_weight=torch.tensor([[1.,1.,0.]]))
    loss,metrics=_loss(output,batch,2)
    loss.backward()
    assert rates.grad[0,0]>0 and rates.grad[0,1]<0 and rates.grad[0,2]==0
    assert float(metrics['rate_loss_contribution'])>0


def test_missed_rest_classification_keeps_its_gradient_when_probability_is_low():
    from test_composer_survival_training import _loss_inputs
    from maichart.composer.train import _loss
    output,batch=_loss_inputs()
    logit=torch.tensor([[-5.]],requires_grad=True)
    output['plan']=torch.cat((torch.zeros(1,1,5),logit.sigmoid()[...,None]),-1)
    output['survival_grid_rates']=torch.ones(1,1)
    batch.update(planner_target=torch.tensor([[[0.,0.,0.,0.,0.,1.]]]),planner_weight=torch.ones(1,1))
    loss,_=_loss(output,batch,2)
    loss.backward()
    # Original plan weight .2 / six channels is retained. A wrong confident
    # prediction must not lose another sigmoid-derivative factor.
    expected=.2/6*(torch.sigmoid(logit.detach())-1.)
    torch.testing.assert_close(logit.grad,expected)


def test_local_onset_supervision_keeps_high_confidence_gradient_independent_of_rate():
    from test_composer_survival_training import _loss_inputs
    from maichart.composer.train import _loss
    output,batch=_loss_inputs()
    direct=torch.tensor([[8.,-8.]],requires_grad=True)
    output['audio_timing_log_probs']=direct
    output['audio_next_event_log_probs']=torch.tensor([[-.1,-10.]])
    _,metrics=_loss(output,batch,2)
    assert float(metrics['audio_pointer_loss'])>15.9
    metrics_loss=torch.nn.functional.cross_entropy(direct,batch['candidate_target'])
    metrics_loss.backward()
    assert direct.grad[0,1]<-.99


def test_acoustic_confidence_reorders_positions_without_overriding_time_band_mass():
    base=torch.tensor([[.05,.15,.3]]).log()
    tail=torch.tensor([.5]).log()
    valid=torch.ones(1,3,dtype=torch.bool)
    bins=torch.tensor([[0,0,1]])
    delta=torch.tensor([[8.,-8.,100.]],requires_grad=True)
    result,waiting=redistribute_survival(base,tail,delta,valid,bins=bins)
    torch.testing.assert_close(result.exp()[0,:2].sum(),torch.tensor(.2),atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(result.exp()[0,2],torch.tensor(.3),atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(waiting,tail,atol=0,rtol=0)
    (-result[0,1]).backward()
    assert delta.grad[0,1]<-.99
