"""Closed-form owner and budget checks for persistent draft execution."""
import pytest
import torch

from maichart.composer.stream_phrase_execution import draft_joint_sequence_loss


def owners():
    return dict(target_kind=torch.tensor([[2,2,3]]),
        draft_departure_group=torch.tensor([[0,0,1]]),
        draft_landing_group=torch.tensor([[0,1,1]]),
        draft_loss_group_count=2,draft_loss_group_rows=torch.tensor([0,0]),
        draft_loss_group_positions=torch.tensor([0,1]),draft_loss_group_side=torch.tensor([0,1]),
        draft_loss_group_complete=torch.tensor([False,True]))


def test_one_mode_exactly_restores_weighted_time_and_note_mark_budget():
    time=torch.tensor([[[1.],[2.],[3.]]],requires_grad=True)
    mark=torch.tensor([[[4.],[5.],[6.]]],requires_grad=True)
    weight=torch.tensor([[1.,.5,2.]])
    prior=torch.randn(1,3,1)
    loss,metrics=draft_joint_sequence_loss(time,mark,prior,prior,owners(),weight,weight,3.5)
    expected=((time[...,0]*weight).sum()+(mark[...,0]*weight*torch.tensor([[1,1,0]])).sum())/3.5
    torch.testing.assert_close(loss,expected,rtol=0,atol=0)
    loss.backward()
    assert mark.grad[0,2,0]==0
    torch.testing.assert_close(metrics['draft_group_weight_mass'],torch.tensor(5./3.5))


def test_crossing_note_lands_in_new_phrase_without_reusing_old_mode():
    time=torch.tensor([[[.1,2.],[.2,1.5],[2.,.3]]],requires_grad=True)
    mark=torch.tensor([[[.1,1.],[3.,.1],[999.,999.]]],requires_grad=True)
    departure=torch.tensor([[[2.,-1.],[7.,-3.],[-3.,7.]]],requires_grad=True)
    landing=torch.tensor([[[-5.,5.],[-2.,2.],[9.,-9.]]],requires_grad=True)
    weight=torch.tensor([[1.,.5,2.]])
    loss,metrics=draft_joint_sequence_loss(time,mark,departure,landing,owners(),weight,weight,3.5)
    # Old phrase: first two decisions + first mark. New phrase: final
    # decision + the earlier crossing NOTE's mark. W is 2.5 for each owner.
    old=time[0,0]+.5*time[0,1]+mark[0,0]
    new=2*time[0,2]+.5*mark[0,1]
    expected=(-2.5*torch.logsumexp(departure[0,0].log_softmax(-1)-old/2.5,0)
              -2.5*torch.logsumexp(landing[0,1].log_softmax(-1)-new/2.5,0))/3.5
    torch.testing.assert_close(loss,expected)
    loss.backward()
    assert landing.grad[0,1].abs().sum()>0
    assert departure.grad[0,0].abs().sum()>0
    assert mark.grad[0,2].abs().sum()==0
    assert metrics['draft_complete_phrase_groups']==1
    assert metrics['draft_partial_phrase_groups']==1


def test_waits_do_not_add_fictional_zero_mark_exposure():
    batch=owners();batch['target_kind'].fill_(3)
    time=torch.ones(1,3,2)
    mark=torch.zeros_like(time)
    priors=torch.zeros_like(time)
    weight=torch.tensor([[1.,.5,2.]])
    _,metrics=draft_joint_sequence_loss(time,mark,priors,priors,batch,weight,weight,1.)
    assert metrics['draft_configuration_weight_mass']==0
    assert metrics['draft_group_weight_mass']==3.5


def test_missing_owner_metadata_is_not_silently_per_event_mode_sampling():
    x=torch.zeros(1,3,2);weights=torch.ones(1,3)
    with pytest.raises(ValueError,match='packed on CPU'):
        draft_joint_sequence_loss(x,x,x,x,{'target_kind':torch.full((1,3),2)},weights,weights,1.)


def test_empty_zero_budget_batch_has_finite_zero_gradient_objective():
    x=torch.randn(1,2,2,requires_grad=True);weights=torch.zeros(1,2)
    batch=dict(target_kind=torch.full((1,2),3),draft_departure_group=torch.full((1,2),-1),
        draft_landing_group=torch.full((1,2),-1),draft_loss_group_count=0,
        draft_loss_group_rows=torch.empty(0,dtype=torch.long),
        draft_loss_group_positions=torch.empty(0,dtype=torch.long),draft_loss_group_side=torch.empty(0,dtype=torch.long))
    loss,_=draft_joint_sequence_loss(x,x,x,x,batch,weights,weights,1.)
    assert loss==0 and torch.isfinite(loss)


def test_committed_modes_use_actual_continuation_without_repaying_prior():
    batch=owners();batch['draft_loss_group_mode']=torch.tensor([1,0])
    time=torch.tensor([[[.01,8.],[.02,9.],[7.,.03]]],requires_grad=True)
    mark=torch.tensor([[[.01,6.],[5.,.01],[999.,999.]]],requires_grad=True)
    departure=torch.tensor([[[20.,-20.],[0.,0.],[0.,0.]]],requires_grad=True)
    landing=torch.tensor([[[0.,0.],[-20.,20.],[0.,0.]]],requires_grad=True)
    weight=torch.tensor([[1.,.5,2.]])
    loss,metrics=draft_joint_sequence_loss(time,mark,departure,landing,batch,weight,weight,3.5)
    expected=(time[0,0,1]+.5*time[0,1,1]+mark[0,0,1]+
              2*time[0,2,0]+.5*mark[0,1,0])/3.5
    torch.testing.assert_close(loss,expected)
    loss.backward()
    # Another mode may fit perfectly; the deployed continuation cannot use it.
    assert time.grad[0,0,0]==time.grad[0,1,0]==time.grad[0,2,1]==0
    assert mark.grad[0,0,0]==mark.grad[0,1,1]==0
    assert departure.grad.abs().sum()==landing.grad.abs().sum()==0
    assert metrics['draft_phrase_mode_kl']==0
    assert metrics['draft_phrase_mode_posterior_entropy']==0
    assert metrics['draft_committed_mode_groups']==2
    changed=time.detach().clone();changed[0,:2,0]=-100.;changed[0,2,1]=-100.
    other,_=draft_joint_sequence_loss(changed,mark.detach(),departure.detach(),landing.detach(),
        batch,weight,weight,3.5)
    torch.testing.assert_close(other,loss.detach(),rtol=0,atol=0)


def test_committed_and_authored_latent_phrases_keep_distinct_responsibilities():
    batch=owners();batch['draft_loss_group_mode']=torch.tensor([1,-1])
    time=torch.tensor([[[.1,2.],[.2,1.5],[2.,.3]]],requires_grad=True)
    mark=torch.tensor([[[.1,1.],[3.,.1],[0.,0.]]],requires_grad=True)
    departure=torch.tensor([[[2.,-1.],[7.,-3.],[-3.,7.]]],requires_grad=True)
    landing=torch.tensor([[[-5.,5.],[-2.,2.],[9.,-9.]]],requires_grad=True)
    weight=torch.tensor([[1.,.5,2.]])
    loss,metrics=draft_joint_sequence_loss(time,mark,departure,landing,batch,weight,weight,3.5)
    old=time[0,0,1]+.5*time[0,1,1]+mark[0,0,1]
    new=2*time[0,2]+.5*mark[0,1]
    expected=(old-2.5*torch.logsumexp(landing[0,1].log_softmax(-1)-new/2.5,0))/3.5
    torch.testing.assert_close(loss,expected)
    loss.backward()
    assert departure.grad.abs().sum()==0
    assert landing.grad[0,1].abs().sum()>0
    assert metrics['draft_committed_mode_groups']==metrics['draft_latent_mode_groups']==1


def test_invalid_committed_mode_is_rejected_before_loss_can_change_owner():
    batch=owners();batch['draft_loss_group_mode']=torch.tensor([2,-1])
    x=torch.zeros(1,3,2);weights=torch.ones(1,3)
    with pytest.raises(ValueError,match='Invalid committed'):
        draft_joint_sequence_loss(x,x,x,x,batch,weights,weights,1.)
