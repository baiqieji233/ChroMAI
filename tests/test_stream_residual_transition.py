import pytest
import torch

from maichart.composer.stream_transition import residual_keep_logit,two_group_log_probs,sample_two_group


def test_zero_residual_restores_original_probabilities_and_base_gradients():
    base=torch.tensor([2.,-.5,1.,.7],requires_grad=True)
    keep=torch.tensor([True,False,True,False])
    correction=torch.tensor(0.,requires_grad=True)
    gate=residual_keep_logit(base,keep,correction)
    result=two_group_log_probs(base,keep,gate)
    ordinary=base.log_softmax(-1)
    torch.testing.assert_close(result,ordinary,rtol=1e-6,atol=1e-7)
    actual_grad=torch.autograd.grad(-result[1],base,retain_graph=True)[0]
    expected_grad=torch.autograd.grad(-ordinary[1],base,retain_graph=True)[0]
    torch.testing.assert_close(actual_grad,expected_grad,rtol=1e-6,atol=1e-7)
    residual_grad=torch.autograd.grad(-result[1],correction)[0]
    assert residual_grad>0  # An exit target must reduce the keep correction.


@pytest.mark.parametrize('correction',[-.2,.2])
def test_residual_changes_odds_in_the_expected_direction_with_gain_eight(correction):
    base=torch.tensor([.7,.1,.2]).log();keep=torch.tensor([True,False,False])
    baseline=base[keep].logsumexp(-1)-base[~keep].logsumexp(-1)
    gate=residual_keep_logit(base,keep,correction)
    torch.testing.assert_close(gate,baseline+8*correction)
    result=two_group_log_probs(base,keep,gate)
    mass=result[keep].exp().sum()
    torch.testing.assert_close(mass,(baseline+8*correction).sigmoid())
    assert bool(mass>.7)==(correction>0)
    torch.testing.assert_close(residual_keep_logit(base,keep,correction,gain=1.),baseline+correction)


def test_batched_zero_correction_broadcasts_without_changing_conditional_probabilities():
    base=torch.tensor([[[1.,2.,3.],[.5,-.2,.1]],[[2.,.2,1.],[-1.,0.,.5]]],requires_grad=True)
    keep=torch.tensor([True,False,True])
    correction=torch.zeros(2,2,1,requires_grad=True)
    gate=residual_keep_logit(base,keep,correction)
    result=two_group_log_probs(base,keep,gate)
    assert gate.shape==(2,2)
    torch.testing.assert_close(result,base.log_softmax(-1),rtol=1e-6,atol=1e-7)
    result[...,1].sum().backward()
    assert torch.isfinite(base.grad).all() and torch.isfinite(correction.grad).all()


def test_empty_and_single_group_bfloat_rows_remain_finite_in_backward():
    base=torch.tensor([[0.,-1e4,-1e4],[-1e4,0.,1.],
                       [-1e4,-1e4,-1e4],[-torch.inf,-torch.inf,-torch.inf]],
        dtype=torch.bfloat16,requires_grad=True)
    correction=torch.tensor([-1000.,1000.,2.,3.],dtype=torch.bfloat16,requires_grad=True)
    keep=torch.tensor([True,False,False])
    gate=residual_keep_logit(base,keep,correction)
    assert gate.dtype==torch.float32 and gate.eq(0).all()
    result=two_group_log_probs(base,keep,gate)
    assert result[0,0]==0 and result[0,1:].eq(-1e4).all()
    torch.testing.assert_close(result[1,1:].exp().sum(),torch.tensor(1.))
    assert result[2].eq(-1e4).all() and torch.isneginf(result[3]).all()
    (result[0,0]+result[1,1:].sum()).backward()
    assert torch.isfinite(base.grad).all() and torch.isfinite(correction.grad).all()
    assert correction.grad.eq(0).all()


def test_explicit_support_uses_only_legal_base_mass_and_preserves_mask_style():
    base=torch.tensor([3.,2.,1.,-1e4])
    keep=torch.tensor([True,True,False,False]);allowed=torch.tensor([False,True,True,False])
    gate=residual_keep_logit(base,keep,0.,allowed=allowed)
    torch.testing.assert_close(gate,torch.tensor(1.))
    result=two_group_log_probs(base,keep,gate,allowed=allowed)
    assert result[0]==-1e4 and result[3]==-1e4
    torch.testing.assert_close(result[1:3],base[1:3].log_softmax(-1))


def test_bfloat_mask_is_canonicalized_in_all_three_public_paths():
    base=torch.tensor([2.,-1e4,1.],dtype=torch.bfloat16,requires_grad=True)
    keep=torch.tensor([False,True,True])
    assert float(base[1])==-9984.  # The real source of the support mismatch.
    gate=residual_keep_logit(base,keep,0.)
    torch.testing.assert_close(gate,torch.tensor(-1.))
    result=two_group_log_probs(base,keep,gate)
    assert result.dtype==torch.float32 and result[1]==-1e4
    torch.testing.assert_close(result[[0,2]],base.float()[[0,2]].log_softmax(-1))
    (-result[0]).backward()
    assert base.grad[1]==0 and torch.isfinite(base.grad).all()
    empty=torch.full((3,),-1e4,dtype=torch.bfloat16)
    generator=torch.Generator().manual_seed(19);before=generator.get_state()
    assert sample_two_group(empty,keep,1.,generator) is None
    assert torch.equal(before,generator.get_state())
    assert sample_two_group(base.detach(),keep,0.)==0


def test_finite_bfloat_mask_does_not_create_false_keep_support():
    base=torch.tensor([-1e4,0.,1.],dtype=torch.bfloat16)
    keep=torch.tensor([True,False,False])
    result=two_group_log_probs(base,keep,1000.)
    assert result[0]==-1e4
    torch.testing.assert_close(result[1:],torch.tensor([0.,1.]).log_softmax(-1))
    assert sample_two_group(result,keep,0.)==2
