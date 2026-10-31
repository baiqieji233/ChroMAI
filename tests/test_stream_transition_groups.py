import math

import pytest
import torch

from maichart.composer.stream_transition import two_group_log_probs,sample_two_group


def test_explicit_mass_and_conditional_ratios_do_not_depend_on_group_size():
    scores=torch.tensor([.12,.18,.10,.20,.40]).log()
    keep=torch.tensor([True,True,False,False,False])
    result=two_group_log_probs(scores,keep,math.log(.7/.3)).exp()
    torch.testing.assert_close(result[keep].sum(),torch.tensor(.7))
    torch.testing.assert_close(result[~keep].sum(),torch.tensor(.3))
    torch.testing.assert_close(result[0]/result[1],torch.tensor(2/3))
    torch.testing.assert_close(result[2:]/result[2:].sum(),torch.tensor([1/7,2/7,4/7]))


def test_base_group_log_odds_reconstruct_original_distribution_without_a_residual():
    scores=torch.tensor([1.,-.5,2.,.7])
    keep=torch.tensor([True,False,True,False])
    gate=scores[keep].logsumexp(-1)-scores[~keep].logsumexp(-1)
    result=two_group_log_probs(scores,keep,gate)
    torch.testing.assert_close(result,scores.log_softmax(-1))


def test_batched_boolean_support_broadcasts_and_empty_groups_fall_back():
    scores=torch.tensor([[[0.,1.,-1e4],[1.,2.,-1e4]],
                         [[-1e4,-1e4,-1e4],[-torch.inf,-torch.inf,-torch.inf]]],requires_grad=True)
    keep=torch.tensor([True,True,False])
    gate=torch.tensor([[100.,-100.],[1.,2.]],requires_grad=True)
    result=two_group_log_probs(scores,keep,gate[...,None])
    torch.testing.assert_close(result[0,:,:2].exp().sum(-1),torch.ones(2))
    assert result[0,:,2].eq(-1e4).all()
    assert result[1,0].eq(-1e4).all()
    assert torch.isneginf(result[1,1]).all()
    result[0,:,:2].sum().backward()
    assert torch.isfinite(scores.grad).all() and torch.isfinite(gate.grad).all()
    assert gate.grad.count_nonzero()==0  # The exit group has no support.


def test_empty_keep_group_and_explicit_legal_mask_keep_original_distribution():
    scores=torch.tensor([1.,2.,3.])
    result=two_group_log_probs(scores,torch.tensor([True,False,False]),1000.,allowed=torch.tensor([False,True,True]))
    assert result[0]==-1e4
    torch.testing.assert_close(result[1:],scores[1:].log_softmax(-1))
    assert sample_two_group(result,torch.tensor([True,False,False]),temperature=0.)==2


def test_group_gate_and_within_choice_have_separate_gradients():
    scores=torch.tensor([1.,2.,3.,4.],requires_grad=True)
    gate=torch.tensor(-.5,requires_grad=True)
    keep=torch.tensor([True,True,False,False])
    loss=-two_group_log_probs(scores,keep,gate)[2]
    loss.backward()
    assert gate.grad>0  # Leaving should decrease retention.
    torch.testing.assert_close(scores.grad[:2],torch.zeros(2),atol=1e-7,rtol=0)
    assert scores.grad[2]<0 and scores.grad[3]>0


@pytest.mark.parametrize('value',[-1000.,1000.])
def test_extreme_bfloat_gate_is_stable_in_fp32(value):
    gate=torch.tensor(value,dtype=torch.bfloat16,requires_grad=True)
    result=two_group_log_probs(torch.tensor([1.,2.,3.],dtype=torch.bfloat16),torch.tensor([True,False,False]),gate)
    assert result.dtype==torch.float32 and torch.isfinite(result).all()
    torch.testing.assert_close(result.exp().sum(),torch.tensor(1.))
    (-result[1]).backward()
    assert torch.isfinite(gate.grad)


def test_greedy_selects_exit_group_even_when_one_keep_atom_is_the_flat_mode():
    probabilities=torch.tensor([.4,.3,.3])
    keep=torch.tensor([True,False,False])
    assert probabilities.argmax()==0
    assert sample_two_group(probabilities.log(),keep,temperature=0.)==1


def test_greedy_selects_retention_mass_even_when_it_is_split_over_many_atoms():
    probabilities=torch.tensor([.13,.13,.13,.13,.13,.35])
    keep=torch.tensor([True,True,True,True,True,False])
    assert probabilities.argmax()==5
    assert sample_two_group(probabilities.log(),keep,temperature=0.)==0


def test_positive_temperature_preserves_group_sampling_and_empty_rows_return_none():
    scores=torch.tensor([.4,.3,.3]).log();mask=torch.tensor([True,False,False])
    generator=torch.Generator().manual_seed(42)
    samples=[sample_two_group(scores,mask,temperature=.1,generator=generator) for _ in range(600)]
    assert .3<samples.count(0)/len(samples)<.5
    assert sample_two_group(torch.full((3,),-1e4),mask,0.) is None
    assert sample_two_group(torch.full((3,),-torch.inf),mask,1.) is None
    with pytest.raises(ValueError,match='one transition row'):
        sample_two_group(torch.zeros(2,3),mask)


def test_nested_greedy_chooses_binary_then_grid_mass_before_its_candidates():
    probabilities=torch.tensor([.4,.16,.16,.28])
    keep=torch.tensor([True,False,False,False])
    ids=torch.tensor([[[0,0,0,0],[0,1,1,2]]])
    # Binary exit wins .6 over .4. Inside exit, grid 1 wins .32 over .28,
    # although each of its two candidates is smaller than the .28 singleton.
    assert sample_two_group(probabilities.log(),keep,0.)==3
    assert sample_two_group(probabilities.log(),keep,0.,within_group_ids=ids,within_group_count=3)==1


def test_nested_grid_temperature_does_not_change_binary_mass_or_reward_singletons():
    scores=torch.tensor([.4,.16,.16,.28]).log()
    keep=torch.tensor([True,False,False,False]);ids=torch.tensor([0,1,1,2])
    generator=torch.Generator().manual_seed(42)
    samples=[sample_two_group(scores,keep,.1,generator,within_group_ids=ids,within_group_count=3)
             for _ in range(600)]
    retention=samples.count(0)
    assert .3<retention/len(samples)<.5  # Binary mass stays .4 at any positive T.
    grid_one=sum(sample in (1,2) for sample in samples)
    assert .65<grid_one/(len(samples)-retention)<.95


def test_nested_grid_support_excludes_masked_candidates_and_requires_static_count():
    scores=torch.tensor([.4,.6,0.]).log()
    keep=torch.tensor([True,False,False]);ids=torch.tensor([0,1,2])
    assert sample_two_group(scores,keep,0.,within_group_ids=ids,within_group_count=3)==1
    with pytest.raises(ValueError,match='static group count'):
        sample_two_group(scores,keep,0.,within_group_ids=ids)
