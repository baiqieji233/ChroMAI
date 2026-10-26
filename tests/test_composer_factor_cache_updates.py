import pytest
import torch

from maichart.composer.action_factors import CompositionalActions


def fresh_weight(model):
    return sum(factor(getattr(model,'atom_'+name)) for name,factor in model.factors.items())


@pytest.mark.parametrize('switch_to_train',[False,True])
def test_evaluation_cache_is_invalidated_before_unversioned_optimizer_updates(switch_to_train):
    model=CompositionalActions(['1','2h[4:1]','A1'],4).eval()
    with torch.no_grad():
        before=model.composed_weight()
        assert model.composed_weight() is before
    if switch_to_train:model.train()
    model(torch.ones(2,4)).sum().backward()
    # Fused AdamW on the installed CUDA build mutates without bumping _version.
    # .data emulates that property on CPU as well, without a CUDA requirement.
    versions=[factor.weight._version for factor in model.factors.values()]
    with torch.no_grad():
        for factor in model.factors.values():factor.weight.data.add_(.25)
    assert versions==[factor.weight._version for factor in model.factors.values()]
    if switch_to_train:model.eval()
    with torch.no_grad():
        actual=model.composed_weight()
        torch.testing.assert_close(actual,fresh_weight(model))
        assert not torch.equal(actual,before)
        assert model.composed_weight() is actual


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_real_fused_adamw_evaluate_train_evaluate_uses_updated_factors():
    model=CompositionalActions(['1','2h[4:1]','A1'],4).cuda().eval()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.01,fused=True)
    with torch.no_grad():before=model.composed_weight().clone()
    model.train()
    model(torch.ones(2,4,device='cuda')).sum().backward()
    optimizer.step();model.eval()
    with torch.no_grad():
        actual=model.composed_weight()
        torch.testing.assert_close(actual,fresh_weight(model))
        assert not torch.equal(actual,before)
