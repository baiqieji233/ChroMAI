import copy

import pytest
import torch
from torch.nn import functional as F

from maichart.composer.action_factors import CompositionalActions


@pytest.mark.parametrize('policy',['marginal_v1','branch_joint_v2'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_shared_table_matches_sparse_embeddings_and_independent_heads(policy,dtype):
    torch.manual_seed(831)
    atoms=['1','2h[4:1]','A1','Ch[4:1]','1-4[8:1]*-6[4:1]']
    reference=CompositionalActions(atoms,8,slide_factor_policy=policy).to(dtype)
    with torch.no_grad():
        for factor in reference.factors.values():factor.weight.normal_(0,.1)
    shared=copy.deepcopy(reference)
    history=torch.tensor([[[1,2,0],[1,5,0]],[[3,4,0],[3,0,0]]])
    targets=torch.tensor([[5,1,0],[3,4,0]])
    slot_context=torch.randn(2,3,8,dtype=dtype)
    outputs=[]
    for model,reuse in ((reference,False),(shared,True)):
        weight=model.composed_weight() if reuse else None
        hist=model.embed(history,composed_weight=weight)*history.ne(0)[...,None]
        prefix=model.embed(targets,composed_weight=weight)*targets.ne(0)[...,None]
        prefix=prefix.cumsum(1)-prefix
        full=slot_context+hist.sum(2).mean(1)[:,None]+prefix
        audio=slot_context+prefix
        main_logits=F.linear(full,weight) if reuse else model(full)
        audio_logits=F.linear(audio,weight) if reuse else model(audio)
        loss=F.cross_entropy(main_logits.flatten(0,1),targets.flatten())
        loss=loss+.15*F.cross_entropy(audio_logits.flatten(0,1),targets.flatten())
        loss.backward();outputs.append((main_logits.detach(),audio_logits.detach()))
    tolerance=1e-6 if dtype==torch.float32 else 1e-12
    for before,after in zip(*outputs):torch.testing.assert_close(before,after,atol=tolerance,rtol=tolerance)
    for (name,before),(_,after) in zip(reference.named_parameters(),shared.named_parameters()):
        assert (before.grad is None)==(after.grad is None),name
        if before.grad is not None:
            torch.testing.assert_close(before.grad,after.grad,atol=tolerance,rtol=tolerance,msg=name)
    assert any(factor.weight.grad.abs().sum()>0 for factor in shared.factors.values())


def test_invalid_composed_table_shape_fails_before_embedding():
    model=CompositionalActions(['1'],4)
    with pytest.raises(ValueError,match='catalog and width'):
        model.embed(torch.tensor([0,1]),composed_weight=torch.zeros(3,4))
