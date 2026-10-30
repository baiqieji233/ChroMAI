import torch
from maichart.composer.event_survival import causal_count_feedback,integrated_survival


def test_actual_overshoot_only_reduces_rate_and_never_fills_under_budget():
    rates=torch.ones(1,100,requires_grad=True)
    edges=torch.arange(101,dtype=torch.float64)[None]*.05
    empty=torch.zeros(1,100,2)
    same,penalty=causal_count_feedback(rates,edges,empty)
    torch.testing.assert_close(same,rates,atol=0,rtol=0)
    assert not penalty.any()
    busy=torch.full_like(empty,20.)
    reduced,cost=causal_count_feedback(rates,edges,busy)
    assert torch.all(reduced<rates) and torch.all(reduced>0) and cost.max()<=4.
    restored,_=causal_count_feedback(rates,edges,busy,torch.zeros(1,1))
    torch.testing.assert_close(restored,rates,atol=0,rtol=0)
    seconds=torch.tensor([[4.2,4.5,5.]])
    valid=torch.ones_like(seconds,dtype=torch.bool)
    _,raw_tail,_=integrated_survival(rates,edges,seconds,valid,torch.tensor([4.]))
    _,soft_tail,_=integrated_survival(reduced,edges,seconds,valid,torch.tensor([4.]))
    assert torch.all(soft_tail>=raw_tail)
    reduced.sum().backward()
    assert torch.isfinite(rates.grad).all() and torch.all(rates.grad>0)
