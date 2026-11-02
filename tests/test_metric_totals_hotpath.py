"""CPU-only checks for packed optional metric updates."""
import torch
from maichart.composer.metric_totals import MetricTotals
from maichart.composer.stream_train import _record_recovery_metrics


def test_optional_subsets_reuse_mapping_after_schema_growth():
    total=MetricTotals();reference={}
    updates=[{'loss':1.,'count':2.},{'loss':-3.},{'recovery_loss':4.,'count':-1.},
             {'loss':7.},{'count':3.,'extra':-5.},{'loss':-2.}]
    cached=None
    for number,update in enumerate(updates):
        values={name:torch.tensor(value) for name,value in update.items()}
        total.add(values)
        for name,value in values.items():
            reference[name]=reference[name]+value if name in reference else value.clone()
        assert total.names==tuple(reference)
        torch.testing.assert_close(total.values,torch.stack(list(reference.values())),rtol=0,atol=0)
        if number==1:cached=total._indices[('loss',)]
    assert total._indices[('loss',)] is cached


def test_recovery_window_commit_and_discard_remain_isolated():
    window=MetricTotals();window.add({'loss':torch.tensor(3.),'base_loss':torch.tensor(3.)})
    window.add({'loss':torch.tensor(5.),'base_loss':torch.tensor(5.)})
    _record_recovery_metrics(window,torch.tensor(.25,requires_grad=True),2)
    assert window.as_floats()=={'loss':8.5,'base_loss':8.,'recovery_loss':.5}
    committed=MetricTotals();committed.add(window)
    saved=committed.as_floats()
    window.add({'loss':torch.tensor(float('nan'))})
    assert committed.as_floats()==saved
    assert committed.values.grad_fn is None
    next_window=MetricTotals();next_window.add({'base_loss':torch.tensor(-1.),'loss':torch.tensor(2.)})
    committed.add(next_window)
    assert committed.as_floats()=={'loss':10.5,'base_loss':7.,'recovery_loss':.5}
