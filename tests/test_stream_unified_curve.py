"""Explicit curve control transforms occurrence and timing as one law."""
from types import SimpleNamespace
import math

import numpy as np
import torch

from maichart.composer.stream_data import NOTE,WAIT
from maichart.composer.stream_generate import _Curve


def fixture(ratio=2.,rests=()):
    curve=object.__new__(_Curve)
    curve.active=True;curve.curve_edges=np.array([0.,1.])
    curve.event_bin_ratios=np.array([ratio]);curve._arrival_curve_tensors={}
    curve.controller=SimpleNamespace(rest_regions=list(rests))
    rate=torch.tensor(math.log(math.expm1(1.)))
    result=dict(arrival_log_rates=rate.expand(1,1,1,2),
        arrival_seconds=torch.tensor([[[.25,.75]]],dtype=torch.float64),
        arrival_allowed=torch.ones(1,1,2,dtype=torch.bool),
        arrival_start=torch.tensor([[0.]],dtype=torch.float64),
        arrival_stop=torch.tensor([[1.]],dtype=torch.float64),
        kind_logits=torch.zeros(1,1,5),delta_logits=torch.zeros(1,1,2),
        execution_kind_logits=torch.zeros(1,1,1,5),execution_delta_logits=torch.zeros(1,1,1,2))
    return curve,result


def test_curve_changes_tail_and_conditional_from_same_final_rate():
    curve,result=fixture()
    kinds,timing=curve.first_arrival(result,0)
    note=kinds[NOTE].exp();tail=kinds[WAIT].exp()
    assert torch.allclose(tail,torch.tensor(math.exp(-2.)),atol=1e-6)
    assert torch.allclose(note+tail,torch.tensor(1.),atol=1e-6)
    joint=note*timing.exp()
    expected=torch.tensor([1-math.exp(-1.),math.exp(-1.)*(1-math.exp(-1.))])
    assert torch.allclose(joint,expected,atol=1e-6)


def test_explicit_rest_moves_all_probability_to_tail_without_forcing_notes():
    curve,result=fixture(rests=[(0.,1.)])
    kinds,timing=curve.first_arrival(result,0)
    assert torch.isneginf(kinds[NOTE])
    assert kinds[WAIT]==0
    assert torch.isneginf(timing).all()


def test_disabled_curve_returns_model_distribution_without_reintegrating():
    curve,result=fixture();curve.active=False
    kinds,timing=curve.first_arrival(result,0)
    assert kinds.data_ptr()==result['execution_kind_logits'].data_ptr()
    assert timing.data_ptr()==result['execution_delta_logits'].data_ptr()
