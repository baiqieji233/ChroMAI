"""CPU checks for the v6 baseline; run with the final coordinated suite."""
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import torch

from maichart.composer.stream_data import NOTE
from maichart.composer.stream_timing import AbsoluteTimingHead, initialize_timing_prior


def test_cell_likelihood_prior_uses_seconds_and_inverse_softplus():
    head=AbsoluteTimingHead(4,(0,1,2),group_policy='phase_only_v1')
    step=SimpleNamespace(kind=NOTE,start=Fraction(0),end=Fraction(1),start_second=0.,weight=1.)
    ds=SimpleNamespace(streams=[[step]],target_window_weights=[[1.]],chart_weights=[1.],unknown_steps=[[False]],
        _end_beats=[4.],clocks=[SimpleNamespace(at_floats=lambda beats:np.asarray(beats)*.5)])
    result=initialize_timing_prior(head,ds,batch_size=1)
    rate=result['timing_prior_rates'][0]
    # BOS owns [0,.25], the observed beat-one cell owns [.25,.75].
    assert abs(result['timing_prior_survival_seconds'][0]-.25)<1e-12
    assert abs(.5/np.expm1(.5*rate)+.5/rate-.75)<1e-10
    torch.testing.assert_close(torch.nn.functional.softplus(head.grid_head.bias.double()),
        torch.tensor(result['timing_prior_rates'],dtype=torch.float64),rtol=1e-6,atol=1e-7)


def test_empty_training_prior_is_finite():
    head=AbsoluteTimingHead(4,(0,1,2),group_policy='phase_gap_v2')
    ds=SimpleNamespace(streams=[],target_window_weights=[],chart_weights=[],unknown_steps=[])
    result=initialize_timing_prior(head,ds)
    np.testing.assert_allclose(result['timing_prior_rates'],np.ones(22),rtol=1e-12)
    combined=head.grid_head.bias.repeat(2)+head.bucket_head.bias
    torch.testing.assert_close(torch.nn.functional.softplus(combined),torch.ones(22))
