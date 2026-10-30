import math

import numpy as np
import pytest
import torch

from maichart.composer.data import event_rate_targets
from maichart.composer.event_views import EventViews
from maichart.composer.train import _loss


def _loss_inputs():
    output = {'candidate_logits': torch.log(torch.tensor([[.1, .2]])),
              'action_logits': torch.zeros(1, 1, 2),
              'future': torch.zeros(1, 8), 'eos_logits': torch.zeros(1)}
    batch = {'candidate_target': torch.tensor([1]),
             'action_target': torch.zeros(1, 1, dtype=torch.long),
             'future_target': torch.zeros(1, 8), 'eos_target': torch.zeros(1)}
    return output, batch


def test_survival_pointer_retains_probability_of_no_event():
    output, batch = _loss_inputs()
    _, legacy = _loss(output, batch, 2)
    joint = output['candidate_logits'].clone().requires_grad_()
    output['next_event_log_probs'] = joint
    output['audio_next_event_log_probs'] = joint
    _, survival = _loss(output, batch, 2)
    assert legacy['pointer_loss'].item() == pytest.approx(-math.log(2 / 3))
    assert survival['pointer_loss'].item() == pytest.approx(-math.log(.2))
    assert survival['audio_pointer_loss'].item() == pytest.approx(-math.log(.2))


@pytest.mark.parametrize('supervised', [True, False])
def test_empty_time_probes_penalize_rate_only_with_known_supervision(supervised):
    output, batch = _loss_inputs()
    output['next_event_log_probs'] = output['candidate_logits']
    lograte = torch.full((1, 2), math.log1p(4.), requires_grad=True)
    output['workload'] = torch.stack([lograte] + [torch.zeros_like(lograte)] * 4, -1)
    batch.update(event_count_target=torch.zeros(1, 2), event_exposure=torch.full((1, 2), .25),
                 event_rate_weight=torch.full((1, 2), float(supervised)))
    loss, metrics = _loss(output, batch, 2)
    assert metrics['rawrate_loss'].item() == pytest.approx(float(supervised))
    assert metrics['rate_loss_contribution'].item() == pytest.approx(.2 * float(supervised))
    loss.backward()
    if supervised:
        assert torch.all(lograte.grad > 0)
    else:
        assert torch.all(lograte.grad == 0)


def test_rate_labels_count_onsets_and_mask_unknown_spans_and_candidate_rows():
    # Use the actual confidence implementation without loading audio or a chart.
    class Views:
        window_weight = EventViews.window_weight
        end_sec = 1.
        unknown_intervals = ((.55, .65),)
        regions = ()

    counts, exposure, weight = event_rate_targets(
        np.array([0., .1, .5, .9]), np.array([.1, .1, .6, 1.]),
        1., Views(), np.ones(4, np.float32), 1)
    np.testing.assert_array_equal(counts, [2., 2., 1., 1.])
    np.testing.assert_allclose(exposure, [.225, .225, .25, .125])
    np.testing.assert_array_equal(weight, [0., 1., 0., 1.])


def test_rate_labels_exclude_right_boundary_and_zero_exposure():
    class Views:
        window_weight = EventViews.window_weight
        end_sec = 1.
        unknown_intervals = ()
        regions = ()

    counts, exposure, weight = event_rate_targets(
        [.125, .375], [.25, 2.], 1., Views(), np.ones(2, np.float32), 0)
    np.testing.assert_array_equal(counts, [1., 0.])
    np.testing.assert_array_equal(exposure, [.25, 0.])
    np.testing.assert_array_equal(weight, [1., 0.])


def test_survival_rate_does_not_also_fit_the_old_smoothed_rate_target():
    output,batch=_loss_inputs()
    output['next_event_log_probs']=output['candidate_logits']
    output['workload']=torch.zeros(1,2,5)
    batch.update(workload_target=torch.zeros(1,2,5),workload_weight=torch.ones(1,2),
        workload_seconds=torch.tensor([[0.,1.]]))
    _,before=_loss(output,batch,2)
    batch['workload_target'][...,0]=100.
    _,after=_loss(output,batch,2)
    assert torch.equal(before['workload_loss'],after['workload_loss'])
    assert torch.equal(before['curve_slope_loss'],after['curve_slope_loss'])
