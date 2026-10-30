"""Rest supervision stays stable without changing planner probabilities or state."""
import copy

import pytest
import torch
from torch.nn import functional as F

from maichart.composer.planning import AudioPlanner
from maichart.composer.song_context import SongContext
from maichart.composer.train import _loss
from test_composer_survival_training import _loss_inputs


def rest_loss(logits, target, *, expose_logits=True, supervised=True):
    output, batch = _loss_inputs()
    output['plan'] = torch.cat((logits.new_zeros((*logits.shape, 5)), logits.sigmoid()[..., None]), -1)
    output['survival_grid_rates'] = logits.new_ones(logits.shape)
    if expose_logits:
        output['plan_rest_logits'] = logits
    batch['planner_target'] = torch.cat((logits.new_zeros((*logits.shape, 5)), target[..., None]), -1)
    batch['planner_weight'] = torch.full_like(logits, float(supervised))
    return _loss(output, batch, 2)


def test_normal_rest_loss_and_gradient_match_probability_bce():
    target = torch.tensor([[0., 1., 0., 1., .3]])
    results = []
    for expose in (False, True):
        logits = torch.tensor([[-4., -1., 0., 2., 4.]], requires_grad=True)
        loss, metrics = rest_loss(logits, target, expose_logits=expose)
        loss.backward()
        results.append((metrics['planner_loss'], logits.grad))
    for actual, expected in zip(results[1], results[0]):
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize('value,target', [(17., 0.), (25., 0.), (-100., 1.)])
def test_confidently_wrong_rest_keeps_finite_corrective_gradient(value, target):
    logits = torch.tensor([[value]], requires_grad=True)
    loss, metrics = rest_loss(logits, torch.full_like(logits, target))
    loss.backward()
    # Keep the existing planner weight .2 and mean over six channels.
    expected = .2 / 6. * (logits.detach().sigmoid() - target)
    torch.testing.assert_close(logits.grad, expected)
    assert logits.grad.abs().item() > .03
    assert torch.isfinite(loss) and torch.isfinite(metrics['planner_loss'])


def test_masked_recovery_rest_labels_do_not_create_gradient():
    logits = torch.tensor([[25.]], requires_grad=True)
    loss, metrics = rest_loss(logits, torch.zeros_like(logits), supervised=False)
    loss.backward()
    assert metrics['planner_loss'] == 0
    assert logits.grad.item() == 0


@pytest.mark.parametrize('song_paired', [False, True])
def test_planner_optional_logits_preserve_api_predictions_and_state(song_paired):
    torch.manual_seed(17)
    if song_paired:
        model = SongContext(3, width=16).eval()
        invoke = lambda enabled: model.heads(torch.zeros(2, 4, 16), torch.zeros(2, 3), return_rest_logits=enabled)
        head = model.plan
    else:
        model = AudioPlanner(16, 3).eval()
        invoke = lambda enabled: model(torch.zeros(2, 4, 104), torch.zeros(2, 3), torch.zeros(2, 4, dtype=torch.bool), return_rest_logits=enabled)
        head = model.output
    with torch.no_grad():
        head.weight[5].zero_()
        head.bias[5] = 17.
    saved = copy.deepcopy(model.state_dict())
    old = invoke(False)
    new = invoke(True)
    torch.testing.assert_close(new[0], old[0] if song_paired else old, atol=0, rtol=0)
    if song_paired:
        torch.testing.assert_close(new[1], old[1], atol=0, rtol=0)
    assert new[0][..., 5].eq(1).all()
    F.binary_cross_entropy_with_logits(new[-1], torch.zeros_like(new[-1])).backward()
    assert head.bias.grad[5] == 1.
    assert model.state_dict().keys() == saved.keys()
    model.load_state_dict(saved, strict=True)


def test_grounded_song_planner_emits_raw_logits_and_respects_override():
    from test_composer_compact_integration import fixture
    model, data = fixture()
    with torch.no_grad():
        model.planner.plan.weight[5].zero_()
        model.planner.plan.bias[5] = 17.
    output = model(**data)
    assert output['plan_rest_logits'].eq(17).all()
    assert output['plan'][..., 5].eq(1).all()
    override = output['plan'].detach().clone()
    override[..., 5] = .25
    changed = model(**data, planner_override=override)
    assert 'plan_rest_logits' not in changed
    torch.testing.assert_close(changed['plan'], override, atol=0, rtol=0)
