import pytest
import torch

from maichart.form.blueprint_batch import collate_blueprint
from maichart.form.rational_planner import (
    RationalPlanner, RationalPlannerConfig, rational_planner_loss,
)


def sample(frames, regions, chart_weight=1., excluded=False):
    return (
        dict(audio=torch.randn(1, frames, 120), form=torch.randn(1, regions, 11),
             condition=torch.randn(1, 16)),
        dict(load_curve_target=torch.rand(1, frames, 4),
             load_curve_mask=torch.ones(1, frames, 4, dtype=torch.bool),
             frame_weight=torch.zeros(1, frames) if excluded else torch.rand(1, frames)+.1),
        chart_weight,
    )


def test_batched_planner_matches_individual_losses_and_gradients():
    torch.manual_seed(42)
    planner = RationalPlanner(RationalPlannerConfig(
        descriptor_dim=8, width=16, form_heads=2, form_layers=1, dropout=0))
    planner.eval()
    rows = [sample(7, 2, 2.), sample(11, 4, .3), sample(5, 1, 1., excluded=True)]
    inputs, targets, weights, trainable = collate_blueprint(rows)
    assert trainable.tolist() == [True, True, False]
    result = planner(**inputs)
    batched = rational_planner_loss(result, targets, reduction='none')['loss']
    individual = torch.stack([
        rational_planner_loss(planner(**inp), target)['loss']
        for inp, target, _weight in rows])
    torch.testing.assert_close(batched, individual, atol=1e-7, rtol=1e-5)
    (batched*weights).sum().backward()
    batch_grads = {name: parameter.grad.clone() for name, parameter in planner.named_parameters()}
    planner.zero_grad()
    for inp, target, weight in rows:
        (rational_planner_loss(planner(**inp), target)['loss']*weight).backward()
    for name, parameter in planner.named_parameters():
        torch.testing.assert_close(parameter.grad, batch_grads[name], atol=1e-6, rtol=1e-4)
    assert torch.all(result['load_curve'][inputs['audio_padding_mask']] == 0)
    assert torch.all(targets['frame_weight'][inputs['audio_padding_mask']] == 0)
    assert not targets['load_curve_mask'][inputs['audio_padding_mask']].any()


def test_blueprint_batch_setting_validation():
    from maichart.form.train import parser, _validate
    args = parser().parse_args(['--manifest', 'unused', '--checkpoint-dir', 'unused',
                               '--blueprint-batch-size', '0'])
    with pytest.raises(ValueError, match='blueprint_batch_size'):
        _validate(args)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA smoke check requires a GPU')
def test_batched_planner_cuda_smoke():
    planner = RationalPlanner(RationalPlannerConfig(
        descriptor_dim=8, width=16, form_heads=2, form_layers=1, dropout=0)).cuda()
    inputs, targets, weights, trainable = collate_blueprint([sample(9, 3), sample(14, 5, .5)])
    inputs = {key: value.cuda() for key, value in inputs.items()}
    targets = {key: value.cuda() for key, value in targets.items()}
    prediction = planner(**inputs)
    loss = (rational_planner_loss(prediction, targets, reduction='none')['loss']
            * weights.cuda()).sum() / trainable.sum()
    loss.backward()
    assert torch.isfinite(loss)
    assert planner.audio_projection.weight.grad is not None
