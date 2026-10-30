import torch

from maichart.composer.button_load import button_load_features, PerButtonLoad


def _timing(pads):
    value = torch.zeros(1, 2, 3*pads)
    value[..., :8] = 1.
    value[..., pads:pads+8] = torch.log1p(torch.tensor(.079))
    value[..., 2*pads:2*pads+8] = torch.log1p(torch.tensor(.25))
    return value


def test_changing_chord_partner_keeps_full_shared_button_load():
    heads = torch.zeros(1, 4, 8)
    heads[0, :, 4] = 1.
    heads[0, :2, 5] = 1.
    heads[0, 2:, 6] = 1.
    ages = torch.tensor([[.75, .5, .25, 0.]])
    features = button_load_features(heads, ages, torch.ones(1, 4, dtype=torch.bool),
                                    _timing(41), torch.tensor([[.25, .5]]))
    assert torch.all(features[..., 4, 7:] > features[..., 5, 7:])
    assert torch.all(features[..., 4, 7:] > features[..., 6, 7:])
    # A chord must not halve the contribution of its shared button.
    single = heads.clone()
    single[..., 5:] = 0
    other = button_load_features(single, ages, torch.ones(1, 4, dtype=torch.bool),
                                _timing(41), torch.tensor([[.25, .5]]))
    torch.testing.assert_close(features[..., 4, :], other[..., 4, :])


def test_layouts_agree_for_matching_button_ages_and_masked_history():
    heads = torch.ones(1, 3, 8)
    ages = torch.tensor([[1., .5, 0.]])
    valid = torch.tensor([[True, False, True]])
    gap = torch.tensor([[.25, .5]])
    old = button_load_features(heads, ages, valid, _timing(33), gap)
    new = button_load_features(heads, ages, valid, _timing(41), gap)
    torch.testing.assert_close(old, new)


def test_zero_initialization_all_slots_gradient_and_context_interaction():
    torch.manual_seed(23)
    model = PerButtonLoad(16)
    features = torch.rand(2, 8, 10)
    context = torch.randn(2, 4, 16)
    scores = model(features, context)
    assert scores.shape == (2, 4, 8)
    assert not scores.count_nonzero()
    scores.sum().backward()
    assert model.output.weight.grad.abs().sum() > 0
    with torch.no_grad():
        model.output.weight.normal_(0., .1)
    context.requires_grad_()
    scores = model(features, context)
    scores.square().sum().backward()
    assert torch.all(context.grad.abs().sum(-1) > 0)
    assert not torch.allclose(scores[:, 0], scores[:, 1])
