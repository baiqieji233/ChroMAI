import torch

from maichart.composer.button_load import button_load_features, PerButtonLoad, PerButtonLoadV2
from maichart.composer.data import ComposerDataset


def test_target_augmentation_is_private_reproducible_and_preserves_stop():
    dataset = object.__new__(ComposerDataset)
    dataset.seed, dataset.epoch, dataset.action_order_policy = 17, 2, 'symmetric_v1'
    canonical = torch.tensor([3, 5, 9, 0, 0])
    orders = [dataset.ordered_action_target(canonical, i) for i in range(64)]
    assert any(torch.equal(row[:3], canonical[:3].flip(0)) for row in orders)
    assert any(torch.equal(row, canonical) for row in orders)
    for index, row in enumerate(orders):
        assert torch.equal(row, dataset.ordered_action_target(canonical, index))
        assert torch.equal(row[3:], canonical[3:])
    assert canonical.tolist() == [3, 5, 9, 0, 0]
    dataset.action_order_policy = 'off'
    assert all(torch.equal(dataset.ordered_action_target(canonical, i), canonical) for i in range(64))


def test_second_load_uses_real_seconds_and_chord_members():
    heads = torch.zeros(1, 2, 8)
    heads[0, :, 0] = 1
    heads[0, 0, 1] = 1
    timing = torch.zeros(1, 2, 123)
    timing[..., :8] = 1
    args = (heads, torch.tensor([[1., 0.]]), torch.ones(1, 2, dtype=torch.bool),
            timing, torch.tensor([[.25, .5]]))
    fast = button_load_features(*args, history_second_ages=torch.tensor([[.08, 0.]]),
                               candidate_gap_seconds=torch.tensor([[.08, .18]]), extended=True)
    slow = button_load_features(*args, history_second_ages=torch.tensor([[.8, 0.]]),
                               candidate_gap_seconds=torch.tensor([[.08, .18]]), extended=True)
    assert fast.shape == (1, 2, 8, 23)
    assert fast[0, 0, 0, 10] > slow[0, 0, 0, 10]
    assert fast[0, 0, 0, 14] > 0  # chord member burden
    assert fast[0, 0, 0, 18] > fast[0, 0, 1, 18]  # shared-key run
    assert button_load_features(*args).shape[-1] == 10


def test_load_heads_preserve_legacy_shapes_and_center_each_kind():
    context = torch.randn(2, 4, 16)
    assert PerButtonLoad(16)(torch.rand(2, 8, 10), context).shape == (2, 4, 8)
    model = PerButtonLoadV2(16)
    features = torch.rand(2, 8, 23)
    features[..., 0] = 1
    assert model(features, context).shape == (2, 4, 3, 8)
    assert not model(features, context).count_nonzero()
    with torch.no_grad():
        model.output.weight.normal_()
    scores = model(features, context)
    torch.testing.assert_close(scores.sum(-1), torch.zeros_like(scores.sum(-1)), atol=5e-5, rtol=0)
    features.zero_()
    assert not model(features, context).count_nonzero()
