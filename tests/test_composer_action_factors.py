import copy

import pytest
import torch
from torch.nn import functional as F

from maichart.composer.action_factors import (
    CompositionalActions, FACTOR_FIELDS, KIND_BITS, TOUCH_RING_BITS,
)


def value(model, field, atom_id):
    return model.factor_values[field][int(getattr(model, f'atom_{field}')[atom_id])]


def test_neutral_residual_keeps_legal_ids_and_has_first_step_gradients():
    model = CompositionalActions(['1', '2', '1h[4:1]', '1-4[8:1]'], 4)
    context = torch.arange(1., 25.).reshape(2, 3, 4).requires_grad_()
    residual = model(context)
    assert residual.shape == (2, 3, 5)
    torch.testing.assert_close(residual, torch.zeros_like(residual))
    baseline = torch.randn_like(residual)
    torch.testing.assert_close(baseline + residual, baseline)
    residual[..., 1].sum().backward()
    for embedding in model.factors.values():
        assert embedding.weight.grad is not None
        assert torch.count_nonzero(embedding.weight.grad) > 0
    for name in FACTOR_FIELDS:
        assert getattr(model, f'atom_{name}').shape == (5,)
        assert int(getattr(model, f'atom_{name}')[0]) == 0


def test_shared_input_output_weights_and_scalar_embedding_ids():
    model = CompositionalActions(['1', '2b', '3x'], 3)
    with torch.no_grad():
        for factor in model.factors.values():
            factor.weight.copy_(torch.arange(factor.weight.numel()).reshape_as(factor.weight))
    context = torch.randn(2, 3)
    torch.testing.assert_close(model(context), F.linear(context, model.embed(torch.arange(4))))
    assert model.embed(torch.tensor(2)).shape == (3,)
    assert model.embed(torch.tensor([[0, 1], [2, 3]])).shape == (2, 2, 3)


def test_parsed_chords_kind_masks_touch_and_modifiers():
    model = CompositionalActions(['15', '1/5', '2xb', 'B3f', 'Ch[4:1]', '1/Ch[4:1]'], 4)
    expected = torch.tensor([True, False, False, False, True, False, False, False])
    torch.testing.assert_close(model.atom_head_lanes[1], expected)
    torch.testing.assert_close(model.atom_head_lanes[2], expected)
    assert value(model, 'kind_mask', 1) == str(KIND_BITS['tap'])
    assert value(model, 'break_ex', 3) == '3'
    assert value(model, 'touch_ring', 4) == str(TOUCH_RING_BITS['B'])
    assert value(model, 'firework', 4) == '1'
    assert not model.atom_head_lanes[4].any()
    assert value(model, 'kind_mask', 6) == str(KIND_BITS['tap'] | KIND_BITS['touch_hold'])
    assert int(model.atom_head_lane_mask[1]) == int(model.atom_head_lane_mask[2])


def test_duration_categories_are_rational_and_keep_timing_semantics():
    atoms = ['1h[4:1]', '2h[8:2]', '3h[#0.5]', '4h[#.500]',
             '5h[120##0.5]', '6h[120.0##.500]', '7h', '8h[1:0]']
    model = CompositionalActions(atoms, 4)
    assert value(model, 'hold_duration', 1) == value(model, 'hold_duration', 2)
    assert value(model, 'hold_duration', 3) == value(model, 'hold_duration', 4)
    assert value(model, 'hold_duration', 5) == value(model, 'hold_duration', 6)
    assert value(model, 'hold_duration', 7) == value(model, 'hold_duration', 8)
    assert len({value(model, 'hold_duration', i) for i in (1, 3, 5)}) == 3


def test_slide_factors_share_relative_geometry_and_keep_chains_and_branches():
    atoms = ['1-4[8:1]', '2-5[16:2]', '1V84[8:1]',
             '2V15[8:1]', '1-4-6[8:1]', '1-4[8:1]*-6[8:1]',
             '1-6[8:1]*-4[8:1]']
    model = CompositionalActions(atoms, 4)
    for field in ('trajectory_shape', 'relative_endpoint', 'slide_duration'):
        assert value(model, field, 1) == value(model, field, 2)
    assert value(model, 'relative_endpoint', 3) == value(model, 'relative_endpoint', 4)
    assert value(model, 'trajectory_shape', 1) != value(model, 'trajectory_shape', 5)
    assert value(model, 'branch_count', 5) == '1'
    assert value(model, 'branch_count', 6) == '2'
    for field in ('trajectory_shape', 'relative_endpoint', 'slide_duration'):
        assert value(model, field, 6) == value(model, field, 7)


def test_relative_logits_use_all_heads_and_have_no_slot_hand_rule():
    model = CompositionalActions(['1', '3', '13', 'C'], 2)
    context = torch.ones(2, 4, 2)
    previous = torch.tensor([[True, False, False, False, True, False, False, False],
                             [False] * 8])
    torch.testing.assert_close(model.relative_logits(context, previous), torch.zeros(2, 4, 5))
    with torch.no_grad():
        model.relative_factors.weight.copy_(torch.arange(16.).reshape(8, 2))
    actual = model.relative_logits(context, previous)
    assert actual.shape == (2, 4, 5)
    torch.testing.assert_close(actual[0, :, 3], (actual[0, :, 1] + actual[0, :, 2]) / 2)
    torch.testing.assert_close(actual[1], torch.zeros_like(actual[1]))
    torch.testing.assert_close(actual[..., 0], torch.zeros_like(actual[..., 0]))
    torch.testing.assert_close(actual[..., 4], torch.zeros_like(actual[..., 4]))
    single_previous = previous[0:1].expand(2, -1)
    changed = model.relative_logits(context, single_previous)
    torch.testing.assert_close(changed[0], changed[1])
    actual.sum().backward()
    assert torch.count_nonzero(model.relative_factors.weight.grad) > 0


def test_mapping_round_trip_and_same_size_catalog_mismatch():
    source = CompositionalActions(['1', '2h[4:1]'], 4)
    with torch.no_grad():
        source.factors['kind_mask'].weight.fill_(.5)
    target = CompositionalActions(['1', '2h[4:1]'], 4)
    target.load_state_dict(copy.deepcopy(source.state_dict()))
    torch.testing.assert_close(source.composed_weight(), target.composed_weight())
    for atoms in (['2h[4:1]', '1'], ['1', '3h[4:1]']):
        other = CompositionalActions(atoms, 4)
        before = other.catalog_fingerprint.clone()
        with pytest.raises(ValueError, match='catalog identity'):
            other.load_state_dict(source.state_dict(), strict=False)
        torch.testing.assert_close(other.catalog_fingerprint, before)


def test_modified_field_buffer_and_missing_identity_are_rejected():
    model = CompositionalActions(['1', '2'], 4)
    state = copy.deepcopy(model.state_dict())
    state['atom_head_lane_mask'][1] = 0
    with pytest.raises(ValueError, match='factor buffer mismatch'):
        model.load_state_dict(state)
    state = copy.deepcopy(model.state_dict())
    state.pop('_extra_state')
    with pytest.raises(ValueError, match='catalog identity'):
        model.load_state_dict(state, strict=False)


@pytest.mark.parametrize('atoms', [[''], ['1', '1'], ['not-a-note'], ['1//2'], ['1h[bad]'], ['STOP']])
def test_invalid_catalogs_fail_explicitly(atoms):
    with pytest.raises(ValueError):
        CompositionalActions(atoms, 4)


def test_empty_catalog_and_invalid_input_shapes():
    model = CompositionalActions([], 4)
    assert model(torch.ones(2, 4)).shape == (2, 1)
    with pytest.raises(ValueError, match='width'):
        model(torch.ones(2, 3))
    with pytest.raises(ValueError, match='eight'):
        model.relative_logits(torch.ones(2, 4), torch.ones(2, 7))
    with pytest.raises(ValueError, match='broadcast'):
        model.relative_logits(torch.ones(2, 3, 4), torch.ones(4, 8))
