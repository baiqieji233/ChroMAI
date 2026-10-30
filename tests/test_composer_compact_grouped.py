"""Compact geometry must equal one post-expansion kind-conserved bias."""
import pytest
import torch
from torch.nn import functional as F

from maichart.composer.action_factors import CompositionalActions
from maichart.composer.conserved_actions import CompactGroupedActionHead, group_logsumexp


ATOMS = ['1', '1b', '2', '1/2', '1h[4:1]', '2h[4:1]',
         '1-5[4:1]', '1-6[4:1]', '2-6[4:1]', 'A1', 'A1f']


def make_head(atoms=ATOMS):
    return CompactGroupedActionHead(CompositionalActions(atoms, 8), 8)


def test_combined_geometry_equals_atom_space_kind_conservation():
    torch.manual_seed(31)
    head = make_head()
    context = torch.randn(2, 3, 8)
    base = torch.randn(2, 3, len(ATOMS) + 1)
    head_bias = torch.randn(2, 3, 3, 8)
    detail_bias = torch.randn_like(base)
    original = head(base, context)
    expected = original.clone()
    for kind in range(3):
        groups = head.macro_kind_category == kind
        atoms = groups[head.atom_group_id].nonzero().flatten()
        projection = head.group_head_projection[head.atom_group_id[atoms]]
        biased = original[..., atoms] + F.linear(head_bias[..., kind, :], projection) + detail_bias[..., atoms]
        expected[..., atoms] = (biased + original[..., atoms].logsumexp(-1, keepdim=True)
                               - biased.logsumexp(-1, keepdim=True))
    actual = head(base, context, head_bias=head_bias, detail_bias=detail_bias)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-6)
    assert head.group_head_projection[head.atom_group_id[4]].sum() == 2
    # Slide endpoint 5 cannot become a starting button.
    assert head.group_head_projection[head.atom_group_id[7], 4] == 0


@pytest.mark.parametrize('atoms', [['1', '2'], ['A1', 'A2'], ['1-5[4:1]']])
def test_absent_kinds_are_finite_and_touch_stop_unchanged(atoms):
    head = make_head(atoms)
    context = torch.randn(2, 8)
    base = torch.randn(2, len(atoms) + 1)
    original = head(base, context)
    actual = head(base, context, head_bias=torch.randn(2, 3, 8), detail_bias=torch.randn_like(base))
    assert torch.isfinite(actual).all()
    untouched = head.macro_kind_category[head.atom_group_id] < 0
    torch.testing.assert_close(actual[..., untouched], original[..., untouched])
    torch.testing.assert_close(actual.exp().sum(-1), torch.ones(2))


def test_audio_reference_group_mass_and_gradient():
    head = make_head()
    context = torch.randn(4, 8, requires_grad=True)
    actual = head.audio_logits(context)
    expected = head.audio_macro_head(head.macro_norm(context)).float().log_softmax(-1)
    torch.testing.assert_close(group_logsumexp(actual, head.audio_macro_group_id,
                                             head.num_audio_groups), expected)
    F.nll_loss(actual, torch.tensor([1, 2, 5, 7])).backward()
    assert head.audio_macro_head.weight.grad.abs().sum() > 0
    assert head.macro_head.weight.grad is None


def test_macro_shared_factors_use_same_normalized_context():
    torch.manual_seed(42)
    factors = CompositionalActions(ATOMS, 8)
    head = CompactGroupedActionHead(factors, 8)
    for name in ('kind_mask', 'head_lane_mask', 'touch_ring'):
        factors.factors[name].weight.data.normal_()
    context = torch.randn(3, 8)
    base = torch.randn(3, len(ATOMS) + 1)
    normed = head.macro_norm(context)
    weight = (factors.factors['kind_mask'].weight[head.group_kind_id]
              + factors.factors['head_lane_mask'].weight[head.group_lane_id]
              + factors.factors['touch_ring'].weight[head.group_ring_id])
    expected = (head.macro_head(normed) + F.linear(normed, weight)).log_softmax(-1)
    actual = group_logsumexp(head(base, context), head.atom_group_id, head.num_groups)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-6)


def test_head_only_reuses_single_detail_partition(monkeypatch):
    import maichart.composer.conserved_actions as module

    head = make_head()
    calls = []
    original = module.group_logsumexp

    def counted(*args):
        calls.append(args[0].shape[-1])
        return original(*args)

    monkeypatch.setattr(module, 'group_logsumexp', counted)
    head(torch.randn(2, len(ATOMS) + 1), torch.randn(2, 8),
         head_bias=torch.randn(2, 3, 8))
    assert calls == [len(ATOMS) + 1]


def test_compact_biases_receive_macro_classification_gradients():
    head = make_head()
    base = torch.randn(2, len(ATOMS) + 1, requires_grad=True)
    context = torch.randn(2, 8)
    bias = torch.randn(2, 3, 8, requires_grad=True)
    logits = head(base, context, head_bias=bias)
    macro = group_logsumexp(logits, head.atom_group_id, head.num_groups)
    F.nll_loss(macro, head.atom_group_id[torch.tensor([1, 3])]).backward()
    assert head.macro_head.weight.grad.abs().sum() > 0
    assert bias.grad[:, 0].abs().sum() > 0


def test_bfloat16_geometry_keeps_fp32_mass_and_gradients():
    head = make_head()
    context = torch.randn(2, 8, requires_grad=True)
    bias = torch.randn(2, 3, 8, requires_grad=True)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        logits = head(torch.randn(2, len(ATOMS) + 1), context,
                      head_bias=bias, detail_bias=torch.randn(2, len(ATOMS) + 1))
        loss = F.nll_loss(logits, torch.tensor([1, 3]))
    assert logits.dtype == torch.float32
    torch.testing.assert_close(logits.exp().sum(-1), torch.ones(2), atol=2e-6, rtol=2e-6)
    loss.backward()
    assert torch.isfinite(bias.grad).all() and bias.grad.abs().sum() > 0
