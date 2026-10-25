import pytest
import torch

from maichart.composer.generate import _grounded_pointer_probabilities


def _mass(probabilities, bins):
    return torch.stack([probabilities[bins == group].sum() for group in range(6)])


@pytest.mark.parametrize('temperature', [.1, .55, 1., 2.])
def test_nonuniform_joint_bin_mass_survives_temperature_and_truncation(temperature):
    bins = torch.tensor([0] * 20 + [1] * 3 + [5] * 25)
    logits = torch.linspace(-5., 4., len(bins))
    expected = _mass(logits.softmax(-1), bins)
    result = _grounded_pointer_probabilities(logits, bins, temperature, max_k=2)
    torch.testing.assert_close(_mass(result, bins), expected)
    torch.testing.assert_close(result.sum(), torch.tensor(1.))
    assert (result[bins == 0] > 0).sum() <= 2


@pytest.mark.parametrize('count,peaked', [(1, False), (6, False), (48, False), (48, True)])
def test_internal_count_and_peak_cannot_change_predicted_bin_mass(count, peaked):
    within = torch.zeros(count)
    if peaked:
        within[0] = 8.
    # Preserve unequal authored model mass while changing only the internal
    # candidate count/entropy; never assume equal probabilities across bins.
    joint = torch.cat((.3 * within.softmax(-1), torch.tensor([.2, .5])))
    bins = torch.tensor([0] * count + [1, 4])
    result = _grounded_pointer_probabilities(joint.log(), bins, .55, max_k=2)
    torch.testing.assert_close(_mass(result, bins), torch.tensor([.3, .2, 0., 0., .5, 0.]))


def test_mask_and_retries_condition_original_joint_mass_without_reviving_candidates():
    joint = torch.tensor([.1, .2, .3, .4])
    bins = torch.tensor([0, 0, 1, 5])
    eligible = torch.tensor([False, True, True, True])
    first = _grounded_pointer_probabilities(joint.log(), bins, .55, eligible=eligible)
    torch.testing.assert_close(first, torch.tensor([0., 2/9, 3/9, 4/9]))
    eligible[1] = False  # Entire first bin is now exhausted.
    retry = _grounded_pointer_probabilities(joint.log(), bins, .55, eligible=eligible)
    torch.testing.assert_close(retry, torch.tensor([0., 0., 3/7, 4/7]))
    masked_logits = joint.log().masked_fill(~eligible, float('-inf'))
    torch.testing.assert_close(retry, _grounded_pointer_probabilities(masked_logits, bins, .55))


def test_zero_temperature_selects_joint_argmax_instead_of_largest_bin():
    logits = torch.tensor([.2, .3, .4, .5]).log()
    bins = torch.tensor([0, 0, 0, 1])
    p = _grounded_pointer_probabilities(logits, bins, 0.)
    assert p.argmax().item() == logits.argmax().item() == 3
    assert torch.multinomial(p, 1, generator=torch.Generator().manual_seed(7)).item() == 3
    eligible = torch.tensor([True, True, True, False])
    masked = _grounded_pointer_probabilities(logits, bins, 0., eligible=eligible)
    assert masked.argmax().item() == 2


def test_t1_with_disabled_truncation_matches_joint_softmax():
    torch.manual_seed(5)
    logits = torch.randn(90)
    bins = torch.arange(90) % 6
    logits[4] = float('-inf')
    result = _grounded_pointer_probabilities(logits, bins, 1., top_p=1., min_p=0., max_k=0)
    torch.testing.assert_close(result, logits.softmax(-1))


def test_empty_support_and_extreme_finite_logits_are_explicit_and_finite():
    for temperature in (0., 1e-300, .55, 1.):
        bins = torch.tensor([0, 1, 5])
        empty = _grounded_pointer_probabilities(torch.full((3,), float('-inf')), bins, temperature)
        assert torch.equal(empty, torch.zeros(3))
        extreme = _grounded_pointer_probabilities(torch.tensor([-3e38, 0., 3e38]), bins, temperature)
        assert torch.isfinite(extreme).all()
        torch.testing.assert_close(extreme.sum(), torch.tensor(1.))
        assert extreme[2] == 1.


@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
def test_model_nonfinite_failures_are_not_hidden_even_by_eligibility_mask(bad):
    with pytest.raises(ValueError, match='NaN or positive infinity'):
        _grounded_pointer_probabilities(torch.tensor([bad, 0.]), torch.tensor([0, 1]), .55,
                                        eligible=torch.tensor([False, True]))


def test_candidate_order_and_bin_labels_do_not_change_probabilities():
    torch.manual_seed(9)
    logits = torch.randn(120)
    bins = torch.arange(120) % 6
    result = _grounded_pointer_probabilities(logits, bins, .8)
    order = torch.randperm(120)
    permuted = _grounded_pointer_probabilities(logits[order], (bins[order] + 3) % 6, .8)
    torch.testing.assert_close(result[order], permuted)
