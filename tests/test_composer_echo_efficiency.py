"""Bounded-query features must preserve the original dense evidence exactly."""
import pytest
import torch

from maichart.composer.phase_echo import (
    ECHO_LAGS, ECHO_PERMUTATIONS, ECHO_TOLERANCE, PhaseEchoPrior,
    phase_echo_features,
)


def _dense_reference(lanes, ages, valid, cand_gap):
    """Return evidence and exact source sets; never read future chart events.

    Histories are chronological, ages measured from the last committed event.
    ``valid`` describes actual history padding, never stochastic dropout.
    """
    lanes, valid = lanes.bool(), valid.bool()
    ages, cand_gap = ages.float(), cand_gap.float()
    b, h, _ = lanes.shape
    c = cand_gap.size(1)
    lag = ages.new_tensor(ECHO_LAGS)
    k = len(ECHO_LAGS)
    permutation = torch.tensor(ECHO_PERMUTATIONS, device=lanes.device)
    button = lanes.any(-1) & valid
    counts = button.sum(-1)
    if not h:
        return dict(evidence=torch.zeros(b,k,4,device=lanes.device,dtype=torch.long),
                    rhythm_evidence=torch.zeros(b,k,device=lanes.device,dtype=torch.long),
                    candidate_source=torch.zeros(b,c,k,8,device=lanes.device,dtype=torch.bool),
                    has_candidate_source=torch.zeros(b,c,k,device=lanes.device,dtype=torch.bool),
                    rhythm_prediction=torch.zeros(b,c,k,device=lanes.device,dtype=torch.bool),
                    has_rhythm_source=torch.zeros(b,k,device=lanes.device,dtype=torch.bool),
                    rhythm_gap=ages.new_zeros(b,k),button_count=counts,event_count=valid.sum(-1))
    # [B,K,J,I]: I precedes J by the specified lag.
    pair = ((ages[:,None,None,:] - ages[:,None,:,None] - lag[None,:,None,None]).abs() < ECHO_TOLERANCE)
    pair = pair & valid[:,None,:,None] & valid[:,None,None,:]
    has_pair = pair.any(-1)
    source = torch.einsum('bkji,bil->bkjl', pair.float(), lanes.float()) > 0
    transformed = source[..., permutation].permute(0,1,3,2,4)
    copied = (transformed == lanes[:,None,None]).all(-1) & has_pair[:,:,None] & button[:,None,None]
    recent_buttons = button & (button.flip(1).long().cumsum(1).flip(1) <= 4)
    evidence = (copied & recent_buttons[:,None,None]).sum(-1)
    recent_events = valid & (valid.flip(1).long().cumsum(1).flip(1) <= 4)
    rhythm_evidence = (has_pair & recent_events[:,None]).sum(-1)
    wanted_age = lag[None,None] - cand_gap[...,None]
    matches = ((wanted_age[...,None] - ages[:,None,None]).abs() < ECHO_TOLERANCE)
    matches = matches & button[:,None,None] & (wanted_age >= 0)[...,None]
    candidate_source = torch.einsum('bcki,bil->bckl', matches.float(), lanes.float()) > 0
    has_source = matches.any(-1)
    anchors = ((ages[:,None] - lag[None,:,None]).abs() < ECHO_TOLERANCE) & valid[:,None]
    positions = torch.arange(h, device=lanes.device)
    anchor_index = positions[None,None].expand(b,k,h).masked_fill(~anchors, -1).max(-1).values
    following = valid[:,None] & (positions[None,None] > anchor_index[...,None])
    next_index = positions[None,None].expand(b,k,h).masked_fill(~following, h).min(-1).values
    rhythm_source = (anchor_index >= 0) & (next_index < h)
    next_age = ages.gather(1, next_index.clamp_max(h-1))
    rhythm_gap = lag[None] - next_age
    rhythm_prediction = ((cand_gap[...,None] - rhythm_gap[:,None]).abs() < ECHO_TOLERANCE) & rhythm_source[:,None]
    return dict(evidence=evidence, rhythm_evidence=rhythm_evidence,
                candidate_source=candidate_source,has_candidate_source=has_source,
                rhythm_prediction=rhythm_prediction,has_rhythm_source=rhythm_source,
                rhythm_gap=rhythm_gap,button_count=counts,event_count=valid.sum(-1))


@pytest.mark.parametrize("h", [0, 1, 3, 4, 5, 17, 64])
@pytest.mark.parametrize("seed", [0, 7, 31])
def test_bounded_queries_equal_dense_random_history(h, seed):
    generator = torch.Generator().manual_seed(seed)
    lanes = torch.rand(4, h, 8, generator=generator) < .18
    # Empty lanes represent Touch-only events. Include empty and padded rows,
    # including padding at age zero, and irregular/non-prefix validity.
    valid = torch.rand(4, h, generator=generator) > .25
    valid[0] = False
    if h:
        lanes[:, ::3] = False
    ages = torch.arange(h - 1, -1, -1).float()[None].expand(4, h).clone() * .5
    ages[~valid] = 0
    gaps = torch.tensor([[0., .5, 1., 2., 3., 6., 12., 20.]]).expand(4, -1)
    actual = phase_echo_features(lanes, ages, valid, gaps)
    expected = _dense_reference(lanes, ages, valid, gaps)
    assert actual.keys() == expected.keys()
    for key in actual:
        assert torch.equal(actual[key], expected[key]), key


def test_non_four_four_transforms_touch_and_future_sources_equal_dense():
    ages = torch.tensor([[18., 16., 15., 12., 10., 9., 6., 4., 3., 2., 1., 0.]])
    lanes = torch.zeros(1, 12, 8, dtype=torch.bool)
    for i in range(12):
        if i not in (7, 9, 10):  # Touch must not consume the four button queries.
            lanes[0, i, (i * 3) % 8] = True
            lanes[0, i, (i * 3 + 2) % 8] = True
    valid = torch.ones(1, 12, dtype=torch.bool)
    gaps = torch.tensor([[0., 1., 2., 3., 6., 12., 16., 17., 100.]])
    actual = phase_echo_features(lanes, ages, valid, gaps)
    expected = _dense_reference(lanes, ages, valid, gaps)
    for key in actual:
        assert torch.equal(actual[key], expected[key]), key
    assert not actual['has_candidate_source'][:, -1].any()
    assert not actual['candidate_source'][:, -1].any()


def test_optimized_features_leave_prior_zero_initialized():
    prior = PhaseEchoPrior(8)
    lanes = torch.ones(2, 16, 8, dtype=torch.bool)
    ages = torch.arange(15, -1, -1).float()[None].expand(2, -1)
    valid = torch.ones(2, 16, dtype=torch.bool)
    gaps = torch.tensor([[0., 1., 3.]]).expand(2, -1)
    lane, rhythm = prior(lanes, ages, valid, gaps, torch.randn(2, 3, 8), torch.ones(2, 1))
    assert torch.equal(lane, torch.zeros_like(lane))
    assert torch.equal(rhythm, torch.zeros_like(rhythm))


@pytest.mark.parametrize("offset", [0., .0005, .002])
def test_duplicate_ages_and_tolerance_keep_source_union(offset):
    # Preserve the dense implementation's union if multiple valid sources
    # match a lag, including entries around the tolerance boundary.
    ages = torch.tensor([[6., 6. + offset, 3., 3. + offset, 0., 0.]])
    lanes = torch.zeros(1, 6, 8, dtype=torch.bool)
    lanes[0, :, 0] = True
    lanes[0, 1::2, 4] = True
    valid = torch.tensor([[True, True, False, True, True, True]])
    gaps = torch.tensor([[0., 3., 6., 6.0005]])
    actual = phase_echo_features(lanes, ages, valid, gaps)
    expected = _dense_reference(lanes, ages, valid, gaps)
    for key in actual:
        assert torch.equal(actual[key], expected[key]), key
