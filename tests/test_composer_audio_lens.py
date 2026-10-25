from fractions import Fraction

import numpy as np
import pytest
import torch

from maichart.composer.audio_lens import AudioLens, LensBank, LENS_DIM, LENS_OFFSETS, PlacementTargets
from maichart.composer.event_views import EventViews
from maichart.event_timing import EventPoint, EventTempo


def _linear_bank(times):
    times = np.asarray(times, dtype=np.float64)
    return np.broadcast_to(times[:, None], (len(times), 120)).astype(np.float32).copy()


def test_lens_interpolates_nonuniform_source_clock_without_patch_normalization():
    times = np.array([0., .07, .19, .4, .73, 1.2])
    patches, valid = LensBank(_linear_bank(times), times, 1.3).query([.4, .7])
    expected = np.array([.4, .7])[:, None] + LENS_OFFSETS
    assert patches.shape == (2, 31, LENS_DIM)
    assert patches.dtype == np.float32 and valid.dtype == np.bool_
    assert valid.all()
    np.testing.assert_allclose(patches, np.broadcast_to(expected[..., None], patches.shape), atol=1e-6)
    assert patches[1].mean() > patches[0].mean()


def test_lens_uses_source_clock_shift_and_preserves_feature_groups():
    times = np.arange(0., 2.01, .02)
    values = np.zeros((len(times), 120), dtype=np.float32)
    values[:, 92:98] = np.arange(6) + 2.
    values[:, :80] = np.repeat(np.arange(8) + 10., 10)
    values[:, 80:92] = np.arange(12) + 20.
    original, valid = LensBank(values, times, 2.1).query([.8])
    shifted, shifted_valid = LensBank(values, times + 1., 3.1).query([1.8])
    np.testing.assert_allclose(original, shifted)
    np.testing.assert_array_equal(valid, shifted_valid)
    np.testing.assert_array_equal(original[0, 10], np.r_[np.arange(6) + 2., np.arange(8) + 10., np.arange(12) + 20.])


def test_lens_audio_and_source_boundaries_are_zero_masked_without_tail_copy():
    times = np.array([.1, .3, .5])
    patches, valid = LensBank(np.ones((3, 120)), times, .5).query([0., .5])
    clock = np.array([0., .5])[:, None] + LENS_OFFSETS
    expected = (clock >= .1) & (clock < .5)
    np.testing.assert_array_equal(valid, expected)
    assert np.all(patches[~valid] == 0.)
    assert np.all(patches[valid] == 1.)
    empty, empty_valid = LensBank(np.empty((0, 120)), [], 0.).query([0.])
    assert not empty_valid.any() and not empty.any()


def test_lens_rejects_ambiguous_source_clock():
    with pytest.raises(ValueError, match='strictly increasing'):
        LensBank(np.ones((2, 120)), [.2, .2], 1.)


def test_audio_lens_preserves_order_masks_missing_audio_and_chunks_candidates():
    torch.manual_seed(17)
    model = AudioLens(width=16, chunk_size=2).eval()
    patches = torch.randn(2, 5, 31, LENS_DIM)
    valid = torch.ones(2, 5, 31, dtype=torch.bool)
    valid[:, 0] = False
    patches[:, 0] = float('nan')
    chunked = model(patches, valid)
    assert chunked.shape == (2, 5, 16)
    assert torch.isfinite(chunked).all()
    assert torch.count_nonzero(chunked[:, 0]) == 0
    model.chunk_size = 32
    torch.testing.assert_close(model(patches, valid), chunked, atol=1e-5, rtol=1e-5)
    reversed_audio = model(patches.flip(-2), valid.flip(-1))
    assert not torch.allclose(chunked[:, 1:], reversed_audio[:, 1:])
    assert model(patches[:, :0], valid[:, :0]).shape == (2, 0, 16)


def test_audio_lens_cpu_amp_is_finite_with_partially_missing_windows():
    model = AudioLens(width=16)
    patches = torch.randn(1, 3, 31, LENS_DIM)
    valid = torch.ones(1, 3, 31, dtype=torch.bool)
    valid[:, :, :10] = False
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        output = model(patches, valid)
    assert torch.isfinite(output).all()
    output.float().square().mean().backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())


def test_placement_tracks_tempo_changes_and_tolerance_equivalent_candidates():
    tempo = [EventTempo(Fraction(0), 120.), EventTempo(Fraction(2), 60.)]
    points = [EventPoint(Fraction(1), '1'), EventPoint(Fraction(2), '2h[4:1]'),
              EventPoint(Fraction(3), '3-6[4:1]'), EventPoint(Fraction(4), 'B1b')]
    labels = PlacementTargets(points, tempo, .1, 4.)
    targets, weights = labels.query([.575, .6, .625, .626, 1.1, 2.1, 3.1])
    np.testing.assert_array_equal(targets[:3, 0], [1., 1., 1.])
    assert targets[3, 0] == 0.
    np.testing.assert_array_equal(targets[4], [1., 1., 0., 0., 0.])
    np.testing.assert_array_equal(targets[5], [1., 0., 1., 0., 0.])
    np.testing.assert_array_equal(targets[6], [1., 0., 0., 1., 1.])
    assert np.all(weights == 1.)


def test_placement_unions_close_events_instead_of_assigning_false_type_negatives():
    tempo = [EventTempo(Fraction(0), 60.)]
    points = [EventPoint(Fraction(1), '1h[4:1]'), EventPoint(Fraction(51, 50), 'B2')]
    targets, _ = PlacementTargets(points, tempo, 0., 3.).query([1.01])
    np.testing.assert_array_equal(targets[0], [1., 1., 0., 1., 0.])


def test_placement_unknown_neighborhoods_never_become_trusted_rests():
    tempo = [EventTempo(Fraction(0), 60.)]
    points = [EventPoint(Fraction(1), '1'), EventPoint(Fraction(2), '2')]
    views = EventViews.build(points, tempo, 0., 3., excluded=[(.98, 1.02)],
                             annotation={'regions': [{'start_sec': 1.8, 'end_sec': 2.2, 'weight': .25}]})
    targets, weights = PlacementTargets(points, tempo, 0., 3., views).query([.96, 1., 1.04, 1.1, 2., -.01, 3.])
    assert not weights[:3].any()
    assert not targets[:3].any()
    np.testing.assert_array_equal(weights[3], np.ones(5))
    np.testing.assert_array_equal(weights[4], np.full(5, .25))
    assert targets[4, 0] == 1.
    assert not weights[5:].any() and not targets[5:].any()
