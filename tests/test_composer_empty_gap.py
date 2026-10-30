from fractions import Fraction

import numpy as np
import torch
import pytest

from maichart.composer.data import empty_gap_inputs
from maichart.composer.event_views import EventViews
from maichart.event_timing import EventPoint, EventTempo


class RecordingAudio:
    def __init__(self):
        self.seconds = []

    def at(self, seconds):
        self.seconds.extend(seconds.tolist())
        return np.repeat(seconds[:, None], 104, axis=1).astype(np.float32)


def views(*, excluded=(), annotation=None):
    return EventViews.build([EventPoint(Fraction(0), '1'), EventPoint(Fraction(8), '2')],
                           [EventTempo(Fraction(0), 120.)], 0., 8.,
                           excluded=excluded, annotation=annotation)


def test_empty_gap_four_midpoints_cover_true_seconds_and_preserve_confidence():
    audio = RecordingAudio()
    result = empty_gap_inputs(audio, views(annotation={
        'regions': [{'start_sec': 1.5, 'end_sec': 2., 'weight': .4}]}), 1., 4.)
    duration = 4. - .025 - 1.
    expected = 1. + (np.arange(4) + .5) * duration / 4.
    np.testing.assert_allclose(audio.seconds, expected)
    np.testing.assert_allclose(result['empty_gap_audio'][:, 0], expected)
    torch.testing.assert_close(result['empty_gap_exposure'], torch.full((4,), duration / 4.))
    torch.testing.assert_close(result['empty_gap_weight'], torch.full((4,), .4))


def test_unknown_gap_masks_all_probes_even_when_midpoints_miss_unknown_span():
    audio = RecordingAudio()
    result = empty_gap_inputs(audio, views(excluded=[(1.01, 1.02)]), 1., 4.)
    assert not audio.seconds
    assert all(not value.any() for value in result.values())


def test_nonpositive_gap_has_no_queries_or_supervision():
    audio = RecordingAudio()
    for previous, target in ((2., 2.), (2., 2.01), (3., 2.), (-1., .01)):
        result = empty_gap_inputs(audio, views(), previous, target)
        assert all(not value.any() for value in result.values())
    assert not audio.seconds


def test_negative_previous_clock_is_clamped_to_audio_start():
    audio = RecordingAudio()
    result = empty_gap_inputs(audio, views(), -2., 1.025)
    np.testing.assert_allclose(audio.seconds, [.125, .375, .625, .875])
    assert float(result['empty_gap_exposure'].sum()) == 1.


@pytest.mark.parametrize('confidence', [0., .4, 1.])
def test_empty_gap_loss_reduces_spurious_rates_and_masks_unknowns(confidence):
    from maichart.composer.train import _loss
    from test_composer_survival_training import _loss_inputs

    output, batch = _loss_inputs()
    rates = torch.full((1, 4), 2., requires_grad=True)
    output['empty_gap_rates'] = rates
    batch.update(empty_gap_exposure=torch.full((1, 4), .25),
                 empty_gap_weight=torch.full((1, 4), confidence))
    total, metrics = _loss(output, batch, 2)
    assert float(metrics['empty_gap_nll']) == pytest.approx(2. * confidence)
    total.backward()
    torch.testing.assert_close(rates.grad, torch.full_like(rates, .05 * confidence))
