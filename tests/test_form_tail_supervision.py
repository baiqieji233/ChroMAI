from fractions import Fraction

import pytest
import torch

from maichart.event_timing import EventPoint, EventTempo
from maichart.form.blueprint import build_blueprint_targets


def targets(second, *, first=0., frame_end=2., count=100):
    tempo = [EventTempo(Fraction(0), 120.)]
    frames = torch.arange(count, dtype=torch.float64) * .02 + first
    point = EventPoint(Fraction(str(second-first)) * 2, '1')
    return build_blueprint_targets([point], tempo, first, frames, frame_end_sec=frame_end)


def test_valid_onset_after_last_frame_center_keeps_tail_supervision():
    value = targets(1.99)
    assert float(value['onset_target'][-1]) == pytest.approx(.8824969)
    assert int(value['type_target'][-1]) == 0


@pytest.mark.parametrize('second', [2., 2.01])
def test_audio_end_is_exclusive_for_tail_targets(second):
    value = targets(second)
    assert not value['onset_target'].any()
    assert (value['type_target'] == -100).all()


def test_single_frame_short_audio_uses_declared_exclusive_end():
    value = targets(.009, frame_end=.01, count=1)
    assert value['onset_target'][0] > .9
    assert int(value['type_target'][0]) == 0


def test_nonzero_first_does_not_change_tail_frame_coverage():
    value = targets(2.49, first=.5, frame_end=2.5)
    assert float(value['onset_target'][-1]) == pytest.approx(.8824969)
    assert int(value['type_target'][-1]) == 0
