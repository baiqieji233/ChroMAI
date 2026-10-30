from fractions import Fraction

import numpy as np
import torch

from maichart.composer.fixed_horizon import FixedFutureAudio
from maichart.composer.quality_inputs import (
    QUALITY_FUTURE_OFFSETS, fixed_future_audio_inputs,
)
from maichart.event_timing import EventTempo
from maichart.tempo_clock import TempoClock


class RecordingAudio:
    def __init__(self):
        self.queries = []

    def at(self, seconds):
        self.queries.extend(seconds.tolist())
        return np.repeat(seconds[:, None], 104, axis=1).astype(np.float32)


def test_fixed_probes_cross_real_tempo_changes_without_candidate_keys():
    clock = TempoClock([EventTempo(Fraction(0), 120.), EventTempo(Fraction(2), 60.)])
    audio = RecordingAudio()
    result = fixed_future_audio_inputs(audio, clock, [1.], 20.)
    expected = clock.at_floats(1. + np.asarray(QUALITY_FUTURE_OFFSETS))
    np.testing.assert_allclose(result['quality_future_audio'][0, :, 0], expected)
    assert result['quality_future_mask'].all()
    assert result['quality_future_audio'].shape == (1, 10, 104)
    assert result['quality_future_audio'][0, -1, 0] == 8.


def test_out_of_audio_probes_are_zero_and_queries_are_shared():
    clock = TempoClock([EventTempo(Fraction(0), 120.)])
    audio = RecordingAudio()
    result = fixed_future_audio_inputs(audio, clock, [0., 0., 1.], 1.)
    mask = result['quality_future_mask']
    assert not result['quality_future_audio'][~mask].any()
    assert len(audio.queries) == len(set(audio.queries))
    before = list(audio.queries)
    repeated = fixed_future_audio_inputs(audio, clock, [0., 0., 1.], 1.)
    assert audio.queries == before
    torch.testing.assert_close(result['quality_future_audio'], repeated['quality_future_audio'])


def test_fixed_future_zero_init_masking_and_candidate_independence():
    module = FixedFutureAudio(16)
    current = torch.randn(2, 3, 16)
    inputs = {'quality_future_audio': torch.randn(2, 3, 10, 104),
              'quality_future_mask': torch.ones(2, 3, 10, dtype=torch.bool),
              'quality_future_offsets': torch.tensor(QUALITY_FUTURE_OFFSETS).expand(2, -1)}
    assert not module(current, inputs).any()
    with torch.no_grad():
        module.output.weight.fill_(.1)
    reference = module(current, inputs)
    inputs['quality_future_audio'][:, 1:] += 100.
    torch.testing.assert_close(module(current, inputs)[:, 0], reference[:, 0])
    inputs['quality_future_mask'].zero_()
    inputs['quality_future_audio'].fill_(float('nan'))
    assert not module(current, inputs).any()
