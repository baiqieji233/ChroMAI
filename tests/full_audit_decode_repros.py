"""Read-only reproductions for song_form_v2 context omissions.

These tests intentionally document current behavior; they do not prescribe a
grid, denominator whitelist, or decoder policy.  They should remain useful
while the numerical-aware decoder is revised on a separate branch.
"""
from fractions import Fraction as F

import numpy as np
import torch

from maichart.event_timing import EventPoint, EventTempo
from maichart.form.data import collate_form, input_example, organize_song
from maichart.form.model import FormConfig, FormTransformer
from maichart.generation.event_audio import ACOUSTIC_DIM, EventAudio


def _song():
    times = np.arange(451, dtype=np.float32) / 50
    features = np.zeros((len(times), ACOUSTIC_DIM), dtype=np.float32)
    return organize_song(
        EventAudio(features, times, 9.0, 120.0),
        [EventTempo(F(0), 120)],
        0,
        min_seconds=0.2,
        max_seconds=0.5,
    )


def _model():
    torch.manual_seed(17)
    return FormTransformer(
        FormConfig(
            vocab_size=55,
            d_model=32,
            nhead=4,
            layers=1,
            dim_feedforward=64,
            dropout=0,
            audio_dim=120,
            descriptor_dim=504,
            condition_dim=16,
            max_tokens=16,
            max_audio_tokens=64,
            max_memory_tokens=64,
        )
    ).eval()


def _example(song, history):
    example = input_example(song, 12, history, max_memory_tokens=64)
    example["load_curve"] = torch.tensor(
        [[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5]], dtype=torch.float32
    )
    example["load_curve_condition_mask"] = torch.ones(2, 4, dtype=torch.bool)
    example["decoder_input"] = torch.tensor([1, 2], dtype=torch.long)
    return example


def test_load_curve_timestamps_are_not_consumed_by_form_transformer():
    """Changing only load_curve_times_sec leaves logits exactly unchanged."""
    song = _song()
    first = _example(song, [EventPoint(F(0), "1"), EventPoint(F(1), "2")])
    second = _example(song, [EventPoint(F(0), "1"), EventPoint(F(1), "2")])
    first["load_curve_times_sec"] = torch.tensor([0.0, 1.0])
    second["load_curve_times_sec"] = torch.tensor([17.25, 91.5])

    model = _model()
    with torch.inference_mode():
        left = model(collate_form([first]))["logits"]
        right = model(collate_form([second]))["logits"]

    torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_history_distance_is_absent_when_relative_tokens_and_action_state_match():
    """Equal relative history tokens hide different distance to the span edge."""
    song = _song()
    near = _example(song, [EventPoint(F(0), "1"), EventPoint(F(1), "2")])
    far = _example(song, [EventPoint(F(5), "1"), EventPoint(F(6), "2")])

    # The encoder receives identical tensors even though the retained history
    # covers different real time.  The metadata proves the semantic difference.
    assert torch.equal(near["history"], far["history"])
    assert torch.equal(near["action_state"], far["action_state"])
    assert near["meta"]["history_coverage_seconds"] != far["meta"]["history_coverage_seconds"]

    model = _model()
    with torch.inference_mode():
        left = model(collate_form([near]))["logits"]
        right = model(collate_form([far]))["logits"]

    torch.testing.assert_close(left, right, rtol=0, atol=0)
