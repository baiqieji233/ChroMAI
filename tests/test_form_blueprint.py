from fractions import Fraction

import pytest
import torch

from maichart.event_timing import EventPoint, EventTempo
from maichart.form.blueprint import (
    BlueprintConfig,
    SongBlueprint,
    build_blueprint_targets,
    compute_blueprint_loss,
)


TEMPO = (EventTempo(Fraction(0), 120.0),)


def _small_model():
    return SongBlueprint(BlueprintConfig(descriptor_dim=8, width=32,
                                         form_heads=4, form_layers=1,
                                         temporal_layers=2, dropout=0.0))


def test_full_song_forward_keeps_time_axis_and_uses_short_form_sequence():
    model = _small_model()
    audio = torch.randn(2, 1025, 120)
    form = torch.randn(2, 4, 8 + 3)
    condition = torch.randn(2, 16)
    audio_mask = torch.zeros(2, 1025, dtype=torch.bool)
    audio_mask[1, 900:] = True
    form_mask = torch.zeros(2, 4, dtype=torch.bool)
    form_mask[1, 3] = True
    output = model(audio, form, condition, audio_mask, form_mask)
    assert output["onset_logits"].shape == (2, 1025)
    assert output["type_logits"].shape == (2, 1025, 4)
    assert output["sustain_seconds"].shape == (2, 1025)
    assert torch.isfinite(output["onset_logits"]).all()
    assert torch.all(output["sustain_seconds"] >= 0)
    assert all(isinstance(block.conv, torch.nn.Conv1d) for block in model.temporal)


def test_masked_audio_values_and_padding_do_not_change_valid_frame_outputs():
    torch.manual_seed(12)
    model = _small_model().eval()
    short_audio = torch.randn(1, 48, 120)
    form = torch.randn(1, 3, 11)
    condition = torch.randn(1, 16)
    with torch.no_grad():
        baseline = model(short_audio, form, condition)
        padded_audio = torch.cat((short_audio, torch.full((1, 17, 120), 1e5)), dim=1)
        mask = torch.zeros(1, 65, dtype=torch.bool)
        mask[:, 48:] = True
        padded = model(padded_audio, form, condition, audio_padding_mask=mask)
    for key in baseline:
        torch.testing.assert_close(baseline[key], padded[key][:, :48], rtol=1e-5, atol=1e-6)


def test_forward_rejects_nonfinite_features():
    model = _small_model()
    audio = torch.zeros(1, 4, 120)
    audio[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="audio features must be finite"):
        model(audio, torch.zeros(1, 1, 11), torch.zeros(1, 16))


def test_form_transformer_has_a_fixed_descriptor_limit():
    model = _small_model()
    with pytest.raises(ValueError, match="max is 128"):
        model(torch.zeros(1, 8, 120), torch.zeros(1, 129, 11), torch.zeros(1, 16))


def test_target_builder_maps_triplet_and_sustain_through_exact_tempo_clock():
    points = [EventPoint(Fraction(1, 3), "1h[4:2]")]
    frames = torch.tensor([0.0, 0.08, 0.16, 0.24, 0.32])
    targets = build_blueprint_targets(points, TEMPO, 0.0, frames)
    assert targets["onset_target"][2] > 0.93
    assert targets["type_target"][2].item() == 1  # Hold.
    assert targets["duration_mask"][2]
    assert targets["duration_target_seconds"][2].item() == pytest.approx(1.0)


def test_chord_type_is_deterministic_and_out_of_clock_events_are_not_labeled():
    points = [EventPoint(Fraction(0), "1/2-6[4:1]"),
              EventPoint(Fraction(8), "5")]
    targets = build_blueprint_targets(points, TEMPO, 0.0, [0.0, 0.5, 1.0])
    assert targets["type_target"][0].item() == 2  # Slide outranks simultaneous tap.
    assert targets["type_target"][2].item() == -100
    assert targets["onset_target"][2].item() == 0.0


def test_empty_chart_produces_silent_targets_without_fabricated_types():
    targets = build_blueprint_targets([], TEMPO, 0.0, [0.0, 0.1, 0.2])
    assert torch.count_nonzero(targets["onset_target"]) == 0
    assert torch.all(targets["type_target"] == -100)
    assert not targets["duration_mask"].any()
    assert torch.count_nonzero(targets["duration_target_seconds"]) == 0


def test_full_song_onset_targets_use_bounded_local_gaussian_windows(monkeypatch):
    points = [EventPoint(Fraction(index, 8), str(index % 8 + 1)) for index in range(1000)]
    frames = torch.arange(5000, dtype=torch.float64) * 0.04
    original_exp = torch.exp
    maximum_window = 0

    def checked_exp(value):
        nonlocal maximum_window
        assert value.ndim == 1
        maximum_window = max(maximum_window, value.numel())
        return original_exp(value)

    monkeypatch.setattr(torch, "exp", checked_exp)
    targets = build_blueprint_targets(points, TEMPO, 0.0, frames)
    assert maximum_window <= 13
    assert targets["onset_target"].shape == (5000,)


def test_composite_loss_masks_padding_and_backpropagates_all_heads():
    torch.manual_seed(4)
    model = _small_model()
    points = [EventPoint(Fraction(1, 3), "A2h[4:2]"), EventPoint(Fraction(2), "3-7[4:1]")]
    target = build_blueprint_targets(points, TEMPO, 0.0, torch.arange(24, dtype=torch.float32) * 0.1)
    targets = {key: value.unsqueeze(0) for key, value in target.items()}
    audio = torch.randn(1, 24, 120)
    form = torch.randn(1, 3, 11)
    condition = torch.randn(1, 16)
    padding = torch.zeros(1, 24, dtype=torch.bool)
    padding[:, -3:] = True
    output = model(audio, form, condition, audio_padding_mask=padding)
    losses = compute_blueprint_loss(output, targets, audio_padding_mask=padding)
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    assert model.onset_head.weight.grad is not None
    assert model.type_head.weight.grad is not None
    assert model.duration_head.weight.grad is not None


def test_load_curve_loss_uses_the_sigmoid_prediction_scale():
    outputs = {
        "onset_logits": torch.zeros(1, 2),
        "type_logits": torch.zeros(1, 2, 4),
        "sustain_seconds": torch.zeros(1, 2),
        "load_curve": torch.tensor([[[-2.0, 2.0, 0.0, 1.0],
                                      [1.0, 0.0, 2.0, -2.0]]]),
    }
    targets = {
        "onset_target": torch.zeros(1, 2),
        "type_target": torch.full((1, 2), -100),
        "duration_target_seconds": torch.zeros(1, 2),
        "duration_mask": torch.zeros(1, 2, dtype=torch.bool),
        "load_curve_target": torch.tensor([[[0.1, 0.9, 0.5, 0.7],
                                            [0.7, 0.5, 0.9, 0.1]]]),
        "load_curve_mask": torch.ones(1, 2, 4, dtype=torch.bool),
    }
    losses = compute_blueprint_loss(outputs, targets)
    expected = torch.nn.functional.smooth_l1_loss(
        outputs["load_curve"].sigmoid(), targets["load_curve_target"])
    torch.testing.assert_close(losses["loss_load_curve"], expected)


def test_row_curve_mask_matches_explicit_channel_mask_normalization():
    outputs = {
        "onset_logits": torch.zeros(1, 2),
        "type_logits": torch.zeros(1, 2, 4),
        "sustain_seconds": torch.zeros(1, 2),
        "load_curve": torch.tensor([[[0.0, 1.0, -1.0, 2.0],
                                      [2.0, -1.0, 1.0, 0.0]]]),
    }
    targets = {
        "onset_target": torch.zeros(1, 2),
        "type_target": torch.full((1, 2), -100),
        "duration_target_seconds": torch.zeros(1, 2),
        "duration_mask": torch.zeros(1, 2, dtype=torch.bool),
        "load_curve_target": torch.full((1, 2, 4), 0.25),
        "load_curve_mask": torch.tensor([[True, False]]),
    }
    row_mask_loss = compute_blueprint_loss(outputs, targets)["loss_load_curve"]
    targets["load_curve_mask"] = targets["load_curve_mask"].unsqueeze(-1).expand_as(
        targets["load_curve_target"])
    channel_mask_loss = compute_blueprint_loss(outputs, targets)["loss_load_curve"]
    torch.testing.assert_close(row_mask_loss, channel_mask_loss)


def test_zero_weight_frames_do_not_affect_any_blueprint_loss():
    outputs = {
        "onset_logits": torch.tensor([[1.0, -1.0]]),
        "type_logits": torch.tensor([[[2.0, 0.0, 0.0, 0.0],
                                      [0.0, 2.0, 0.0, 0.0]]]),
        "sustain_seconds": torch.tensor([[2.0, 3.0]]),
        "load_curve": torch.tensor([[[0.0, 0.0, 0.0, 0.0],
                                      [1.0, 1.0, 1.0, 1.0]]]),
    }
    targets = {
        "onset_target": torch.tensor([[1.0, 0.0]]),
        "type_target": torch.tensor([[0, 1]]),
        "duration_target_seconds": torch.tensor([[2.0, 1.0]]),
        "duration_mask": torch.ones(1, 2, dtype=torch.bool),
        "load_curve_target": torch.tensor([[[0.5, 0.5, 0.5, 0.5],
                                            [0.0, 0.0, 0.0, 0.0]]]),
        "load_curve_mask": torch.ones(1, 2, 4, dtype=torch.bool),
        "frame_weight": torch.tensor([[1.0, 0.0]]),
    }
    baseline = compute_blueprint_loss(outputs, targets)
    excluded_changed = {key: value.clone() for key, value in outputs.items()}
    excluded_changed["onset_logits"][0, 1] = 25.0
    excluded_changed["type_logits"][0, 1] = torch.tensor([20.0, -20.0, 20.0, -20.0])
    excluded_changed["sustain_seconds"][0, 1] = 100.0
    excluded_changed["load_curve"][0, 1] = -20.0
    changed_targets = {key: value.clone() for key, value in targets.items()}
    changed_targets["onset_target"][0, 1] = 1.0
    changed_targets["type_target"][0, 1] = 3
    changed_targets["duration_target_seconds"][0, 1] = 50.0
    changed_targets["load_curve_target"][0, 1] = 1.0
    altered = compute_blueprint_loss(excluded_changed, changed_targets)
    for name in baseline:
        torch.testing.assert_close(baseline[name], altered[name])


def test_all_zero_frame_weights_produce_finite_zero_losses():
    outputs = {
        "onset_logits": torch.zeros(1, 2),
        "type_logits": torch.zeros(1, 2, 4),
        "sustain_seconds": torch.zeros(1, 2),
        "load_curve": torch.zeros(1, 2, 4),
    }
    targets = {
        "onset_target": torch.ones(1, 2),
        "type_target": torch.zeros(1, 2, dtype=torch.long),
        "duration_target_seconds": torch.ones(1, 2),
        "duration_mask": torch.ones(1, 2, dtype=torch.bool),
        "load_curve_target": torch.ones(1, 2, 4),
        "load_curve_mask": torch.ones(1, 2, 4, dtype=torch.bool),
        "frame_weight": torch.zeros(1, 2),
    }
    losses = compute_blueprint_loss(outputs, targets)
    assert all(torch.isfinite(value) for value in losses.values())
    assert all(value.item() == 0.0 for value in losses.values())

