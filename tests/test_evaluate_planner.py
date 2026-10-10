from pathlib import Path
import sys

import torch
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.evaluate_planner import _channel_metrics, _condition_for_evaluation
from maichart.generation.conditions import GENERATION_DEFAULTS, RADAR_KEYS, STYLE_KEYS


def test_generation_defaults_ignore_chart_derived_condition_values():
    authored_a = torch.tensor([0.8] + [0.0] * (len(RADAR_KEYS) + len(STYLE_KEYS)))
    authored_b = torch.tensor([0.6] + [1.0] * (len(RADAR_KEYS) + len(STYLE_KEYS)))
    contract = {"condition_params": {"mask_prob": 0.5, "default": 0.5}}

    requested_a, effective_a, _ = _condition_for_evaluation(
        "generation_defaults", contract=contract, level=13.2, difficulty=4,
        authored_vector=authored_a)
    requested_b, effective_b, _ = _condition_for_evaluation(
        "generation_defaults", contract=contract, level=13.2, difficulty=4,
        authored_vector=authored_b)

    expected_radar = {key: GENERATION_DEFAULTS[key] for key in RADAR_KEYS}
    expected_style = {key: GENERATION_DEFAULTS[key] for key in STYLE_KEYS}
    assert requested_a.to_dict() == requested_b.to_dict()
    assert effective_a.to_vector().equal(effective_b.to_vector())
    assert requested_a.level == 13.2
    assert requested_a.difficulty == 4
    assert requested_a.radar == expected_radar
    assert requested_a.style == expected_style


def test_generation_defaults_still_apply_checkpoint_condition_guard():
    requested, effective, note = _condition_for_evaluation(
        "generation_defaults",
        contract={"condition_params": {"mask_prob": 1.0, "default": 0.5}},
        level=12.5,
        difficulty=5,
    )

    assert note == "neutralised"
    assert requested.level == effective.level == 12.5
    assert all(value == 0.5 for value in {**effective.radar, **effective.style}.values())


def test_authored_mode_is_explicitly_chart_controlled():
    vector = torch.tensor([13.0 / 15.0] + [0.8] * (len(RADAR_KEYS) + len(STYLE_KEYS)))
    requested, effective, note = _condition_for_evaluation(
        "authored", contract={"condition_params": {"mask_prob": 0.5}},
        level=13.3, difficulty=5, authored_vector=vector)

    assert all(abs(value - 0.8) < 1e-6 for value in requested.radar.values())
    assert all(abs(value - 0.8) < 1e-6 for value in requested.style.values())
    assert torch.allclose(effective.to_vector(), requested.to_vector())
    assert not note


def test_fine_metrics_report_mse_baselines_and_positive_target_subset():
    metrics = _channel_metrics(
        np.asarray([0.2, 0.4, 0.8], dtype=float),
        np.asarray([0.0, 0.5, 1.0], dtype=float),
        "fine_rhythm",
    )

    assert metrics["mse"] == pytest.approx(0.03)
    assert metrics["zero_baseline_mse"] == pytest.approx(0.4166666667)
    assert metrics["sample_oracle_mean_baseline_mse"] == pytest.approx(0.1666666667)
    assert metrics["positive_target_count"] == 2
    assert metrics["positive_target_fraction"] == pytest.approx(2 / 3)
    assert metrics["positive_target_mae"] == pytest.approx(0.15)
    assert metrics["positive_target_prediction_mean"] == pytest.approx(0.6)
