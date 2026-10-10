import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from evaluate_onset_alignment import score_frame_predictions, weighted_average_precision


def test_frame_metrics_exclude_margin_padding_and_zero_quality_bins():
    result = score_frame_predictions(
        probabilities=[0.9, 0.1, 0.8, 0.4, 0.2],
        target=[1.0, 0.0, 1.0, 0.0, 0.0],
        quality=[1.0, 1.0, 0.0, 1.0, 1.0],
        valid=[1.0, 1.0, 1.0, 0.0, 1.0],
    )
    assert result["valid_frame_bins"] == 3
    assert result["positive_target_frame_bins"] == 1
    assert result["positive_target_fraction"] == 1 / 3
    assert result["average_precision"] == 1.0
    assert np.isfinite(result["soft_bce"])
    assert np.isfinite(result["brier"])


def test_same_score_average_precision_groups_ties_and_matches_prevalence():
    score = weighted_average_precision(
        scores=[0.5, 0.5, 0.5, 0.5],
        labels=[True, False, True, False],
        weights=[1.0, 1.0, 1.0, 1.0],
    )
    assert score == 0.5


def test_all_negative_slice_has_defined_brier_but_no_average_precision_or_recall():
    result = score_frame_predictions(
        probabilities=[0.1, 0.2, 0.3],
        target=[0.0, 0.0, 0.0],
        quality=[1.0, 1.0, 1.0],
        valid=[1.0, 1.0, 1.0],
    )
    assert result["positive_target_fraction"] == 0.0
    assert result["average_precision"] is None
    assert result["thresholds"]["0.5"]["recall"] is None
    assert result["brier"] > 0.0
    assert result["constant_sample_mean_brier_oracle"] == 0.0


def test_zero_quality_slice_returns_null_metrics_and_zero_counts():
    result = score_frame_predictions(
        probabilities=[0.8, 0.2], target=[1.0, 0.0],
        quality=[0.0, 0.0], valid=[1.0, 1.0],
    )
    assert result["valid_frame_bins"] == 0
    assert result["soft_bce"] is None
    assert result["average_precision"] is None
    assert result["thresholds"]["0.3"]["predicted_positive_frame_bins"] == 0
