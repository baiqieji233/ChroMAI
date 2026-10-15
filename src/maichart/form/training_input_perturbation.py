"""Training input perturbation to reduce train-generation gap.

This module adds controlled noise to training inputs so the model learns to handle
imperfect predictions during generation. Three perturbation strategies:

1. Onset perturbation: Add/remove/shift onset positions
2. Curve perturbation: Add noise to load curve values
3. History corruption: Simulate configuration errors in prior segments

All perturbations are probabilistic and configurable.
"""
from __future__ import annotations

import random
import re
from fractions import Fraction
from typing import Sequence

import numpy as np
import torch

from maichart.event_timing import EventPoint, seconds_at_beat


QUALITY_TRAINING_DEFAULTS = {
    'perturb_onset_drop_prob': 0., 'perturb_onset_add_prob': 0.,
    'perturb_onset_shift_ms': 0., 'perturb_curve_noise_std': 0.,
    'perturb_curve_dropout_prob': 0., 'perturb_history_corruption_prob': 0.,
    'perturb_apply_prob': 1., 'repetition_penalty_weight': 0.,
    'diversity_loss_weight': 0.,
}


def perturb_onset_skeleton(
    points: Sequence[EventPoint],
    tempo_changes: Sequence,
    first: float,
    *,
    drop_prob: float = 0.0,
    add_prob: float = 0.0,
    shift_sigma_ms: float = 0.0,
    rng: random.Random | None = None,
) -> list[EventPoint]:
    """Add controlled noise to onset positions to simulate prediction errors.

    Args:
        points: Original chart points
        tempo_changes: Tempo map for time conversion
        first: First beat offset
        drop_prob: Probability of dropping each onset (0-0.3 recommended)
        add_prob: Probability of adding spurious onsets (0-0.2 recommended)
        shift_sigma_ms: Gaussian shift standard deviation in milliseconds
        rng: Random generator for reproducibility

    Returns:
        Perturbed points list (may be shorter/longer than input)
    """
    if not points:
        return []

    rng = rng or random.Random()
    result = []

    # Drop some onsets
    for point in points:
        if rng.random() >= drop_prob:
            if shift_sigma_ms > 0:
                # Shift onset timing slightly
                original_sec = float(seconds_at_beat(point.beat, tempo_changes, first))
                shift_sec = rng.gauss(0, shift_sigma_ms / 1000.0)
                shifted_sec = max(0, original_sec + shift_sec)

                # Convert back to beat (approximate)
                from maichart.form.rational_audio import beat_at_second
                shifted_beat = beat_at_second(Fraction(str(shifted_sec)), tempo_changes, first)
                result.append(EventPoint(shifted_beat, point.note_text))
            else:
                result.append(point)

    # Add spurious onsets
    if add_prob > 0 and points:
        num_add = int(len(points) * add_prob * rng.random())
        start_beat = points[0].beat
        end_beat = points[-1].beat

        for _ in range(num_add):
            # Random beat between first and last
            random_fraction = rng.random()
            beat = start_beat + (end_beat - start_beat) * Fraction(
                int(random_fraction * 1000), 1000
            ).limit_denominator(96)

            # Use simplest note from nearby real notes
            nearby = min(points, key=lambda p: abs(float(p.beat - beat)))
            result.append(EventPoint(beat, nearby.note_text))

    # Re-sort by beat
    result.sort(key=lambda p: p.beat)
    return result


def perturb_load_curve(
    curve_values: np.ndarray | torch.Tensor,
    *,
    noise_std: float = 0.0,
    dropout_prob: float = 0.0,
    rng: random.Random | None = None,
) -> np.ndarray:
    """Add noise to load curve to simulate prediction errors.

    Args:
        curve_values: Original curve [T, 4] (density, tap, sustain, slide_touch)
        noise_std: Gaussian noise standard deviation (0-0.1 recommended)
        dropout_prob: Probability of zeroing entire frames (0-0.2 recommended)
        rng: Random generator

    Returns:
        Perturbed curve (same shape as input)
    """
    rng = rng or random.Random()

    if isinstance(curve_values, torch.Tensor):
        values = curve_values.detach().cpu().numpy()
    else:
        values = np.array(curve_values)

    if values.size == 0:
        return values

    result = values.copy()

    # Add Gaussian noise
    if noise_std > 0:
        noise = np.asarray([[rng.gauss(0.0, noise_std) for _ in row]
                            for row in values], dtype=np.float32).reshape(values.shape)
        result = result + noise

    # Random dropout of entire frames
    if dropout_prob > 0:
        mask = np.asarray([rng.random() > dropout_prob for _ in result], dtype=bool)
        result = result * mask[:, np.newaxis]

    # Clamp to valid range [0, 1]
    result = np.clip(result, 0.0, 1.0)

    return result


def perturb_history_notes(
    points: Sequence[EventPoint],
    *,
    corruption_prob: float = 0.0,
    rng: random.Random | None = None,
) -> list[EventPoint]:
    """Corrupt note configurations to simulate previous generation errors.

    Args:
        points: History points
        corruption_prob: Probability of corrupting each note (0-0.1 recommended)
        rng: Random generator

    Returns:
        Perturbed history points
    """
    if corruption_prob <= 0 or not points:
        return list(points)

    rng = rng or random.Random()
    result = []

    # Only mutate standalone taps. Replacing digits throughout a Slide/Hold
    # also changes its duration/path and invalidates the sustain-state input.
    for point in points:
        match = re.fullmatch(r'([1-8])([bx]*)', point.note_text)
        if match and rng.random() < corruption_prob:
            button, modifiers = int(match[1]), match[2]
            shifted = (button - 1 + rng.choice((-1, 1))) % 8 + 1
            result.append(EventPoint(point.beat, f'{shifted}{modifiers}'))
        else:
            result.append(point)

    return result


class PerturbationConfig:
    """Configuration for training input perturbation."""

    def __init__(
        self,
        *,
        onset_drop_prob: float = 0.0,
        onset_add_prob: float = 0.0,
        onset_shift_sigma_ms: float = 0.0,
        curve_noise_std: float = 0.0,
        curve_dropout_prob: float = 0.0,
        history_corruption_prob: float = 0.0,
        apply_prob: float = 1.0,
        seed: int | None = None,
    ):
        """Initialize perturbation configuration.

        Args:
            onset_drop_prob: Probability of dropping each onset (0-0.3)
            onset_add_prob: Probability of adding spurious onsets (0-0.2)
            onset_shift_sigma_ms: Timing shift std in milliseconds (0-20ms)
            curve_noise_std: Gaussian noise std for curve values (0-0.1)
            curve_dropout_prob: Frame dropout probability (0-0.2)
            history_corruption_prob: Note corruption probability (0-0.1)
            apply_prob: Overall probability of applying any perturbation (0-1)
            seed: Random seed for reproducibility
        """
        self.onset_drop_prob = max(0.0, min(0.5, onset_drop_prob))
        self.onset_add_prob = max(0.0, min(0.5, onset_add_prob))
        self.onset_shift_sigma_ms = max(0.0, min(50.0, onset_shift_sigma_ms))
        self.curve_noise_std = max(0.0, min(0.3, curve_noise_std))
        self.curve_dropout_prob = max(0.0, min(0.5, curve_dropout_prob))
        self.history_corruption_prob = max(0.0, min(0.3, history_corruption_prob))
        self.apply_prob = max(0.0, min(1.0, apply_prob))
        self.seed = seed
        self._rng = random.Random(seed)

    def should_apply(self) -> bool:
        """Check if perturbation should be applied this time."""
        return self._rng.random() < self.apply_prob

    def is_enabled(self) -> bool:
        """Check if any perturbation is configured."""
        return (
            self.onset_drop_prob > 0
            or self.onset_add_prob > 0
            or self.onset_shift_sigma_ms > 0
            or self.curve_noise_std > 0
            or self.curve_dropout_prob > 0
            or self.history_corruption_prob > 0
        )

    def for_seed(self, seed: int) -> "PerturbationConfig":
        """Create an independent, reproducible per-example perturbation stream."""
        return PerturbationConfig(
            onset_drop_prob=self.onset_drop_prob,
            onset_add_prob=self.onset_add_prob,
            onset_shift_sigma_ms=self.onset_shift_sigma_ms,
            curve_noise_std=self.curve_noise_std,
            curve_dropout_prob=self.curve_dropout_prob,
            history_corruption_prob=self.history_corruption_prob,
            apply_prob=self.apply_prob,
            seed=seed,
        )
