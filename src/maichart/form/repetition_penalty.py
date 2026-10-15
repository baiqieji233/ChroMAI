"""Configuration repetition detection and penalty for training/generation.

Addresses the issue where models generate 5+ consecutive identical configurations,
which is a serious design flaw in rhythm game charts.
"""
from __future__ import annotations

from collections import Counter
from typing import Sequence

import torch
import torch.nn.functional as F


def detect_repetitions(
    token_sequence: Sequence[int],
    window_size: int = 8,
    min_repeat: int = 3,
) -> list[tuple[int, int, int]]:
    """Detect repetitive configuration patterns in token sequence.

    Args:
        token_sequence: Sequence of decoded tokens
        window_size: Size of pattern window to check (4-12 recommended)
        min_repeat: Minimum repetitions to flag (2-3 recommended)

    Returns:
        List of (start_pos, end_pos, repeat_count) tuples
    """
    if len(token_sequence) < window_size * 2:
        return []

    repetitions = []
    seq = list(token_sequence)

    for pattern_len in range(3, window_size + 1):
        i = 0
        while i <= len(seq) - pattern_len * 2:
            pattern = seq[i:i + pattern_len]

            # Count consecutive repetitions
            repeat_count = 1
            j = i + pattern_len

            while j + pattern_len <= len(seq):
                if seq[j:j + pattern_len] == pattern:
                    repeat_count += 1
                    j += pattern_len
                else:
                    break

            if repeat_count >= min_repeat:
                repetitions.append((i, j, repeat_count))
                i = j  # Skip past this repetition
            else:
                i += 1

    return repetitions


def compute_repetition_penalty_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    generated_so_far: torch.Tensor | None = None,
    penalty_weight: float = 0.1,
    lookback_window: int = 16,
) -> torch.Tensor:
    """Compute training loss penalty for repetitive configurations.

    Args:
        logits: Model output logits [B, L, V]
        targets: Target token IDs [B, L]
        generated_so_far: Previously generated tokens [B, L_prev]
        penalty_weight: Strength of repetition penalty (0.05-0.2 recommended)
        lookback_window: How far back to check for repetitions

    Returns:
        Penalty loss scalar
    """
    if penalty_weight <= 0:
        return torch.tensor(0.0, device=logits.device)

    batch_size, seq_len, vocab_size = logits.shape
    device = logits.device

    # Analyze target sequences for repetition patterns
    penalties = []

    for b in range(batch_size):
        target_seq = targets[b].cpu().tolist()

        # Combine with previously generated tokens if available
        if generated_so_far is not None:
            prev_seq = generated_so_far[b].cpu().tolist()
            full_seq = prev_seq + target_seq
        else:
            full_seq = target_seq

        # Detect repetitions
        reps = detect_repetitions(full_seq, window_size=8, min_repeat=3)

        # Compute penalty for each repetition
        for start, end, count in reps:
            if start < len(full_seq) - len(target_seq):
                # Repetition started in previous context
                # Penalize continuation in current sequence
                overlap_start = len(full_seq) - len(target_seq)
                overlap_len = end - overlap_start
                if overlap_len > 0:
                    penalties.append(count * overlap_len)

    if not penalties:
        return torch.tensor(0.0, device=device)

    # Average penalty across batch
    avg_penalty = sum(penalties) / batch_size

    return torch.tensor(avg_penalty * penalty_weight, device=device)


def apply_repetition_penalty_sampling(
    logits: torch.Tensor,
    generated_tokens: list[int],
    penalty_strength: float = 1.2,
    lookback_window: int = 32,
) -> torch.Tensor:
    """Apply repetition penalty during sampling/generation.

    Args:
        logits: Next-token logits [V]
        generated_tokens: Previously generated token sequence
        penalty_strength: Multiplicative penalty (1.0-1.5 recommended)
        lookback_window: How far back to check

    Returns:
        Penalized logits [V]
    """
    if penalty_strength <= 1.0 or len(generated_tokens) < 6:
        return logits

    # Count recent token frequencies
    recent = generated_tokens[-lookback_window:]
    token_counts = Counter(recent)

    # Penalize tokens that appeared recently
    penalized_logits = logits.clone()

    for token_id, count in token_counts.items():
        if count >= 2:  # Only penalize if appeared multiple times
            penalty = penalty_strength ** count
            penalized_logits[token_id] = penalized_logits[token_id] / penalty

    return penalized_logits


def compute_pattern_diversity_loss(
    hidden_states: torch.Tensor,
    segment_size: int = 8,
    diversity_weight: float = 0.05,
    valid_mask: torch.Tensor | None = None,
    reduction: str = 'mean',
) -> torch.Tensor:
    """Encourage diversity in hidden representations to avoid repetition.

    Args:
        hidden_states: Model hidden states [B, L, D]
        segment_size: Size of segments to compare
        diversity_weight: Loss weight (0.01-0.1 recommended)

    Returns:
        Diversity loss scalar
    """
    batch_size, seq_len, d_model = hidden_states.shape
    if reduction not in {'mean', 'none'}:
        raise ValueError('diversity reduction must be mean or none')
    if segment_size < 1:
        raise ValueError('segment_size must be positive')
    losses = hidden_states.float().sum((1, 2)) * 0
    if diversity_weight > 0 and seq_len >= segment_size * 2:
        mask = (torch.ones((batch_size, seq_len), dtype=torch.bool, device=hidden_states.device)
                if valid_mask is None else valid_mask.to(device=hidden_states.device, dtype=torch.bool))
        if mask.shape != hidden_states.shape[:2]:
            raise ValueError('diversity valid_mask must match token dimensions')
        num_segments = seq_len // segment_size
        length = num_segments * segment_size
        segments = hidden_states[:, :length].float().reshape(batch_size, num_segments, segment_size, d_model)
        mask = mask[:, :length].reshape(batch_size, num_segments, segment_size)
        counts = mask.sum(2)
        reps = (segments * mask[..., None]).sum(2) / counts.clamp_min(1)[..., None]
        reps = F.normalize(reps, dim=-1)
        pair_counts = counts.new_zeros(batch_size)
        for offset in range(1, min(4, num_segments)):
            selected = (counts[:, :-offset] > 0) & (counts[:, offset:] > 0)
            similarity = (reps[:, :-offset] * reps[:, offset:]).sum(-1)
            losses = losses + (similarity.clamp(.5, 1.) * selected).sum(1)
            pair_counts = pair_counts + selected.sum(1)
        losses = losses / pair_counts.clamp_min(1) * diversity_weight
    return losses if reduction == 'none' else losses.mean()


class RepetitionMonitor:
    """Runtime monitor for tracking repetition statistics during training."""

    def __init__(self):
        self.total_sequences = 0
        self.sequences_with_repetition = 0
        self.max_consecutive_repeat = 0
        self.repeat_counts = Counter()

    def update(self, token_sequences: list[list[int]]):
        """Update statistics with a batch of sequences."""
        for seq in token_sequences:
            self.total_sequences += 1
            reps = detect_repetitions(seq, window_size=8, min_repeat=2)

            if reps:
                self.sequences_with_repetition += 1
                max_repeat = max(r[2] for r in reps)
                self.max_consecutive_repeat = max(self.max_consecutive_repeat, max_repeat)
                self.repeat_counts[max_repeat] += 1

    def get_stats(self) -> dict:
        """Get current statistics."""
        if self.total_sequences == 0:
            return {
                "total_sequences": 0,
                "repetition_rate": 0.0,
                "max_consecutive_repeat": 0,
            }

        return {
            "total_sequences": self.total_sequences,
            "sequences_with_repetition": self.sequences_with_repetition,
            "repetition_rate": self.sequences_with_repetition / self.total_sequences,
            "max_consecutive_repeat": self.max_consecutive_repeat,
            "repeat_distribution": dict(self.repeat_counts),
        }

    def reset(self):
        """Reset all statistics."""
        self.total_sequences = 0
        self.sequences_with_repetition = 0
        self.max_consecutive_repeat = 0
        self.repeat_counts.clear()
