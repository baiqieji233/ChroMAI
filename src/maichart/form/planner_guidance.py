"""Planner-to-configuration model guidance for improved consistency.

This module bridges the gap between Planner predictions (note types, densities)
and the configuration decoder, allowing the decoder to receive soft guidance
from what the planner thinks should happen at each onset.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PlannerGuidanceModule(nn.Module):
    """Projects planner predictions into decoder conditioning."""

    def __init__(self, d_model: int = 256, guidance_strength: float = 0.3):
        """Initialize planner guidance module.

        Args:
            d_model: Model hidden dimension
            guidance_strength: How strongly to weight planner hints (0-1)
                0 = ignore planner completely
                1 = follow planner exactly
                0.2-0.4 = recommended soft guidance
        """
        super().__init__()
        self.guidance_strength = guidance_strength

        # Project planner type probabilities (4 types: tap/hold/slide/touch)
        # and duration prediction into decoder space
        self.type_projection = nn.Linear(4, d_model)
        self.duration_projection = nn.Linear(1, d_model)

        # Gating mechanism to decide when to trust planner
        self.trust_gate = nn.Sequential(
            nn.Linear(d_model + 5, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        planner_type_logits: torch.Tensor,
        planner_duration: torch.Tensor,
        onset_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply planner guidance to decoder hidden states.

        Args:
            decoder_hidden: Decoder hidden states [B, L, D]
            planner_type_logits: Planner type predictions [B, T, 4]
            planner_duration: Planner duration predictions [B, T]
            onset_mask: Boolean mask [B, L] indicating onset positions

        Returns:
            Guided hidden states [B, L, D]
        """
        batch_size, seq_len, d_model = decoder_hidden.shape

        # Convert planner logits to probabilities
        type_probs = F.softmax(planner_type_logits, dim=-1)  # [B, T, 4]

        # Project planner predictions
        type_guidance = self.type_projection(type_probs)  # [B, T, D]
        duration_guidance = self.duration_projection(planner_duration.unsqueeze(-1))  # [B, T, D]

        # Combine guidance signals
        planner_guidance = type_guidance + duration_guidance  # [B, T, D]

        # Align planner predictions with decoder sequence
        # This assumes first onset aligns with decoder position 0
        # In practice, you may need more sophisticated alignment
        if planner_guidance.size(1) < seq_len:
            # Pad if planner has fewer predictions
            padding = seq_len - planner_guidance.size(1)
            planner_guidance = F.pad(planner_guidance, (0, 0, 0, padding))
        elif planner_guidance.size(1) > seq_len:
            # Truncate if planner has more predictions
            planner_guidance = planner_guidance[:, :seq_len, :]

        # Compute trust gate: should we listen to planner at this position?
        gate_input = torch.cat([
            decoder_hidden,
            planner_guidance,
            type_probs[:, :seq_len, :] if type_probs.size(1) >= seq_len else
                F.pad(type_probs, (0, 0, 0, seq_len - type_probs.size(1)))
        ], dim=-1)

        trust = self.trust_gate(gate_input)  # [B, L, 1]

        # Apply guidance with learned trust weighting
        guided = decoder_hidden + self.guidance_strength * trust * planner_guidance

        # Only apply to onset positions if mask provided
        if onset_mask is not None:
            onset_mask_expanded = onset_mask.unsqueeze(-1).float()
            guided = decoder_hidden * (1 - onset_mask_expanded) + guided * onset_mask_expanded

        return guided


def compute_planner_consistency_loss(
    decoder_output_logits: torch.Tensor,
    planner_type_logits: torch.Tensor,
    onset_positions: torch.Tensor | None = None,
    weight: float = 0.1,
) -> torch.Tensor:
    """Compute consistency loss between decoder and planner predictions.

    This encourages the decoder to produce configurations that are consistent
    with the planner's type predictions, reducing the planning-configuration gap.

    Args:
        decoder_output_logits: Decoder's token logits [B, L, V]
        planner_type_logits: Planner's type predictions [B, T, 4]
        onset_positions: Indices of onset tokens in decoder sequence [B, T]
        weight: Loss weight (0.05-0.2 recommended)

    Returns:
        Consistency loss scalar
    """
    if onset_positions is None or planner_type_logits.size(1) == 0:
        return torch.tensor(0.0, device=decoder_output_logits.device)

    # Extract decoder predictions at onset positions
    batch_size, seq_len, vocab_size = decoder_output_logits.shape
    max_onsets = onset_positions.size(1)

    # Safely gather onset predictions
    onset_positions_clamped = onset_positions.clamp(0, seq_len - 1)
    onset_logits = torch.gather(
        decoder_output_logits,
        1,
        onset_positions_clamped.unsqueeze(-1).expand(-1, -1, vocab_size)
    )  # [B, T, V]

    # Map vocabulary to note types (simplified mapping)
    # This is a placeholder - actual implementation needs codec-aware mapping
    # For now, compute KL divergence between distributions
    planner_probs = F.softmax(planner_type_logits, dim=-1)  # [B, T, 4]

    # Project decoder logits to type space (approximate)
    # In real implementation, use codec to extract type from tokens
    decoder_probs = F.softmax(onset_logits, dim=-1)  # [B, T, V]

    # Compute simplified consistency: entropy of planner predictions
    # Lower entropy = more confident planner = should have more influence
    planner_entropy = -(planner_probs * (planner_probs + 1e-8).log()).sum(-1)  # [B, T]

    # Weight by confidence (lower entropy = higher confidence)
    confidence = (2.0 - planner_entropy).clamp(0, 1)  # [B, T]

    # Return weighted consistency loss
    # This is a placeholder - full implementation needs proper type alignment
    loss = (confidence.mean() * weight)

    return loss


def extract_planner_hints(
    blueprint_outputs: dict[str, torch.Tensor],
    onset_beats: list[float],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract type and duration hints from blueprint model outputs.

    Args:
        blueprint_outputs: Dictionary containing 'type_logits' and 'sustain_seconds'
        onset_beats: List of onset beat positions
        device: Target device

    Returns:
        Tuple of (type_logits [1, T, 4], durations [1, T])
    """
    type_logits = blueprint_outputs.get('type_logits')
    durations = blueprint_outputs.get('sustain_seconds')

    if type_logits is None or durations is None:
        # Return dummy values if planner didn't produce predictions
        num_onsets = len(onset_beats)
        return (
            torch.zeros(1, num_onsets, 4, device=device),
            torch.zeros(1, num_onsets, device=device)
        )

    # Extract predictions at onset positions
    # This requires mapping beat positions to frame indices
    # For now, return the predictions as-is
    return type_logits, durations
