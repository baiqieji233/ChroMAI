"""Shared contracts for the music-structure planner.

The legacy planner is deliberately kept byte-for-byte compatible at the model
interface level.  ``music_structure_v2`` adds four local action statistics and
uses a separate configuration distribution.
"""
from __future__ import annotations

import torch

LEGACY_PROFILE = "legacy_v1"
MUSIC_STRUCTURE_PROFILE = "music_structure_v2"
CONTINUOUS_KEYS = ("count", "density", "chord", "slide", "last", "touch", "hold", "intensity")
CONFIG_KEYS = ("single", "chord", "slide", "touch")
PLAN_DIM = {LEGACY_PROFILE: 9, MUSIC_STRUCTURE_PROFILE: 12}


def validate_profile(profile: str) -> str:
    if profile not in PLAN_DIM:
        raise ValueError(f"unknown planner_profile {profile!r}; expected legacy_v1 or music_structure_v2")
    return profile


def target_to_logits(target: torch.Tensor, *, profile: str = MUSIC_STRUCTURE_PROFILE) -> torch.Tensor:
    """Convert Bx9 target values to planner logits (Bx12 for v2)."""
    validate_profile(profile)
    if target.ndim != 2 or target.size(1) != 9:
        raise ValueError("plan_target must have shape [B, 9].")
    eps = torch.finfo(target.dtype).eps if target.is_floating_point() else 1e-6
    continuous = target[:, :8].to(dtype=torch.float32).clamp(eps, 1.0 - eps)
    if profile == LEGACY_PROFILE:
        # Legacy targets are already the historical nine planner channels;
        # only the five continuous channels need inverse-sigmoid conversion.
        return torch.cat((continuous[:, :5].logit(), target[:, 5:9].to(torch.float32)), dim=1)
    category = target[:, 8].to(torch.float32).mul(3.0).round().long().clamp(0, 3)
    config = torch.full((target.size(0), 4), -8.0, device=target.device, dtype=torch.float32)
    config.scatter_(1, category.unsqueeze(1), 8.0)
    return torch.cat((continuous.logit(), config), dim=1)


def plan_values(logits: torch.Tensor, *, profile: str = MUSIC_STRUCTURE_PROFILE) -> torch.Tensor:
    """Return probability-space values: continuous controls then config probs."""
    validate_profile(profile)
    expected = PLAN_DIM[profile]
    if logits.ndim != 2 or logits.size(1) not in ({6, 9} if profile == LEGACY_PROFILE else {expected}):
        raise ValueError(f"plan logits must have shape [B, {expected}] (legacy also accepts 6).")
    if profile == LEGACY_PROFILE:
        return logits.sigmoid()
    return torch.cat((logits[:, :8].sigmoid(), logits[:, 8:].softmax(dim=1)), dim=1)
