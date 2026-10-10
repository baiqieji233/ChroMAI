"""Shared contracts for the music-structure planner.

Three profiles exist and each one is a strict extension of the previous:

``legacy_v1``          five continuous controls plus a four-way configuration
                       one-hot, mixed into six wide planner channels.
``music_structure_v2`` eight continuous controls (the legacy five plus Touch,
                       Hold and a local-workload intensity) and a separate
                       four-way configuration *distribution*.
``music_structure_v3`` adds a ninth continuous control, ``fine_rhythm``: the
                       share of onsets that sit closer together than a quarter
                       beat.  It is the one channel that lets the model express
                       "this passage wants sixteenth-note figuration" instead of
                       flattening it into quarter/eighth notes.

Only ``music_structure_v3`` can be conditioned on fine-note bias at generation
time, because conditioning needs a matching training target; the earlier
profiles keep their exact historical dimensions and semantics.
"""
from __future__ import annotations

import torch

LEGACY_PROFILE = "legacy_v1"
MUSIC_STRUCTURE_PROFILE = "music_structure_v2"
MUSIC_STRUCTURE_V3_PROFILE = "music_structure_v3"

# Profiles that share the structure-planner head (pooled local/global/condition
# encoder regions, sigmoid-regressed continuous channels, softmax configuration
# distribution).  Everything that used to test ``== MUSIC_STRUCTURE_PROFILE``
# should test membership here instead, so a future extension does not silently
# fall back to the legacy path.
MUSIC_STRUCTURE_PROFILES = frozenset({MUSIC_STRUCTURE_PROFILE, MUSIC_STRUCTURE_V3_PROFILE})

CONTINUOUS_KEYS = ("count", "density", "chord", "slide", "last", "touch", "hold", "intensity")
CONTINUOUS_KEYS_V3 = CONTINUOUS_KEYS + ("fine_rhythm",)
CONFIG_KEYS = ("single", "chord", "slide", "touch")
PLAN_DIM = {LEGACY_PROFILE: 9, MUSIC_STRUCTURE_PROFILE: 12, MUSIC_STRUCTURE_V3_PROFILE: 13}
# Length of the raw plan target tensor the datasets emit (continuous values plus
# one configuration-class index).  The planner head is four wider because the
# configuration index is expanded into logits.
PLAN_TARGET_DIM = {LEGACY_PROFILE: 9, MUSIC_STRUCTURE_PROFILE: 9, MUSIC_STRUCTURE_V3_PROFILE: 10}
# Number of leading channels that are sigmoid-regressed continuous controls.
CONTINUOUS_COUNT = {LEGACY_PROFILE: 5, MUSIC_STRUCTURE_PROFILE: 8, MUSIC_STRUCTURE_V3_PROFILE: 9}


def is_music_structure(profile: str) -> bool:
    """True for the structure-planner profiles (v2 and later)."""

    return profile in MUSIC_STRUCTURE_PROFILES


def continuous_keys(profile: str) -> tuple[str, ...]:
    """Continuous planner channels for a profile, in tensor order."""

    validate_profile(profile)
    if profile == MUSIC_STRUCTURE_V3_PROFILE:
        return CONTINUOUS_KEYS_V3
    if profile == MUSIC_STRUCTURE_PROFILE:
        return CONTINUOUS_KEYS
    return CONTINUOUS_KEYS[:5]


def validate_profile(profile: str) -> str:
    if profile not in PLAN_DIM:
        raise ValueError(
            f"unknown planner_profile {profile!r}; expected legacy_v1, "
            "music_structure_v2 or music_structure_v3"
        )
    return profile


def target_to_logits(target: torch.Tensor, *, profile: str = MUSIC_STRUCTURE_PROFILE) -> torch.Tensor:
    """Convert a raw plan target into planner-head logits for ``profile``.

    The target carries one entry per continuous channel plus a single
    configuration-class index; the returned tensor carries one entry per
    continuous channel plus four configuration logits.
    """

    validate_profile(profile)
    expected_target = PLAN_TARGET_DIM[profile]
    if target.ndim != 2 or target.size(1) != expected_target:
        raise ValueError(f"plan_target must have shape [B, {expected_target}] for {profile}.")
    eps = torch.finfo(target.dtype).eps if target.is_floating_point() else 1e-6
    continuous_count = CONTINUOUS_COUNT[profile]
    continuous = target[:, :continuous_count].to(dtype=torch.float32).clamp(eps, 1.0 - eps)
    if profile == LEGACY_PROFILE:
        # Legacy targets *are* the historical nine planner channels: the last
        # four entries already hold the configuration one-hot, so only the five
        # continuous channels need inverse-sigmoid conversion.
        return torch.cat((continuous.logit(), target[:, 5:9].to(torch.float32)), dim=1)
    category = target[:, continuous_count].to(torch.float32).mul(3.0).round().long().clamp(0, 3)
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
    continuous_count = CONTINUOUS_COUNT[profile]
    return torch.cat(
        (logits[:, :continuous_count].sigmoid(), logits[:, continuous_count:].softmax(dim=1)), dim=1)
