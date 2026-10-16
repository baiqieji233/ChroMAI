"""Versioned musical-evidence contracts; absent metadata always means legacy."""
from __future__ import annotations

LEGACY = 'legacy'
MUSICAL_EVIDENCE = 'musical_evidence_v1'
QUALITY_PROFILES = (LEGACY, MUSICAL_EVIDENCE)
DEFAULT_ALIGNMENT_WEIGHT = .15
DEFAULT_ALIGNMENT_TOLERANCE_MS = 15.


def validate_quality_profile(profile):
    if profile not in QUALITY_PROFILES:
        raise ValueError('quality_profile must be legacy or musical_evidence_v1')
    return profile


def enhanced(profile):
    return validate_quality_profile(profile) == MUSICAL_EVIDENCE


def evidence_profile(profile):
    return 'balanced_evidence_v1' if enhanced(profile) else 'legacy'


def density_profile(profile):
    return 'musical_log_v1' if enhanced(profile) else 'legacy_linear_v1'


def duration_semantics(profile):
    return 'miacode_head_bpm_v1' if enhanced(profile) else 'legacy_tempo_integrated_v1'


def default_curve_bin_seconds(profile):
    return 1. if enhanced(profile) else 2.


def descriptor_dim(profile):
    return 512 if enhanced(profile) else 504
