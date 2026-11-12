"""Versioned Composer v2 conditions; unknown means infer from music/history.

Values use the shared exact-event normalization. Unknown slots contain zero
and are distinguished from measured zero by an explicit boolean mask.
"""
from __future__ import annotations

import math
import numpy as np
from maichart.generation.conditions import CONDITION_KEYS

LEVEL_ONLY = 'level_only_v1'
RADAR_STYLE = 'radar_style_v1'
POLICIES = (LEVEL_ONLY, RADAR_STYLE)
SUPPORTED_KEYS = tuple(CONDITION_KEYS)


def condition_contract(policy):
    if policy not in POLICIES:
        raise ValueError('Unknown Composer v2 condition policy: ' + str(policy))
    return dict(policy=policy, keys=list(CONDITION_KEYS),
        supported_keys=['level'] if policy == LEVEL_ONLY else list(SUPPORTED_KEYS),
        normalization='level_div_15_zero_optional' if policy == LEVEL_ONLY
            else 'level_div_15_shared_exact_event_radar_style_v1',
        missing='zero_value_with_explicit_known_mask',
        duration_semantics=None if policy == LEVEL_ONLY else 'miacode_head_bpm_v1',
        duration_source=None if policy == LEVEL_ONLY else 'full_audio_duration_sec',
        melody_following='unsupported_no_onset_features' if policy == LEVEL_ONLY else dict(
            meaning='rhythm_onset_alignment_proxy_not_melody_or_pitch',
            source='v2_logmel_positive_flux_column_96', times='mel_times_sec_absolute_audio_clock',
            threshold='mean_plus_0.5_std_finite_nonnegative_flux', tolerance_sec=.08,
            chart_to_audio_offset='audio_first', unavailable='known_false'),
        training_mask_policy='whole_optional_25pct_independent_optional_15pct_v1'
            if policy == RADAR_STYLE else 'none')


def policy_from_config(config):
    policy = (config or {}).get('condition_policy', LEVEL_ONLY)
    condition_contract(policy)
    return policy


def resolve_condition(level, radar=None, style=None, *, policy=RADAR_STYLE, max_level=15.):
    """Resolve explicit requests; omitted/null optional values stay unknown."""
    contract = condition_contract(policy)
    values = np.zeros(len(CONDITION_KEYS), dtype=np.float32)
    known = np.zeros(len(CONDITION_KEYS), dtype=np.bool_)
    raw = {'level': level}
    for section in (radar, style):
        if section is not None:
            if not isinstance(section, dict):
                raise ValueError('Radar / Style must be objects')
            for key, value in section.items():
                if key not in CONDITION_KEYS[1:]:
                    raise ValueError('Unknown condition: ' + str(key))
                raw[key] = value
    for index, key in enumerate(CONDITION_KEYS):
        value = raw.get(key)
        if value is None:
            continue
        if key not in contract['supported_keys']:
            raise ValueError('Checkpoint does not support condition: ' + key)
        if isinstance(value, bool):
            raise ValueError('Condition must be a finite number: ' + key)
        number = float(value)
        lower, upper = (1., max_level) if key == 'level' else (0., 1.)
        if not math.isfinite(number) or not lower <= number <= upper:
            raise ValueError(f'Condition {key} must be finite and in [{lower}, {upper}]')
        values[index] = number / 15. if key == 'level' else number
        known[index] = True
    return values, known


def resolve_generation_condition(level, overrides=None, policy=RADAR_STYLE):
    overrides = {} if overrides is None else overrides
    if not isinstance(overrides, dict):
        raise ValueError('Condition overrides must be an object')
    if 'level' in overrides:
        raise ValueError('Use the separate level argument')
    if policy == LEVEL_ONLY and overrides:
        raise ValueError('Level-only checkpoint has no trained Radar / Style controls')
    values, known = resolve_condition(level, overrides, policy=policy)
    report = dict(policy=policy, values=values.tolist(), known=known.tolist(),
        overrides={key: float(value) for key, value in overrides.items() if value is not None},
        automatic_keys=[key for key, available in zip(CONDITION_KEYS, known) if not available
                        and key in condition_contract(policy)['supported_keys']],
        unsupported_keys=[key for key in CONDITION_KEYS if key not in condition_contract(policy)['supported_keys']])
    return values, known, report


def _audio_onset_proxy(audio, first):
    """Use cached v2 positive spectral flux; never compute a new spectrogram.

    Every above-threshold frame is retained, matching the existing shared
    event-quality onset proxy. This is rhythm alignment, not melody analysis.
    """
    if audio is None:
        return None
    try:
        mel = np.asarray(audio.mel)
        times = np.asarray(audio.mel_times_sec, dtype=np.float64)
        if mel.ndim != 2 or mel.shape[1] <= 96 or times.ndim != 1 or len(mel) != len(times):
            return None
        flux = np.asarray(mel[:, 96], dtype=np.float64)
        valid = np.isfinite(times) & (times >= 0) & np.isfinite(flux) & (flux >= 0)
        if valid.sum() < 2:
            return None
        measured = flux[valid]
        deviation = float(measured.std())
        if float(measured.max()) <= 0 or deviation <= 0:
            return None
        threshold = float(measured.mean()) + .5 * deviation
        selected = valid & (flux >= threshold) & (flux > 0)
        if not selected.any():
            return None
        offset = float(first)
        if not math.isfinite(offset):
            raise ValueError('audio_first must be finite')
        return dict(onsets=times[selected].tolist(), first=offset)
    except (AttributeError, TypeError, IndexError):
        return None


def training_condition(points, tempo_changes, *, level, duration_sec, policy=RADAR_STYLE,
                       audio=None, first=0., max_level=15.):
    if policy == LEVEL_ONLY:
        return resolve_condition(level, policy=policy, max_level=max_level)
    from maichart.generation.event_quality import condition_from_events, MIACODE_DURATION_SEMANTICS
    proxy = _audio_onset_proxy(audio, first)
    condition = condition_from_events(points, tempo_changes, level=level,
        duration_sec=duration_sec, duration_semantics=MIACODE_DURATION_SEMANTICS,
        audio=proxy)
    style = dict(condition.style)
    if proxy is None:
        style.pop('melody_following', None)
    return resolve_condition(level, condition.radar, style, policy=policy, max_level=max_level)
