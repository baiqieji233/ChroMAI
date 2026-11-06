"""Condition contracts use real chart measurements and explicit unknowns."""
from fractions import Fraction
from types import SimpleNamespace
import numpy as np
import pytest
from maichart.event_timing import EventPoint, EventTempo
from maichart.composer_v2.conditions import (
    CONDITION_KEYS, LEVEL_ONLY, RADAR_STYLE, condition_contract,
    policy_from_config, resolve_generation_condition, training_condition,
)


def test_automatic_is_unknown_and_explicit_zero_is_known():
    values, known, report = resolve_generation_condition(13.)
    assert known.tolist() == [True] + [False] * 15
    assert not values[1:].any()
    assert 'note_density' in report['automatic_keys']
    values, known, report = resolve_generation_condition(13., {'note_density': 0.})
    assert values[1] == 0. and known[1]
    assert 'note_density' not in report['automatic_keys']


@pytest.mark.parametrize('overrides', [{'note_density': float('nan')},
    {'peak': 1.01}, {'foo': .2}, {'melody_following': float('inf')}, {'level': 13.}])
def test_invalid_or_unmeasured_overrides_rejected(overrides):
    with pytest.raises(ValueError):
        resolve_generation_condition(13., overrides)


def test_legacy_is_level_only_and_does_not_claim_new_controls():
    assert policy_from_config({}) == LEVEL_ONLY
    assert condition_contract(LEVEL_ONLY)['supported_keys'] == ['level']
    with pytest.raises(ValueError, match='Level-only'):
        resolve_generation_condition(13., {'tap_bias': .8}, LEVEL_ONLY)
    assert not resolve_generation_condition(13., policy=LEVEL_ONLY)[1][1:].any()


def test_resume_inherits_condition_policy_and_init_migrates_only_forward():
    from maichart.composer_v2.train import resolve_training_policy
    assert resolve_training_policy(None) == (RADAR_STYLE, False)
    assert resolve_training_policy(None, {}, resume=True) == (LEVEL_ONLY, False)
    assert resolve_training_policy(None, {}) == (RADAR_STYLE, True)
    with pytest.raises(ValueError, match='init-checkpoint'):
        resolve_training_policy(RADAR_STYLE, {}, resume=True)
    with pytest.raises(ValueError, match='init-checkpoint'):
        resolve_training_policy(LEVEL_ONLY, {'condition_policy': RADAR_STYLE})


def test_training_measurements_and_generation_use_same_scale():
    points = [EventPoint(Fraction(0), '1'), EventPoint(Fraction(1), '2/6')]
    tempo = [EventTempo(Fraction(0), 120.)]
    values, known = training_condition(points, tempo, level=13., duration_sec=10.)
    melody = CONDITION_KEYS.index('melody_following')
    assert not known[melody] and values[melody] == 0.
    assert known.sum() == 15
    assert values[CONDITION_KEYS.index('tap_bias')] == 1.
    assert values[CONDITION_KEYS.index('chord')] == .5
    longer, _ = training_condition(points, tempo, level=13., duration_sec=20.)
    assert longer[1] == pytest.approx(values[1] / 2)
    requested = {key: float(values[index]) for index, key in enumerate(CONDITION_KEYS)
                 if index and known[index]}
    generated, generation_known, _ = resolve_generation_condition(13., requested, RADAR_STYLE)
    np.testing.assert_array_equal(generated, values)
    np.testing.assert_array_equal(generation_known, known)


def _flux_audio(onsets):
    times = np.arange(0., 4., .01)
    mel = np.zeros((len(times), 98), dtype=np.float32)
    for onset in onsets:
        mel[int(round(onset / .01)), 96] = 1.
    return SimpleNamespace(mel=mel, mel_times_sec=times)


def test_cached_flux_proxy_aligns_chart_first_on_audio_clock():
    points = [EventPoint(Fraction(0), '1'), EventPoint(Fraction(1), '2')]
    tempo = [EventTempo(Fraction(0), 120.)]
    key = CONDITION_KEYS.index('melody_following')
    audio = _flux_audio([.75, 1.25])
    values, known = training_condition(points, tempo, level=13., duration_sec=4.,
        audio=audio, first=.75)
    assert known[key] and values[key] == 1.
    # A real measured mismatch is a known zero, rather than missing audio.
    wrong, known = training_condition(points, tempo, level=13., duration_sec=4.,
        audio=audio, first=0.)
    assert known[key] and wrong[key] == 0.
    # Translating both the chart offset and the audio peaks preserves the score.
    shifted, shifted_known = training_condition(points, tempo, level=13., duration_sec=4.,
        audio=_flux_audio([1.25, 1.75]), first=1.25)
    np.testing.assert_array_equal(shifted, values)
    np.testing.assert_array_equal(shifted_known, known)
    generated, generated_known, _ = resolve_generation_condition(13., {'melody_following': 1.})
    assert generated[key] == 1. and generated_known[key]


@pytest.mark.parametrize('audio', [None, SimpleNamespace(), _flux_audio([]),
    SimpleNamespace(mel=np.ones((20, 98)), mel_times_sec=np.arange(20) * .01),
    SimpleNamespace(mel=np.full((20, 98), np.nan), mel_times_sec=np.arange(20) * .01)])
def test_unavailable_or_unusable_flux_is_unknown_not_measured_zero(audio):
    key = CONDITION_KEYS.index('melody_following')
    values, known = training_condition([EventPoint(Fraction(0), '1')],
        [EventTempo(Fraction(0), 120.)], level=13., duration_sec=4., audio=audio)
    assert not known[key] and values[key] == 0.


def test_flux_proxy_is_explicitly_not_a_melody_measurement():
    contract = condition_contract(RADAR_STYLE)['melody_following']
    assert contract['source'] == 'v2_logmel_positive_flux_column_96'
    assert contract['tolerance_sec'] == .08
    assert contract['meaning'] == 'rhythm_onset_alignment_proxy_not_melody_or_pitch'
