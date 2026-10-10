import json

import pytest

from maichart.generation.conditions import GENERATION_DEFAULTS, RADAR_KEYS, STYLE_KEYS
from maichart.generation.event_cli import resolve_generation_condition


def write_json(tmp_path, name, data):
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_level_only_file_keeps_generation_controls_without_profile(tmp_path):
    request = write_json(tmp_path, "request.json", {"level": 14.2, "difficulty": 4})
    condition, _ = resolve_generation_condition(level=13, condition_file=request)
    assert condition.level == 14.2 and condition.difficulty == 4
    assert condition.radar == {key: GENERATION_DEFAULTS[key] for key in RADAR_KEYS}
    assert condition.style == {key: GENERATION_DEFAULTS[key] for key in STYLE_KEYS}


def test_sparse_and_wrapped_explicit_controls_override_only_supplied_values(tmp_path):
    request = write_json(tmp_path, "request.json", {
        "radar": {"chord": {"target": 0.2}},
        "style": {"melody_following": 0, "touch_bias": {"value": 0.07}},
    })
    condition, _ = resolve_generation_condition(level=14.7, condition_file=request)
    assert condition.level == 14.7
    assert condition.radar["chord"] == .2
    assert condition.style["melody_following"] == 0
    assert condition.style["touch_bias"] == .07
    assert condition.radar["note_density"] == GENERATION_DEFAULTS["note_density"]


def test_profile_and_explicit_values_are_merged_before_contract_guard(tmp_path):
    profile = write_json(tmp_path, "profile.json", {
        "schema": "maichart-condition-profile-v1", "controls": {"note_density": {"median": .22}},
    })
    request = write_json(tmp_path, "request.json", {"difficulty": 4, "style": {"touch_bias": .1}})
    condition, _ = resolve_generation_condition(
        level=12.6, condition_file=request, condition_profile=profile)
    assert condition.radar["note_density"] == .22
    assert condition.style["touch_bias"] == .1
    neutral, source = resolve_generation_condition(
        level=12.6, condition_file=request, condition_profile=profile,
        contract_params={"mask_prob": 1., "default": .25})
    assert neutral.level == 12.6 and neutral.difficulty == 4
    assert set(neutral.radar.values()) == {.25}
    assert set(neutral.style.values()) == {.25}
    assert source.endswith("contract_mask_1.0")


def test_non_object_request_fails_explicitly(tmp_path):
    request = write_json(tmp_path, "request.json", [])
    with pytest.raises(ValueError, match="JSON object"):
        resolve_generation_condition(level=13, condition_file=request)
