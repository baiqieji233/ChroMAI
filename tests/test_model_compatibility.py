import pytest

from maichart.generation.model_compatibility import (
    FeatureSupport,
    ModelFamily,
    checkpoint_capabilities,
    detect_model_family,
    get_generation_config_for_model,
    validate_args_for_model,
)


EVENT_SCHEMA = "maichart-event-generator-checkpoint-v1"


def test_checkpoint_schema_selects_one_of_the_preserved_event_families():
    assert detect_model_family({"schema": EVENT_SCHEMA,
                                "data_contract": {"target_schema": "event_v1"}}) is ModelFamily.EVENT_V1
    assert detect_model_family({"schema": EVENT_SCHEMA,
                                "data_contract": {"target_schema": "event_ctx_v1"}}) is ModelFamily.EVENT_CTX_V1


def test_v3_capability_requires_matching_profile_and_dimensions():
    payload = {
        "schema": EVENT_SCHEMA,
        "model_config": {"planner_profile": "music_structure_v3", "plan_dim": 13},
        "data_contract": {"target_schema": "event_ctx_v1",
                           "planner_params": {"profile": "music_structure_v3", "plan_dim": 13}},
    }
    capabilities = checkpoint_capabilities(payload)
    assert capabilities["model_family"] == "event_ctx_v1"
    assert capabilities["fine_note_strength"] is True
    assert capabilities["rhythm_bias"] is False

    payload["model_config"]["plan_dim"] = 12
    capabilities = checkpoint_capabilities(payload)
    assert capabilities["valid"] is False
    assert capabilities["fine_note_strength"] is False
    assert capabilities["conflict_detection"] is True


def test_flags_and_oversampling_metadata_cannot_upgrade_v2():
    payload = {
        "schema": EVENT_SCHEMA,
        "supports_rhythm_features": True,
        "model_config": {"planner_profile": "music_structure_v2", "plan_dim": 12},
        "data_contract": {"target_schema": "event_ctx_v1",
                           "rhythm_features_v1": True,
                           "family_params": {"fine_rhythm_oversample": 3}},
    }
    capabilities = checkpoint_capabilities(payload)
    assert capabilities["valid"] is True
    assert capabilities["fine_note_strength"] is False
    assert capabilities["rhythm_bias"] is False


def test_unknown_or_legacy_payload_does_not_default_to_event_v1():
    with pytest.raises(ValueError, match="无法确认事件模型族"):
        detect_model_family({"schema": "future-unknown"})
    assert checkpoint_capabilities({"schema": "legacy-state-dict", "legacy": True})["model_family"] == "legacy_frame_v2"


@pytest.mark.parametrize("plan_dim", [None, 6, 9])
def test_legacy_event_profile_keeps_historical_head_widths_and_post_process(plan_dim):
    model_config = {"planner_profile": "legacy_v1"}
    if plan_dim is not None:
        model_config["plan_dim"] = plan_dim
    capabilities = checkpoint_capabilities({
        "schema": EVENT_SCHEMA,
        "model_config": model_config,
        "data_contract": {"target_schema": "event_v1"},
    })
    assert capabilities["valid"] is True
    assert capabilities["conflict_detection"] is True
    assert capabilities["fine_note_strength"] is False


def test_schema_less_checkpoint_needs_known_legacy_config_and_weight_signature():
    payload = {
        "schema": None,
        "model_config": {
            "input_dim": 32, "condition_dim": 4, "num_note_types": 7,
            "d_model": 16, "nhead": 4, "num_layers": 1,
            "dim_feedforward": 32, "dropout": 0.1, "max_len": 32,
            "positional_encoding": "sinusoidal",
        },
        "model_state_dict": {key: object() for key in {
            "input_projection.weight", "condition_projection.0.weight",
            "encoder.layers.0.self_attn.in_proj_weight", "final_norm.weight",
            "note_presence_head.weight", "button_head.weight", "type_head.weight", "density_head.weight",
        }},
    }
    assert detect_model_family(payload) is ModelFamily.LEGACY_FRAME_V2
    assert checkpoint_capabilities({"schema": None, "model_config": {"d_model": 16}})["model_family"] == "unknown"


def test_compatibility_enum_remains_three_families_with_deprecated_alias():
    assert ModelFamily.EVENT_CTX_V1_EXTENDED is ModelFamily.EVENT_CTX_V1
    assert FeatureSupport.get_features(ModelFamily.EVENT_CTX_V1)["rhythm_bias"] is False
    assert detect_model_family({"schema": "legacy-state-dict", "legacy": True}) is ModelFamily.LEGACY_FRAME_V2


def test_non_auto_rhythm_bias_is_rejected_and_v3_control_uses_payload():
    class Args:
        rhythm_bias = "triplet"
        fine_note_strength = 0.75
        enable_conflict_detection = True

    payload = {
        "schema": EVENT_SCHEMA,
        "model_config": {"planner_profile": "music_structure_v3", "plan_dim": 13},
        "data_contract": {"target_schema": "event_v1",
                           "planner_params": {"profile": "music_structure_v3", "plan_dim": 13}},
    }
    valid, warnings = validate_args_for_model(ModelFamily.EVENT_V1, Args(), checkpoint=payload)
    assert valid is False
    assert len(warnings) == 1 and "不受支持" in warnings[0]
    config = get_generation_config_for_model(ModelFamily.EVENT_V1, Args(), checkpoint=payload)
    assert config["fine_note_strength"] == 0.75
    assert config["rhythm_bias"] == "auto"
