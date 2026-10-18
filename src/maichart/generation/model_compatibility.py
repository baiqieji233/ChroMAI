"""Checkpoint family and generation capability detection."""

from __future__ import annotations

from enum import Enum
from typing import Any

from maichart.generation.music_plan import MUSIC_STRUCTURE_V3_PROFILE, PLAN_DIM


EVENT_CHECKPOINT_SCHEMA = "maichart-event-generator-checkpoint-v1"
STREAM_CHECKPOINT_SCHEMA = "maichart-chart-stream-checkpoint-v1"
EVENT_FAMILIES = {"event_v1", "event_ctx_v1"}
LEGACY_CHECKPOINT_SCHEMAS = {
    "legacy-state-dict",
    "maichart-conditioned-generator-v1",
    "maichart-conditioned-generator-checkpoint-v2",
}
EVENT_TARGET_ALIASES = {
    "event_v1": "event_v1", "maichart-event-v1": "event_v1",
    "event_ctx_v1": "event_ctx_v1", "maichart-event-ctx-v1": "event_ctx_v1",
}
LEGACY_MODEL_CONFIG_KEYS = {
    "input_dim", "condition_dim", "num_note_types", "d_model", "nhead",
    "num_layers", "dim_feedforward", "dropout", "max_len", "positional_encoding",
}
LEGACY_STATE_KEYS = {
    "input_projection.weight", "condition_projection.0.weight",
    "encoder.layers.0.self_attn.in_proj_weight", "final_norm.weight",
    "note_presence_head.weight", "button_head.weight", "type_head.weight", "density_head.weight",
}


class ModelFamily(Enum):
    SONG_FORM_V1 = "song_form_v1"
    SONG_FORM_V2 = "song_form_v2"
    EVENT_V1 = "event_v1"
    EVENT_CTX_V1 = "event_ctx_v1"
    LEGACY_FRAME_V2 = "legacy_frame_v2"
    CHART_STREAM_V1 = "chart_stream_v1"
    # Deprecated compatibility alias. Capability is determined from checkpoint
    # contents, never from a separate "extended" family label.
    EVENT_CTX_V1_EXTENDED = "event_ctx_v1"


def checkpoint_capabilities(checkpoint_data: dict[str, Any]) -> dict[str, Any]:
    """Derive generation capabilities from the checkpoint's declared schema."""
    if checkpoint_data.get('schema') == STREAM_CHECKPOINT_SCHEMA:
        config = checkpoint_data.get('model_config') or {}
        valid = checkpoint_data.get('family') == 'chart_stream_v1' and checkpoint_data.get('input_contract') == 'chart_stream_inputs_v1_exact_fraction'
        grounding=config.get('grounding_profile','none')
        grounded=valid and grounding=='causal_audio_v1'
        training=checkpoint_data.get('grounding_training') or {}
        return {'model_family':'chart_stream_v1','valid':bool(valid),'errors':[] if valid else ['chart_stream checkpoint contract mismatch'],
                'continuous_stream':bool(valid), 'rational_event_positions':bool(valid),
                'experimental':True,'conflict_detection':False,'conflict_repair':False,
                'fine_note_strength':False,'onset_profile':'continuous_event_stream',
                'local_onset_guidance':False,'rhythm_bias':False,'action_state_profile':'sustain_state_v1',
                'sustained_action_memory':True,'rhythm_geometry_profile':'exact_fraction_stream_v1',
                'generation_entry':'/chart-stream','quality_profile':'musical_evidence_v1',
                'descriptor_dim':0,'curve_bin_seconds':1.,'planner_context_seconds':0.,
                'audio_evidence_profile':'balanced_evidence_v1','position_alignment_supervision':bool(grounded),
                'grounding_profile':grounding,'causal_numeric_clock':bool(grounded),
                'native_local_audio':bool(grounded),'causal_archive_retrieval':bool(grounded),
                'dynamic_intent_routing':bool(grounded),
                'position_alignment_trained':bool(grounded and training.get('profile')==grounding
                    and training.get('alignment_supervised') is True and training.get('optimizer_steps',0)>0)}
    if checkpoint_data.get("schema") in {"maichart-song-form-checkpoint-v1","maichart-song-form-checkpoint-v2"}:
        from maichart.form.checkpoint import LEGACY_INPUT_CONTRACT, input_contract_for_configs
        rational=checkpoint_data.get('schema')=='maichart-song-form-checkpoint-v2'
        family='song_form_v2' if rational else 'song_form_v1'
        errors=[]
        if checkpoint_data.get("family") != family: errors.append("新架构模型族声明不一致")
        config = checkpoint_data.get('model_config') or {}
        planner_config = checkpoint_data.get('blueprint_config') or {}
        profile=config.get('rhythm_geometry_profile','none')
        quality=config.get('quality_profile','legacy')
        try:
            expected = input_contract_for_configs(config, planner_config, rational=rational)
        except (ValueError, TypeError, KeyError) as error:
            expected = None
            errors.append(str(error))
        contract=checkpoint_data.get('input_contract')
        from maichart.form.rhythm_geometry import GEOMETRY_PROFILES
        if profile not in GEOMETRY_PROFILES or (not rational and profile!='none'):
            errors.append('新架构节奏几何合同不匹配')
        if contract!=expected and not (not rational and contract==LEGACY_INPUT_CONTRACT):
            errors.append("新架构输入合同不匹配")
        alignment_capability = (checkpoint_data.get('capabilities') or {}).get('position_alignment') or {}
        alignment_available = quality == 'musical_evidence_v1' and not errors
        return {"model_family":family,"planner_profile":("musical_query_workload_temporal_v2"
                    if rational and quality != 'legacy' else "musical_query_workload_v1" if rational else "whole_song_blueprint_v1"),
                "valid":not errors,"errors":errors,"experimental":True,
                "conflict_detection":True,"conflict_repair":True,"fine_note_strength":False,
                "onset_profile":"none","local_onset_guidance":False,"rhythm_bias":False,
                "action_state_profile":"sustain_state_v1","sustained_action_memory":False,
                "rhythm_geometry_profile":profile,"generation_entry":"/",
                "quality_profile":quality,"descriptor_dim":config.get('descriptor_dim',504),
                "curve_bin_seconds":config.get('curve_bin_seconds',2.),
                "planner_context_seconds":planner_config.get('context_seconds',0.),
                "audio_evidence_profile":"balanced_evidence_v1" if quality != 'legacy' else 'legacy',
                "position_alignment_supervision":alignment_available,
                "position_alignment_trained":bool(alignment_available
                    and alignment_capability.get('contract') == 'authored_audio_queries_v1'
                    and alignment_capability.get('steps',0) > 0),
                "position_alignment_tolerance_ms":config.get('position_alignment_tolerance_ms',15.)}
    contract = checkpoint_data.get("data_contract") or {}
    if not isinstance(contract, dict):
        contract = {}
    target_schema = contract.get("target_schema") or checkpoint_data.get("target_schema")
    schema = checkpoint_data.get("schema")
    errors: list[str] = []

    config = checkpoint_data.get("model_config") or {}
    if not isinstance(config, dict):
        config = {}
        errors.append("model_config 不是对象")
    state = checkpoint_data.get("model_state_dict") or {}
    state_keys = set(state.keys()) if hasattr(state, "keys") else set()
    known_schema_less_legacy = (
        schema is None and (target_schema is None or target_schema == "legacy_frame_v2")
        and LEGACY_MODEL_CONFIG_KEYS <= set(config)
        and LEGACY_STATE_KEYS <= state_keys
    )
    if (isinstance(target_schema, str) and target_schema in EVENT_TARGET_ALIASES
            and (schema is None or schema == EVENT_CHECKPOINT_SCHEMA)):
        family = EVENT_TARGET_ALIASES[target_schema]
    elif (isinstance(schema, str) and schema in LEGACY_CHECKPOINT_SCHEMAS) or (
        checkpoint_data.get("checkpoint_kind") == "legacy_weights" or checkpoint_data.get("legacy")
    ) or known_schema_less_legacy:
        family = "legacy_frame_v2"
    else:
        family = "unknown"
        errors.append(f"无法确认 checkpoint schema/family: schema={schema!r}, target_schema={target_schema!r}")

    planner = contract.get("planner_params") or {}
    if not isinstance(planner, dict):
        planner = {}
        errors.append("planner_params 不是对象")
    config_profile = config.get("planner_profile")
    contract_profile = planner.get("profile")
    if config_profile and contract_profile and config_profile != contract_profile:
        errors.append("checkpoint model_config 与 data_contract 的 planner_profile 不一致")
    profile = config_profile or contract_profile or "legacy_v1"
    if str(profile) not in PLAN_DIM:
        errors.append(f"未知 planner_profile: {profile!r}")

    dimensions = []
    for name, value in (("model_config.plan_dim", config.get("plan_dim")),
                        ("planner_params.plan_dim", planner.get("plan_dim"))):
        if value is not None:
            try:
                dimensions.append((name, int(value)))
            except (TypeError, ValueError):
                errors.append(f"{name} 不是有效整数")
    head = state.get("plan_head.2.weight") if hasattr(state, "get") else None
    if head is not None and getattr(head, "ndim", 0) == 2:
        dimensions.append(("plan_head.2.weight", int(head.shape[0])))
    expected = PLAN_DIM.get(str(profile))
    allowed_dimensions = {6, 9} if profile == "legacy_v1" else ({expected} if expected is not None else set())
    if allowed_dimensions:
        errors.extend(f"{name}={value} 与 planner_profile={profile} 允许的 plan_dim={sorted(allowed_dimensions)} 矛盾"
                      for name, value in dimensions if value not in allowed_dimensions)
    if len({value for _, value in dimensions}) > 1:
        errors.append("checkpoint 中记录的 planner plan_dim 相互矛盾")

    event_family = family in EVENT_FAMILIES
    onset_profile = str(config.get("onset_profile", "none"))
    onset_contract = contract.get("onset_params") or {}
    onset_training = checkpoint_data.get("onset_training") or {}
    onset_trained = False
    if onset_profile not in {"none", "chart_onsets_v1"}:
        errors.append(f"未知 onset_profile: {onset_profile!r}")
    elif onset_profile == "chart_onsets_v1":
        if not event_family:
            errors.append("局部采音头仅支持事件模型")
        if profile != MUSIC_STRUCTURE_V3_PROFILE:
            errors.append("chart_onsets_v1 需要 music_structure_v3 规划器")
        if not isinstance(onset_contract, dict) or onset_contract.get("profile") != onset_profile:
            errors.append("checkpoint 缺少匹配的局部采音目标合同")
        if config.get("local_pooling_policy") != "per_sample_v2":
            errors.append("局部采音头需要 per_sample_v2 音频池化")
        try:
            width = int(config.get("d_model") or 0)
        except (TypeError, ValueError):
            width = 0
            errors.append("局部采音头 d_model 不是有效整数")
        hidden = max(1, width // 2)
        expected_onset_shapes = {"onset_head.0.weight": (hidden, width), "onset_head.0.bias": (hidden,),
                                 "onset_head.2.weight": (1, hidden), "onset_head.2.bias": (1,)}
        for key, shape in expected_onset_shapes.items():
            weight = state.get(key) if hasattr(state, "get") else None
            if weight is None or tuple(getattr(weight, "shape", ())) != shape:
                errors.append(f"局部采音头缺少有效权重: {key}")
        if isinstance(onset_training, dict) and isinstance(onset_contract, dict):
            try:
                onset_trained = (onset_training.get("profile") == onset_profile
                                 and onset_training.get("trained") is True
                                 and int(onset_training.get("optimizer_steps", 0)) > 0
                                 and float(onset_contract.get("loss_weight", 0)) > 0)
            except (TypeError, ValueError):
                errors.append("局部采音训练记录无效")
    action_profile = config.get("action_state_profile", "none")
    action_trained = False
    from maichart.generation.action_state_spec import STATE_DIM, STATE_KEYS
    action_weights = {key: next((state[name] for name in state_keys
                                if name == key or str(name).endswith('.' + key)), None)
                      for key in STATE_KEYS}
    if action_profile not in {"none", "sustain_state_v1"}:
        errors.append("未知持续动作摘要 profile")
    elif action_profile == "sustain_state_v1":
        from maichart.training.action_state_policy import state_contract
        if not event_family:
            errors.append("持续动作摘要仅支持事件模型")
        if contract.get("action_state_params") != state_contract():
            errors.append("持续动作摘要合同缺失或不兼容")
        try:
            width = int(config.get("d_model") or 0)
        except (TypeError, ValueError):
            width = 0
        expected_shapes = {"action_state_projection.weight": (width, STATE_DIM),
                           "action_state_projection.bias": (width,),
                           "action_state_type_embedding": (width,)}
        for key, shape in expected_shapes.items():
            if width <= 0 or tuple(getattr(action_weights[key], "shape", ())) != shape:
                errors.append(f"持续动作摘要缺少有效权重：{key}")
        action_training = checkpoint_data.get("action_state_training")
        steps = action_training.get("optimizer_steps") if isinstance(action_training, dict) else None
        if (not isinstance(action_training, dict) or action_training.get("profile") != action_profile
                or isinstance(steps, bool) or not isinstance(steps, int) or steps < 0
                or action_training.get("trained") is not (steps > 0)):
            errors.append("持续动作摘要训练记录无效")
        else:
            action_trained = steps > 0
    elif any(weight is not None for weight in action_weights.values()):
        errors.append("持续动作摘要已关闭但 checkpoint 仍包含其权重")
    usable_event_family = event_family
    v3_valid = (event_family and profile == MUSIC_STRUCTURE_V3_PROFILE and expected is not None
                and bool(dimensions)
                and (not dimensions or all(value == expected for _, value in dimensions))
                and not errors)
    return {
        "model_family": family,
        "planner_profile": str(profile),
        "valid": not errors,
        "errors": errors,
        "conflict_detection": usable_event_family,
        "conflict_repair": usable_event_family,
        "fine_note_strength": bool(v3_valid),
        "onset_profile": onset_profile,
        "local_onset_guidance": bool(event_family and onset_trained and not errors),
        "rhythm_bias": False,
        "action_state_profile": action_profile,
        "sustained_action_memory": bool(event_family and action_trained and not errors),
    }


def detect_model_family(checkpoint_data: dict[str, Any]) -> ModelFamily:
    """Return the event family; refuse unknown or legacy checkpoints."""
    family = checkpoint_capabilities(checkpoint_data)["model_family"]
    try:
        return ModelFamily(family)
    except ValueError as exc:
        raise ValueError(f"无法确认事件模型族: {family}") from exc


class FeatureSupport:
    """Compatibility API for callers that only know an event family."""
    @staticmethod
    def get_features(model_family: ModelFamily) -> dict[str, bool]:
        event_family = model_family in {ModelFamily.EVENT_V1, ModelFamily.EVENT_CTX_V1}
        return {
            "conflict_detection": event_family,
            "conflict_repair": event_family,
            "post_process": event_family,
            "rhythm_bias": False,
            "fine_note_strength": False,
            "local_onset_guidance": False,
        }


def validate_args_for_model(model_family: ModelFamily, args,
                            checkpoint: dict[str, Any] | None = None) -> tuple[bool, list[str]]:
    features = FeatureSupport.get_features(model_family)
    if checkpoint is not None:
        capabilities = checkpoint_capabilities(checkpoint)
        features.update({key: bool(capabilities[key]) for key in features if key in capabilities})
    warnings = []
    if hasattr(args, "rhythm_bias") and args.rhythm_bias != "auto":
        warnings.append("--rhythm-bias 已不受支持；请使用自动节奏跟随")
    if (hasattr(args, "fine_note_strength") and args.fine_note_strength != 0.5
            and not features["fine_note_strength"]):
        warnings.append(f"当前模型({model_family.value})不支持--fine-note-strength参数，将被忽略")
    if (hasattr(args, "enable_conflict_detection") and args.enable_conflict_detection
            and not features["conflict_detection"]):
        warnings.append(f"当前模型({model_family.value})不支持冲突检测")
    return not warnings, warnings


def get_generation_config_for_model(model_family: ModelFamily, args,
                                    checkpoint: dict[str, Any] | None = None) -> dict[str, Any]:
    config = {
        "model_family": model_family,
        "rhythm_bias": "auto",
        "fine_note_strength": 0.5,
        "local_onset_bias_strength": 0.0,
        "enable_conflict_detection": False,
    }
    capabilities = checkpoint_capabilities(checkpoint) if checkpoint is not None else None
    if (hasattr(args, "fine_note_strength") and capabilities is not None
            and capabilities["fine_note_strength"]):
        config["fine_note_strength"] = args.fine_note_strength
    onset_strength = float(getattr(args, "local_onset_bias_strength", 0.0))
    if onset_strength:
        if capabilities is None or not capabilities["local_onset_guidance"]:
            raise ValueError("局部采音对齐指导需要包含已训练采音头的 checkpoint；旧模型请使用强度 0")
        config["local_onset_bias_strength"] = onset_strength
    for key in ("enable_conflict_detection", "detect_hard_conflicts", "detect_soft_conflicts",
                "conflict_repair_mode", "conflict_retries"):
        if hasattr(args, key):
            config[key] = getattr(args, key)
    return config
