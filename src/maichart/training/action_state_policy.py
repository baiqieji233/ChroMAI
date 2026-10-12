"""Explicit, optional migration to a sustained-action memory projection."""
from maichart.generation.action_state_spec import MAX_STATES, PROFILE, STATE_DIM, STATE_KEYS

PROFILES = ("none", PROFILE)


def resolve_action_state_profile(value="auto", *, checkpoint=None, mode="fresh"):
    stored = ((checkpoint or {}).get("model_config") or {}).get("action_state_profile", "none")
    if value not in ("auto", *PROFILES) or stored not in PROFILES:
        raise ValueError("持续动作摘要只能是 auto、none 或 sustain_state_v1")
    profile = stored if value == "auto" else value
    if mode == "resume" and profile != stored:
        raise ValueError("断点续训不能切换持续动作摘要；新增摘要请使用已有模型加强训练（finetune）")
    if checkpoint and stored == PROFILE and profile == "none":
        raise ValueError("当前 checkpoint 已包含持续动作输入，请保留该能力；无摘要模型仍可使用原路径生成")
    return profile


def state_contract():
    return {"profile": PROFILE, "feature_version": PROFILE, "state_dim": STATE_DIM,
            "max_states": MAX_STATES, "clock": "chart_audio_clock_v2", "causal": True}
