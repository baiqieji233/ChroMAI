"""Versioned input-only migrations without changing event model tensor shapes."""
from __future__ import annotations

POOLING_POLICIES = ("legacy_uniform", "per_sample_v2")
CONTEXT_BUDGET_POLICIES = ("legacy_dynamic_v1", "bounded_v2")
CHART_FILTER_POLICIES = ("legacy_strict_v1", "drop_headless_stars_v1")
POLICIES = {
    "local_pooling_policy": POOLING_POLICIES,
    "context_budget_policy": CONTEXT_BUDGET_POLICIES,
    "chart_filter_policy": CHART_FILTER_POLICIES,
}
NEW_INPUT_POLICIES = {key: values[-1] for key, values in POLICIES.items()}


def stored_input_policies(checkpoint: dict | None) -> dict[str, str]:
    checkpoint = checkpoint or {}
    recorded = dict((checkpoint.get("data_contract") or {}).get("input_policies") or {})
    defaults = {key: values[0] for key, values in POLICIES.items()}
    defaults["local_pooling_policy"] = (checkpoint.get("model_config") or {}).get(
        "local_pooling_policy", ((checkpoint.get("data_contract") or {}).get("onset_params") or {}).get(
            "local_pooling_policy", "legacy_uniform"))
    result = {key: recorded.get(key, value) for key, value in defaults.items()}
    for key, value in result.items():
        if value not in POLICIES[key]:
            raise ValueError(f"checkpoint {key} 无效：{value!r}")
    if result["local_pooling_policy"] != defaults["local_pooling_policy"]:
        raise ValueError("checkpoint 音频池化策略与模型配置不一致")
    return result


def resolve_input_policies(requested: dict, *, checkpoint: dict | None = None,
                           mode: str = "fresh", onset_profile: str = "none") -> dict[str, str]:
    """Fresh/finetune default to corrected inputs; resume inherits exactly."""
    stored = stored_input_policies(checkpoint)
    result = {}
    for key, choices in POLICIES.items():
        request = str(requested.get(key, "auto")).strip().lower()
        if request not in ("auto", *choices):
            raise ValueError(f"{key} 只能是 auto 或 {' / '.join(choices)}")
        value = (stored[key] if mode == "resume" else NEW_INPUT_POLICIES[key]) if request == "auto" else request
        if mode == "resume" and value != stored[key]:
            raise ValueError(f"断点继续须继承 {key}={stored[key]}；切换输入策略请使用已有模型加强训练（finetune），无需从头训练。")
        result[key] = value
    effective_onset = ((checkpoint or {}).get("model_config") or {}).get("onset_profile", "none") if onset_profile == "auto" else onset_profile
    if effective_onset == "chart_onsets_v1" and result["local_pooling_policy"] != "per_sample_v2":
        raise ValueError("局部采音头需要 per_sample_v2 音频池化；可独立使用逐样本池化而不启用采音头。")
    return result
