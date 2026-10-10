"""Strict checkpoint contracts for the event generator."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from maichart.generation.event_model import EventConditionedTransformer, EventTransformerConfig, EventVocabulary
from maichart.training.checkpoints import load_checkpoint

EVENT_CHECKPOINT_SCHEMA = "maichart-event-generator-checkpoint-v1"

# Contract keys that only exist for some families.  They are compared when
# either side carries them, and ignored otherwise so older checkpoints keep
# working.
OPTIONAL_CONTRACT_KEYS = ("family_params", "condition_params", "quality_params", "planner_params", "onset_params")
LEGACY_CONDITION_TARGET_VERSION = "legacy_unshifted_v1"
CURRENT_CONDITION_TARGET_VERSION = "chart_audio_clock_v2"
INIT_MIGRATABLE_FAMILY_KEYS = frozenset({
    "slide_oversample", "touch_oversample", "oversample_policy",
    "fine_rhythm_oversample", "fine_rhythm_ratio_threshold", "fine_rhythm_stats_version",
    "onset_profile",
})
LEGACY_TOUCH_PRIORITY_POLICY = "independent_note_types_touch_priority_v1"
CURRENT_TOUCH_PRIORITY_POLICY = "independent_note_types_touch_priority_fine_rhythm_v2"


def _normalize_resume_sampling(params: dict[str, Any] | None, *, side: str) -> dict[str, Any]:
    """Canonicalize only known, distribution-equivalent sampling contracts."""
    normalized = dict(params or {})
    raw_multiplier = normalized.get("fine_rhythm_oversample", 1)
    try:
        number = float(raw_multiplier)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Checkpoint {side} has an invalid fine-rhythm sampling multiplier; use finetune.") from exc
    if isinstance(raw_multiplier, bool) or not math.isfinite(number) or not number.is_integer() or number < 1:
        raise ValueError(f"Checkpoint {side} has an invalid fine-rhythm sampling multiplier; use finetune.")
    fine_multiplier = int(number)
    if fine_multiplier <= 1:
        for key in ("fine_rhythm_oversample", "fine_rhythm_ratio_threshold", "fine_rhythm_stats_version"):
            normalized.pop(key, None)
    elif not normalized.get("fine_rhythm_stats_version"):
        raise ValueError(
            f"Checkpoint {side} uses fine-rhythm oversampling without a statistics version; "
            "its sampling distribution cannot be proven equivalent. Use finetune (--init-checkpoint)."
        )
    policy = normalized.get("oversample_policy")
    if policy == LEGACY_TOUCH_PRIORITY_POLICY:
        normalized["oversample_policy"] = CURRENT_TOUCH_PRIORITY_POLICY
    return normalized


def vocabulary_signature(vocabulary: EventVocabulary) -> str:
    return hashlib.sha256(json.dumps(vocabulary.to_dict(), sort_keys=True).encode()).hexdigest()


def load_event_checkpoint(path: str | Path, *, device: str | torch.device = "cpu") -> dict[str, Any]:
    payload = load_checkpoint(path, map_location=device)
    if payload.get("schema") != EVENT_CHECKPOINT_SCHEMA:
        raise ValueError("需要 event_v1 checkpoint；旧固定帧模型不能作为事件模型续训/初始化。请选择旧版基线或重新训练事件模型。")
    vocabulary = EventVocabulary.from_dict(payload["vocabulary"])
    config = dict(payload.get("model_config", {}))
    # Planner was added after the first event checkpoints. If an old state dict
    # carries a planner head, infer its width before constructing the model;
    # otherwise the current default remains a safe fresh-training width.
    if "plan_dim" not in config:
        head = payload.get("model_state_dict", {}).get("plan_head.2.weight")
        if head is not None and getattr(head, "ndim", 0) == 2:
            config["plan_dim"] = int(head.shape[0])
        payload["model_config"] = config
    if config.get("vocab_size") != len(vocabulary):
        raise ValueError("Checkpoint vocabulary/config size mismatch.")
    if payload.get("data_contract", {}).get("vocabulary_signature") != vocabulary_signature(vocabulary):
        raise ValueError("Checkpoint vocabulary signature mismatch.")
    condition_target_version = payload.get("data_contract", {}).get(
        "condition_target_version", LEGACY_CONDITION_TARGET_VERSION)
    if condition_target_version not in {LEGACY_CONDITION_TARGET_VERSION, CURRENT_CONDITION_TARGET_VERSION}:
        raise ValueError(f"Unknown condition_target_version {condition_target_version!r}.")
    onset_profile = str(config.get("onset_profile", "none"))
    if onset_profile not in {"none", "chart_onsets_v1"}:
        raise ValueError(f"Unknown checkpoint onset_profile {onset_profile!r}.")
    state = payload.get("model_state_dict", {})
    onset_keys = {"onset_head.0.weight", "onset_head.0.bias", "onset_head.2.weight", "onset_head.2.bias"}
    present = {key for key in onset_keys if key in state or any(str(name).endswith("." + key) for name in state)}
    if onset_profile == "chart_onsets_v1":
        if present != onset_keys:
            raise ValueError("Checkpoint enables chart_onsets_v1 but its onset head weights are incomplete or missing.")
        contract = payload.get("data_contract", {}).get("onset_params")
        if not isinstance(contract, dict):
            raise ValueError("Checkpoint enables chart_onsets_v1 but has no onset_params data contract.")
        if contract.get("profile") != onset_profile or contract.get("target_version") != onset_profile:
            raise ValueError("Checkpoint onset target profile and target version disagree.")
        if contract.get("local_pooling_policy") != config.get("local_pooling_policy", "legacy_uniform"):
            raise ValueError("Checkpoint onset pooling policy disagrees with model_config.")
        try:
            sigma = float(contract.get("sigma_sec"))
            loss_weight = float(contract.get("loss_weight"))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Checkpoint onset sigma/loss weight is invalid.") from exc
        if not math.isfinite(sigma) or not math.isclose(sigma, 0.02, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("Checkpoint chart_onsets_v1 requires sigma_sec=0.02.")
        if not math.isfinite(loss_weight) or loss_weight < 0.0:
            raise ValueError("Checkpoint onset loss weight must be finite and nonnegative.")
        training = payload.get("onset_training")
        if (not isinstance(training, dict) or training.get("profile") != onset_profile
                or not isinstance(training.get("trained"), bool)):
            raise ValueError("Checkpoint onset_training metadata is missing or inconsistent.")
        try:
            steps = int(training.get("optimizer_steps", -1))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Checkpoint onset optimizer step count is invalid.") from exc
        if steps < 0 or training["trained"] != (steps > 0):
            raise ValueError("Checkpoint onset trained status disagrees with optimizer step count.")
    elif present:
        raise ValueError("Checkpoint contains onset head weights while onset_profile is disabled.")
    return payload


def load_event_model(path: str | Path, device: torch.device):
    checkpoint = load_event_checkpoint(path, device=device)
    model = EventConditionedTransformer(EventTransformerConfig(**checkpoint["model_config"])).to(device)
    # A checkpoint trained before the condition-token architecture has no
    # ``condition_type_embedding``; the compat loader switches the model to the
    # per-frame conditioning those weights were trained with instead of
    # failing, so an existing model keeps generating.
    compatibility = model.load_weights_compatible(checkpoint["model_state_dict"])
    model.planner_enabled = bool(compatibility.get("planner", False))
    if compatibility["legacy"]:
        print("checkpoint_condition_fusion=per_frame_bias（旧结构 checkpoint：条件作为逐帧常量注入；"
              "按旧结构生成，行为与训练时一致。结构升级需重新训练。）", flush=True)
    model.eval()
    return model, EventVocabulary.from_dict(checkpoint["vocabulary"]), checkpoint


def compare_event_contract(stored: dict[str, Any], current: dict[str, Any], *, mode: str) -> None:
    contract = stored.get("data_contract", {})
    stored_condition_version = contract.get("condition_target_version", LEGACY_CONDITION_TARGET_VERSION)
    current_condition_version = current.get("condition_target_version", LEGACY_CONDITION_TARGET_VERSION)
    if stored_condition_version != current_condition_version:
        allowed_migration = (
            mode in {"init", "finetune"}
            and stored_condition_version == LEGACY_CONDITION_TARGET_VERSION
            and current_condition_version == CURRENT_CONDITION_TARGET_VERSION
        )
        if not allowed_migration:
            raise ValueError(
                "事件模型数据合同 condition_target_version 不兼容："
                f"已有={stored_condition_version!r}，当前={current_condition_version!r}。"
                "仅允许通过 finetune 从 legacy_unshifted_v1 迁移到 chart_audio_clock_v2；resume 必须继承原版本。"
            )
    keys = ("target_schema", "feature_profile", "input_dim", "sample_rate", "hop_length", "segment_beats", "max_tokens", "vocabulary_signature", "condition_keys")
    for key in keys:
        if contract.get(key) != current.get(key):
            raise ValueError(f"事件模型数据合同 {key} 不兼容：已有={contract.get(key)!r}，当前={current.get(key)!r}。")
    # Family-specific settings (window context, margins) are optional so that
    # checkpoints written before a family existed still resume: the check only
    # applies when either side actually carries them.
    for key in OPTIONAL_CONTRACT_KEYS:
        if key in contract or key in current:
            before, after = contract.get(key), current.get(key)
            if key == "family_params" and mode in {"init", "finetune"}:
                # Resampling changes sample weights, not tensor shapes or the
                # temporal layout. Fine-tuning loads weights and creates a new
                # optimizer, so it may explicitly migrate a legacy OR sampler
                # or change either multiplier. All non-sampling keys stay strict.
                before = {name: value for name, value in (before or {}).items()
                          if name not in INIT_MIGRATABLE_FAMILY_KEYS}
                after = {name: value for name, value in (after or {}).items()
                         if name not in INIT_MIGRATABLE_FAMILY_KEYS}
            elif key == "family_params" and mode == "resume":
                before = _normalize_resume_sampling(before, side="checkpoint")
                after = _normalize_resume_sampling(after, side="current run")
            if key == "quality_params" and mode in {"init", "finetune"}:
                # A finetune is the supported way to change chart quality
                # labels or add corrected preference pairs. Ordinary resume
                # stays strict so optimizer state never crosses regimes.
                before, after = None, None
            if key == "onset_params" and mode in {"init", "finetune"}:
                # Explicit initialization may add or retune this auxiliary
                # objective; resume remains strict to preserve optimizer semantics.
                before, after = None, None
            if before != after:
                if key == "condition_params":
                    raise ValueError(
                        f"事件模型数据合同 condition_params 不兼容：已有={contract.get(key)!r}，当前={current.get(key)!r}。"
                        "其中 dropout 对应‘条件随机丢弃（CFG 锚）/ --condition-dropout’，"
                        "与模型内部 --dropout、上下文 --context-dropout 独立。"
                        "resume 和 finetune 都须继承这份 checkpoint 的条件合同；"
                        "请检查并回填 condition_mask_prob、condition_dropout 与 condition_default。"
                    )
                raise ValueError(
                    f"事件模型数据合同 {key} 不兼容：已有={contract.get(key)!r}，当前={current.get(key)!r}。"
                    "更改 Slide/Touch 采样倍率或迁移旧采样策略请用 finetune（--init-checkpoint）；"
                    "音频、词表、窗口布局和条件合同仍须兼容。"
                )
    if mode == "resume" and contract.get("dataset_signature") != current.get("dataset_signature"):
        raise ValueError("断点继续要求相同数据集内容；新数据集请选择‘已有模型加强训练’。")
    if mode == "resume" and not stored.get("optimizer_state_dict"):
        raise ValueError("Checkpoint lacks optimizer state; use finetuning instead.")


def atomic_save_checkpoint(payload: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(target)
