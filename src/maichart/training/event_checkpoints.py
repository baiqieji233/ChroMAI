"""Strict checkpoint contracts for the event generator."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from maichart.generation.event_model import EventConditionedTransformer, EventTransformerConfig, EventVocabulary
from maichart.training.checkpoints import load_checkpoint

EVENT_CHECKPOINT_SCHEMA = "maichart-event-generator-checkpoint-v1"

# Contract keys that only exist for some families.  They are compared when
# either side carries them, and ignored otherwise so older checkpoints keep
# working.
OPTIONAL_CONTRACT_KEYS = ("family_params", "condition_params", "quality_params", "planner_params")
SAMPLING_CONTRACT_KEYS = frozenset({
    "slide_oversample", "touch_oversample", "oversample_policy",
    "fine_rhythm_oversample", "fine_rhythm_ratio_threshold",
})


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
                          if name not in SAMPLING_CONTRACT_KEYS}
                after = {name: value for name, value in (after or {}).items()
                         if name not in SAMPLING_CONTRACT_KEYS}
            if key == "quality_params" and mode in {"init", "finetune"}:
                # A finetune is the supported way to change chart quality
                # labels or add corrected preference pairs. Ordinary resume
                # stays strict so optimizer state never crosses regimes.
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
