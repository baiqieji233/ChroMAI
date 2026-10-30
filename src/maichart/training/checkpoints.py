"""Checkpoint contracts shared by training, inference, and the WebUI."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

CHECKPOINT_SCHEMA = "maichart-conditioned-generator-checkpoint-v2"
LEGACY_SCHEMA = "maichart-conditioned-generator-v1"


def load_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a v1/v2 checkpoint or a legacy raw state dict.

    The return value always contains ``model_state_dict`` and a small amount of
    classification metadata. Keeping this normalization in one place avoids
    subtly different loading behavior between the trainer and generator.
    """

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"Checkpoint 文件不存在: {source}")
    try:
        payload = torch.load(source, map_location=map_location, weights_only=False)
    except Exception as exc:  # noqa: BLE001 - convert torch's low-level error
        raise ValueError(f"无法读取 checkpoint: {source}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"Checkpoint 必须是字典或 state_dict: {source}")
    payload = dict(payload)

    if _is_state_dict(payload.get("model_state_dict")):
        payload["model_state_dict"] = dict(payload["model_state_dict"])
        payload.setdefault("checkpoint_kind", "training_checkpoint")
        return payload
    if _is_state_dict(payload.get("model")):
        payload["model_state_dict"] = dict(payload["model"])
        payload.setdefault("checkpoint_kind", "training_checkpoint")
        return payload
    if _is_state_dict(payload):
        return {
            "schema": "legacy-state-dict",
            "checkpoint_kind": "legacy_weights",
            "model_state_dict": dict(payload),
            "legacy": True,
        }
    raise ValueError(
        f"无法识别 checkpoint 格式: {source}。需要 model_state_dict、model，或纯 PyTorch state_dict。"
    )


def checkpoint_kind(payload: Mapping[str, Any]) -> str:
    if payload.get("checkpoint_kind") == "legacy_weights" or payload.get("legacy"):
        return "legacy_weights"
    if payload.get("schema") == CHECKPOINT_SCHEMA:
        return "training_checkpoint_v2"
    return "training_checkpoint_v1"


def checkpoint_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    data = payload.get("data_contract")
    return dict(data) if isinstance(data, Mapping) else {}


def checkpoint_model_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    data = payload.get("model_config")
    return dict(data) if isinstance(data, Mapping) else {}


def compare_checkpoint_contract(
    payload: Mapping[str, Any],
    *,
    model_config: Mapping[str, Any] | None = None,
    data_contract: Mapping[str, Any] | None = None,
    mode: str = "init",
) -> tuple[list[str], list[str]]:
    """Return human-readable incompatibility errors and non-fatal warnings."""

    errors: list[str] = []
    warnings: list[str] = []
    stored_model = checkpoint_model_config(payload)
    stored_data = checkpoint_contract(payload)
    if not stored_model:
        warnings.append("这是旧版纯权重 checkpoint，没有保存模型结构合同；只能在结构完全相同的前提下尝试加载。")
    if not stored_data:
        warnings.append("checkpoint 没有保存数据合同，无法自动确认 division、音频特征和条件字段完全一致。")

    if model_config and stored_model:
        keys = ("input_dim", "condition_dim", "num_note_types", "d_model", "nhead", "num_layers", "dim_feedforward", "max_len")
        for key in keys:
            if key in stored_model and key in model_config and stored_model[key] != model_config[key]:
                errors.append(f"模型参数 {key} 不兼容：checkpoint={stored_model[key]}，当前={model_config[key]}。")

    if data_contract and stored_data:
        for key in (
            "division",
            "sample_rate",
            "feature_profile",
            "audio_feature_dim",
            "audio_feature_keys",
            "condition_dim",
            "condition_keys",
            "note_type_vocab",
            "target_schema",
        ):
            if key in stored_data and key in data_contract and stored_data[key] != data_contract[key]:
                errors.append(f"数据合同 {key} 不兼容：checkpoint={stored_data[key]}，当前={data_contract[key]}。")

    if mode == "resume" and checkpoint_kind(payload) == "legacy_weights":
        errors.append("这个 checkpoint 只有模型权重，没有优化器、学习率和 epoch 状态，不能用于断点继续；请选择‘已有模型加强训练’。")
    if mode == "resume":
        for key in ("optimizer_state_dict", "epoch"):
            if key not in payload:
                errors.append(f"断点缺少 {key}，不能恢复完整训练状态。")
        if "scheduler_state_dict" not in payload:
            warnings.append("旧版 checkpoint 没有学习率调度器状态，将从当前学习率重新创建调度器。")
        if "amp_scaler_state_dict" not in payload:
            warnings.append("旧版 checkpoint 没有 AMP scaler 状态，将重新创建 scaler。")
    return errors, warnings


def _is_state_dict(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and bool(value)
        and any(torch.is_tensor(item) for item in value.values())
        and all(
            torch.is_tensor(item)
            or (isinstance(key, str) and (key == "_extra_state" or key.endswith("._extra_state")))
            for key, item in value.items()
        )
    )
