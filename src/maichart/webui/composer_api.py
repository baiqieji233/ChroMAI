"""WebUI adapter for the independent chart_composer_v1 workflow."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


FAMILY = "chart_composer_v1"
SCHEMA = "maichart-chart-composer-checkpoint-v1"


def _path(value: Any, label: str, *, exists: bool = False, directory: bool = False) -> Path:
    if not value or not str(value).strip():
        raise ValueError(f"缺少 {label} 路径")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        # Match server._resolve_path so a service started outside the repo still
        # resolves relative UI paths against the project root.
        path = Path(__file__).resolve().parents[3] / path
    path = path.resolve()
    if exists and not path.exists():
        raise ValueError(f"{label} 不存在: {path}")
    if directory and exists and not path.is_dir():
        raise ValueError(f"{label} 必须是目录: {path}")
    return path


def _int(payload: dict[str, Any], key: str, default: int, *, minimum: int | None = None) -> int:
    try:
        value = int(payload.get(key, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} 必须是整数") from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"{key} 不能小于 {minimum}")
    return value


def _float(payload: dict[str, Any], key: str, default: float, *, minimum: float | None = None) -> float:
    try:
        value = float(payload.get(key, default))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} 必须是数字") from exc
    if not math.isfinite(value) or (minimum is not None and value < minimum):
        raise ValueError(f"{key} 数值无效")
    return value


def _manifest(payload: dict[str, Any]) -> Path:
    value = payload.get("manifest")
    if isinstance(value, list):
        value = next((item for item in value if str(item).strip()), None)
    return _path(value, "manifest", exists=True)


def _device(value: Any) -> str:
    result = str(value or "auto").strip().lower()
    if result not in {"auto", "cpu", "cuda"}:
        raise ValueError("device 只能是 auto、cpu 或 cuda")
    return result


def _precision(value: Any) -> str:
    result = str(value or "auto").strip().lower()
    if result not in {"auto", "fp32", "fp16", "bf16"}:
        raise ValueError("precision 只能是 auto、fp32、fp16 或 bf16")
    return result


def _arg(args: list[str], name: str, value: Any) -> None:
    args.extend([name, str(value)])


def start_train(handler, payload: dict[str, Any], *, runtime: Any) -> None:
    requested_family = payload.get("model_family")
    if requested_family not in {None, FAMILY}:
        raise ValueError(f"Composer endpoint 只接受 model_family={FAMILY}")
    manifest = _manifest(payload)
    checkpoint_dir = _path(payload.get("checkpoint_dir"), "checkpoint-dir")
    cache_dir = _path(payload.get("cache_dir"), "cache-dir")
    mode = str(payload.get("mode", "fresh")).strip().lower()
    if mode not in {"fresh", "resume", "finetune"}:
        raise ValueError("训练模式只能是 fresh、resume 或 finetune")
    d_model = _int(payload, "d_model", 192, minimum=1)
    nhead = _int(payload, "nhead", 6, minimum=1)
    if d_model % nhead:
        raise ValueError("d_model 必须能被 nhead 整除")
    if mode != "fresh" and not (payload.get("resume") or payload.get("init_checkpoint") or payload.get("checkpoint")):
        raise ValueError("resume / finetune 模式必须提供 checkpoint")
    parent = payload.get("resume") or payload.get("init_checkpoint") or payload.get("checkpoint")
    parent_path = _path(parent, "checkpoint", exists=True) if mode != "fresh" else None

    args = ["-m", "maichart.composer.train"]
    _arg(args, "--manifest", manifest)
    _arg(args, "--checkpoint-dir", checkpoint_dir)
    _arg(args, "--cache-dir", cache_dir)
    for key, flag, default, minimum in (
        ("epochs", "--epochs", 10, 1), ("batch_size", "--batch-size", 4, 1),
        ("d_model", "--d-model", d_model, 1), ("layers", "--layers", 3, 1),
        ("nhead", "--nhead", nhead, 1), ("history_events", "--history-events", 128, 1),
        ("candidate_count", "--candidate-count", 64, 1), ("action_slots", "--action-slots", 8, 1),
        ("checkpoint_interval_steps", "--checkpoint-interval-steps", 500, 0),
        ("limit_charts", "--limit-charts", 0, 0), ("prepare_workers", "--prepare-workers", 4, 0),
    ):
        _arg(args, flag, _int(payload, key, default, minimum=minimum))
    _arg(args, "--lr", _float(payload, "lr", 0.0002, minimum=0.0))
    _arg(args, "--audio-cache-gb", _float(payload, "audio_cache_gb", 4.0, minimum=0.0))
    _arg(args, "--device", _device(payload.get("device")))
    _arg(args, "--precision", _precision(payload.get("precision")))
    _arg(args, "--seed", _int(payload, "seed", 0, minimum=0))
    if mode == "resume":
        _arg(args, "--resume", parent_path)
    elif mode == "finetune":
        _arg(args, "--init-checkpoint", parent_path)

    task = runtime._create_task("composer_train")
    task.result.update({"family": FAMILY, "schema": SCHEMA, "mode": mode,
                        "manifest": str(manifest), "checkpoint_dir": str(checkpoint_dir),
                        "epoch_target": _int(payload, "epochs", 10, minimum=1)})
    runtime._run_subprocess_task(task, args, expected_epoch=task.result["epoch_target"],
                                 result_paths=[checkpoint_dir / "composer_last.pt"])
    handler._json_response({"task_id": task.task_id, "family": FAMILY})


def start_generate(handler, payload: dict[str, Any], *, runtime: Any) -> None:
    requested_family = payload.get("model_family")
    if requested_family not in {None, FAMILY}:
        raise ValueError(f"Composer endpoint 只接受 model_family={FAMILY}")
    audio = _path(payload.get("audio"), "audio", exists=True)
    checkpoint = _path(payload.get("checkpoint"), "checkpoint", exists=True)
    output = _path(payload.get("output"), "output")
    output.parent.mkdir(parents=True, exist_ok=True)
    args = ["-m", "maichart.composer.generate"]
    for flag, value in (("--audio", audio), ("--checkpoint", checkpoint), ("--output", output)):
        _arg(args, flag, value)
    if payload.get("bpm") not in (None, ""):
        _arg(args, "--bpm", _float(payload, "bpm", 120.0, minimum=0.0))
    for key, flag, default, minimum in (("first", "--first", 0.0, None), ("level", "--level", 13.0, 0.0),
                                        ("temperature", "--temperature", 0.8, 0.0), ("seed", "--seed", 0, 0),
                                        ("max_events", "--max-events", 0, 0)):
        value = _float(payload, key, default, minimum=minimum) if isinstance(default, float) else _int(payload, key, default, minimum=minimum)
        _arg(args, flag, value)
    _arg(args, "--device", _device(payload.get("device")))
    condition = payload.get("condition_file")
    if condition:
        _arg(args, "--condition-file", _path(condition, "condition-file", exists=True))
    sidecar = output.with_suffix(".composer.json")
    task = runtime._create_task("composer_generate")
    task.result.update({"family": FAMILY, "checkpoint": str(checkpoint), "output": str(output)})
    runtime._run_subprocess_task(task, args, expected_epoch=None, result_paths=[output, sidecar])
    handler._json_response({"task_id": task.task_id, "family": FAMILY})


def inspect(handler, payload: dict[str, Any]) -> None:
    checkpoint = _path(payload.get("checkpoint"), "checkpoint", exists=True)
    try:
        import torch
        from maichart.composer.checkpoint import inspect_payload
        data = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(data, dict):
            raise ValueError("checkpoint payload 必须是对象")
        result = inspect_payload(data)
        result.update({"path": str(checkpoint), "family": result.get("family") or data.get("family"),
                       "schema": result.get("schema") or data.get("schema")})
    except Exception as exc:  # inspection must return a useful UI error
        result = {"valid": False, "path": str(checkpoint), "family": None,
                  "schema": None, "error": str(exc)}
    result["can_resume"] = bool(result.get("valid"))
    result["can_finetune"] = bool(result.get("valid"))
    result["can_generate"] = bool(result.get("valid"))
    handler._json_response(result)
