"""Dependency-light localhost WebUI for preprocessing, training, and generation."""

from __future__ import annotations

import argparse
import json
import math
import mimetypes
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from email.parser import BytesParser
from email.policy import default as email_default
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Sequence
from urllib.parse import parse_qs, quote, urlparse

from maichart.generation.conditions import CONDITION_KEYS, GENERATION_DEFAULTS, GENERATION_SAMPLING_DEFAULTS
from maichart.generation.music_plan import MUSIC_STRUCTURE_V3_PROFILE, is_music_structure
from maichart.generation.model_compatibility import checkpoint_capabilities
from maichart.generation.miacode_native import engine_status
from maichart.generation.timeline import serialize_tempo_map, tempo_map_fingerprint
from maichart.generation.model import AUDIO_FEATURE_KEYS
from maichart.training.checkpoints import (
    checkpoint_contract,
    checkpoint_kind,
    checkpoint_model_config,
    compare_checkpoint_contract,
    load_checkpoint,
)
from maichart.training.frame_label_codec import DEFAULT_NOTE_TYPES
from maichart.training.dataset_catalog import assert_no_validation_overlap
from maichart.webui.dataset_api import DatasetAPI
from maichart.webui.quality_api import (
    QualityAPI,
    quality_audio,
    quality_catalog,
    quality_detail,
    quality_preview,
    quality_save,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STATIC_ROOT = Path(__file__).resolve().parent / "static"
WEBUI_ROOT = PROJECT_ROOT / "webui_workspace"
UPLOAD_ROOT = WEBUI_ROOT / "uploads"
DATASET_ROOT = WEBUI_ROOT / "datasets"
CONDITION_ROOT = WEBUI_ROOT / "conditions"
GENERATED_ROOT = WEBUI_ROOT / "generated"
INTENSITY_ROOT = WEBUI_ROOT / "intensity_curves"
# Run plans for the one-click preprocessing+training path.
PLAN_ROOT = WEBUI_ROOT / "plans"
# Every task's full log is appended here, one file per run, plus a sidecar JSON
# with status/argv/exit code.  The in-memory ring keeps only the last 300 lines
# for the live view; disk keeps everything, so a finished (or crashed) run can
# still be read after the interface is restarted.
LOG_ROOT = WEBUI_ROOT / "logs"
# Keep the log directory bounded: oldest runs are pruned first, whole runs at a
# time so a log never loses its sidecar.
LOG_KEEP_FILES = 400
LOG_KEEP_BYTES = 2 * 1024 * 1024 * 1024
# Uploads are streamed straight to disk, so this limit only guards against
# runaway requests rather than memory; it no longer forces users to split a
# large training ZIP into pieces.
MAX_UPLOAD_BYTES = 20 * 1024 * 1024 * 1024
EVENT_CHECKPOINT_SCHEMA = "maichart-event-generator-checkpoint-v1"
EVENT_MODEL_FAMILY = "event_v1"
EVENT_CTX_MODEL_FAMILY = "event_ctx_v1"
EVENT_FAMILIES = frozenset({EVENT_MODEL_FAMILY, EVENT_CTX_MODEL_FAMILY})
ONSET_PROFILES = frozenset({"auto", "none", "chart_onsets_v1"})
LEGACY_MODEL_FAMILY = "legacy_frame_v2"
for _directory in (UPLOAD_ROOT, DATASET_ROOT, CONDITION_ROOT, GENERATED_ROOT):
    _directory.mkdir(parents=True, exist_ok=True)


@dataclass
class TaskState:
    task_id: str
    kind: str
    status: str = "queued"
    progress: float = 0.0
    message: str = "等待启动"
    logs: list[str] = field(default_factory=list)
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    process: subprocess.Popen[str] | None = field(default=None, repr=False)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    # Persistence: the full log lives on disk, the in-memory list stays a ring.
    argv: list[str] = field(default_factory=list)
    log_path: Path | None = None
    started_at: str = ""
    ended_at: str = ""
    exit_code: int | None = None
    _handle: Any = field(default=None, repr=False, compare=False)

    def log(self, text: str) -> None:
        line = text.rstrip("\r\n")
        if not line:
            return
        # Parse the raw stream first: progress detection keys on the original
        # format, then the stored line gains a wall-clock prefix so the UI log
        # doubles as a timeline.
        stamp = time.strftime("%H:%M:%S")
        with self.lock:
            self.message = line[-240:]
            epoch_match = re.search(r"epoch=(\d+)", line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                self.result["last_epoch"] = current_epoch
                target_epoch = self.result.get("epoch_target")
                if target_epoch:
                    self.result["epoch_progress"] = min(1.0, current_epoch / int(target_epoch))
            self.logs.append(f"[{stamp}] {line}")
            # The in-memory ring drops old lines; the file keeps them all.
            self.logs = self.logs[-300:]
            handle = self._handle
            if handle is not None and not getattr(handle, "closed", False):
                try:
                    handle.write(f"[{stamp}] {line}\n")
                except (OSError, ValueError):
                    self._handle = None  # disk problem: keep the run, lose the file

    def to_dict(self) -> dict[str, Any]:
        with self.lock:
            return {
                "task_id": self.task_id,
                "kind": self.kind,
                "status": self.status,
                "progress": self.progress,
                "message": self.message,
                "logs": list(self.logs),
                "result": dict(self.result),
                "error": self.error,
            }


TASKS: dict[str, TaskState] = {}
TASKS_LOCK = threading.Lock()
_QUALITY_API = QualityAPI()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args(argv)
    server = ThreadingHTTPServer((args.host, args.port), WebUIHandler)
    print(f"maichart WebUI: http://{args.host}:{args.port}", flush=True)
    print(f"workspace: {WEBUI_ROOT}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


class WebUIHandler(BaseHTTPRequestHandler):
    server_version = "maichart-webui/0.1"

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        parsed = urlparse(self.path)
        if parsed.path == "/":
            return self._serve_static("index.html")
        if parsed.path in {"/app.js", "/styles.css"}:
            return self._serve_static(parsed.path.lstrip("/"))
        if parsed.path == "/api/status":
            return self._json_response(
                {
                    "workspace": str(WEBUI_ROOT),
                    "project_root": str(PROJECT_ROOT),
                    "training_capabilities": {"independent_touch_oversample": True,
                                              "fine_rhythm_oversample": True,
                                              "quality_training_capability": True,
                                              "local_onset_training": True,
                                              "event_learning_rate_control": True,
                                              "multi_dataset_training": True},
                    "intensity_curve_capability": True,
                    "music_structure_capability": True,
                    "density_feedback_capability": True,
                    "playability_engine": engine_status(),
                    "playability_policy_version": 2,
                    "tasks": [_task.to_dict() for _task in _all_tasks()],
                    "datasets": _uploaded_datasets(),
                    "extractions": {key: dict(value) for key, value in _EXTRACTIONS.items()},
                }
            )
        if parsed.path == "/api/logs":
            # Persisted run logs survive an interface restart and are never
            # truncated to the 300-line live ring.
            return self._json_response({
                "logs": _persisted_logs(limit=int(parse_qs(parsed.query).get("limit", ["100"])[0] or 100)),
                "root": str(LOG_ROOT),
            })
        if parsed.path == "/api/task":
            task_id = parse_qs(parsed.query).get("id", [""])[0]
            task = _get_task(task_id)
            if task is None:
                return self._json_response({"error": "任务不存在"}, HTTPStatus.NOT_FOUND)
            return self._json_response(task.to_dict())
        if parsed.path == "/api/download":
            return self._download(parse_qs(parsed.query).get("path", [""])[0])
        if parsed.path == "/api/intensity/audio":
            return self._intensity_audio(parse_qs(parsed.query).get("id", [""])[0])
        if parsed.path == "/api/quality/audio":
            return self._quality_audio(parse_qs(parsed.query), send_body=True)
        self._json_response({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/upload":
                return self._upload()
            if parsed.path == "/api/generate":
                return self._start_generate(self._json_body())
            if parsed.path == "/api/intensity/analyze":
                return self._start_intensity_analysis(self._json_body())
            if parsed.path.startswith("/api/quality/"):
                status, result = _QUALITY_API.post(parsed.path, self._json_body())
                return self._json_response(result, HTTPStatus(status))
            if parsed.path.startswith("/api/datasets/"):
                status, result = _dataset_api().post(parsed.path, self._json_body())
                return self._json_response(result, HTTPStatus(status))
            if parsed.path == "/api/preprocess":
                return self._start_preprocess(self._json_body())
            if parsed.path == "/api/train":
                return self._start_train(self._json_body())
            if parsed.path == "/api/gold/build":
                return self._start_gold_build(self._json_body())
            if parsed.path == "/api/gold/designers":
                return self._gold_designers(self._json_body())
            if parsed.path == "/api/gold/scan":
                return self._gold_scan(self._json_body())
            if parsed.path == "/api/acceptance/evaluate":
                return self._start_acceptance(self._json_body())
            if parsed.path == "/api/checkpoint/inspect":
                return self._inspect_checkpoint(self._json_body())
            if parsed.path == "/api/inspect-manifest":
                return self._inspect_manifest(self._json_body())
            if parsed.path == "/api/cancel":
                return self._cancel_task(self._json_body())
            self._json_response({"error": "Not found"}, HTTPStatus.NOT_FOUND)
        except (ValueError, OSError, KeyError) as exc:
            self._json_response({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # noqa: BLE001 - return a usable local error
            self._json_response({"error": f"服务器错误: {exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def _quality_audio(self, query: dict[str, list[str]], *, send_body: bool) -> None:
        status, headers, path, byte_range = quality_audio(
            query, method="GET" if send_body else "HEAD", range_header=self.headers.get("Range"),
        )
        if path is None or byte_range is None:
            return self._json_response({"error": "谱面音频不可用"}, HTTPStatus(status))
        start, end = byte_range
        self.send_response(status)
        for key, value in headers.items():
            self.send_header(key, value)
        self.end_headers()
        if not send_body:
            return
        try:
            with path.open("rb") as handle:
                handle.seek(start)
                remaining = end - start + 1
                while remaining:
                    block = handle.read(min(1024 * 1024, remaining))
                    if not block:
                        break
                    self.wfile.write(block)
                    remaining -= len(block)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def _start_intensity_analysis(self, payload: dict[str, Any]) -> None:
        audio = _required_path(payload.get("audio"), "分析音频", must_exist=True)
        source = str(payload.get("curve_source", "audio"))
        if source not in {"auto", "audio", "model"}:
            raise ValueError("曲线来源只能是 auto、audio 或 model")
        checkpoint_path = (_required_path(payload["checkpoint"], "曲线规划 checkpoint", must_exist=True)
                           if source != "audio" and payload.get("checkpoint") else None)
        if source == "model" and checkpoint_path is None:
            raise ValueError("模型预测曲线需要选择 music_structure_v2/v3 checkpoint")
        resolution = float(payload.get("resolution_sec", 2.0))
        context = float(payload.get("local_context_sec", 12.0))
        first = float(payload.get("first", 0.0))
        tempo_mode = str(payload.get("tempo_mode", "constant")).strip().lower()
        explicit_tempo_map = (serialize_tempo_map(payload["tempo_map"])
                              if payload.get("tempo_map") is not None else None)
        tempo_map_hash = tempo_map_fingerprint(explicit_tempo_map)
        if not math.isfinite(resolution) or not 0.5 <= resolution <= 16:
            raise ValueError("曲线时间间隔必须在 0.5–16 秒之间")
        if not math.isfinite(context) or not 2 <= context <= 120:
            raise ValueError("局部比较范围必须在 2–120 秒之间")
        if not math.isfinite(first):
            raise ValueError("首拍偏移必须是有限值")
        if tempo_mode not in {"auto", "constant"}:
            raise ValueError("tempo-mode 只能是 auto 或 constant")
        if payload.get("bpm") not in {None, ""}:
            supplied_bpm = float(payload["bpm"])
            if not math.isfinite(supplied_bpm) or supplied_bpm <= 0:
                raise ValueError("BPM 必须是正的有限值")
        task = _create_task("intensity")
        task.result["audio"] = str(audio)
        task.result["audio_preview_url"] = f"/api/intensity/audio?id={task.task_id}"

        def worker() -> None:
            _open_task_log(task, ["intensity_analyze", str(audio),
                                  f"resolution_sec={resolution}", f"local_context_sec={context}"])
            task.status = "running"
            task.message = "正在分析节奏、频谱、和声变化和局部段落张力"
            try:
                from maichart.generation.music_intensity import generate_intensity_profile
                selected_source = source
                if selected_source == "auto":
                    profile_name = (_load_checkpoint_payload(checkpoint_path).get("model_config", {}).get("planner_profile")
                                    if checkpoint_path else None)
                    selected_source = ("model" if is_music_structure(str(profile_name or ""))
                                       else "audio")
                if selected_source == "model":
                    from maichart.generation.learned_curve import generate_learned_profile
                    from maichart.generation.event_cli import resolve_generation_condition
                    checkpoint_payload = _load_checkpoint_payload(checkpoint_path)
                    condition, condition_source = resolve_generation_condition(
                        level=float(payload.get("level", GENERATION_DEFAULTS["level"])),
                        condition_data=payload.get("condition"),
                        condition_profile=_resolve_condition_profile(payload),
                        contract_params=(checkpoint_payload.get("data_contract", {}).get("condition_params") or {}),
                    )
                    profile = generate_learned_profile(
                        audio, checkpoint_path, resolution_sec=resolution, local_context_sec=context,
                        condition=condition, bpm=payload.get("bpm"),
                        first=first, tempo_mode=tempo_mode, tempo_map=explicit_tempo_map,
                        device="cpu")
                    task.result["condition_source"] = condition_source
                else:
                    profile = generate_intensity_profile(audio, resolution_sec=resolution,
                                                         local_context_sec=context)
                profile["explicit_tempo_map"] = explicit_tempo_map
                profile["tempo_map_sha256"] = tempo_map_hash
                profile["tempo_mode"] = tempo_mode
                profile["first"] = first
                profile["bpm_input"] = (float(payload["bpm"])
                                        if payload.get("bpm") not in {None, ""} else None)
                if selected_source != "model":
                    profile["bpm"] = (float(explicit_tempo_map[0]["bpm"]) if explicit_tempo_map
                                       else float(payload["bpm"]) if payload.get("bpm") not in {None, ""}
                                       else None)
                INTENSITY_ROOT.mkdir(parents=True, exist_ok=True)
                output = INTENSITY_ROOT / f"{task.task_id}_automatic.json"
                output.write_text(json.dumps(profile, ensure_ascii=False), encoding="utf-8")
                with task.lock:
                    task.result["intensity_profile"] = profile
                    task.result["curve_source"] = selected_source
                    task.result["intensity_profile_path"] = str(output)
                    task.status = "completed"
                    task.progress = 1.0
                    task.message = "模型负荷曲线已生成，可以拖动编辑" if selected_source == "model" else "音频参考曲线已生成，可以拖动编辑"
                task.log(f"intensity_profile method={profile.get('method')} "
                         f"duration_sec={profile.get('duration_sec')} points={len(profile.get('points', []))}")
                _close_task_log(task, status="completed", exit_code=0)
            except Exception as exc:  # task errors are visible to its separate poller
                task.status = "failed"
                task.error = str(exc)
                task.message = "张力分析失败"
                _close_task_log(task, status="failed", error=str(exc))

        threading.Thread(target=worker, name=f"intensity-{task.task_id}", daemon=True).start()
        self._json_response({"task_id": task.task_id})

    def _intensity_audio(self, task_id: str, *, send_body: bool = True) -> None:
        # Playback is limited to the audio explicitly selected for an analysis
        # task. This also supports external D: songs without opening arbitrary
        # filesystem downloads through the project-only download endpoint.
        task = _get_task(task_id)
        if task is None or task.kind != "intensity":
            return self._json_response({"error": "音频分析任务不存在"}, HTTPStatus.NOT_FOUND)
        path = Path(str(task.result.get("audio", "")))
        if not path.is_file():
            return self._json_response({"error": "分析音频已不存在"}, HTTPStatus.NOT_FOUND)
        size = path.stat().st_size
        start, end = 0, size - 1
        range_header = self.headers.get("Range")
        if range_header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
            if not match or not any(match.groups()) or not size:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if match.group(1):
                start = int(match.group(1))
                end = min(size - 1, int(match.group(2))) if match.group(2) else size - 1
            else:
                start = max(0, size - int(match.group(2)))
            if start >= size or end < start:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
        length = max(0, end - start + 1)
        self.send_response(HTTPStatus.PARTIAL_CONTENT if range_header else HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if range_header:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if send_body:
            with path.open("rb") as handle:
                handle.seek(start)
                remaining = length
                try:
                    while remaining:
                        block = handle.read(min(1024 * 1024, remaining))
                        if not block:
                            break
                        self.wfile.write(block)
                        remaining -= len(block)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass  # seeking in the player intentionally aborts old streams

    def do_HEAD(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/intensity/audio":
            return self._intensity_audio(parse_qs(parsed.query).get("id", [""])[0], send_body=False)
        if parsed.path == "/api/quality/audio":
            return self._quality_audio(parse_qs(parsed.query), send_body=False)
        self.send_response(HTTPStatus.NOT_FOUND)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _start_generate(self, payload: dict[str, Any]) -> None:
        checkpoint = _required_path(payload.get("checkpoint"), "checkpoint", must_exist=True)
        checkpoint_payload = _load_checkpoint_payload(checkpoint)
        model_family = _checkpoint_model_family(checkpoint_payload)
        capabilities = checkpoint_capabilities(checkpoint_payload)
        if model_family == "unknown":
            raise ValueError(
                f"不支持的 checkpoint schema: {checkpoint_payload.get('schema')!r}。"
                f"只接受 {EVENT_CHECKPOINT_SCHEMA} 或已知旧固定帧 checkpoint。"
            )
        if capabilities["errors"]:
            raise ValueError("checkpoint capability contract invalid: " + "; ".join(capabilities["errors"]))
        explicit_tempo_map = (serialize_tempo_map(payload["tempo_map"])
                              if payload.get("tempo_map") is not None else None)
        tempo_map_hash = tempo_map_fingerprint(explicit_tempo_map)
        audio_features = payload.get("audio_features")
        audio = payload.get("audio")
        if audio_features and model_family == LEGACY_MODEL_FAMILY:
            audio_features_path = _required_path(audio_features, "audio features", must_exist=True)
        else:
            audio_features_path = None
        audio_path = _required_path(audio, "audio", must_exist=True) if audio else None
        if _is_event_family(model_family) and audio_path is None:
            raise ValueError("事件模型生成必须提供音频路径；它不读取旧版 audio_features 输入")
        if not _is_event_family(model_family) and audio_path is None and audio_features_path is None:
            raise ValueError("请导入音频或预计算的 audio_features.json")
        curve = None
        if payload.get("intensity_curve_enabled", False):
            if not _is_event_family(model_family):
                raise ValueError("谱面张力曲线目前支持 event_v1 / event_ctx_v1")
            from maichart.generation.music_intensity import audio_sha256, validate_intensity_profile

            curve = validate_intensity_profile(payload.get("intensity_curve"))
            if curve.get("audio_sha256") != audio_sha256(audio_path):
                raise ValueError("张力曲线与当前音频不匹配：更换歌曲后请重新生成曲线")
            if curve.get("method") in {"learned_music_structure_v2", "learned_music_structure_v3"} and curve.get("checkpoint"):
                if curve.get("tempo_map_sha256") != tempo_map_hash:
                    raise ValueError("曲线预测使用的 tempo map 已变化，请用当前节拍设置重新生成曲线")
                curve_checkpoint = _resolve_path(curve["checkpoint"])
                if curve_checkpoint != checkpoint:
                    raise ValueError("曲线预测使用了另一个 checkpoint 路径，请重新生成曲线")
                from maichart.generation.event_cli import _file_identity
                identity = _file_identity(checkpoint)
                saved_identity = curve.get("checkpoint_identity")
                if not isinstance(saved_identity, dict):
                    raise ValueError("曲线缺少可验证的 checkpoint 身份信息，请重新生成曲线")
                saved_hash = saved_identity.get("sha256")
                same_size = saved_identity.get("size") == identity["size"]
                same_hash = isinstance(saved_hash, str) and saved_hash == identity["sha256"]
                old_browser_mtime_match = (
                    saved_hash is None and isinstance(saved_identity.get("mtime_ns"), (int, float))
                    and float(saved_identity["mtime_ns"]) == float(identity["mtime_ns"]))
                if not same_size or not (same_hash or old_browser_mtime_match):
                    raise ValueError("曲线预测所用 checkpoint 文件已被覆盖或身份不匹配，请用当前 checkpoint 重新生成曲线")
                from maichart.generation.event_cli import resolve_generation_condition
                current_condition, _ = resolve_generation_condition(
                    level=float(payload.get("level", GENERATION_DEFAULTS["level"])),
                    condition_data=payload.get("condition"),
                    condition_profile=_resolve_condition_profile(payload),
                    contract_params=(checkpoint_payload.get("data_contract", {}).get("condition_params") or {}),
                )
                context_differences = _learned_curve_context_differences(
                    curve, current_condition, payload, tempo_map_hash)
                if context_differences:
                    detail = "；".join(context_differences[:12])
                    if len(context_differences) > 12:
                        detail += f"；另有 {len(context_differences) - 12} 项"
                    raise ValueError(
                        "曲线预测所用等级、风格或节拍设置已变化，请重新生成曲线后继续编辑"
                        f"（具体差异：{detail}）"
                    )
        tempo_mode = str(payload.get("tempo_mode", "constant")).strip().lower()
        intensity_mode = str(payload.get("intensity_mode", GENERATION_SAMPLING_DEFAULTS["intensity_mode"]))
        if _is_event_family(model_family):
            if tempo_mode not in {"auto", "constant"}:
                raise ValueError("事件模型的 tempo-mode 只能是 auto 或 constant")
            if intensity_mode not in {"auto", "model", "loudness", "multifeature"}:
                raise ValueError("强度跟随模式只能是 auto、model、multifeature 或 loudness")
        curve_strength = float(payload.get("intensity_curve_strength", 0.65))
        curve_mode = str(payload.get("intensity_curve_mode", "follow"))
        fine_note_strength = float(payload.get("fine_note_strength", 0.5))
        if not math.isfinite(fine_note_strength) or not 0.0 <= fine_note_strength <= 1.0:
            raise ValueError("细碎音符强度必须是 0–1 之间的有限值")
        local_onset_bias_strength = float(payload.get("local_onset_bias_strength", 0.0))
        if not math.isfinite(local_onset_bias_strength) or not 0.0 <= local_onset_bias_strength <= 1.0:
            raise ValueError("局部采音对齐指导强度必须是 0–1 之间的有限值")
        local_onset_guidance = (_is_event_family(model_family)
                                and bool(capabilities.get("local_onset_guidance", False)))
        if local_onset_bias_strength > 0 and not local_onset_guidance:
            raise ValueError(
                "当前 checkpoint 未训练可用的局部采音对齐辅助头；请检查 checkpoint 能力，"
                "并使用训练过 chart_onsets_v1 的 event checkpoint。"
            )
        conflict_enabled = bool(payload.get("conflict_detection_enabled", False))
        conflict_repair_mode = str(payload.get("conflict_repair_mode", "none"))
        conflict_retries = int(payload.get("conflict_retries", 0))
        conflict_kind_modes = payload.get("conflict_kind_modes")
        tap_on_slide_threshold_ms = float(payload.get("tap_on_slide_threshold_ms", 200.0))
        conflict_backtrack_windows = int(payload.get("conflict_backtrack_windows", 0))
        conflict_backtrack_budget = int(payload.get("conflict_backtrack_budget", 4))
        sustain_repair_mode = str(payload.get("sustain_repair_mode", "off"))
        note_repair_mode = str(payload.get("note_repair_mode", "delete")).strip().lower()
        if note_repair_mode not in {"delete", "edit", "edit_then_delete"}:
            raise ValueError("单 note 修复方式只能是 delete、edit 或 edit_then_delete")
        if conflict_repair_mode not in {"none", "local", "reject"}:
            raise ValueError("冲突处理方式只能是 none、local 或 reject")
        if not 0 <= conflict_retries <= 5:
            raise ValueError("冲突重试次数必须是 0–5")
        runtime_reference_mode = str(payload.get("runtime_reference_mode", "off")).strip().lower()
        if runtime_reference_mode not in {"off", "report", "reject"}:
            raise ValueError("运行时参考检查模式只能是 off、report 或 reject")
        # Old browser settings migrate to the embedded judge. Never execute a
        # persisted MiaCode.exe path, even if the previous UI supplied one.
        if runtime_reference_mode != "off" and not conflict_enabled:
            conflict_enabled = True
            conflict_repair_mode = "reject" if runtime_reference_mode == "reject" else "none"
        if conflict_enabled and not _is_event_family(model_family):
            raise ValueError("内嵌无理检测目前支持 event_v1/event_ctx_v1；旧固定帧模型请关闭此选项")
        if conflict_enabled and _is_event_family(model_family):
            from maichart.generation.miacode_native import ensure_available
            ensure_available()
        if not conflict_enabled and conflict_repair_mode != "none":
            raise ValueError("选择局部修复或拒绝生成时，请同时启用无理检测")
        if conflict_repair_mode == "none":
            conflict_retries = 0
        # Fine-grained note repair is meaningful only for local regeneration;
        # reject/detect-only keep the legacy delete policy for compatibility.
        effective_note_repair_mode = note_repair_mode if conflict_repair_mode == "local" else "delete"
        from maichart.generation.post_process import PostProcessConfig
        post_config_kwargs = dict(
            enable_conflict_detection=conflict_enabled,
            conflict_repair_mode=conflict_repair_mode,
            conflict_retries=conflict_retries,
            conflict_engine="miacode" if conflict_enabled else "static",
            conflict_kind_modes=conflict_kind_modes,
            tap_on_slide_threshold_ms=tap_on_slide_threshold_ms,
            conflict_backtrack_windows=conflict_backtrack_windows,
            conflict_backtrack_budget=conflict_backtrack_budget,
            sustain_repair_mode=sustain_repair_mode,
        )
        # Keep this bridge loadable while an older checkout is being upgraded;
        # the current PostProcessConfig includes this field and validates it.
        post_config_kwargs["note_repair_mode"] = effective_note_repair_mode
        PostProcessConfig(**post_config_kwargs)
        if not math.isfinite(curve_strength) or not 0.0 <= curve_strength <= 1.0:
            raise ValueError("曲线控制力度必须在 0–1 之间")
        if curve_mode not in {"follow", "hard", "soft", "strict"}:
            raise ValueError("曲线控制方式只能是 follow、hard、soft 或 strict")
        task = _create_task("generate")
        task.result.update({"model_family": model_family, "checkpoint_schema": checkpoint_payload.get("schema")})
        seed, seed_randomized = _resolve_seed(payload)
        task.result.update({"seed": seed, "seed_randomized": seed_randomized})
        task.result.update({"conflict_engine": "miacode" if conflict_enabled else "disabled",
                            "runtime_reference_migrated": runtime_reference_mode != "off"})
        condition = payload.get("condition") or {"level": float(payload.get("level", GENERATION_DEFAULTS["level"]))}
        condition_path = CONDITION_ROOT / f"{task.task_id}.json"
        condition_path.write_text(json.dumps(condition, ensure_ascii=False, indent=2), encoding="utf-8")
        curve_path = None
        if curve is not None:
            INTENSITY_ROOT.mkdir(parents=True, exist_ok=True)
            curve_path = INTENSITY_ROOT / f"{task.task_id}_edited.json"
            stored_curve = {key: curve[key] for key in
                            ("schema", "audio_sha256", "duration_sec", "points", "method", "edited",
                             "checkpoint", "checkpoint_identity", "condition", "bpm", "bpm_input", "first", "tempo_mode",
                             "rest_regions", "tempo_changes", "explicit_tempo_map", "tempo_map_sha256")
                            if key in curve}
            curve_path.write_text(json.dumps(stored_curve, ensure_ascii=False), encoding="utf-8")
            task.result["intensity_curve_path"] = str(curve_path)
        output = payload.get("output") or str(GENERATED_ROOT / f"{task.task_id}_maidata.txt")
        output_path = _resolve_path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        candidate_count = max(1, int(payload.get("candidates", 1)))
        output_ir = output_path.with_suffix(".chart.json")
        result_paths = _generation_result_paths(output_path, output_ir, candidate_count, model_family)
        if _is_event_family(model_family):
            args = [
                "-m", ("maichart.generation.event_ctx_cli" if model_family == EVENT_CTX_MODEL_FAMILY
                       else "maichart.generation.event_cli"),
                "--checkpoint", str(checkpoint),
                "--audio", str(audio_path),
                "--output", str(output_path),
                "--output-ir", str(output_ir),
                "--condition-file", str(condition_path),
                "--first", str(payload.get("first", 0.0)),
                "--guidance-scale", str(payload.get("guidance_scale", GENERATION_SAMPLING_DEFAULTS["guidance_scale"])),
                "--candidates", str(candidate_count),
                "--seed", str(seed),
                "--device", str(payload.get("device", "auto")),
                "--title", str(payload.get("title", "Generated maimai chart")),
                "--artist", str(payload.get("artist", "")),
                "--chart-level", str(payload.get("chart_level", "")),
                "--designer", str(payload.get("designer", "AI")),
                "--tempo-mode", tempo_mode,
                # Corpus-measured defaults (generation/conditions.py): the same
                # numbers the panel shows, so an API caller with a bare payload
                # gets a typical chart instead of a 0.5-style outlier.
                "--temperature", str(payload.get("temperature", GENERATION_SAMPLING_DEFAULTS["temperature"])),
                # Nucleus truncation keeps the frequent-token tail of the
                # distribution without the long-shot tokens; 0 disables it.
                "--top-p", str(payload.get("top_p", GENERATION_SAMPLING_DEFAULTS["top_p"])),
                # One CTRL-style pass over the newest N tokens: damps
                # note-level stutter without touching chorus-scale repeats.
                "--repetition-penalty", str(payload.get("repetition_penalty",
                                                        GENERATION_SAMPLING_DEFAULTS["repetition_penalty"])),
                "--repetition-window", str(payload.get("repetition_window",
                                                       GENERATION_SAMPLING_DEFAULTS["repetition_window"])),
                "--max-tokens", str(max(1, int(payload.get("max_tokens", GENERATION_SAMPLING_DEFAULTS["max_tokens"])))),
            ]
            if capabilities["fine_note_strength"]:
                args.extend(["--fine-note-strength", str(fine_note_strength)])
                task.result["fine_note_strength"] = fine_note_strength
            elif fine_note_strength != 0.5:
                warning = "当前 checkpoint 未训练 music_structure_v3 fine_rhythm；细碎音符控制已忽略。"
                task.result.setdefault("warnings", []).append(warning)
                print(f"warning: {warning}", flush=True)
            if local_onset_guidance:
                args.extend(["--local-onset-bias-strength", str(local_onset_bias_strength)])
                task.result["local_onset_bias_strength"] = local_onset_bias_strength
            if curve_path is not None:
                args.extend(["--intensity-curve", str(curve_path),
                             "--intensity-curve-strength", str(curve_strength),
                             "--intensity-curve-mode", curve_mode])
            if explicit_tempo_map is not None:
                tempo_path = CONDITION_ROOT / f"{task.task_id}_tempo.json"
                tempo_path.write_text(json.dumps(explicit_tempo_map, ensure_ascii=False, indent=2), encoding="utf-8")
                args.extend(["--tempo-map", str(tempo_path)])
            # Intensity following: the condition vector is song-level, so this is
            # what tells each window how loud the music actually is.  Passed
            # explicitly in both directions so the log always states the mode.
            if payload.get("intensity_follow", GENERATION_SAMPLING_DEFAULTS["intensity_follow"]):
                args.append("--intensity-follow")
            else:
                args.append("--no-intensity-follow")
            args.extend([
                "--intensity-follow-span",
                str(payload.get("intensity_follow_span",
                                GENERATION_SAMPLING_DEFAULTS["intensity_follow_span"])),
            ])
            args.extend(["--intensity-mode", intensity_mode])
            if payload.get("tail_damping", GENERATION_SAMPLING_DEFAULTS["tail_damping"]):
                args.append("--tail-damping")
            args.extend([
                "--tail-damping-windows",
                str(payload.get("tail_damping_windows",
                                GENERATION_SAMPLING_DEFAULTS["tail_damping_windows"])),
                "--tail-damping-strength",
                str(payload.get("tail_damping_strength",
                                GENERATION_SAMPLING_DEFAULTS["tail_damping_strength"])),
            ])
            # Audible-span trim: the clock is the file, so without this the silent
            # lead-in and the trailing silence are asked for notes (measured: 8
            # notes before the music started, 9 after it ended).
            if payload.get("audio_bounds_trim", GENERATION_SAMPLING_DEFAULTS["audio_bounds_trim"]):
                args.append("--audio-bounds-trim")
            else:
                args.append("--no-audio-bounds-trim")
            args.extend([
                "--audio-bounds-lead",
                str(payload.get("audio_bounds_lead", GENERATION_SAMPLING_DEFAULTS["audio_bounds_lead"])),
                "--silence-threshold",
                str(payload.get("silence_threshold", GENERATION_SAMPLING_DEFAULTS["silence_threshold"])),
                "--silence-window-share",
                str(payload.get("silence_window_share",
                                GENERATION_SAMPLING_DEFAULTS["silence_window_share"])),
            ])
            if payload.get("sample", GENERATION_SAMPLING_DEFAULTS["sample"]):
                args.append("--sample")
            # Embedded playability diagnostics are available to every event
            # checkpoint; the fine-rhythm control is gated by v3 capability.
            if conflict_enabled:
                args.extend(["--enable-conflict-detection", "--conflict-engine", "miacode"])
                if payload.get("detect_hard_conflicts", True):
                    args.append("--detect-hard-conflicts")
                else:
                    args.append("--no-detect-hard-conflicts")
                if payload.get("detect_soft_conflicts", False):
                    args.append("--detect-soft-conflicts")
                args.extend(["--conflict-repair-mode", conflict_repair_mode])
                if conflict_repair_mode != "none" and conflict_retries:
                    args.extend(["--conflict-retries", str(conflict_retries)])
                args.extend(["--conflict-kind-modes", json.dumps(conflict_kind_modes, ensure_ascii=False)
                             if conflict_kind_modes is not None else "null",
                             "--tap-on-slide-threshold-ms", str(tap_on_slide_threshold_ms),
                             "--conflict-backtrack-windows", str(conflict_backtrack_windows),
                             "--conflict-backtrack-budget", str(conflict_backtrack_budget),
                             "--sustain-repair-mode", sustain_repair_mode,
                             "--note-repair-mode", effective_note_repair_mode])
                task.result["conflict_detection_enabled"] = True
                task.result["conflict_repair_mode"] = conflict_repair_mode
                task.result["conflict_retries"] = conflict_retries if conflict_repair_mode != "none" else 0
                task.result["conflict_kind_modes"] = conflict_kind_modes
                task.result["tap_on_slide_threshold_ms"] = tap_on_slide_threshold_ms
                task.result["conflict_backtrack_windows"] = conflict_backtrack_windows
                task.result["conflict_backtrack_budget"] = conflict_backtrack_budget
                task.result["sustain_repair_mode"] = sustain_repair_mode
                task.result["note_repair_mode"] = effective_note_repair_mode
            if model_family == EVENT_CTX_MODEL_FAMILY:
                # The window layout is part of the data contract, so the
                # checkpoint is authoritative: the UI fields are only a copy, and
                # a stale or empty field used to put generation out of
                # distribution (the model then emitted a couple of notes per
                # window and stopped).  Mismatches are recorded, not obeyed.
                trained = (checkpoint_payload.get("data_contract", {}) or {}).get("family_params") or {}
                layout: dict[str, Any] = {}
                mismatches: list[str] = []
                for key, default in (("context_beats", 8), ("audio_margin_beats", 4),
                                     ("context_tokens", 192), ("overlap_beats", 0)):
                    provided = payload.get(key)
                    expected = trained.get(key)
                    value = int(expected) if expected is not None else int(provided if provided is not None else default)
                    layout[key] = value
                    if provided is not None and int(provided) != value:
                        mismatches.append(f"{key}: UI={int(provided)} 训练={value}")
                args.extend([
                    "--context-beats", str(layout["context_beats"]),
                    "--audio-margin-beats", str(layout["audio_margin_beats"]),
                    "--context-tokens", str(max(1, layout["context_tokens"])),
                    "--overlap-beats", str(max(0, layout["overlap_beats"])),
                ])
                task.result["context_layout"] = layout
                if mismatches:
                    task.result["context_layout_warnings"] = mismatches
                    print(f"generation_layout=from_checkpoint {'; '.join(mismatches)} "
                          f"(checkpoint values are used; retrain to change the layout)", flush=True)
            if payload.get("bpm") not in {None, ""}:
                args.extend(["--bpm", str(payload["bpm"])])
            # The corpus condition profile supplies the control values the model
            # was actually trained with.  Without it every control fell back to
            # 0.5, while the corpus medians sit 0.2-0.5 lower, and the decoder
            # held one arbitrary style for hundreds of beats.
            profile = _resolve_condition_profile(payload)
            if profile is not None:
                args.extend(["--condition-profile", str(profile)])
                task.result["condition_profile"] = str(profile)
            # A gold reference turns the quality report into an acceptance
            # verdict comparing this chart with the held-out gold charts.
            if payload.get("gold_stats_path"):
                gold_path = _required_path(payload["gold_stats_path"], "金标统计", must_exist=True)
                args.extend(["--gold-stats", str(gold_path)])
                task.result["gold_stats"] = str(gold_path)
        else:
            args = [
                "-m", "maichart.generation.cli",
                "--checkpoint", str(checkpoint),
                "--output", str(output_path),
                "--output-ir", str(output_ir),
                "--condition-file", str(condition_path),
                "--level", str(payload.get("level", GENERATION_DEFAULTS["level"])),
                "--first", str(payload.get("first", 0.0)),
                "--division", str(payload.get("division", GENERATION_SAMPLING_DEFAULTS["division"])),
                "--candidates", str(candidate_count),
                "--seed", str(seed),
                "--presence-threshold", str(payload.get("presence_threshold", 0.45)),
                "--button-threshold", str(payload.get("button_threshold", 0.45)),
                "--guidance-scale", str(payload.get("guidance_scale", 1.0)),
                "--device", str(payload.get("device", "auto")),
                "--title", str(payload.get("title", "Generated maimai chart")),
                "--artist", str(payload.get("artist", "")),
                "--chart-level", str(payload.get("chart_level", "")),
                "--designer", str(payload.get("designer", "AI")),
            ]
            if payload.get("bpm") not in {None, ""}:
                args.extend(["--bpm", str(payload["bpm"])])
            if explicit_tempo_map is not None:
                tempo_path = CONDITION_ROOT / f"{task.task_id}_tempo.json"
                tempo_path.write_text(json.dumps(explicit_tempo_map, ensure_ascii=False, indent=2), encoding="utf-8")
                args.extend(["--tempo-map", str(tempo_path)])
            if audio_features_path is not None:
                args.extend(["--audio-features", str(audio_features_path)])
            else:
                args.extend(["--audio", str(audio_path)])
        _run_subprocess_task(task, args, expected_epoch=None, result_paths=result_paths)
        self._json_response({"task_id": task.task_id})

    def _gold_designers(self, payload: dict[str, Any]) -> None:
        """Designer histogram for the manifest, so gold criteria are discoverable."""

        manifest = _required_path(payload.get("manifest"), "训练 manifest", must_exist=True)
        data = json.loads(manifest.read_text(encoding="utf-8"))
        counter: Counter[str] = Counter()
        usable = 0
        for song in data.get("songs", []):
            for difficulty in song.get("difficulties", []):
                if not (difficulty.get("event_usable_for_training") or difficulty.get("usable_for_training")):
                    continue
                usable += 1
                name = str(difficulty.get("designer") or "").strip()
                counter[name or "<未填写>"] += 1
        self._json_response({
            "manifest": str(manifest),
            "usable_difficulties": usable,
            "designers": [{"name": name, "count": count} for name, count in counter.most_common(40)],
        })

    def _gold_scan(self, payload: dict[str, Any]) -> None:
        """Report what the gold folder holds, so the UI can preview it."""

        manifest = _required_path(payload.get("manifest"), "训练 manifest", must_exist=True)
        if not manifest.is_file():
            raise ValueError(f"训练 manifest 不存在：{manifest}")
        gold_root = payload.get("gold_root") or _default_gold_root(manifest, payload.get("raw_root"))
        gold_path = _resolve_path(gold_root)
        if not gold_path.is_dir():
            self._json_response({
                "gold_root": str(gold_path),
                "exists": False,
                "message": f"还没有 gold 文件夹：{gold_path}。请先创建它，并把每首金标曲放成独立文件夹"
                           f"（内含 maidata.txt 与 track.mp3/track.wav/track.ogg）。",
            })
            return
        from maichart.training.gold_set import scan_gold_folder

        report = scan_gold_folder(manifest, gold_root=gold_path)
        report["gold_root"] = str(gold_path)
        if not report.get("songs"):
            report["message"] = ("gold 文件夹里没有找到可用谱面：请确认每首歌是独立文件夹、"
                                 "里面有 maidata.txt 与 track.mp3/track.wav/track.ogg。")
        self._json_response(report)

    def _start_gold_build(self, payload: dict[str, Any]) -> None:
        """Split a manifest into pretrain / gold-finetune / gold-validation."""

        manifest = _required_path(payload.get("manifest"), "训练 manifest", must_exist=True)
        if not manifest.is_file():
            raise ValueError(f"训练 manifest 不存在：{manifest}")
        designers = [str(item).strip() for item in (payload.get("designers") or []) if str(item).strip()]
        song_ids = [str(item).strip() for item in (payload.get("song_ids") or []) if str(item).strip()]
        mode = str(payload.get("mode", "folder")).strip().lower()
        gold_root: Path | None = None
        if mode == "folder":
            gold_root = _resolve_path(
                payload.get("gold_root") or _default_gold_root(manifest, payload.get("raw_root"))
            )
            if not gold_root.is_dir():
                raise ValueError(
                    f"找不到 gold 文件夹：{gold_root}。请先把金标谱面按「每首一个文件夹」放进去"
                    f"（内含 maidata.txt 与 track.mp3/track.wav/track.ogg），或改用在训练集里按谱师名筛选。"
                )
        elif not designers and not song_ids:
            raise ValueError("请至少填写一位谱师名或一个 song_id：金标集必须来自明确的筛选条件")
        out_dir = _resolve_path(payload.get("out_dir") or _default_gold_out_dir(manifest))
        cache_dir = _resolve_path(payload.get("cache_dir") or str(WEBUI_ROOT / "cache"))
        # Writing the manifests into the folder that holds the gold charts is how
        # the outputs ended up mixed with the inputs and a second run nested one
        # level deeper ("gold/gold").  Refuse it here.
        if gold_root is not None and out_dir == gold_root:
            raise ValueError(
                f"输出目录不能与放谱面的 gold 文件夹相同：{out_dir}\n"
                "请把输出改到别处（例如训练集目录下的 gold_set），否则三份 manifest 会写进你的谱面文件夹。"
            )
        if gold_root is not None and gold_root in out_dir.parents:
            raise ValueError(
                f"输出目录不能位于 gold 谱面文件夹内部：{out_dir}\n"
                "否则重复执行会不断嵌套（gold/gold/…），并且清单里的相对路径会失效。"
            )
        val_ratio = float(payload.get("val_ratio", 0.3))
        if not 0.0 < val_ratio < 1.0:
            raise ValueError("验证集比例必须在 0 和 1 之间")
        task = _create_task("gold")
        pretrain = out_dir / "pretrain_manifest.json"
        finetune = out_dir / "gold_finetune_manifest.json"
        validation = out_dir / "gold_val_manifest.json"
        stats = out_dir / "gold_stats.json"
        task.result.update({
            "gold_dir": str(out_dir),
            "pretrain_manifest": str(pretrain),
            "gold_finetune_manifest": str(finetune),
            "gold_val_manifest": str(validation),
            "gold_stats": str(stats),
            "criteria": {"mode": mode, "gold_root": str(gold_root) if gold_root else None,
                         "designers": designers, "song_ids": song_ids,
                         "val_ratio": val_ratio, "min_note_count": int(payload.get("min_note_count", 0))},
        })
        args = [
            "-m", "maichart.training.gold_set", str(manifest),
            "--out-dir", str(out_dir), "--cache-dir", str(cache_dir),
            "--designers", ",".join(designers), "--song-ids", ",".join(song_ids),
            "--val-ratio", str(val_ratio), "--seed", str(int(payload.get("seed", 42))),
            "--min-note-count", str(max(0, int(payload.get("min_note_count", 0)))),
        ]
        if gold_root is not None:
            args.extend(["--gold-root", str(gold_root)])
        if payload.get("no_stats"):
            args.append("--no-stats")
        _run_subprocess_task(task, args, expected_epoch=None,
                             result_paths=[pretrain, finetune, validation, stats,
                                           out_dir / "gold_summary.json"])
        self._json_response({"task_id": task.task_id})

    def _start_acceptance(self, payload: dict[str, Any]) -> None:
        """Measure one chart and judge it against the gold reference."""

        maidata = _required_path(payload.get("maidata"), "谱面文件", must_exist=True)
        if not maidata.is_file():
            raise ValueError(f"谱面文件不存在：{maidata}")
        audio_value = payload.get("audio")
        audio = _required_path(audio_value, "音频", must_exist=True) if audio_value else None
        gold_value = payload.get("gold_stats")
        gold = _required_path(gold_value, "金标统计", must_exist=True) if gold_value else None
        cache_dir = _resolve_path(payload.get("cache_dir") or str(WEBUI_ROOT / "cache"))
        task = _create_task("acceptance")
        report = _resolve_path(payload.get("report") or str(GENERATED_ROOT / f"{task.task_id}_acceptance.json"))
        report.parent.mkdir(parents=True, exist_ok=True)
        task.result.update({
            "maidata": str(maidata), "audio": str(audio) if audio else None,
            "gold_stats": str(gold) if gold else None, "report": str(report),
        })
        args = [
            "-m", "maichart.generation.acceptance", str(maidata),
            "--difficulty", str(int(payload.get("difficulty", 5))),
            "--cache-dir", str(cache_dir), "-o", str(report),
            "--resolution", str(max(24, int(payload.get("resolution", 240)))),
        ]
        if audio is not None:
            args.extend(["--audio", str(audio)])
        if gold is not None:
            args.extend(["--gold-stats", str(gold)])
        if payload.get("bpm") not in {None, ""}:
            args.extend(["--bpm", str(payload["bpm"])])
        _run_subprocess_task(task, args, expected_epoch=None, result_paths=[report])
        self._json_response({"task_id": task.task_id})

    def _start_preprocess(self, payload: dict[str, Any]) -> None:
        model_family = _requested_model_family(payload)
        feature_profile = str(payload.get("feature_profile", "basic_v1"))
        if model_family == LEGACY_MODEL_FAMILY and feature_profile != "basic_v1":
            raise ValueError("当前可运行的音频特征方案是 basic_v1；rhythm_v2 还未接入预处理，请先使用 basic_v1。")
        raw_root = _required_path(payload.get("raw_root"), "raw dataset root", must_exist=True)
        cache_dir = _resolve_path(payload.get("cache_dir") or str(PROJECT_ROOT / "cache"))
        output = _resolve_path(payload.get("manifest") or str(raw_root / "training_manifest.json"))
        if output.is_file():
            existing = _read_training_manifest(output, raw_root)
            if existing.get("merged_from"):
                raise ValueError("联合训练清单不能作为预处理输出；请在‘当前预处理数据集’中选择一个原始数据集")
        existing_task = _find_preprocess_task(output)
        if existing_task is not None and existing_task.status in {"queued", "running"}:
            raise ValueError("这个训练集正在预处理，请等待当前任务完成，不需要再次点击预处理")
        task = _create_task("preprocess")
        # Keep the requested paths while the subprocess is running so the UI
        # can explain which manifest is being built before the file exists.
        task.result.update(
            {
                "raw_root": str(raw_root),
                "cache_dir": str(cache_dir),
                "manifest": str(output),
                "model_family": model_family,
            }
        )
        args = _preprocess_args(payload, raw_root=raw_root, cache_dir=cache_dir, output=output,
                                model_family=model_family)
        _run_subprocess_task(task, args, expected_epoch=None, result_paths=[output])
        self._json_response({"task_id": task.task_id})

    def _start_train(self, payload: dict[str, Any]) -> None:
        payload, dataset_selection = _prepare_training_sources(payload)
        model_family = _requested_model_family(payload)
        if model_family == LEGACY_MODEL_FAMILY and str(payload.get("feature_profile", "basic_v1")) != "basic_v1":
            raise ValueError("当前可运行的音频特征方案是 basic_v1；请先使用 basic_v1 训练。")
        manifest_value = payload.get("manifest")
        if not manifest_value:
            raise ValueError("请先填写训练 manifest 路径，并先完成训练集预处理")
        manifest = _resolve_path(str(manifest_value))
        auto_preprocess = bool(payload.get("auto_preprocess"))
        if auto_preprocess:
            # One click: build the manifest, then train, in a single task.  The
            # manifest legitimately does not exist yet, so nothing is validated
            # beyond the raw folder being present -- the output folder is created
            # here rather than making the user prepare it by hand.
            manifest.parent.mkdir(parents=True, exist_ok=True)
        elif not manifest.is_file():
            raise ValueError(
                f"训练 manifest 尚未生成: {manifest}。请先点击‘预处理训练集’，等待任务完成后再开始训练"
            )
        if manifest.is_file():
            pending = _manifest_pending_status(manifest)
            if pending is not None:
                raise ValueError(pending["message"])
        if manifest.is_file():
            data = _read_training_manifest(manifest, payload.get("raw_root"))
        else:
            data = {"songs": []}
        if auto_preprocess and data.get("merged_from"):
            raise ValueError("联合训练清单不能重新预处理；请分别预处理原始数据集，再使用‘开始训练’")
        if not _is_event_family(model_family) and not payload.get("include_unusable") and not any(
            difficulty.get("usable_for_training")
            for song in data.get("songs", [])
            for difficulty in song.get("difficulties", [])
        ):
            raise ValueError("当前训练集没有可训练难度。请点击‘检查训练集状态’，修复显示的过滤原因后重新预处理")
        quality_path = None
        preference_enabled = bool(payload.get("preference_enabled"))
        if bool(payload.get("quality_enabled")) or preference_enabled:
            if not _is_event_family(model_family):
                raise ValueError("质量加权和反面偏好训练目前只支持 event_v1 / event_ctx_v1")
            from maichart.training.quality_annotations import annotation_path, load_annotations

            quality_path = annotation_path(manifest)
            if payload.get("quality_annotations") and _resolve_path(str(payload["quality_annotations"])) != quality_path:
                raise ValueError("质量标注属于另一份训练集；请为当前 manifest 打开‘训练素材’并重新加载标注")
            annotations = load_annotations(quality_path)
            if not annotations["records"]:
                raise ValueError("当前训练集还没有已保存的质量标注；请先打开‘训练素材’标记谱面")
            if preference_enabled and not any(
                item.get("label") == "negative" and item.get("corrected_path")
                for item in annotations["records"]
            ):
                raise ValueError("偏好训练至少需要一个已上传修正版的反例；没有修正版的反例只会从正向学习中排除")
            preference_weight = float(payload.get("preference_weight", 0.05))
            preference_max_pairs = int(payload.get("preference_max_pairs", 32))
            if not math.isfinite(preference_weight) or not 0.0 <= preference_weight <= 1.0:
                raise ValueError("偏好训练强度必须是 0–1 的有限数字")
            if not 1 <= preference_max_pairs <= 512:
                raise ValueError("每轮反例窗口数必须在 1–512 之间")
        cache_dir = _resolve_path(payload.get("cache_dir") or str(PROJECT_ROOT / "cache"))
        checkpoint_dir = _resolve_path(payload.get("checkpoint_dir") or str(PROJECT_ROOT / "checkpoints" / model_family))
        planner_profile = str(payload.get("planner_profile", "auto"))
        teacher_mix = float(payload.get("plan_teacher_mix", 0.5))
        if planner_profile not in {"auto", "legacy_v1", "music_structure_v2",
                                   "music_structure_v3"}:
            raise ValueError("规划器模式只能是 auto、legacy_v1、music_structure_v2 或 music_structure_v3")
        if not math.isfinite(teacher_mix) or not 0.0 <= teacher_mix <= 1.0:
            raise ValueError("规划目标混合比例必须在 0–1 之间")
        validation_manifest = None
        if payload.get("validation_manifest"):
            if not _is_event_family(model_family):
                raise ValueError("旧固定帧基线不支持 validation-manifest；请选择事件模型，或清空验证路径")
            validation_manifest = _required_path(payload["validation_manifest"], "验证 manifest", must_exist=True)
            pending = _manifest_pending_status(validation_manifest)
            if pending is not None:
                raise ValueError(pending["message"])
            _read_training_manifest(validation_manifest)
            if validation_manifest == manifest:
                raise ValueError("训练和验证 manifest 不能是同一个文件；请先按歌曲划分数据集")
            assert_no_validation_overlap(manifest, validation_manifest)
        training_mode = str(payload.get("training_mode", "fresh")).strip().lower()
        if training_mode in {"from_scratch", "scratch", "new"}:
            training_mode = "fresh"
        if training_mode in {"continue", "checkpoint"}:
            training_mode = "resume"
        if training_mode in {"finetune", "fine_tune", "init"}:
            training_mode = "finetune"
        if training_mode not in {"fresh", "resume", "finetune"}:
            raise ValueError("训练方式只能是：fresh（从头）、resume（断点继续）或 finetune（已有模型加强训练）")
        condition_params = _event_condition_parameters(payload) if _is_event_family(model_family) else None
        try:
            learning_rate = float(payload.get("lr", 0.00002 if training_mode == "finetune" else 0.0002))
        except (TypeError, ValueError) as exc:
            raise ValueError("学习率必须是大于 0、且不超过 1 的有限数值") from exc
        if not math.isfinite(learning_rate) or not 0 < learning_rate <= 1:
            raise ValueError("学习率必须是大于 0、且不超过 1 的有限数值")
        if preference_enabled and training_mode == "fresh":
            raise ValueError("反面偏好训练需要已训练 checkpoint；先 fresh 做质量加权，再选择 finetune 启用修正版配对。")
        training_checkpoint = payload.get("training_checkpoint") or payload.get("checkpoint")
        training_start_epoch = 0
        inspection = None
        if training_mode != "fresh":
            if not training_checkpoint:
                raise ValueError("当前训练方式需要选择 checkpoint 文件或填写完整路径")
            training_checkpoint_path = _required_path(training_checkpoint, "训练 checkpoint", must_exist=True)
            inspection = _checkpoint_inspection(
                training_checkpoint_path, mode=training_mode, manifest=data, model_family=model_family,
            )
            if inspection["errors"]:
                raise ValueError("checkpoint 不兼容：\n- " + "\n- ".join(inspection["errors"]))
            if training_mode == "resume":
                training_start_epoch = max(0, int(inspection.get("epoch") or 0))
            if condition_params is not None:
                _check_event_condition_parameters(condition_params, inspection.get("data_contract", {}).get("condition_params"))
            stored_planner = inspection.get("model_config", {}).get("planner_profile", "legacy_v1")
            if planner_profile != "auto" and planner_profile != stored_planner:
                raise ValueError("改变音乐结构规划器需要 fresh 重新训练；原 checkpoint 可使用 auto 或其原规划器继续训练。")
        else:
            training_checkpoint_path = None
        onset_profile, onset_loss_weight, onset_weight_override = _resolve_onset_training_options(
            payload, model_family=model_family, training_mode=training_mode, inspection=inspection)
        if model_family in EVENT_FAMILIES and onset_profile == "chart_onsets_v1":
            stored_planner = ((inspection or {}).get("model_config") or {}).get("planner_profile")
            effective_planner = (stored_planner if planner_profile == "auto" and training_mode != "fresh"
                                 else MUSIC_STRUCTURE_V3_PROFILE if planner_profile == "auto"
                                 else planner_profile)
            if effective_planner != MUSIC_STRUCTURE_V3_PROFILE:
                raise ValueError("chart_onsets_v1 需要 music_structure_v3 planner；请将训练 profile 设为 v3 或使用兼容 checkpoint。")
        task = _create_task("train")
        if dataset_selection is not None:
            task.result["dataset_selection"] = dataset_selection
            task.result["manifest"] = str(manifest)
        epochs = max(1, int(payload.get("epochs", 20)))
        epoch_target = training_start_epoch + epochs
        seed, seed_randomized = _resolve_seed(payload)
        task.result.update({"seed": seed, "seed_randomized": seed_randomized,
                            "epoch_start": training_start_epoch,
                            "epochs_requested": epochs,
                            "epoch_target": epoch_target,
                            "epoch_semantics": ("additional_from_checkpoint" if training_mode == "resume"
                                                 else "stage_total")})
        if onset_profile is not None:
            task.result.update({"onset_profile": onset_profile, "onset_loss_weight": onset_loss_weight})
            if onset_weight_override:
                task.result.setdefault("warnings", []).append(
                    "resume 已使用 checkpoint 保存的 onset_loss_weight，忽略本次请求值。")
        if _is_event_family(model_family):
            try:
                cuda_memory_budget_gb = float(payload.get("cuda_memory_budget_gb", 0.0))
            except (TypeError, ValueError) as exc:
                raise ValueError("CUDA 显存预算必须是非负 GiB 数值；0 表示自动 60%。") from exc
            if not math.isfinite(cuda_memory_budget_gb) or cuda_memory_budget_gb < 0:
                raise ValueError("CUDA 显存预算必须是非负 GiB 数值；0 表示自动 60%。")
            task.result["cuda_memory_budget_gb"] = cuda_memory_budget_gb
            args = [
                # The cross-window family is the same trainer with a different
                # dataset and target_schema, so only the module name differs.
                "-m", ("maichart.training.train_event_ctx" if model_family == EVENT_CTX_MODEL_FAMILY
                       else "maichart.training.train_events"),
                "--manifest", str(manifest), "--cache-dir", str(cache_dir),
                "--checkpoint-dir", str(checkpoint_dir), "--epochs", str(epochs),
                "--lr", str(learning_rate),
                "--max-steps", str(max(0, int(payload.get("max_steps", 0)))),
                "--validation-max-steps", str(max(0, int(payload.get("validation_max_steps", 0)))),
                "--batch-size", str(max(1, int(payload.get("batch_size", 2)))),
                "--segment-beats", str(payload.get("segment_beats", 16)),
                "--d-model", str(int(payload.get("d_model", 256))),
                "--nhead", str(int(payload.get("nhead", 4))),
                "--num-layers", str(int(payload.get("num_layers", 4))),
                "--dim-feedforward", str(int(payload.get("dim_feedforward", 1024))),
                "--dropout", str(payload.get("dropout", 0.1)),
                "--condition-dropout", str(condition_params["dropout"]),
                "--plan-loss-weight", str(max(0.0, float(payload.get("plan_loss_weight", 0.1)))),
                "--planner-profile", planner_profile,
                "--plan-teacher-mix", str(teacher_mix),
                "--onset-profile", onset_profile,
                "--onset-loss-weight", str(onset_loss_weight),
                "--weight-decay", str(payload.get("weight_decay", 0.01)),
                # max_tokens is both the position grid and the padded length of
                # every batch, so it belongs to the training contract rather
                # than being a fixed 2048 the UI cannot reach.
                "--max-tokens", str(max(64, int(payload.get("max_tokens", 768)))),
                "--max-audio-tokens", str(max(128, int(payload.get("max_audio_tokens", 1024)))),
                 "--min-difficulty-index", str(max(1, min(6, int(payload.get("min_difficulty_index", 4))))),
                 "--max-difficulty-index", str(max(1, min(6, int(payload.get("max_difficulty_index", 6))))),
                 "--min-level", str(float(payload.get("min_level", 12.0))),
                 "--max-level", str(float(payload.get("max_level", 15.0))),
                 "--condition-default", str(condition_params["default"]),

                "--grad-accumulation", str(max(1, int(payload.get("grad_accumulation", 1)))),
                "--device", str(payload.get("device", "auto")),
                "--cuda-memory-budget-gb", str(cuda_memory_budget_gb),
                "--save-every", str(max(0, int(payload.get("save_every", 5)))),
                "--patience", str(max(0, int(payload.get("patience", 10)))),
                "--audio-cache-gb", str(float(payload.get("audio_cache_gb", 4.0))),
                "--prewarm-workers", str(payload.get("prewarm_workers", "auto")),
                "--seed", str(seed),
                "--condition-mask-prob", str(condition_params["mask_prob"]),
            ]
            if quality_path is not None:
                args.extend(["--quality-annotations", str(quality_path)])
            if preference_enabled:
                args.extend([
                    "--preference-enabled",
                    "--preference-weight", str(preference_weight),
                    "--preference-max-pairs", str(preference_max_pairs),
                ])
            if model_family == EVENT_CTX_MODEL_FAMILY:
                # Cross-window layout.  These are part of the contract, so they
                # are recorded in the checkpoint and compared on resume.
                args.extend([
                    "--context-beats", str(max(0, int(payload.get("context_beats", 8)))),
                    "--audio-margin-beats", str(max(0, int(payload.get("audio_margin_beats", 4)))),
                    "--context-tokens", str(max(1, int(payload.get("context_tokens", 192)))),
                    "--overlap-beats", str(max(0, int(payload.get("overlap_beats", 0)))),
                    "--context-dropout",
                    str(min(1.0, max(0.0, float(payload.get("context_dropout", 0.15))))),
                    "--slide-oversample",
                    str(max(1, int(payload.get("slide_oversample", 2)))),
                    "--touch-oversample",
                    str(max(1, int(payload.get("touch_oversample", 1)))),
                    "--fine-rhythm-oversample",
                    str(max(1, int(payload.get("fine_rhythm_oversample", 1)))),
                    "--fine-rhythm-ratio-threshold",
                    str(min(1.0, max(0.0, float(payload.get("fine_rhythm_ratio_threshold", 0.25))))),
                ])
            if payload.get("no_amp"):
                args.append("--no-amp")
            if validation_manifest is not None:
                args.extend(["--validation-manifest", str(validation_manifest)])
            if training_mode == "resume":
                args.extend(["--resume", str(training_checkpoint_path)])
            elif training_mode == "finetune":
                args.extend(["--init-checkpoint", str(training_checkpoint_path)])
            result_paths = [checkpoint_dir / "event_last.pt", checkpoint_dir / "event_best.pt"]
            task.result.update({"model_family": model_family, "checkpoint_dir": str(checkpoint_dir)})
            self._dispatch_training(
                task, args, result_paths=result_paths, epochs=epochs, epoch_target=epoch_target,
                auto_preprocess=auto_preprocess,
                payload=payload, model_family=model_family, manifest=manifest, cache_dir=cache_dir,
            )
            return
        args = [
            "-m", "maichart.training.train_generator",
            "--manifest", str(manifest), "--cache-dir", str(cache_dir),
            "--checkpoint-dir", str(checkpoint_dir), "--epochs", str(epochs),
            "--batch-size", str(max(1, int(payload.get("batch_size", 2)))),
            "--segment-frames", str(max(1, int(payload.get("segment_frames", 512)))),
            "--max-len", str(max(1, int(payload.get("max_len", 4096)))),
            "--grad-accumulation", str(max(1, int(payload.get("grad_accumulation", 1)))),
            "--lr", str(learning_rate),
            "--weight-decay", str(payload.get("weight_decay", 0.01)),
            "--d-model", str(int(payload.get("d_model", 256))),
            "--nhead", str(int(payload.get("nhead", 4))),
            "--num-layers", str(int(payload.get("num_layers", 4))),
            "--dim-feedforward", str(int(payload.get("dim_feedforward", 1024))),
            "--dropout", str(payload.get("dropout", 0.1)),
            "--device", str(payload.get("device", "auto")),
            "--save-every", str(max(0, int(payload.get("save_every", 5)))),
            "--seed", str(seed),
            "--condition-dropout", str(payload.get("condition_dropout", 0.15)),
        ]
        if training_mode == "resume":
            args.extend(["--resume", str(training_checkpoint_path)])
        elif training_mode == "finetune":
            args.extend(["--init-checkpoint", str(training_checkpoint_path)])
        if payload.get("reset_optimizer"):
            args.append("--reset-optimizer")
        if payload.get("include_unusable"):
            args.append("--include-unusable")
        if payload.get("no_amp"):
            args.append("--no-amp")
        task.result.update({"model_family": model_family, "checkpoint_dir": str(checkpoint_dir)})
        self._dispatch_training(
            task, args, result_paths=[checkpoint_dir / "generator_last.pt"], epochs=epochs,
            epoch_target=epoch_target,
            auto_preprocess=auto_preprocess, payload=payload, model_family=model_family,
            manifest=manifest, cache_dir=cache_dir,
        )

    def _dispatch_training(self, task, args: list[str], *, result_paths: list[Path], epochs: int,
                           epoch_target: int,
                           auto_preprocess: bool, payload: dict[str, Any], model_family: str,
                           manifest: Path, cache_dir: Path) -> None:
        """Start training, optionally preceded by preprocessing in the same task.

        The one-click path writes both argument lists to a plan file and runs
        ``maichart.training.auto_pipeline``: preprocessing takes minutes and
        training hours, so chaining them inside the browser would leave the run
        orphaned by a page reload.
        """

        if auto_preprocess:
            raw_root = _required_path(payload.get("raw_root"), "raw dataset root", must_exist=True)
            prepare_argv = _preprocess_args(payload, raw_root=raw_root, cache_dir=cache_dir,
                                            output=manifest, model_family=model_family)
            # `_preprocess_args` returns subprocess-shaped argv (leading
            # `-m <module>`), but the wrapper calls the stage in-process, which
            # takes only the real arguments.  Passing the launcher prefix through
            # made argparse fail with "unrecognized arguments: -m <raw_root>"
            # before any preprocessing ran.  The trainer list below has always
            # stripped its own prefix (`args[2:]`); the prepare list simply
            # never got the same treatment.
            if _is_event_family(model_family):
                prepare_stage = prepare_argv[2:]
            else:
                # legacy launcher is `-m maichart.cli preprocess build ...`
                prepare_stage = prepare_argv[4:]
            plan = {
                "family": model_family,
                "prepare": prepare_stage,
                # args starts with "-m <module>"; the wrapper calls the trainer
                # directly, so only the real arguments are forwarded.
                "train": list(args[2:]),
            }
            plan_path = PLAN_ROOT / f"{task.task_id}_plan.json"
            plan_path.parent.mkdir(parents=True, exist_ok=True)
            plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
            task.result.update({
                "auto_preprocess": True,
                "raw_root": str(raw_root),
                "manifest": str(manifest),
                "plan": str(plan_path),
            })
            args = ["-m", "maichart.training.auto_pipeline", "--plan", str(plan_path)]
            # The manifest is produced by the first stage, so it is watched too.
            result_paths = [manifest, *result_paths]
        _run_subprocess_task(task, args, expected_epoch=epoch_target, result_paths=result_paths)
        self._json_response({"task_id": task.task_id})

    def _inspect_checkpoint(self, payload: dict[str, Any]) -> None:
        payload, dataset_selection = _prepare_training_sources(payload)
        checkpoint_path = _required_path(
            payload.get("checkpoint_path") or payload.get("checkpoint"),
            "checkpoint",
            must_exist=True,
        )
        manifest = None
        manifest_value = payload.get("manifest")
        if manifest_value:
            manifest_path = _resolve_path(str(manifest_value))
            if manifest_path.is_file():
                manifest = _read_training_manifest(manifest_path, payload.get("raw_root"))
        mode = str(payload.get("mode", "finetune")).strip().lower()
        if mode in {"continue", "checkpoint"}:
            mode = "resume"
        if mode in {"init", "fine_tune"}:
            mode = "finetune"
        if mode not in {"resume", "finetune", "generate"}:
            mode = "finetune"
        inspection = _checkpoint_inspection(
            checkpoint_path, mode=mode, manifest=manifest,
            model_family=None if mode == "generate" else _requested_model_family(payload),
        )
        if dataset_selection is not None:
            inspection["dataset_selection"] = dataset_selection
            inspection["manifest"] = payload["manifest"]
        self._json_response(inspection)

    def _inspect_manifest(self, payload: dict[str, Any]) -> None:
        payload, dataset_selection = _prepare_training_sources(payload)
        model_family = _requested_model_family(payload)
        manifest_value = payload.get("manifest")
        if not manifest_value:
            raise ValueError("请先填写训练 manifest 路径")
        manifest_path = _resolve_path(str(manifest_value))
        pending = _manifest_pending_status(manifest_path)
        if pending is not None:
            self._json_response(pending)
            return
        data = _read_training_manifest(manifest_path, payload.get("raw_root"))
        songs = [song for song in data.get("songs", []) if isinstance(song, dict)]
        difficulty_records = [
            (song, difficulty)
            for song in songs
            for difficulty in song.get("difficulties", [])
            if isinstance(difficulty, dict)
        ]
        difficulties = [difficulty for _, difficulty in difficulty_records]
        is_event_manifest = (data.get("preprocessing") or {}).get("target_schema") == EVENT_MODEL_FAMILY
        if _is_event_family(model_family) and is_event_manifest:
            event_feature_profile = (data.get("preprocessing") or {}).get("feature_profile")
            if event_feature_profile not in {None, "event_audio_v1", "event_audio_v2_rhythm"}:
                raise ValueError("事件模型 manifest 的 feature_profile 不受支持")
        reasons = Counter(
            str(reason)
            for difficulty in difficulties
            for reason in difficulty.get("filter_reasons", [])
        )
        audio_statuses = Counter(
            str((song.get("audio") or {}).get("status"))
            if (song.get("audio") or {}).get("status") is not None
            else "source_present" if song.get("audio_path") else "unknown"
            for song in songs
        )
        difficulty_details = [
            {
                "song_id": str(song.get("song_id") or "unknown"),
                "maidata_path": song.get("maidata_path"),
                "audio_path": song.get("audio_path"),
                "difficulty_index": difficulty.get("difficulty_index"),
                "level_raw": difficulty.get("level_raw"),
                "note_count": difficulty.get("note_count"),
                "usable_for_training": bool(difficulty.get("usable_for_training")),
                "event_usable_for_training": difficulty.get("event_usable_for_training"),
                "partial_parse": bool(difficulty.get("partial_parse")),
                "filter_reasons": list(difficulty.get("filter_reasons") or []),
                "parse_coverage": difficulty.get("parse_coverage"),
                "validation_issues": list(difficulty.get("validation_issues") or []),
                "unknown_token_samples": list(difficulty.get("unknown_token_samples") or []),
                "processing_error": difficulty.get("processing_error"),
            }
            for song in songs
            for difficulty in song.get("difficulties", [])
            if isinstance(difficulty, dict)
        ]
        self._json_response(
            {
                "manifest": str(manifest_path),
                "dataset_selection": dataset_selection,
                "schema": data.get("schema"),
                "ready": True,
                "state": "prepared",
                "source_root": _manifest_relative_path(data.get("source_root"), manifest_path),
                "cache_dir": _manifest_relative_path(data.get("cache_dir"), manifest_path),
                "song_count": len(songs),
                "difficulty_count": len(difficulties),
                "usable_difficulty_count": (
                    sum(1 for difficulty in difficulties if difficulty.get("usable_for_training"))
                    if not (_is_event_family(model_family) and is_event_manifest)
                    else sum(1 for song, difficulty in difficulty_records if _event_difficulty_usable(song, difficulty))
                ),
                "filtered_difficulty_count": (
                    sum(1 for difficulty in difficulties if not difficulty.get("usable_for_training"))
                    if not (_is_event_family(model_family) and is_event_manifest)
                    else sum(1 for song, difficulty in difficulty_records if not _event_difficulty_usable(song, difficulty))
                ),
                "model_family": model_family,
                "event_manifest": bool(is_event_manifest),
                "top_filter_reasons": reasons.most_common(8),
                "audio_statuses": dict(audio_statuses),
                "preprocessing": data.get("preprocessing") or {},
                "difficulty_details": difficulty_details[:100],
                "processing_errors": _manifest_processing_errors(data),
            }
        )

    def _cancel_task(self, payload: dict[str, Any]) -> None:
        task = _get_task(str(payload.get("task_id", "")))
        if task is None:
            raise ValueError("任务不存在")
        process = task.process
        if process is not None and process.poll() is None:
            process.terminate()
            task.status = "cancelled"
            task.message = "已请求中断"
        self._json_response(task.to_dict())

    def _upload(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > MAX_UPLOAD_BYTES:
            raise ValueError("上传文件为空或超过 20 GB 限制")
        boundary = _multipart_boundary(self.headers.get("Content-Type", ""))
        UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
        # The body is streamed part by part: file parts go to a temp file on
        # disk, so a multi-GB ZIP never sits in RAM.  Buffering the whole body
        # (the previous behaviour) made the browser drop the connection with
        # "Failed to fetch" on large training sets.
        with tempfile.TemporaryDirectory(dir=UPLOAD_ROOT, prefix="upload-") as tmp:
            fields = _parse_multipart_stream(self.rfile, boundary, length, Path(tmp))
            kind = str(fields.get("kind", "file"))
            file_info = fields.get("file")
            if not isinstance(file_info, tuple):
                raise ValueError("缺少 file 字段")
            filename, part_path = file_info
            safe_name = _safe_filename(filename)
            allowed = {
                "audio": {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".wma", ".mp4"},
                "audio_features": {".json"},
                "dataset_zip": {".zip"},
                "checkpoint": {".pt", ".pth"},
                "condition": {".json"},
                "maidata": {".txt", ".maidata"},
            }
            suffix = Path(safe_name).suffix.lower()
            if kind not in allowed or suffix not in allowed[kind]:
                raise ValueError(f"不支持的 {kind} 文件类型: {suffix}")
            stored = UPLOAD_ROOT / f"{uuid.uuid4().hex[:12]}_{safe_name}"
            shutil.move(str(part_path), stored)
        if kind == "dataset_zip":
            # Extraction runs in the background so the upload response returns
            # as soon as the bytes are on disk; the browser keeps the connection
            # open for minutes otherwise and large requests died mid-flight.
            upload_id = uuid.uuid4().hex[:8]
            final_root = DATASET_ROOT / Path(safe_name).stem / upload_id
            _start_extraction(stored, final_root, upload_id)
            return self._json_response(
                {
                    "kind": kind,
                    "path": str(final_root),
                    "name": safe_name,
                    "cache_dir": str(final_root / "cache"),
                    "manifest": str(final_root / "training_manifest.json"),
                    "upload_id": upload_id,
                    "extracting": True,
                }
            )
        self._json_response({"kind": kind, "path": str(stored), "name": safe_name})

    def _download(self, raw_path: str) -> None:
        path = _resolve_path(raw_path)
        if not _is_relative_to(path, PROJECT_ROOT):
            return self._json_response({"error": "只允许下载项目目录中的文件"}, HTTPStatus.FORBIDDEN)
        if not path.is_file():
            return self._json_response({"error": "文件不存在"}, HTTPStatus.NOT_FOUND)
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Disposition", f"attachment; filename*=UTF-8''{quote(path.name)}")
        self.end_headers()
        self.wfile.write(data)

    def _serve_static(self, name: str) -> None:
        path = (STATIC_ROOT / name).resolve()
        if not _is_relative_to(path, STATIC_ROOT) or not path.is_file():
            return self._json_response({"error": "静态文件不存在"}, HTTPStatus.NOT_FOUND)
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "text/plain")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 16 * 1024 * 1024:
            raise ValueError("请求体为空或过大")
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("请求体必须是 JSON 对象")
        return payload

    def _json_response(self, value: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:
        return


def _create_task(kind: str) -> TaskState:
    task = TaskState(task_id=uuid.uuid4().hex[:12], kind=kind)
    with TASKS_LOCK:
        TASKS[task.task_id] = task
        for old_id, old_task in list(TASKS.items()):
            if old_id != task.task_id and old_task.status in {"completed", "failed", "cancelled"} and len(TASKS) > 20:
                del TASKS[old_id]
    return task


def _run_subprocess_task(task: TaskState, args: list[str], *, expected_epoch: int | None, result_paths: list[Path]) -> None:
    def worker() -> None:
        task.status = "running"
        task.message = "启动任务"
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        current_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + (os.pathsep + current_pythonpath if current_pythonpath else "")
        # Windows OpenMP defaults to active spin-waiting: a GPU-bound training
        # loop measured ~6 cores of CPU time doing nothing, stealing issue slots
        # from the thread that feeds the GPU.  Passive waiting and a zero
        # blocktime remove the spin; both are user-overridable.
        env.setdefault("OMP_WAIT_POLICY", "PASSIVE")
        env.setdefault("KMP_BLOCKTIME", "0")
        # With VRAM at 95% the caching allocator has no headroom, and its default
        # block-splitting fragments what little is left; measured on this project's
        # run that showed up as a saturated GPU with ~5% of peak throughput.
        # Expandable segments let blocks grow and merge instead.
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        _open_task_log(task, args)
        try:
            process = subprocess.Popen(
                [sys.executable, *args],
                cwd=PROJECT_ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
            task.process = process
            assert process.stdout is not None
            for line in process.stdout:
                task.log(line)
                if expected_epoch:
                    match = re.search(r"epoch=(\d+)", line)
                    if match:
                        task.progress = min(0.99, int(match.group(1)) / expected_epoch)
            code = process.wait()
            if task.status == "cancelled":
                _close_task_log(task, status="cancelled", exit_code=code)
                return
            if code != 0:
                task.status = "failed"
                tail = "\n".join(task.logs[-6:])
                task.error = f"子进程退出码 {code}" + (f"\n{tail}" if tail else "")
                task.message = "任务失败"
                _close_task_log(task, status="failed", exit_code=code, error=task.error)
                return
            task.status = "completed"
            task.progress = 1.0
            task.message = "任务完成"
            _close_task_log(task, status="completed", exit_code=0)
            _collect_task_results(task, result_paths)
        except Exception as exc:  # noqa: BLE001 - task state carries the error
            task.status = "failed"
            task.error = str(exc)
            task.message = "任务启动失败"
            _close_task_log(task, status="failed", error=str(exc))
        finally:
            task.process = None

    threading.Thread(target=worker, name=f"maichart-{task.kind}-{task.task_id}", daemon=True).start()


def _generation_result_paths(output: Path, output_ir: Path, count: int, model_family: str) -> list[Path]:
    paths: list[Path] = []
    for index in range(count):
        candidate = _candidate_path(output, index, count)
        paths.extend([candidate, _candidate_path(output_ir, index, count)])
        if _is_event_family(model_family):
            paths.extend([
                candidate.with_suffix(".quality.json"),
                candidate.with_name(candidate.stem + ".original.txt"),
                candidate.with_name(candidate.stem + ".original.chart.json"),
                candidate.with_name(candidate.stem + ".repaired.txt"),
            ])
    return paths


def _logs_root() -> Path:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    return LOG_ROOT


def _log_group_key(path: Path) -> str:
    """Both ``x.log`` and ``x.meta.json`` share this key, so pruning keeps them together."""

    return path.name.split(".")[0]


def _open_task_log(task: TaskState, argv: Sequence[str]) -> Path | None:
    """Start persisting a task's log; the file outlives the process and the UI."""

    try:
        root = _logs_root()
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = root / f"{stamp}_{task.kind}_{task.task_id}.log"
        task.argv = [str(value) for value in argv]
        task.started_at = time.strftime("%Y-%m-%d %H:%M:%S")
        handle = path.open("a", encoding="utf-8", buffering=1)
        handle.write(
            "# maichart task log (full stream; the UI keeps only the last 300 lines)\n"
            f"# task_id={task.task_id}\n# kind={task.kind}\n# started={task.started_at}\n"
            f"# cwd={PROJECT_ROOT}\n# argv={' '.join(task.argv)}\n"
        )
        task._handle = handle
        task.log_path = path
        task.result["log_path"] = str(path)
        return path
    except OSError as exc:  # a read-only or full disk must not break the task
        task.result["log_path_error"] = str(exc)
        return None


def _close_task_log(task: TaskState, *, status: str, exit_code: int | None = None,
                    error: str | None = None) -> None:
    """Finish the log file and write the sidecar the history list reads."""

    handle = task._handle
    ended = time.strftime("%Y-%m-%d %H:%M:%S")
    if handle is not None:
        try:
            handle.write(f"# finished={ended}\n# status={status}\n"
                         f"# exit_code={'' if exit_code is None else exit_code}\n")
            if error:
                first = str(error).splitlines()[0] if str(error).splitlines() else ""
                handle.write(f"# error={first}\n")
        except (OSError, ValueError):
            pass
        finally:
            try:
                handle.close()
            except OSError:
                pass
    task._handle = None
    task.ended_at = ended
    task.exit_code = exit_code
    path = task.log_path
    if path is None:
        return
    meta = {
        "task_id": task.task_id, "kind": task.kind, "status": status,
        "started": task.started_at, "ended": ended, "exit_code": exit_code,
        "argv": task.argv, "log": str(path),
        "log_bytes": path.stat().st_size if path.is_file() else 0,
        "error": (str(error) if error else "")[:2000],
    }
    try:
        path.with_suffix(".meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass
    _prune_logs()


def _prune_logs(max_files: int | None = None, max_bytes: int | None = None) -> None:
    """Drop the oldest whole runs once the log directory exceeds its budget."""

    files_budget = LOG_KEEP_FILES if max_files is None else max_files
    bytes_budget = LOG_KEEP_BYTES if max_bytes is None else max_bytes
    if not LOG_ROOT.is_dir():
        return
    groups: dict[str, list[Path]] = {}
    for path in LOG_ROOT.iterdir():
        if path.is_file():
            groups.setdefault(_log_group_key(path), []).append(path)
    ordered = sorted(groups.items(),
                     key=lambda item: max(p.stat().st_mtime for p in item[1]))
    total = sum(p.stat().st_size for group in groups.values() for p in group)
    count = len(groups)
    for key, group in ordered:
        if count <= files_budget and total <= bytes_budget:
            break
        if count <= 1:
            break  # never delete the newest run
        for path in group:
            try:
                total -= path.stat().st_size
                path.unlink()
            except OSError:
                pass
        count -= 1


def _persisted_logs(limit: int = 100) -> list[dict[str, Any]]:
    """List persisted runs, newest first, for the interface's history panel."""

    if not LOG_ROOT.is_dir():
        return []
    entries: list[dict[str, Any]] = []
    meta_paths = sorted(LOG_ROOT.glob("*.meta.json"),
                        key=lambda path: path.stat().st_mtime, reverse=True)
    for meta_path in meta_paths[:limit]:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        log_path = Path(str(meta.get("log") or ""))
        exists = log_path.is_file()
        entries.append({
            "task_id": meta.get("task_id"), "kind": meta.get("kind"),
            "status": meta.get("status"), "started": meta.get("started"),
            "ended": meta.get("ended"), "exit_code": meta.get("exit_code"),
            "error": meta.get("error") or "",
            "log_path": str(log_path) if log_path else "",
            "bytes": log_path.stat().st_size if exists else 0,
            "download": (f"/api/download?path={quote(str(log_path))}"
                         if exists and _is_relative_to(log_path.resolve(), PROJECT_ROOT) else None),
        })
    return entries


def _collect_task_results(task: TaskState, result_paths: list[Path]) -> None:
    paths = list(result_paths)
    if task.kind == "train" and _is_event_family(task.result.get("model_family")):
        paths.extend(sorted(Path(task.result["checkpoint_dir"]).glob("event_epoch_[0-9][0-9][0-9].pt")))
    existing = list(dict.fromkeys(path for path in paths if path.is_file()))
    task.result["paths"] = [str(path) for path in existing]
    task.result["files"] = [path.name for path in existing]
    task.result["download_urls"] = [
        f"/api/download?path={quote(str(path))}" if _is_relative_to(path.resolve(), PROJECT_ROOT) else None
        for path in existing
    ]
    if task.kind == "acceptance":
        # Hand the report straight to the interface; the panel draws it.
        report_path = Path(str(task.result.get("report", "")))
        if report_path.is_file():
            try:
                task.result["acceptance_report"] = json.loads(report_path.read_text(encoding="utf-8-sig"))
            except (OSError, ValueError) as exc:
                task.result["acceptance_error"] = f"无法读取验收报告: {exc}"
        else:
            task.result["acceptance_error"] = "验收任务没有写出报告文件"
        return
    if task.kind == "gold":
        summary_path = Path(str(task.result.get("gold_dir", ""))) / "gold_summary.json"
        if summary_path.is_file():
            try:
                task.result["gold_summary"] = json.loads(summary_path.read_text(encoding="utf-8-sig"))
            except (OSError, ValueError) as exc:
                task.result["gold_error"] = f"无法读取金标汇总: {exc}"
        for key in ("pretrain_manifest", "gold_finetune_manifest", "gold_val_manifest", "gold_stats"):
            value = task.result.get(key)
            if value and Path(str(value)).is_file():
                task.result[f"{key}_exists"] = True
        return
    if task.kind != "generate":
        return
    quality_paths = [path for path in paths if path.name.endswith(".quality.json")]
    task.result["quality_report_expected"] = bool(quality_paths)
    task.result["quality_report_exists"] = any(path.is_file() for path in quality_paths)
    reports = []
    for path in quality_paths:
        report: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
        if path.is_file():
            try:
                report["diagnostics"] = json.loads(path.read_text(encoding="utf-8-sig"))
            except (OSError, ValueError) as exc:
                report["error"] = f"无法读取质量诊断: {exc}"
        reports.append(report)
    task.result["quality_reports"] = reports
    task.result["quality_notice"] = "语法通过不代表可玩、难度达标或音乐贴合；请检查诊断并人工试玩。"


def _all_tasks() -> list[TaskState]:
    with TASKS_LOCK:
        return list(TASKS.values())


def _get_task(task_id: str) -> TaskState | None:
    with TASKS_LOCK:
        return TASKS.get(task_id)


def _find_preprocess_task(manifest_path: Path) -> TaskState | None:
    """Return the newest preprocess task that targets ``manifest_path``."""

    target = manifest_path.resolve()
    with TASKS_LOCK:
        candidates = [
            task
            for task in TASKS.values()
            if task.kind == "preprocess"
            and task.result.get("manifest")
            and Path(str(task.result["manifest"])).resolve() == target
        ]
    return candidates[-1] if candidates else None


def _manifest_pending_status(manifest_path: Path) -> dict[str, Any] | None:
    task = _find_preprocess_task(manifest_path)
    if task is not None and task.status in {"queued", "running"}:
        state = "processing"
        message = "正在预处理这个训练集；请等待右侧任务完成。预处理完成后才能检查和训练。"
    elif task is not None and task.status in {"failed", "cancelled"}:
        state = "preprocess_failed"
        message = "这个训练集的预处理失败或已中断，请查看任务日志，修复后重新点击‘预处理训练集’。"
    elif not manifest_path.is_file():
        state = "not_preprocessed"
        message = "尚未生成训练清单（training_manifest.json）。上传 ZIP 只会解压文件；请先点击‘预处理训练集’，完成后会自动检查。"
    else:
        return None
    return {
        "ready": False,
        "state": state,
        "manifest": str(manifest_path),
        "message": message,
        "task_id": task.task_id if task is not None else None,
    }


def _manifest_relative_path(value: Any, manifest_path: Path) -> str | None:
    if not value:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = manifest_path.parent / path
    return str(path.resolve())


def _read_training_manifest(manifest_path: Path, raw_root: Any = None) -> dict[str, Any]:
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"训练清单不是有效 JSON，请重新预处理: {exc}") from exc
    if not isinstance(data, dict) or data.get("schema") != "maichart-training-manifest-v1":
        raise ValueError("这不是训练清单，请先点击‘预处理训练集’")
    source = _manifest_relative_path(data.get("source_root"), manifest_path)
    if raw_root and source and _resolve_path(str(raw_root)) != Path(source):
        raise ValueError(
            f"当前训练清单属于另一份原始训练集（数据源: {source}）。请重新选择对应的已上传训练集，或点击‘预处理训练集’为当前目录生成新清单"
        )
    return data


def _dataset_api() -> DatasetAPI:
    # Resolve roots at call time, including isolated test/QA workspaces.
    return DatasetAPI(project_root=PROJECT_ROOT, dataset_root=DATASET_ROOT, workspace_root=WEBUI_ROOT)


def _prepare_training_sources(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Use the same selected sources for inspection and actual training."""
    manifests = payload.get("manifests")
    if manifests is None or manifests == []:
        return payload, None
    if not isinstance(manifests, list) or any(not isinstance(value, str) or not value.strip() for value in manifests):
        raise ValueError("训练数据集列表必须是非空路径组成的列表")
    manifests = list(dict.fromkeys(str(_resolve_path(value)) for value in manifests))
    if payload.get("auto_preprocess") and len(manifests) > 1:
        raise ValueError("联合训练前请分别预处理选中的数据集，然后点击‘开始训练’；一键预处理只处理当前单个数据集。")
    if payload.get("auto_preprocess") and len(manifests) == 1:
        if _resolve_path(str(payload.get("manifest") or "")) != Path(manifests[0]):
            raise ValueError("一键预处理的当前目录与勾选的数据集不一致，请先切换到这份数据集。")
        return payload, None
    for value in manifests:
        pending = _manifest_pending_status(Path(value))
        if pending is not None:
            raise ValueError(pending["message"] + f" 数据集：{Path(value).parent.name}")
    prepared = _dataset_api().prepare(manifests, model_family=_requested_model_family(payload))
    result = dict(payload)
    result.update({"manifest": prepared["manifest"], "raw_root": prepared.get("raw_root") or "",
                   "cache_dir": prepared["cache_dir"], "auto_preprocess": False})
    if result.get("quality_enabled") or result.get("preference_enabled"):
        result["quality_annotations"] = prepared.get("quality_annotations") or prepared.get("annotation_path")
    return result, prepared


def _event_difficulty_usable(song: dict[str, Any], difficulty: dict[str, Any]) -> bool:
    """Use event-preprocessor QC fields without trusting fixed-frame flags."""

    if difficulty.get("event_usable_for_training") is not None:
        return bool(difficulty.get("event_usable_for_training"))
    if difficulty.get("event_qc"):
        return not bool((difficulty.get("event_qc") or {}).get("errors"))
    # New event manifests may intentionally keep the shared field false for
    # fixed-frame compatibility; source paths and the explicit event marker are
    # the gate until the new CLI adds richer QC fields.
    return bool(
        song.get("maidata_path") and song.get("audio_path")
    ) or bool(difficulty.get("maidata_path") or difficulty.get("source_path"))


def _event_condition_parameters(payload: dict[str, Any]) -> dict[str, Any]:
    controls = {"mask_prob": ("condition_mask_prob", "条件随机遮罩", 0.5),
                "dropout": ("condition_dropout", "条件随机丢弃（CFG 锚）", 0.0),
                "default": ("condition_default", "被遮罩控制默认值", 0.5)}
    values: dict[str, Any] = {"null_embedding": True, "null_policy": "masked_condition_controls"}
    for key, (field, label, fallback) in controls.items():
        try:
            value = float(payload.get(field, fallback))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label}（{field}）必须是 0–1 的有限数字") from exc
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{label}（{field}）必须是 0–1 的有限数字")
        values[key] = value
    return values


def _resolve_onset_training_options(
    payload: dict[str, Any], *, model_family: str, training_mode: str,
    inspection: dict[str, Any] | None,
) -> tuple[str | None, float | None, bool]:
    has_options = "onset_profile" in payload or "onset_loss_weight" in payload
    if not _is_event_family(model_family):
        if has_options:
            raise ValueError("局部采音对齐训练选项只支持 event_v1 / event_ctx_v1")
        return None, None, False

    profile = str(payload.get("onset_profile", "auto")).strip().lower()
    if profile not in ONSET_PROFILES:
        raise ValueError("onset_profile 只能是 auto、none 或 chart_onsets_v1")
    try:
        requested_weight = float(payload.get("onset_loss_weight", 0.2))
    except (TypeError, ValueError) as exc:
        raise ValueError("onset_loss_weight 必须是非负有限数值") from exc
    if not math.isfinite(requested_weight) or requested_weight < 0:
        raise ValueError("onset_loss_weight 必须是非负有限数值")

    weight_overridden = False
    if training_mode == "resume":
        inspection = inspection or {}
        contract = inspection.get("data_contract") or {}
        stored_params = contract.get("onset_params") or {}
        model_config = inspection.get("model_config") or {}
        training = inspection.get("onset_training") or {}
        stored_profile = (stored_params.get("profile") or model_config.get("onset_profile") or "none")
        if stored_profile not in {"none", "chart_onsets_v1"}:
            raise ValueError(f"checkpoint 保存了不支持的 onset_profile：{stored_profile!r}")
        stored_weight = stored_params.get("loss_weight", training.get("loss_weight", 0.2))
        try:
            stored_weight = float(stored_weight)
        except (TypeError, ValueError) as exc:
            raise ValueError("checkpoint 的 onset_loss_weight 无效，不能 resume") from exc
        if not math.isfinite(stored_weight) or stored_weight < 0:
            raise ValueError("checkpoint 的 onset_loss_weight 无效，不能 resume")
        if profile != "auto" and profile != stored_profile:
            raise ValueError(
                "resume 不能改变 checkpoint 的 onset_profile；要启用或关闭 onset head，请使用 finetune。"
            )
        profile = stored_profile
        weight_overridden = not math.isclose(requested_weight, stored_weight, rel_tol=0.0, abs_tol=1e-12)
        requested_weight = stored_weight

    return profile, requested_weight, weight_overridden


def _check_event_condition_parameters(current: dict[str, Any], stored: Any) -> None:
    if not isinstance(stored, dict):
        return  # Older incomplete contracts remain guarded by the trainer.
    labels = {"dropout": "条件随机丢弃（CFG 锚，condition_dropout）",
              "mask_prob": "条件随机遮罩（condition_mask_prob）",
              "default": "被遮罩控制默认值（condition_default）"}
    differences = [f"{labels.get(key, key)}：checkpoint={stored.get(key)!r}，本次={current.get(key)!r}"
                   for key in sorted(stored.keys() | current.keys()) if stored.get(key) != current.get(key)]
    if differences:
        raise ValueError("条件合同不兼容：\n- " + "\n- ".join(differences)
                         + "\n请点击‘检查 checkpoint’回填原模型参数，或将上述控件设为 checkpoint 的值。"
                         "模型内部 dropout、上下文随机丢弃 context_dropout 与条件随机丢弃是三个独立参数；"
                         "resume 和 finetune 都需要保持当前条件合同。")


def _is_event_family(family: str | None) -> bool:
    """Both event families share the pipeline: audio in, Simai events out."""

    return family in EVENT_FAMILIES


def _requested_model_family(payload: dict[str, Any]) -> str:
    family = str(payload.get("model_family", EVENT_MODEL_FAMILY)).strip().lower()
    if family not in EVENT_FAMILIES | {LEGACY_MODEL_FAMILY}:
        raise ValueError("model family 只能是 event_v1、event_ctx_v1（跨窗口）或 legacy_frame_v2（旧固定帧基线）")
    return family


def _load_checkpoint_payload(path: Path) -> dict[str, Any]:
    # The shared loader uses torch.load and preserves unknown schema fields.
    return load_checkpoint(path, map_location="cpu")


def _generation_capabilities(model_family: str, payload: dict[str, Any]) -> dict[str, bool]:
    """Expose schema-level tools and gate planner controls on a valid contract."""
    detected = checkpoint_capabilities(payload)
    if detected["model_family"] != model_family:
        return {"conflict_detection": False, "conflict_repair": False,
                "rhythm_bias": False, "fine_note_strength": False,
                "local_onset_guidance": False}
    return {
        "conflict_detection": bool(detected["conflict_detection"]),
        "conflict_repair": bool(detected["conflict_repair"]),
        "rhythm_bias": False,
        "fine_note_strength": bool(detected["fine_note_strength"] and not detected["errors"]),
        "local_onset_guidance": bool(_is_event_family(model_family)
                                      and detected.get("local_onset_guidance", False)
                                      and not detected["errors"]),
    }


def _checkpoint_model_family(payload: dict[str, Any]) -> str:
    return str(checkpoint_capabilities(payload)["model_family"])


def _checkpoint_inspection(
    checkpoint_path: Path,
    *,
    mode: str,
    manifest: dict[str, Any] | None,
    model_family: str | None = EVENT_MODEL_FAMILY,
) -> dict[str, Any]:
    try:
        payload = _load_checkpoint_payload(checkpoint_path)
    except ValueError as exc:
        return {
            "valid": False,
            "path": str(checkpoint_path),
            "schema": None,
            "checkpoint_kind": None,
            "model_family": None,
            "can_resume": False,
            "can_init": False,
            "model_config": {},
            "data_contract": {},
            "onset_training": {},
            "warnings": [],
            "errors": [str(exc)],
        }

    detected_family = _checkpoint_model_family(payload)
    capabilities = checkpoint_capabilities(payload)
    if detected_family == "unknown":
        init_errors = [
            f"无法确认 checkpoint schema {payload.get('schema')!r}；拒绝把未知模型当作旧固定帧或 event_v1。"
        ]
        init_warnings = []
        resume_errors = list(init_errors)
        resume_warnings = []
    elif detected_family in EVENT_FAMILIES:
        init_errors: list[str] = []
        init_warnings: list[str] = []
        if not checkpoint_model_config(payload):
            init_errors.append("event checkpoint 缺少 model_config，无法确认事件模型结构")
        stored_contract = checkpoint_contract(payload)
        if not stored_contract:
            init_warnings.append("event checkpoint 未记录 data_contract；音频与事件词表兼容性需由 event CLI 再检查")
        init_errors.extend(capabilities["errors"])
        if stored_contract.get("target_schema") not in {
            None, EVENT_MODEL_FAMILY, EVENT_CTX_MODEL_FAMILY,
            "maichart-event-v1", "maichart-event-ctx-v1",
        }:
            init_errors.append(
                f"event checkpoint 的 target_schema 是 {stored_contract.get('target_schema')!r}，"
                "不属于任何事件模型族，拒绝作为事件模型加载"
            )
        if (detected_family == EVENT_CTX_MODEL_FAMILY
                and capabilities["model_family"] != EVENT_CTX_MODEL_FAMILY):
            init_errors.append("跨窗口 checkpoint 的 data_contract 缺少 target_schema=event_ctx_v1")
        if detected_family == EVENT_CTX_MODEL_FAMILY:
            init_warnings.append(
                "跨窗口模型（event_ctx_v1）：生成时必须使用同样的 context-beats / audio-margin-beats，"
                "否则窗口切分与训练不一致。生成请用 maichart.generation.event_ctx_cli。"
            )
        # Frame division is not part of the event model's training contract.
        preprocessing = (manifest or {}).get("preprocessing") or {}
        for key in ("sample_rate", "feature_profile", "hop_length"):
            if key in preprocessing and key in stored_contract and preprocessing[key] != stored_contract[key]:
                init_errors.append(f"数据合同 {key} 不兼容：checkpoint={stored_contract[key]}，当前={preprocessing[key]}")
        resume_errors = list(init_errors)
        resume_warnings = list(init_warnings)
        if detected_family == EVENT_CTX_MODEL_FAMILY:
            family_params = stored_contract.get("family_params") or {}
            if ("slide_oversample" in family_params and
                    ("touch_oversample" not in family_params or
                     family_params.get("oversample_policy") not in
                     {"independent_note_types_touch_priority_v1",
                      "independent_note_types_touch_priority_fine_rhythm_v2"})):
                resume_errors.append(
                    "checkpoint 使用旧的 Slide/Touch 合并采样合同，不能普通 resume；请改用 finetune 迁移权重并设置新的 Slide/Touch 倍率。"
                )
        for key in ("optimizer_state_dict", "epoch"):
            if key not in payload:
                resume_errors.append(f"断点缺少 {key}，不能恢复完整训练状态；请使用 finetune")
    else:
        data_contract = _manifest_data_contract(manifest) if manifest else None
        init_errors, init_warnings = compare_checkpoint_contract(
            payload, data_contract=data_contract, mode="finetune",
        )
        resume_errors, resume_warnings = compare_checkpoint_contract(
            payload, data_contract=data_contract, mode="resume",
        )
        init_warnings.append("旧固定帧基线：不是 event 模型。使用 event_v1 必须从头重新训练，不能跨 schema 续训或微调。")
        resume_warnings.append(init_warnings[-1])
    if model_family is not None and detected_family != model_family:
        mismatch = (
            f"跨 schema 加载被拒绝：当前选择 {model_family}，checkpoint 属于 {detected_family}。"
            "旧模型不能转成 event 模型；请从头训练 event_v1，或显式选择旧固定帧基线。"
        )
        init_errors.append(mismatch)
        resume_errors.append(mismatch)
    errors = resume_errors if mode == "resume" else init_errors
    warnings = list(dict.fromkeys((resume_warnings if mode == "resume" else init_warnings)))
    return {
        "valid": not bool(errors),
        "path": str(checkpoint_path),
        "schema": payload.get("schema"),
        "checkpoint_kind": (
            "event_training_checkpoint_v1" if detected_family in EVENT_FAMILIES
            else "unknown" if detected_family == "unknown"
            else checkpoint_kind(payload)
        ),
        "model_family": detected_family,
        # Capability report consumed by the generation panel.  Playability
        # post-processing rides on top of any event checkpoint (it runs after
        # decoding), while the sampling-bias flags need a checkpoint trained on
        # the fine-rhythm target.
        "generation_capabilities": _generation_capabilities(detected_family, payload),
        "capability_errors": capabilities["errors"],
        "model_family_label": (
            "event_ctx_v1 跨窗口事件模型" if detected_family == EVENT_CTX_MODEL_FAMILY
            else "event_v1 事件模型" if detected_family == EVENT_MODEL_FAMILY
            else "未知模型族（拒绝加载）" if detected_family == "unknown"
            else "旧固定帧基线（非 event 模型）"
        ),
        "generation_cli": (
            "maichart.generation.event_ctx_cli" if detected_family == EVENT_CTX_MODEL_FAMILY
            else "maichart.generation.event_cli" if detected_family == EVENT_MODEL_FAMILY
            else "maichart.generation.cli" if detected_family == LEGACY_MODEL_FAMILY
            else None
        ),
        "can_resume": not bool(resume_errors),
        "can_init": not bool(init_errors),
        "epoch": payload.get("epoch", 0),
        "global_step": payload.get("global_step", 0),
        "model_config": checkpoint_model_config(payload),
        "data_contract": checkpoint_contract(payload),
        "onset_training": payload.get("onset_training") or {},
        "warnings": warnings,
        "errors": errors,
        "mode": mode,
        "resume_errors": resume_errors,
        "init_errors": init_errors,
    }


def _manifest_data_contract(manifest: dict[str, Any] | None) -> dict[str, Any]:
    preprocessing = (manifest or {}).get("preprocessing") or {}
    return {
        "division": preprocessing.get("division"),
        "sample_rate": preprocessing.get("sample_rate"),
        "feature_profile": "basic_v1",
        "audio_feature_dim": len(AUDIO_FEATURE_KEYS),
        "audio_feature_keys": list(AUDIO_FEATURE_KEYS),
        "condition_keys": list(CONDITION_KEYS),
        "condition_dim": len(CONDITION_KEYS),
        "note_type_vocab": list(DEFAULT_NOTE_TYPES),
        "target_schema": "frame_v2",
    }


def _manifest_processing_errors(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()

    def add(item: dict[str, Any]) -> None:
        key = (
            str(item.get("song_id", "")),
            str(item.get("difficulty_index", "")),
            str(item.get("message", item.get("error", ""))),
        )
        if key not in seen:
            seen.add(key)
            errors.append(item)

    for item in manifest.get("errors") or []:
        if isinstance(item, dict):
            add(item)
    for song in manifest.get("songs") or []:
        if not isinstance(song, dict):
            continue
        for item in song.get("processing_errors") or []:
            if isinstance(item, dict):
                add(item)
        for difficulty in song.get("difficulties") or []:
            if not isinstance(difficulty, dict):
                continue
            item = difficulty.get("processing_error")
            if isinstance(item, dict):
                add({
                    "song_id": song.get("song_id"),
                    "difficulty_index": difficulty.get("difficulty_index"),
                    **item,
                })
    return errors[:100]


def _looks_like_dataset(root: Path) -> bool:
    """Does this folder hold a chart corpus (or an already built manifest)?

    Only the first two levels are probed: the upload layout puts the songs one
    level down, and a manually dropped corpus has them one or two levels down
    (``<version>/<song>/maidata.txt``).  Probing deeper would walk an entire
    corpus on every status request.
    """

    if (root / "training_manifest.json").is_file():
        return True
    for pattern in ("maidata.txt", "*/maidata.txt", "*/*/maidata.txt"):
        if next(root.glob(pattern), None) is not None:
            return True
    return False


def _uploaded_datasets() -> list[dict[str, Any]]:
    """Expose the same discovered and registered library in every UI entry."""
    return _dataset_api().catalog({})[1]["datasets"]


def _resolve_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _required_path(value: Any, label: str, *, must_exist: bool) -> Path:
    if not value:
        raise ValueError(f"缺少 {label} 路径")
    path = _resolve_path(str(value))
    if must_exist and not path.exists():
        raise ValueError(f"{label} 不存在: {path}")
    return path


def _curve_values_equal(left: Any, right: Any) -> bool:
    """Compare curve metadata without rejecting harmless JSON float noise."""

    if left is None or right is None:
        return left is right
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-6)
    except (TypeError, ValueError):
        return left == right


def _curve_value_text(value: Any) -> str:
    if value is None or value == "":
        return "空"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _curve_condition_differences(saved: Any, current: Any) -> list[str]:
    """Return human-readable condition differences for a learned curve.

    The old WebUI collapsed every mismatch into one generic message, which made
    a valid saved curve look broken.  Keep the strict rejection but expose the
    exact field that changed so the user can correct the generation panel.
    """

    if not isinstance(saved, dict):
        return ["曲线缺少 condition 条件快照"]
    if not isinstance(current, dict):
        return ["当前生成条件无效"]
    differences: list[str] = []
    for key, label in (("level", "等级"), ("difficulty", "difficulty")):
        if key in saved and key in current and not _curve_values_equal(saved[key], current[key]):
            differences.append(f"{label} {_curve_value_text(saved[key])}→{_curve_value_text(current[key])}")
        elif key not in saved:
            differences.append(f"曲线缺少 {label} 元数据")
    for section, label in (("radar", "Radar"), ("style", "Style")):
        saved_values = saved.get(section)
        current_values = current.get(section)
        if not isinstance(saved_values, dict) or not isinstance(current_values, dict):
            differences.append(f"{label} 条件快照缺失")
            continue
        for key in sorted(set(saved_values) | set(current_values)):
            if key not in saved_values:
                differences.append(f"{label}.{key} 曲线缺失")
            elif key not in current_values:
                differences.append(f"当前缺少 {label}.{key}")
            elif not _curve_values_equal(saved_values[key], current_values[key]):
                differences.append(
                    f"{label}.{key} {_curve_value_text(saved_values[key])}→{_curve_value_text(current_values[key])}"
                )
    return differences


def _learned_curve_context_differences(curve: dict[str, Any], current_condition: Any,
                                       payload: dict[str, Any], tempo_map_hash: str | None) -> list[str]:
    differences = _curve_condition_differences(curve.get("condition"), current_condition.to_dict())
    if not _curve_values_equal(curve.get("first", 0), payload.get("first", 0)):
        differences.append(f"first {_curve_value_text(curve.get('first', 0))}→{_curve_value_text(payload.get('first', 0))}")
    if tempo_map_hash is None:
        saved_mode = str(curve.get("tempo_mode", "constant"))
        current_mode = str(payload.get("tempo_mode", "constant"))
        if saved_mode != current_mode:
            differences.append(f"节拍模式 {saved_mode}→{current_mode}")
        current_bpm = payload.get("bpm")
        current_bpm = None if current_bpm in {None, ""} else float(current_bpm)
        if "bpm_input" in curve:
            saved_bpm = curve.get("bpm_input")
            if not _curve_values_equal(saved_bpm, current_bpm):
                differences.append(f"BPM {_curve_value_text(saved_bpm)}→{_curve_value_text(current_bpm)}")
        elif current_bpm is not None and not _curve_values_equal(curve.get("bpm"), current_bpm):
            differences.append(f"BPM {_curve_value_text(curve.get('bpm'))}→{_curve_value_text(current_bpm)}")
    return differences


def _safe_filename(value: str) -> str:
    name = PurePosixPath((value or "upload.bin").replace("\\", "/")).name
    name = re.sub(r"[^A-Za-z0-9_.()\-\u4e00-\u9fff ]+", "_", name).strip(" .")
    return name or "upload.bin"


def _resolve_condition_profile(payload: dict[str, Any]) -> Path | None:
    """Find the corpus condition profile for a generation request.

    Explicit path first, then ``condition_profile.json`` next to the training
    manifest the user is working with.  Returns ``None`` when there is none, in
    which case the generator keeps its old 0.5 defaults.
    """

    explicit = payload.get("condition_profile")
    if explicit not in {None, ""}:
        path = _resolve_path(explicit)
        if not path.is_file():
            raise ValueError(f"条件基线档案不存在：{path}")
        return path
    for key in ("manifest", "training_manifest"):
        value = payload.get(key)
        if value in {None, ""}:
            continue
        candidate = _resolve_path(value).parent / "condition_profile.json"
        if candidate.is_file():
            return candidate
    return None


def _preprocess_args(payload: dict[str, Any], *, raw_root: Path, cache_dir: Path, output: Path,
                     model_family: str) -> list[str]:
    """Arguments for the preprocessing stage, shared by the two entry points."""

    if _is_event_family(model_family):
        # Event preprocessing owns its parser/QC and deliberately does not
        # create fixed-frame labels or alignment reports.
        args = [
            "-m", "maichart.training.prepare_events", str(raw_root),
            "--cache-dir", str(cache_dir), "-o", str(output),
            "--sample-rate", str(int(payload.get("sample_rate", 22050))),
        ]
        if payload.get("limit"):
            args.extend(["--limit", str(int(payload["limit"]))])
        if payload.get("force"):
            args.append("--force")
        return args
    args = [
        "-m", "maichart.cli", "preprocess", "build", str(raw_root),
        "--cache-dir", str(cache_dir), "-o", str(output),
        "--division", str(int(payload.get("division", 16))),
        "--sample-rate", str(int(payload.get("sample_rate", 22050))),
    ]
    if payload.get("limit"):
        args.extend(["--limit", str(int(payload["limit"]))])
    if payload.get("skip_audio"):
        args.append("--skip-audio")
    if payload.get("skip_alignment"):
        args.append("--skip-alignment")
    if payload.get("force"):
        args.append("--force")
    return args


def _default_gold_out_dir(manifest: Path) -> Path:
    """Where gold manifests go by default.

    ``<dataset>/gold_set`` -- deliberately *not* ``<dataset>/gold``, which is the
    folder users put their gold charts in (writing the outputs there mixed inputs
    with outputs).  If the manifest already sits in a gold output folder (the UI
    points the training manifest at ``pretrain_manifest.json`` after a build),
    keep writing there instead of nesting another level, which is how
    ``gold/gold/gold`` and its unusable paths came about.
    """

    parent = manifest.parent
    markers = ("gold_val_manifest.json", "gold_finetune_manifest.json", "pretrain_manifest.json")
    if any((parent / name).is_file() for name in markers):
        return parent
    return parent / "gold_set"


def _default_gold_root(manifest: Path, raw_root: Any) -> Path:
    """Where the gold charts live by default: ``<dataset root>/gold``.

    The dataset root is the raw folder the manifest was built from, so adding a
    ``gold`` folder next to the songs is all a user has to do.
    """

    if raw_root not in {None, ""}:
        return _resolve_path(raw_root) / "gold"
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        source_root = data.get("source_root")
    except (OSError, ValueError):
        source_root = None
    if source_root:
        candidate = Path(str(source_root))
        if not candidate.is_absolute():
            candidate = manifest.parent / candidate
        return candidate.resolve() / "gold"
    return manifest.parent / "gold"


def _resolve_seed(payload: dict[str, Any]) -> tuple[int, bool]:
    """Return ``(seed, randomized)`` for a train or generate request.

    An empty or absent seed draws a fresh one, so repeated runs are genuinely
    different draws instead of replaying the hard-coded 42 every time.  Any
    explicit integer is honoured verbatim, which is what makes a good draw
    reproducible.  The resolved value is recorded on the task (and in the
    checkpoint and generation report) so nothing is lost.
    """

    raw = payload.get("seed")
    if raw is None or (isinstance(raw, str) and raw.strip().lower() in {"", "random", "randomize", "rand"}):
        return secrets.randbelow(2**31 - 1), True
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise ValueError(f"随机种子必须是整数，或留空表示每次随机：{raw!r}") from None
    if value < 0:
        raise ValueError("随机种子不能为负数")
    return value, False


def _candidate_path(path: Path, index: int, count: int) -> Path:
    if count == 1:
        return path
    return path.with_name(f"{path.stem}_candidate_{index + 1:03d}{path.suffix}")


def _multipart_boundary(content_type: str) -> bytes:
    match = re.search(r"boundary=(?:\"([^\"]+)\"|([^;]+))", content_type)
    if not match:
        raise ValueError("上传请求缺少 multipart boundary")
    return (match.group(1) or match.group(2)).strip().encode("utf-8")


_MULTIPART_CHUNK = 1024 * 1024
_MULTIPART_FIELD_MAX = 256 * 1024
_MULTIPART_HEADER_MAX = 64 * 1024


def _parse_multipart_stream(rfile, boundary: bytes, length: int, tmpdir: Path) -> dict[str, Any]:
    """Parse ``multipart/form-data`` without holding the body in memory.

    File parts are written straight into ``tmpdir`` and returned as
    ``(filename, Path)``; ordinary form fields stay in memory.  Buffering the
    whole body (the previous implementation) made large dataset uploads fail
    in the browser with "Failed to fetch" because the server stalled while
    gigabytes accumulated in RAM.
    """

    delimiter = b"\r\n--" + boundary
    opening = b"--" + boundary
    fields: dict[str, Any] = {}
    buf = bytearray()
    remaining = int(length)

    def fill() -> bool:
        nonlocal remaining
        if remaining <= 0:
            return False
        chunk = rfile.read(min(_MULTIPART_CHUNK, remaining))
        if not chunk:
            return False
        buf.extend(chunk)
        remaining -= len(chunk)
        return True

    # Skip the preamble and consume the first boundary line.
    while True:
        index = buf.find(opening)
        if index >= 0:
            del buf[:index + len(opening)]
            break
        excess = len(buf) - (len(opening) - 1)
        if excess > 0:
            del buf[:excess]
        if not fill():
            raise ValueError("上传数据被截断：找不到 multipart 起始边界")

    while True:
        # Right after a boundary line: "--" ends the body, otherwise a CRLF
        # precedes the next part's headers.
        while len(buf) < 2:
            if not fill():
                raise ValueError("上传数据被截断：boundary 之后没有数据")
        if buf.startswith(b"--"):
            break
        if buf.startswith(b"\r\n"):
            del buf[:2]
        while b"\r\n\r\n" not in buf:
            if len(buf) > _MULTIPART_HEADER_MAX:
                raise ValueError("multipart part 头过大")
            if not fill():
                raise ValueError("上传数据被截断：part 头不完整")
        header_bytes, _, rest = bytes(buf).partition(b"\r\n\r\n")
        buf = bytearray(rest)
        headers = BytesParser(policy=email_default).parsebytes(header_bytes + b"\r\n\r\n")
        disposition = str(headers.get("Content-Disposition", ""))
        name_match = re.search(r'name="([^"]+)"', disposition)
        filename_match = re.search(r'filename="([^"]*)"', disposition)
        name = name_match.group(1) if name_match else None
        keep_tail = len(delimiter) - 1

        if filename_match and name is not None:
            destination = tmpdir / f"part-{len(fields):04d}-{_safe_filename(filename_match.group(1))}"
            with open(destination, "wb") as stream:
                while True:
                    index = buf.find(delimiter)
                    if index >= 0:
                        stream.write(buf[:index])
                        del buf[:index + len(delimiter)]
                        break
                    if not fill():
                        raise ValueError("上传数据被截断：文件部分不完整")
                    # A delimiter can arrive wholly inside the bytes just read,
                    # so search again before flushing anything.
                    index = buf.find(delimiter)
                    if index >= 0:
                        stream.write(buf[:index])
                        del buf[:index + len(delimiter)]
                        break
                    # Flush everything that cannot be the head of a delimiter
                    # straddling the chunk edge; keep only a short tail.
                    if len(buf) > keep_tail:
                        stream.write(buf[:len(buf) - keep_tail])
                        del buf[:len(buf) - keep_tail]
            fields[name] = (filename_match.group(1), destination)
        else:
            data = bytearray()
            while True:
                index = buf.find(delimiter)
                if index >= 0:
                    data += buf[:index]
                    del buf[:index + len(delimiter)]
                    break
                if not fill():
                    raise ValueError("上传数据被截断：表单字段不完整")
                index = buf.find(delimiter)
                if index >= 0:
                    data += buf[:index]
                    del buf[:index + len(delimiter)]
                    break
                if len(buf) > keep_tail:
                    data += buf[:len(buf) - keep_tail]
                    del buf[:len(buf) - keep_tail]
                if len(data) > _MULTIPART_FIELD_MAX:
                    raise ValueError("multipart 表单字段过大")
            if name is not None:
                fields[name] = bytes(data).decode("utf-8", errors="replace")

    # Drain the trailer (usually just the final CRLF) so the socket is clean.
    while remaining > 0:
        chunk = rfile.read(min(_MULTIPART_CHUNK, remaining))
        if not chunk:
            break
        remaining -= len(chunk)
    return fields


def _extract_dataset_zip(source: Path, final_root: Path) -> Path:
    """Extract into a staging directory, then rename it into place atomically.

    The staging name (``.tmp-<id>``) never matches the ``[a-f0-9]{8}`` pattern
    the UI lists, so a half-extracted dataset can never be selected for
    preprocessing; the rename is the single moment the dataset becomes usable.
    """

    final_root.parent.mkdir(parents=True, exist_ok=True)
    staging = final_root.parent / f".tmp-{final_root.name}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        staging_root = staging.resolve()
        with zipfile.ZipFile(source) as archive:
            for member in archive.infolist():
                destination = (staging / member.filename).resolve()
                if not _is_relative_to(destination, staging_root):
                    raise ValueError("数据集压缩包包含越界路径")
            archive.extractall(staging)
        if final_root.exists():
            shutil.rmtree(final_root)
        staging.rename(final_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return final_root


_EXTRACTIONS_LOCK = threading.Lock()
_EXTRACTIONS: dict[str, dict[str, Any]] = {}
_EXTRACTION_KEEP_SECONDS = 3600


def _start_extraction(stored: Path, final_root: Path, upload_id: str) -> None:
    """Extract in a background thread; the UI polls /api/status for progress.

    Extraction of a multi-GB ZIP takes minutes.  Doing it inside the request
    (the previous behaviour) kept the browser waiting with no data flowing,
    which is another way large uploads die with "Failed to fetch".
    """

    with _EXTRACTIONS_LOCK:
        stale = [
            key for key, value in _EXTRACTIONS.items()
            if value.get("status") in {"done", "failed"}
            and time.time() - float(value.get("finished", 0.0)) > _EXTRACTION_KEEP_SECONDS
        ]
        for key in stale:
            _EXTRACTIONS.pop(key, None)
        _EXTRACTIONS[upload_id] = {
            "status": "extracting", "name": stored.name,
            "path": str(final_root), "started": time.time(),
        }

    def _worker() -> None:
        try:
            _extract_dataset_zip(stored, final_root)
            with _EXTRACTIONS_LOCK:
                _EXTRACTIONS[upload_id].update(status="done", finished=time.time())
        except BaseException as exc:  # reported verbatim in /api/status
            with _EXTRACTIONS_LOCK:
                _EXTRACTIONS[upload_id].update(status="failed", error=str(exc), finished=time.time())

    threading.Thread(target=_worker, name=f"extract-{upload_id}", daemon=True).start()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
