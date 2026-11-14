"""Read published audit coverage and history independently of training tasks."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from maichart.training.dataset_catalog import _snapshot_stamp
from maichart.training.quality_annotations import resolve_auto_audit

from .dataset_api import _json_data


def audit_state_path(manifest, preview):
    return manifest.with_name("auto_quality_audit_preview_state.json" if preview else "auto_quality_audit_state.json")


def audit_state(manifest, preview):
    try:
        state = _json_data(audit_state_path(manifest, preview))
    except (FileNotFoundError, ValueError):
        return None
    if state.get("manifest") != str(manifest):
        return None
    return task_state(state)


def task_state(state):
    if state.get("status") in {"queued", "running"} and state.get("pid"):
        try:
            _check_process(int(state["pid"]))
        except (TypeError, ValueError, ProcessLookupError):
            return dict(state, status="interrupted")
        except PermissionError:
            pass
    return state


def _check_process(pid):
    if pid <= 0:
        raise ValueError("invalid pid")
    if os.name != "nt":
        os.kill(pid, 0)
        return
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = kernel.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        if ctypes.get_last_error() == 87:  # ERROR_INVALID_PARAMETER, no such process
            raise ProcessLookupError(pid)
        return
    try:
        code = wintypes.DWORD()
        if kernel.GetExitCodeProcess(handle, ctypes.byref(code)) and code.value != 259:  # STILL_ACTIVE
            raise ProcessLookupError(pid)
    finally:
        kernel.CloseHandle(handle)


def audit_task_lock(preview):
    return ".materials-audit-preview-task.lock" if preview else ".materials-audit-task.lock"


def report_summary(report):
    if report is None:
        return None
    from collections import Counter

    detail_fields = {"audited_chart_ids", "audited_chart_audio_stats", "audited_chart_audio_paths",
                     "unverified_charts", "review_regions"}
    summary = {key: value for key, value in report.items() if key not in detail_fields}
    if "unverified_by_reason" not in summary:
        summary["unverified_by_reason"] = dict(Counter(
            str(item.get("reason") or "unverified") for item in (report.get("unverified_charts") or [])
            if isinstance(item, dict)
        ))
    if not summary.get("created_at") and report.get("report_path"):
        try:
            summary["created_at"] = datetime.fromtimestamp(Path(report["report_path"]).stat().st_mtime, timezone.utc).isoformat()
        except OSError:
            pass
    if "review_regions" in report:
        summary["reported_prompts"] = sum(len(row.get("regions", ())) for row in report["review_regions"])
    return summary


def published_audits(manifest, data):
    formal_path, formal, formal_report = resolve_auto_audit(manifest, data)
    _, preview, preview_report = resolve_auto_audit(manifest, data, preview=True)
    inherited = (formal or {}).get("audit_metadata", {}).get("composed_sources") or []
    return dict(
        available=any(item is not None for item in (formal, preview, formal_report, preview_report)),
        manifest=str(manifest), has_annotations=formal is not None, has_preview_annotations=preview is not None,
        annotations_path=str(formal_path) if formal_path else None,
        record_count=len((formal or {}).get("records", ())),
        report=report_summary(formal_report), preview_report=report_summary(preview_report),
        formal_task=audit_state(manifest, False), preview_task=audit_state(manifest, True),
        inherited_auto_sources=[item.get("source_manifest") for item in inherited],
    ), formal, formal_report, preview, preview_report


def _selection_key(paths):
    return frozenset(Path(path).resolve() for path in paths)


def _has_audit_files(manifest):
    return any(manifest.with_name(name).is_file() for name in (
        "auto_quality_audit_report.json", "auto_quality_audit_preview_report.json",
        "auto_quality_audit_state.json", "auto_quality_audit_preview_state.json",
    ))


def selection_audits(manifests, workspace):
    """Find existing compositions without preparing or writing any snapshot."""
    paths = _selection_key(manifests)
    current = None
    historic = {"report": None, "preview_report": None}
    tasks = {"formal_task": None, "preview_task": None}
    for path in (Path(workspace) / "composed_manifests").glob("composed_*/training_manifest.json"):
        try:
            versions_path = path.with_name("source_versions.json")
            if not versions_path.is_file() and not _has_audit_files(path):
                continue
            data = _json_data(path)
            if not data.get("merged_from") or _selection_key(data["merged_from"]) != paths:
                continue
            versions = _json_data(versions_path) if versions_path.is_file() else {}
            meta, *_ = published_audits(path, data)
            for name in historic:
                report = meta[name]
                if report and (historic[name] is None or report.get("created_at", "") > historic[name][name].get("created_at", "")):
                    historic[name] = meta
            for name in tasks:
                state = meta[name]
                if state and (tasks[name] is None or state.get("started_at", "") > tasks[name].get("started_at", "")):
                    tasks[name] = state
            if versions.get("model_family") == "chart_composer_v2" and versions.get("files") and all(
                _snapshot_stamp(file) == stamp for group in ("files", "assets")
                for file, stamp in versions.get(group, {}).items()
            ):
                if current is None or versions_path.stat().st_mtime_ns > current[0]:
                    current = (versions_path.stat().st_mtime_ns, meta)
        except (OSError, TypeError, ValueError, KeyError):
            continue
    return dict(
        current_snapshot=current[1] if current else None,
        historical_formal=historic["report"], historical_preview=historic["preview_report"],
        **tasks,
    )
