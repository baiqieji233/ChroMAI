"""Optional external MiaCode runtime-reference checks for generated charts."""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any

REFERENCE_COMMIT = "6573efb611056bccd046dd17bc2f4cceda78bf88"
REPORT_SCHEMA = "miacode-runtime-reference-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate(raw: Any) -> tuple[dict[str, Any] | None, list[str]]:
    if not isinstance(raw, dict):
        return None, ["driver JSON root must be an object"]
    errors: list[str] = []
    if not isinstance(raw.get("ok"), bool):
        errors.append("driver schema error: root.ok must be boolean")
    parse = raw.get("parse")
    if not isinstance(parse, dict):
        errors.append("driver schema error: parse must be an object")
    else:
        if not isinstance(parse.get("ok"), bool):
            errors.append("driver schema error: parse.ok must be boolean")
        if not isinstance(parse.get("error_count"), int) or isinstance(parse.get("error_count"), bool):
            errors.append("driver schema error: parse.error_count must be integer")
        if not isinstance(parse.get("errors"), list) or not isinstance(parse.get("warnings"), list):
            errors.append("driver schema error: parse errors/warnings must be arrays")
    for key in ("diagnostics", "static_references", "marker_states"):
        if not isinstance(raw.get(key), list):
            errors.append(f"driver schema error: {key} must be an array")
    if not isinstance(raw.get("summary"), dict):
        errors.append("driver schema error: summary must be an object")
    return raw, errors


def run_reference(
    executable: str | Path,
    maidata_text: str,
    *,
    difficulty: int,
    first: float,
    timeout_sec: float = 30.0,
    mode: str = "report",
) -> dict[str, Any]:
    """Run the user-provided reference executable against one complete maidata text.

    The temporary input is deleted after the subprocess exits. The result keeps
    raw driver fields and never claims hardware playability.
    """
    if mode not in {"report", "reject"}:
        raise ValueError("runtime reference mode must be report or reject")
    executable_path = Path(executable).expanduser().resolve()
    if not executable_path.is_file():
        raise FileNotFoundError(f"runtime reference executable does not exist: {executable_path}")
    if not math.isfinite(float(timeout_sec)) or not 0 < float(timeout_sec) <= 60:
        raise ValueError("runtime-reference-timeout-sec must be finite and in (0, 60].")
    before_executable = _sha256(executable_path)
    result: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "mode": mode,
        "status": "failed",
        "reference": {"name": "MiaCode Muri headless reference dumper",
                       "declared_commit": REFERENCE_COMMIT,
                       "commit_verification": "declared_only",
                       "engine_reference_only": True, "playable_verified": False},
        "invocation": {"executable": str(executable_path),
                        "executable_sha256": before_executable,
                        "difficulty": int(difficulty), "first": float(first),
                        "timeout_sec": float(timeout_sec), "with_markers": True,
                        "argv": None},
        "input": {"temp_source": True, "sha256_before": None, "sha256_after": None},
        "parse": None, "summary": None, "diagnostics": None,
        "static_references": None, "marker_states": None,
        "raw": None, "errors": [], "reject_reason": None,
    }
    with tempfile.TemporaryDirectory(prefix="maichart-muri-") as directory:
        source = Path(directory) / "generated.maidata.txt"
        source.write_text(str(maidata_text), encoding="utf-8", newline="\n")
        result["input"]["sha256_before"] = _sha256(source)
        argv = [str(executable_path), "--maidata", str(source),
                "--difficulty", str(int(difficulty)), "--first", format(float(first), ".17g"),
                "--with-markers"]
        result["invocation"]["argv"] = argv
        try:
            completed = subprocess.run(argv, shell=False, capture_output=True, text=True,
                                       encoding="utf-8", errors="replace",
                                       timeout=float(timeout_sec), check=False)
            if completed.returncode != 0:
                result["errors"].append(f"driver exited with status {completed.returncode}")
            try:
                decoded = json.loads(completed.stdout)
            except (TypeError, json.JSONDecodeError) as exc:
                decoded = None
                result["errors"].append(f"driver stdout is not valid JSON: {exc}")
            raw, schema_errors = _validate(decoded) if decoded is not None else (None, [])
            result["errors"].extend(schema_errors)
            if raw is not None:
                result["raw"] = raw
                for key in ("parse", "summary", "diagnostics", "static_references", "marker_states"):
                    result[key] = raw.get(key)
                parse = raw.get("parse") or {}
                if (raw.get("ok") is False or parse.get("ok") is False
                        or parse.get("error_count", 0) > 0 or parse.get("errors")):
                    result["errors"].append("reference root/parse status failed or parser errors were reported")
            if completed.stderr:
                result["stderr"] = completed.stderr
        except subprocess.TimeoutExpired as exc:
            result["timed_out"] = True
            result["errors"].append(f"driver timed out after {timeout_sec:g} seconds")
            result["stderr"] = str(exc.stderr) if exc.stderr else None
        except (OSError, TimeoutError) as exc:
            result["errors"].append(str(exc))
        result["input"]["sha256_after"] = _sha256(source)
        if result["input"]["sha256_after"] != result["input"]["sha256_before"]:
            result["errors"].append("temporary maidata changed while reference driver ran")
    after_executable = _sha256(executable_path)
    result["invocation"]["executable_sha256_after"] = after_executable
    if after_executable != before_executable:
        result["errors"].append("reference executable changed while it was running")
    diagnostics = result.get("diagnostics") if isinstance(result.get("diagnostics"), list) else []
    alert_counts: dict[str, int] = {}
    for diagnostic in diagnostics:
        if isinstance(diagnostic, dict):
            level = str(diagnostic.get("alert_level", "unknown"))
            alert_counts[level] = alert_counts.get(level, 0) + 1
    summary = dict(result.get("summary") or {})
    summary.setdefault("diagnostic_count", len(diagnostics))
    summary["alert_counts"] = alert_counts
    summary.setdefault("static_reference_count", len(result.get("static_references") or []))
    summary.setdefault("visible_panel_count", len((result.get("raw") or {}).get("visible_panel_entries") or []))
    result["summary"] = summary
    if result["errors"]:
        result["status"] = "failed"
        result["reject_reason"] = "reference_failed" if mode == "reject" else None
    else:
        muri = [item for item in diagnostics
                if isinstance(item, dict) and item.get("alert_level") == "Muri"]
        if mode == "reject" and muri:
            result["status"] = "rejected"
            result["reject_reason"] = "muri_alert"
        else:
            result["status"] = "completed"
    return result

