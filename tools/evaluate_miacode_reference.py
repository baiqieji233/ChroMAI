"""Run the headless MiaCode Muri reference dumper and preserve raw diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


REFERENCE_COMMIT = "6573efb611056bccd046dd17bc2f4cceda78bf88"
REPORT_SCHEMA = "miacode-muri-engine-reference-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _validate_paths(executable: Path, maidata_paths: Sequence[Path], output: Path) -> None:
    output_resolved = output.resolve()
    protected = {executable.resolve(), *(path.resolve() for path in maidata_paths)}
    if output_resolved in protected:
        raise ValueError("--output must not overwrite the executable or any maidata input.")


def _validate_driver_report(raw: Any) -> tuple[dict[str, Any] | None, list[str]]:
    if not isinstance(raw, dict):
        return None, ["driver JSON root must be an object"]
    errors = []
    if not isinstance(raw.get("ok"), bool):
        errors.append("driver schema error: root.ok must be boolean")
    parse = raw.get("parse")
    if not isinstance(parse, dict):
        errors.append("driver schema error: parse must be an object")
    else:
        if not isinstance(parse.get("ok"), bool):
            errors.append("driver schema error: parse.ok must be boolean")
        count = parse.get("error_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            errors.append("driver schema error: parse.error_count must be a nonnegative integer")
        for key in ("errors", "warnings"):
            if not isinstance(parse.get(key), list):
                errors.append(f"driver schema error: parse.{key} must be an array")
        warning_count = parse.get("warning_count")
        if isinstance(warning_count, bool) or not isinstance(warning_count, int) or warning_count < 0:
            errors.append("driver schema error: parse.warning_count must be a nonnegative integer")
    for key in ("diagnostics", "static_references", "marker_states"):
        if not isinstance(raw.get(key), list):
            errors.append(f"driver schema error: {key} must be an array")
    if not isinstance(raw.get("summary"), dict):
        errors.append("driver schema error: summary must be an object")
    return raw, errors


def _parse_failed(raw: dict[str, Any]) -> bool:
    parse = raw["parse"]
    return (raw["ok"] is False or parse["ok"] is False or parse["error_count"] > 0
            or bool(parse["errors"]))


def _run_chart(executable: Path, source: Path, difficulty: int, first: float | None,
               timeout_sec: float, executable_sha256: str) -> dict[str, Any]:
    item: dict[str, Any] = {
        "input_path": str(source), "input_sha256": None,
        "input_sha256_after": None, "difficulty": difficulty,
        "first_override": first, "argv": None, "exitcode": None,
        "timed_out": False, "raw_stdout": None, "stderr": None, "raw": None,
        "parse": None, "errors": None, "diagnostics": None,
        "static_references": None, "marker_states": None, "summary": None,
        "executable_sha256": executable_sha256, "engine_reference_only": True,
        "playable_verified": False, "status": "failed", "error": None,
    }
    errors: list[str] = []
    try:
        if not source.is_file():
            raise FileNotFoundError(f"maidata input does not exist: {source}")
        item["input_sha256"] = _sha256(source)
        argv = [str(executable), "--maidata", str(source), "--difficulty", str(difficulty), "--with-markers"]
        if first is not None:
            argv.extend(["--first", format(first, ".17g")])
        item["argv"] = argv
        completed = subprocess.run(
            argv, shell=False, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout_sec, check=False,
        )
        item["exitcode"] = int(completed.returncode)
        item["raw_stdout"] = completed.stdout
        item["stderr"] = completed.stderr
        if completed.returncode != 0:
            errors.append(f"driver exited with status {completed.returncode}")
        try:
            decoded = json.loads(completed.stdout)
        except (json.JSONDecodeError, TypeError) as exc:
            errors.append(f"driver stdout is not valid JSON: {exc}")
            decoded = None
        raw, schema_errors = _validate_driver_report(decoded) if decoded is not None else (None, [])
        errors.extend(schema_errors)
        if raw is not None:
            item["raw"] = raw
            for key in ("parse", "diagnostics", "static_references", "marker_states", "summary"):
                item[key] = raw.get(key)
            item["errors"] = raw["parse"].get("errors") if isinstance(raw.get("parse"), dict) else None
            if not schema_errors and _parse_failed(raw):
                errors.append("engine root/parse status failed or parser errors were reported")
        item["input_sha256_after"] = _sha256(source)
        if item["input_sha256_after"] != item["input_sha256"]:
            errors.append("maidata input changed while the reference driver was running")
    except subprocess.TimeoutExpired as exc:
        item["timed_out"] = True
        item["raw_stdout"] = _decode_partial(exc.stdout)
        item["stderr"] = _decode_partial(exc.stderr)
        errors.append(f"driver timed out after {timeout_sec:g} seconds")
        try:
            if source.is_file():
                item["input_sha256_after"] = _sha256(source)
                if item["input_sha256"] and item["input_sha256_after"] != item["input_sha256"]:
                    errors.append("maidata input changed while the reference driver was running")
        except OSError as hash_exc:
            errors.append(f"could not hash input after timeout: {hash_exc}")
    except (OSError, ValueError) as exc:
        errors.append(str(exc))
    item["error"] = errors or None
    item["status"] = "completed" if not errors else "failed"
    return item


def _decode_partial(value: bytes | str | None) -> str | None:
    if value is None:
        return None
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def evaluate(executable: str | Path, maidata_paths: Sequence[str | Path], *, difficulty: int = 5,
             first: float | None = None, output: str | Path, timeout_sec: float = 30.0) -> dict[str, Any]:
    executable_path = _canonical_path(executable)
    sources = [_canonical_path(path) for path in maidata_paths]
    output_path = _canonical_path(output)
    if not sources:
        raise ValueError("At least one --maidata input is required.")
    if isinstance(difficulty, bool) or int(difficulty) != difficulty or not 1 <= int(difficulty) <= 7:
        raise ValueError("difficulty must be an integer in 1..7.")
    if first is not None and not math.isfinite(float(first)):
        raise ValueError("first override must be finite.")
    if not math.isfinite(float(timeout_sec)) or not 0 < float(timeout_sec) <= 60:
        raise ValueError("timeout-sec must be finite and in (0, 60].")
    _validate_paths(executable_path, sources, output_path)
    if not executable_path.is_file():
        raise FileNotFoundError(f"headless reference executable does not exist: {executable_path}")
    executable_before = _sha256(executable_path)
    charts = [_run_chart(executable_path, source, int(difficulty),
                         None if first is None else float(first), float(timeout_sec), executable_before)
              for source in sources]
    try:
        executable_after = _sha256(executable_path)
    except OSError:
        executable_after = None
    executable_changed = executable_after != executable_before
    if executable_changed:
        for chart in charts:
            changed_error = "reference executable changed while it was running"
            chart["error"] = [*(chart["error"] or []), changed_error]
            chart["status"] = "failed"
    report = {
        "schema": REPORT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reference": {
            "name": "MiaCode Muri headless reference dumper",
            "declared_commit": REFERENCE_COMMIT,
            "commit_verification": "declared_only",
            "engine_reference_only": True,
            "playable_verified": False,
        },
        "invocation": {
            "executable": str(executable_path),
            "executable_sha256": executable_before,
            "executable_sha256_after": executable_after,
            "difficulty": int(difficulty), "first_override": first,
            "with_markers": True, "timeout_sec": float(timeout_sec),
        },
        "summary": {
            "input_count": len(charts),
            "completed_count": sum(chart["status"] == "completed" for chart in charts),
            "error_count": sum(chart["status"] != "completed" for chart in charts),
        },
        "charts": charts,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True,
                        help="User-provided headless miacode_muri_dump developer build.")
    parser.add_argument("--maidata", type=Path, nargs="+", action="extend", required=True,
                        help="One or more maidata source files; inputs are read-only.")
    parser.add_argument("--difficulty", type=int, default=5, choices=range(1, 8))
    parser.add_argument("--first", type=float, help="Optional explicit first/audio-clock override.")
    parser.add_argument("--output", type=Path, required=True, help="Independent JSON report path.")
    parser.add_argument("--timeout-sec", type=float, default=30.0,
                        help="Per-chart subprocess timeout in (0, 60] seconds.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = evaluate(args.executable, args.maidata, difficulty=args.difficulty,
                          first=args.first, output=args.output, timeout_sec=args.timeout_sec)
    except (OSError, ValueError) as exc:
        print(f"miacode_reference_error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {args.output}: completed={report['summary']['completed_count']} "
          f"errors={report['summary']['error_count']}")
    return 0 if report["summary"]["error_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
