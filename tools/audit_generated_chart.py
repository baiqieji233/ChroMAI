"""Print exact-event diagnostics for a Maidata file, without generation.

Usage: python tools/audit_generated_chart.py maidata.txt --checkpoint model.pt
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

# Support direct execution from a source checkout without an editable install.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maichart.generation.conditions import GenerationCondition
from maichart.generation.event_quality import assess_event_chart
from maichart.maidata import parse_maidata_file
from maichart.timing import _parse_point_directives, _split_top_level_commas


@dataclass(frozen=True, slots=True)
class _AuditPoint:
    beat: Fraction
    note_text: str


def _events_from_inote(raw: str, initial_bpm: float) -> tuple[list[_AuditPoint], list[tuple[Fraction, float]]]:
    """Reuse timing grammar but never call its integer-tick conversion.

    This duck-type adapter can audit charts before event_timing is available.
    The final empty point preserves the chart's explicit/inferred end beat.
    """
    if not math.isfinite(initial_bpm) or initial_bpm <= 0:
        raise ValueError("Initial BPM must be finite and positive.")
    bpm, division, beat = initial_bpm, 4, Fraction(0)
    points, changes = [], []
    chunks = _split_top_level_commas(raw)
    for index, chunk in enumerate(chunks):
        directives, text = _parse_point_directives(chunk)
        end = False
        for directive in directives:
            if directive.directive_type == "bpm":
                bpm = float(directive.value)
            elif directive.directive_type == "grid":
                division = int(directive.value)
            elif directive.directive_type == "end":
                end = True
        if not changes or changes[-1][1] != bpm:
            changes.append((beat, bpm))
        points.append(_AuditPoint(beat, text))
        if end:
            if any(chunk.strip() for chunk in chunks[index + 1:]):
                raise ValueError("Nonempty timing data follows the E end marker.")
            break
        beat += Fraction(4, division)
    else:
        points.append(_AuditPoint(beat, ""))
    return points, changes


def _json_metadata(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_metadata(item) for item in value]
    if hasattr(value, "numel") and value.numel() == 1:
        return _json_metadata(value.item())
    return {"type": type(value).__name__, "shape": list(getattr(value, "shape", ()))}


def checkpoint_summary(path: str | Path, *, recent: int = 5) -> dict[str, Any]:
    """Read checkpoint metadata only; never construct or run a model.

    Safe weights-only loading deliberately has no unrestricted-pickle fallback.
    Missing paradigm/history fields are reported as unavailable, not inferred.
    """
    import torch

    load_error = None
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping):
            raise ValueError("Checkpoint must contain a mapping.")
    except Exception as exc:  # noqa: BLE001 - audit can still use sidecar metrics
        payload = {}
        load_error = str(exc)
    model = payload.get("model_config") or {}
    contract = payload.get("data_contract") or {}
    paradigm = payload.get("paradigm", payload.get("training_paradigm"))
    if paradigm is None:
        paradigm = model.get("paradigm", contract.get("paradigm"))
    history = {}
    for key in ("paradigm_stats", "paradigm_statistics", "paradigm_history",
                "recent_paradigm_statistics", "training_history", "history", "metrics"):
        if key in payload:
            records = payload[key]
            history[key] = _json_metadata(records[-recent:] if isinstance(records, (list, tuple)) else records)
    metrics_path = Path(path).resolve().parent / "metrics.jsonl"
    if metrics_path.is_file():
        records = []
        for line_number, line in enumerate(metrics_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid metrics JSON at {metrics_path}:{line_number}: {exc}") from exc
            if isinstance(record, Mapping):
                records.append(record)
        history["metrics.jsonl"] = _json_metadata(records[-recent:])
    elif isinstance(payload.get("metrics"), Mapping):
        history["checkpoint.metrics"] = _json_metadata(payload["metrics"])
    result = {"path": str(Path(path).resolve()), "schema": payload.get("schema"),
            "epoch": payload.get("epoch"), "global_step": payload.get("global_step"),
            "paradigm": _json_metadata(paradigm),
            "paradigm_available": paradigm is not None,
            "recent_statistics": history, "recent_limit": recent,
            "statistics_available": bool(history),
            "note": "Stored checkpoint metadata only; no model inference or quality ranking."}
    if load_error is not None:
        result["load_error"] = load_error
        result["note"] += " Checkpoint payload was not safely deserialized; sidecar metrics were still inspected."
    return result


def _level(text: str | None) -> float:
    if not text:
        return 13.0
    return float(text[:-1]) + 0.5 if text.endswith("+") else float(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("maidata", type=Path)
    parser.add_argument("--difficulty", type=int, choices=range(1, 7))
    parser.add_argument("--duration-sec", type=float)
    parser.add_argument("--target-condition", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--recent", type=int, default=5)
    args = parser.parse_args(argv)
    if args.recent <= 0:
        parser.error("--recent must be positive.")
    result: dict[str, Any] = {"maidata": str(args.maidata.resolve()), "charts": []}
    try:
        chart = parse_maidata_file(args.maidata)
        result["title"] = chart.title
        target = GenerationCondition.from_json(args.target_condition) if args.target_condition else None
        selected = [block for block in chart.difficulties if block.inote is not None and
                    (args.difficulty is None or block.index == args.difficulty)]
        if not selected:
            raise ValueError("No matching inote difficulty blocks.")
        for block in selected:
            try:
                points, tempo = _events_from_inote(block.inote, float(chart.wholebpm or 120))
                report = assess_event_chart(points, tempo, first=float(chart.first or 0),
                                            duration_sec=args.duration_sec, target_condition=target)
                # The source level is metadata, not an estimated actual difficulty.
                report["source_level"] = _level(block.level)
            except (ValueError, TypeError, OverflowError, ZeroDivisionError) as exc:
                report = {"syntax_valid": False, "quality_status": "needs_review",
                          "playable_verified": False, "fatal_errors": [
                              {"code": "invalid_maidata_timing", "message": str(exc)}]}
            result["charts"].append({"difficulty": block.index, **report})
        if args.checkpoint:
            try:
                result["checkpoint"] = checkpoint_summary(args.checkpoint, recent=args.recent)
            except Exception as exc:
                result["checkpoint"] = {"path": str(args.checkpoint.resolve()), "error": str(exc)}
    except (OSError, ValueError, TypeError) as exc:
        result["error"] = str(exc)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    if result.get("error") or result.get("checkpoint", {}).get("load_error"):
        return 2
    return int(any(not chart["syntax_valid"] or chart["quality_status"] == "degenerate"
                   for chart in result["charts"]))


if __name__ == "__main__":
    raise SystemExit(main())
