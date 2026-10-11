"""Replay generation's embedded detector/repair on an existing chart, without a model."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from maichart.event_timing import parse_event_chart
from maichart.generation.event_compile import compile_event_points
from maichart.generation.miacode_native import ensure_available
from maichart.generation.post_process import PostProcessConfig, process_chart_points


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("maidata", type=Path)
    parser.add_argument("--difficulty", type=int, default=5)
    parser.add_argument("--mode", choices=("none", "local", "reject"), default="none")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.maidata.resolve()
    output = args.output_dir.resolve()
    if source.parent == output:
        raise ValueError("Use a separate output directory to preserve the input")
    output.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256(source.read_bytes()).hexdigest()
    chart = parse_event_chart(source, args.difficulty)
    ensure_available()
    print(f"playability_engine engine=miacode mode={args.mode} input={source}", flush=True)
    started = time.perf_counter()
    result = process_chart_points(chart.points, chart.tempo_changes, chart.first,
        PostProcessConfig(True, detect_soft_conflicts=True, conflict_repair_mode=args.mode,
                          conflict_engine="miacode"))
    for issue in result.conflicts_detected:
        print("playability_issue " + json.dumps(issue.to_dict(), ensure_ascii=False), flush=True)
    summary = {"source": str(source), "source_sha256": identity, "mode": args.mode,
               "engine": "miacode_native", "in_process": True,
               "elapsed_sec": time.perf_counter() - started,
               "before": dict(Counter(c.conflict_type for c in result.conflicts_detected)),
               "after": dict(Counter(c.conflict_type for c in result.remaining_conflicts)),
               "removed": len(result.repairs_applied),
               "remaining_hard": sum(c.severity == "hard" for c in result.remaining_conflicts),
               "playable_verified": False}
    (output / "report.json").write_text(json.dumps({"summary": summary, "playability": result.to_dict()},
        ensure_ascii=False, indent=2), encoding="utf-8")
    if args.mode == "local":
        text, _ = compile_event_points(result.points, tempo_changes=chart.tempo_changes,
            first=chart.first, title=chart.title, level=chart.level,
            difficulty=chart.difficulty, artist=chart.artist)
        (output / "repaired.maidata.txt").write_text(text, encoding="utf-8")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == identity
    print("playability_final " + json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if result.success else 2


if __name__ == "__main__":
    raise SystemExit(main())
