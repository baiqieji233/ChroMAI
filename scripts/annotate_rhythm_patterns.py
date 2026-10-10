"""Write a read-only rhythm statistics report for a training manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maichart.training.rhythm_report import analyze_training_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="training_manifest.json or its dataset directory")
    parser.add_argument("--output", help="report JSON path (defaults to <manifest>.rhythm_report.json)")
    parser.add_argument("--window-beats", type=int, default=16, help="window length in beats (default: 16)")
    parser.add_argument("--difficulty", type=int, action="append", dest="difficulties",
                        help="difficulty index to include; repeat to select several")
    parser.add_argument("--min-level", type=float, help="minimum chart level, inclusive")
    parser.add_argument("--max-level", type=float, help="maximum chart level, inclusive")
    parser.add_argument("--limit", type=int, help="maximum eligible chart difficulties to analyze")
    args = parser.parse_args(argv)
    try:
        report = analyze_training_manifest(
            args.dataset,
            args.output,
            window_beats=args.window_beats,
            difficulties=set(args.difficulties) if args.difficulties else None,
            min_level=args.min_level,
            max_level=args.max_level,
            limit=args.limit,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(
        f"rhythm_report charts={report['summary']['processed_difficulties']} "
        f"windows={report['summary']['processed_windows']} "
        f"skipped={report['summary']['skipped']} output={report['report_path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
