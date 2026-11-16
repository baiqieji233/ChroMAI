"""Compare generated charts on the metrics that actually matter for maimai.

Built for A/B testing: point it at two or more generated charts (or the folders
holding them) and it prints one row per chart, so a decoding change, a training
change, or a parameter change can be judged by numbers instead of impression.

The metrics are the ones that were measured to separate this project's output
from real maimai charts:

  * arrangement diversity -- configuration types, top-3 share, longest run of an
    identical bar configuration, distinct note texts, buttons used;
  * note-type mix -- chord / hold / slide / touch shares;
  * structural health -- blank fraction, longest blank run, syntax validity,
    candidate-selection reachability;
  * audio following -- whether the chart gets denser where the music does, which
    is what tells legitimate repetition (a chorus) from a stuck model.

Usage:
    python tools/compare_charts.py chart.txt [chart2.txt ...] [--gold --gold-dir DIR]
    python tools/compare_charts.py dir1 dir2
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maichart.event_timing import parse_event_chart  # noqa: E402

BAR = Fraction(4)
SLIDE_CHARS = set("<>vVpqszw^")
HOLD_CHARS = set("hH")
TOUCH_CHARS = set("E!")


def note_kinds(text: str) -> set[str]:
    kinds: set[str] = set()
    if any(char in text for char in HOLD_CHARS):
        kinds.add("hold")
    if any(char in text for char in SLIDE_CHARS):
        kinds.add("slide")
    if any(char in text for char in TOUCH_CHARS):
        kinds.add("touch")
    if text and not kinds:
        kinds.add("tap")
    return kinds


def configuration(cells: list[str]) -> tuple:
    return (
        min(len(cells), 8),
        min(sum(1 for text in cells if "/" in text), 4),
        sum(1 for text in cells if "hold" in note_kinds(text)),
        sum(1 for text in cells if "slide" in note_kinds(text)),
        sum(1 for text in cells if "touch" in note_kinds(text)),
    )


def chart_metrics(path: Path, difficulty: int | None = None) -> dict:
    text = path.read_text(encoding="utf-8")
    if difficulty is None:
        difficulty = next((value for value in (5, 4, 3) if f"&inote_{value}" in text), 4)
    chart = parse_event_chart(path, difficulty)

    bars: dict[int, list[str]] = {}
    texts: Counter = Counter()
    buttons: set[str] = set()
    for point in chart.points:
        raw = "".join(str(point.note_text).split())
        if not raw:
            continue
        texts[raw] += 1
        buttons.update(char for char in raw if char.isdigit() and char != "0")
        bars.setdefault(int(Fraction(point.beat) // BAR), []).append(raw)

    rows = [{"bar": index, "cells": cells, "config": configuration(cells)} for index, cells in sorted(bars.items())]
    if not rows:
        return {"path": str(path), "bars": 0}

    configs = Counter(row["config"] for row in rows)
    top = configs.most_common(3)
    longest = run = 1
    for previous, current in zip(rows, rows[1:]):
        run = run + 1 if previous["config"] == current["config"] else 1
        longest = max(longest, run)
    notes = sum(len(row["cells"]) for row in rows)
    counts = Counter()
    for row in rows:
        for cell in row["cells"]:
            counts.update(note_kinds(cell))
    metric = {
        "path": str(path),
        "bars": len(rows),
        "configs": len(configs),
        "config_ratio": len(configs) / len(rows),
        "top3_share": sum(count for _, count in top) / len(rows),
        "longest_run": longest,
        "distinct_texts": len(texts),
        "buttons": len(buttons),
        "notes_per_bar": notes / len(rows),
        "chord_share": sum(1 for row in rows for cell in row["cells"] if "/" in cell) / notes,
        "hold_share": counts["hold"] / notes,
        "slide_share": counts["slide"] / notes,
        "touch_share": counts["touch"] / notes,
        "top3": [f"{config}x{count}" for config, count in top],
    }

    # A quality report next to the chart carries the structural metrics.
    for candidate in (path.parent / f"{path.stem}.quality.json", path.with_suffix(".quality.json")):
        if candidate.is_file():
            quality = json.loads(candidate.read_text(encoding="utf-8"))
            structure = quality.get("structure") or {}
            metric["blank_fraction"] = structure.get("blank_fraction")
            metric["max_blank_sec"] = structure.get("max_blank_run_sec")
            metric["syntax_valid"] = quality.get("syntax_valid")
            metric["windows"] = quality.get("windows_decoded")
            # Absolute-time density: the only密度口径 comparable across songs,
            # because a bar lasts a different number of seconds at every BPM.
            metric["nps"] = quality.get("note_density_nps")
            metric["peak_nps"] = quality.get("peak_nps")
            follow = quality.get("audio_follow") or {}
            if isinstance(follow, dict):
                metric["audio_follow"] = (follow.get("rms_density_correlation")
                                          or follow.get("correlation") or follow.get("score"))
                metric["audio_follow_keys"] = sorted(follow.keys())
            break
    return metric


def find_charts(target: Path) -> list[Path]:
    if target.is_dir():
        return sorted(path for path in target.glob("*.txt") if path.name.startswith("maidata") or path.suffix == ".txt")
    return [target]


def collect(targets: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for target in targets:
        for chart in find_charts(target):
            metric = chart_metrics(chart)
            if metric.get("bars"):
                metric["label"] = target.name if target.is_dir() else target.stem
                rows.append(metric)
    return rows


COLUMNS = [
    ("label", "chart", 26),
    ("bars", "bars", 5),
    ("configs", "cfg", 5),
    ("top3_share", "top3", 6),
    ("longest_run", "run", 5),
    ("distinct_texts", "texts", 6),
    ("buttons", "btn", 4),
    ("notes_per_bar", "npb", 6),
    ("chord_share", "chord", 6),
    ("hold_share", "hold", 6),
    ("slide_share", "slide", 6),
    ("touch_share", "touch", 6),
    ("blank_fraction", "blank", 6),
    ("max_blank_sec", "maxblk", 7),
    ("nps", "nps", 6),
    ("peak_nps", "peak", 6),
    ("audio_follow", "follow", 7),
]

# Keys printed with three decimals when they carry a fractional value.
FRACTIONAL = {"blank_fraction", "max_blank_sec", "audio_follow"}
THREE_DECIMALS = {"top3_share", "chord_share", "hold_share", "slide_share", "touch_share"}
ONE_DECIMAL = {"notes_per_bar", "nps"}


def cell_text(key: str, value) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        if key in FRACTIONAL:
            return f"{float(value):.3f}"
        if key in THREE_DECIMALS:
            return f"{float(value):.3f}"
        if key in ONE_DECIMAL:
            return f"{float(value):.1f}"
        return str(int(value)) if float(value).is_integer() else f"{float(value):.2f}"
    return str(value)


def print_table(rows: list[dict]) -> None:
    header = " ".join(name.rjust(width) for _, name, width in COLUMNS)
    print(header)
    print("-" * len(header))
    for row in rows:
        print(" ".join(cell_text(key, row.get(key)).rjust(width) for key, _, width in COLUMNS))
    for row in rows:
        if row.get("top3"):
            print(f"  {row['label']}: top3 = {row['top3']}")
        if row.get("audio_follow_keys"):
            print(f"  {row['label']}: audio_follow keys = {row['audio_follow_keys']}")


def gold_reference(probe: Path) -> list[dict]:
    manifest = probe / "gold_val_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"gold reference manifest not found: {manifest}")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    rows = []
    for song in data["songs"]:
        for difficulty in song["difficulties"]:
            if not difficulty.get("event_usable_for_training"):
                continue
            metric = chart_metrics(probe / song["maidata_path"], difficulty["difficulty_index"])
            if metric.get("bars"):
                metric["label"] = f"GOLD {str(song.get('song_id'))[:10]}"
                rows.append(metric)
            break
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare generated maimai charts by the numbers.")
    parser.add_argument("targets", nargs="+", help="chart files or folders to compare")
    parser.add_argument("--gold", action="store_true", help="also print reference charts from --gold-dir")
    parser.add_argument("--gold-dir", type=Path, help="reference directory containing gold_val_manifest.json")
    args = parser.parse_args(argv)

    if args.gold and args.gold_dir is None:
        parser.error("--gold requires --gold-dir <dir> containing gold_val_manifest.json")
    gold_rows: list[dict] = []
    if args.gold or args.gold_dir is not None:
        assert args.gold_dir is not None
        manifest = args.gold_dir / "gold_val_manifest.json"
        if not manifest.is_file():
            parser.error(f"gold reference directory or manifest not found: {manifest}")
        gold_rows = gold_reference(args.gold_dir)

    rows = collect([Path(target) for target in args.targets])
    if not rows:
        print("no charts with notes found")
        return 1
    if gold_rows:
        rows = gold_rows + rows
    print_table(rows)

    # Aggregate so a multi-song A/B is readable at a glance.
    if len(rows) > 1:
        print()
        print("aggregate over the non-gold charts:")
        for key in ("configs", "top3_share", "longest_run", "distinct_texts", "slide_share",
                    "blank_fraction", "notes_per_bar"):
            values = [row[key] for row in rows if key in row and not row["label"].startswith("GOLD")]
            if values:
                print(f"  {key:<16} mean={statistics.mean(values):>8.3f}  min={min(values):>8.3f}  max={max(values):>8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
