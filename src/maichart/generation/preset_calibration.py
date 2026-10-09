"""Calibrate level-dependent generation controls from original charts.

This describes the training corpus; it does not estimate official difficulty.
Audio headers supply the same duration denominator used by event training.
No waveforms, feature extraction, training or GPU work are required.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
from typing import Any

from maichart.generation.conditions import RADAR_KEYS, STYLE_KEYS

SCHEMA = "maichart-level-presets-v1"
CONTROL_KEYS = RADAR_KEYS + STYLE_KEYS


def weighted_quantile(values: list[float], weights: list[float], q: float) -> float:
    pairs = sorted((float(value), float(weight)) for value, weight in zip(values, weights) if weight > 0)
    if not pairs:
        raise ValueError("Cannot calibrate an empty level neighborhood")
    threshold = sum(weight for _, weight in pairs) * q
    cumulative = 0.0
    for value, weight in pairs:
        cumulative += weight
        if cumulative >= threshold:
            return value
    return pairs[-1][0]


def isotonic(values: list[float], weights: list[float]) -> list[float]:
    """Weighted pooled-adjacent-violators fit, preserving a nondecreasing trend."""
    blocks: list[list[float]] = []
    for index, (value, weight) in enumerate(zip(values, weights)):
        blocks.append([float(index), float(index), value * max(weight, 1e-9), max(weight, 1e-9)])
        while len(blocks) > 1 and blocks[-2][2] / blocks[-2][3] > blocks[-1][2] / blocks[-1][3]:
            right = blocks.pop()
            left = blocks.pop()
            blocks.append([left[0], right[1], left[2] + right[2], left[3] + right[3]])
    result = [0.0] * len(values)
    for start, end, total, weight in blocks:
        result[int(start):int(end) + 1] = [total / weight] * (int(end) - int(start) + 1)
    return result


def _audio_duration(path: Path) -> float:
    try:
        import soundfile as sf

        return float(sf.info(str(path)).duration)
    except Exception:
        ffprobe = shutil.which("ffprobe")
        if not ffprobe:
            raise ValueError("Audio header duration unavailable")
        result = subprocess.run([ffprobe, "-v", "error", "-show_entries", "format=duration",
                                 "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                                capture_output=True, text=True, check=True, timeout=15)
        return float(result.stdout.strip())


def collect_chart_controls(manifest_path: Path) -> tuple[list[dict], dict]:
    from maichart.event_timing import parse_event_chart
    from maichart.generation.event_quality import condition_from_events
    from maichart.training.event_dataset import inspect_chart, numeric_level

    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    bases = [manifest_path.parent]
    declared = manifest.get("source_root")
    if declared and not str(declared).startswith("<"):
        root = Path(declared)
        bases.append(root if root.is_absolute() else manifest_path.parent / root)
    bases.append(manifest_path.parent.parent)

    def resolve(value):
        path = Path(str(value))
        if path.is_absolute():
            return path
        return next((base / path for base in bases if (base / path).is_file()), bases[0] / path)

    rows, skipped, durations, seen = [], [], {}, set()
    label_counts: Counter = Counter()
    for song in manifest.get("songs", []):
        for difficulty in song.get("difficulties", []):
            if not (difficulty.get("event_usable_for_training") or difficulty.get("usable_for_training")):
                continue
            index = int(difficulty.get("difficulty_index", difficulty.get("index", 0)))
            if not 4 <= index <= 6:
                continue
            try:
                raw = str(difficulty.get("level_raw", difficulty.get("level", ""))).strip()
                level = numeric_level(raw)
                if not math.isfinite(level) or not 12 <= level <= 15:
                    continue
                source, audio = resolve(song["maidata_path"]), resolve(song["audio_path"])
                if not source.is_file() or not audio.is_file():
                    raise ValueError("missing_source")
                identity = (str(source.resolve()), index)
                if identity in seen:
                    continue
                if audio not in durations:
                    durations[audio] = _audio_duration(audio)
                duration = durations[audio]
                if not math.isfinite(duration) or duration <= 0:
                    raise ValueError("invalid_audio_duration")
                chart = parse_event_chart(source, index)
                inspect_chart(chart)
                condition = condition_from_events(chart.points, chart.tempo_changes,
                                                  level=level, duration_sec=duration, audio=None)
                kind = "display_plus" if raw.endswith("+") else "decimal" if "." in raw else "display_integer"
                controls = {**condition.radar, **condition.style}
                # This feature needs onset analysis. Never calibrate it from the
                # no-audio zero placeholder; retain the audio-measured default.
                controls["melody_following"] = 0.9
                rows.append({"song_id": str(song.get("song_id", "")), "difficulty_index": index,
                             "level": level, "level_raw": raw, "label_kind": kind,
                             "label_weight": 1.0 if kind == "decimal" else 0.35,
                             "audio_duration_sec": duration, "controls": controls})
                seen.add(identity)
                label_counts[kind] += 1
            except Exception as exc:
                skipped.append({"song_id": str(song.get("song_id", "")), "difficulty_index": index,
                                "reason": str(exc)[:300]})
        if len(rows) and len(rows) % 200 == 0:
            print(f"calibrated_charts={len(rows)} skipped={len(skipped)}", flush=True)
    return rows, {"charts": len(rows), "skipped": len(skipped), "label_counts": dict(label_counts),
                  "duration_source": "audio_header", "chart_duration_fallback": 0,
                  "skip_reasons": dict(Counter(item["reason"] for item in skipped)),
                  "skipped_examples": skipped[:20]}


def calibrate_levels(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("No usable original charts with audio duration in 12–15")
    levels = []
    for index in range(31):
        level = round(12 + index / 10, 1)
        direct = [row for row in rows if abs(row["level"] - level) < 0.049999]
        bandwidth = 0.15
        while True:
            local = [row for row in rows if abs(row["level"] - level) <= 3 * bandwidth + 1e-9]
            weights = [math.exp(-0.5 * ((row["level"] - level) / bandwidth) ** 2) * row.get("label_weight", 1)
                       for row in local]
            support = sum(weights) ** 2 / max(sum(weight * weight for weight in weights), 1e-12)
            if support >= 45 or bandwidth >= 0.45:
                break
            bandwidth = round(bandwidth + 0.05, 2)
        controls, bounds = {}, {}
        for key in CONTROL_KEYS:
            values = [row["controls"][key] for row in local]
            controls[key] = weighted_quantile(values, weights, .5)
            bounds[key] = {"p10": weighted_quantile(values, weights, .1),
                           "p90": weighted_quantile(values, weights, .9)}
        levels.append({"level": level, "radar": {key: controls[key] for key in RADAR_KEYS},
                       "style": {key: controls[key] for key in STYLE_KEYS}, "controls": bounds,
                       "support": round(support, 2), "raw_count": len(direct),
                       "exact_count": sum(row.get("label_kind") == "decimal" for row in direct),
                       "bandwidth": bandwidth, "fallback": len(direct) < 20,
                       "examples": [row["song_id"] for row in sorted(local, key=lambda row: abs(row["level"] - level))[:3]]})
    # Enforce a smooth nondecreasing difficulty budget only for the two amount
    # controls. Configuration and action-type shares retain their observed trend.
    for key in ("note_density", "peak"):
        fitted = isotonic([row["radar"][key] for row in levels], [row["support"] for row in levels])
        for row, value in zip(levels, fitted):
            row["radar"][key] = value
    for row in levels:
        for section in ("radar", "style"):
            for key, value in row[section].items():
                value = round(min(1.0, max(0.0, value)), 6)
                row[section][key] = value
                row["controls"][key]["p10"] = round(min(value, row["controls"][key]["p10"]), 6)
                row["controls"][key]["p90"] = round(max(value, row["controls"][key]["p90"]), 6)
    return levels


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args(argv)
    source = args.manifest.resolve()
    rows, audit = collect_chart_controls(source)
    profile = {"schema": SCHEMA, "source": {"manifest": str(source),
               "sha256": hashlib.sha256(source.read_bytes()).hexdigest(), **audit},
               "method": {"level_grid": "12.0–15.0 by 0.1", "statistic": "Gaussian weighted median",
                          "bandwidth": "0.15, expanded to at most 0.45 when effective support <45",
                          "amount_trend": "weighted isotonic fit for note_density and peak only",
                          "label_weight": {"decimal": 1.0, "display_integer": .35, "display_plus": .35},
                          "label_limit": "Decimal labels are provided chart metadata, not independently verified official constants",
                          "melody_following": "0.9 retained; no audio onset measurement in this calibration"},
               "levels": calibrate_levels(rows)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.audit:
        args.audit.parent.mkdir(parents=True, exist_ok=True)
        args.audit.write_text(json.dumps({"summary": audit, "charts": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=True), flush=True)
    print(f"level_presets_written={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
