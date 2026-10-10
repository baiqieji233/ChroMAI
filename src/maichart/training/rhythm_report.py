"""Read-only rhythm statistics for maimai training manifests."""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any

from maichart.event_timing import parse_event_chart, seconds_at_beat
from maichart.generation.event_quality import parse_event_notes
from maichart.training.dataset_catalog import _source_asset
from maichart.training.event_dataset import inspect_chart, numeric_level
from maichart.training.music_plan_targets import fine_note_ratio


def _resolve_manifest(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_dir():
        path = path / "training_manifest.json"
    return path.resolve()


def _resolve_maidata(value: str, manifest_path: Path, manifest: dict[str, Any]) -> Path:
    resolved = _source_asset(value, manifest_path, manifest)
    if resolved.is_file():
        return resolved
    path = Path(value)
    if path.is_absolute():
        return resolved
    bases = [manifest_path.parent]
    source_root = manifest.get("source_root")
    if source_root and not str(source_root).startswith("<"):
        root = Path(str(source_root))
        bases.append(root if root.is_absolute() else manifest_path.parent / root)
    bases.append(manifest_path.parent.parent)
    for base in bases:
        candidate = (base / path).resolve()
        if candidate.is_file():
            return candidate
    try:
        for child in manifest_path.parent.iterdir():
            if child.is_dir() and (child / path).is_file():
                return (child / path).resolve()
    except OSError:
        pass
    return resolved


def _number_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def analyze_training_manifest(
    dataset: str | Path,
    output: str | Path | None = None,
    *,
    window_beats: int = 16,
    difficulties: set[int] | None = None,
    min_level: float | None = None,
    max_level: float | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Analyze charts in a training manifest without changing source files."""
    manifest_path = _resolve_manifest(dataset)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Training manifest not found: {manifest_path}")
    if window_beats <= 0:
        raise ValueError("window_beats must be a positive integer")
    if limit is not None and limit < 0:
        raise ValueError("limit must be nonnegative")
    if min_level is not None and max_level is not None and min_level > max_level:
        raise ValueError("min_level must not exceed max_level")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("songs"), list):
        raise ValueError("Manifest must contain a songs list")
    output_path = Path(output).expanduser().resolve() if output else manifest_path.with_name(
        f"{manifest_path.stem}.rhythm_report.json"
    )
    if output_path == manifest_path:
        raise ValueError("Report output must not overwrite the source manifest")

    skipped: Counter[str] = Counter()
    candidates: list[dict[str, Any]] = []
    filtered: Counter[str] = Counter()
    for song_index, song in enumerate(manifest["songs"]):
        if not isinstance(song, dict):
            skipped["invalid_song_record"] += 1
            continue
        for difficulty in song.get("difficulties", []):
            if not isinstance(difficulty, dict):
                skipped["invalid_difficulty_record"] += 1
                continue
            raw_index = difficulty.get("difficulty_index", difficulty.get("index"))
            try:
                difficulty_index = int(raw_index)
            except (TypeError, ValueError):
                skipped["difficulty_index_missing_or_invalid"] += 1
                continue
            if difficulties is not None and difficulty_index not in difficulties:
                filtered["difficulty"] += 1
                continue
            raw_level = difficulty.get("level_raw", difficulty.get("level"))
            level: float | None
            try:
                level = numeric_level(raw_level)
            except (TypeError, ValueError):
                level = None
            if (min_level is not None or max_level is not None) and level is None:
                skipped["level_missing_or_invalid"] += 1
                continue
            if min_level is not None and level is not None and level < min_level:
                filtered["level"] += 1
                continue
            if max_level is not None and level is not None and level > max_level:
                filtered["level"] += 1
                continue
            candidates.append({
                "song_index": song_index,
                "song": song,
                "difficulty": difficulty,
                "difficulty_index": difficulty_index,
                "level": level,
            })

    selected = candidates if limit is None else candidates[:limit]
    limited_out = max(0, len(candidates) - len(selected))
    rows: list[dict[str, Any]] = []
    all_ratios: list[float] = []
    example_candidates: list[dict[str, Any]] = []
    per_level_ratios: dict[str, list[float]] = defaultdict(list)
    per_level_charts: Counter[str] = Counter()
    for item in selected:
        song = item["song"]
        song_id = str(song.get("song_id", item["song_index"]))
        chart_path_value = song.get("maidata_path")
        if not chart_path_value:
            skipped["maidata_path_missing"] += 1
            continue
        try:
            chart_path = _resolve_maidata(str(chart_path_value), manifest_path, manifest)
            if not chart_path.is_file():
                raise FileNotFoundError(f"Maidata file not found: {chart_path}")
            chart = parse_event_chart(chart_path, item["difficulty_index"])
            inspect_chart(chart)
            parsed = parse_event_notes(chart.points, chart.tempo_changes, first=chart.audio_first)
            if not parsed.syntax_valid:
                raise ValueError("Chart note syntax is invalid")
        except Exception as exc:
            reason = "source_missing" if isinstance(exc, FileNotFoundError) else "chart_parse_or_validation_error"
            skipped[reason] += 1
            continue

        level_key = f"{item['level']:g}" if item["level"] is not None else "unknown"
        per_level_charts[level_key] += 1
        chart_windows: list[dict[str, Any]] = []
        start = Fraction(0)
        end_beat = Fraction(chart.end_beat)
        while start < end_beat:
            end = min(start + window_beats, end_beat)
            notes = [note for note in parsed.notes if start <= note.beat < end]
            beats = sorted({note.beat for note in notes})
            ratio = fine_note_ratio(note.beat for note in notes)
            start_sec = seconds_at_beat(start, chart.tempo_changes, chart.audio_first)
            end_sec = seconds_at_beat(end, chart.tempo_changes, chart.audio_first)
            chart_windows.append({
                "start_beat": str(start),
                "end_beat": str(end),
                "start_sec": float(start_sec),
                "end_sec": float(end_sec),
                "onset_count": len(beats),
                "note_count": len(notes),
                "fine_note_ratio": ratio,
                "first_onset": ({
                    "beat": str(beats[0]),
                    "sec": float(seconds_at_beat(beats[0], chart.tempo_changes, chart.audio_first)),
                } if beats else None),
                "last_onset": ({
                    "beat": str(beats[-1]),
                    "sec": float(seconds_at_beat(beats[-1], chart.tempo_changes, chart.audio_first)),
                } if beats else None),
            })
            all_ratios.append(ratio)
            per_level_ratios[level_key].append(ratio)
            start = end
        for window in chart_windows:
            if window["onset_count"]:
                example_candidates.append({
                    "song_id": song_id,
                    "title": str(song.get("title", "")),
                    "difficulty_index": item["difficulty_index"],
                    "level": item["level"],
                    **window,
                })
        rows.append({
            "song_id": song_id,
            "title": str(song.get("title", "")),
            "difficulty_index": item["difficulty_index"],
            "level": item["level"],
            "source_path": str(chart_path),
            "window_count": len(chart_windows),
            "nonempty_window_count": sum(bool(window["onset_count"]) for window in chart_windows),
            "fine_note_ratio": _number_summary([window["fine_note_ratio"] for window in chart_windows]),
        })

    examples = sorted(
        example_candidates,
        key=lambda window: (window["fine_note_ratio"], window["onset_count"], window["start_beat"]),
        reverse=True,
    )[:8]
    window_count = sum(row["window_count"] for row in rows)
    nonempty_count = sum(row["nonempty_window_count"] for row in rows)
    level_distribution = {}
    for level_key in sorted(per_level_charts, key=lambda key: (key == "unknown", float(key) if key != "unknown" else 0)):
        ratios = per_level_ratios[level_key]
        level_distribution[level_key] = {
            "chart_count": per_level_charts[level_key],
            "window_count": len(ratios),
            "fine_note_ratio": _number_summary(ratios),
            "windows_ratio_at_least_0_25": sum(value >= 0.25 for value in ratios),
        }

    report: dict[str, Any] = {
        "schema": "maichart-rhythm-dataset-report-v1",
        "manifest": str(manifest_path),
        "report_path": str(output_path),
        "window_beats": window_beats,
        "filters": {
            "difficulty_indices": sorted(difficulties) if difficulties else None,
            "min_level": min_level,
            "max_level": max_level,
            "limit": limit,
        },
        "summary": {
            "candidate_difficulties": len(candidates),
            "selected_difficulties": len(selected),
            "processed_difficulties": len(rows),
            "processed_windows": window_count,
            "nonempty_windows": nonempty_count,
            "fine_note_ratio": _number_summary(all_ratios),
            "windows_ratio_at_least_0_25": sum(value >= 0.25 for value in all_ratios),
            "limited_out": limited_out,
            "filtered": dict(sorted(filtered.items())),
            "skipped": dict(sorted(skipped.items())),
        },
        "per_level": level_distribution,
        "window_examples": examples,
        "charts": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report
