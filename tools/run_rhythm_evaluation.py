"""Describe generated chart structure without producing a single quality score."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maichart.event_timing import EventChart, EventPoint, parse_event_chart, parse_event_chart_text, seconds_at_beat
from maichart.generation.event_quality import assess_event_chart, parse_event_notes
from maichart.generation.acceptance import structural_metrics
from maichart.training.dataset_catalog import _source_asset
from maichart.training.event_dataset import inspect_chart, numeric_level
from maichart.training.music_plan_targets import fine_note_ratio


SUPPORTED_SUFFIXES = (".chart.json", ".chart_ir.json", ".ir.json")


def _decode_embedded_event(payload: dict[str, Any]) -> dict[str, Any] | None:
    if isinstance(payload.get("tempo_changes"), list) and isinstance(payload.get("points"), list):
        return payload
    for container in (payload, payload.get("chart_ir"), payload.get("ir")):
        if not isinstance(container, dict):
            continue
        metadata = container.get("metadata") or {}
        difficulty = container.get("difficulty") or {}
        for raw in (metadata.get("raw"), container.get("raw")):
            if not isinstance(raw, str) or not raw.strip():
                continue
            try:
                decoded = json.loads(raw)
            except json.JSONDecodeError:
                decoded = None
            if isinstance(decoded, dict) and isinstance(decoded.get("points"), list):
                return decoded
            if raw.lstrip().startswith("&"):
                index = difficulty.get("index") or payload.get("difficulty_index")
                if index is not None:
                    chart = parse_event_chart_text(raw, int(index))
                    return chart.to_dict()
    return None


def _duration(payload: dict[str, Any], *, audio_first: float) -> tuple[float | None, str | None]:
    quality = payload.get("quality") if isinstance(payload.get("quality"), dict) else {}
    for key, source in (("audio_duration_sec", "payload.audio_duration_sec"),
                        ("duration_sec", "payload.duration_sec")):
        value = payload.get(key)
        if value is not None:
            duration = float(value)
            if math.isfinite(duration) and duration > 0:
                return duration, source
    value = quality.get("duration_sec")
    if value is not None and audio_first == 0.0:
        duration = float(value)
        if math.isfinite(duration) and duration > 0:
            return duration, "embedded_quality.duration_sec_legacy_first_zero"
    return None, None


def _onset_rhythm(notes) -> dict[str, Any]:
    onsets: dict[Fraction, list[str]] = defaultdict(list)
    for note in notes:
        onsets[Fraction(note.beat)].append(note.note_type)
    beats = sorted(onsets)
    gaps = [right - left for left, right in zip(beats, beats[1:]) if right > left]
    gap_counts = Counter(gaps)
    patterns = [tuple(sorted(onsets[beat])) for beat in beats]
    repeated_pattern = {"cycles": 1 if patterns else 0, "onset_window_length": 0, "pattern": []}
    for start in range(len(patterns)):
        for width in range(1, min(8, (len(patterns) - start) // 2) + 1):
            cycles = 1
            while (start + (cycles + 1) * width <= len(patterns)
                   and patterns[start:start + width] == patterns[start + cycles * width:start + (cycles + 1) * width]):
                cycles += 1
            if cycles > repeated_pattern["cycles"] or (
                    cycles == repeated_pattern["cycles"] and cycles > 1 and width > repeated_pattern["onset_window_length"]):
                repeated_pattern = {
                    "cycles": cycles,
                    "onset_window_length": width,
                    "pattern": [list(types) for types in patterns[start:start + width]],
                }
    named_gaps = {"12th": Fraction(1, 3), "16th": Fraction(1, 4), "24th": Fraction(1, 6)}
    return {
        "distinct_onset_count": len(beats),
        "fine_note_ratio": fine_note_ratio(beats),
        "short_gap_share": sum(gap <= Fraction(1, 2) for gap in gaps) / len(gaps) if gaps else 0.0,
        "gap_sample_count": len(gaps),
        "gap_histogram_beats": {str(gap): gap_counts[gap] for gap in sorted(gap_counts)},
        "named_subdivision_gap_counts": {name: gap_counts.get(gap, 0) for name, gap in named_gaps.items()},
        "max_repeated_onset_type_pattern": repeated_pattern,
    }


def _playability_evidence(quality_payload: dict[str, Any]) -> dict[str, Any]:
    stored = quality_payload.get("playability")
    if not isinstance(stored, dict):
        return {"status": "unavailable", "verified": False}
    coverage = stored.get("coverage") if isinstance(stored.get("coverage"), dict) else {}
    return {
        "status": stored.get("status", coverage.get("status", "unknown")),
        "verified": False,
        "scope": stored.get("scope"),
        "coverage": coverage,
        "counts": {key: stored.get(key) for key in
                   ("notes", "original_notes", "conflicts", "hard", "remaining_hard") if key in stored},
        "repair_count": len(stored.get("repairs", [])) if isinstance(stored.get("repairs"), list) else None,
    }


def _metric_row(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("chart JSON root must be an object")
    event_data = _decode_embedded_event(payload)
    if event_data is None:
        raise ValueError(
            "no exact event points and tempo map found; expected points/tempo_changes or a serialized event chart in ChartIR.raw"
        )
    quality_payload = payload.get("quality") if isinstance(payload.get("quality"), dict) else {}
    declared_end = event_data.get("end_beat")
    end_beat_source = "event_chart.end_beat" if declared_end is not None else None
    if declared_end is None:
        declared_end = event_data.get("song_beats", payload.get("song_beats"))
        if declared_end is None:
            declared_end = quality_payload.get("song_beats")
            if declared_end is not None:
                end_beat_source = "quality.song_beats"
        elif declared_end is not None:
            end_beat_source = "song_beats"
    chart_data = dict(event_data)
    if declared_end is not None:
        chart_data["end_beat"] = str(declared_end)
    else:
        chart_data["end_beat"] = str(max(
            (Fraction(point["beat"]) for point in chart_data.get("points", []) if point.get("beat") is not None),
            default=Fraction(0),
        ))
    chart = EventChart.from_dict(chart_data)
    chart.time_offset = float(event_data.get("time_offset", 0.0))
    # New output IR defines top-level `first` as the audio timestamp of beat 0.
    top_level_audio_first = (event_data is payload
                             and payload.get("schema") == "maichart-event-chart-v1"
                             and payload.get("first") is not None)
    audio_first = float(payload["first"]) if top_level_audio_first else chart.audio_first
    parsed = parse_event_notes(chart.points, chart.tempo_changes, first=audio_first)
    duration, duration_source = _duration(payload, audio_first=audio_first)
    quality = assess_event_chart(
        chart.points,
        chart.tempo_changes,
        first=audio_first,
        duration_sec=duration,
    )
    notes = parsed.notes
    note_seconds = [float(note.time_sec) for note in notes]
    before_audio = sum(second < 0 for second in note_seconds)
    after_audio = sum(second >= duration for second in note_seconds) if duration is not None else None
    beyond_end_beat = (sum(note.beat >= chart.end_beat for note in notes)
                       if end_beat_source is not None else None)
    structure = structural_metrics(
        chart.points,
        chart.tempo_changes,
        audio_first,
        audio_duration_sec=duration,
    )

    per_second_counts: dict[int, Counter[str]] = defaultdict(Counter)
    for note in notes:
        per_second_counts[math.floor(float(note.time_sec))][note.note_type] += 1
    if duration is not None:
        first_second, last_second = min(0, min(per_second_counts, default=0)), math.ceil(duration)
    elif per_second_counts:
        first_second, last_second = min(0, min(per_second_counts)), max(per_second_counts) + 1
    else:
        first_second = last_second = 0
    second_rows = []
    for second in range(first_second, last_second):
        counts = per_second_counts[second]
        second_rows.append({
            "second": second,
            "note_count": sum(counts.values()),
            "types": {key: counts[key] for key in ("tap", "hold", "slide", "touch", "touch_hold")},
        })
    playable_span_rows = [row for row in second_rows if row["second"] >= 0]
    rhythm = _onset_rhythm(notes)
    quality_types = quality.get("type_ratios") or {}
    counts = quality.get("counts") or {}
    metadata = (payload.get("chart_ir", {}).get("metadata", {})
                if isinstance(payload.get("chart_ir"), dict) else {}) or {}
    identity = {
        "title": chart.title or metadata.get("title") or path.stem,
        "difficulty_index": chart.difficulty,
        "level": chart.level or None,
        "audio_sha256": payload.get("audio_sha256") or event_data.get("audio_sha256"),
        "checkpoint_identity": payload.get("checkpoint_identity") or quality_payload.get("checkpoint_identity"),
        "checkpoint_sha256": payload.get("checkpoint_sha256") or quality_payload.get("checkpoint_sha256"),
    }
    return {
        "path": str(path.resolve()),
        "identity": identity,
        "timing": {
            "tempo_map": [{"beat": str(change.beat), "bpm": change.bpm} for change in chart.tempo_changes],
            "first_sec": chart.first,
            "time_offset_sec": chart.time_offset,
            "audio_first_sec": audio_first,
            "end_beat": str(chart.end_beat) if end_beat_source is not None else None,
            "end_beat_source": end_beat_source or "missing",
            "audio_duration_sec": duration,
            "duration_source": duration_source,
        },
        "syntax": {
            "valid": bool(quality.get("syntax_valid")),
            "duplicate_head_count": sum(
                issue.get("code") == "duplicate_button_head"
                for issue in quality.get("fatal_errors", [])
            ),
            "fatal_errors": quality.get("fatal_errors", []),
            "unsupported": quality.get("unsupported", []),
        },
        "distribution": {
            "note_count": counts.get("notes", len(notes)),
            "event_count": counts.get("events"),
            "note_density_per_sec": quality.get("note_density_nps"),
            "peak_notes_per_sec": quality.get("peak_nps"),
            "type_counts": quality.get("types", {}),
            "type_ratios": quality_types,
            "chord_ratio": quality.get("chord_ratio"),
            "button_entropy": (quality.get("entropy") or {}).get("buttons"),
            "slide_path_count": counts.get("slide_branches"),
            "onset_rhythm": rhythm,
        },
        "structure": {
            "measure_count": structure["measures"],
            "blank_measure_count": structure["blank_measures"],
            "blank_fraction": structure["blank_fraction"],
            "longest_blank_run_measures": structure["max_blank_run_measures"],
            "longest_blank_run_sec": structure["max_blank_run_sec"],
            "coverage_ratio": structure["coverage_ratio"],
            "repeated_event_signature_run": quality.get("max_repeated_run"),
            "repeated_measure_windows": (quality.get("motif") or {}).get("repeated_windows"),
            "repeated_measure_window_ratio": (quality.get("motif") or {}).get("reuse_ratio"),
        },
        "boundaries": {
            "first_note_sec": min(note_seconds) if note_seconds else None,
            "last_note_sec": max(note_seconds) if note_seconds else None,
            "notes_before_audio_start": before_audio,
            "notes_at_or_after_audio_end": after_audio,
            "audio_end_check_available": duration is not None,
            "notes_at_or_after_declared_end_beat": beyond_end_beat,
        },
        "per_second": {
            "window_seconds": 1,
            "windows": second_rows,
            "empty_seconds_within_audio": (sum(row["note_count"] == 0 for row in playable_span_rows)
                                             if duration is not None else None),
        },
        "playability_evidence": _playability_evidence(quality_payload),
        "interpretation": "descriptive only; note count is not a quality score and playable quality is unverified",
        "_density_counts": [row["note_count"] for row in playable_span_rows],
        "_onset_seconds": sorted({float(note.time_sec) for note in notes}),
    }


def _discover(inputs: list[Path]) -> tuple[list[Path], Path | None]:
    files: list[Path] = []
    roots = [path.resolve() for path in inputs if path.is_dir()]
    for path in inputs:
        if path.is_dir():
            files.extend(candidate for candidate in path.rglob("*.json")
                         if candidate.name.endswith(SUPPORTED_SUFFIXES) and not _is_shadow(candidate))
        elif path.is_file():
            files.append(path.resolve())
        else:
            raise FileNotFoundError(path)
    unique = sorted(set(files))
    return unique, roots[0] if len(roots) == 1 else None


def _is_shadow(path: Path) -> bool:
    return any(part.casefold() == ".history" for part in path.parts) or path.name.casefold().endswith(".original.chart.json")


def _onset_alignment(generated: dict[str, Any], reference: dict[str, Any], *, tolerance_sec: float = 0.05) -> dict[str, Any]:
    left = sorted(generated["_onset_seconds"])
    right = sorted(reference["_onset_seconds"])
    i = j = matched = 0
    while i < len(left) and j < len(right):
        delta = left[i] - right[j]
        if abs(delta) <= tolerance_sec:
            matched += 1
            i += 1
            j += 1
        elif delta < 0:
            i += 1
        else:
            j += 1
    precision = matched / len(left) if left else 0.0
    recall = matched / len(right) if right else 0.0
    return {"tolerance_sec": tolerance_sec, "generated_onsets": len(left), "reference_onsets": len(right),
            "matched_onsets": matched, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "interpretation": "onset agreement with this reference only; reference intent may be incomplete"}


def _pair_metrics(generated: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    generated_hash = generated["identity"].get("audio_sha256")
    reference_hash = reference["identity"].get("audio_sha256")
    if generated_hash and reference_hash and generated_hash != reference_hash:
        return {"status": "incompatible", "identity_status": "audio_sha256_mismatch",
                "reference_path": reference["path"]}
    scalar_pairs = {
        "blank_fraction": (generated["structure"]["blank_fraction"], reference["structure"]["blank_fraction"]),
        "coverage_ratio": (generated["structure"]["coverage_ratio"], reference["structure"]["coverage_ratio"]),
        "note_density_per_sec": (generated["distribution"]["note_density_per_sec"],
                                 reference["distribution"]["note_density_per_sec"]),
        "duplicate_head_count": (generated["syntax"]["duplicate_head_count"],
                                 reference["syntax"]["duplicate_head_count"]),
        "repeated_event_signature_run": (generated["structure"]["repeated_event_signature_run"],
                                         reference["structure"]["repeated_event_signature_run"]),
    }
    deltas = {
        key: (float(left) - float(right) if left is not None and right is not None else None)
        for key, (left, right) in scalar_pairs.items()
    }
    generated_ratios = generated["distribution"]["type_ratios"]
    reference_ratios = reference["distribution"]["type_ratios"]
    deltas["type_ratio"] = {
        key: float(generated_ratios.get(key, 0.0)) - float(reference_ratios.get(key, 0.0))
        for key in sorted(set(generated_ratios) | set(reference_ratios))
    }
    left_counts, right_counts = generated["_density_counts"], reference["_density_counts"]
    shared = min(len(left_counts), len(right_counts))
    return {
        "status": "matched" if generated_hash and reference_hash else "matched_unverified",
        "identity_status": "audio_sha256_match" if generated_hash and reference_hash else "filename_only_unverified",
        "reference_path": reference["path"],
        "paired_metric_deltas": deltas,
        "per_second_density_mae": (
            sum(abs(left_counts[index] - right_counts[index]) for index in range(shared)) / shared
            if shared else None
        ),
        "per_second_density_compared_seconds": shared,
        "duration_difference_sec": (
            generated["timing"]["audio_duration_sec"] - reference["timing"]["audio_duration_sec"]
            if generated["timing"]["audio_duration_sec"] is not None
            and reference["timing"]["audio_duration_sec"] is not None else None
        ),
        "onset_alignment": _onset_alignment(generated, reference),
    }


def evaluate_paths(inputs: list[Path], *, reference_dir: Path | None = None) -> dict[str, Any]:
    paths, input_root = _discover(inputs)
    if not paths:
        raise ValueError("No supported chart JSON files found.")
    reference_files: dict[str, Path] = {}
    if reference_dir is not None:
        if not reference_dir.is_dir():
            raise FileNotFoundError(reference_dir)
        reference_files = {
            path.relative_to(reference_dir.resolve()).as_posix(): path.resolve()
            for path in reference_dir.rglob("*.json") if path.name.endswith(SUPPORTED_SUFFIXES) and not _is_shadow(path)
        }
    rows = []
    errors = []
    for path in paths:
        try:
            row = _metric_row(path)
            if reference_dir is not None:
                if input_root is not None:
                    relative = path.relative_to(input_root).as_posix()
                else:
                    relative = path.name
                reference_path = reference_files.get(relative)
                if reference_path is None:
                    row["reference_pair"] = {"status": "missing", "key": relative}
                else:
                    reference = _metric_row(reference_path)
                    row["reference_pair"] = _pair_metrics(row, reference)
            row.pop("_density_counts", None)
            row.pop("_onset_seconds", None)
            rows.append(row)
        except (OSError, ValueError, TypeError, json.JSONDecodeError, ZeroDivisionError) as exc:
            errors.append({"path": str(path.resolve()), "error": str(exc)})
    if reference_dir is not None:
        paired = sum(row.get("reference_pair", {}).get("status") in {"matched", "matched_unverified"} for row in rows)
        missing = sum(row.get("reference_pair", {}).get("status") == "missing" for row in rows)
        incompatible = sum(row.get("reference_pair", {}).get("status") == "incompatible" for row in rows)
        verified = sum(row.get("reference_pair", {}).get("status") == "matched" for row in rows)
        pairing = {"status": "requested", "reference_dir": str(reference_dir.resolve()),
                   "matched": paired, "verified_matched": verified, "missing": missing,
                   "incompatible": incompatible,
                   "method": "exact relative filename; audio SHA-256 verified when both files provide it; otherwise filename-only and unverified"}
    else:
        pairing = {"status": "not_requested", "matched": 0, "missing": 0}
    return {
        "schema": "maichart-rhythm-evaluation-v1",
        "metric_policy": "descriptive metrics only; no aggregate quality score",
        "input_count": len(paths),
        "evaluated_count": len(rows),
        "error_count": len(errors),
        "reference_pairing": pairing,
        "charts": rows,
        "errors": errors,
    }


def _sha256_file(path: Path, cache: dict[Path, str]) -> str:
    path = path.resolve()
    if path not in cache:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        if path.stat().st_size == 0:
            raise ValueError("audio file is empty")
        cache[path] = digest.hexdigest()
    return cache[path]


def build_fixed_panel(
    manifest_path: Path,
    *,
    size: int = 10,
    split_report_path: Path | None = None,
) -> dict[str, Any]:
    """Choose reproducible reference songs by measured chart traits, not labels."""
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("songs"), list):
        raise ValueError("Panel source must be a training manifest with songs[].")
    if size < 1:
        raise ValueError("panel size must be positive")
    forbidden_audio_hashes: set[str] = set()
    split_report = None
    if split_report_path is not None:
        split_report_path = split_report_path.resolve()
        split_report = json.loads(split_report_path.read_text(encoding="utf-8-sig"))
        expected_manifest_hash = split_report.get("validation_manifest_sha256")
        manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        if expected_manifest_hash and manifest_hash != expected_manifest_hash:
            raise ValueError("panel manifest does not match split report validation_manifest_sha256")
        forbidden_audio_hashes = set(split_report.get("train_audio_hashes", []))
    candidates = []
    skipped: Counter[str] = Counter()
    audio_hash_cache: dict[Path, str] = {}
    seen_audio_hashes: set[str] = set()
    for song in manifest["songs"]:
        if not isinstance(song, dict):
            skipped["invalid_song"] += 1
            continue
        source_value = song.get("maidata_path")
        audio_value = song.get("audio_path")
        for difficulty in song.get("difficulties", []):
            if not isinstance(difficulty, dict) or not difficulty.get("event_usable_for_training"):
                continue
            try:
                index = int(difficulty.get("difficulty_index", difficulty.get("index")))
                level = numeric_level(difficulty.get("level_raw", difficulty.get("level")))
            except (TypeError, ValueError):
                skipped["difficulty_metadata_invalid"] += 1
                continue
            if index not in {4, 5, 6} or not 12 <= level <= 15:
                continue
            if not source_value or not audio_value:
                skipped["source_path_missing"] += 1
                continue
            source_path = _source_asset(source_value, manifest_path, manifest)
            audio_path = _source_asset(audio_value, manifest_path, manifest)
            if not source_path.is_file() or not audio_path.is_file():
                skipped["source_file_missing"] += 1
                continue
            try:
                chart = parse_event_chart(source_path, index)
                inspect_chart(chart)
                parsed = parse_event_notes(chart.points, chart.tempo_changes, first=chart.audio_first)
                if not parsed.syntax_valid or not parsed.notes:
                    skipped["chart_invalid_or_empty"] += 1
                    continue
                quality = assess_event_chart(chart.points, chart.tempo_changes, first=chart.audio_first)
            except (OSError, ValueError, TypeError, ZeroDivisionError):
                skipped["chart_invalid_or_empty"] += 1
                continue
            try:
                audio_hash = _sha256_file(audio_path, audio_hash_cache)
            except OSError:
                skipped["audio_unreadable"] += 1
                continue
            except ValueError:
                skipped["audio_invalid"] += 1
                continue
            if audio_hash in forbidden_audio_hashes:
                skipped["training_audio_hash_excluded"] += 1
                continue
            if audio_hash in seen_audio_hashes:
                skipped["duplicate_audio_content"] += 1
                continue
            seen_audio_hashes.add(audio_hash)
            total_duration = max(
                float(seconds_at_beat(chart.end_beat, chart.tempo_changes, chart.audio_first)
                      - chart.audio_first),
                1e-6,
            )
            types = quality.get("type_ratios") or {}
            candidates.append({
                "song_id": str(song.get("song_id", source_path.parent.name)),
                "title": str(song.get("title") or chart.title or source_path.parent.name),
                "difficulty_index": index,
                "level": level,
                "maidata_path": str(source_path),
                "audio_path": str(audio_path),
                "audio_sha256": audio_hash,
                "note_count": len(parsed.notes),
                "note_density_per_sec": len(parsed.notes) / total_duration,
                "hold_ratio": types.get("hold", 0.0) + types.get("touch_hold", 0.0),
                "touch_ratio": types.get("touch", 0.0) + types.get("touch_hold", 0.0),
                "slide_ratio": types.get("slide", 0.0),
                "chord_ratio": quality.get("chord_ratio", 0.0),
                "fine_note_ratio": fine_note_ratio(note.beat for note in parsed.notes),
                "tempo_map": [{"beat": str(change.beat), "bpm": change.bpm}
                              for change in chart.tempo_changes],
                "first_sec": chart.first,
                "time_offset_sec": chart.time_offset,
                "audio_first_sec": chart.audio_first,
                "end_beat": str(chart.end_beat),
            })
    if not candidates:
        raise ValueError("No readable event-usable charts matched difficulty 4..6 and level 12..15.")

    selected: list[dict[str, Any]] = []
    used_songs: set[str] = set()
    criteria = [
        ("lowest_density", "note_density_per_sec", False, False),
        ("highest_density", "note_density_per_sec", True, False),
        ("highest_hold_ratio", "hold_ratio", True, True),
        ("highest_touch_ratio", "touch_ratio", True, True),
        ("highest_slide_ratio", "slide_ratio", True, True),
        ("highest_chord_ratio", "chord_ratio", True, True),
        ("highest_fine_ratio", "fine_note_ratio", True, True),
    ]

    def add(record: dict[str, Any], basis: str) -> None:
        if record["song_id"] in used_songs or len(selected) >= size:
            return
        used_songs.add(record["song_id"])
        selected.append({**record, "selection_basis": basis})

    for basis, metric, reverse, require_positive in criteria:
        eligible = [item for item in candidates if not require_positive or item[metric] > 0]
        ordered = sorted(eligible, key=lambda item: (item[metric], item["song_id"], item["difficulty_index"]),
                         reverse=reverse)
        record = next((item for item in ordered if item["song_id"] not in used_songs), None)
        if record is not None:
            add(record, basis)
    density_values = sorted(item["note_density_per_sec"] for item in candidates)
    for basis, fraction in (("density_quantile_25", 0.25), ("density_quantile_50", 0.5),
                            ("density_quantile_75", 0.75)):
        target = density_values[round(fraction * (len(density_values) - 1))]
        record = min(
            (item for item in candidates if item["song_id"] not in used_songs),
            key=lambda item: (abs(item["note_density_per_sec"] - target), item["song_id"]),
            default=None,
        )
        if record is not None:
            add(record, basis)
    for record in sorted(candidates, key=lambda item: (item["song_id"], item["difficulty_index"])):
        if len(selected) >= size:
            break
        add(record, "stable_fill")
    return {
        "schema": "maichart-rhythm-evaluation-panel-v1",
        "source_manifest": str(manifest_path),
        "split_report": str(split_report_path) if split_report_path is not None else None,
        "candidate_filter": {"event_usable_for_training": True, "difficulty_indices": [4, 5, 6],
                             "level_min": 12, "level_max": 15},
        "selection_policy": "distinct songs selected by measured density/type-ratio extrema and density quantiles; no subjective tags",
        "requested_size": size,
        "candidate_chart_count": len(candidates),
        "selected_song_count": len(selected),
        "skipped_counts": dict(sorted(skipped.items())),
        "audio_hash_audit": {
            "hash_algorithm": "sha256",
            "distinct_candidate_audio_hashes": len(seen_audio_hashes),
            "train_hashes_checked": len(forbidden_audio_hashes),
            "excluded_training_audio_candidates": skipped.get("training_audio_hash_excluded", 0),
            "selected_overlap_count": len({item["audio_sha256"] for item in selected} & forbidden_audio_hashes),
            "status": "disjoint_from_training" if split_report is not None else "not_checked",
        },
        "songs": selected,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, help="chart JSON files or directories of generated charts")
    parser.add_argument("--reference-dir", type=Path,
                        help="optional paired reference directory with matching relative filenames")
    parser.add_argument("--panel-manifest", type=Path,
                        help="also choose a fixed, measured-feature panel from an event training manifest")
    parser.add_argument("--panel-output", type=Path,
                        help="write the optional panel JSON here")
    parser.add_argument("--panel-split-report", type=Path,
                        help="verify the panel manifest against this split report and exclude training audio hashes")
    parser.add_argument("--panel-size", type=int, default=10,
                        help="number of distinct songs in the optional panel (default: 10)")
    parser.add_argument("-o", "--output", type=Path, help="JSON report path; defaults to stdout")
    args = parser.parse_args(argv)
    if not args.inputs and not args.panel_manifest:
        parser.error("provide chart inputs or --panel-manifest")
    try:
        report = (evaluate_paths(args.inputs, reference_dir=args.reference_dir) if args.inputs else {
            "schema": "maichart-rhythm-evaluation-v1",
            "metric_policy": "descriptive metrics only; no aggregate quality score",
            "input_count": 0, "evaluated_count": 0, "error_count": 0,
            "reference_pairing": {"status": "not_requested", "matched": 0, "missing": 0},
            "charts": [], "errors": [],
        })
        panel = (build_fixed_panel(args.panel_manifest, size=args.panel_size,
                                   split_report_path=args.panel_split_report)
                 if args.panel_manifest else None)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if panel is not None:
        report["fixed_panel"] = panel
        if args.panel_output:
            args.panel_output.parent.mkdir(parents=True, exist_ok=True)
            args.panel_output.write_text(json.dumps(panel, ensure_ascii=False, indent=2, allow_nan=False),
                                         encoding="utf-8")
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized)
    return 1 if report["error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
