"""Select a small, tagged manifest for fast training smoke tests."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


def build_small_batch_manifest(
    manifest_path: str | Path,
    *,
    cache_dir: str | Path,
    output_path: str | Path,
    target_count: int = 24,
    seed: int = 7,
) -> dict[str, Any]:
    source_path = Path(manifest_path).resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    candidates: list[dict[str, Any]] = []
    warnings: list[str] = []
    missing_fields: Counter[str] = Counter()
    for song in source.get("songs", []):
        if not isinstance(song, dict):
            continue
        music_id = song.get("song_id") or song.get("music_id")
        if not music_id:
            missing_fields["song_id"] += 1
            warnings.append("Skipped song with missing song_id")
            continue
        for difficulty in song.get("difficulties", []):
            if not isinstance(difficulty, dict):
                continue
            index = difficulty.get("difficulty_index", difficulty.get("index"))
            if index is None:
                missing_fields["difficulty_index"] += 1
                warnings.append(f"Skipped {music_id}: missing difficulty_index")
                continue
            if difficulty.get("usable_for_training") is False:
                warnings.append(f"Skipped {music_id} difficulty {index}: marked unusable")
                continue
            level = _level(difficulty.get("level"))
            warning_codes = difficulty.get("warning_codes") or []
            tags = _tags(level, difficulty.get("type_counts") or {}, warning_codes)
            for code in warning_codes:
                warnings.append(f"{music_id} difficulty {index}: {code}")
            sample_id = f"{music_id}_d{index}"
            candidates.append(
                {
                    "sample_id": sample_id,
                    "music_id": str(music_id),
                    "difficulty": int(index),
                    "level": level,
                    "title": song.get("title", ""),
                    "tags": tags,
                    "selection_reason": "eligible difficulty with cached chart and audio references",
                    "paths": {
                        "chart_ir": difficulty.get("chart_ir_path"),
                        "frame_labels": difficulty.get("frame_labels_path"),
                        "alignment_report": difficulty.get("alignment_report_path"),
                        "audio_features": (song.get("audio") or {}).get("audio_features_path"),
                    },
                }
            )

    rng = random.Random(seed)
    rng.shuffle(candidates)
    samples = candidates[: max(0, int(target_count))]
    if len(candidates) <= target_count:
        samples = candidates
    result = {
        "schema": "maichart-small-batch-manifest-v1",
        "source_manifest": str(source_path),
        "cache_dir": str(Path(cache_dir).resolve()),
        "samples": samples,
        "actual_sample_count": len(samples),
        "metadata": {
            "candidate_count": len(candidates),
            "usable_candidate_count": len(candidates),
            "target_count": int(target_count),
            "seed": int(seed),
            "warnings": warnings,
            "missing_fields": dict(missing_fields),
        },
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def _level(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _tags(level: float | None, type_counts: dict[str, Any], warning_codes: list[Any]) -> list[str]:
    tags: list[str] = []
    numeric_level = level or 0.0
    if numeric_level < 8:
        tags.append("low_difficulty")
    elif numeric_level < 12:
        tags.append("medium_difficulty")
    else:
        tags.append("high_difficulty")
    total = sum(_count(value) for value in type_counts.values()) or 1
    if _count(type_counts.get("slide")) / total >= 0.2:
        tags.append("slide_heavy")
    if _count(type_counts.get("hold")) / total >= 0.2:
        tags.append("hold_heavy")
    tags.extend(str(code) for code in warning_codes if str(code))
    return list(dict.fromkeys(tags))


def _count(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0
