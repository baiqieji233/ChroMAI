from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_v25_fixture(root: Path, *, sample_count: int = 2) -> Path:
    """Write a small, self-contained V2.5 manifest and cache tree."""
    if sample_count not in {1, 2}:
        raise ValueError("sample_count must be 1 or 2")

    cache = root / "cache"
    songs: list[dict[str, Any]] = []
    for sample_index in range(sample_count):
        song_id = f"song{sample_index + 1:03d}"
        difficulty = 4 + sample_index
        audio_path = cache / "audio_features" / f"{song_id}.audio_features.json"
        labels_path = cache / "frame_labels" / song_id / f"difficulty_{difficulty}.frame_labels.json"
        align_path = cache / "alignment_reports" / song_id / f"difficulty_{difficulty}.alignment_report.json"
        chart_path = cache / "chart_ir" / song_id / f"difficulty_{difficulty}.chart_ir.json"
        _write_json(
            audio_path,
            {
                "schema": "maichart-audio-features-v1",
                "feature_frames": [_audio_frame(index, index * 0.1 + sample_index * 0.01) for index in range(4)],
            },
        )
        _write_json(
            labels_path,
            {
                "schema": "maichart-frame-labels-v1",
                "song_id": song_id,
                "difficulty": difficulty,
                "frames": [
                    _label_frame(0, 0.0, False, [], {}),
                    _label_frame(1, 0.1, True, ["1", "3"] if sample_index == 0 else ["4", "6"], {"tap_count": 1}),
                    _label_frame(2, 0.2, True, ["2"], {"hold_start_count": 1}),
                    _label_frame(3, 0.3, False, [], {}),
                ],
            },
        )
        _write_json(align_path, {"schema": "maichart-alignment-report-v1", "summary": {}})
        _write_json(chart_path, {"schema_version": 1, "notes": []})
        songs.append(
            {
                "song_id": song_id,
                "audio": {"audio_features_path": _rel(audio_path, root)},
                "difficulties": [
                    {
                        "difficulty_index": difficulty,
                        "level": 12.0 + sample_index,
                        "usable_for_training": True,
                        "frame_labels_path": _rel(labels_path, root),
                        "alignment_report_path": _rel(align_path, root),
                        "chart_ir_path": _rel(chart_path, root),
                    }
                ],
            }
        )

    manifest_path = root / "manifest.json"
    _write_json(
        manifest_path,
        {
            "schema": "maichart-training-manifest-v1",
            "song_count": sample_count,
            "difficulty_count": sample_count,
            "usable_difficulty_count": sample_count,
            "songs": songs,
        },
    )
    return manifest_path


def _audio_frame(index: int, time_sec: float) -> dict[str, float | int]:
    return {
        "frame_index": index,
        "time_sec": time_sec,
        "onset_strength": 0.1 * index,
        "rms": 0.01 * (index + 1),
        "percussive_rms": 0.02 * (index + 1),
        "harmonic_rms": 0.03 * (index + 1),
        "spectral_centroid": 1000.0 + index,
        "spectral_bandwidth": 2000.0 + index,
        "zero_crossing_rate": 0.05,
    }


def _label_frame(
    index: int,
    time_sec: float,
    has_note: bool,
    positions: list[str],
    counts: dict[str, int],
) -> dict[str, Any]:
    labels: dict[str, Any] = {
        "has_note": has_note,
        "note_count": 1 if has_note else 0,
        "tap_count": 0,
        "break_count": 0,
        "hold_start_count": 0,
        "hold_active_count": 0,
        "slide_start_count": 0,
        "slide_active_count": 0,
        "touch_count": 0,
        "touch_hold_start_count": 0,
        "positions": positions,
    }
    labels.update(counts)
    return {
        "frame_index": index,
        "beat": str(index),
        "tick": index * 480,
        "time_sec": time_sec,
        "labels": labels,
    }


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()
