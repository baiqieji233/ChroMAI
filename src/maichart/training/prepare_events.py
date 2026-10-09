"""Build an event-training manifest from a raw chart dataset.

Why this module exists
----------------------
``event_dataset.EventDataset`` reads a manifest whose songs carry
``maidata_path`` and ``audio_path``, and the WebUI dispatches event
preprocessing to ``python -m maichart.training.prepare_events``.  That entry
point did not exist, so the WebUI's "预处理训练集" button failed with
``No module named maichart.training.prepare_events`` and no fresh event manifest
could be produced.

The event path deliberately does not reuse the fixed-frame preprocessor: it
needs no ``division``, no frame labels and no alignment reports.  It parses each
chart into exact beat events itself, which is what makes ``{16}`` and ``{12}``
region changes representable at all.

What the manifest must contain
------------------------------
``preprocessing.target_schema == "event_v1"`` and
``preprocessing.feature_profile == "event_audio_v2_rhythm"``.  The WebUI branches on
those two fields to decide whether a manifest is an event manifest; without them
it treats an event dataset as the legacy fixed-frame format and rejects it.
Paths are stored relative to the manifest so the dataset can be moved, and each
difficulty carries an explicit ``event_usable_for_training`` flag so the WebUI
does not have to fall back to the fixed-frame ``usable_for_training`` field.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any

from maichart.event_timing import parse_event_chart
from maichart.maidata import parse_maidata_metadata
from maichart.preprocess import (
    AUDIO_FILENAMES,
    _discover_sample_dirs,
    _raw_song_id,
)

# Master and Re:Master only.  The event vocabulary is fitted on authored expert
# charts, and the dataset loader applies the same two filters, so applying them
# here keeps the reported usable count equal to what training will actually use.
MIN_DIFFICULTY_INDEX = 4
MIN_LEVEL = 12.0

MANIFEST_SCHEMA = "maichart-training-manifest-v1"
PREPROCESSING_SCHEMA = "maichart-preprocessing-contract-v1"

DIFFICULTY_NAMES = {0: "basic", 1: "advanced", 2: "expert", 3: "master", 4: "remaster"}


def _numeric_level(value: str | None) -> float:
    """Convert an authored level such as ``13+`` to a sortable number."""

    if value is None:
        raise ValueError("missing level")
    text = str(value).strip()
    if not text:
        raise ValueError("empty level")
    # ``13+`` is one step above ``13`` but below ``14``; the convention used
    # across this project is +0.7, matching event_dataset.numeric_level.
    if text.endswith("+"):
        return float(text[:-1]) + 0.7
    return float(text)


def _display_path(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _inspect_difficulty(source: Path, index: int) -> dict[str, Any]:
    """Parse one difficulty and report whether it can become training data."""

    record: dict[str, Any] = {
        "difficulty_index": index,
        "difficulty_name": DIFFICULTY_NAMES.get(index, f"index_{index}"),
        "level_raw": None,
        "level": None,
        "designer": None,
        "has_chart": False,
        "note_count": None,
        "end_beat": None,
        "tempo_changes": None,
        "event_usable_for_training": False,
        "filter_reasons": [],
        "error": None,
    }
    reasons: list[str] = []
    try:
        chart = parse_event_chart(source, index)
    except Exception as exc:  # noqa: BLE001 - every failure is a QC record
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["filter_reasons"] = ["parse_failed"]
        return record

    record["has_chart"] = True
    record["note_count"] = len(chart.points)
    record["end_beat"] = str(chart.end_beat)
    record["tempo_changes"] = len(chart.tempo_changes)
    if chart.points:
        # The event timebase is the authored tempo, which is also what training
        # conditions on, so surface it for review.
        record["initial_bpm"] = float(chart.tempo_changes[0].bpm) if chart.tempo_changes else None
    if not chart.points:
        reasons.append("empty_chart")
    if record["note_count"] is not None and record["note_count"] < 50:
        reasons.append("note_count_too_low")
    record["filter_reasons"] = reasons
    record["event_usable_for_training"] = not reasons
    return record


def _inspect_song(directory: Path, root: Path, *, limit: int | None) -> dict[str, Any] | None:
    maidata = directory / "maidata.txt"
    if not maidata.is_file():
        return None
    audio = next(
        (candidate for name in AUDIO_FILENAMES if (candidate := directory / name).is_file()),
        None,
    )
    song_id = _raw_song_id(directory, None)
    try:
        metadata = parse_maidata_metadata(maidata.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:  # noqa: BLE001
        return {
            "song_id": song_id,
            "title": None,
            "artist": None,
            "maidata_path": _display_path(maidata, root),
            "audio_path": _display_path(audio, root) if audio else None,
            "wholebpm": None,
            "dataset_usable": False,
            "audio": {"readable": False, "status": "unreadable_maidata", "error": str(exc)},
            "difficulties": [],
        }

    difficulties: list[dict[str, Any]] = []
    for block in metadata.difficulties:
        if limit is not None and len(difficulties) >= limit:
            break
        record = _inspect_difficulty(maidata, block.index)
        record["level_raw"] = block.level
        record["designer"] = block.designer
        try:
            level = _numeric_level(block.level)
        except ValueError:
            level = None
        record["level"] = level
        # The dataset loader applies these two filters itself, so mirror them
        # here rather than showing difficulties that will silently vanish.
        if block.index < MIN_DIFFICULTY_INDEX:
            record["event_usable_for_training"] = False
            record["filter_reasons"] = sorted(set(record["filter_reasons"]) | {"difficulty_index_below_master"})
        elif level is None:
            record["event_usable_for_training"] = False
            record["filter_reasons"] = sorted(set(record["filter_reasons"]) | {"level_not_numeric"})
        elif level < MIN_LEVEL:
            record["event_usable_for_training"] = False
            record["filter_reasons"] = sorted(set(record["filter_reasons"]) | {"level_below_12"})
        if audio is None:
            record["event_usable_for_training"] = False
            record["filter_reasons"] = sorted(set(record["filter_reasons"]) | {"audio_missing"})
        difficulties.append(record)

    usable = any(item["event_usable_for_training"] for item in difficulties)
    return {
        "song_id": song_id,
        "title": metadata.title,
        "artist": metadata.artist,
        "maidata_path": _display_path(maidata, root),
        "audio_path": _display_path(audio, root) if audio else None,
        "wholebpm": metadata.wholebpm,
        "dataset_usable": usable,
        "audio": {
            "readable": audio is not None,
            "status": "source_present" if audio is not None else "missing",
            "path": _display_path(audio, root) if audio else None,
        },
        "difficulties": difficulties,
    }


def build_event_manifest(
    raw_root: Path,
    *,
    cache_dir: Path,
    sample_rate: int = 22050,
    limit: int | None = None,
) -> dict[str, Any]:
    directories = _discover_sample_dirs(raw_root)
    songs: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for directory in directories:
        if limit is not None and len(songs) >= limit:
            break
        try:
            song = _inspect_song(directory, raw_root, limit=None)
        except Exception as exc:  # noqa: BLE001
            errors.append({"stage": "inspect_song", "path": str(directory), "message": str(exc)})
            continue
        if song is not None:
            songs.append(song)

    difficulties = [item for song in songs for item in song["difficulties"]]
    usable = [item for item in difficulties if item["event_usable_for_training"]]
    reason_counts = Counter(
        reason for item in difficulties for reason in item.get("filter_reasons", [])
    )
    return {
        "schema": MANIFEST_SCHEMA,
        "source_root": raw_root.resolve().as_posix(),
        "cache_dir": cache_dir.as_posix(),
        "song_count": len(songs),
        "difficulty_count": len(difficulties),
        "usable_difficulty_count": len(usable),
        "preprocessing": {
            "schema": PREPROCESSING_SCHEMA,
            # These two fields are what make the WebUI recognise this as an event
            # manifest instead of the legacy fixed-frame format.
            "target_schema": "event_v1",
            "feature_profile": "event_audio_v2_rhythm",
            "sample_rate": int(sample_rate),
            "hop_length": 441,
            "feature_rate_hz": 50,
            "min_difficulty_index": MIN_DIFFICULTY_INDEX,
            "min_level": MIN_LEVEL,
        },
        "songs": songs,
        "summary": {
            "raw_sample_count": len(directories),
            "dataset_usable_song_count": sum(1 for song in songs if song["dataset_usable"]),
            "difficulty_count": len(difficulties),
            "usable_difficulty_count": len(usable),
            "filtered_difficulty_count": len(difficulties) - len(usable),
            "filter_reason_counts": dict(reason_counts),
        },
        "warnings": [],
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build an event_v1 training manifest from a raw chart dataset.",
    )
    parser.add_argument("raw_root", help="raw dataset root containing per-song maidata.txt")
    parser.add_argument("--cache-dir", required=True, help="cache directory for audio features")
    parser.add_argument("-o", "--output", required=True, help="output manifest JSON path")
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument(
        "--force", action="store_true",
        help=(
            "accepted for CLI compatibility with the WebUI; this preprocessor "
            "writes no caches of its own, and the derived event-audio caches "
            "invalidate themselves when the feature or tempo contract changes"
        ),
    )
    parser.add_argument("--limit", type=int, help="limit samples for debugging")
    args = parser.parse_args(argv)

    raw_root = Path(args.raw_root).resolve()
    if not raw_root.exists():
        print(f"[prepare_events] raw root does not exist: {raw_root}", file=sys.stderr)
        return 2
    output = Path(args.output).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Paths are stored relative to the manifest so the dataset survives a move;
    # the loader resolves them against the manifest's own directory.
    manifest = build_event_manifest(
        raw_root, cache_dir=cache_dir, sample_rate=int(args.sample_rate), limit=args.limit,
    )
    base = output.parent
    for song in manifest["songs"]:
        for key in ("maidata_path", "audio_path"):
            value = song.get(key)
            if value:
                song[key] = _relative_to(value, raw_root, base)
        for difficulty in song["difficulties"]:
            if difficulty.get("error"):
                continue
    manifest["source_root"] = _relative_or_absolute(raw_root, base)
    manifest["cache_dir"] = _relative_or_absolute(cache_dir, base)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = manifest["summary"]
    print(
        f"[prepare_events] songs={manifest['song_count']} "
        f"difficulties={manifest['difficulty_count']} "
        f"usable={summary['usable_difficulty_count']} "
        f"filtered={summary['filtered_difficulty_count']} "
        f"output={output}",
        flush=True,
    )
    for reason, count in Counter(summary["filter_reason_counts"]).most_common(8):
        print(f"[prepare_events]   filtered {reason}: {count}", flush=True)
    return 0


def _relative_to(value: str, raw_root: Path, base: Path) -> str:
    """Express a raw-root-relative path relative to the output manifest."""

    candidate = (raw_root / value).resolve() if not Path(value).is_absolute() else Path(value)
    return _relative_or_absolute(candidate, base)


def _relative_or_absolute(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base).as_posix()
    except ValueError:
        return path.resolve().as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
