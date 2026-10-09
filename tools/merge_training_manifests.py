"""Merge compatible preprocessing manifests for one generator run."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


SCHEMA = "maichart-training-manifest-v1"
PATH_KEYS = ("maidata_path", "audio_path", "convert_report_path")


def merge_training_manifests(
    manifest_paths: list[str | Path],
    output_path: str | Path,
) -> dict[str, Any]:
    """Merge manifests only when their frame/audio contracts agree."""

    if not manifest_paths:
        raise ValueError("At least one training manifest is required.")
    loaded = []
    for raw_path in manifest_paths:
        path = Path(raw_path).resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != SCHEMA:
            raise ValueError(f"{path} is not a {SCHEMA}.")
        loaded.append((path, payload))

    contracts = [_contract(payload, manifest_path) for manifest_path, payload in loaded]
    reference = contracts[0]
    for path, contract in zip((item[0] for item in loaded), contracts):
        differences = [
            f"{key}: {reference[key]!r} != {contract[key]!r}"
            for key in reference
            if reference[key] is not None and contract[key] is not None and reference[key] != contract[key]
        ]
        if differences:
            raise ValueError(
                f"Cannot merge {path}: preprocessing contracts differ ({'; '.join(differences)}). "
                "Use the same division, sample rate, and feature/label schema."
            )

    output = Path(output_path).resolve()
    songs: list[dict[str, Any]] = []
    seen_song_ids: set[str] = set()
    for manifest_path, payload in loaded:
        for song in payload.get("songs", []):
            copied = copy.deepcopy(song)
            song_id = str(copied.get("song_id") or "unknown")
            if song_id in seen_song_ids:
                raise ValueError(f"Duplicate song_id {song_id!r}; rename or deduplicate before merging.")
            seen_song_ids.add(song_id)
            _absolutize_song_paths(copied, manifest_path.parent)
            songs.append(copied)

    merged = {
        "schema": SCHEMA,
        "source_root": "<merged>",
        "cache_dir": str(output.parent),
        "song_count": len(songs),
        "difficulty_count": sum(len(song.get("difficulties", [])) for song in songs),
        "usable_difficulty_count": sum(
            1
            for song in songs
            for difficulty in song.get("difficulties", [])
            if difficulty.get("usable_for_training")
        ),
        "preprocessing": {
            key: reference[key]
            for key in reference
            if reference[key] is not None
        },
        "merged_from": [str(path) for path, _ in loaded],
        "songs": songs,
        "summary": {
            "merged_manifest_count": len(loaded),
            "song_count": len(songs),
            "difficulty_count": sum(len(song.get("difficulties", [])) for song in songs),
            "usable_difficulty_count": sum(
                1
                for song in songs
                for difficulty in song.get("difficulties", [])
                if difficulty.get("usable_for_training")
            ),
        },
        "warnings": [],
        "errors": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return merged


def _contract(payload: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    metadata = payload.get("preprocessing") if isinstance(payload.get("preprocessing"), dict) else {}
    first_audio = _first_cache_payload(payload, manifest_path, "audio_features_path")
    first_labels = _first_cache_payload(payload, manifest_path, "frame_labels_path")
    return {
        "division": metadata.get("division") or _value(first_audio, "division") or _frame_division(first_labels),
        "sample_rate": metadata.get("sample_rate") or _value(first_audio, "sample_rate"),
        "audio_features_schema": metadata.get("audio_features_schema") or _value(first_audio, "schema"),
        "frame_labels_schema": metadata.get("frame_labels_schema") or _value(first_labels, "schema"),
        "audio_feature_keys": tuple(metadata.get("audio_feature_keys", ())),
    }


def _first_cache_payload(payload: dict[str, Any], manifest_path: Path, key: str) -> dict[str, Any]:
    for song in payload.get("songs", []):
        audio = song.get("audio") if isinstance(song.get("audio"), dict) else {}
        raw_path = audio.get(key) or song.get(key)
        if not raw_path:
            for difficulty in song.get("difficulties", []):
                raw_path = difficulty.get(key)
                if raw_path:
                    break
        if raw_path:
            path = Path(raw_path)
            if not path.is_absolute():
                path = manifest_path.parent / path
            if path.is_file():
                try:
                    value = json.loads(path.read_text(encoding="utf-8"))
                    return value if isinstance(value, dict) else {}
                except (OSError, json.JSONDecodeError):
                    return {}
    return {}


def _value(payload: dict[str, Any], key: str) -> Any:
    return payload.get(key) if isinstance(payload, dict) else None


def _frame_division(payload: dict[str, Any]) -> int | None:
    frame_grid = payload.get("frame_grid") if isinstance(payload, dict) else None
    value = frame_grid.get("division") if isinstance(frame_grid, dict) else None
    return int(value) if value is not None else None


def _absolutize_song_paths(song: dict[str, Any], base: Path) -> None:
    for key in PATH_KEYS:
        if song.get(key):
            song[key] = str(_absolute(song[key], base))
    audio = song.get("audio")
    if isinstance(audio, dict) and audio.get("audio_features_path"):
        audio["audio_features_path"] = str(_absolute(audio["audio_features_path"], base))
    for difficulty in song.get("difficulties", []):
        if not isinstance(difficulty, dict):
            continue
        for key in ("chart_ir_path", "frame_labels_path", "alignment_report_path"):
            if difficulty.get(key):
                difficulty[key] = str(_absolute(difficulty[key], base))


def _absolute(value: str | Path, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+", help="compatible training manifest JSON files")
    parser.add_argument("-o", "--output", required=True, help="merged manifest output path")
    args = parser.parse_args(argv)
    result = merge_training_manifests(args.manifests, args.output)
    print(
        f"merged={len(result['merged_from'])} songs={result['song_count']} "
        f"difficulties={result['difficulty_count']} usable={result['usable_difficulty_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
