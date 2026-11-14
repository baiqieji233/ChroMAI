"""Merge compatible preprocessing manifests for one generator run."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import tempfile
import time
import threading
from pathlib import Path
from typing import Any
from maichart.portable_paths import material_path


SCHEMA = "maichart-training-manifest-v1"
PATH_KEYS = ("maidata_path", "audio_path", "convert_report_path")
_COMPOSE_LOCK = threading.RLock()
SONG_FORM_V2_AUDIO_PROFILE = "rational_audio_queries_v1"
SONG_FORM_V2_IGNORED_AUDIO_CONTRACT_KEYS = frozenset({
    "feature_profile",
    "hop_length",
    "feature_rate_hz",
    "division",
    "sample_rate",
    "audio_features_schema",
    "frame_labels_schema",
    "audio_feature_keys",
})
SONG_FORM_V2_LEGACY_AUDIO_METADATA_KEYS = frozenset({
    *SONG_FORM_V2_IGNORED_AUDIO_CONTRACT_KEYS,
    "input_dim",
    "feature_config",
    "audio_feature_dim",
    "feature_dim",
})


def _replace_file(temporary, path):
    for attempt in range(8):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if os.name != "nt" or attempt == 7:
                raise
            # A concurrent JSON reader can briefly block rename on Windows.
            time.sleep(.02 * (attempt + 1))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.stem + "_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(payload, output, ensure_ascii=False, indent=2)
        _replace_file(temporary, path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def _source_root(manifest: Path, data: dict[str, Any]) -> Path:
    value = str(data.get("source_root") or ".")
    if value.startswith("<"):
        return manifest.parent
    path = material_path(value)
    return (path if path.is_absolute() else manifest.parent / path).resolve()


def _source_asset(value: Any, manifest: Path, data: dict[str, Any]) -> Path:
    path = material_path(str(value))
    if path.is_absolute():
        return path.resolve()
    bases = (manifest.parent, _source_root(manifest, data), manifest.parent.parent)
    return next(((base / path).resolve() for base in bases if (base / path).exists()), (bases[0] / path).resolve())


def _source_asset_cache():
    resolved = {}

    def asset(value, manifest, data):
        raw = material_path(str(value))
        key = (None if raw.is_absolute() else manifest, raw)
        if key not in resolved:
            path = _source_asset(raw, manifest, data)
            resolved[key] = path
            resolved.setdefault((None, path), path)
        return resolved[key]

    return asset


def manifest_dataset_id(manifest_path: str | Path, payload: dict[str, Any] | None = None) -> str:
    """Return an internal source identity, separate from its display name."""
    data = payload or {}
    for key in ("dataset_id", "source_dataset"):
        value = data.get(key)
        if isinstance(value, dict):
            value = value.get("id") or value.get("key")
        elif key == "source_dataset":
            value = None
        value = str(value or "").strip()
        if value:
            return value
    manifest = Path(manifest_path).resolve()
    return "ds-" + hashlib.sha1(str(_source_root(manifest, data)).casefold().encode("utf-8")).hexdigest()[:12]


def manifest_dataset_name(manifest_path: str | Path, payload: dict[str, Any] | None = None) -> str:
    data = payload or {}
    for key in ("dataset_name", "source_dataset", "name"):
        value = data.get(key)
        if isinstance(value, dict):
            value = value.get("name") or value.get("label")
        value = str(value or "").strip()
        if value:
            return value
    manifest = Path(manifest_path).resolve()
    root = _source_root(manifest, data)
    return root.name or manifest.stem


def _quality_annotation_path(manifest: Path) -> Path:
    return manifest.with_name("quality_annotations.json")


def _namespaced_chart_id(song_id: str, difficulty_index: int, chart_hash: str,
                         audio_hash: Any, namespace: str) -> str:
    payload = f"{namespace}\0{song_id}\0{int(difficulty_index)}\0{chart_hash}\0{audio_hash or ''}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _extract_zero_unmarked_contributions(record: dict[str, Any]) -> tuple[float, list[dict[str, Any]]] | None:
    from maichart.training.quality_annotations import (
        DEFAULT_WEIGHTS, LABELS, _validate_regions, zero_unmarked_region_weights,
    )
    if not isinstance(record, dict):
        return None
    label = str(record.get("label", "normal"))
    if label not in LABELS:
        raise ValueError(f"未知质量标签: {label}")
    if label in {"excluded", "negative"}:
        return None
    try:
        outer_weight = float(record.get("weight", DEFAULT_WEIGHTS[label]))
    except (TypeError, ValueError) as exc:
        raise ValueError("weight 必须是有限数字") from exc
    if not math.isfinite(outer_weight) or outer_weight < 0 or outer_weight > 4:
        raise ValueError("weight 必须在 0–4 之间")
    if outer_weight <= 0:
        return None
    raw_regions = record.get("regions", [])
    if not isinstance(raw_regions, list):
        raise ValueError("regions 必须是数组")
    validated = _validate_regions(raw_regions)
    contribs: list[dict[str, Any]] = []
    for raw_row, val_row in zip(raw_regions, validated):
        t, c = zero_unmarked_region_weights(val_row)
        eff_t = outer_weight * t
        eff_c = outer_weight * c
        if not (eff_t > 0 or eff_c > 0):
            continue
        entry_start = val_row.get("entry_start_sec") if eff_t > 0 and "entry_start_sec" in val_row else None
        contribs.append({
            "start_sec": float(val_row["start_sec"]),
            "end_sec": float(val_row["end_sec"]),
            "eff_timing": eff_t,
            "eff_component": eff_c,
            "entry_start_sec": float(entry_start) if entry_start is not None else None,
        })
    if not contribs:
        return None
    return outer_weight, contribs


def _merge_zero_unmarked_contributions(contribs: list[dict[str, Any]]) -> tuple[float, list[dict[str, Any]]]:
    from maichart.training.quality_annotations import _validate_regions
    bounds = sorted({float(row[key]) for row in contribs for key in ("start_sec", "end_sec")})
    raw_spans: list[dict[str, Any]] = []
    peak_eff = 0.0
    for start, end in zip(bounds, bounds[1:]):
        covering = [row for row in contribs if row["start_sec"] <= start and end <= row["end_sec"]]
        if not covering:
            continue
        eff_t = max((float(row["eff_timing"]) for row in covering), default=0.0)
        eff_c = max((float(row["eff_component"]) for row in covering), default=0.0)
        if not (eff_t > 0 or eff_c > 0):
            continue
        peak_eff = max(peak_eff, eff_t, eff_c)
        entries = [
            float(row["entry_start_sec"])
            for row in covering
            if row.get("entry_start_sec") is not None and float(row["eff_timing"]) > 0
            and float(row["entry_start_sec"]) <= start + 1e-9
        ]
        raw_spans.append({
            "start_sec": start,
            "end_sec": end,
            "eff_timing": eff_t,
            "eff_component": eff_c,
            "entry_start_sec": min(entries) if entries and eff_t > 0 else None,
        })
    outer_weight = 1.0 if peak_eff <= 16.0 else min(4.0, peak_eff / 16.0)
    result: list[dict[str, Any]] = []
    for span in raw_spans:
        t_scaled = span["eff_timing"] / outer_weight
        c_scaled = span["eff_component"] / outer_weight
        reg_weight = min(4.0, max(1.0, max(t_scaled, c_scaled) / 4.0))
        merged_row: dict[str, Any] = {
            "start_sec": span["start_sec"],
            "end_sec": span["end_sec"],
            "label": "good",
            "reason": "",
            "weight": reg_weight,
            "timing_weight": min(4.0, t_scaled / reg_weight),
            "component_weight": min(4.0, c_scaled / reg_weight),
        }
        if span["entry_start_sec"] is not None:
            merged_row["entry_start_sec"] = span["entry_start_sec"]
        if (
            result
            and result[-1]["end_sec"] == span["start_sec"]
            and {k: v for k, v in result[-1].items() if k not in ("start_sec", "end_sec")}
            == {k: v for k, v in merged_row.items() if k not in ("start_sec", "end_sec")}
        ):
            result[-1]["end_sec"] = span["end_sec"]
        else:
            result.append(merged_row)
    return outer_weight, _validate_regions(result)


def merge_training_manifests(
    manifest_paths: list[str | Path],
    output_path: str | Path,
    *,
    dataset_ids: list[str] | None = None,
    dataset_names: list[str] | None = None,
    merge_annotations: bool = True,
    model_family: str | None = None,
    _composition_identity: str | None = None,
    _source_versions: dict[str, Any] | None = None,
    _published_root: Path | None = None,
    _preview_audio_digest=None,
    _resolve_asset=_source_asset,
) -> dict[str, Any]:
    """Merge compatible manifests for one training run.

    Paths in the resulting manifest are absolute, while every song keeps
    ``source_dataset``, ``source_manifest``, ``original_song_id`` and
    ``source_path_key``.  The latter is important: quality annotations made
    against an individual source manifest continue to resolve after the
    merged manifest is moved elsewhere.  Duplicate ``maidata + difficulty``
    entries are ignored deterministically; this handles the common case where
    a curated set is a subset of the full corpus without silently doubling a
    chart's training weight.
    """

    if not manifest_paths:
        raise ValueError("At least one training manifest is required.")
    loaded = []
    for position, raw_path in enumerate(manifest_paths):
        path = Path(raw_path).resolve()
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if payload.get("schema") != SCHEMA:
            raise ValueError(f"{path} is not a {SCHEMA}.")
        source_id = (dataset_ids[position].strip() if dataset_ids and position < len(dataset_ids)
                     and dataset_ids[position].strip() else manifest_dataset_id(path, payload))
        source_name = (dataset_names[position].strip() if dataset_names and position < len(dataset_names)
                       and dataset_names[position].strip() else manifest_dataset_name(path, payload))
        loaded.append((path, payload, source_id, source_name))

    # If two selected manifests use the same explicit id, retain a stable
    # namespace for both.  The suffix depends on the canonical manifest path,
    # never on selection order.
    id_counts: dict[str, int] = {}
    for _, _, source_id, _ in loaded:
        id_counts[source_id] = id_counts.get(source_id, 0) + 1
    normalized_loaded = []
    for path, payload, source_id, source_name in loaded:
        if id_counts[source_id] > 1:
            source_id = f"{source_id}-{hashlib.sha1(str(path).encode('utf-8')).hexdigest()[:8]}"
        normalized_loaded.append((path, payload, source_id, source_name))
    loaded = sorted(
        normalized_loaded,
        key=lambda item: (
            bool(item[1].get("candidate_pool") or item[1].get("reviewed_training")),
            bool(item[1].get("reviewed_training")),
            str(item[0]).casefold(),
        ),
    )

    contracts = [_contract(payload, manifest_path) for manifest_path, payload, _, _ in loaded]
    reference = contracts[0]
    for path, contract in zip((item[0] for item in loaded), contracts):
        differences = [
            f"{key}: {reference[key]!r} != {contract[key]!r}"
            for key in reference
            if reference[key] != contract[key]
            and not (model_family in {"song_form_v2", "chart_composer_v2"}
                     and key in SONG_FORM_V2_IGNORED_AUDIO_CONTRACT_KEYS)
            and (key in {"target_schema", "feature_profile"}
                 or reference[key] is not None and contract[key] is not None)
        ]
        if differences:
            raise ValueError(
                f"Cannot merge {path}: preprocessing contracts differ ({'; '.join(differences)}). "
                "请使用相同模型族、采样率和音频特征重新预处理这些数据集。"
            )

    if not merge_annotations and any(
        payload.get("candidate_pool")
        or payload.get("reviewed_training")
        or any(isinstance(s, dict) and (s.get("reviewed_only") or s.get("candidate_pool")) for s in payload.get("songs", ()))
        for _, payload, _, _ in loaded
    ):
        raise ValueError("候选库或人工确认片段必须保留并合并质量标注；未确认部分仅作上下文")

    from maichart.training.quality_annotations import (
        _auto_annotation_owner, annotations_fingerprint, audio_sha256, auto_annotation_path, chart_sha256,
        remove_composed_auto_annotations, resolve_auto_annotations, save_auto_annotations,
        zero_unmarked_region_weights,
    )
    chart_hash_cache: dict[tuple[str, int], str | None] = {}
    audio_hash_cache: dict[str, str | None] = {}

    def _disk_chart_sha(path: Path, slot: int) -> str | None:
        key = (str(path), int(slot))
        if key not in chart_hash_cache:
            try:
                chart_hash_cache[key] = chart_sha256(path, slot)
            except (OSError, ValueError, UnicodeError):
                chart_hash_cache[key] = None
        return chart_hash_cache[key]

    def _disk_audio_sha(path_str: Any) -> str | None:
        if not path_str:
            return None
        resolved = _resolve_asset(path_str, output, {})
        key = str(resolved)
        if key not in audio_hash_cache:
            try:
                digest = _preview_audio_digest or audio_sha256
                audio_hash_cache[key] = digest(resolved) if resolved.is_file() else None
            except OSError:
                audio_hash_cache[key] = None
        return audio_hash_cache[key]

    output = Path(output_path).resolve()
    songs: list[dict[str, Any]] = []
    seen_charts: set[tuple[str, int]] = set()
    zero_owner_keys: set[tuple[str, int]] = set()
    zero_content_owners: dict[tuple[str, str, int], tuple[str, int]] = {}
    zero_owner_records: dict[tuple[str, int], dict[str, Any]] = {}
    zero_owner_hashes: dict[tuple[str, int], str] = {}
    zero_contributions: dict[tuple[str, int], list[dict[str, Any]]] = {}
    zero_contributor_counts: dict[tuple[str, int], int] = {}
    preview_contributors: dict[tuple[str, int], list[tuple[str, Path, list[dict[str, Any]]]]] = {}
    preview_annotation_owners: dict[tuple[str, int], dict[str, str]] = {}
    annotation_records: list[dict[str, Any]] = []
    auto_sources: dict[str, dict[str, Any]] = {}
    auto_candidates: dict[tuple[str, int], list[tuple[str, dict[str, Any]]]] = {}
    source_entries: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    owners: dict[tuple[str, int], dict[str, Any]] = {}
    duplicate_count = 0
    multi_source = len(loaded) > 1
    for manifest_path, payload, source_id, source_name in loaded:
        is_candidate = bool(payload.get("candidate_pool"))
        is_reviewed_manifest = bool(payload.get("reviewed_training"))
        candidate_records = {}
        if is_candidate:
            from maichart.training.candidate_pool import _valid_regions
        manifest_records_by_key: dict[tuple[str, int], dict[str, Any]] = {}
        auto_records_by_key: dict[tuple[str, int], list[dict[str, Any]]] = {}
        source_root_path = _source_root(manifest_path, payload)
        source_entries.append({
            "dataset_id": source_id, "name": source_name,
            "manifest": str(manifest_path), "source_root": str(source_root_path),
            "song_count": 0, "difficulty_count": 0,
        })
        if merge_annotations:
            annotation_file = _quality_annotation_path(manifest_path)
            if annotation_file.is_file():
                try:
                    annotation_payload = json.loads(annotation_file.read_text(encoding="utf-8-sig"))
                    if annotation_payload.get("schema") != "maichart-quality-annotations-v1":
                        raise ValueError(f"质量标注 schema 无效：{annotation_file}")
                    for record in annotation_payload.get("records", []):
                        if isinstance(record, dict):
                            copied_record = copy.deepcopy(record)
                            # Older hand-created annotation files sometimes
                            # stored source_path relative to their manifest.
                            copied_record["_annotation_manifest"] = str(manifest_path)
                            for key in ("source_path", "audio_path", "corrected_path"):
                                if copied_record.get(key):
                                    copied_record[key] = str(_resolve_asset(copied_record[key], manifest_path, payload))
                            annotation_records.append(copied_record)
                            try:
                                source_path = copied_record.get("source_path")
                                rec_key = (str(_resolve_asset(source_path, manifest_path, payload)) if source_path else str(Path.cwd()),
                                           int(copied_record.get("difficulty_index")))
                                manifest_records_by_key[rec_key] = copied_record
                                if is_candidate and _valid_regions(copied_record):
                                    candidate_records.setdefault((rec_key[0].casefold(), rec_key[1]), []).append(copied_record)
                            except (TypeError, ValueError, OSError):
                                pass
                except (OSError, json.JSONDecodeError) as exc:
                    raise ValueError(f"无法读取质量标注 {annotation_file}：{exc}") from exc
            if not (is_candidate or is_reviewed_manifest):
                auto_file, auto_payload = resolve_auto_annotations(manifest_path, payload)
                if auto_payload is not None:
                    annotation_owner = _auto_annotation_owner(auto_file, auto_payload)
                    owner_payload = payload
                    if annotation_owner != manifest_path and annotation_owner.is_file():
                        owner_payload = json.loads(annotation_owner.read_text(encoding="utf-8-sig"))
                    auto_sources[str(manifest_path)] = {
                        "source_manifest": str(manifest_path),
                        "audit_manifest": str(annotation_owner),
                        "audit_metadata": auto_payload.get("audit_metadata", {}),
                    }
                    for record in auto_payload["records"]:
                        if not record.get("source_path") or "difficulty_index" not in record:
                            continue
                        for key in ("source_path", "audio_path"):
                            if record.get(key):
                                record[key] = str(_resolve_asset(record[key], annotation_owner, owner_payload))
                        key = (record["source_path"], record["difficulty_index"])
                        auto_records_by_key.setdefault(key, []).append(record)
        for song in payload.get("songs", []):
            copied = copy.deepcopy(song)
            is_reviewed = is_reviewed_manifest or bool(copied.get("reviewed_only"))
            original_song_id = str(copied.get("song_id") or "unknown")
            copied["original_song_id"] = original_song_id
            copied["source_dataset"] = {"id": source_id, "name": source_name}
            copied["source_manifest"] = str(manifest_path)
            copied["song_id"] = f"{source_id}::{original_song_id}" if multi_source else original_song_id
            _absolutize_song_paths(copied, manifest_path, payload, resolve_asset=_resolve_asset)
            maidata_path = Path(str(copied.get("maidata_path") or "")).resolve()
            try:
                source_key = maidata_path.relative_to(source_root_path).as_posix()
            except ValueError:
                source_key = maidata_path.as_posix()
            copied["source_path_key"] = source_key
            kept_difficulties = []
            for difficulty in copied.get("difficulties", []):
                if not isinstance(difficulty, dict):
                    continue
                raw_index = difficulty.get("difficulty_index", difficulty.get("index"))
                try:
                    index = int(raw_index)
                except (TypeError, ValueError):
                    index = -1
                chart_key = (str(maidata_path), index)
                if not (is_reviewed or is_candidate or song.get("candidate_pool")):
                    for record in auto_records_by_key.get(chart_key, ()):
                        if record.get("audio_path") == copied.get("audio_path") or not record.get("audio_path"):
                            auto_candidates.setdefault(chart_key, []).append((str(manifest_path), record))
                # An ordinary source already owning the exact physical slot always wins over later duplicates.
                if chart_key in seen_charts and chart_key not in zero_owner_keys:
                    duplicate_count += 1
                    if owners[chart_key].get("audio_path") != copied.get("audio_path"):
                        raise ValueError(f"同一原谱槽位指向不同音频：{maidata_path}:{index}，请统一数据源")
                    warnings.append({"reason": "duplicate_source_chart", "source_path": str(maidata_path),
                                     "difficulty_index": index, "source_manifest": str(manifest_path)})
                    continue
                rec_for_chart = manifest_records_by_key.get(chart_key) or {}
                is_strict_zero = is_candidate or is_reviewed
                is_zero_source = is_strict_zero or (bool(rec_for_chart) and rec_for_chart.get("unmarked_weight") == 0)
                if is_zero_source:
                    if is_candidate:
                        from maichart.training.candidate_pool import candidate_record
                        records = candidate_records.get((chart_key[0].casefold(), index))
                        cand_rec = candidate_record(manifest_path, copied, difficulty, records) if records else None
                        if cand_rec is None:
                            warnings.append({"reason": "candidate_not_approved_or_stale", "source_path": str(maidata_path),
                                             "difficulty_index": index, "source_manifest": str(manifest_path)})
                            continue
                        rec_for_chart = cand_rec
                    if not rec_for_chart:
                        if is_strict_zero:
                            warnings.append({"reason": "reviewed_not_approved_or_stale", "source_path": str(maidata_path),
                                             "difficulty_index": index, "source_manifest": str(manifest_path)})
                            continue
                    else:
                        valid_c_sha = _disk_chart_sha(maidata_path, index)
                        if not valid_c_sha or rec_for_chart.get("chart_sha256") != valid_c_sha:
                            if is_strict_zero:
                                warnings.append({"reason": "stale_quality_annotation", "source_path": str(maidata_path),
                                                 "difficulty_index": index, "source_manifest": str(manifest_path)})
                                continue
                            is_zero_source = False
                        elif rec_for_chart.get("audio_path") and rec_for_chart["audio_path"] != copied.get("audio_path"):
                            if is_strict_zero:
                                warnings.append({"reason": "quality_audio_changed", "source_path": str(maidata_path),
                                                 "difficulty_index": index, "source_manifest": str(manifest_path)})
                                continue
                            is_zero_source = False
                        else:
                            valid_a_sha = _disk_audio_sha(copied.get("audio_path"))
                            cand_ident = difficulty.get("candidate_identity") if isinstance(difficulty.get("candidate_identity"), dict) else {}
                            expected_a_sha = rec_for_chart.get("audio_sha256") or cand_ident.get("audio_sha256")
                            if expected_a_sha and (not valid_a_sha or expected_a_sha != valid_a_sha):
                                if is_strict_zero:
                                    warnings.append({"reason": "quality_audio_changed", "source_path": str(maidata_path),
                                                     "difficulty_index": index, "source_manifest": str(manifest_path)})
                                    continue
                                is_zero_source = False
                            if is_zero_source:
                                extracted = _extract_zero_unmarked_contributions(rec_for_chart)
                                if extracted is None:
                                    warnings.append({"reason": "reviewed_excluded_or_empty", "source_path": str(maidata_path),
                                                     "difficulty_index": index, "source_manifest": str(manifest_path)})
                                    if is_strict_zero:
                                        continue
                                    is_zero_source = False
                                else:
                                    _, contribs = extracted
                                    content_key = (valid_c_sha, valid_a_sha, index) if valid_c_sha and valid_a_sha else None
                                    if _published_root is not None:
                                        preview_owner = chart_key
                                        if chart_key not in zero_owner_keys and content_key is not None:
                                            preview_owner = zero_content_owners.get(content_key, chart_key)
                                        preview_contributors.setdefault(preview_owner, []).append(
                                            (str(manifest_path), maidata_path, contribs))
                                    if chart_key in zero_owner_keys:
                                        duplicate_count += 1
                                        if owners[chart_key].get("audio_path") != copied.get("audio_path"):
                                            raise ValueError(f"同一原谱槽位指向不同音频：{maidata_path}:{index}，请统一数据源")
                                        zero_contributions[chart_key].extend(contribs)
                                        zero_contributor_counts[chart_key] += 1
                                        warnings.append({"reason": "duplicate_source_chart", "source_path": str(maidata_path),
                                                         "difficulty_index": index, "source_manifest": str(manifest_path)})
                                        continue
                                    if content_key is not None and content_key in zero_content_owners:
                                        duplicate_count += 1
                                        owner_key = zero_content_owners[content_key]
                                        zero_contributions[owner_key].extend(contribs)
                                        zero_contributor_counts[owner_key] += 1
                                        warnings.append({"reason": "duplicate_reviewed_content_chart", "source_path": str(maidata_path),
                                                         "difficulty_index": index, "source_manifest": str(manifest_path)})
                                        continue
                                    copied["reviewed_only"] = True
                                    seen_charts.add(chart_key)
                                    owners[chart_key] = copied
                                    zero_owner_keys.add(chart_key)
                                    zero_owner_records[chart_key] = copy.deepcopy(rec_for_chart)
                                    zero_owner_hashes[chart_key] = valid_c_sha
                                    zero_contributions[chart_key] = list(contribs)
                                    zero_contributor_counts[chart_key] = 1
                                    if content_key is not None:
                                        zero_content_owners[content_key] = chart_key
                                    kept_difficulties.append(difficulty)
                                    source_entries[-1]["difficulty_count"] += 1
                                    continue
                if chart_key in seen_charts:
                    duplicate_count += 1
                    if owners[chart_key].get("audio_path") != copied.get("audio_path"):
                        raise ValueError(f"同一原谱槽位指向不同音频：{maidata_path}:{index}，请统一数据源")
                    warnings.append({"reason": "duplicate_source_chart", "source_path": str(maidata_path),
                                     "difficulty_index": index, "source_manifest": str(manifest_path)})
                    continue
                seen_charts.add(chart_key)
                owners[chart_key] = copied
                kept_difficulties.append(difficulty)
                source_entries[-1]["difficulty_count"] += 1
            if kept_difficulties:
                copied["difficulties"] = kept_difficulties
                songs.append(copied)
                source_entries[-1]["song_count"] += 1

    # Annotation records refer to source song ids.  Resolve their source by
    # absolute path, namespace ids exactly as quality catalog does, and reject
    # contradictory labels for a duplicate source chart instead of silently
    # choosing whichever manifest happened to be listed first.
    seen_annotation_keys: dict[tuple[str, int], dict[str, Any]] = {}
    accepted_records = []
    training_fields = ("label", "weight", "regions", "chart_sha256", "audio_sha256",
                       "corrected_path", "corrected_difficulty", "corrected_sha256")
    for record in annotation_records:
        annotation_manifest = record.pop("_annotation_manifest", None)
        try:
            index = int(record.get("difficulty_index"))
        except (TypeError, ValueError):
            warnings.append({"reason": "annotation_missing_difficulty", "chart_id": record.get("chart_id")})
            continue
        record_path = _resolve_asset(record["source_path"], output, {}) if record.get("source_path") else Path.cwd()
        key = (str(record_path), index)
        if key in zero_owner_keys:
            # Verified zero-unmarked owners are emitted below from their validated contribution table.
            continue
        owner = owners.get(key)
        # A sidecar may also annotate charts outside this selected split.
        if owner is None:
            continue
        # Candidate / reviewed sidecars deliberately contain zero-weight unknown records.
        # They must never override a duplicate owned by an ordinary source.
        if annotation_manifest != owner.get("source_manifest") and any(
            str(p) == annotation_manifest and (data.get("candidate_pool") or data.get("reviewed_training"))
            for p, data, _, _ in loaded
        ):
            continue
        valid_hash = _disk_chart_sha(record_path, index)
        if not valid_hash or record.get("chart_sha256") != valid_hash:
            warnings.append({"reason": "stale_quality_annotation", "source_path": str(record_path), "difficulty_index": index})
            continue
        if record.get("audio_path") and record["audio_path"] != owner.get("audio_path"):
            warnings.append({"reason": "quality_audio_changed", "source_path": str(record_path), "difficulty_index": index})
            continue
        if record.get("audio_sha256"):
            valid_audio = _disk_audio_sha(owner.get("audio_path"))
            if not valid_audio or record["audio_sha256"] != valid_audio:
                warnings.append({"reason": "quality_audio_changed", "source_path": str(record_path), "difficulty_index": index})
                continue
        prior = seen_annotation_keys.get(key)
        if prior is not None:
            if any(prior.get(field) != record.get(field) for field in training_fields):
                raise ValueError(f"同一谱面存在冲突质量标注（标签、区域或修正版不同）：{record_path}:{index}")
            continue
        seen_annotation_keys[key] = dict(record)
        if _published_root is not None and annotation_manifest is not None:
            preview_annotation_owners.setdefault(key, {})["manual"] = annotation_manifest
        record.update({"original_song_id": owner["original_song_id"], "song_id": owner["song_id"],
                       "source_dataset": owner["source_dataset"], "source_path": str(record_path),
                       "source_manifest": owner["source_manifest"]})
        namespace = f"{owner['source_dataset']['id']}\0{owner['source_path_key']}"
        record["chart_id"] = _namespaced_chart_id(owner["song_id"], index, valid_hash, None, namespace)
        accepted_records.append(record)

    for key, owner in owners.items():
        if key not in zero_owner_keys:
            continue
        record_path_str, index = key
        record = copy.deepcopy(zero_owner_records[key])
        record.pop("_annotation_manifest", None)
        record["unmarked_weight"] = 0.0
        if zero_contributor_counts[key] > 1:
            merged_weight, merged_regions = _merge_zero_unmarked_contributions(zero_contributions[key])
            record["label"] = "normal"
            record["weight"] = merged_weight
            record["regions"] = merged_regions
        else:
            cleaned_regions: list[dict[str, Any]] = []
            for raw_region in record.get("regions", []):
                if not isinstance(raw_region, dict):
                    continue
                t_w, c_w = zero_unmarked_region_weights(raw_region)
                if not (t_w > 0 or c_w > 0):
                    continue
                kept_region = dict(raw_region)
                kept_region.setdefault("label", "good")
                cleaned_regions.append(kept_region)
            record["regions"] = cleaned_regions
        valid_hash = zero_owner_hashes[key]
        record.update({"original_song_id": owner["original_song_id"], "song_id": owner["song_id"],
                       "source_dataset": owner["source_dataset"], "source_path": record_path_str,
                       "source_manifest": owner["source_manifest"]})
        namespace = f"{owner['source_dataset']['id']}\0{owner['source_path_key']}"
        record["chart_id"] = _namespaced_chart_id(owner["song_id"], index, valid_hash, None, namespace)
        accepted_records.append(record)
    annotation_records = accepted_records

    # Audits omit clean charts from records. A formal audit on a retained
    # ordinary source still binds an empty composed sidecar in that case.
    used_auto_sources = {
        owner["source_manifest"] for key, owner in owners.items()
        if key not in zero_owner_keys and not owner.get("candidate_pool")
    } & auto_sources.keys()
    auto_records = []
    for key, candidates in sorted(auto_candidates.items()):
        owner = owners.get(key)
        if owner is None or key in zero_owner_keys:
            continue
        record_path, index = key
        chart_hash = _disk_chart_sha(Path(record_path), index)
        audio_path = owner.get("audio_path")
        if not chart_hash or not audio_path or not Path(audio_path).is_file():
            continue
        # An audit of the retained entry takes precedence over a duplicate
        # ordinary source's audit of the same physical chart and audio.
        candidates.sort(key=lambda item: item[0] != owner["source_manifest"])
        for source_manifest, record in candidates:
            if record.get("chart_sha256") != chart_hash:
                continue
            if record.get("audio_path") and record["audio_path"] != audio_path:
                continue
            if record.get("audio_sha256") and record["audio_sha256"] != _disk_audio_sha(audio_path):
                continue
            if not (record.get("audio_path") or record.get("audio_sha256")):
                continue
            record = {
                **record,
                "original_song_id": owner["original_song_id"],
                "song_id": owner["song_id"],
                "source_dataset": owner["source_dataset"],
                "source_manifest": owner["source_manifest"],
                "audio_path": audio_path,
            }
            namespace = f"{owner['source_dataset']['id']}\0{owner['source_path_key']}"
            record["chart_id"] = _namespaced_chart_id(owner["song_id"], index, chart_hash, None, namespace)
            auto_records.append(record)
            used_auto_sources.add(source_manifest)
            if _published_root is not None:
                preview_annotation_owners.setdefault(key, {})["auto"] = source_manifest
            break
    auto_metadata = {"composed_sources": [
        metadata for source, metadata in auto_sources.items() if source in used_auto_sources
    ]}
    if _composition_identity is not None:
        if used_auto_sources:
            _composition_identity = hashlib.sha256(
                (_composition_identity + annotations_fingerprint({
                    "records": auto_records, "audit_metadata": auto_metadata,
                })).encode("ascii")
            ).hexdigest()
        output = output / f"composed_{_composition_identity[:16]}" / "training_manifest.json"

    published = (_published_root / output.parent.name / output.name
                 if _published_root is not None and _composition_identity is not None else None)

    merged_preprocessing = {**(loaded[0][1].get("preprocessing") or {}), **{
        key: reference[key]
        for key in reference
        if reference[key] is not None
    }}
    if model_family == "song_form_v2":
        for key in SONG_FORM_V2_LEGACY_AUDIO_METADATA_KEYS:
            merged_preprocessing.pop(key, None)
        merged_preprocessing["feature_profile"] = SONG_FORM_V2_AUDIO_PROFILE

    merged = {
        "schema": SCHEMA,
        # All source paths are absolute after composition.  Keeping the output
        # directory as source_root makes older path resolvers harmless instead
        # of pointing them at a literal ``<merged>`` folder.
        "source_root": str(output.parent),
        "cache_dir": str(output.parent),
        "song_count": len(songs),
        "difficulty_count": sum(len(song.get("difficulties", [])) for song in songs),
        "usable_difficulty_count": sum(
            1
            for song in songs
            for difficulty in song.get("difficulties", [])
            if difficulty.get("event_usable_for_training", difficulty.get("usable_for_training"))
        ),
        "preprocessing": merged_preprocessing,
        "merged_from": [str(path) for path, _, _, _ in loaded],
        "datasets": source_entries,
        "songs": songs,
        "summary": {
            "merged_manifest_count": len(loaded),
            "song_count": len(songs),
            "difficulty_count": sum(len(song.get("difficulties", [])) for song in songs),
            "usable_difficulty_count": sum(
                1
                for song in songs
                for difficulty in song.get("difficulties", [])
                if difficulty.get("event_usable_for_training", difficulty.get("usable_for_training"))
            ),
            "duplicate_source_charts": duplicate_count,
            "quality_records": len(annotation_records),
            "candidate_pool_charts": sum(len(song.get("difficulties", [])) for song in songs if song.get("candidate_pool")),
            "stale_quality_annotations": sum(item["reason"] in {"stale_quality_annotation", "quality_audio_changed"} for item in warnings),
        },
        "warnings": warnings,
        "errors": [],
    }
    merged["manifest"] = str(output)
    merged["raw_root"] = str(output.parent)
    merged["raw_roots"] = [entry["source_root"] for entry in source_entries]
    # Reuse the largest source's content-addressed cache when available.
    primary = max(loaded, key=lambda item: len(item[1].get("songs", [])))
    primary_cache = _source_asset(primary[1].get("cache_dir") or "cache", primary[0], primary[1])
    merged["cache_dir"] = str(primary_cache if primary_cache.is_dir() else output.parent / "cache")
    if merge_annotations:
        annotation_output = _quality_annotation_path(output)
        _write_json(annotation_output, {
            "schema": "maichart-quality-annotations-v1",
            "manifest": str(output), "records": annotation_records,
            "merged_from": [str(path) for path, _, _, _ in loaded],
        })
        merged["quality_annotations"] = str(annotation_output)
    if published is None and any(payload.get("candidate_pool") for _, payload, _, _ in loaded):
        from maichart.training.candidate_pool import supervision_budget
        merged["summary"]["supervision"] = supervision_budget(
            [path for path, _, _, _ in loaded],
            _accepted={(owner["source_manifest"], (key[0].casefold(), key[1])) for key, owner in owners.items()},
        )
    if used_auto_sources:
        inherited_path = save_auto_annotations(output, auto_records, audit_metadata=auto_metadata, composed=True)
        legacy_path = auto_annotation_path(output)
        if _composition_identity is not None and output.is_file() and legacy_path.is_file():
            previous = json.loads(output.read_text(encoding="utf-8-sig"))
            if previous.get("auto_quality_annotations") == str(legacy_path):
                # Preserve the manifest hash of an existing immutable snapshot.
                inherited_path = legacy_path
        merged["auto_quality_annotations"] = str(inherited_path)
    elif _composition_identity is None and all(output != item[0] for item in loaded):
        remove_composed_auto_annotations(output)
    _write_json(output, merged)
    if _source_versions is not None:
        _write_json(output.with_name("source_versions.json"), _source_versions)
    if published is not None:
        return {**merged, "published_manifest": str(published), "preview_contributors": preview_contributors,
                "preview_annotation_owners": preview_annotation_owners}
    return merged


def _snapshot_stamp(path):
    try:
        stat = Path(path).stat()
        return [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino]
    except FileNotFoundError:
        return None


def _composition_source_versions(manifests, model_family):
    from maichart.training.quality_annotations import (
        annotation_path, auto_annotation_path, split_annotation_source,
    )

    files, assets = {}, {}
    for value in manifests:
        manifest = Path(value)
        payload = json.loads(manifest.read_text(encoding="utf-8-sig"))
        paths = [manifest, annotation_path(manifest)]
        owners = [manifest]
        origin = split_annotation_source(manifest, payload)
        if origin is not None:
            owners.append(origin)
        for owner in owners:
            paths.extend((owner, auto_annotation_path(owner), auto_annotation_path(owner, composed=True)))
        for path in paths:
            files[str(path)] = _snapshot_stamp(path)
        for song in payload.get("songs", []):
            # File stats are also needed for ordinary sources: the composer
            # validates their audio and chart hashes before retaining weights.
            for field in ("maidata_path", "audio_path"):
                if song.get(field):
                    raw = material_path(song[field])
                    path = raw if raw.is_absolute() else _source_asset(raw, manifest, payload)
                    if str(path) not in assets:
                        assets[str(path)] = _snapshot_stamp(path)
    return dict(model_family=model_family, files=files, assets=assets)


def compose_datasets(
    manifests: list[str | Path],
    output_root: str | Path,
    model_family: str | None = None,
    *,
    dataset_ids: list[str] | None = None,
    dataset_names: list[str] | None = None,
    published_root: Path | None = None,
    preview_audio_digest=None,
) -> dict[str, Any]:
    """Prepare a stable source selection without changing a single source."""
    if not manifests:
        raise ValueError("至少需要一份训练 manifest")
    manifests = list(dict.fromkeys(str(Path(value).resolve()) for value in manifests))
    for value in manifests:
        path = Path(value)
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        target = _contract(payload, path).get("target_schema")
        if model_family in {"event_v1", "event_ctx_v1"} and target != "event_v1":
            raise ValueError(f"{path.parent.name} 不是事件模型预处理清单，请选择当前模型族重新预处理")
        if model_family == "legacy_frame_v2" and target == "event_v1":
            raise ValueError("事件模型清单不能用于旧固定帧模型，请重新预处理")
    if len(manifests) == 1 and not json.loads(Path(manifests[0]).read_text(encoding="utf-8-sig")).get("candidate_pool"):
        # Selecting one prepared dataset must not rewrite song ids, paths, or
        # checkpoint contracts.  The trainer can consume its original file
        # directly; composition is only needed when sources are combined.
        manifest = Path(manifests[0]).resolve()
        payload = json.loads(manifest.read_text(encoding="utf-8-sig"))
        if payload.get("schema") != SCHEMA:
            raise ValueError(f"{manifest} is not a {SCHEMA}.")
        source_root = payload.get("source_root") or manifest.parent
        source_root_path = Path(str(source_root))
        if not source_root_path.is_absolute():
            source_root_path = (manifest.parent / source_root_path).resolve()
        cache_value = payload.get("cache_dir") or str(manifest.parent / "cache")
        cache_path = Path(str(cache_value))
        if not cache_path.is_absolute():
            cache_path = (manifest.parent / cache_path).resolve()
        return {
            "manifest": str(manifest), "raw_root": str(source_root_path), "raw_roots": [str(source_root_path)],
            "source_root": str(source_root_path), "cache_dir": str(cache_path),
            "quality_annotations": str(_quality_annotation_path(manifest)),
            "datasets": catalog_datasets([manifest]),
            "summary": payload.get("summary") or {
                "merged_manifest_count": 1, "song_count": payload.get("song_count", len(payload.get("songs", []))),
                "difficulty_count": payload.get("difficulty_count", 0),
                "usable_difficulty_count": payload.get("usable_difficulty_count", 0),
            },
            "warnings": [],
        }
    target = Path(output_root).resolve()
    resolve_asset = _source_asset_cache() if published_root is not None else _source_asset
    composition_identity = None
    source_versions = None
    if target.suffix.lower() != ".json":
        if model_family == "chart_composer_v2" and published_root is None:
            source_versions = _composition_source_versions(manifests, model_family)
        identity = hashlib.sha256()
        for value in sorted(manifests, key=str.casefold):
            identity.update(value.casefold().encode("utf-8"))
            identity.update(Path(value).read_bytes())
            annotation = _quality_annotation_path(Path(value))
            if annotation.is_file():
                identity.update(annotation.read_bytes())
            candidate_payload = json.loads(Path(value).read_text(encoding="utf-8-sig"))
            if candidate_payload.get("candidate_pool") or candidate_payload.get("reviewed_training"):
                # A changed source must never reuse and overwrite the output
                # identity of an earlier frozen selection.
                from maichart.training.candidate_pool import _charts, _digest, _stat
                digest = preview_audio_digest or _digest
                for song, difficulty, index in _charts(value, resolve_asset=resolve_asset):
                    for asset in (song.get("maidata_path"), song.get("audio_path")):
                        identity.update(str(_stat(asset)).encode())
                        if asset and Path(asset).is_file():
                            identity.update(digest(asset).encode())
        # A saved label change creates a new snapshot, so checking or preparing
        # another run cannot overwrite a running trainer's annotation file.
        composition_identity = identity.hexdigest()
    with _COMPOSE_LOCK:
        return merge_training_manifests(
            manifests, target, dataset_ids=dataset_ids, dataset_names=dataset_names,
            merge_annotations=True, model_family=model_family,
            _composition_identity=composition_identity, _source_versions=source_versions,
            _published_root=published_root,
            _preview_audio_digest=preview_audio_digest,
            _resolve_asset=resolve_asset,
        )


def catalog_datasets(manifests: list[str | Path]) -> list[dict[str, Any]]:
    """Return compact dataset cards for the WebUI selection page."""
    cards: list[dict[str, Any]] = []
    for raw_path in manifests:
        path = Path(raw_path).resolve()
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if payload.get("schema") != SCHEMA:
            raise ValueError(f"{path} is not a {SCHEMA}.")
        source_root = payload.get("source_root")
        root = Path(str(source_root)) if source_root else path.parent
        if not root.is_absolute():
            root = (path.parent / root).resolve()
        songs = [item for item in payload.get("songs", []) if isinstance(item, dict)]
        difficulties = [difficulty for song in songs for difficulty in song.get("difficulties", [])
                        if isinstance(difficulty, dict)]
        usable = [item for item in difficulties if item.get("usable_for_training") or item.get("event_usable_for_training")]
        dataset_id = manifest_dataset_id(path, payload)
        annotation = _quality_annotation_path(path)
        cards.append({
            "dataset_id": dataset_id,
            "name": manifest_dataset_name(path, payload),
            "manifest": str(path),
            "source_root": str(root),
            "song_count": len(songs),
            "difficulty_count": len(difficulties),
            "usable_difficulty_count": len(usable),
            "annotation_path": str(annotation),
            "annotation_exists": annotation.is_file(),
            "prepared": path.is_file(),
            "preprocessing": payload.get("preprocessing") or {},
            "merged": bool(payload.get("merged_from")),
        })
    return cards


def _manifest_chart_sources(manifest_path: str | Path) -> dict[tuple[str, int], dict[str, Any]]:
    """Collect resolved maidata/audio identities for overlap checks."""
    manifest_file = Path(manifest_path).resolve()
    payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    def resolve(value: Any) -> str | None:
        if not value:
            return None
        return str(_source_asset(value, manifest_file, payload))

    result: dict[tuple[str, int], dict[str, Any]] = {}
    for song in payload.get("songs", []):
        maidata = resolve(song.get("maidata_path"))
        audio = resolve(song.get("audio_path"))
        for difficulty in song.get("difficulties", []):
            try:
                index = int(difficulty.get("difficulty_index", difficulty.get("index")))
            except (TypeError, ValueError):
                continue
            if maidata:
                result[(maidata, index)] = {"song_id": song.get("song_id"), "difficulty_index": index,
                                            "maidata_path": maidata, "audio_path": audio}
    return result


def validation_overlap(train_manifest: str | Path, validation_manifest: str | Path) -> list[dict[str, Any]]:
    """Return chart/audio identities present in both manifests."""
    train = _manifest_chart_sources(train_manifest)
    validation = _manifest_chart_sources(validation_manifest)
    overlaps: list[dict[str, Any]] = []
    for key in sorted(set(train) & set(validation)):
        overlaps.append({"train": train[key], "validation": validation[key], "reason": "same_maidata_difficulty"})
    train_audio = {value["audio_path"] for value in train.values() if value.get("audio_path")}
    for value in validation.values():
        if (value.get("audio_path") in train_audio and value.get("audio_path")) or any(
            key[0] == value.get("maidata_path") for key in train
        ):
            if not any(item["validation"] == value for item in overlaps):
                overlaps.append({"train": {"audio_path": value["audio_path"]}, "validation": value,
                                 "reason": "same_audio"})
    return overlaps


def assert_no_validation_overlap(train_manifest: str | Path, validation_manifest: str | Path) -> None:
    """Raise a clear error when training and validation share source content."""
    overlaps = validation_overlap(train_manifest, validation_manifest)
    if overlaps:
        first = overlaps[0]
        raise ValueError(
            "训练和验证集存在歌曲或谱面重叠："
            f"{first.get('reason')} {first.get('validation', {}).get('song_id', '')}。"
            "请按歌曲划分验证集，避免验证指标失真。"
        )


def _contract(payload: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    metadata = payload.get("preprocessing") if isinstance(payload.get("preprocessing"), dict) else {}
    first_audio = _first_cache_payload(payload, manifest_path, "audio_features_path")
    first_labels = _first_cache_payload(payload, manifest_path, "frame_labels_path")
    target = metadata.get("target_schema")
    if target == "event_ctx_v1":
        target = "event_v1"
    if target is None and (metadata.get("feature_profile") == "basic_v1" or first_labels):
        target = "frame_v2"
    return {
        "target_schema": target,
        "feature_profile": metadata.get("feature_profile"),
        "hop_length": metadata.get("hop_length"),
        "feature_rate_hz": metadata.get("feature_rate_hz"),
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
            path = _source_asset(raw_path, manifest_path, payload)
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


def _absolutize_song_paths(song: dict[str, Any], manifest: Path, payload: dict[str, Any], *, resolve_asset=_source_asset) -> None:
    for key in (*PATH_KEYS, "audio_features_path", "chart_ir_path", "frame_labels_path", "alignment_report_path"):
        if song.get(key):
            song[key] = str(resolve_asset(song[key], manifest, payload))
    audio = song.get("audio")
    if isinstance(audio, dict):
        for key in ("path", "audio_features_path"):
            if audio.get(key):
                audio[key] = str(resolve_asset(audio[key], manifest, payload))
    for difficulty in song.get("difficulties", []):
        if not isinstance(difficulty, dict):
            continue
        for key in ("chart_ir_path", "frame_labels_path", "alignment_report_path", "source_path", "maidata_path", "audio_path"):
            if difficulty.get(key):
                difficulty[key] = str(resolve_asset(difficulty[key], manifest, payload))


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


