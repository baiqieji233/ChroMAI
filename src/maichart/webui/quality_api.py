"""WebUI-facing quality annotation service.

This module contains no HTTP server coupling.  ``QualityAPI`` returns ordinary
status/data pairs so the existing dependency-light handler can wire it in with
two small branches.
"""

from __future__ import annotations

import json
import threading
from collections import Counter
from http import HTTPStatus
from pathlib import Path
from typing import Any
from urllib.parse import quote

from maichart.maidata import parse_maidata_file
from maichart.timing import tokenize_chart_difficulty_timing
from maichart.training.quality_annotations import (
    DEFAULT_WEIGHTS,
    LABELS,
    CatalogChart,
    annotation_path,
    audio_sha256,
    chart_duration,
    chart_sha256,
    find_annotation,
    iter_catalog,
    load_annotations,
    save_annotations,
    validate_record,
)

_CATALOG_LOCK = threading.RLock()
_CATALOGS: dict[str, tuple[int, int, list[CatalogChart], dict[str, tuple[int, int]]]] = {}


class _AnnotationLookup:
    """Prefilter candidates, preserving the existing validator and record order."""
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = list(records)
        self.ids: dict[str, set[int]] = {}
        self.sources: dict[tuple[Path, int], set[int]] = {}
        for index, record in enumerate(self.records):
            self._register(index, record)

    def _register(self, index: int, record: dict[str, Any]) -> None:
        record_id = record.get("chart_id")
        if isinstance(record_id, str):
            self.ids.setdefault(record_id, set()).add(index)
        if record.get("source_path") and record.get("difficulty_index") is not None:
            try:
                key = (Path(record["source_path"]).resolve(), int(record["difficulty_index"]))
                self.sources.setdefault(key, set()).add(index)
            except (OSError, ValueError, TypeError):
                pass

    def update(self, record: dict[str, Any]) -> None:
        indices = self.ids.get(record.get("chart_id"))
        if indices:
            for index in list(indices):
                self.records[index] = record
                self._register(index, record)
        else:
            index = len(self.records)
            self.records.append(record)
            self._register(index, record)

    def find(self, row: CatalogChart, active: dict[str, dict[str, Any]] | None = None) -> dict[str, Any] | None:
        indices = self.ids.get(row.chart_id, set()) | self.ids.get(row.legacy_chart_id, set())
        indices |= self.sources.get((Path(row.source_path).resolve(), int(row.difficulty_index)), set())
        candidates = [self.records[index] for index in sorted(indices)]
        if active is not None:
            candidates = [record for record in candidates if active.get(record.get("chart_id")) is record]
        return find_annotation(row, candidates)


def _catalog(manifest: Path) -> list[CatalogChart]:
    """Reuse parsed catalog rows while cheaply noticing edited chart files."""
    key = str(manifest.resolve())
    stat = manifest.stat()
    cached = _CATALOGS.get(key)
    if cached and cached[0] == stat.st_mtime_ns and cached[1] == stat.st_size:
        old_rows, old_stats = cached[2], cached[3]
        changed = False
        for row in old_rows:
            try:
                current = Path(row.source_path).stat()
                stamp = (current.st_size, current.st_mtime_ns)
            except OSError:
                stamp = (-1, -1)
            if old_stats.get(row.source_path) != stamp:
                changed = True
                break
        if not changed:
            return old_rows
    rows = iter_catalog(manifest)
    stats = {}
    for row in rows:
        try:
            info = Path(row.source_path).stat()
            stats[row.source_path] = (info.st_size, info.st_mtime_ns)
        except OSError:
            stats[row.source_path] = (-1, -1)
    with _CATALOG_LOCK:
        _CATALOGS[key] = (stat.st_mtime_ns, stat.st_size, rows, stats)
    return rows


class QualityAPI:
    """Stateless quality annotation operations for one local WebUI."""

    def post(self, endpoint: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        if endpoint == "/api/quality/catalog":
            return self.catalog(payload)
        if endpoint == "/api/quality/selection":
            return self.selection(payload)
        if endpoint == "/api/quality/save":
            return self.save(payload)
        if endpoint == "/api/quality/detail":
            return self.detail(payload)
        if endpoint == "/api/quality/preview":
            return self.preview(payload)
        return HTTPStatus.NOT_FOUND, {"error": "Not found"}

    def get_audio(self, query: dict[str, Any], *, method: str = "GET", range_header: str | None = None) -> tuple[int, dict[str, str], Path | None, tuple[int, int] | None]:
        manifest = _required_manifest(query)
        chart_id = _first(query.get("chart_id"))
        row = next((item for item in _catalog(manifest) if item.chart_id == chart_id), None)
        if row is None or not row.audio_path:
            return HTTPStatus.NOT_FOUND, {"Content-Type": "application/json"}, None, None
        path = Path(row.audio_path)
        if not path.is_file():
            return HTTPStatus.NOT_FOUND, {"Content-Type": "application/json"}, None, None
        size = path.stat().st_size
        start, end, status = _parse_range(range_header, size)
        headers = {"Content-Type": _content_type(path), "Accept-Ranges": "bytes", "Content-Length": str(end - start + 1)}
        if status == HTTPStatus.PARTIAL_CONTENT:
            headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        return status, headers, path, (start, end)

    def catalog(self, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        manifest = _required_manifest(payload)
        query = str(payload.get("query", "")).strip().casefold()
        label_filter = str(payload.get("label_filter") or "").strip()
        if label_filter and label_filter not in LABELS:
            raise ValueError("未知质量标签")
        try:
            page = max(1, int(payload.get("page", 1)))
            page_size = min(500, max(1, int(payload.get("page_size", 50))))
        except (TypeError, ValueError) as exc:
            raise ValueError("page/page_size 必须是整数") from exc
        sort_by = str(payload.get("sort_by", "title") or "title").strip().casefold()
        sort_aliases = {"level": "level", "难度": "level", "title": "title", "标题": "title",
                        "designer": "designer", "谱师": "designer", "folder": "folder", "文件夹": "folder",
                        "song_id": "song_id", "song": "song_id", "歌曲id": "song_id"}
        sort_by = sort_aliases.get(sort_by, sort_by)
        if sort_by not in {"level", "title", "designer", "folder", "song_id"}:
            raise ValueError("sort_by 必须是 level/title/designer/folder/song_id")
        sort_order = str(payload.get("sort_order", payload.get("order", "asc")) or "asc").strip().casefold()
        if sort_order in {"升序", "ascending"}:
            sort_order = "asc"
        elif sort_order in {"降序", "descending"}:
            sort_order = "desc"
        if sort_order not in {"asc", "desc"}:
            raise ValueError("sort_order 必须是 asc 或 desc")
        records = load_annotations(manifest)["records"]
        all_rows = _catalog(manifest)
        lookup = _AnnotationLookup(records)
        annotation_by_id = {row.chart_id: lookup.find(row) for row in all_rows}
        catalog_values = [row.to_dict(annotation_by_id[row.chart_id]) for row in all_rows]
        rows = []
        for value in catalog_values:
            haystack = " ".join(str(value.get(key, "")) for key in ("song_id", "title", "maidata_title", "artist", "level", "designer", "folder_name", "source_dataset", "source_path")).casefold()
            if query and query not in haystack:
                continue
            if label_filter and value["label"] != label_filter:
                continue
            rows.append(value)
        _sort_catalog_rows(rows, sort_by, sort_order == "desc")
        total = len(rows)
        pages = max(1, (total + page_size - 1) // page_size)
        page = min(page, pages)
        start = (page - 1) * page_size
        summary = self._summary(catalog_values, records)
        return HTTPStatus.OK, {
            "schema": "maichart-quality-annotations-v1", "manifest": str(manifest), "annotation_path": str(annotation_path(manifest)),
            "page": page, "pages": pages, "page_size": page_size, "total": total,
            "folder_total": len({row.chart_id for row in all_rows}),
            "sort_by": sort_by, "sort_order": sort_order,
            "items": rows[start:start + page_size],
            "rows": rows[start:start + page_size], "summary": summary,
        }

    def selection(self, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        """Return every chart identity in this source, without page/filter caps."""
        manifest = _required_manifest(payload)
        chart_ids = list(dict.fromkeys(row.chart_id for row in _catalog(manifest)))
        return HTTPStatus.OK, {"manifest": str(manifest), "chart_ids": chart_ids,
                               "total": len(chart_ids), "scope": "current_dataset"}

    def save(self, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        manifest = _required_manifest(payload)
        chart_ids = payload.get("chart_ids")
        if isinstance(chart_ids, str):
            chart_ids = [chart_ids]
        if not isinstance(chart_ids, list) or not chart_ids:
            raise ValueError("chart_ids 不能为空")
        label = str(payload.get("label", "normal"))
        if label not in LABELS:
            raise ValueError("未知质量标签")
        weight = float(payload.get("weight", DEFAULT_WEIGHTS[label]))
        if not 0 <= weight <= 4 or weight != weight or weight in (float("inf"), float("-inf")):
            raise ValueError("weight 必须是 0–4 的有限数字")
        rows = _catalog(manifest)
        by_id = {row.chart_id: row for row in rows}
        missing = [item for item in chart_ids if item not in by_id]
        if missing:
            raise ValueError("存在不属于当前 manifest 的 chart_id")
        current = load_annotations(manifest)["records"]
        by_record = {item.get("chart_id"): item for item in current}
        lookup = _AnnotationLookup(list(by_record.values()))
        audio_hashes: dict[str, str] = {}
        for chart_id in chart_ids:
            row = by_id[chart_id]
            duration = chart_duration(row.source_path, row.difficulty_index)
            previous = lookup.find(row, by_record) or {}
            # If this is an old or differently namespaced id, replace it with
            # the current identity on save instead of leaving duplicate rows.
            old_id = previous.get("chart_id")
            if previous and old_id != chart_id and by_record.get(old_id) is previous:
                by_record.pop(old_id, None)
            regions = payload.get("regions", previous.get("regions", []))
            # Validate against exact chart duration before writing anything.
            record = validate_record({**previous, "chart_id": chart_id,
                                      "regions": regions, "label": label, "weight": weight})
            if duration is not None:
                from maichart.training.quality_annotations import _validate_regions
                record["regions"] = _validate_regions(regions, duration)
            record.update({"song_id": row.song_id, "difficulty_index": row.difficulty_index, "source_path": row.source_path,
                           "audio_path": row.audio_path, "chart_sha256": row.chart_sha256,
                           "source_dataset": row.source_dataset,
                           "source_dataset_id": row.source_dataset_id, "folder_name": row.folder_name,
                           "maidata_title": row.maidata_title,
                           "reason": str(payload.get("reason", previous.get("reason", "")))})
            if row.audio_path and Path(row.audio_path).is_file():
                # Hash only explicitly selected charts; catalog browsing stays
                # cheap even for a large corpus of long audio files.
                if row.audio_path not in audio_hashes:
                    audio_hashes[row.audio_path] = audio_sha256(row.audio_path)
                record["audio_sha256"] = audio_hashes[row.audio_path]
            corrected = payload.get("corrected_path")
            if label != "negative" and "corrected_path" not in payload:
                record.pop("corrected_path", None)
                record.pop("corrected_difficulty", None)
                record.pop("corrected_sha256", None)
            if corrected:
                corrected_path = Path(str(corrected))
                if not corrected_path.is_absolute():
                    corrected_path = (manifest.parent / corrected_path).resolve()
                else:
                    corrected_path = corrected_path.resolve()
                if not corrected_path.is_file():
                    raise ValueError("corrected_path 不存在")
                corrected_index = int(payload.get("corrected_difficulty", row.difficulty_index))
                corrected_chart = parse_maidata_file(corrected_path)
                target = next((item for item in corrected_chart.difficulties if item.index == corrected_index), None)
                if target is None or not target.inote:
                    raise ValueError("corrected_path 缺少指定难度槽位")
                corrected_duration = chart_duration(corrected_path, corrected_index)
                if duration is not None and corrected_duration is not None and abs(duration - corrected_duration) > 0.02:
                    raise ValueError("修正版与原谱面时长不一致，无法建立偏好对")
                # Saving should catch incompatible timing now, rather than
                # letting the trainer discover it after it has prepared the
                # full audio cache. The preference builder repeats this check
                # during training because files may change after annotation.
                from maichart.event_timing import parse_event_chart
                from maichart.training.event_preferences import _validate_corrected

                original_events = parse_event_chart(
                    row.source_path, row.difficulty_index, strict_notes=True)
                corrected_events = parse_event_chart(
                    corrected_path, corrected_index, strict_notes=True)
                incompatibility = _validate_corrected(original_events, corrected_events)
                if incompatibility:
                    raise ValueError(f"修正版与原谱无法建立偏好对：{incompatibility}")
                record["corrected_path"] = str(corrected_path)
                record["corrected_difficulty"] = corrected_index
                record["corrected_sha256"] = chart_sha256(corrected_path, corrected_index)
            by_record[chart_id] = record
            lookup.update(record)
        output = save_annotations(manifest, by_record.values())
        current_rows = _catalog(manifest)
        values = [row.to_dict(lookup.find(row, by_record)) for row in current_rows]
        summary = self._summary(values, list(by_record.values()))
        return HTTPStatus.OK, {"saved": len(chart_ids), "annotation_path": str(output),
                               "records": list(by_record.values()), "summary": summary}

    def detail(self, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        manifest = _required_manifest(payload)
        chart_id = _first(payload.get("chart_id"))
        records = load_annotations(manifest)["records"]
        row = next((item for item in _catalog(manifest) if item.chart_id == chart_id), None)
        if row is None:
            raise ValueError("chart_id 不属于当前 manifest")
        chart = parse_maidata_file(row.source_path)
        difficulty = next(item for item in chart.difficulties if item.index == row.difficulty_index)
        points = []
        try:
            points = [{"time_sec": point.time_sec, "beat": float(point.beat), "raw": point.raw, "note_text": point.note_text}
                      for point in tokenize_chart_difficulty_timing(
                          chart, difficulty.index, initial_bpm=float(chart.wholebpm or 120))]
        except Exception:
            points = []
        annotation = find_annotation(row, records, verify_audio=True)
        return HTTPStatus.OK, {**row.to_dict(annotation), "chart": row.to_dict(annotation),
            "record": annotation, "maidata": chart.raw or "",
            "preview": {"events": points, "duration_sec": chart_duration(row.source_path, row.difficulty_index)},
            "slot": {"index": row.difficulty_index,
            "inote": difficulty.inote or "", "events": points, "duration_sec": chart_duration(row.source_path, row.difficulty_index)},
            "regions": annotation.get("regions", []) if annotation else [],
            "audio_url": f"/api/quality/audio?manifest={quote(str(manifest))}&chart_id={quote(chart_id)}" if row.audio_path else None,
            "corrected": self._corrected_detail(annotation), "annotation_path": str(annotation_path(manifest))}

    def preview(self, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        manifest = _required_manifest(payload)
        records = load_annotations(manifest)["records"]
        rows = _catalog(manifest)
        lookup = _AnnotationLookup(records)
        annotations = {item.chart_id: lookup.find(item) for item in rows}
        def dist(values: list[Any]) -> dict[str, int]:
            return dict(sorted(Counter(str(value or "") for value in values).items()))
        type_counts: Counter[str] = Counter()
        for row in rows:
            type_counts.update(row.type_counts or {})
        return HTTPStatus.OK, {"manifest": str(manifest), "annotation_path": str(annotation_path(manifest)), "total": len(rows),
            "labels": dist([annotations[row.chart_id].get("label") if annotations[row.chart_id] else "normal" for row in rows]),
            "weights": dist([annotations[row.chart_id].get("weight", 1) if annotations[row.chart_id] else 1 for row in rows]),
            "type_distribution": dict(sorted(type_counts.items())), "level_distribution": dist([row.level for row in rows]),
            "designer_distribution": dist([row.designer for row in rows]),
            "training_note": "区域权重会与现有 slide/touch 增采样倍率叠加；预览不重建大数据集。negative 无修正版只保存为待修正反例，不会作为真负样本直接进入 SFT。"}

    @staticmethod
    def _summary(rows: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, Any]:
        counts = Counter(str(row.get("label", "normal")) for row in rows)
        return {
            "total": len(rows), "annotated": len(records), "counts": dict(counts),
            "preference_pairs": sum(1 for record in records
                                     if record.get("label") == "negative" and record.get("corrected_path")),
            "negative_without_correction": sum(1 for record in records
                                                if record.get("label") == "negative" and not record.get("corrected_path")),
        }

    @staticmethod
    def _corrected_detail(annotation: dict[str, Any] | None) -> dict[str, Any] | None:
        if not annotation or not annotation.get("corrected_path"):
            return None
        path = Path(annotation["corrected_path"])
        if not path.is_file():
            return {"path": str(path), "missing": True}
        return {"path": str(path), "difficulty_index": annotation.get("corrected_difficulty"), "maidata": path.read_text(encoding="utf-8", errors="replace")}


# Small functional exports keep server.py integration explicit and testable.
_DEFAULT_API = QualityAPI()


def quality_catalog(payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    return _DEFAULT_API.catalog(payload)


def quality_save(payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    return _DEFAULT_API.save(payload)


def quality_detail(payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    return _DEFAULT_API.detail(payload)


def quality_preview(payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    return _DEFAULT_API.preview(payload)


def quality_audio(query: dict[str, Any], *, method: str = "GET", range_header: str | None = None,
                  send_body: bool = True) -> tuple[int, dict[str, str], Path | None, tuple[int, int] | None]:
    del send_body  # body streaming belongs to the HTTP handler
    return _DEFAULT_API.get_audio(query, method=method, range_header=range_header)


def _required_manifest(payload: dict[str, Any]) -> Path:
    value = payload.get("manifest")
    if not value:
        raise ValueError("manifest 不能为空")
    path = Path(str(value)).resolve()
    if not path.is_file():
        raise ValueError("manifest 不存在")
    return path


def _first(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value or "")


def _content_type(path: Path) -> str:
    return {".mp3": "audio/mpeg", ".wav": "audio/wav", ".ogg": "audio/ogg", ".flac": "audio/flac"}.get(path.suffix.lower(), "application/octet-stream")


def _sort_catalog_rows(rows: list[dict[str, Any]], sort_by: str, descending: bool) -> None:
    """Sort in-place with deterministic ascending tie breakers.

    Sorting ties first and the requested field second keeps pagination stable
    when several charts share a title/level while still making the selected
    field descend without reversing the tie order.
    """
    def text(value: Any) -> tuple[int, str]:
        normalized = str(value or "").strip().casefold()
        return (1 if not normalized else 0, normalized)

    def level(value: Any) -> tuple[int, float, str]:
        raw = str(value or "").strip().casefold()
        if not raw:
            return (1, float("inf"), "")
        # Maimai labels may be decimal levels or values such as 14+.
        number = raw.rstrip("+").strip()
        try:
            return (0, float(number), raw)
        except ValueError:
            return (0, float("inf"), raw)

    def selected(row: dict[str, Any]) -> Any:
        return level(row.get("level")) if sort_by == "level" else text(row.get(sort_by if sort_by != "folder" else "folder_name"))

    def tie(row: dict[str, Any]) -> tuple[Any, ...]:
        return (text(row.get("title")), text(row.get("folder_name")), text(row.get("song_id")),
                int(row.get("difficulty_index", 0) or 0), text(row.get("designer")), text(row.get("source_path")),
                text(row.get("chart_id")))

    rows.sort(key=tie)
    rows.sort(key=selected, reverse=descending)


def _parse_range(value: str | None, size: int) -> tuple[int, int, HTTPStatus]:
    if not value or not value.startswith("bytes="):
        return 0, max(0, size - 1), HTTPStatus.OK
    spec = value[6:].split(",", 1)[0].strip()
    try:
        left, right = spec.split("-", 1)
        if not left:
            length = int(right)
            start, end = max(0, size - length), size - 1
        else:
            start, end = int(left), int(right) if right else size - 1
        if start < 0 or start >= size or end < start:
            raise ValueError
        return start, min(end, size - 1), HTTPStatus.PARTIAL_CONTENT
    except (ValueError, TypeError):
        return 0, max(0, size - 1), HTTPStatus.OK

