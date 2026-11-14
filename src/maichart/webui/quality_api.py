"""WebUI-facing quality annotation service.

This module contains no HTTP server coupling.  ``QualityAPI`` returns ordinary
status/data pairs so the existing dependency-light handler can wire it in with
two small branches.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from collections import Counter, OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
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
    resolve_auto_audit,
    save_annotations,
    split_annotation_source,
    validate_record,
)

from .dataset_api import _json_data
from .material_audit_status import audit_state, published_audits
from .material_reviews import ReviewStore, chart_provenance, decision_signature, report_version, review_prompts, source_key

AUTO_REASON_LABELS: dict[str, str] = {
    "auto_audio_misalignment": "旧体检曾因疑似音画偏移排除，请重新体检",
    "auto_global_offset_review": "疑似整谱音画偏移",
    "auto_excessive_muri": "问题段过多（整谱排除）",
    "auto_hard_muri": "硬无理冲突 / 撞尾误触",
    "auto_unquantized_timing": "疑似节奏网格偏移",
    "auto_sensor_jackhammer": "疑似高速同点连打",
    "auto_local_tempo_drift": "疑似局部对音漂移",
    "auto_abandoned_gap": "音乐活跃处留白",
    "comfort": "手顺吃力（轻降权）",
    "auto_worthy_phrase": "优选好段落（采音准·手顺佳）",
    "auto_segment_muri_mask": "局部问题段已屏蔽",
    "auto_hand_flow_weight": "局部手顺已轻降权",
    "auto_worthy_highlight": "含优选好段落",
}


def _auto_scope(payload: dict[str, Any], *, allow_auto=False) -> str:
    scope = payload.get("auto_scope", "formal")
    if scope not in ("formal", "preview") and not (allow_auto and scope == "auto"):
        raise ValueError("未知自动体检结果范围")
    return scope


def auto_quality_view(manifest: Path, manifest_data: dict[str, Any], scope: str = "formal"):
    meta, formal_data, formal_report, preview_data, preview_report = published_audits(manifest, manifest_data)
    if scope == "auto":
        scope = "preview" if formal_data is None and preview_data is not None else "formal"
    meta["auto_scope"] = scope
    data = preview_data if scope == "preview" else formal_data
    report = preview_report if scope == "preview" else formal_report
    return meta, data, report if data is not None else None


def _audit_coverage(report: dict[str, Any] | None, rows: list[CatalogChart]) -> tuple[set[str], dict[str, str]]:
    report = report or {}
    current_ids = {row.chart_id for row in rows}
    verified = {chart_id for chart_id in (report.get("audited_chart_ids") or ())
                if isinstance(chart_id, str) and chart_id in current_ids}
    unverified = {item["chart_id"]: str(item.get("reason") or "unverified")
                  for item in (report.get("unverified_charts") or [])
                  if isinstance(item, dict) and item.get("chart_id") in current_ids}
    audio_stats = report.get("audited_chart_audio_stats")
    if isinstance(audio_stats, dict):
        audio_paths = report.get("audited_chart_audio_paths") or {}
        observed: dict[str, list[int] | None] = {}
        for row in rows:
            if row.chart_id not in audio_stats:
                continue
            current = None
            if row.audio_path:
                if row.audio_path not in observed:
                    try:
                        stat = Path(row.audio_path).stat()
                        observed[row.audio_path] = [stat.st_size, stat.st_mtime_ns]
                    except OSError:
                        observed[row.audio_path] = None
                current = observed[row.audio_path]
            if current != audio_stats[row.chart_id] or (row.chart_id in audio_paths and row.audio_path != audio_paths[row.chart_id]):
                verified.discard(row.chart_id)
                unverified[row.chart_id] = "audio_changed"
    return verified, unverified


def _audit_review_regions(report: dict[str, Any] | None) -> dict[str, list[dict[str, Any]]]:
    return {item["chart_id"]: item["regions"] for item in ((report or {}).get("review_regions") or [])
            if isinstance(item, dict) and isinstance(item.get("chart_id"), str)
            and isinstance(item.get("regions"), list)}


def _prompt_audio_version(row, audio_versions, stamps):
    if row.chart_id in audio_versions:
        return audio_versions[row.chart_id]
    if row.audio_path not in stamps and row.audio_path:
        stamps[row.audio_path] = _file_version(Path(row.audio_path))
    return stamps.get(row.audio_path)


def _review_spans(auto, report_spans):
    return [
        *(dict(span, review_kind="auto_weight") for span in (auto or {}).get("regions", ())),
        *(dict(span, review_kind="advisory") for span in report_spans),
    ]


def _review_settings(report):
    return {**(report.get("settings") or {}), "scope": report.get("scope") or None,
            "preview": bool(report.get("preview"))}


def _manual_override_scope(auto_rec: dict[str, Any] | None, manual_rec: dict[str, Any] | None) -> str | None:
    if auto_rec is None or manual_rec is None:
        return None
    if manual_rec.get("unmarked_weight") == 0 or (
        manual_rec.get("chart_override") is not False
        and (manual_rec["label"], manual_rec["weight"]) != (auto_rec["label"], auto_rec["weight"])
    ):
        return "chart"
    if any(a["start_sec"] < m["end_sec"] and m["start_sec"] < a["end_sec"]
           for a in auto_rec["regions"] for m in manual_rec["regions"]):
        return "region"
    return None


def _auto_region_effect_label(region: dict[str, Any]) -> str:
    label = str(region.get("label", "normal"))
    tw = float(region.get("timing_weight", 1.0))
    cw = float(region.get("component_weight", 1.0))
    if label == "good":
        return f"采音 {tw:g}× · 配置 {cw:g}× 优先学习"
    if tw <= 0.0 and cw <= 0.0:
        return "屏蔽采音与配置学习"
    if cw <= 0.0 and tw > 0.0:
        return f"保留采音 {tw:g}× · 屏蔽配置"
    if tw <= 0.0 and cw > 0.0:
        return f"屏蔽采音 · 保留配置 {cw:g}×"
    return f"采音 {tw:g}× · 配置 {cw:g}× 轻降权"


def _summarize_auto_row(
    auto_rec: dict[str, Any] | None,
    manual_rec: dict[str, Any] | None,
    *,
    verified: bool,
    unverified_reason: str | None,
    review_count: int = 0,
    review_resolved: int = 0,
    provenance: str | None = None,
) -> dict[str, Any]:
    override = _manual_override_scope(auto_rec, manual_rec)
    pending = review_count - review_resolved
    summary = {"auto_problem_count": 0, "auto_good_count": 0, "auto_review_count": pending,
               "auto_review_total": review_count, "auto_review_resolved_count": review_resolved,
               "auto_manual_override": override is not None, "auto_override_scope": override,
               "manual_provenance": provenance, "has_manual_regions": bool((manual_rec or {}).get("regions"))}
    if unverified_reason:
        reason = {"audio_unavailable": "缺少可用音频", "audio_changed": "音频已变化",
                  "partial_coverage": "检查覆盖不足", "parse_error": "谱面解析失败"}.get(unverified_reason)
        return {**summary, "auto_status": "unverified", "auto_unverified_reason": unverified_reason,
                "auto_badge": "待核对 · " + (reason or "体检未完成校验")}
    if auto_rec is None:
        if pending:
            return {**summary, "auto_status": "unverified", "auto_badge": f"待复核 {pending} 段"}
        if review_resolved:
            return {**summary, "auto_status": "reviewed", "auto_badge": f"复核完成 {review_resolved}/{review_count} 条"}
        return {**summary, "auto_status": "clean" if verified else "unreviewed",
                "auto_badge": "检查通过" if verified else "未确认体检覆盖"}
    regions = auto_rec.get("regions") or []
    hard_count = sum(1 for r in regions if r.get("label") == "problem" and r.get("reason") != "comfort")
    comfort_count = sum(1 for r in regions if r.get("label") == "problem" and r.get("reason") == "comfort")
    good_count = sum(1 for r in regions if r.get("label") == "good")
    if auto_rec.get("label") == "excluded" or float(auto_rec.get("weight", 1.0)) <= 0.0:
        reason = str(auto_rec.get("reason") or "")
        badge = (AUTO_REASON_LABELS[reason] if reason == "auto_audio_misalignment"
                 else "自动排除 · 问题段过多")
        if override == "chart":
            badge += " · 已采用人工整谱标注" if provenance == "manual_chart" else " · 整谱权重另有设置"
        return {
            **summary,
            "auto_status": "excluded",
            "auto_badge": badge,
            "auto_problem_count": hard_count + comfort_count,
        }
    parts: list[str] = []
    if hard_count:
        parts.append(f"局部屏蔽 {hard_count} 段")
    if comfort_count:
        parts.append(f"手顺降权 {comfort_count} 段")
    if good_count:
        parts.append(f"优选好段 {good_count} 段")
    if pending:
        parts.append(f"待复核 {pending} 段")
    elif review_resolved:
        parts.append(f"复核完成 {review_resolved}/{review_count} 条")
    status = ("masked" if hard_count else "downweighted" if comfort_count else "unverified" if pending
              else "good" if good_count else "reviewed" if review_resolved else "clean")
    return {
        **summary,
        "auto_status": status,
        "auto_badge": " · ".join(parts) if parts else "干净通过",
        "auto_problem_count": hard_count + comfort_count,
        "auto_good_count": good_count,
    }


def _format_auto_detail(
    auto_rec: dict[str, Any] | None,
    manual_rec: dict[str, Any] | None,
    *,
    verified: bool,
    scope: str,
    unverified_reason: str | None = None,
    review_regions: tuple[dict[str, Any], ...] = (),
    provenance: str | None = None,
) -> dict[str, Any]:
    summary = _summarize_auto_row(
        auto_rec, manual_rec, verified=verified, unverified_reason=unverified_reason,
        review_count=len(review_regions), review_resolved=sum(bool(item.get("reviewed")) for item in review_regions),
        provenance=provenance,
    )
    enriched_regions: list[dict[str, Any]] = []
    for item in (auto_rec or {}).get("regions") or []:
        if any(prompt["review_kind"] == "auto_weight"
               and (prompt["start_sec"], prompt["end_sec"], prompt["reason"])
               == (item["start_sec"], item["end_sec"], str(item.get("reason") or ""))
               for prompt in review_regions):
            continue
        reason_key = str(item.get("reason") or "")
        enriched_regions.append({
            **item,
            "reason_label": AUTO_REASON_LABELS.get(reason_key, reason_key or "自动标记片段"),
            "effect_label": _auto_region_effect_label(item),
        })
    for item in review_regions:
        reason_key = str(item.get("reason") or "")
        reason_label = AUTO_REASON_LABELS.get(reason_key, reason_key or "待复核片段")
        offset_ms = item.get("offset_ms")
        if reason_key == "auto_global_offset_review" and isinstance(offset_ms, (int, float)):
            reason_label += f"，估计 {offset_ms:+g} ms"
        if item["review_kind"] == "auto_weight":
            enriched_regions.append({**item, "reason_label": reason_label, "effect_label": _auto_region_effect_label(item)})
        else:
            enriched_regions.append({**item, "label": "uncertain", "reason_label": reason_label, "effect_label": "仅作复核提示"})
    enriched_regions.sort(key=lambda item: item["start_sec"])
    chart_reason = str((auto_rec or {}).get("reason") or "")
    return {
        **summary,
        "has_audit": verified or auto_rec is not None or bool(unverified_reason),
        "auto_scope": scope,
        "unverified_reason": unverified_reason,
        "chart_label": (auto_rec or {}).get("label", "normal"),
        "chart_reason": chart_reason,
        "chart_reason_label": AUTO_REASON_LABELS.get(chart_reason, chart_reason),
        "regions": enriched_regions,
    }


_CATALOG_LOCK = threading.RLock()
_CHART_INDEX_LOCK = threading.RLock()
_MUTATION_LOCKS = {}
_MATERIAL_LOCKS = threading.local()


def _file_version(path: Path):
    try:
        stat = path.stat()
        return stat.st_mtime_ns, stat.st_size, stat.st_ctime_ns, stat.st_ino
    except FileNotFoundError:
        return None


@dataclass(frozen=True, eq=False)
class _ChartCatalog:
    rows: list[CatalogChart]
    by_id: dict[str, CatalogChart]
    versions: dict[str, tuple | None]
    checked_at: float


def _read_catalog(manifest: Path) -> _ChartCatalog:
    rows = iter_catalog(manifest)
    assets = {row.source_path for row in rows}
    from maichart.training.quality_annotations import _resolve_manifest_path, _source_root

    data = _json_data(manifest)
    root = _source_root(manifest, data)
    for song in data.get("songs", ()):
        path = song.get("maidata_path")
        if path:
            raw = Path(path)
            assets.add(str(raw) if raw.is_absolute() else str(_resolve_manifest_path(raw, manifest, root)))
    asset_list = list(assets)
    if len(asset_list) > 256:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=8) as pool:
            stamps = list(pool.map(lambda p: _file_version(Path(p)), asset_list))
        versions = dict(zip(asset_list, stamps))
    else:
        versions = {path: _file_version(Path(path)) for path in asset_list}
    return _ChartCatalog(rows, {row.chart_id: row for row in rows}, versions, time.monotonic())


_CHART_CATALOGS: OrderedDict[Path, tuple] = OrderedDict()


def _mutation_lock(manifest):
    with _CATALOG_LOCK:
        return _MUTATION_LOCKS.setdefault(annotation_path(manifest), threading.RLock())


@contextmanager
def _material_write(manifest):
    from maichart.composer.training_lock import TrainingOutputBusyError, _directory_lock

    target = annotation_path(manifest)
    with _mutation_lock(manifest):
        held = getattr(_MATERIAL_LOCKS, "targets", set())
        if target in held:
            yield
            return
        _MATERIAL_LOCKS.targets = held
        try:
            with _directory_lock(target.parent, ".materials-review.lock"):
                held.add(target)
                try:
                    yield
                finally:
                    held.remove(target)
        except TrainingOutputBusyError as exc:
            raise ValueError("另一个 WebUI 正在保存这个训练集，请重试") from exc


class _AnnotationLookup:
    """Prefilter candidates, preserving the existing validator and record order."""
    def __init__(self, records: list[dict[str, Any]], catalog_rows=()) -> None:
        self.records: list[dict[str, Any] | None] = list(records)
        self.ids: dict[str, set[int]] = {}
        self.sources: dict[tuple[Path, int], set[int]] = {}
        self._resolved_paths: dict[str, Path] = {}
        self._path_lock = threading.Lock()
        for row in catalog_rows:
            self._resolved_paths[row.source_path] = Path(row.source_path)
            if row.audio_path:
                self._resolved_paths[row.audio_path] = Path(row.audio_path)
        for index, record in enumerate(self.records):
            if record is not None:
                self._register(index, record)

    def _resolve(self, value: str) -> Path:
        if value not in self._resolved_paths:
            with self._path_lock:
                if value not in self._resolved_paths:
                    self._resolved_paths[value] = Path(value).resolve()
        return self._resolved_paths[value]

    def _register(self, index: int, record: dict[str, Any]) -> None:
        record_id = record.get("chart_id")
        if isinstance(record_id, str) and record_id:
            self.ids.setdefault(record_id, set()).add(index)
        if record.get("source_path") and record.get("difficulty_index") is not None:
            try:
                key = (self._resolve(record["source_path"]), int(record["difficulty_index"]))
                self.sources.setdefault(key, set()).add(index)
            except (OSError, ValueError, TypeError):
                pass

    def _unregister(self, index, record):
        if record is None:
            return
        record_id = record.get("chart_id")
        if record_id in self.ids:
            self.ids[record_id].discard(index)
        if record.get("source_path") and record.get("difficulty_index") is not None:
            try:
                slot = self._resolve(record["source_path"]), int(record["difficulty_index"])
            except (OSError, ValueError, TypeError):
                return
            if slot in self.sources:
                self.sources[slot].discard(index)

    def _candidate_indices(self, row: CatalogChart, extra_id: str | None = None) -> list[int]:
        indices: set[int] = set()
        for cid in (extra_id, row.chart_id, row.legacy_chart_id):
            if isinstance(cid, str) and cid:
                indices |= self.ids.get(cid, set())
        try:
            indices |= self.sources.get((self._resolve(row.source_path), int(row.difficulty_index)), set())
        except (OSError, ValueError, TypeError):
            pass
        return [index for index in sorted(indices) if self.records[index] is not None]

    def _matching_indices(self, row: CatalogChart, chart_id: str) -> list[int]:
        try:
            row_slot = (self._resolve(row.source_path), int(row.difficulty_index))
        except (OSError, ValueError, TypeError):
            row_slot = None
        matched: list[int] = []
        for index in self._candidate_indices(row, extra_id=chart_id):
            rec = self.records[index]
            if rec is None:
                continue
            rec_id = rec.get("chart_id")
            if isinstance(rec_id, str) and rec_id == chart_id:
                matched.append(index)
                continue
            if find_annotation(row, [rec], resolve_path=self._resolve) is not None:
                matched.append(index)
                continue
            if not rec_id and row_slot is not None and rec.get("source_path") and rec.get("difficulty_index") is not None:
                try:
                    rec_slot = (self._resolve(rec["source_path"]), int(rec["difficulty_index"]))
                except (OSError, ValueError, TypeError):
                    rec_slot = None
                if rec_slot == row_slot:
                    matched.append(index)
        return matched

    def replace_for_row(self, row: CatalogChart, chart_id: str, record: dict[str, Any]) -> None:
        matched = self._matching_indices(row, chart_id)
        if matched:
            primary = matched[0]
            for index in matched:
                self._unregister(index, self.records[index])
            self.records[primary] = record
            for extra in matched[1:]:
                self.records[extra] = None
            self._register(primary, record)
        else:
            index = len(self.records)
            self.records.append(record)
            self._register(index, record)

    def remove_for_row(self, row: CatalogChart, chart_id: str) -> None:
        for index in self._matching_indices(row, chart_id):
            self._unregister(index, self.records[index])
            self.records[index] = None

    def active_records(self) -> list[dict[str, Any]]:
        return [record for record in self.records if record is not None]

    def find(self, row: CatalogChart, *, verify_audio=False, audio_sha256=None) -> dict[str, Any] | None:
        candidates = [self.records[index] for index in self._candidate_indices(row)]
        return find_annotation(row, [c for c in candidates if c is not None], resolve_path=self._resolve,
                               verify_audio=verify_audio, verified_audio_sha256=audio_sha256)


def _is_zero_unmarked_manifest(manifest: Path) -> bool:
    try:
        data = json.loads(Path(manifest).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, TypeError):
        return False
    return bool(data.get("candidate_pool") or data.get("reviewed_training"))


def _sanitize_zero_unmarked_regions(regions: Any) -> Any:
    if not isinstance(regions, list):
        return regions
    sanitized: list[Any] = []
    for region in regions:
        if not isinstance(region, dict):
            sanitized.append(region)
            continue
        item = dict(region)
        label = str(item.get("label", item.get("reason") or "manual")).strip()
        if label != "good":
            item["timing_weight"] = 0.0
            item["component_weight"] = 0.0
            item.pop("entry_start_sec", None)
            item.pop("learn_entry", None)
        sanitized.append(item)
    return sanitized


def _catalog_snapshot(manifest: Path, *, force=False) -> _ChartCatalog:
    with _CHART_INDEX_LOCK:
        version = _file_version(manifest)
        previous = _CHART_CATALOGS.get(manifest)
        if previous is None or version != previous[0]:
            cached = _read_catalog(manifest)
            checked_at = cached.checked_at
        else:
            _, cached, checked_at = previous
            ttl = 0.0 if len(cached.versions) <= 256 else 60.0
            if force or time.monotonic() - checked_at >= ttl:
                if any(_file_version(Path(path)) != stamp for path, stamp in cached.versions.items()):
                    cached = _read_catalog(manifest)
                checked_at = time.monotonic()
        _CHART_CATALOGS[manifest] = version, cached, checked_at
        _CHART_CATALOGS.move_to_end(manifest)
        while len(_CHART_CATALOGS) > 6:
            _CHART_CATALOGS.popitem(last=False)
        return cached


def _catalog(manifest: Path) -> list[CatalogChart]:
    return _catalog_snapshot(manifest).rows


def _catalog_row(manifest: Path, chart_id: str) -> CatalogChart | None:
    catalog = _catalog_snapshot(manifest)
    row = catalog.by_id.get(chart_id)
    if row and _file_version(Path(row.source_path)) != catalog.versions.get(row.source_path):
        return _catalog_snapshot(manifest, force=True).by_id.get(chart_id)
    return row


@lru_cache(maxsize=8)
def _annotation_snapshot(manifest: Path, version, catalog: _ChartCatalog) -> _AnnotationLookup:
    return _AnnotationLookup(load_annotations(manifest)["records"], catalog.rows)


@lru_cache(maxsize=6)
def _origin_rows(catalog: _ChartCatalog):
    return {(row.source_path, row.difficulty_index, row.chart_sha256): row for row in catalog.rows}


def _annotations(manifest: Path, catalog=None) -> _AnnotationLookup:
    return _annotation_snapshot(manifest, _file_version(annotation_path(manifest)),
                                catalog if catalog is not None else _catalog_snapshot(manifest))


_SEARCH_FIELDS = ("song_id", "title", "maidata_title", "artist", "level", "designer",
                  "folder_name", "source_dataset", "source_path")


@dataclass
class _CatalogView:
    chart_catalog: _ChartCatalog
    version: tuple
    rows: list[tuple[str, dict[str, Any]]]
    summary: dict[str, Any]
    audit: dict[str, Any]
    built_at: float
    audio_versions: dict[str, tuple | None]
    checked_assets_at: float = 0.0
    candidate_asset_stats: dict[str, Any] | None = None
    sorted_indices: dict[tuple[str, bool], tuple[int, ...]] | None = None
    filter_cache: OrderedDict[tuple, tuple[dict[str, Any], ...]] | None = None


_VIEWS: OrderedDict[tuple, _CatalogView] = OrderedDict()
_CANDIDATE_ASSET_STATS: OrderedDict[Path, tuple[float, dict[str, Any]]] = OrderedDict()


def _compute_row_training_effect(
    value: dict[str, Any],
    *,
    is_candidate: bool,
    is_reviewed_training: bool,
) -> dict[str, Any]:
    """Summarize whether a catalog row enters training under default Composer v2 settings and its weight."""
    from maichart.training.event_dataset import numeric_level
    from maichart.training.quality_annotations import zero_unmarked_region_weights

    slot = int(value.get("difficulty_index") or 0)
    if slot not in (4, 5, 6):
        return {"status": "excluded", "weight": 0.0, "text": f"不入训 · 槽 {slot} 非常规"}
    raw_level = value.get("level")
    lvl: float | None = None
    if raw_level is not None and str(raw_level).strip():
        try:
            lvl = float(numeric_level(raw_level))
        except ValueError:
            lvl = -1.0
    if lvl is not None and not (12.0 <= lvl <= 15.0):
        label_str = str(raw_level).strip() or "?"
        return {"status": "excluded", "weight": 0.0, "text": f"默认不入训 · 定数 {label_str}"}

    rec = value.get("annotation") or {}
    if is_candidate:
        c_status = value.get("candidate_status")
        r_status = value.get("candidate_review_status")
        if c_status in {"stale", "pending", "error"} or r_status in {"unreviewed", "exclude"}:
            if r_status == "exclude":
                text = "不入训 · 0×（已排除）"
            elif c_status == "error":
                text = "不入训 · 0×（扫描异常）"
            elif rec.get("auto_screened") and str(rec.get("reason") or "").startswith("deferred:"):
                text = "待核对 · 0×（暂不入训）"
            else:
                text = "未放行 · 0×"
            return {"status": "excluded", "weight": 0.0, "text": text}
        raw_regs = rec.get("regions") or ()
        reg_max = max((max(zero_unmarked_region_weights(r)) for r in raw_regs), default=1.0)
        eff_w = round(float(rec.get("weight", 1.0)) * reg_max, 3)
        masked_n = sum(1 for r in raw_regs if r.get("label") == "problem" or max(zero_unmarked_region_weights(r)) <= 0)
        if slot == 4:
            return {"status": "expert_opt_in", "weight": eff_w, "text": f"默认不入训（勾 EXPERT 为 {eff_w:g}×）"}
        if r_status == "learn_all" and masked_n == 0:
            return {
                "status": "boosted" if eff_w > 1.0 + 1e-6 else "included",
                "weight": eff_w,
                "text": f"入训 · {eff_w:g}×（整谱）",
            }
        if masked_n > 0:
            return {"status": "masked", "weight": eff_w, "text": f"入训 · {eff_w:g}×（屏蔽 {masked_n} 段）"}
        return {"status": "masked", "weight": eff_w, "text": f"入训 · {eff_w:g}×（仅学 {value.get('confirmed_regions', 1)} 段）"}

    if is_reviewed_training:
        from maichart.training.candidate_pool import _valid_regions

        if not _valid_regions(rec):
            return {"status": "excluded", "weight": 0.0, "text": "不入训 · 0×"}
        eff_w = round(float(rec.get("weight", 1.0)), 3)
        if slot == 4:
            return {"status": "expert_opt_in", "weight": eff_w, "text": f"默认不入训（勾 EXPERT 为 {eff_w:g}×）"}
        return {"status": "included", "weight": eff_w, "text": f"入训 · {eff_w:g}×（示范片段）"}

    has_manual = value.get("annotation") is not None
    auto_st = value.get("auto_status")
    if has_manual and (rec.get("label") in {"excluded", "negative"} or float(rec.get("weight", 1.0)) <= 0):
        return {"status": "excluded", "weight": 0.0, "text": "不入训 · 0×（人工排除）"}
    if auto_st == "excluded" and not (has_manual and rec.get("chart_override") is not False):
        return {"status": "excluded", "weight": 0.0, "text": "不入训 · 0×（体检排除）"}

    eff_w = round(float(rec.get("weight", 1.0)) if has_manual else 1.0, 3)
    manual_probs = sum(1 for r in (rec.get("regions") or ()) if r.get("label") == "problem")
    auto_probs = int(value.get("auto_problem_count") or 0)
    if slot == 4:
        return {"status": "expert_opt_in", "weight": eff_w, "text": f"默认不入训（勾 EXPERT 为 {eff_w:g}×）"}
    if manual_probs > 0 or auto_probs > 0 or auto_st in {"masked", "downweighted"}:
        return {"status": "masked", "weight": eff_w, "text": f"入训 · {eff_w:g}×（含局部屏蔽/降权）"}
    if eff_w > 1.0 + 1e-6:
        return {"status": "boosted", "weight": eff_w, "text": f"加权入训 · {eff_w:g}×"}
    if eff_w < 1.0 - 1e-6:
        return {"status": "masked", "weight": eff_w, "text": f"降权入训 · {eff_w:g}×"}
    return {"status": "included", "weight": 1.0, "text": "入训 · 1×"}


def _view_version(manifest, data):
    paths = [annotation_path(manifest)]
    if data.get("candidate_pool"):
        paths.extend(manifest.parent / name for name in ("scan_state.json", "pool.json"))
        paths.append(manifest.parent / "scan" / "results.json")
    elif not data.get("reviewed_training"):
        owners = [manifest]
        origin = split_annotation_source(manifest, data)
        if origin is not None and origin != manifest:
            owners.append(origin)
        for owner in owners:
            paths.append(owner)
            paths.extend(owner.with_name(name) for name in (
                "auto_quality_annotations.json", "composed_auto_quality_annotations.json",
                "auto_quality_annotations_preview.json", "auto_quality_audit_report.json",
                "auto_quality_audit_preview_report.json", "auto_quality_audit_state.json",
                "auto_quality_audit_preview_state.json",
                "auto_quality_audit_report_previous.json", "auto_quality_audit_preview_report_previous.json",
            ))
    return tuple((path, _file_version(path)) for path in paths)


class QualityAPI:
    """Stateless quality annotation operations for one local WebUI."""

    def __init__(self, workspace_root=None):
        self.workspace_root=Path(workspace_root).absolute() if workspace_root else Path(__file__).resolve().parents[3]/'webui_workspace'
        self.reviews = ReviewStore(self.workspace_root)

    @staticmethod
    def _review_source(manifest, row):
        return source_key(row.source_manifest or manifest, row)

    @staticmethod
    def _require_editable(manifest):
        data = _json_data(manifest)
        if data.get("merged_from") or data.get("split_metadata"):
            raise ValueError("这份训练快照为只读，请回到原训练素材保存")

    def _chart_review_context(self, manifest, row, scope, *, audio_sha256=None):
        _, data, report = resolve_auto_audit(manifest, _json_data(manifest), preview=scope == "preview")
        record = _annotations(manifest).find(row, verify_audio=True, audio_sha256=audio_sha256)
        chart_history, prompt_history = self.reviews.read()
        source = self._review_source(manifest, row)
        verified, unverified = _audit_coverage(report if data is not None else None, [row])
        reason = unverified.get(row.chart_id)
        auto = (find_annotation(row, data["records"], verify_audio=True, verified_audio_sha256=audio_sha256)
                if data is not None and not reason else None)
        report_spans = _audit_review_regions(report).get(row.chart_id, ())
        spans = _review_spans(auto, report_spans) if report is not None and row.chart_id in verified else ()
        audio_versions = (report or {}).get("audited_chart_audio_stats") or {}
        prompts = review_prompts(row, source, spans, _review_settings(report) if report is not None else {},
                                 _prompt_audio_version(row, audio_versions, {}) if spans else None,
                                 prompt_history, record, audio_sha256=audio_sha256)
        detail = _format_auto_detail(auto, record, verified=row.chart_id in verified, scope=scope,
                                     unverified_reason=reason, review_regions=tuple(prompts),
                                     provenance=chart_provenance(source, record, chart_history))
        if prompts:
            detail["report_version"] = report_version(report)
        if report_spans and row.chart_id not in verified:
            detail["review_unavailable"] = True
        return detail, prompts, source

    def _catalog_view(self, manifest, scope, *, force=False):
        manifest_data = _json_data(manifest)
        version = _view_version(manifest, manifest_data)
        version += ((self.reviews.path, _file_version(self.reviews.path)),)
        key = (self.workspace_root, manifest, scope)
        # Candidate status and reported audio coverage can change when source
        # assets are edited without touching the manifest or its sidecars.
        is_candidate = bool(manifest_data.get("candidate_pool"))
        has_audit = any(
            file_version and "report" in path.name for path, file_version in version
        )
        chart_catalog = _catalog_snapshot(manifest, force=force)
        with _CATALOG_LOCK:
            cached = _VIEWS.get(key)
        if cached is not None and not force and cached.chart_catalog is chart_catalog and cached.version == version:
            small_corpus = len(chart_catalog.rows) <= 256
            asset_ttl = 0.0 if small_corpus else 60.0
            now_mono = time.monotonic()
            assets_ok = True
            if now_mono - cached.checked_assets_at >= asset_ttl:
                if cached.audio_versions and not all(
                    _file_version(Path(path)) == stamp for path, stamp in cached.audio_versions.items()
                ):
                    assets_ok = False
                if assets_ok and cached.candidate_asset_stats is not None:
                    from maichart.training.candidate_pool import _read, _stat

                    meta = _read(manifest.parent / "pool.json")
                    if bool(meta) and meta.get("source_manifest_stat") != _stat(meta.get("source_manifest")):
                        assets_ok = False
                    elif not all(_stat(path) == stamp for path, stamp in cached.candidate_asset_stats.items()):
                        assets_ok = False
                if assets_ok:
                    cached.checked_assets_at = now_mono
            if assets_ok:
                with _CATALOG_LOCK:
                    if _VIEWS.get(key) is cached:
                        _VIEWS.move_to_end(key)
                return cached
        lookup = _annotations(manifest, chart_catalog)
        chart_rows = chart_catalog.rows
        matched = [(row, lookup.find(row)) for row in chart_rows]
        values = [row.to_dict(record) for row, record in matched]
        audit = dict(available=False, manifest=str(manifest), auto_scope="formal" if scope == "auto" else scope,
                     has_annotations=False, has_preview_annotations=False, report=None, preview_report=None)
        audio_stamps = {}
        candidate_stats_map = None
        is_reviewed_training = bool(manifest_data.get("reviewed_training"))
        if is_candidate:
            from maichart.training.candidate_pool import _read, _resolved_slot, _stat, _valid_regions, pool_charts, scan_issues
            scan_issues(manifest)
            entries = pool_charts(manifest)
            metadata = _read(manifest.parent / "pool.json")
            state = _read(manifest.parent / "scan_state.json").get("status")
            source_changed = bool(metadata) and metadata.get("source_manifest_stat") != _stat(metadata.get("source_manifest"))
            stats: dict[str, Any] = {}
            if metadata.get("scan_complete") and state not in {"queued", "running"}:
                large_pool = len(values) > 256
                reused_stats = None
                if large_pool and not force and not source_changed:
                    with _CATALOG_LOCK:
                        prev_stats = _CANDIDATE_ASSET_STATS.get(manifest)
                    if prev_stats is not None and time.monotonic() - prev_stats[0] < 60.0:
                        reused_stats = prev_stats[1]
                if reused_stats is not None:
                    stats = reused_stats
                else:
                    needed_paths: list[str] = []
                    seen_needed: set[str] = set()
                    for value in values:
                        entry = entries.get(_resolved_slot(value["source_path"], value["difficulty_index"]))
                        if not entry:
                            continue
                        song_obj, _ = entry
                        for p_val in (song_obj.get("maidata_path"), song_obj.get("audio_path")):
                            if p_val and p_val not in seen_needed:
                                seen_needed.add(p_val)
                                needed_paths.append(p_val)
                    if len(needed_paths) > 256:
                        from concurrent.futures import ThreadPoolExecutor

                        with ThreadPoolExecutor(max_workers=8) as pool:
                            stat_vals = list(pool.map(_stat, needed_paths))
                        stats = dict(zip(needed_paths, stat_vals))
                    else:
                        stats = {p_val: _stat(p_val) for p_val in needed_paths}
                    if large_pool:
                        with _CATALOG_LOCK:
                            _CANDIDATE_ASSET_STATS[manifest] = (time.monotonic(), stats)
                            _CANDIDATE_ASSET_STATS.move_to_end(manifest)
                            while len(_CANDIDATE_ASSET_STATS) > 6:
                                _CANDIDATE_ASSET_STATS.popitem(last=False)
                candidate_stats_map = stats
            for value in values:
                record = value.get("annotation") or {}
                entry = entries.get(_resolved_slot(value["source_path"], value["difficulty_index"]))
                status = "pending"
                if entry and state not in {"queued", "running"}:
                    song, difficulty = entry
                    status = difficulty.get("candidate_status", "pending")
                    if metadata.get("scan_complete"):
                        identity = difficulty.get("candidate_identity", {})
                        source, audio = song.get("maidata_path"), song.get("audio_path")
                        for path in (source, audio):
                            if path and path not in stats:
                                stats[path] = _stat(path)
                        if source_changed or stats.get(source) != identity.get("source_stat") or stats.get(audio) != identity.get("audio_stat"):
                            status = "stale"
                value["candidate_status"] = status
                value["confirmed_regions"] = len(_valid_regions(record)) if status != "stale" else 0
                value["candidate_chart_review"] = record.get("candidate_chart_review") if status != "stale" else None
                value["candidate_review_status"] = (
                    "unreviewed" if status == "stale"
                    else "exclude" if record.get("label") in {"excluded", "negative"} or record.get("weight", 1.) <= 0
                    else "learn_all" if record.get("candidate_chart_review") == "learn_all"
                    else "partial" if value["confirmed_regions"] else "unreviewed"
                )
        elif not is_reviewed_training:
            audit, auto_data, report = auto_quality_view(manifest, manifest_data, scope)
            auto_lookup = _AnnotationLookup(auto_data["records"], chart_rows) if auto_data is not None else None
            verified, unverified = _audit_coverage(report, chart_rows)
            prompts = _audit_review_regions(report)
            chart_history, prompt_history = self.reviews.read()
            settings = _review_settings(report) if report is not None else {}
            audio_versions = (report or {}).get("audited_chart_audio_stats") or {}
            if has_audit:
                audio_paths_list = list({
                    row.audio_path for row in chart_rows if row.chart_id in audio_versions and row.audio_path
                })
                if len(audio_paths_list) > 256:
                    from concurrent.futures import ThreadPoolExecutor

                    with ThreadPoolExecutor(max_workers=8) as pool:
                        audio_vers = list(pool.map(lambda p: _file_version(Path(p)), audio_paths_list))
                    audio_stamps = dict(zip(audio_paths_list, audio_vers))
                else:
                    audio_stamps = {path: _file_version(Path(path)) for path in audio_paths_list}
            for (row, record), value in zip(matched, values):
                auto_rec = auto_lookup.find(row) if auto_lookup is not None and row.chart_id not in unverified else None
                source = self._review_source(manifest, row)
                spans = _review_spans(auto_rec, prompts.get(row.chart_id, ())) if report is not None and row.chart_id in verified else ()
                reviewed = review_prompts(row, source, spans, settings,
                                          _prompt_audio_version(row, audio_versions, audio_stamps), prompt_history, record
                                          ) if spans else ()
                value.update(_summarize_auto_row(
                    auto_rec, record, verified=row.chart_id in verified, unverified_reason=unverified.get(row.chart_id),
                    review_count=len(reviewed), review_resolved=sum(item["reviewed"] for item in reviewed),
                    provenance=chart_provenance(source, record, chart_history),
                ))
            if report is not None and ("review_regions" in report or any(row["auto_review_total"] for row in values)):
                report_key = "preview_report" if audit["auto_scope"] == "preview" else "report"
                total_prompts = sum(row["auto_review_total"] for row in values)
                resolved = sum(row["auto_review_resolved_count"] for row in values)
                unavailable = sum(len(spans) for chart_id, spans in prompts.items() if chart_id not in verified)
                audit[report_key]["prompt_review_summary"] = dict(
                    total=total_prompts, resolved=resolved, pending=total_prompts - resolved,
                    pending_charts=sum(row["auto_review_count"] > 0 for row in values),
                    unavailable=unavailable,
                )
                audit[report_key]["audio_changed_charts"] = sum(reason == "audio_changed" for reason in unverified.values())
        for value in values:
            value["training_effect"] = _compute_row_training_effect(
                value,
                is_candidate=is_candidate,
                is_reviewed_training=is_reviewed_training,
            )
        rows = [
            (
                (" ".join(str(value.get(k, "")) for k in _SEARCH_FIELDS) + " " + str((value.get("training_effect") or {}).get("text", ""))).casefold(),
                value,
            )
            for value in values
        ]
        now_ts = time.monotonic()
        view = _CatalogView(
            chart_catalog,
            version,
            rows,
            self._summary(values, lookup.active_records()),
            audit,
            now_ts,
            audio_stamps,
            checked_assets_at=now_ts,
            candidate_asset_stats=candidate_stats_map,
            sorted_indices={},
            filter_cache=OrderedDict(),
        )
        with _CATALOG_LOCK:
            _VIEWS[key] = view
            _VIEWS.move_to_end(key)
            while len(_VIEWS) > 6:
                _VIEWS.popitem(last=False)
        return view

    def post(self, endpoint: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        if endpoint == '/api/quality/chart-preview':return self.chart_preview(payload)
        if endpoint == '/api/quality/source-feedback':return self.source_feedback(payload)
        if endpoint == "/api/quality/generation-feedback":
            return self.generation_feedback(payload)
        if endpoint == "/api/quality/candidate-feedback":
            return self.candidate_feedback(payload)
        if endpoint in {"/api/quality/reviewed-collection", "/api/quality/export-reviewed"}:
            from maichart.training.reviewed_generation import export_reviewed, library_state, update_library
            root=self.workspace_root/'reviewed_training'
            inspect=endpoint.endswith("reviewed-collection")
            result = export_reviewed(payload.get("candidates"),root,inspect_only=inspect,allow_empty=True)
            if inspect:result.update(library_state(root,payload.get('task_id'),result))
            else:result=update_library(result,payload.get('task_id'),root)
            return HTTPStatus.OK, result
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

    def chart_preview(self, payload):
        """Open one actual dataset slot in the same chart review workbench."""
        import hashlib
        from maichart.maidata import resolve_chart_first
        from maichart.webui.chart_preview import _resource
        from maichart.webui.chart_timeline import timeline
        manifest=_required_manifest(payload)
        row = _catalog_row(manifest, payload.get("chart_id"))
        if row is None:raise ValueError('谱面不属于当前训练来源')
        identity=_feedback_chart(dict(path=row.source_path,difficulty_index=row.difficulty_index))
        parsed=parse_maidata_file(row.source_path)
        difficulty=next(item for item in parsed.difficulties if item.index==row.difficulty_index)
        # The renderer opens the first populated slot. Make a read-only view of
        # the chosen slot, never accidentally play an easier slot in the file.
        text=f'&title={parsed.title or "训练素材"}\n&wholebpm={parsed.wholebpm or 120}\n&first={resolve_chart_first(parsed,row.difficulty_index) or 0}\n&lv_5={difficulty.level or ""}\n&inote_5={difficulty.inote or ""}\n'
        digest=hashlib.sha256(text.encode()).hexdigest()
        root=self.workspace_root/'preview_resources'
        root.mkdir(parents=True,exist_ok=True);view=root/f'training_{digest}.txt'
        if not view.exists():view.write_text(text,encoding='utf-8')
        audio=_feedback_audio(dict(audio_path=row.audio_path)) if row.audio_path else {}
        notes=timeline(row.source_path,row.difficulty_index)
        _feedback_chart(identity)  # Reject replacement while building the view.
        version=dict(**identity,**audio,title=parsed.title or row.song_id,label=f'训练素材 · {difficulty.level or ""}',
                     version_id=digest[:24],content_kind='complete_chart',url=_resource(view,difficulty_index=5),timeline=notes,
                     audio_url=_resource(Path(row.audio_path)) if row.audio_path else None,
                     audio_name=Path(row.audio_path).name if row.audio_path else None)
        annotation = _annotations(manifest).find(row, verify_audio=True, audio_sha256=audio.get("audio_sha256")) or {}
        manifest_data = _json_data(manifest)
        context=dict(manifest=str(manifest),chart_id=row.chart_id,
            chart_review=annotation.get('candidate_chart_review'),
            learning_excluded=annotation.get('label') in {'excluded','negative'} or annotation.get('weight',1)==0,
            candidate_pool=bool(manifest_data.get('candidate_pool')),
            auto_scope=_auto_scope(payload),
            zero_unmarked=bool(manifest_data.get('candidate_pool') or manifest_data.get('reviewed_training')
                               or annotation.get('unmarked_weight')==0))
        visible=annotation.get('regions',[])
        if context['candidate_pool']:
            from maichart.training.candidate_pool import _read, _resolved_slot, _stale, pool_charts, scan_issues
            key = _resolved_slot(row.source_path, row.difficulty_index)
            scanned = scan_issues(manifest).get(key)
            status, issues = scanned[1:] if scanned is not None and scanned[0] == row.chart_id else ("pending", [])
            metadata = _read(manifest.parent / "pool.json")
            pool_entry = pool_charts(manifest).get(key)
            if pool_entry and metadata.get("scan_complete") and _stale(manifest.parent, metadata, *pool_entry):
                status, issues = "stale", []
            context.update(candidate_status=status, issues=issues)
            visible=[*visible,*[item for item in annotation.get('candidate_reviews',[]) if item.get('label')=='uncertain']]
        elif not manifest_data.get('reviewed_training'):
            detail, _, _ = self._chart_review_context(manifest, row, _auto_scope(payload), audio_sha256=audio.get("audio_sha256"))
            context["auto_annotation"] = detail
            context["audit_read_only"] = bool(manifest_data.get("merged_from") or manifest_data.get("split_metadata"))
            origin_value = row.source_manifest or (manifest_data.get("split_metadata") or {}).get("source_manifest")
            if context["audit_read_only"] and origin_value:
                origin = Path(origin_value).resolve()
                if origin.is_file() and origin != manifest:
                    original = _origin_rows(_catalog_snapshot(origin)).get(
                        (row.source_path, row.difficulty_index, row.chart_sha256))
                    if original is not None and os.path.normcase(original.audio_path or "") == os.path.normcase(row.audio_path or ""):
                        context["origin_jump"] = dict(manifest=str(origin), chart_id=original.chart_id,
                                                      name=row.source_dataset or origin.parent.name)
        return HTTPStatus.OK,dict(task_id='source-'+row.chart_id,revision=identity['file_sha256'],annotations=annotation.get('regions',[]),
            source_kind='training_source',training_context=context,charts=[version],saved_reviews=visible)

    def source_feedback(self,payload):
        manifest = _required_manifest(payload)
        from .candidate_api import _is_pool, remember_undo
        self._require_editable(manifest)
        with _material_write(manifest):
            action = payload.get("prompt_action")
            if action in ("keep", "defer", "reopen"):
                receipt = self._review_prompt(payload, manifest)
                return HTTPStatus.OK, self._review_response(manifest, receipt)
            if action is not None and action != "annotate":
                raise ValueError("未知复核操作")
            if action == "annotate" and payload.get("label") not in {"good", "problem"}:
                raise ValueError("请选择明确的片段评价后再完成复核")
            receipt = self._review_prompt(payload, manifest, save=False) if action == "annotate" else None
            candidate = _is_pool(manifest)
            row = _catalog_row(manifest, payload.get("chart_id"))
            old = _annotations(manifest).find(row) if candidate and row is not None else None
            status, result = self._source_feedback(payload)
            if candidate and status == HTTPStatus.OK:
                current = next((r for r in result['records'] if r.get('chart_id') == payload.get('chart_id')), None)
                result['undo_token'] = remember_undo(manifest, payload['chart_id'], old, current)
                from .dataset_api import _pool_card

                result["candidate_pool_summary"] = _pool_card(manifest.parent, _json_data(manifest))
            if receipt is not None and status == HTTPStatus.OK:
                source, prompt, audio_sha, *_ = receipt
                try:
                    self.reviews.save_prompt(source, prompt, "annotated", audio_sha)
                except (OSError, sqlite3.Error) as exc:
                    raise ValueError(f"片段标注已保存，复核状态保存失败，请重试。{exc}") from exc
                result.update(self._review_response(manifest, receipt))
            elif not candidate and not _is_zero_unmarked_manifest(manifest) and row is not None:
                result["auto_annotation"] = self._chart_review_context(manifest, row, _auto_scope(payload))[0]
            return status, result

    def _review_prompt(self, payload, manifest, *, save=True):
        row = _catalog_row(manifest, payload.get("chart_id"))
        if row is None or _json_data(manifest).get("reviewed_training"):
            raise ValueError("请在原训练库打开要复核的谱面")
        if not payload.get("file_sha256") or not payload.get("chart_sha256") or (row.audio_path and not payload.get("audio_sha256")):
            raise ValueError("请重新打开谱面和音频后复核")
        identity = _feedback_chart({**payload, "path": row.source_path, "difficulty_index": row.difficulty_index})
        if identity["chart_sha256"] != row.chart_sha256:
            raise ValueError("谱面已变化，请重新打开")
        audio = _feedback_audio({"audio_path": row.audio_path, "audio_sha256": payload.get("audio_sha256")})
        audio_sha = audio.get("audio_sha256") or "no-audio"
        target = payload.get("review_target")
        if not isinstance(target, dict):
            raise ValueError("缺少提示的报告版本")
        audit_manifest = _required_manifest(target)
        if _is_zero_unmarked_manifest(audit_manifest):
            raise ValueError("请选择 Composer 正式体检或抽查产生的提示")
        audit_row = _catalog_row(audit_manifest, target.get("chart_id"))
        if audit_row is None or self._review_source(audit_manifest, audit_row) != self._review_source(manifest, row):
            raise ValueError("体检提示不属于当前来源，请重新打开")
        if (audit_row.chart_sha256 != row.chart_sha256
                or os.path.normcase(audit_row.audio_path or "") != os.path.normcase(row.audio_path or "")):
            raise ValueError("体检提示的谱面或音频版本已变，请重新体检")
        scope = _auto_scope(target)
        detail, prompts, source = self._chart_review_context(audit_manifest, audit_row, scope, audio_sha256=audio_sha)
        prompt = next((item for item in prompts if item["prompt_id"] == target.get("prompt_id")), None)
        if prompt is None or not target.get("report_version") or detail.get("report_version") != target["report_version"]:
            raise ValueError("体检结果已更新，请重新打开这条提示")
        if payload["prompt_action"] == "annotate":
            bounds = _feedback_span(payload)
            if ("start_sec" not in bounds or bounds["start_sec"] > prompt["start_sec"] + 1e-6
                    or bounds["end_sec"] < prompt["end_sec"] - 1e-6):
                raise ValueError("完成此条复核时，标注范围须覆盖整条提示；也可只保存局部标注")
        if save:
            self.reviews.save_prompt(source, prompt, {"keep": "kept", "defer": "deferred", "reopen": "reopened"}[
                payload["prompt_action"]
            ], audio_sha)
        return source, prompt, audio_sha, audit_manifest, audit_row, scope

    def _review_response(self, manifest, receipt):
        _, prompt, audio_sha, audit_manifest, audit_row, scope = receipt
        detail, updated, _ = self._chart_review_context(audit_manifest, audit_row, scope, audio_sha256=audio_sha)
        result = dict(prompt_review=next((item for item in updated if item["prompt_id"] == prompt["prompt_id"]), None),
                      review_manifest=str(audit_manifest), review_scope=scope, review_chart_id=audit_row.chart_id)
        if audit_manifest == manifest:
            result["auto_annotation"] = detail
        return result

    def _source_feedback(self,payload):
        manifest=_required_manifest(payload)
        row = _catalog_row(manifest, payload.get("chart_id"))
        if row is None:raise ValueError('谱面不属于当前训练来源')
        identity = _feedback_chart({**payload,'path':row.source_path,'difficulty_index':row.difficulty_index})
        if identity["chart_sha256"] != row.chart_sha256:
            raise ValueError("谱面已变化，请重新打开")
        if payload.get('chart_action') is not None:
            return self._candidate_chart_feedback(payload, manifest, row)
        span=_feedback_span(payload)
        if 'start_sec' not in span:raise ValueError('请先在谱面时间轴圈选片段；整谱质量可在素材管理中设置')
        label=payload.get('label');reason=_feedback_reason(payload)
        if label not in {'good','problem','uncertain'}:raise ValueError('未知片段评价')
        from .candidate_api import _is_pool
        candidate = _is_pool(manifest)
        if candidate:
            if any(not payload.get(key) for key in ('file_sha256', 'chart_sha256', 'audio_sha256')):
                raise ValueError('请重新打开候选谱面与音频，再保存已试听的片段')
            _feedback_audio(dict(audio_path=row.audio_path, audio_sha256=payload['audio_sha256']))
            from maichart.training.candidate_pool import review_guard
            review_guard(manifest, row.chart_id, positive=label=='good', row=row)
            if label == 'good':
                from maichart.webui.chart_timeline import timeline
                notes = timeline(row.source_path, row.difficulty_index)['notes']
                if not any(note.get('has_head', True) and span['start_sec'] <= note['start_sec'] < span['end_sec'] for note in notes):
                    raise ValueError('这个范围没有新的音符头，请带上休止两端的音符再确认学习范围')
        old = _annotations(manifest).find(row) or {}
        if candidate or _is_zero_unmarked_manifest(manifest):
            old = {**old, 'unmarked_weight': 0.}
        region=dict(start_sec=span['start_sec'],end_sec=span['end_sec'],label=label,reason=reason,weight=1.)
        learn_entry = payload.get('learn_entry', False)
        if not isinstance(learn_entry, bool):
            raise ValueError('学习前文接入须为布尔值')
        if learn_entry:
            if label != 'good' or reason in {'layout', 'comfort'} or not row.audio_path:
                raise ValueError('确认接入需要对应音频，并选择学习采音')
            region.update(learn_entry=True, entry_start_sec=span['start_sec'])
        if payload.get('note'):
            region['note'] = str(payload['note'])[:500]
        if old.get('unmarked_weight')==0:
            from maichart.training.reviewed_generation import approved_regions
            from maichart.training.quality_annotations import zero_unmarked_region_weights
            approvals=[]
            prior_regions = [] if candidate and (old.get('label') in {'excluded','negative'} or old.get('weight',1.)<=0) else old.get('regions',[])
            for previous in prior_regions:
                timing,components=zero_unmarked_region_weights(previous)
                if timing or components:
                    approvals.append({**previous,'scope':'region','label':'good','training_use':'both' if timing and components else 'timing' if timing else 'layout'})
            action={**region,'scope':'region'}
            if label=='good':action['training_use']='timing' if reason=='timing' else 'layout' if reason in {'layout','comfort'} else 'both'
            approvals.append(action)
            return self.save(dict(manifest=str(manifest),chart_ids=[row.chart_id],regions=approved_regions(approvals),
                                  label='normal' if candidate and label=='good' and old.get('label') in {'excluded','negative'} else old.get('label','normal'),
                                  weight=(old.get('weight',1.) or 1.) if candidate and label=='good' else old.get('weight',1.),
                                  candidate_review=region if candidate else None))
        if label=='good' and reason:
            region.update(timing_weight=1.5 if reason=='timing' else 1.,component_weight=1. if reason=='timing' else 1.5)
        regions=[]
        for previous in old.get('regions',[]):
            a,b=previous['start_sec'],previous['end_sec']
            if b<=region['start_sec'] or a>=region['end_sec']:regions.append(previous);continue
            if a<region['start_sec']:regions.append({**previous,'end_sec':region['start_sec']})
            if b>region['end_sec']:regions.append({**previous,'start_sec':region['end_sec']})
        regions.append(region);regions.sort(key=lambda item:item['start_sec'])
        return self.save(dict(manifest=str(manifest),chart_ids=[row.chart_id],regions=regions,
                              label=old.get('label','normal'),weight=old.get('weight',1.),
                              annotation_scope='region'))

    def _candidate_chart_feedback(self, payload, manifest, row):
        """A whole-chart decision is independent of the currently selected span."""
        from .candidate_api import _is_pool
        from maichart.training.candidate_pool import review_guard
        action = payload['chart_action']
        if not _is_pool(manifest) or action not in {'learn_all', 'exclude'}:
            raise ValueError('整谱审阅操作只适用于候选库')
        review_guard(manifest, row.chart_id, positive=action=='learn_all', row=row)
        old = find_annotation(row, load_annotations(manifest)['records']) or {}
        regions = []
        if action == 'learn_all':
            if any(not payload.get(key) for key in ('file_sha256','chart_sha256','audio_sha256')):
                raise ValueError('请重新打开谱面和音频后确认整谱学习')
            _feedback_audio(dict(audio_path=row.audio_path, audio_sha256=payload['audio_sha256']))
            from maichart.webui.chart_timeline import timeline
            notes = timeline(row.source_path, row.difficulty_index)['notes']
            if not notes:
                raise ValueError('这张谱面没有可学习的音符')
            end = max(float(n.get('end_sec', n['start_sec'])) for n in notes) + .25
            regions = [dict(start_sec=0., end_sec=end, label='good', reason='', weight=1.,
                            timing_weight=1., component_weight=1., learn_entry=True, entry_start_sec=0.)]
        return self.save(dict(manifest=str(manifest), chart_ids=[row.chart_id], regions=regions,
                              label='normal' if action=='learn_all' else 'excluded',
                              weight=(old.get('weight',1.) or 1.) if action=='learn_all' else 0.,
                              candidate_chart_review=action, reason=str(payload.get('note',''))[:500]))

    def candidate_feedback(self, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        """Compare real complete versions; save evidence outside training labels."""
        import hashlib
        from datetime import datetime, timezone
        candidates = payload.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != 2:
            raise ValueError("candidates 必须包含 A、B 两个谱面")
        preferred = payload.get("preferred")
        if preferred not in {"a", "b", "both_good", "both_bad", "uncertain"}:
            raise ValueError("未知候选反馈")
        reason = _feedback_reason(payload)
        identities = [_feedback_chart(candidate) for candidate in candidates]
        if (identities[0]['path'],identities[0]['difficulty_index']) == (identities[1]['path'],identities[1]['difficulty_index']):
            raise ValueError("请选择两个不同的完整谱面版本")
        audio_identity = _feedback_audio(payload)
        candidate_audio = [_feedback_audio(candidate) for candidate in candidates]
        audio_hashes = {item['audio_sha256'] for item in [audio_identity,*candidate_audio] if item}
        if len(audio_hashes)>1:
            raise ValueError("A、B 必须对应同一份音频")
        if not audio_identity:
            audio_identity=next((item for item in candidate_audio if item),{})
        for identity,audio in zip(identities,candidate_audio):identity.update(audio)
        span = _feedback_span(payload)
        identity = dict(candidates=identities, **span, **audio_identity)
        pair_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        record = dict(**identity, pair_id=pair_id, client_pair_id=payload.get("pair_id"),
                      scope="region" if 'start_sec' in span else "whole_chart",
                      preferred=preferred, reason=reason,
                      created_at=datetime.now(timezone.utc).isoformat())
        path = Path(identities[0]["path"]).with_name("candidate_feedback.json")
        _append_feedback(path,"composer-v2-candidate-feedback-v1",record)
        return HTTPStatus.OK, {"feedback_path": str(path), "record": record}

    def generation_feedback(self, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        """Mark one complete output, optionally within a region, for later review."""
        import hashlib
        from datetime import datetime, timezone
        label=payload.get('label')
        if label not in {'good','problem','uncertain'}:
            raise ValueError('未知谱面反馈')
        identity=dict(**_feedback_chart(payload),**_feedback_audio(payload),**_feedback_span(payload))
        feedback_id=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        record=dict(**identity,feedback_id=feedback_id,label=label,reason=_feedback_reason(payload),
                    scope='region' if 'start_sec' in identity else 'whole_chart',
                    created_at=datetime.now(timezone.utc).isoformat())
        if 'training_use' in payload:
            if label != 'good' or 'start_sec' not in identity or payload['training_use'] not in {'none','timing','layout','both'}:
                raise ValueError('训练示范需要圈选片段，并明确选择学习内容')
            record['training_use'] = payload['training_use']
        learn_entry=payload.get('learn_entry',False)
        if not isinstance(learn_entry,bool):
            raise ValueError('学习前文接入须为布尔值')
        if learn_entry:
            if label!='good' or record.get('training_use') not in {'timing','both'} or record['scope']!='region':
                raise ValueError('学习前文接入需要收集好片段，并选择只学采音或采音和配置都学')
            if not identity.get('audio_sha256'):
                raise ValueError('确认前文接入需要对应音频，请重新载入谱面与音频')
            record.update(learn_entry=True,entry_start_sec=identity['start_sec'])
        elif 'training_use' in payload:
            record['learn_entry']=False
        path=Path(identity['path']).with_name('generation_feedback.json')
        _append_feedback(path,'maichart-generation-feedback-v1',record)
        return HTTPStatus.OK, {"feedback_path": str(path), "record": record}

    def get_audio(self, query: dict[str, Any], *, method: str = "GET", range_header: str | None = None) -> tuple[int, dict[str, str], Path | None, tuple[int, int] | None]:
        manifest = _required_manifest(query)
        chart_id = _first(query.get("chart_id"))
        row = _catalog_row(manifest, chart_id)
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
        query = str(payload.get("query") or payload.get("search") or "").strip().casefold()
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
        candidate_filter=payload.get('candidate_status') or ''
        review_filter=payload.get('candidate_review') or ''
        auto_filter = str(payload.get('auto_filter') or '').strip()
        if review_filter not in {'','unreviewed','partial','learn_all','exclude'}:
            raise ValueError('未知审阅决定')
        if candidate_filter not in ('','pending','error','suspicious','ok','stale'):
            raise ValueError('未知自动检查状态')
        if auto_filter not in {'', 'flagged', 'excluded', 'problem', 'good', 'clean', 'unverified',
                               'pending_review', 'reviewed_prompts', 'coverage_issue'}:
            raise ValueError('未知自动体检筛选')
        view = self._catalog_view(manifest, _auto_scope(payload, allow_auto=True), force=payload.get("refresh") is True)
        descending = sort_order == "desc"
        sort_key = (sort_by, descending)
        filter_key = (query, label_filter, candidate_filter, review_filter, auto_filter, sort_by, descending)
        with _CATALOG_LOCK:
            sorted_idx = (view.sorted_indices or {}).get(sort_key)
            cached_rows = (view.filter_cache or {}).get(filter_key)
            if cached_rows is not None and view.filter_cache is not None:
                view.filter_cache.move_to_end(filter_key)
        if cached_rows is not None:
            rows = cached_rows
        else:
            if sorted_idx is None:
                indexed = [dict(row, _view_idx=idx) for idx, (_, row) in enumerate(view.rows)]
                _sort_catalog_rows(indexed, sort_by, descending)
                sorted_idx = tuple(item["_view_idx"] for item in indexed)
                with _CATALOG_LOCK:
                    if view.sorted_indices is not None:
                        view.sorted_indices[sort_key] = sorted_idx
            matched_rows = []
            view_rows = view.rows
            for idx in sorted_idx:
                haystack, value = view_rows[idx]
                if query and query not in haystack:
                    continue
                if label_filter and value["label"] != label_filter:
                    continue
                if candidate_filter and value.get('candidate_status') != candidate_filter:
                    continue
                if review_filter and value.get('candidate_review_status') != review_filter:
                    continue
                if auto_filter:
                    st = value.get('auto_status', 'unreviewed')
                    if auto_filter == 'flagged' and st not in {'excluded', 'masked', 'downweighted'}:
                        continue
                    if auto_filter == 'excluded' and st != 'excluded':
                        continue
                    if auto_filter == 'problem' and st not in {'masked', 'downweighted'}:
                        continue
                    if auto_filter == 'good' and int(value.get('auto_good_count') or 0) <= 0:
                        continue
                    if auto_filter == 'clean' and st != 'clean':
                        continue
                    if auto_filter == 'unverified' and st != 'unverified' and not value.get('auto_review_count'):
                        continue
                    if auto_filter == 'pending_review' and not value.get('auto_review_count'):
                        continue
                    if auto_filter == 'reviewed_prompts' and not value.get('auto_review_resolved_count'):
                        continue
                    if auto_filter == 'coverage_issue' and not value.get('auto_unverified_reason'):
                        continue
                matched_rows.append(value)
            rows = tuple(matched_rows)
            with _CATALOG_LOCK:
                if view.filter_cache is not None:
                    view.filter_cache[filter_key] = rows
                    view.filter_cache.move_to_end(filter_key)
                    while len(view.filter_cache) > 32:
                        view.filter_cache.popitem(last=False)
        total = len(rows)
        pages = max(1, (total + page_size - 1) // page_size)
        page = min(page, pages)
        start = (page - 1) * page_size
        items = list(rows[start:start + page_size])
        if items and "candidate_status" in items[0]:
            from maichart.training.candidate_pool import _resolved_slot, scan_issues
            indexed = scan_issues(manifest)
            items = [dict(value, candidate_issues=(
                scanned[2] if (scanned := indexed.get(_resolved_slot(
                    value["source_path"], value["difficulty_index"]))) is not None
                and value["candidate_status"] not in {"stale", "pending"} else []
            )) for value in items]
        return HTTPStatus.OK, {
            "schema": "maichart-quality-annotations-v1", "manifest": str(manifest), "annotation_path": str(annotation_path(manifest)),
            "page": page, "pages": pages, "page_size": page_size, "total": total,
            "folder_total": len(view.chart_catalog.by_id),
            "sort_by": sort_by, "sort_order": sort_order,
            "items": items, "rows": items, "summary": view.summary,
            "auto_quality": dict(view.audit, formal_task=audit_state(manifest, False), preview_task=audit_state(manifest, True)),
        }

    def selection(self, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        """Return every chart identity in this source, without page/filter caps."""
        manifest = _required_manifest(payload)
        chart_ids = list(dict.fromkeys(row.chart_id for row in _catalog(manifest)))
        return HTTPStatus.OK, {"manifest": str(manifest), "chart_ids": chart_ids,
                               "total": len(chart_ids), "scope": "current_dataset"}

    def save(self, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        manifest = _required_manifest(payload)
        self._require_editable(manifest)
        with _material_write(manifest):
            return self._save(payload)

    def _save(self, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        manifest = _required_manifest(payload)
        from .candidate_api import _is_pool
        candidate = _is_pool(manifest)
        zero_manifest = candidate or _is_zero_unmarked_manifest(manifest)
        chart_ids = payload.get("chart_ids")
        if isinstance(chart_ids, str):
            chart_ids = [chart_ids]
        if not isinstance(chart_ids, list) or not chart_ids or any(not isinstance(value, str) or not value for value in chart_ids):
            raise ValueError("chart_ids 不能为空")
        label = str(payload.get("label", "normal"))
        if label not in LABELS:
            raise ValueError("未知质量标签")
        weight = float(payload.get("weight", DEFAULT_WEIGHTS[label]))
        if not 0 <= weight <= 4 or weight != weight or weight in (float("inf"), float("-inf")):
            raise ValueError("weight 必须是 0–4 的有限数字")
        scope = payload.get("annotation_scope", "chart")
        if scope not in ("chart", "region"):
            raise ValueError("未知人工标注范围")
        chart_ids = list(dict.fromkeys(chart_ids))
        by_id = {chart_id: _catalog_row(manifest, chart_id) for chart_id in chart_ids}
        missing = [item for item in chart_ids if by_id[item] is None]
        if missing:
            raise ValueError("存在不属于当前 manifest 的 chart_id")
        current = load_annotations(manifest)["records"]
        lookup = _AnnotationLookup(current, _catalog(manifest))
        audio_hashes: dict[str, str] = {}
        decisions = []
        bulk_mode = len(chart_ids) > 1 or payload.get("bulk_mode") is True
        for chart_id in chart_ids:
            row = by_id[chart_id]
            if chart_sha256(row.source_path, row.difficulty_index) != row.chart_sha256:
                raise ValueError("谱面已变化，请重新载入后保存标注")
            duration = chart_duration(row.source_path, row.difficulty_index)
            if candidate:
                from maichart.webui.chart_timeline import timeline
                notes = timeline(row.source_path, row.difficulty_index)['notes']
                duration = max(float(duration or 0.), max((float(n.get('end_sec', n['start_sec'])) for n in notes), default=0.) + .25)
            matched_previous = lookup.find(row)
            previous = dict(matched_previous) if matched_previous is not None else {}
            zero_unmarked = zero_manifest or previous.get("unmarked_weight") == 0
            if zero_unmarked:
                previous["unmarked_weight"] = 0.
            raw_regions = payload.get("regions", previous.get("regions", []))
            regions = _sanitize_zero_unmarked_regions(raw_regions) if zero_unmarked else raw_regions
            if candidate:
                from maichart.training.candidate_pool import review_guard
                from maichart.training.quality_annotations import zero_unmarked_region_weights
                positive = isinstance(regions, list) and any(
                    isinstance(r, dict) and any(zero_unmarked_region_weights(r)) for r in regions
                )
                review_guard(manifest, chart_id, positive=positive, row=row)
            # Validate against exact chart duration before writing anything.
            chart_label, chart_weight = label, weight
            if not zero_unmarked and scope == "region":
                chart_label = previous.get("label", "normal")
                chart_weight = previous.get("weight", 1.0)
                if matched_previous is None:
                    previous["chart_override"] = False
            elif not zero_unmarked and previous.get("chart_override") is False:
                previous["chart_override"] = True
            record = validate_record({**previous, "chart_id": chart_id,
                                      "regions": regions, "label": chart_label, "weight": chart_weight})
            if candidate and 'regions' in payload:
                record.pop('candidate_chart_review', None)
            if candidate and payload.get('candidate_review'):
                record.pop('candidate_chart_review', None)
                record['candidate_reviews'] = [*previous.get('candidate_reviews', []), payload['candidate_review']]
            if candidate and payload.get('candidate_chart_review') in {'learn_all','exclude'}:
                record['candidate_chart_review'] = payload['candidate_chart_review']
                record['candidate_reviews'] = []
            if duration is not None:
                from maichart.training.quality_annotations import _validate_regions
                record["regions"] = _validate_regions(regions, duration)
            record.update({"song_id": row.song_id, "difficulty_index": row.difficulty_index, "source_path": row.source_path,
                           "audio_path": row.audio_path, "chart_sha256": row.chart_sha256,
                           "source_dataset": row.source_dataset,
                           "source_dataset_id": row.source_dataset_id, "folder_name": row.folder_name,
                           "maidata_title": row.maidata_title,
                           "reason": str(previous.get("reason", "") if scope == "region"
                                         else payload.get("reason", previous.get("reason", "")))})
            if row.audio_path and Path(row.audio_path).is_file():
                # Hash only explicitly selected charts; catalog browsing stays
                # cheap even for a large corpus of long audio files.
                if row.audio_path not in audio_hashes:
                    audio_hashes[row.audio_path] = audio_sha256(row.audio_path)
                record["audio_sha256"] = audio_hashes[row.audio_path]
            corrected = payload.get("corrected_path")
            if chart_label != "negative" and "corrected_path" not in payload:
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
            lookup.replace_for_row(row, chart_id, record)
            if not zero_unmarked and scope == "chart":
                decisions.append((self._review_source(manifest, row), decision_signature(record),
                                  "bulk_chart" if bulk_mode else "manual_chart"))
        saved_records = lookup.active_records()
        path = annotation_path(manifest)
        output = path if path.is_file() and saved_records == current else save_annotations(manifest, saved_records)
        if decisions:
            try:
                self.reviews.save_charts(decisions)
            except (OSError, sqlite3.Error) as exc:
                raise ValueError(f"训练标注已保存，来源记录保存失败，请重试。{exc}") from exc
        current_rows = _catalog(manifest)
        chart_history = self.reviews.read()[0] if not zero_manifest else {}
        values = []
        for row in current_rows:
            record = lookup.find(row)
            values.append(dict(row.to_dict(record), manual_provenance=chart_provenance(
                self._review_source(manifest, row), record, chart_history,
            )))
        summary = self._summary(values, saved_records)
        return HTTPStatus.OK, {"saved": len(chart_ids), "annotation_path": str(output),
                               "records": saved_records, "summary": summary}

    def detail(self, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        manifest = _required_manifest(payload)
        chart_id = _first(payload.get("chart_id"))
        row = _catalog_row(manifest, chart_id)
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
        lookup = _annotations(manifest)
        annotation = find_annotation(row, [lookup.records[index] for index in lookup._candidate_indices(row)],
                                     verify_audio=True, resolve_path=lookup._resolve)
        auto_detail = (self._chart_review_context(manifest, row, _auto_scope(payload))[0]
                       if not _is_zero_unmarked_manifest(manifest) else None)
        duration = chart_duration(row.source_path, row.difficulty_index)
        return HTTPStatus.OK, {**row.to_dict(annotation), "chart": row.to_dict(annotation),
            "record": annotation, "auto_annotation": auto_detail, "maidata": chart.raw or "",
            "zero_unmarked": bool(_is_zero_unmarked_manifest(manifest) or (annotation or {}).get("unmarked_weight") == 0),
            "preview": {"events": points, "duration_sec": duration},
            "slot": {"index": row.difficulty_index,
            "inote": difficulty.inote or "", "events": points, "duration_sec": duration},
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
        matched = [row["annotation"] for row in rows if row.get("annotation") is not None]
        result = {
            "total": len(rows), "annotated": len(matched), "counts": dict(counts),
            "unmatched_records": len(records) - len({id(record) for record in matched}),
            "preference_pairs": sum(1 for record in matched
                                     if record.get("label") == "negative" and record.get("corrected_path")),
            "negative_without_correction": sum(1 for record in matched
                                                if record.get("label") == "negative" and not record.get("corrected_path")),
        }
        for name in ("manual_chart", "bulk_chart", "legacy"):
            result[name + "_charts"] = sum(row.get("manual_provenance") == name for row in rows)
        result["region_charts"] = sum(bool(record.get("regions")) for record in matched)
        result["entering_training_charts"] = sum(
            (row.get("training_effect") or {}).get("status") in {"included", "boosted", "masked"}
            for row in rows
        )
        result["expert_opt_in_charts"] = sum(
            (row.get("training_effect") or {}).get("status") == "expert_opt_in"
            for row in rows
        )
        return result

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


def _feedback_reason(payload: dict[str, Any]) -> str:
    reason=str(payload.get('reason',''))
    if reason not in {'','timing','layout','comfort'}:
        raise ValueError('未知反馈原因')
    return reason


def _feedback_chart(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value,dict) or not value.get('path'):
        raise ValueError('反馈缺少谱面文件')
    path=Path(value['path']).resolve()
    if not path.is_file() or path.suffix.lower() not in {'.txt','.maidata'}:
        raise ValueError('反馈谱面文件不存在或格式不支持')
    if value.get('partial_preview') or 'phrase_candidates' in {part.casefold() for part in path.parts}:
        raise ValueError('试写片段不能作为完整谱面反馈')
    if value.get('content_kind','complete_chart')!='complete_chart':
        raise ValueError('请选择完整谱面')
    before=path.stat()
    raw_index=value.get('difficulty_index',5)
    try:index=int(raw_index)
    except (TypeError,ValueError):raise ValueError('difficulty_index 必须是整数')
    if isinstance(raw_index,bool) or str(index)!=str(raw_index):
        raise ValueError('difficulty_index 必须是整数')
    chart=parse_maidata_file(path)
    difficulty=next((item for item in chart.difficulties if item.index==index),None)
    if difficulty is None or not (difficulty.inote or '').strip():
        raise ValueError('反馈谱面缺少指定难度槽位')
    identity=dict(path=str(path),difficulty_index=index,chart_sha256=chart_sha256(path,index),
                  file_sha256=audio_sha256(path))
    after=path.stat()
    if (before.st_mtime_ns,before.st_size)!=(after.st_mtime_ns,after.st_size):
        raise ValueError('谱面文件已改变，请重新载入后反馈')
    for key in ('chart_sha256','file_sha256'):
        if value.get(key) and value[key]!=identity[key]:
            raise ValueError('谱面文件已改变，请重新载入后反馈')
    if value.get('version_id'):
        version_id=value['version_id']
        if not isinstance(version_id,str) or len(version_id)>128:
            raise ValueError('version_id 无效')
        identity['version_id']=version_id
    return identity


def _feedback_audio(value: dict[str, Any]) -> dict[str, Any]:
    if not value.get('audio_path'):
        if value.get('audio_sha256'):raise ValueError('音频身份缺少 audio_path')
        return {}
    path=Path(value['audio_path']).resolve()
    if not path.is_file():raise ValueError('原任务音频不存在，请重新载入后反馈')
    before=path.stat()
    digest=audio_sha256(path)
    after=path.stat()
    if (before.st_mtime_ns,before.st_size)!=(after.st_mtime_ns,after.st_size):
        raise ValueError('音频文件已改变，请重新载入后反馈')
    if value.get('audio_sha256') and value['audio_sha256']!=digest:
        raise ValueError('音频文件已改变，请重新载入后反馈')
    return dict(audio_path=str(path),audio_sha256=digest)


def _feedback_span(payload: dict[str, Any]) -> dict[str, float]:
    from maichart.training.quality_annotations import _finite
    span={}
    for left,right in [('start_sec','end_sec'),('context_start_sec','context_end_sec')]:
        if left not in payload and right not in payload:continue
        if left not in payload or right not in payload:
            raise ValueError(f'{left}、{right} 必须一起提供')
        if isinstance(payload[left],bool) or isinstance(payload[right],bool):
            raise ValueError('反馈区间必须是有限数字')
        start,end=_finite(payload[left],left),_finite(payload[right],right)
        if not 0<=start<end:raise ValueError(f'反馈区间必须满足 0 <= {left} < {right}')
        span.update({left:start,right:end})
    if 'context_start_sec' in span:
        if 'start_sec' not in span or not (span['context_start_sec']<=span['start_sec']<span['end_sec']<=span['context_end_sec']):
            raise ValueError('上下文播放区间必须包含标记区间')
    return span


def _append_feedback(path: Path, schema: str, record: dict[str, Any]) -> None:
    """Publish one atomic sidecar update without touching dataset annotations."""
    import os
    import tempfile
    from maichart.training.quality_annotations import _lock_for
    with _lock_for(path):
        data=json.loads(path.read_text(encoding='utf-8')) if path.exists() else dict(schema=schema,records=[])
        if data.get('schema')!=schema or not isinstance(data.get('records'),list):
            raise ValueError('反馈文件 schema 不兼容')
        data['records'].append(record)
        fd,temporary=tempfile.mkstemp(prefix='.feedback-',suffix='.tmp',dir=path.parent)
        try:
            with os.fdopen(fd,'w',encoding='utf-8') as handle:
                json.dump(data,handle,ensure_ascii=False,indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary,path)
        finally:
            if os.path.exists(temporary):os.unlink(temporary)


def _required_manifest(payload: dict[str, Any]) -> Path:
    value = _first(payload.get("manifest"))
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

