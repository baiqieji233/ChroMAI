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

    def __init__(self, workspace_root=None):
        self.workspace_root=Path(workspace_root) if workspace_root else Path(__file__).resolve().parents[3]/'webui_workspace'

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
        row=next((row for row in _catalog(manifest) if row.chart_id==payload.get('chart_id')),None)
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
                     version_id=digest[:24],content_kind='complete_chart',url=_resource(view),timeline=notes,
                     audio_url=_resource(Path(row.audio_path)) if row.audio_path else None,
                     audio_name=Path(row.audio_path).name if row.audio_path else None)
        annotation=find_annotation(row,load_annotations(manifest)['records']) or {}
        return HTTPStatus.OK,dict(task_id='source-'+row.chart_id,revision=identity['file_sha256'],annotations=annotation.get('regions',[]),
            source_kind='training_source',training_context=dict(manifest=str(manifest),chart_id=row.chart_id),charts=[version])

    def source_feedback(self,payload):
        manifest=_required_manifest(payload)
        row=next((row for row in _catalog(manifest) if row.chart_id==payload.get('chart_id')),None)
        if row is None:raise ValueError('谱面不属于当前训练来源')
        _feedback_chart({**payload,'path':row.source_path,'difficulty_index':row.difficulty_index})
        span=_feedback_span(payload)
        if 'start_sec' not in span:raise ValueError('请先在谱面时间轴圈选片段；整谱质量可在素材管理中设置')
        label=payload.get('label');reason=_feedback_reason(payload)
        if label not in {'good','problem','uncertain'}:raise ValueError('未知片段评价')
        old=find_annotation(row,load_annotations(manifest)['records']) or {}
        region=dict(start_sec=span['start_sec'],end_sec=span['end_sec'],label=label,reason=reason,weight=1.)
        if old.get('unmarked_weight')==0:
            from maichart.training.reviewed_generation import approved_regions
            from maichart.training.quality_annotations import region_component_weights
            approvals=[]
            for previous in old.get('regions',[]):
                timing,components=region_component_weights(previous)
                if timing or components:
                    approvals.append({**previous,'scope':'region','label':'good','training_use':'both' if timing and components else 'timing' if timing else 'layout'})
            action={**region,'scope':'region'}
            if label=='good':action['training_use']='timing' if reason=='timing' else 'layout' if reason in {'layout','comfort'} else 'both'
            approvals.append(action)
            return self.save(dict(manifest=str(manifest),chart_ids=[row.chart_id],regions=approved_regions(approvals),
                                  label=old.get('label','normal'),weight=old.get('weight',1.)))
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
                              label=old.get('label','normal'),weight=old.get('weight',1.)))

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

