"""Manage dataset directories and prepare selected sources for the WebUI."""
from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from collections import OrderedDict
from http import HTTPStatus
from pathlib import Path
from typing import Any

from maichart.training.dataset_catalog import catalog_datasets, compose_datasets, manifest_dataset_id

_REGISTRY_LOCK = threading.Lock()
_MANIFEST_NAMES = ("pretrain_manifest.json", "training_manifest.json")
_JSON_CACHE = OrderedDict()
_JSON_LOCK = threading.RLock()


def _json_data(path):
    """Cache immutable JSON snapshots; never parse or hash source charts here."""
    path = Path(path)
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size, stat.st_ctime_ns)
    with _JSON_LOCK:
        if key in _JSON_CACHE:
            _JSON_CACHE.move_to_end(key)
            return _JSON_CACHE[key][0]
    value = json.loads(path.read_text(encoding='utf-8-sig'))
    with _JSON_LOCK:
        for old in list(_JSON_CACHE):
            if old[0] == key[0]:_JSON_CACHE.pop(old)
        if stat.st_size <= 16*1024*1024:_JSON_CACHE[key] = (value,stat.st_size)
        while len(_JSON_CACHE)>16 or sum(item[1] for item in _JSON_CACHE.values())>16*1024*1024:
            _JSON_CACHE.popitem(last=False)
    return value


def _display_root(manifest, data):
    if data.get('candidate_pool') or data.get('reviewed_training'):
        return manifest.parent
    value = Path(str(data.get('source_root') or '.'))
    return (value if value.is_absolute() else manifest.parent / value).resolve()


def _pool_card(root, data):
    """Last-scan counts, not a full identity/geometry audit during page loading."""
    from .material_audit_status import task_state

    metadata = _json_data(root / 'pool.json') if (root / 'pool.json').is_file() else {}
    state = task_state(_json_data(root / 'scan_state.json')) if (root / 'scan_state.json').is_file() else {}
    annotations = _json_data(root / 'quality_annotations.json') if (root / 'quality_annotations.json').is_file() else {}
    counts = dict(charts=0,error=0,suspicious=0,ok=0,pending=0,duplicate=len(metadata.get('excluded',[])))
    by_source = {(str(record.get('source_path','')).casefold(),record.get('difficulty_index')):record for record in annotations.get('records',[])}
    regions = 0
    decisions = dict(learn_all=0, partial=0, exclude=0)
    from maichart.training.candidate_pool import _stat, _valid_regions

    for song in data.get('songs',[]):
        for difficulty in song.get('difficulties',[]):
            counts['charts'] += 1
            status = difficulty.get('candidate_status','pending')
            counts[status] = counts.get(status,0)+1
            index=difficulty.get('difficulty_index',difficulty.get('index'))
            record=by_source.get((str(song.get('maidata_path','')).casefold(),index),{})
            identity=difficulty.get('candidate_identity',{})
            if record.get('chart_sha256')==identity.get('chart_sha256') and record.get('audio_sha256')==identity.get('audio_sha256'):
                confirmed = len(_valid_regions(record))
                regions += confirmed
                if record.get("label") in {"excluded", "negative"} or record.get("weight", 1.) <= 0:
                    decisions["exclude"] += 1
                elif record.get("candidate_chart_review") == "learn_all":
                    decisions["learn_all"] += 1
                elif confirmed:
                    decisions["partial"] += 1
    stale = bool(metadata.get("scan_complete") and metadata.get("source_manifest")
                 and metadata.get("source_manifest_stat") != _stat(metadata["source_manifest"]))
    return dict(pool_id=data['candidate_pool']['pool_id'],counts=counts,confirmed_regions=regions,
                scan_complete=bool(metadata.get('scan_complete')),scan_state=state.get('status'),scan_task=state,
                decisions=decisions, stale=stale, updated_at=state.get("updated_at") or metadata.get("updated_at"),
                source_manifest=metadata.get("source_manifest"),
                reviewable_charts=counts['ok']+counts['suspicious'] if metadata.get('scan_complete') else 0,
                summary_only=True)


class DatasetAPI:
    def __init__(self, project_root: Path, dataset_root: Path, workspace_root: Path) -> None:
        self.project_root = Path(project_root).resolve()
        self.dataset_root = Path(dataset_root).resolve()
        self.workspace_root = Path(workspace_root).resolve()
        self.registry_path = self.workspace_root / "dataset_registry.json"

    def post(self, endpoint: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        if endpoint in {"/api/datasets/list", "/api/datasets/catalog"}:
            return self.catalog(payload)
        if endpoint == "/api/datasets/register":
            return self.register(payload)
        if endpoint in {"/api/datasets/compose", "/api/datasets/prepare"}:
            return self.compose(payload)
        return HTTPStatus.NOT_FOUND, {"error": "Not found"}

    def _resolve(self, value: Any) -> Path:
        path = Path(str(value)).expanduser()
        return (path if path.is_absolute() else self.project_root / path).resolve()

    def _registry(self) -> list[dict[str, Any]]:
        if not self.registry_path.is_file():
            return []
        data = json.loads(self.registry_path.read_text(encoding="utf-8-sig"))
        return [row for row in data.get("datasets", []) if isinstance(row, dict) and row.get("path")]

    def _save_registry(self, entries: list[dict[str, Any]]) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(dir=self.registry_path.parent, prefix="dataset_registry_", suffix=".tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as output:
                json.dump({"schema": "maichart-dataset-registry-v1", "datasets": entries}, output, ensure_ascii=False, indent=2)
            os.replace(temporary, self.registry_path)
        finally:
            if Path(temporary).exists():
                Path(temporary).unlink()

    @staticmethod
    def _looks_like_dataset(root: Path) -> bool:
        return (any((root / name).is_file() for name in _MANIFEST_NAMES)
                or any(next(root.glob(pattern), None) is not None
                       for pattern in ("maidata.txt", "*/maidata.txt", "*/*/maidata.txt")))

    def _roots(self) -> list[dict[str, Any]]:
        entries = self._registry()
        if self.dataset_root.is_dir():
            for child in self.dataset_root.iterdir():
                if not child.is_dir():
                    continue
                upload_roots = [sub for sub in child.iterdir()
                                if sub.is_dir() and re.fullmatch(r"[a-f0-9]{8}", sub.name)]
                if upload_roots:
                    entries.extend({"path": str(root), "name": re.sub(r"^[a-f0-9]{12}_", "", child.name)}
                                   for root in upload_roots)
                elif self._looks_like_dataset(child):
                    entries.append({"path": str(child), "name": child.name})
        return entries

    def _card(self, entry: dict[str, Any], current: Path | None = None) -> dict[str, Any]:
        root = self._resolve(entry["path"])
        manifest = self._resolve(entry["manifest"]) if entry.get("manifest") else None
        if manifest is None or not manifest.is_file():
            manifest = next((root / name for name in _MANIFEST_NAMES if (root / name).is_file()), root / "training_manifest.json")
        card = {"path": str(root), "source_root": str(root), "raw_root": str(root),
                "manifest": str(manifest), "name": str(entry.get("name") or root.name),
                "cache_dir": str(root / "cache"), "prepared": False}
        if entry.get('candidate_pool'):
            card['candidate_pool'] = dict(entry['candidate_pool'])
            state = root / 'scan_state.json'
            if state.is_file():
                card['candidate_pool']['scan_state'] = json.loads(state.read_text(encoding='utf-8')).get('status')
        if entry.get('review_group'):card['review_group']=entry['review_group']
        if manifest.is_file():
            try:
                data = _json_data(manifest)
                if data.get('schema') != 'maichart-training-manifest-v1':
                    raise ValueError('这份文件不是可用的训练清单')
                songs=[song for song in data.get('songs',[]) if isinstance(song,dict)]
                difficulties=[d for song in songs for d in song.get('difficulties',[]) if isinstance(d,dict)]
                card.update(dataset_id=manifest_dataset_id(manifest,data),song_count=len(songs),
                    difficulty_count=len(difficulties),usable_difficulty_count=sum(bool(d.get('usable_for_training') or d.get('event_usable_for_training')) for d in difficulties),
                    annotation_path=str(manifest.with_name('quality_annotations.json')),
                    annotation_exists=manifest.with_name('quality_annotations.json').is_file(),
                    preprocessing=data.get('preprocessing') or {},merged=bool(data.get('merged_from')))
                card.update({"path": str(root), "raw_root": str(root), "manifest": str(manifest),
                             "name": str(entry.get("name") or root.name), "prepared": True})
                if data.get('candidate_pool'):
                    summary = _pool_card(manifest.parent,data)
                    card['candidate_pool'] = {**data['candidate_pool'], **summary}
                    card['candidate_pool_summary'] = summary
                if data.get('reviewed_training'):
                    card['reviewed_regions']=data['reviewed_training'].get('regions',0)
                    card['reviewed_only']=True
                if not (data.get("candidate_pool") or data.get("reviewed_training")):
                    from .material_audit_status import published_audits

                    try:
                        audit, *_ = published_audits(manifest, data)
                        card["audit_summary"] = audit
                    except (OSError, ValueError, TypeError) as exc:
                        card["audit_summary_error"] = str(exc)
                cache = Path(str(data.get("cache_dir") or "cache"))
                card["cache_dir"] = str((cache if cache.is_absolute() else manifest.parent / cache).resolve())
            except (OSError, ValueError, TypeError, KeyError) as exc:
                card["error"] = str(exc)
                card["prepared"] = False
        card["available_manifests"] = [str(root / name) for name in _MANIFEST_NAMES if (root / name).is_file()]
        card["current"] = current == manifest
        return card

    def catalog(self, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        current = self._resolve(payload["current_manifest"]) if payload.get("current_manifest") else None
        current_root = None
        entries = self._roots()
        warnings = []
        values = payload.get("manifests") or []
        if not isinstance(values, list):
            raise ValueError("manifests 必须是数组")
        for value in values:
            if value:
                path = self._resolve(value)
                if path.is_file():
                    try:
                        data = _json_data(path)
                    except (OSError,ValueError,TypeError) as exc:
                        warnings.append(dict(manifest=str(path),error=str(exc)))
                        continue
                    if data.get("schema") != "maichart-training-manifest-v1" or data.get("merged_from"):
                        continue
                    root = _display_root(path,data)
                    entries.append({"path": str(root), "manifest": str(path)})
        if current is not None and current.is_file():
            try:
                data = _json_data(current)
                if not data.get("merged_from") and data.get("schema") == "maichart-training-manifest-v1":
                    source = _display_root(current,data)
                    current_root = source
                    entries.append({"path": str(source), "manifest": str(current)})
            except (OSError, ValueError, TypeError):
                pass
        unique: dict[str, dict[str, Any]] = {}
        for entry in entries:
            path = self._resolve(entry["path"])
            key = str(path).casefold()
            if key not in unique or entry.get("manifest"):
                unique[key] = {**unique.get(key, {}), **entry}
        cards=[]
        for entry in unique.values():
            if not self._resolve(entry["path"]).is_dir():
                cards.append(dict(path=entry["path"], name=entry.get("name") or Path(entry["path"]).name,
                                  manifest=entry.get("manifest", ""), prepared=False, error="目录不存在"))
                continue
            try:
                cards.append(self._card({**entry,'manifest':str(current)} if current_root==self._resolve(entry['path']) else entry,current))
            except (OSError,ValueError,TypeError,KeyError) as exc:
                cards.append(dict(path=entry['path'],name=entry.get('name') or Path(entry['path']).name,
                                  manifest=entry.get('manifest',''),prepared=False,error=str(exc)))
        cards.sort(key=lambda row: (not row.get("current"), str(row["name"]).casefold(), row["path"].casefold()))
        return HTTPStatus.OK, {"datasets": cards, "items": cards, "total": len(cards),
                               "warnings": warnings,
                               "current_manifest": str(current) if current else ""}

    def register(self, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        value = payload.get("path") or payload.get("raw_root") or payload.get("manifest")
        if not value:
            raise ValueError("请填写已有训练素材目录")
        path = self._resolve(value)
        manifest = None
        if path.is_file():
            manifest = path
            root = path.parent
            catalog_datasets([path])
        elif path.is_dir():
            root = path
            if not self._looks_like_dataset(root) and next(root.rglob("maidata.txt"), None) is None:
                raise ValueError("这个目录没有找到 maidata.txt 或训练清单，请选择包含歌曲目录的训练素材根目录")
        else:
            raise ValueError(f"训练素材目录不存在：{path}")
        entry = {"path": str(root), "name": str(payload.get("name") or root.name)}
        group=payload.get('review_group')
        if group:
            if not manifest or not isinstance(group,str) or len(group)>256:
                raise ValueError('校准素材组需要有效的训练清单与任务标识')
            data=json.loads(manifest.read_text(encoding='utf-8-sig'))
            if data.get('reviewed_training',{}).get('policy')!='approved_regions_only_v1':
                raise ValueError('只有已确认片段素材可以更新校准素材组')
            entry['review_group']=group
        if manifest:
            entry["manifest"] = str(manifest)
        with _REGISTRY_LOCK:
            entries = [row for row in self._registry() if self._resolve(row["path"]) != root
                       and not (group and row.get('review_group')==group)]
            entries.append(entry)
            self._save_registry(entries)
        card = self._card(entry)
        return HTTPStatus.OK, {**card, "dataset": card,
                               "message": "已导入数据集" if card["prepared"] else "目录已导入，请先预处理后加入训练"}

    def _manifests(self, payload: dict[str, Any]) -> list[str]:
        values = payload.get("manifests")
        if values is None:
            value = payload.get("manifest")
            values = [value] if value else []
        if not isinstance(values, list) or any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError("manifests 必须是非空路径组成的数组")
        output = list(dict.fromkeys(str(self._resolve(value)) for value in values))
        for value in output:
            if not Path(value).is_file():
                raise ValueError(f"数据集尚未预处理：{Path(value).parent.name}，请先点击预处理训练集")
        return output

    def compose(self, payload: dict[str, Any], *, preview_root: Path | None = None,
                preview_audio_digest=None) -> tuple[int, dict[str, Any]]:
        manifests = self._manifests(payload)
        if not manifests:
            raise ValueError("请至少勾选一个训练数据集")
        family = payload.get("model_family")
        # song_form_v1 consumes the same event-preprocessed source contract;
        # composition must validate it as event audio, never as legacy frames.
        if family in {"song_form_v1", "chart_stream_v1", "chart_composer_v1", "chart_composer_v2"}:
            family = "event_v1"
        published = self.workspace_root / "composed_manifests"
        if preview_root is None:
            result = compose_datasets(manifests, published, family)
        else:
            result = compose_datasets(manifests, preview_root, family, published_root=published,
                                      preview_audio_digest=preview_audio_digest)
        return HTTPStatus.OK, result

    def prepare(self, manifests: list[str], model_family: str = "event_v1", *, preview_root: Path | None = None,
                preview_audio_digest=None) -> dict[str, Any]:
        payload = {"manifests": manifests, "model_family": model_family}
        if preview_root is None:
            return self.compose(payload)[1]
        return self.compose(payload, preview_root=preview_root, preview_audio_digest=preview_audio_digest)[1]
