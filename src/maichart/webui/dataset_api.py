"""Manage dataset directories and prepare selected sources for the WebUI."""
from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from http import HTTPStatus
from pathlib import Path
from typing import Any

from maichart.training.dataset_catalog import catalog_datasets, compose_datasets

_REGISTRY_LOCK = threading.Lock()
_MANIFEST_NAMES = ("pretrain_manifest.json", "training_manifest.json")


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
        if manifest.is_file():
            try:
                card.update(catalog_datasets([manifest])[0])
                card.update({"path": str(root), "raw_root": str(root), "manifest": str(manifest),
                             "name": str(entry.get("name") or root.name), "prepared": True})
                data = json.loads(manifest.read_text(encoding="utf-8-sig"))
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
        values = payload.get("manifests") or []
        if not isinstance(values, list):
            raise ValueError("manifests 必须是数组")
        for value in values:
            if value:
                path = self._resolve(value)
                if path.is_file():
                    data = json.loads(path.read_text(encoding="utf-8-sig"))
                    if data.get("schema") != "maichart-training-manifest-v1" or data.get("merged_from"):
                        continue
                    root = Path(str(data.get("source_root") or "."))
                    root = (root if root.is_absolute() else path.parent / root).resolve()
                    entries.append({"path": str(root), "manifest": str(path)})
        if current is not None and current.is_file():
            try:
                data = json.loads(current.read_text(encoding="utf-8-sig"))
                if not data.get("merged_from") and data.get("schema") == "maichart-training-manifest-v1":
                    source = Path(str(data.get("source_root") or "."))
                    source = (source if source.is_absolute() else current.parent / source).resolve()
                    current_root = source
                    entries.append({"path": str(source), "manifest": str(current)})
            except (OSError, ValueError, TypeError):
                pass
        unique: dict[str, dict[str, Any]] = {}
        for entry in entries:
            path = self._resolve(entry["path"])
            if path.is_dir():
                key = str(path).casefold()
                if key not in unique or entry.get("manifest"):
                    unique[key] = {**unique.get(key, {}), **entry}
        cards = [self._card({**entry, "manifest": str(current)} if current_root == self._resolve(entry["path"]) else entry, current)
                 for entry in unique.values()]
        cards.sort(key=lambda row: (not row.get("current"), str(row["name"]).casefold(), row["path"].casefold()))
        return HTTPStatus.OK, {"datasets": cards, "items": cards, "total": len(cards),
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
        if manifest:
            entry["manifest"] = str(manifest)
        with _REGISTRY_LOCK:
            entries = [row for row in self._registry() if self._resolve(row["path"]) != root]
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

    def compose(self, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        manifests = self._manifests(payload)
        if not manifests:
            raise ValueError("请至少勾选一个训练数据集")
        family = payload.get("model_family")
        # song_form_v1 consumes the same event-preprocessed source contract;
        # composition must validate it as event audio, never as legacy frames.
        if family in {"song_form_v1", "chart_stream_v1", "chart_composer_v1"}:
            family = "event_v1"
        result = compose_datasets(manifests, self.workspace_root / "composed_manifests", family)
        return HTTPStatus.OK, result

    def prepare(self, manifests: list[str], model_family: str = "event_v1") -> dict[str, Any]:
        return self.compose({"manifests": manifests, "model_family": model_family})[1]
