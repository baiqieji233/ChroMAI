"""Human-friendly quality annotations for training manifests.

The annotation file is deliberately derived from a dataset manifest.  Users
edit records through the WebUI; they never need to maintain a second manifest.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from maichart.maidata import parse_maidata_file, resolve_chart_first
from maichart.timing import tokenize_chart_difficulty_timing

SCHEMA = "maichart-quality-annotations-v1"
AUTO_ANNOTATION_FILENAME = "auto_quality_annotations.json"
AUTO_ANNOTATION_PREVIEW_FILENAME = "auto_quality_annotations_preview.json"
COMPOSED_AUTO_ANNOTATION_FILENAME = "composed_auto_quality_annotations.json"
DEFAULT_WEIGHTS = {"normal": 1.0, "excellent": 2.0, "reduced": 0.25, "excluded": 0.0, "negative": 0.0}
LABELS = frozenset(DEFAULT_WEIGHTS)
_WRITE_LOCKS: dict[str, threading.Lock] = {}
_WRITE_LOCKS_GUARD = threading.Lock()


def annotation_path(manifest_path: str | Path) -> Path:
    """Return the automatically managed annotation path for a manifest."""
    return Path(manifest_path).resolve().with_name("quality_annotations.json")


def auto_annotation_path(
    manifest_path: str | Path, *, preview: bool = False, composed: bool = False
) -> Path:
    """Return the machine-generated audit annotation path (never overwrites manual annotations)."""
    if preview and composed:
        raise ValueError("组合继承标注不支持抽查模式")
    name = (COMPOSED_AUTO_ANNOTATION_FILENAME if composed else
            AUTO_ANNOTATION_PREVIEW_FILENAME if preview else AUTO_ANNOTATION_FILENAME)
    return Path(manifest_path).resolve().with_name(name)


def split_annotation_source(manifest: Path, payload: dict[str, Any]) -> Path | None:
    from maichart.portable_paths import material_path

    split = payload.get("split_metadata")
    origin = split.get("source_manifest") if isinstance(split, dict) else None
    return (manifest.parent / material_path(origin)).resolve() if origin else None


def _lock_for(path: Path) -> threading.Lock:
    key = str(path.resolve())
    with _WRITE_LOCKS_GUARD:
        return _WRITE_LOCKS.setdefault(key, threading.Lock())


def _finite(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是有限数字") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} 必须是有限数字")
    return number


def _resolve_manifest_path(value: Any, manifest_path: Path, source_root: Path | None = None) -> Path:
    from maichart.portable_paths import material_path

    raw = material_path(str(value))
    if raw.is_absolute():
        return raw.resolve()
    bases = [manifest_path.parent]
    if source_root is not None:
        bases.append(source_root)
    bases.append(manifest_path.parent.parent)
    for base in bases:
        candidate = (base / raw).resolve()
        if candidate.exists():
            return candidate
    return (bases[0] / raw).resolve()


def _source_root(manifest_path: Path, manifest: dict[str, Any]) -> Path:
    return _resolve_manifest_path(manifest.get("source_root", "."), manifest_path)


def _canonical_chart(maidata_path: Path, difficulty_index: int) -> tuple[str, Any]:
    chart = parse_maidata_file(maidata_path)
    difficulty = next((item for item in chart.difficulties if item.index == difficulty_index), None)
    if difficulty is None:
        raise ValueError(f"找不到难度槽位 {difficulty_index}")
    # Timing metadata is part of the chart identity, while other difficulty
    # blocks are intentionally excluded.
    canonical = "\n".join([
        f"slot={difficulty_index}",
        f"wholebpm={chart.wholebpm or ''}",
        f"first={resolve_chart_first(chart, difficulty_index) or ''}",
        f"level={difficulty.level or ''}",
        f"designer={difficulty.designer or ''}",
        (difficulty.inote or "").replace("\r\n", "\n").replace("\r", "\n").strip(),
    ])
    return canonical, chart


def chart_sha256(maidata_path: str | Path, difficulty_index: int) -> str:
    canonical, _ = _canonical_chart(Path(maidata_path), int(difficulty_index))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def hash_chart(chart_or_path: Any, difficulty_index: int | None = None) -> str:
    """Compatibility export for dataset/training code.

    ``chart_or_path`` may be a maidata path or a parsed RawMaidataChart.  The
    latter is intentionally hashed from one difficulty only, so adding another
    difficulty to the same file cannot invalidate unrelated annotations.
    """
    if isinstance(chart_or_path, (str, Path)):
        if difficulty_index is None:
            raise ValueError("difficulty_index is required for a path")
        return chart_sha256(chart_or_path, difficulty_index)
    if difficulty_index is None:
        raise ValueError("difficulty_index is required")
    difficulty = next((item for item in getattr(chart_or_path, "difficulties", []) if item.index == int(difficulty_index)), None)
    if difficulty is None:
        raise ValueError(f"找不到难度槽位 {difficulty_index}")
    canonical = "\n".join([f"slot={int(difficulty_index)}", f"wholebpm={getattr(chart_or_path, 'wholebpm', '') or ''}",
                             f"first={resolve_chart_first(chart_or_path, difficulty_index) or ''}", f"level={difficulty.level or ''}",
                             f"designer={difficulty.designer or ''}", (difficulty.inote or "").replace("\r\n", "\n").replace("\r", "\n").strip()])
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def audio_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def chart_id(song_id: str, difficulty_index: int, chart_hash: str,
             audio_hash: str | None = None, source_dataset: str | None = None) -> str:
    """Build a stable chart identity, optionally namespaced by dataset.

    The optional namespace lets a combined training selection contain two
    manifests with the same song id and chart bytes without making their UI
    rows or annotations collide.  Omitting it preserves the original v1 id
    contract for callers that do not have dataset metadata.
    """
    namespace = str(source_dataset or "").strip()
    payload = f"{namespace}\0{song_id}\0{int(difficulty_index)}\0{chart_hash}\0{audio_hash or ''}" if namespace else f"{song_id}\0{int(difficulty_index)}\0{chart_hash}\0{audio_hash or ''}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_regions(regions: Any, duration: float | None = None) -> list[dict[str, Any]]:
    if regions is None:
        return []
    if not isinstance(regions, list):
        raise ValueError("regions 必须是数组")
    result: list[dict[str, Any]] = []
    previous_end = -float("inf")
    for region in regions:
        if not isinstance(region, dict):
            raise ValueError("region 必须是对象")
        start = _finite(region.get("start_sec"), "region.start_sec")
        end = _finite(region.get("end_sec"), "region.end_sec")
        if start < 0 or end <= start:
            raise ValueError("region 必须满足 0 <= start_sec < end_sec")
        if duration is not None and end > duration + 1e-6:
            raise ValueError("region 超出谱面时长")
        if start < previous_end - 1e-9:
            raise ValueError("regions 不能重叠，且必须按时间排序")
        reason = str(region.get("reason", "")).strip()
        label = str(region.get("label", reason or "manual")).strip()
        if not label:
            raise ValueError("region.label 不能为空")
        weight = _finite(region.get("weight", 1.0), "region.weight")
        if weight < 0 or weight > 4:
            raise ValueError("region.weight 必须在 0–4 之间")
        item = {"start_sec": start, "end_sec": end, "label": label,
                "reason": reason, "weight": weight}
        for component in ("timing_weight", "component_weight"):
            if component in region:
                value = _finite(region[component], f"region.{component}")
                if not 0 <= value <= 4:
                    raise ValueError(f"region.{component} 必须在 0–4 之间")
                item[component] = value
        if 'entry_start_sec' in region:
            entry_start = _finite(region['entry_start_sec'], 'region.entry_start_sec')
            if not 0 <= entry_start <= start + 1e-9:
                raise ValueError('确认的接入起点须不晚于该片段起点')
            if label != 'good' or region_component_weights(item)[0] <= 0:
                raise ValueError('确认前文接入需要保留采音监督的好片段')
            item['entry_start_sec'] = entry_start
        result.append(item)
        previous_end = end
    return result


def validate_record(record: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError("annotation record 必须是对象")
    label = str(record.get("label", "normal"))
    if label not in LABELS:
        raise ValueError(f"未知质量标签: {label}")
    weight = _finite(record.get("weight", DEFAULT_WEIGHTS[label]), "weight")
    if weight < 0 or weight > 4:
        raise ValueError("weight 必须在 0–4 之间")
    out = dict(record)
    out["label"] = label
    out["weight"] = weight
    out["reason"] = str(record.get("reason", ""))
    out["regions"] = _validate_regions(record.get("regions", []))
    if "chart_override" in out and not isinstance(out["chart_override"], bool):
        raise ValueError("chart_override 必须是布尔值")
    if "unmarked_weight" in record:
        value = _finite(record["unmarked_weight"], "unmarked_weight")
        if not 0 <= value <= 4:
            raise ValueError("unmarked_weight 必须在 0–4 之间")
        out["unmarked_weight"] = value
    for key in ("chart_id", "song_id", "source_path", "audio_path", "chart_sha256", "audio_sha256"):
        if key in out and out[key] is not None:
            out[key] = str(out[key])
    if "difficulty_index" in out:
        try:
            out["difficulty_index"] = int(out["difficulty_index"])
        except (TypeError, ValueError) as exc:
            raise ValueError("difficulty_index 必须是整数") from exc
    return out


def load_annotations(manifest_path: str | Path) -> dict[str, Any]:
    candidate = Path(manifest_path).resolve()
    path = candidate if candidate.name == "quality_annotations.json" else annotation_path(candidate)
    if not path.exists():
        return {"schema": SCHEMA, "manifest": str(Path(manifest_path).resolve()), "records": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA:
        raise ValueError("质量标注文件 schema 不兼容")
    records = [validate_record(item) for item in data.get("records", [])]
    return {"schema": SCHEMA, "manifest": str(Path(manifest_path).resolve()), "records": records}


def save_annotations(manifest_path: str | Path, records: Iterable[dict[str, Any]]) -> Path:
    from maichart.training.dataset_catalog import _replace_file

    manifest = Path(manifest_path).resolve()
    path = annotation_path(manifest)
    payload = {"schema": SCHEMA, "manifest": str(manifest), "records": [validate_record(item) for item in records]}
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = _lock_for(path)
    with lock:
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            _replace_file(temp_name, path)
        finally:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
    return path


@dataclass
class CatalogChart:
    chart_id: str
    song_id: str
    difficulty_index: int
    source_path: str
    audio_path: str | None
    title: str | None
    artist: str | None
    level: str | None
    designer: str | None
    chart_sha256: str
    audio_sha256: str | None = None
    type_counts: dict[str, int] | None = None
    source_dataset: str | None = None
    source_dataset_id: str | None = None
    folder_name: str | None = None
    maidata_title: str | None = None
    legacy_chart_id: str | None = None
    source_manifest: str | None = None

    def to_dict(self, annotation: dict[str, Any] | None = None) -> dict[str, Any]:
        value = self.__dict__.copy()
        # Kept internally only for reading annotations created before dataset
        # namespacing was introduced.
        value.pop("legacy_chart_id", None)
        value["annotation"] = annotation
        value["label"] = annotation.get("label", "normal") if annotation else "normal"
        value["weight"] = annotation.get("weight", 1.0) if annotation else 1.0
        value["annotated"] = annotation is not None
        return value


def iter_catalog(manifest_path: str | Path) -> list[CatalogChart]:
    manifest_file = Path(manifest_path).resolve()
    data = json.loads(manifest_file.read_text(encoding="utf-8-sig"))
    source_root = _source_root(manifest_file, data)
    manifest_dataset, manifest_dataset_id = _manifest_dataset_metadata(manifest_file, data)
    rows: list[CatalogChart] = []
    songs = data.get("songs", [])
    parsed_sources: dict[Path, Any] = {}
    if len(songs) > 256:
        from concurrent.futures import ThreadPoolExecutor

        unique_maidata: list[Path] = []
        seen_maidata: set[Path] = set()
        for song in songs:
            maidata = _resolve_manifest_path(song.get("maidata_path", ""), manifest_file, source_root)
            if maidata not in seen_maidata:
                seen_maidata.add(maidata)
                unique_maidata.append(maidata)

        def _parse_one(path: Path) -> tuple[Path, Any]:
            if not path.is_file():
                return path, None
            try:
                return path, parse_maidata_file(path)
            except (OSError, ValueError, UnicodeError):
                return path, None

        with ThreadPoolExecutor(max_workers=8) as pool:
            for m_path, parsed in pool.map(_parse_one, unique_maidata):
                parsed_sources[m_path] = parsed

    for song in songs:
        maidata = _resolve_manifest_path(song.get("maidata_path", ""), manifest_file, source_root)
        if maidata not in parsed_sources:
            if not maidata.is_file():
                parsed_sources[maidata] = None
            else:
                try:
                    parsed_sources[maidata] = parse_maidata_file(maidata)
                except (OSError, ValueError, UnicodeError):
                    parsed_sources[maidata] = None
        parsed_chart = parsed_sources[maidata]
        if parsed_chart is None:
            continue
        audio_value = song.get("audio_path")
        audio = _resolve_manifest_path(audio_value, manifest_file, source_root) if audio_value else None
        for difficulty in song.get("difficulties", []):
            index = difficulty.get("difficulty_index", difficulty.get("index"))
            if index is None:
                continue
            index = int(index)
            try:
                c_hash = hash_chart(parsed_chart, index)
            except (OSError, ValueError, UnicodeError):
                continue
            a_hash = None
            # Audio hashing is lazy: only reuse a hash already persisted in a
            # record; catalog browsing never reads large audio files.
            song_id = str(song.get("song_id", "unknown"))
            source_dataset, source_dataset_id = _dataset_metadata(song.get("source_dataset"), manifest_dataset, manifest_dataset_id)
            legacy_id = chart_id(song_id, index, c_hash, a_hash)
            maidata_title = _clean_text(getattr(parsed_chart, "title", None))
            manifest_title = _clean_text(song.get("title"))
            display_title = maidata_title or manifest_title
            parsed_difficulty = next((item for item in parsed_chart.difficulties if item.index == index), None)
            parsed_level = _clean_text(getattr(parsed_difficulty, "level", None)) if parsed_difficulty else None
            parsed_designer = _clean_text(getattr(parsed_difficulty, "designer", None)) if parsed_difficulty else None
            global_designer = _clean_text(getattr(parsed_chart, "fields", {}).get("des"))
            level = parsed_level or _clean_text(difficulty.get("level", difficulty.get("level_raw")))
            designer = parsed_designer or global_designer or _clean_text(difficulty.get("designer"))
            source_key = _clean_text(song.get("source_path_key")) or _relative_source_key(maidata, source_root)
            rows.append(CatalogChart(
                # Include the relative source path in the namespace.  This
                # prevents duplicate song ids and identical chart bytes from
                # two folders in the same dataset from sharing annotations.
                chart_id=chart_id(song_id, index, c_hash, a_hash,
                                  f"{source_dataset_id}\0{source_key}"),
                song_id=song_id, difficulty_index=index,
                source_path=str(maidata), audio_path=str(audio) if audio else None,
                title=display_title, artist=song.get("artist"), level=level,
                designer=designer, chart_sha256=c_hash, audio_sha256=a_hash,
                type_counts=dict(difficulty.get("type_counts") or {}),
                source_dataset=source_dataset, source_dataset_id=source_dataset_id,
                folder_name=maidata.parent.name,
                maidata_title=maidata_title, legacy_chart_id=legacy_id,
                source_manifest=song.get("source_manifest"),
            ))
    return rows


def find_annotation(
    row: CatalogChart, records: Iterable[dict[str, Any]], *, verify_audio: bool = False,
    resolve_path=None, verified_audio_sha256: str | None = None,
) -> dict[str, Any] | None:
    if resolve_path is None:
        resolve_path = lambda value: Path(value).resolve()
    for record in records:
        record_id = record.get("chart_id")
        if record.get("chart_sha256") != row.chart_sha256:
            continue
        id_match = record_id in {row.chart_id, row.legacy_chart_id}
        path_match = False
        if record.get("source_path") and record.get("difficulty_index") is not None:
            try:
                path_match = (resolve_path(record["source_path"]) == resolve_path(row.source_path)
                              and int(record["difficulty_index"]) == int(row.difficulty_index))
            except (TypeError, ValueError, OSError):
                path_match = False
        if not id_match and not path_match:
            continue
        # A legacy id has no dataset namespace.  Only migrate it when its
        # stored source path identifies the same chart; otherwise two copied
        # charts with equal bytes could inherit one another's label.
        if record_id == row.legacy_chart_id and record.get("source_path"):
            try:
                if resolve_path(record["source_path"]) != resolve_path(row.source_path):
                    continue
            except (TypeError, OSError):
                continue
        if record.get("audio_path") and row.audio_path and resolve_path(record["audio_path"]) != resolve_path(row.audio_path):
            continue
        if verify_audio and record.get("audio_sha256") and row.audio_path:
            try:
                if record["audio_sha256"] != (verified_audio_sha256 or audio_sha256(row.audio_path)):
                    continue
            except OSError:
                continue
        return record
    return None


def _clean_text(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _dataset_value(value: Any) -> str | None:
    if isinstance(value, dict):
        value = value.get("name") or value.get("id") or value.get("label")
    return _clean_text(value)


def _dataset_metadata(value: Any, fallback_name: str, fallback_id: str) -> tuple[str, str]:
    """Split dataset display name from its optional collision-resistant id."""
    if isinstance(value, dict):
        display = _clean_text(value.get("name") or value.get("label") or value.get("id")) or fallback_name
        identity = _clean_text(value.get("id") or value.get("key") or value.get("name")) or fallback_id
        return display, identity
    text = _clean_text(value)
    return (text or fallback_name, fallback_id)


def _relative_source_key(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _manifest_dataset_name(manifest_file: Path, data: dict[str, Any]) -> str:
    """Return a human-readable and identity-stable dataset namespace."""
    for key in ("source_dataset", "dataset_name", "dataset_id"):
        value = data.get(key)
        value = _dataset_value(value)
        if value:
            return value
    return manifest_file.parent.name or manifest_file.stem


def _manifest_dataset_metadata(manifest_file: Path, data: dict[str, Any]) -> tuple[str, str]:
    fallback_name = manifest_file.parent.name or manifest_file.stem
    fallback_id = _fallback_dataset_id(manifest_file, data)
    for key in ("source_dataset", "dataset_name", "dataset_id"):
        if key in data:
            value = data.get(key)
            # dataset_id is explicitly an identity, while source_dataset and
            # dataset_name are ordinarily display labels unless represented as
            # an object with an id/key.
            if key == "dataset_id" and not isinstance(value, dict):
                explicit = _clean_text(value)
                if explicit:
                    return explicit, explicit
            return _dataset_metadata(value, fallback_name, fallback_id)
    return fallback_name, fallback_id


def _fallback_dataset_id(manifest_file: Path, data: dict[str, Any]) -> str:
    """Namespace unnamed manifests by their resolved source location.

    The folder name remains a display label, while this digest prevents two
    same-named dataset folders from sharing chart identities.
    """
    source_root = _source_root(manifest_file, data)
    identity = f"{manifest_file.resolve()}\0{source_root.resolve()}"
    return "path-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]


def resolve_annotation(record: dict[str, Any] | None, source_path: str | Path, chart: Any,
                       difficulty_index: int, audio_path: str | Path | None = None) -> dict[str, Any] | None:
    """Resolve only a record with matching source, slot, and content identity.

    Records without ``source_path`` are treated as stale because a chart hash
    alone can match copies in unrelated folders. If an audio path or hash is
    stored, the caller must provide the same audio identity as well.
    """
    if not record:
        return None
    try:
        if int(record.get("difficulty_index", -1)) != int(difficulty_index):
            return None
        stored_source = record.get("source_path")
        if not stored_source or Path(stored_source).resolve() != Path(source_path).resolve():
            return None
        stored_audio = record.get("audio_path")
        if stored_audio:
            if not audio_path or Path(stored_audio).resolve() != Path(audio_path).resolve():
                return None
    except (OSError, TypeError, ValueError):
        return None
    # Use the selected source slot's canonical text identity. EventChart is an
    # already parsed, timing-normalized view and cannot reproduce the authored
    # raw inote byte-for-byte for this identity contract.
    current_chart_hash = chart_sha256(source_path, difficulty_index)
    if record.get("chart_sha256") != current_chart_hash:
        return None
    if record.get("audio_sha256"):
        if not audio_path:
            return None
        try:
            if record["audio_sha256"] != audio_sha256(audio_path):
                return None
        except OSError:
            return None
    return validate_record(record)


def load_quality_annotations(path: str | Path) -> dict[str, Any]:
    """Alias used by dataset/training code when it receives the annotation file."""
    return load_annotations(path)


def load_auto_annotations(manifest_path: str | Path) -> dict[str, Any]:
    """Load machine-generated quality audit records without touching manual annotations."""
    candidate = Path(manifest_path)
    if candidate.name not in (AUTO_ANNOTATION_FILENAME, AUTO_ANNOTATION_PREVIEW_FILENAME,
                               COMPOSED_AUTO_ANNOTATION_FILENAME):
        if candidate.is_file():
            _, data = resolve_auto_annotations(candidate)
            return data if data is not None else {"schema": SCHEMA, "manifest": str(candidate.resolve()), "records": []}
        path = auto_annotation_path(candidate)
    else:
        path = candidate
    if not path.exists():
        return {"schema": SCHEMA, "manifest": str(candidate.resolve()), "records": []}
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    return _validate_auto_annotations(data)


def _validate_auto_annotations(data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise ValueError("自动质检标注文件 schema 不兼容")
    records = [validate_record(item) for item in data.get("records", [])]
    out: dict[str, Any] = {"schema": SCHEMA, "manifest": data.get("manifest"), "records": records}
    if "audit_metadata" in data and isinstance(data["audit_metadata"], dict):
        out["audit_metadata"] = dict(data["audit_metadata"])
    return out


def _auto_annotation_owner(path: Path, data: dict[str, Any]) -> Path | None:
    from maichart.portable_paths import material_path

    saved_manifest = data.get("manifest") if isinstance(data, dict) else None
    if isinstance(saved_manifest, str) and saved_manifest:
        return (path.parent / material_path(saved_manifest)).resolve()
    return None


def resolve_auto_annotations(
    manifest_path: str | Path, payload: dict[str, Any] | None = None, *, preview: bool = False,
) -> tuple[Path | None, dict[str, Any] | None]:
    """Resolve a matching direct, inherited, or original split-source sidecar."""
    manifest = Path(manifest_path).resolve()
    if payload is None:
        payload = json.loads(manifest.read_text(encoding="utf-8-sig"))
    owners = [manifest]
    if not preview:
        origin = split_annotation_source(manifest, payload)
        if origin is not None and origin != manifest:
            owners.append(origin)
    for owner in owners:
        paths = [auto_annotation_path(owner, preview=preview)]
        if not preview:
            paths.append(auto_annotation_path(owner, composed=True))
        for path in paths:
            if not path.is_file():
                continue
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            if _auto_annotation_owner(path, data) != owner:
                continue
            result = _validate_auto_annotations(data)
            digest = (result.get("audit_metadata") or {}).get("manifest_sha256")
            if digest and (not owner.is_file() or digest != hashlib.sha256(owner.read_bytes()).hexdigest()):
                continue
            return path, result
    return None, None


def auto_audit_report_path(manifest: Path, *, preview=False, previous=False) -> Path:
    suffix = "_preview" if preview else ""
    name = f"auto_quality_audit{suffix}_report"
    return manifest.with_name(f"{name}{'_previous' if previous else ''}.json")


def resolve_auto_audit(
    manifest_path: str | Path, payload: dict[str, Any] | None = None, *, preview: bool = False,
) -> tuple[Path | None, dict[str, Any] | None, dict[str, Any] | None]:
    """Read only reports for the requested manifest and the published audit, if available."""
    manifest = Path(manifest_path).resolve()
    if payload is None:
        payload = json.loads(manifest.read_text(encoding="utf-8-sig"))
    try:
        auto_path, auto_data = resolve_auto_annotations(manifest, payload, preview=preview)
    except (OSError, ValueError, TypeError):
        auto_path, auto_data = None, None
    owner = _auto_annotation_owner(auto_path, auto_data) if auto_path is not None else manifest
    if owner is None:
        return auto_path, auto_data, None
    metadata = (auto_data or {}).get("audit_metadata") or {}
    if metadata.get("composed_sources") is not None:
        # A report from a direct audit cannot describe the inherited fallback.
        return auto_path, auto_data, None
    for previous in (False, True):
        if previous and auto_data is None:
            break
        path = auto_audit_report_path(owner, preview=preview, previous=previous)
        try:
            report = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        if (not isinstance(report, dict) or report.get("schema") != "composer-v2-auto-quality-audit-v1"
                or _auto_annotation_owner(path, report) != owner or bool(report.get("preview")) != preview):
            continue
        audit_id = report.get("audit_id")
        if (audit_id or metadata.get("audit_id")) and audit_id != metadata.get("audit_id"):
            continue
        digest = report.get("manifest_sha256")
        if digest and (not owner.is_file() or digest != hashlib.sha256(owner.read_bytes()).hexdigest()):
            continue
        report["report_path"] = str(path)
        return auto_path, auto_data, report
    return auto_path, auto_data, None


def save_auto_annotations(
    manifest_path: str | Path,
    records: Iterable[dict[str, Any]],
    *,
    audit_metadata: dict[str, Any] | None = None,
    preview: bool = False,
    composed: bool = False,
) -> Path:
    """Atomically write audit annotations alongside their manifest."""
    from maichart.training.dataset_catalog import _replace_file

    manifest = Path(manifest_path).resolve()
    path = auto_annotation_path(manifest, preview=preview, composed=composed)
    if path == manifest:
        raise ValueError("自动质检标注不能覆盖输入清单")
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "manifest": str(manifest),
        "records": [validate_record(item) for item in records],
    }
    if audit_metadata is not None:
        payload["audit_metadata"] = dict(audit_metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = _lock_for(path)
    with lock:
        if not preview and path.is_file():
            try:
                existing = json.loads(path.read_text(encoding="utf-8-sig"))
            except (UnicodeError, ValueError):
                existing = None
            owner = _auto_annotation_owner(path, existing)
            if owner is not None and owner != manifest:
                raise ValueError(f"自动质检标注已绑定另一训练清单：{owner}")
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            _replace_file(temp_name, path)
        finally:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
    return path


def remove_composed_auto_annotations(manifest_path: Path) -> None:
    """Clear only a generated sidecar when an explicit composition is replaced."""
    manifest = manifest_path.resolve()
    for composed in (True, False):
        path = auto_annotation_path(manifest, composed=composed)
        with _lock_for(path):
            try:
                data = json.loads(path.read_text(encoding="utf-8-sig"))
            except (FileNotFoundError, UnicodeError, ValueError):
                continue
            if _auto_annotation_owner(path, data) != manifest:
                continue
            metadata = data.get("audit_metadata")
            if composed or (isinstance(metadata, dict) and "composed_sources" in metadata):
                path.unlink(missing_ok=True)


def _subtract_intervals(
    start: float,
    end: float,
    cutouts: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    pieces = [(start, end)]
    for cut_lo, cut_hi in cutouts:
        next_pieces: list[tuple[float, float]] = []
        for lo, hi in pieces:
            if cut_hi <= lo or cut_lo >= hi:
                next_pieces.append((lo, hi))
                continue
            if cut_lo > lo:
                next_pieces.append((lo, cut_lo))
            if hi > cut_hi:
                next_pieces.append((cut_hi, hi))
        pieces = next_pieces
        if not pieces:
            break
    return pieces


def merge_annotation_records(
    manual: dict[str, Any] | None,
    auto: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Merge manual and automated quality records with strict ``manual > auto`` priority.

    * Manual chart-level exclusions and ``unmarked_weight == 0`` reviewed-only charts
      remain 100% governed by ``manual``.
    * Explicit chart-level decisions override automatic chart-level exclusions.
      Interval-only edits keep the automatic chart-level decision. Legacy
      records without a scope retain the saved chart-level interpretation.
    * Any overlap between an automated region and a manual region is subtracted from
      the automated region so human ``good`` / ``problem`` spans win unconditionally
      and the merged ``regions`` array remains sorted and strictly disjoint.
    """
    if auto is None:
        return validate_record(manual) if manual is not None else None
    valid_auto = validate_record(auto)
    if manual is None:
        return valid_auto
    valid_manual = validate_record(manual)
    if (
        valid_manual.get("label") in ("excluded", "negative")
        or float(valid_manual.get("weight", 1.0)) <= 0.0
        or valid_manual.get("unmarked_weight") == 0
    ):
        return valid_manual

    manual_regions = list(valid_manual.get("regions", ()))
    cutouts = [(float(r["start_sec"]), float(r["end_sec"])) for r in manual_regions]
    kept_auto: list[dict[str, Any]] = []
    for ar in valid_auto.get("regions", ()):
        for sub_lo, sub_hi in _subtract_intervals(float(ar["start_sec"]), float(ar["end_sec"]), cutouts):
            item = dict(ar)
            item.pop("entry_start_sec", None)
            item["start_sec"] = sub_lo
            item["end_sec"] = sub_hi
            kept_auto.append(item)

    merged_regions = sorted(
        [*manual_regions, *kept_auto],
        key=lambda r: (float(r["start_sec"]), float(r["end_sec"])),
    )
    out = dict(valid_manual)
    if valid_manual.get("chart_override") is False:
        out.update(label=valid_auto["label"], weight=valid_auto["weight"], reason=valid_auto["reason"])
    out["regions"] = merged_regions
    return validate_record(out)


def annotations_fingerprint(payload: dict[str, Any] | list[dict[str, Any]]) -> str:
    """Stable digest for including annotation content in a training contract."""
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def chart_duration(maidata_path: str | Path, difficulty_index: int) -> float | None:
    chart = parse_maidata_file(maidata_path)
    difficulty = next((item for item in chart.difficulties if item.index == difficulty_index), None)
    if difficulty is None or not difficulty.inote:
        return None
    try:
        points = tokenize_chart_difficulty_timing(
            chart, difficulty_index, initial_bpm=float(chart.wholebpm or 120))
        return max((float(point.time_sec or 0.0) for point in points), default=0.0, key=float)
    except Exception:
        return None



def region_component_weights(region: dict[str, Any]) -> tuple[float, float]:
    """Ordinary CE multipliers; negative feedback never reverses CE gradients.

    Explicit legacy weight-only regions retain their original meaning. Light
    labels provide conservative defaults and can be overridden per component.
    """
    label = region.get("label")
    reason = region.get("reason", "")
    default = (1.5, 1.5) if label == "good" else (1., 1.) if label == "uncertain" else (1., 1.)
    if label == "problem":
        default = (0., 1.) if reason == "timing" else (1., 0.) if reason in ("layout", "comfort") else (0., 0.)
    base = float(region.get("weight", 1.))
    return tuple(base * float(region.get(key, value)) for key, value in
                 zip(("timing_weight", "component_weight"), default))


def zero_unmarked_region_weights(
    region: dict[str, Any], *, require_explicit_good: bool = True
) -> tuple[float, float]:
    """Effective (timing, component) weights on an unmarked_weight=0 chart.

    Persisted regions carry a ``label`` after validation; on zero-unmarked
    sources only explicit ``label == 'good'`` (with ``training_use != 'none'``)
    counts as positive supervision. Legacy in-memory callers may explicitly
    opt out; persisted data must never infer approval from weights alone.
    """
    if not isinstance(region, dict) or region.get("training_use") == "none":
        return (0.0, 0.0)
    if require_explicit_good:
        if region.get("label") != "good":
            return (0.0, 0.0)
    elif "label" in region and region.get("label") != "good":
        return (0.0, 0.0)
    timing, components = region_component_weights(region)
    return (max(0.0, float(timing)), max(0.0, float(components)))
