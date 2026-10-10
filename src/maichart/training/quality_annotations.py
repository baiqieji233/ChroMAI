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

from maichart.maidata import parse_maidata_file
from maichart.timing import tokenize_inote_timing

SCHEMA = "maichart-quality-annotations-v1"
DEFAULT_WEIGHTS = {"normal": 1.0, "excellent": 2.0, "reduced": 0.25, "excluded": 0.0, "negative": 0.0}
LABELS = frozenset(DEFAULT_WEIGHTS)
_WRITE_LOCKS: dict[str, threading.Lock] = {}
_WRITE_LOCKS_GUARD = threading.Lock()


def annotation_path(manifest_path: str | Path) -> Path:
    """Return the automatically managed annotation path for a manifest."""
    return Path(manifest_path).resolve().with_name("quality_annotations.json")


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
    raw = Path(str(value))
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
        f"first={chart.first or ''}",
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
                             f"first={getattr(chart_or_path, 'first', '') or ''}", f"level={difficulty.level or ''}",
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
        result.append({"start_sec": start, "end_sec": end, "label": label,
                       "reason": reason, "weight": weight})
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
            os.replace(temp_name, path)
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
    data = json.loads(manifest_file.read_text(encoding="utf-8"))
    source_root = _source_root(manifest_file, data)
    manifest_dataset, manifest_dataset_id = _manifest_dataset_metadata(manifest_file, data)
    rows: list[CatalogChart] = []
    parsed_sources: dict[Path, Any] = {}
    for song in data.get("songs", []):
        maidata = _resolve_manifest_path(song.get("maidata_path", ""), manifest_file, source_root)
        if not maidata.is_file():
            continue
        if maidata not in parsed_sources:
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
            ))
    return rows


def find_annotation(row: CatalogChart, records: Iterable[dict[str, Any]], *, verify_audio: bool = False) -> dict[str, Any] | None:
    for record in records:
        record_id = record.get("chart_id")
        if record.get("chart_sha256") != row.chart_sha256:
            continue
        id_match = record_id in {row.chart_id, row.legacy_chart_id}
        path_match = False
        if record.get("source_path") and record.get("difficulty_index") is not None:
            try:
                path_match = (Path(record["source_path"]).resolve() == Path(row.source_path).resolve()
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
                if Path(record["source_path"]).resolve() != Path(row.source_path).resolve():
                    continue
            except (TypeError, OSError):
                continue
        if record.get("audio_path") and row.audio_path and Path(record["audio_path"]).resolve() != Path(row.audio_path).resolve():
            continue
        if verify_audio and record.get("audio_sha256") and row.audio_path:
            try:
                if record["audio_sha256"] != audio_sha256(row.audio_path):
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
    """Resolve one stored record against current content, never by path alone.

    A changed maidata file therefore cannot inherit an old label merely because
    it was written at the same path.  If a record has an audio hash, callers may
    pass ``audio_path`` to enforce the corresponding content identity too.
    """
    if not record:
        return None
    # Use the selected source slot's canonical text identity. EventChart is an
    # already parsed, timing-normalized view and cannot reproduce the authored
    # raw inote byte-for-byte for this identity contract.
    current_chart_hash = chart_sha256(source_path, difficulty_index)
    if record.get("chart_sha256") != current_chart_hash:
        return None
    if record.get("audio_sha256") and audio_path:
        try:
            if record["audio_sha256"] != audio_sha256(audio_path):
                return None
        except OSError:
            return None
    return validate_record(record)


def load_quality_annotations(path: str | Path) -> dict[str, Any]:
    """Alias used by dataset/training code when it receives the annotation file."""
    return load_annotations(path)


def annotations_fingerprint(payload: dict[str, Any] | list[dict[str, Any]]) -> str:
    """Stable digest for including annotation content in a training contract."""
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def chart_duration(maidata_path: str | Path, difficulty_index: int) -> float | None:
    chart = parse_maidata_file(maidata_path)
    difficulty = next((item for item in chart.difficulties if item.index == difficulty_index), None)
    if difficulty is None or not difficulty.inote:
        return None
    try:
        points = tokenize_inote_timing(difficulty.inote, initial_bpm=float(chart.wholebpm or 120))
        return max((float(point.time_sec or 0.0) for point in points), default=0.0) + max(float(chart.first or 0), 0.0)
    except Exception:
        return None

