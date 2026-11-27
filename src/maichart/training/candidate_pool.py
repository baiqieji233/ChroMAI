"""Isolated candidate libraries: unknown charts are context, never negative examples.

Scanning is explicit and CPU-only. Browsing never rewrites identities or reviews.
Budget units are weighted note heads, an approximation, not token loss weights.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import re
import threading
import time
import unicodedata
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from maichart.training.dataset_catalog import _absolutize_song_paths, _source_asset, _write_json
from maichart.training.quality_annotations import (
    chart_sha256, audio_sha256, iter_catalog, region_component_weights,
    zero_unmarked_region_weights,
)

SCHEMA = "maichart-candidate-pool-v1"
POLICY = "candidate_pool_v1_reviewed_only"


def _read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else copy.deepcopy(default or {})


def _now():
    return datetime.now(timezone.utc).isoformat()


def _pool(value):
    path = Path(value).resolve()
    return path.parent if path.name in {"pool.json", "training_manifest.json"} else path


def _stat(path):
    try:
        stat = Path(path).stat()
        return [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
    except (OSError, TypeError):
        return None


def _file_version(path):
    try:
        stat = Path(path).stat()
        return stat.st_mtime_ns, stat.st_size, stat.st_ctime_ns, stat.st_ino
    except FileNotFoundError:
        return None


def _hash(path, cache):
    if not path or not Path(path).is_file():
        return None
    key = str(Path(path).resolve())
    signature = _stat(path)
    prior = cache.get(key, {})
    if prior.get("stat") != signature:
        prior = {"stat": signature, "sha256": audio_sha256(path)}
        cache[key] = prior
    return prior["sha256"]


@lru_cache(maxsize=1024)
def _cached_digest(path, signature):
    return audio_sha256(path)


def _digest(path):
    return _cached_digest(str(path), tuple(_stat(path) or ()))


def _charts(manifest, payload=None, *, resolve_asset=_source_asset):
    path = Path(manifest).resolve()
    payload = _read(path) if payload is None else payload
    for song in payload.get("songs", []):
        song = copy.deepcopy(song)
        _absolutize_song_paths(song, path, payload, resolve_asset=resolve_asset)
        for difficulty in song.get("difficulties", []):
            index = difficulty.get("difficulty_index", difficulty.get("index"))
            if index is not None:
                yield song, difficulty, int(index)


def _key(path, index):
    return str(Path(path).resolve()).casefold(), int(index)


def _resolved_slot(path, index):
    return os.path.normcase(os.path.normpath(path)), int(index)


@lru_cache(maxsize=2)
def _pool_charts(manifest, version):
    payload = _read(manifest)
    index = {}
    for song in payload.get("songs", ()):
        source = song.get("maidata_path")
        if not source:
            continue
        if not Path(source).is_absolute() or (song.get("audio_path") and not Path(song["audio_path"]).is_absolute()):
            _absolutize_song_paths(song, manifest, payload)
        for difficulty in song.get("difficulties", ()):
            slot = difficulty.get("difficulty_index", difficulty.get("index"))
            if slot is not None:
                index[_resolved_slot(song["maidata_path"], slot)] = song, difficulty
    return index


def pool_charts(manifest):
    manifest = Path(manifest)
    return _pool_charts(manifest, _file_version(manifest))


@dataclass(eq=False)
class _ScanChart:
    info: dict
    fields: tuple
    segments: tuple
    cached_choices: tuple | None = None


_SCAN_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def _scan_index(path, version):
    rows = {}
    for item in _read(path).get("charts", []):
        info = {key: item[key] for key in (
            "chart_id", "source_path", "difficulty_index", "status", "title", "issues",
            "level", "audio_duration_sec",
        ) if key in item}
        info["bpm"] = item.get("stats", {}).get("bpm") or 120
        segments = [value for value in item.get("segments", [])
                    if value.get("queue_eligible") is not False and value.get("onsets", 0) >= 2]
        # A tuple of values avoids retaining a dict per segment. Preserve all
        # fields for the few recommendations returned to existing callers.
        fields = tuple(dict.fromkeys(key for segment in segments for key in segment))
        compact = tuple(tuple(segment.get(key) for key in fields) for segment in segments)
        rows[_resolved_slot(item["source_path"], item["difficulty_index"])] = _ScanChart(info, fields, compact)
    return rows


def scan_index(manifest):
    path = Path(manifest).parent / "scan" / "results.json"
    with _SCAN_LOCK:
        return _scan_index(path, _file_version(path))


@lru_cache(maxsize=2)
def _scan_issues(path, version):
    return {
        _resolved_slot(item["source_path"], item["difficulty_index"]): (item.get("chart_id"), item["status"], item.get("issues", []))
        for item in _read(path).get("charts", [])
    }


def scan_issues(manifest):
    path = Path(manifest).parent / "scan" / "results.json"
    version = _file_version(path)
    with _SCAN_LOCK:
        # Browsing keeps only issues, so it need not hold the segment index.
        return _scan_issues(path, version)


def _records(manifest):
    return _read(Path(manifest).with_name("quality_annotations.json"), {"records": []}).get("records", [])


@lru_cache(maxsize=4096)
def _authored_key(path, index, signature):
    from maichart.maidata import parse_maidata_file, resolve_chart_first
    chart = parse_maidata_file(path)
    slot = next(d for d in chart.difficulties if d.index == index)
    text = re.sub(r"\s+", "", slot.inote or "")
    if not text:
        return None
    canonical = json.dumps([str(chart.wholebpm or ""), str(resolve_chart_first(chart, index) or ""), text])
    return hashlib.sha256(canonical.encode()).hexdigest()


def _content_identity(path, index):
    return _authored_key(str(path), int(index), tuple(_stat(path) or ()))


def _title_key(title):
    title = unicodedata.normalize("NFKC", str(title or "")).casefold()
    return "".join(c for c in title if c.isalnum())


@lru_cache(maxsize=4096)
def _reference_features(path, index, signature):
    from maichart.maidata import parse_maidata_file
    from maichart.event_timing import parse_event_chart
    raw = parse_maidata_file(path)
    chart = parse_event_chart(path, index, strict_notes=False)
    events = sorted({json.dumps([str(point.beat), re.sub(r"\s+", "", point.note_text)], ensure_ascii=False)
                     for point in chart.points})
    return dict(title=_title_key(raw.title), events=events,
                tempo_changes=[[str(change.beat), float(change.bpm)] for change in chart.tempo_changes])


def _valid_regions(record):
    if not isinstance(record, dict):
        return []
    if record.get("label") in {"excluded", "negative"}:
        return []
    try:
        outer_weight = float(record.get("weight", 1.0))
    except (TypeError, ValueError):
        return []
    if not math.isfinite(outer_weight) or outer_weight <= 0:
        return []
    result = []
    for region in record.get("regions", []):
        try:
            a, b = float(region["start_sec"]), float(region["end_sec"])
            t, c = zero_unmarked_region_weights(region)
            if all(math.isfinite(x) for x in (a, b, t, c)) and 0 <= a < b and (t > 0 or c > 0):
                result.append(region)
        except (KeyError, ValueError, TypeError):
            continue
    return result


def candidate_record(manifest, song, difficulty, records=None, *, _metadata=None):
    """Return current, positively approved candidate supervision or None.

    Shared with composition; audio identity is mandatory for candidate reviews.
    """
    if difficulty.get("candidate_status") in {None, "pending", "error", "stale"}:
        return None
    metadata = _read(Path(manifest).with_name("pool.json")) if _metadata is None else _metadata
    if not metadata.get("scan_complete") or _stale(Path(manifest).parent, metadata, song, difficulty):
        return None
    source = song.get("maidata_path")
    index = int(difficulty.get("difficulty_index", difficulty.get("index")))
    try:
        current = _chart_digest(source, index)
        audio = _digest(song["audio_path"])
    except (OSError, ValueError, UnicodeError, KeyError):
        return None
    identity = difficulty.get("candidate_identity", {})
    if current != identity.get("chart_sha256") or audio != identity.get("audio_sha256"):
        return None
    for record in records if records is not None else _records(manifest):
        if (_key(record.get("source_path", ""), record.get("difficulty_index", -1)) == _key(source, index)
                and record.get("chart_sha256") == current and record.get("audio_sha256") == audio
                and _valid_regions(record)):
            try:
                valid_regions = _valid_regions(record)
                for sec, count in _head_events(source, index):
                    region = next((r for r in reversed(record.get("regions", []))
                                   if float(r["start_sec"]) <= sec < float(r["end_sec"])), None)
                    if count and region and region in valid_regions:
                        return record
            except (OSError, ValueError, UnicodeError):
                return None
    return None


def pool_identity(manifest, workspace_root):
    source = Path(manifest).resolve()
    pool_id = hashlib.sha256(str(source).casefold().encode()).hexdigest()[:12]
    root = Path(workspace_root).resolve() / "candidate_pools" / pool_id
    return dict(pool_id=pool_id, pool_dir=str(root), manifest=str(root / "training_manifest.json"))


def create_pool(manifest, workspace_root, name=None, reference_manifests=(), *, progress=None):
    source = Path(manifest).resolve()
    payload = _read(source)
    if payload.get("schema") != "maichart-training-manifest-v1":
        raise ValueError("需要已预处理的训练清单")
    if payload.get("candidate_pool"):
        raise ValueError("候选库不能再次作为候选库来源")
    identity = pool_identity(source, workspace_root)
    pool_id, root = identity["pool_id"], Path(identity["pool_dir"])
    if (root / "pool.json").is_file():
        return pool_status(root)
    root.mkdir(parents=True, exist_ok=True)
    metadata = dict(schema=SCHEMA, pool_id=pool_id, name=name or f"候选库 · {source.parent.name}",
                    source_manifest=str(source), reference_manifests=[str(Path(p).resolve()) for p in reference_manifests
                        if Path(p).resolve() != source], scan_policy=None, policy=POLICY,
                    created_at=_now(), updated_at=_now(), excluded=[])
    _write_json(root / "pool.json", metadata)
    _synchronize(root, progress=progress)
    return pool_status(root)


def _synchronize(root, progress=None):
    """Explicitly rebuild from the source, archiving invalidated review records."""
    metadata = _read(root / "pool.json")
    source = Path(metadata["source_manifest"])
    payload = _read(source)
    old_records = _records(root / "training_manifest.json")
    # Preserve the previous first-match rule, without searching all records
    # (and resolving their Windows paths) once for every chart.
    old_by_key = {}
    for record in old_records:
        old_by_key.setdefault(_key(record.get("source_path", ""), record.get("difficulty_index", -1)), record)
    hash_cache = _read(root / "identity_cache.json")
    reference_cache_path = root.parent / "_reference_index.json"
    reference_cache = _read(reference_cache_path)
    reference_dirty = False
    reference_keys, reference_hashes, reference_contents, reference_titles = {}, {}, {}, {}
    references = []
    for reference in metadata.get("reference_manifests", []):
        if not Path(reference).is_file():
            raise ValueError(f"参考清单已不存在：{reference}")
        reference_payload = _read(reference)
        if reference_payload.get("candidate_pool"):
            continue
        references.append((reference, reference_payload))
    total_references = sum(len(s.get('difficulties', [])) for _, data in references for s in data.get('songs', []))
    completed_references = 0
    if progress:
        progress(dict(stage='references', completed=0, total=total_references))
    for reference, reference_payload in references:
        for song, difficulty, index in _charts(reference, reference_payload):
            path = song.get("maidata_path", "")
            reference_keys[_key(path, index)] = dict(manifest=reference, source_path=path, slot=index,
                                                    audio_path=song.get("audio_path"))
            try:
                cache_key = f"{str(Path(path).resolve()).casefold()}\0{index}"
                entry = reference_cache.get(cache_key, {})
                if entry.get("stat") != _stat(path) or "features" not in entry:
                    entry = dict(stat=_stat(path), chart_sha256=chart_sha256(path, index),
                                 authored_content_sha256=_content_identity(path, index))
                    try:
                        entry["features"] = _reference_features(str(path), index, tuple(_stat(path) or ()))
                    except (OSError, ValueError, UnicodeError):
                        entry["features"] = {}
                    reference_cache[cache_key] = entry
                    reference_dirty = True
                match_entry = dict(reference_keys[_key(path, index)], features=entry.get("features", {}))
                if not match_entry["features"].get("title") and song.get("title"):
                    match_entry["features"] = dict(match_entry["features"], title=_title_key(song["title"]))
                reference_hashes.setdefault(entry["chart_sha256"], []).append(match_entry)
                if entry.get("authored_content_sha256"):
                    reference_contents.setdefault(entry["authored_content_sha256"], []).append(match_entry)
                title = match_entry["features"].get("title")
                if title:
                    reference_titles.setdefault(title, []).append(match_entry)
            except (OSError, ValueError, UnicodeError, StopIteration):
                pass
            completed_references += 1
            if progress:
                progress(dict(stage='references', completed=completed_references, total=total_references))
    if reference_dirty:
        _write_json(reference_cache_path, reference_cache)
    songs, excluded, identities = [], [], {}
    total_source = sum(len(s.get('difficulties', [])) for s in payload.get('songs', []))
    completed_source = 0
    if progress:
        progress(dict(stage='identities', completed=0, total=total_source))
    for original in payload.get("songs", []):
        song = copy.deepcopy(original)
        _absolutize_song_paths(song, source, payload)
        song.update(reviewed_only=True, candidate_pool=metadata["pool_id"])
        kept = []
        for difficulty in song.get("difficulties", []):
            completed_source += 1
            if progress:
                progress(dict(stage='identities', completed=completed_source - 1, total=total_source,
                              title=song.get('title') or Path(song.get('maidata_path', '')).parent.name))
            index = int(difficulty.get("difficulty_index", difficulty.get("index", -1)))
            path = song.get("maidata_path", "")
            try:
                digest = chart_sha256(path, index)
                content = _content_identity(path, index)
            except (OSError, ValueError, UnicodeError, StopIteration):
                digest = None
                content = None
            audio = song.get("audio_path")
            match = reference_keys.get(_key(path, index))
            if not match:
                # Chart bytes alone are not song identity: sparse charts can
                # coincide on unrelated music. Require equal audio content.
                exact = reference_hashes.get(digest, []) + reference_contents.get(content, [])
                for reference in exact:
                    if audio and reference.get("audio_path") and _stat(audio) and _stat(reference["audio_path"]):
                        if _hash(audio, hash_cache) == _hash(reference["audio_path"], hash_cache):
                            match = {k: v for k, v in reference.items() if k != "features"}
                            break
            if match:
                excluded.append(dict(source_path=path, slot=index, reason="existing_exact_chart", match=match))
                continue
            difficulty.pop("similar_reference", None)
            try:
                features = _reference_features(str(path), index, tuple(_stat(path) or ()))
                if not features.get("title") and song.get("title"):
                    features = dict(features, title=_title_key(song["title"]))
                events = set(features["events"])
                similar = []
                for reference in reference_titles.get(features["title"], []):
                    reference_events = set(reference["features"].get("events", []))
                    union = events | reference_events
                    score = len(events & reference_events) / len(union) if union else 0.
                    if score >= .8:
                        similar.append(dict(manifest=reference["manifest"], source_path=reference["source_path"],
                                            slot=reference["slot"], similarity=round(score, 6),
                                            method="normalized_title_event_set_jaccard_v1", reference_stat=_stat(reference["source_path"]),
                                            same_tempo_map=features["tempo_changes"] == reference["features"].get("tempo_changes"),
                                            same_audio_path=bool(audio and reference.get("audio_path") and
                                                Path(audio).resolve() == Path(reference["audio_path"]).resolve())))
                if similar:
                    difficulty["similar_reference"] = sorted(similar, key=lambda row: (-row["similarity"], row["source_path"], row["slot"]))[:8]
            except (OSError, ValueError, UnicodeError):
                pass
            identity = dict(chart_sha256=digest, audio_sha256=_hash(audio, hash_cache),
                            source_stat=_stat(path), audio_stat=_stat(audio), audio_path=audio)
            difficulty["candidate_status"] = "pending"
            difficulty["candidate_identity"] = identity
            identities[_key(path, index)] = identity
            kept.append(difficulty)
        if kept:
            song["difficulties"] = kept
            songs.append(song)
    output = root / "training_manifest.json"
    prepared = copy.deepcopy(payload)
    # Keep original asset root: all copied asset paths are absolute, while
    # catalog namespaces remain stable when an annotation is saved later.
    from maichart.training.dataset_catalog import _source_root
    prepared.update(songs=songs, source_root=str(_source_root(source, payload)),
                    dataset_id="candidate-" + metadata["pool_id"], dataset_name=metadata["name"],
                    candidate_pool=dict(pool_id=metadata["pool_id"], policy=POLICY),
                    manifest=str(output), quality_annotations=str(root / "quality_annotations.json"),
                    song_count=len(songs), difficulty_count=sum(len(s["difficulties"]) for s in songs))
    prepared["usable_difficulty_count"] = sum(bool(d.get("event_usable_for_training", d.get("usable_for_training")))
                                              for song in songs for d in song["difficulties"])
    prepared["summary"] = dict(song_count=prepared["song_count"], difficulty_count=prepared["difficulty_count"],
                               usable_difficulty_count=prepared["usable_difficulty_count"], reviewed_only=True)
    _write_json(output, prepared)
    records, invalid = [], []
    if progress:
        progress(dict(stage='annotations', completed=0, total=len(identities)))
    catalog = iter_catalog(output)
    for row in catalog:
        key = _key(row.source_path, row.difficulty_index)
        identity = identities[key]
        prior = old_by_key.get(key)
        if prior and (prior.get("chart_sha256") != identity["chart_sha256"] or
                      prior.get("audio_sha256") != identity["audio_sha256"]):
            invalid.append(prior)
            prior = None
        record = copy.deepcopy(prior) if prior else dict(row.to_dict(), label="normal", weight=1., unmarked_weight=0., regions=[])
        record.pop("annotation", None)
        record.update(chart_id=row.chart_id, audio_sha256=identity["audio_sha256"], unmarked_weight=0.)
        records.append(record)
        if progress:
            progress(dict(stage='annotations', completed=len(records), total=len(identities)))
    active_keys = {_key(r["source_path"], r["difficulty_index"]) for r in records}
    invalid.extend(r for r in old_records if _key(r.get("source_path", ""), r.get("difficulty_index", -1)) not in active_keys)
    if invalid:
        archive = root / "invalidated" / (_now().replace(":", "-") + ".json")
        _write_json(archive, dict(reason="source_or_audio_identity_changed_or_removed", records=invalid))
    _write_json(root / "quality_annotations.json", dict(schema="maichart-quality-annotations-v1", manifest=str(output), records=records))
    _write_json(root / "identity_cache.json", hash_cache)
    metadata.update(excluded=excluded, invalidated_records=len(invalid), updated_at=_now(),
                    source_manifest_sha256=audio_sha256(source), source_manifest_stat=_stat(source), scan_complete=False)
    _write_json(root / "pool.json", metadata)
    return prepared


def _stale(root, metadata, song, difficulty):
    identity = difficulty.get("candidate_identity", {})
    return (identity.get("source_stat") != _stat(song.get("maidata_path")) or
            identity.get("audio_stat") != _stat(song.get("audio_path")) or
            metadata.get("source_manifest_stat") != _stat(metadata["source_manifest"]))


def review_guard(manifest, chart_id, *, positive=True, row=None):
    manifest = Path(manifest).resolve()
    root = manifest.parent
    metadata = _read(root / "pool.json")
    if _read(root / 'scan_state.json').get('status') in {'queued', 'running'}:
        raise ValueError('候选库正在检查，请等待扫描完成后再保存评价')
    if metadata.get("schema") != SCHEMA or manifest != root / "training_manifest.json":
        raise ValueError("不是当前候选库清单")
    if not metadata.get("scan_complete"):
        raise ValueError("请先完成候选库扫描")
    if row is None:
        row = next((r for r in iter_catalog(manifest) if r.chart_id == chart_id), None)
    if row is None or row.chart_id != chart_id:
        raise ValueError("谱面身份已变化，请重新扫描候选库")
    entry = pool_charts(manifest).get(_resolved_slot(row.source_path, row.difficulty_index))
    if entry is not None:
        song, difficulty = entry
        identity = difficulty.get("candidate_identity", {})
        try:
            valid = (not _stale(root, metadata, song, difficulty)
                     and _chart_digest(row.source_path, row.difficulty_index) == row.chart_sha256 == identity.get("chart_sha256")
                     and _digest(row.audio_path) == identity.get("audio_sha256"))
        except (OSError, TypeError):
            valid = False
        if not valid:
            raise ValueError("源谱或音频已变化，标注身份已失效，请先同步")
        status = difficulty.get("candidate_status", "pending")
        if positive and status in {"error", "pending", "stale"}:
            raise ValueError("错误或未扫描谱面不能保存正向学习结论，请先修正源谱并同步")
        return dict(status=status, chart_id=chart_id, identity=identity)
    raise ValueError("候选谱不存在")


def _scan_job(job):
    """Isolated process: native chart analysis has process-global state."""
    from maichart.training import candidate_scan
    root, path, index, audio, identity = job
    root = Path(root)
    cache_identity = dict(identity, source_stat=None, audio_stat=None,
                          file_sha256=audio_sha256(path) if not identity.get('chart_sha256') and Path(path).is_file() else None,
                          policy=candidate_scan.POLICY, slot=index)
    cache_key = hashlib.sha256(json.dumps(cache_identity, sort_keys=True).encode()).hexdigest()
    cached = root / 'scan' / 'charts' / f'{cache_key}.json'
    result = _read(cached)
    hit = bool(result)
    if not hit:
        result = candidate_scan.scan_chart(path, index, audio, cache_dir=root / 'scan' / 'cache')
        _write_json(cached, result)
    return result, hit


def _scan_worker_init():
    import torch
    torch.set_num_threads(1)


def scan_pool(pool_dir, progress=None, *, workers=1, synchronize=True):
    from maichart.training import candidate_scan
    root = _pool(pool_dir)
    prepared = _synchronize(root, progress=progress) if synchronize else _read(root / 'training_manifest.json')
    metadata = _read(root / "pool.json")
    rows = {_key(r.source_path, r.difficulty_index): r for r in iter_catalog(root / "training_manifest.json")}
    results = []
    count = sum(len(s["difficulties"]) for s in prepared["songs"])
    work = [(song, difficulty) for song in prepared['songs'] for difficulty in song['difficulties']]
    cached_count = 0
    counts = dict(error=0, suspicious=0, ok=0)
    def report(**extra):
        if progress:
            progress(dict(stage='scanning', done=len(results), completed=len(results), total=count,
                          counts=dict(counts), cache_hits=cached_count, workers=max(1, workers), **extra))
    def attach(song, difficulty, result, hit):
        nonlocal cached_count
        cached_count += int(hit)
        index = int(difficulty.get('difficulty_index', difficulty.get('index')))
        identity = difficulty['candidate_identity']
        row = rows.get(_key(song['maidata_path'], index))
        result.update(chart_id=row.chart_id if row else None, song_id=song.get('song_id'), identity=identity)
        if difficulty.get('similar_reference'):
            result.setdefault('issues', []).append(dict(severity='suspicious', code='similar_reference',
                message='与同标题参考谱的事件集合高度相似，仅供人工核对，不自动排除或定性搬运',
                matches=difficulty['similar_reference']))
            result['similar_reference'] = difficulty['similar_reference']
            if result['status'] != 'error':
                result['status'] = 'suspicious'
        difficulty['candidate_status'] = result['status']
        counts[result['status']] += 1
        return result
    def job(song, difficulty):
        return (str(root), song['maidata_path'], int(difficulty.get('difficulty_index', difficulty.get('index'))),
                song.get('audio_path'), difficulty['candidate_identity'])
    report()
    if workers <= 1:
        for song, difficulty in work:
            result, hit = _scan_job(job(song, difficulty))
            results.append(attach(song, difficulty, result, hit))
            report(title=result.get('title') or song.get('title'))
    else:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
        # Process isolation avoids unsafe threads in native analysis; bound the
        # pending queue and restore source ordering before publishing results.
        ordered = {}
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=_scan_worker_init) as executor:
            pending = {}
            iterator = iter(enumerate(work))
            def fill():
                while len(pending) < workers * 2:
                    item = next(iterator, None)
                    if item is None:break
                    number, (song, difficulty) = item
                    pending[executor.submit(_scan_job, job(song, difficulty))] = (number, song, difficulty)
            fill()
            while pending:
                done, _ = wait(pending, timeout=1., return_when=FIRST_COMPLETED)
                if not done:report()
                for future in done:
                    number, song, difficulty = pending.pop(future)
                    result, hit = future.result()
                    ordered[number] = attach(song, difficulty, result, hit)
                    results.append(ordered[number])
                    report(title=result.get('title') or song.get('title'))
                fill()
        results = [ordered[number] for number in range(len(work))]
    report()
    if progress:progress(dict(stage='finalizing', completed=0, total=1, counts=counts))
    _write_json(root / "scan" / "results.json", dict(policy=candidate_scan.POLICY, charts=results))
    _write_json(root / "training_manifest.json", prepared)
    metadata.update(scan_policy=candidate_scan.POLICY, scan_complete=True, updated_at=_now())
    _write_json(root / "pool.json", metadata)
    return pool_status(root)


def pool_status(pool_dir, *, include_supervision=True):
    root = _pool(pool_dir)
    metadata = _read(root / "pool.json")
    if metadata.get("schema") != SCHEMA:
        raise ValueError("候选库不存在或格式不兼容")
    manifest = root / "training_manifest.json"
    records = _records(manifest)
    record_index = {}
    for record in records:
        record_index.setdefault(_resolved_slot(record.get("source_path") or "", record.get("difficulty_index", -1)), record)
    stat_cache = {}
    def stat_once(path):
        key = str(path)
        if key not in stat_cache:
            stat_cache[key] = _stat(path)
        return stat_cache[key]
    source_changed = metadata.get("source_manifest_stat") != stat_once(metadata.get("source_manifest"))
    counts = dict(charts=0, error=0, suspicious=0, ok=0, pending=0, stale=0, duplicate=len(metadata.get("excluded", [])))
    regions = []
    decisions = dict(learn_all=0, exclude=0, partial=0)
    timing_heads = layout_heads = 0.
    for key, (song, difficulty) in pool_charts(manifest).items():
        index = key[1]
        counts["charts"] += 1
        identity = difficulty.get("candidate_identity", {})
        stale = (source_changed or identity.get("source_stat") != stat_once(song.get("maidata_path")) or
                 identity.get("audio_stat") != stat_once(song.get("audio_path")))
        state = "stale" if stale else difficulty.get("candidate_status", "pending")
        counts[state] = counts.get(state, 0) + 1
        if stale:
            continue
        record = record_index.get(key)
        if (not record or record.get("chart_sha256") != identity.get("chart_sha256") or
                record.get("audio_sha256") != identity.get("audio_sha256")):
            continue
        valid_regions = _valid_regions(record)
        if record.get('label') in {'excluded','negative'} or record.get('weight',1.) <= 0:
            decisions['exclude'] += 1
        elif record.get('candidate_chart_review') == 'learn_all':
            decisions['learn_all'] += 1
        elif valid_regions:
            decisions['partial'] += 1
        regions.extend(valid_regions)
        if (not include_supervision or not metadata.get("scan_complete") or not valid_regions or
                difficulty.get("candidate_status") not in {"ok", "suspicious"}):
            continue
        try:
            weighted_regions = []
            for region in record.get("regions", []):
                t, c = zero_unmarked_region_weights(region) if region in valid_regions else (0., 0.)
                weighted_regions.append((float(region["start_sec"]), float(region["end_sec"]), max(0., t), max(0., c)))
            weight = max(0., float(record.get("weight", 1.)))
            for sec, count in _head_events(song["maidata_path"], index):
                region = next((r for r in reversed(weighted_regions) if r[0] <= sec < r[1]), None)
                if region:
                    timing_heads += region[2] * count * weight
                    layout_heads += region[3] * count * weight
        except (OSError, ValueError, UnicodeError):
            pass
    result = dict(metadata, pool_dir=str(root), manifest=str(root / "training_manifest.json"), counts=counts, decisions=decisions,
                stale=bool(counts["stale"]), confirmed_regions=len(regions),
                reviewable_charts=counts["ok"] + counts["suspicious"] if metadata.get("scan_complete") else 0,
                timing_regions=sum(zero_unmarked_region_weights(r)[0] > 0 for r in regions),
                layout_regions=sum(zero_unmarked_region_weights(r)[1] > 0 for r in regions))
    if include_supervision:
        result["supervision"] = dict(unit="weighted_note_heads_approximation",
                                     timing_note_heads=timing_heads, layout_note_heads=layout_heads)
    return result


AUTO_SCREEN_DEFER_CODES = frozenset({
    "global_offset",
    "tempo_drift",
    "alignment_uncertain",
    "alignment_unjudgeable",
    "alignment_failed",
    "action_after_audio",
    "similar_reference",
    "timeline_failed",
})


def _has_manual_candidate_decision(record: dict | None) -> bool:
    if not isinstance(record, dict) or record.get("auto_screened"):
        return False
    if record.get("candidate_chart_review") in {"learn_all", "exclude"}:
        return True
    if record.get("candidate_reviews") or record.get("regions"):
        return True
    if record.get("label") in {"excluded", "negative"}:
        return True
    return float(record.get("weight", 1.0)) != 1.0


_AUTO_SCREEN_CONFLICT_LOCK = threading.Lock()
_AUTO_SCREEN_CONFLICT_CACHE: dict[tuple, tuple[str, dict | None]] = {}


def _auto_screen_conflict_worker(task: tuple[str, int, bool, float]) -> tuple[str, dict | None]:
    """Top-level worker for parallel dialect + MiaCode conflict extraction on uncached charts."""
    from fractions import Fraction
    import numpy as np
    from maichart.composer_v2 import chart_dialect
    from maichart.event_timing import EventTempo, parse_event_chart
    from maichart.generation.post_process import PostProcessConfig, detect_chart_conflicts
    from maichart.tempo_clock import TempoClock

    source_path, slot_index, detect_soft, tap_on_slide_ms = task
    pp_config = PostProcessConfig(
        enable_conflict_detection=True,
        detect_hard_conflicts=True,
        detect_soft_conflicts=bool(detect_soft),
        conflict_engine="miacode",
        tap_on_slide_threshold_ms=float(tap_on_slide_ms),
    )
    try:
        raw_chart = parse_event_chart(source_path, int(slot_index), strict_notes=False)
        norm_chart, dialect_edits = chart_dialect.normalize_training_chart(raw_chart)
        if not norm_chart.points:
            return ("error", None)
        changes = {c.beat: c.bpm for c in norm_chart.tempo_changes}
        tempo = [EventTempo(b, v) for b, v in sorted(changes.items())]
        clock = TempoClock(tempo, norm_chart.audio_first)
        points = list(norm_chart.points)
        point_secs = clock.at_floats(np.fromiter((float(p.beat) for p in points), dtype=np.float64, count=len(points)))
        end_beat_val = float(Fraction(str(norm_chart.end_beat))) if norm_chart.end_beat is not None else 0.0
        total_beats = max(end_beat_val, max(float(p.beat) for p in points) + 1.0)
        det = detect_chart_conflicts(points, tempo, norm_chart.audio_first, pp_config)
    except (OSError, ValueError, KeyError, ZeroDivisionError, OverflowError, UnicodeError):
        return ("error", None)

    if det.coverage.get("status") == "partial":
        if det.coverage.get("parse_errors"):
            return ("error", None)
        return ("defer_partial", None)
    if dialect_edits.get("unsupported", 0) > 0:
        return ("defer_unsupported", None)

    sustain_max = max(
        (float(a.get("end_sec", 0.0)) for a in (det.coverage.get("sustained_actions") or ())),
        default=0.0,
    )
    conflicts_data = [
        dict(
            kind=c.conflict_type,
            severity=str(c.severity),
            time_sec=float(c.time),
            notes=[dict(beat=str(n.beat), raw=n.raw) for n in (c.notes or ())[:6]],
        )
        for c in det.conflicts_detected
    ]
    return (
        "ok",
        dict(
            conflicts=conflicts_data,
            tempo=[[str(t.beat), float(t.bpm)] for t in tempo],
            audio_first=float(norm_chart.audio_first),
            point_secs=[round(float(x), 6) for x in point_secs],
            total_beats=float(total_beats),
            sustain_max=float(sustain_max),
        ),
    )


def auto_screen_pool(
    pool_dir,
    *,
    difficulty_slots=(5, 6),
    min_level: float = 12.0,
    max_level: float = 15.0,
    tap_on_slide_mode: str = "mask_hard",
    tap_on_slide_ms: float = 200.0,
    inner_muri_mode: str = "allow",
    max_offset_ms: float = 35.0,
    max_problem_fraction: float = 0.35,
    preserve_manual: bool = True,
    dry_run: bool = False,
):
    """Batch-screen a scanned candidate pool into clean whole charts, Muri-masked charts, deferred, and excluded."""
    from collections import Counter
    from concurrent.futures import ProcessPoolExecutor
    from fractions import Fraction
    from types import SimpleNamespace
    import numpy as np
    from maichart.composer_v2 import chart_dialect
    from maichart.composer_v2.auto_quality_audit import (
        INNER_MURI_MODES,
        TAP_ON_SLIDE_MODES,
        _causal_conflict_beat_span,
        _merge_intervals,
        _should_mask_issue,
    )
    from maichart.composer_v2.difficulty import DIFFICULTY_SLOTS
    from maichart.event_timing import EventTempo, parse_event_chart
    from maichart.tempo_clock import TempoClock
    from maichart.training.event_dataset import numeric_level
    from maichart.training.quality_annotations import _subtract_intervals, save_annotations

    if tap_on_slide_mode not in TAP_ON_SLIDE_MODES:
        raise ValueError(f"tap_on_slide_mode 须为 {TAP_ON_SLIDE_MODES}")
    if inner_muri_mode not in INNER_MURI_MODES:
        raise ValueError(f"inner_muri_mode 须为 {INNER_MURI_MODES}")
    if not isinstance(difficulty_slots, (list, tuple)) or not difficulty_slots or any(
        type(s) is not int or s not in DIFFICULTY_SLOTS for s in difficulty_slots
    ):
        raise ValueError("请选择有效的难度槽位（4 EXPERT、5 MASTER、6 Re:MASTER）")
    slots_set = frozenset(difficulty_slots)
    if not (1.0 <= float(min_level) <= float(max_level) <= 15.0):
        raise ValueError("定数范围须满足 1 ≤ 最低定数 ≤ 最高定数 ≤ 15")
    if not (10.0 <= float(max_offset_ms) <= 60.0):
        raise ValueError("最大允许声学偏移须在 10–60 ms 之间")
    if not (0.05 <= float(max_problem_fraction) <= 1.0):
        raise ValueError("局部无理屏蔽比例上限须在 0.05–1.0 之间")

    root = _pool(pool_dir)
    manifest = root / "training_manifest.json"
    metadata = _read(root / "pool.json")
    if metadata.get("schema") != SCHEMA:
        raise ValueError("候选库不存在或格式不兼容")
    if _read(root / "scan_state.json").get("status") in {"queued", "running"}:
        raise ValueError("候选库正在扫描中，请等待扫描完成后再自动筛选")
    if not metadata.get("scan_complete") or not (root / "scan" / "results.json").is_file():
        raise ValueError("请先完成候选库扫描后再执行自动筛选")
    if metadata.get("source_manifest_stat") != _stat(metadata.get("source_manifest")):
        raise ValueError("来源清单已变化，请先重新同步检查候选库")

    charts_map = pool_charts(manifest)
    raw_scan = _read(root / "scan" / "results.json")
    scanned_map = {
        _resolved_slot(item["source_path"], item["difficulty_index"]): item
        for item in raw_scan.get("charts", ())
        if item.get("source_path") and item.get("difficulty_index") is not None
    }
    existing_records = _records(manifest)
    existing_by_key = {}
    for rec in existing_records:
        existing_by_key.setdefault(_resolved_slot(rec.get("source_path") or "", rec.get("difficulty_index", -1)), rec)

    if len(existing_by_key) >= len(charts_map) and charts_map:
        work_items = []
        for key, (song, difficulty) in charts_map.items():
            rec = existing_by_key.get(key)
            info = scanned_map.get(key) or {}
            if rec is None:
                continue
            work_items.append((key, song, difficulty, rec, info))
    else:
        work_items = []
        for row in iter_catalog(manifest):
            key = _resolved_slot(row.source_path, row.difficulty_index)
            song, difficulty = charts_map.get(key, ({}, {}))
            rec = existing_by_key.get(key) or row.to_dict()
            rec.setdefault("chart_id", row.chart_id)
            info = scanned_map.get(key) or {}
            work_items.append((key, song, difficulty, rec, info))

    detect_soft = tap_on_slide_mode == "mask_all"
    can_reuse_scan_global = (not detect_soft) and abs(float(tap_on_slide_ms) - 200.0) < 1e-6

    def _dict_to_issue(entry: dict):
        return SimpleNamespace(
            conflict_type=str(entry.get("kind") or ""),
            severity=str(entry.get("severity") or "hard"),
            time=float(entry.get("time_sec", 0.0)),
            notes=[
                SimpleNamespace(beat=Fraction(str(n["beat"])), raw=str(n.get("raw") or ""))
                for n in (entry.get("notes") or ())
                if n.get("beat") is not None
            ],
        )

    # Pre-collect any charts that cannot reuse scan/results.json conflict records and evaluate them in parallel.
    pending_eval_keys: list[tuple] = []
    pending_eval_tasks: list[tuple[str, int, bool, float]] = []
    seen_eval_keys: set[tuple] = set()
    for key, song, difficulty, base_source, info in work_items:
        slot_index = int(key[1])
        prior = existing_by_key.get(key)
        if preserve_manual and _has_manual_candidate_decision(prior):
            continue
        if slot_index not in slots_set:
            continue
        raw_lvl = base_source.get("level") or difficulty.get("level_raw") or difficulty.get("level") or info.get("level")
        try:
            lvl = float(numeric_level(raw_lvl))
        except Exception:
            lvl = -1.0
        if not (float(min_level) <= lvl <= float(max_level)):
            continue
        cand_status = difficulty.get("candidate_status") or info.get("status") or "pending"
        if cand_status in {"error", "pending", "stale"}:
            continue
        issue_codes = {str(item.get("code") or "") for item in (info.get("issues") or ())}
        align = info.get("alignment") or {}
        if (issue_codes & AUTO_SCREEN_DEFER_CODES) or (
            align.get("judgeable") and align.get("confident") and abs(float(align.get("offset_ms", 0.0))) > float(max_offset_ms)
        ) or ("small_offset" in issue_codes and not align and float(max_offset_ms) < 40.0):
            continue
        can_reuse_this = can_reuse_scan_global and not (
            issue_codes & {"conflict_check_failed", "nonstandard_notation", "unsupported_notes"}
        )
        stats = info.get("stats") or {}
        fast_end = max(
            float(stats.get("last_action_end_sec") or 0.0),
            float(stats.get("last_onset_sec") or 0.0),
        )
        if can_reuse_this and fast_end > 0.0:
            continue
        src_path = str(base_source.get("source_path") or song.get("maidata_path") or "")
        chart_sha = str(
            (difficulty.get("candidate_identity") or {}).get("chart_sha256")
            or base_source.get("chart_sha256")
            or ""
        )
        cache_key = (_resolved_slot(src_path, slot_index), chart_sha, detect_soft, round(float(tap_on_slide_ms), 3))
        if cache_key in seen_eval_keys:
            continue
        with _AUTO_SCREEN_CONFLICT_LOCK:
            if cache_key in _AUTO_SCREEN_CONFLICT_CACHE:
                continue
        seen_eval_keys.add(cache_key)
        pending_eval_keys.append(cache_key)
        pending_eval_tasks.append((src_path, slot_index, detect_soft, float(tap_on_slide_ms)))

    if pending_eval_tasks:
        import sys
        from concurrent.futures.process import BrokenProcessPool

        main_file = getattr(sys.modules.get("__main__"), "__file__", None)
        can_spawn = bool(main_file and not str(main_file).endswith("<stdin>") and Path(main_file).is_file())
        workers = min(6, os.cpu_count() or 1, len(pending_eval_tasks))
        if can_spawn and workers >= 2 and len(pending_eval_tasks) >= 8:
            try:
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    eval_results = list(pool.map(_auto_screen_conflict_worker, pending_eval_tasks, chunksize=8))
            except (BrokenProcessPool, OSError):
                eval_results = [_auto_screen_conflict_worker(t) for t in pending_eval_tasks]
        else:
            eval_results = [_auto_screen_conflict_worker(t) for t in pending_eval_tasks]
        with _AUTO_SCREEN_CONFLICT_LOCK:
            for c_key, c_val in zip(pending_eval_keys, eval_results):
                if len(_AUTO_SCREEN_CONFLICT_CACHE) >= 16384:
                    _AUTO_SCREEN_CONFLICT_CACHE.clear()
                _AUTO_SCREEN_CONFLICT_CACHE[c_key] = c_val

    counts = Counter(
        total_charts=0,
        preserved_manual=0,
        approved_clean=0,
        approved_masked=0,
        deferred_review=0,
        excluded_slot=0,
        excluded_level=0,
        excluded_error=0,
        excluded_muri=0,
        masked_regions=0,
    )
    deferred_by_reason: Counter[str] = Counter()
    approved_by_slot: Counter[int] = Counter()
    updated_records = []

    for key, song, difficulty, base_source, info in work_items:
        counts["total_charts"] += 1
        slot_index = int(key[1])
        prior = existing_by_key.get(key)
        identity = difficulty.get("candidate_identity", {})
        audio_sha = identity.get("audio_sha256") or (prior or {}).get("audio_sha256") or base_source.get("audio_sha256")

        if preserve_manual and _has_manual_candidate_decision(prior):
            updated_records.append(copy.deepcopy(prior))
            counts["preserved_manual"] += 1
            if _valid_regions(prior):
                approved_by_slot[slot_index] += 1
            continue

        base_rec = dict(
            base_source,
            difficulty_index=slot_index,
            audio_sha256=audio_sha,
            unmarked_weight=0.0,
            auto_screened=True,
        )
        base_rec.pop("annotation", None)
        base_rec.pop("annotated", None)
        base_rec.pop("candidate_reviews", None)

        if slot_index not in slots_set:
            base_rec.update(
                label="excluded",
                weight=0.0,
                candidate_chart_review="exclude",
                reason="difficulty_slot_excluded",
                regions=[],
            )
            updated_records.append(base_rec)
            counts["excluded_slot"] += 1
            continue

        raw_lvl = base_rec.get("level") or difficulty.get("level_raw") or difficulty.get("level") or info.get("level")
        try:
            lvl = float(numeric_level(raw_lvl))
        except Exception:
            lvl = -1.0
        if not (float(min_level) <= lvl <= float(max_level)):
            base_rec.update(
                label="excluded",
                weight=0.0,
                candidate_chart_review="exclude",
                reason="level_out_of_range",
                regions=[],
            )
            updated_records.append(base_rec)
            counts["excluded_level"] += 1
            continue

        cand_status = difficulty.get("candidate_status") or info.get("status") or "pending"
        if cand_status in {"error", "pending", "stale"}:
            base_rec.update(
                label="excluded",
                weight=0.0,
                candidate_chart_review="exclude",
                reason="scan_error",
                regions=[],
            )
            updated_records.append(base_rec)
            counts["excluded_error"] += 1
            continue

        issue_codes = {str(item.get("code") or "") for item in (info.get("issues") or ())}
        align = info.get("alignment") or {}
        defer_reason = None
        matched_defer = sorted(issue_codes & AUTO_SCREEN_DEFER_CODES)
        if matched_defer:
            defer_reason = matched_defer[0]
        elif align.get("judgeable") and align.get("confident") and abs(float(align.get("offset_ms", 0.0))) > float(max_offset_ms):
            defer_reason = "offset_exceeds_threshold"
        elif "small_offset" in issue_codes and not align and float(max_offset_ms) < 40.0:
            defer_reason = "small_offset"

        if defer_reason is not None:
            base_rec.update(
                label="normal",
                weight=1.0,
                reason=f"deferred:{defer_reason}",
                regions=[],
            )
            base_rec.pop("candidate_chart_review", None)
            updated_records.append(base_rec)
            counts["deferred_review"] += 1
            deferred_by_reason[defer_reason] += 1
            continue

        stats = info.get("stats") or {}
        fast_end = max(
            float(stats.get("last_action_end_sec") or 0.0),
            float(stats.get("last_onset_sec") or 0.0),
        )
        can_reuse_this = can_reuse_scan_global and not (
            issue_codes & {"conflict_check_failed", "nonstandard_notation", "unsupported_notes"}
        )

        if can_reuse_this and fast_end > 0.0:
            raw_scan_conflicts = list(info.get("conflicts") or ()) + list(info.get("inner_slide_conflicts") or ())
            active_issues = [
                iss for entry in raw_scan_conflicts
                if _should_mask_issue(
                    iss := _dict_to_issue(entry),
                    tap_on_slide_mode=tap_on_slide_mode,
                    inner_muri_mode=inner_muri_mode,
                )
            ]
            if not active_issues:
                end_sec = round(fast_end + 0.25, 6)
                base_rec.update(
                    label="normal",
                    weight=1.0,
                    candidate_chart_review="learn_all",
                    reason="auto_screen_clean",
                    regions=[
                        dict(
                            start_sec=0.0,
                            end_sec=end_sec,
                            label="good",
                            reason="auto_screen_clean",
                            weight=1.0,
                            timing_weight=1.0,
                            component_weight=1.0,
                            learn_entry=True,
                            entry_start_sec=0.0,
                        )
                    ],
                )
                updated_records.append(base_rec)
                counts["approved_clean"] += 1
                approved_by_slot[slot_index] += 1
                continue
            try:
                raw_chart = parse_event_chart(base_rec["source_path"], slot_index, strict_notes=False)
                norm_chart, dialect_edits = chart_dialect.normalize_training_chart(raw_chart)
                if not norm_chart.points:
                    raise ValueError("empty_after_dialect")
                if dialect_edits.get("unsupported", 0) > 0:
                    base_rec.update(label="normal", weight=1.0, reason="deferred:unsupported_dialect_atoms", regions=[])
                    base_rec.pop("candidate_chart_review", None)
                    updated_records.append(base_rec)
                    counts["deferred_review"] += 1
                    deferred_by_reason["unsupported_dialect_atoms"] += 1
                    continue
                changes = {c.beat: c.bpm for c in norm_chart.tempo_changes}
                tempo = [EventTempo(b, v) for b, v in sorted(changes.items())]
                clock = TempoClock(tempo, norm_chart.audio_first)
                points = list(norm_chart.points)
                point_secs = clock.at_floats(np.fromiter((float(p.beat) for p in points), dtype=np.float64, count=len(points)))
                end_beat_val = float(Fraction(str(norm_chart.end_beat))) if norm_chart.end_beat is not None else 0.0
                total_beats = max(end_beat_val, max(float(p.beat) for p in points) + 1.0)
                chart_end = round(max(float(point_secs.max()), fast_end) + 0.25, 6)
            except (OSError, ValueError, KeyError, ZeroDivisionError, OverflowError, UnicodeError):
                base_rec.update(
                    label="excluded",
                    weight=0.0,
                    candidate_chart_review="exclude",
                    reason="invalid_sustain_syntax",
                    regions=[],
                )
                updated_records.append(base_rec)
                counts["excluded_error"] += 1
                continue
        else:
            src_path = str(base_rec.get("source_path") or song.get("maidata_path") or "")
            chart_sha = str(identity.get("chart_sha256") or base_rec.get("chart_sha256") or "")
            cache_key = (_resolved_slot(src_path, slot_index), chart_sha, detect_soft, round(float(tap_on_slide_ms), 3))
            with _AUTO_SCREEN_CONFLICT_LOCK:
                status_tag, payload_data = _AUTO_SCREEN_CONFLICT_CACHE.get(cache_key, ("error", None))
            if status_tag == "error" or payload_data is None:
                base_rec.update(
                    label="excluded",
                    weight=0.0,
                    candidate_chart_review="exclude",
                    reason="invalid_sustain_syntax",
                    regions=[],
                )
                updated_records.append(base_rec)
                counts["excluded_error"] += 1
                continue
            if status_tag == "defer_partial":
                base_rec.update(label="normal", weight=1.0, reason="deferred:partial_coverage", regions=[])
                base_rec.pop("candidate_chart_review", None)
                updated_records.append(base_rec)
                counts["deferred_review"] += 1
                deferred_by_reason["partial_coverage"] += 1
                continue
            if status_tag == "defer_unsupported":
                base_rec.update(label="normal", weight=1.0, reason="deferred:unsupported_dialect_atoms", regions=[])
                base_rec.pop("candidate_chart_review", None)
                updated_records.append(base_rec)
                counts["deferred_review"] += 1
                deferred_by_reason["unsupported_dialect_atoms"] += 1
                continue

            tempo = [EventTempo(Fraction(str(b)), float(v)) for b, v in payload_data["tempo"]]
            clock = TempoClock(tempo, payload_data["audio_first"])
            point_secs = np.asarray(payload_data["point_secs"], dtype=np.float64)
            total_beats = float(payload_data["total_beats"])
            chart_end = round(max(float(point_secs.max()), fast_end, float(payload_data["sustain_max"])) + 0.25, 6)
            active_issues = [
                iss for entry in payload_data["conflicts"]
                if _should_mask_issue(
                    iss := _dict_to_issue(entry),
                    tap_on_slide_mode=tap_on_slide_mode,
                    inner_muri_mode=inner_muri_mode,
                )
            ]

        raw_muri: list[tuple[float, float]] = []
        for conflict in active_issues:
            b_lo, b_hi = _causal_conflict_beat_span(conflict, clock, total_beats)
            s_lo = max(0.0, float(clock.at(Fraction(str(round(b_lo, 4))))))
            s_hi = min(chart_end, float(clock.at(Fraction(str(round(b_hi, 4))))))
            if s_hi > s_lo + 0.02:
                raw_muri.append((s_lo, s_hi))

        merged_muri = [
            (round(max(0.0, lo), 6), round(min(chart_end, hi), 6))
            for lo, hi in _merge_intervals(raw_muri, merge_gap=0.08)
            if round(min(chart_end, hi), 6) > round(max(0.0, lo), 6) + 0.02
        ]

        if not merged_muri:
            base_rec.update(
                label="normal",
                weight=1.0,
                candidate_chart_review="learn_all",
                reason="auto_screen_clean",
                regions=[
                    dict(
                        start_sec=0.0,
                        end_sec=chart_end,
                        label="good",
                        reason="auto_screen_clean",
                        weight=1.0,
                        timing_weight=1.0,
                        component_weight=1.0,
                        learn_entry=True,
                        entry_start_sec=0.0,
                    )
                ],
            )
            updated_records.append(base_rec)
            counts["approved_clean"] += 1
            approved_by_slot[slot_index] += 1
            continue

        masked_sec = sum(hi - lo for lo, hi in merged_muri)
        active_duration = max(1.0, chart_end - float(point_secs.min()))
        if masked_sec / active_duration > float(max_problem_fraction):
            base_rec.update(
                label="excluded",
                weight=0.0,
                candidate_chart_review="exclude",
                reason="auto_excessive_muri",
                regions=[],
            )
            updated_records.append(base_rec)
            counts["excluded_muri"] += 1
            continue

        clean_spans = [
            (round(lo, 6), round(hi, 6))
            for lo, hi in _subtract_intervals(0.0, chart_end, merged_muri)
            if round(hi, 6) - round(lo, 6) >= 0.05
        ]
        has_clean_head = any(
            any(lo <= float(sec) < hi for sec in point_secs)
            for lo, hi in clean_spans
        )
        if not clean_spans or not has_clean_head:
            base_rec.update(
                label="excluded",
                weight=0.0,
                candidate_chart_review="exclude",
                reason="auto_excessive_muri",
                regions=[],
            )
            updated_records.append(base_rec)
            counts["excluded_muri"] += 1
            continue

        regions = []
        for lo, hi in clean_spans:
            reg = dict(
                start_sec=lo,
                end_sec=hi,
                label="good",
                reason="auto_screen_clean_span",
                weight=1.0,
                timing_weight=1.0,
                component_weight=1.0,
            )
            if lo <= 1e-6:
                reg["learn_entry"] = True
                reg["entry_start_sec"] = 0.0
            regions.append(reg)
        for lo, hi in merged_muri:
            regions.append(
                dict(
                    start_sec=lo,
                    end_sec=hi,
                    label="problem",
                    reason="auto_hard_muri",
                    weight=1.0,
                    timing_weight=0.0,
                    component_weight=0.0,
                )
            )
        regions.sort(key=lambda r: (r["start_sec"], r["end_sec"]))
        base_rec.update(
            label="normal",
            weight=1.0,
            reason="auto_segment_muri_mask",
            regions=regions,
        )
        base_rec.pop("candidate_chart_review", None)
        updated_records.append(base_rec)
        counts["approved_masked"] += 1
        counts["masked_regions"] += len(merged_muri)
        approved_by_slot[slot_index] += 1

    manual_entering = sum(
        1 for rec in updated_records
        if not rec.get("auto_screened") and bool(_valid_regions(rec))
    )
    summary = dict(
        policy="candidate_auto_screen_v1",
        dry_run=bool(dry_run),
        settings=dict(
            difficulty_slots=sorted(slots_set),
            min_level=float(min_level),
            max_level=float(max_level),
            tap_on_slide_mode=tap_on_slide_mode,
            tap_on_slide_ms=float(tap_on_slide_ms),
            inner_muri_mode=inner_muri_mode,
            max_offset_ms=float(max_offset_ms),
            max_problem_fraction=float(max_problem_fraction),
            preserve_manual=bool(preserve_manual),
        ),
        counts=dict(counts),
        deferred_by_reason=dict(sorted(deferred_by_reason.items())),
        approved_by_slot={str(k): v for k, v in sorted(approved_by_slot.items())},
        total_entering_training=int(counts["approved_clean"] + counts["approved_masked"] + manual_entering),
        manual_entering_training=int(manual_entering),
        updated_at=_now(),
    )

    if not dry_run:
        save_annotations(manifest, updated_records)
        metadata["auto_screen"] = {k: v for k, v in summary.items() if k != "dry_run"}
        metadata["updated_at"] = summary["updated_at"]
        _write_json(root / "pool.json", metadata)
        summary["pool_status"] = pool_status(root, include_supervision=False)
    return summary


def _review_choices(scanned, record):
    result = scanned.info
    spans = record.get("candidate_reviews", []) + record.get("regions", [])
    if not spans and scanned.cached_choices is not None:
        return scanned.cached_choices
    segments = [dict(zip(scanned.fields, values)) for values in scanned.segments]
    if spans:
        eligible = []
        for segment in segments:
            a, b = segment["start_sec"], segment["end_sec"]
            covered = sorted((max(a, float(r["start_sec"])), min(b, float(r["end_sec"])))
                             for r in spans
                             if float(r["end_sec"]) > a and float(r["start_sec"]) < b)
            total, end = 0., a
            for left, right in covered:
                total += max(0., right - max(end, left)); end = max(end, right)
            if total < .5 * (b - a):
                eligible.append(segment)
        segments = eligible
    if not segments:
        return ()
    rates = sorted(s.get("onset_rate") or 0 for s in segments)
    median = rates[len(rates) // 2]
    representative = min(segments, key=lambda s: (abs((s.get("onset_rate") or 0) - median), -(s.get("attack_support") or 0)))
    risk = max(segments, key=lambda s: (s.get("conflicts") or 0, bool(s.get("mixed_16_24")), s.get("onset_rate") or 0))
    rng = random.Random(result["chart_id"])
    choices, seen = [], set()
    for segment, why in ((representative, "代表性片段"), (risk, "复杂或风险片段"), (rng.choice(segments), "随机抽查")):
        if segment["start_sec"] in seen:
            continue
        seen.add(segment["start_sec"])
        choices.append(dict(segment, chart_id=result["chart_id"], source_path=result["source_path"],
                            difficulty_index=result["difficulty_index"], title=result.get("title"), why=why,
                            candidate_status=result["status"], issues=result.get("issues", []),
                            context_start_sec=max(0, segment["start_sec"] - 4),
                            context_end_sec=min(result.get("audio_duration_sec") or segment["end_sec"] + 4, segment["end_sec"] + 4)))
    if not spans:
        scanned.cached_choices = tuple(choices)
    return choices


def review_queue(pool_dir, offset=0, limit=1, *, status=None, exclude_chart_id=None,
                 prefer_chart_id=None, seen_segments=()):
    root = _pool(pool_dir)
    manifest = root / "training_manifest.json"
    metadata = _read(root / "pool.json")
    seen_segments = set(seen_segments)
    offset, limit = max(0, int(offset)), min(100, max(1, int(limit)))
    source_changed = metadata.get("source_manifest_stat") != _stat(metadata.get("source_manifest"))
    scan_state = _read(root / "scan_state.json").get("status")
    scanning = scan_state in {"queued", "running"}
    if not metadata.get("scan_complete") or scanning or source_changed or not (root / "scan" / "results.json").is_file():
        return dict(items=[], total=0, offset=offset, limit=limit, stale=source_changed,
                    next_offset=offset, scan_complete=bool(metadata.get("scan_complete")) and not scanning)
    charts = pool_charts(manifest)
    records = {}
    for record in _records(manifest):
        records.setdefault(record.get("chart_id"), record)
    groups = {}
    preferred = []
    stats = {}
    ranked = {}
    for key, scanned in scan_index(manifest).items():
        result = scanned.info
        chart_id = result.get("chart_id")
        entry = charts.get(key)
        if entry is None:
            continue
        if not chart_id or chart_id == exclude_chart_id or result["status"] == "error" or (status and result["status"] != status):
            continue
        record = records.get(chart_id, {})
        if (record.get("label") in {"excluded", "negative"} or record.get("weight", 1.) <= 0
                or record.get("candidate_chart_review") in {"learn_all", "exclude"}):
            continue
        choices = [choice for choice in _review_choices(scanned, record)
                   if (chart_id, choice["start_sec"], choice["end_sec"]) not in seen_segments]
        if not choices:
            continue
        ranked[chart_id] = entry
        if chart_id == prefer_chart_id:
            preferred = choices
            continue
        try:
            level = float(str(result.get("level", "0")).replace("+", ".7"))
        except ValueError:
            level = 0
        bucket = (int(level // 3), int(result["bpm"] // 40))
        groups.setdefault(bucket, []).append((hashlib.sha256(chart_id.encode()).digest(), choices))
    buckets = [deque(sorted(rows, key=lambda row: row[0])) for _, rows in sorted(groups.items())]
    items = list(preferred)
    while any(buckets):
        for rows in buckets:
            if rows:
                items.extend(rows.popleft()[1])
    page = []
    stale_ids = set()
    next_offset = offset
    for next_offset, item in enumerate(items[offset:], start=offset + 1):
        chart_id = item["chart_id"]
        if chart_id not in stats:
            song, difficulty = ranked[chart_id]
            stats[chart_id] = _stale(root, metadata, song, difficulty)
        if stats[chart_id]:
            stale_ids.add(chart_id)
            continue
        page.append(item)
        if len(page) == limit:
            break
    return dict(items=page, total=len(items), offset=offset, limit=limit,
                stale=bool(stale_ids), next_offset=next_offset)


@lru_cache(maxsize=16384)
def _cached_chart_digest(path, index, signature):
    return chart_sha256(path, index)


def _chart_digest(path, index):
    return _cached_chart_digest(str(path), int(index), tuple(_stat(path) or ()))


@lru_cache(maxsize=16384)
def _cached_heads(path, index, signature):
    # Use playback's exact parser/head semantics, without building preview tails,
    # slide geometry, inverse beat maps or ruler marks for every single note.
    from maichart.event_timing import EventTempo, parse_event_chart
    from maichart.generation.muri.parser import _parse_one, _reference_data
    from maichart.notes import expand_compact_taps, _split_top_level_slashes
    from maichart.tempo_clock import TempoClock
    from maichart.webui.chart_timeline import _unknown_head
    chart = parse_event_chart(path, index, strict_notes=False)
    changes = {change.beat: change.bpm for change in chart.tempo_changes}
    clock = TempoClock([EventTempo(beat, bpm) for beat, bpm in sorted(changes.items())], chart.audio_first)
    try:
        refs = _reference_data()
    except (OSError, ValueError, RuntimeError):
        refs = {}
    from array import array
    heads = array("d")
    for point_index, point in enumerate(chart.points):
        second = clock.at(point.beat)
        for atom_index, raw in enumerate(_split_top_level_slashes(point.note_text)):
            if not raw:
                continue
            for atom in expand_compact_taps(raw):
                try:
                    markers = _parse_one(atom, point_index + 1, atom_index + 1, second, 0, clock.bpm(point.beat), refs)
                    if not markers:
                        raise ValueError('Unsupported note syntax')
                except (ValueError, TypeError, KeyError, IndexError, OverflowError) as exc:
                    if _unknown_head(raw, point.beat, second, '', str(exc)) is not None:
                        heads.append(float(second))
                    continue
                for branch, marker in enumerate(markers):
                    slide = marker['type'] in {'wifi', 'slide'}
                    if not slide or (not branch and marker['has_head_star']):
                        heads.append(float(marker['second']))
    if tuple(_stat(path) or ()) != signature:
        raise ValueError('谱面文件已改变，请重新估算监督占比')
    return heads


def _head_events(path, index):
    seconds = _cached_heads(str(path), int(index), tuple(_stat(path) or ()))
    return ((second, 1) for second in seconds)


def _budget_charts(manifest, payload):
    """Only budget inputs need paths; absolute metadata requires no file probe.

    Canonical identity matching still uses _key below whenever reviews or an
    accepted-owner filter exist, including aliases through symbolic links.
    """
    from maichart.portable_paths import material_path
    from maichart.training.dataset_catalog import _source_asset
    for original in payload.get("songs", []):
        song = dict(original)
        for key in ("maidata_path", "audio_path"):
            if song.get(key):
                path = material_path(str(song[key]))
                song[key] = str(path if path.is_absolute() else _source_asset(song[key], manifest, payload))
        for difficulty in song.get("difficulties", []):
            index = difficulty.get("difficulty_index", difficulty.get("index"))
            if index is not None:
                yield song, difficulty, int(index)


def supervision_budget(manifests, *, _accepted=None, progress=None, mechanics_manifests=()):
    """Estimate weighted heads; accepted owners are supplied by dataset composition.

    Standalone estimation keeps each source's occurrences, as before. Composition
    passes _accepted to count only the selected owner of duplicate charts.
    """
    sources, warnings = [], []
    mechanics = {Path(path).resolve() for path in mechanics_manifests}
    loaded = []
    if progress:
        progress(dict(completed=0, total=0, stage="loading", source=""))
    for manifest in manifests:
        manifest = Path(manifest).resolve()
        payload = _read(manifest)
        charts = list(_budget_charts(manifest, payload))
        loaded.append((manifest, payload, charts))
    total = sum(len(charts) for _, _, charts in loaded)
    completed = 0
    for manifest, payload, charts in loaded:
        candidate = bool(payload.get("candidate_pool"))
        records = _records(manifest)
        record_index = {}
        for row in records:
            if candidate and not _valid_regions(row):
                continue
            record_index.setdefault(_key(row.get("source_path", ""), row.get("difficulty_index", -1)), []).append(row)
        metadata = _read(manifest.with_name("pool.json")) if candidate else None
        timing = layout = 0.
        chart_count = 0
        mechanics_count = 0
        if candidate and not record_index:
            completed += len(charts)
            if progress:
                progress(dict(completed=completed, total=total, stage="estimating", source=str(manifest)))
            sources.append(dict(manifest=str(manifest), candidate_pool=True, charts=0,
                                timing_note_heads=0., layout_note_heads=0.))
            continue
        for song, difficulty, index in charts:
            completed += 1
            if progress:
                progress(dict(completed=completed - 1, total=total, stage="estimating", source=str(manifest)))
            if _accepted is not None and (str(manifest), _key(song["maidata_path"], index)) not in _accepted:
                continue
            matches = record_index.get(_key(song["maidata_path"], index), []) if record_index else []
            if candidate and not any(_valid_regions(row) for row in matches):
                continue
            record = (candidate_record(manifest, song, difficulty, matches, _metadata=metadata)
                      if candidate else next(iter(matches), None))
            if candidate and record is None:
                continue
            if manifest in mechanics or difficulty.get('learning_role', song.get('learning_role')) == 'mechanics':
                mechanics_count += 1
                chart_count += 1
                continue
            if record and not candidate:
                try:
                    if record.get("chart_sha256") != _chart_digest(song["maidata_path"], index):
                        record = None
                    elif record.get("audio_sha256") and record["audio_sha256"] != _digest(song.get("audio_path")):
                        record = None
                except (OSError, ValueError, UnicodeError):
                    record = None
            if not candidate and not (record or {}).get("regions") and difficulty.get("note_count") is not None:
                count = max(0., float(difficulty["note_count"]))
                outer = 0. if (record or {}).get("label") in {"excluded", "negative"} else max(0., float((record or {}).get("weight", 1.)))
                weight = outer * max(0., float((record or {}).get("unmarked_weight", 1.)))
                timing += count * weight
                layout += count * weight
                chart_count += 1
                continue
            try:
                heads = _head_events(song["maidata_path"], index)
            except (OSError, ValueError, UnicodeError) as exc:
                warnings.append(dict(manifest=str(manifest), source_path=song["maidata_path"], reason=str(exc)))
                continue
            chart_count += 1
            zero_unmarked = candidate or (record or {}).get("unmarked_weight") == 0
            base = 0. if zero_unmarked else float((record or {}).get("unmarked_weight", 1.))
            weight = 0. if (record or {}).get("label") in {"excluded", "negative"} else max(0., float((record or {}).get("weight", 1.)))
            valid_regions = _valid_regions(record) if candidate else None
            regions = []
            for region in (record or {}).get("regions", []):
                t, c = ((0., 0.) if candidate and region not in valid_regions
                        else zero_unmarked_region_weights(region) if zero_unmarked
                        else region_component_weights(region))
                regions.append((float(region["start_sec"]), float(region["end_sec"]), max(0., t), max(0., c)))
            for sec, count in heads:
                t = c = max(0., base)
                # Later annotations override earlier overlapping regions.
                for start, end, rt, rc in reversed(regions):
                    if start <= sec < end:
                        t, c = rt, rc
                        break
                timing += count * t * weight
                layout += count * c * weight
        sources.append(dict(manifest=str(manifest), candidate_pool=candidate, charts=chart_count,
                            timing_note_heads=timing, layout_note_heads=layout,
                            **({'mechanics_charts': mechanics_count} if mechanics_count else {})))
    total_t, total_c = sum(s["timing_note_heads"] for s in sources), sum(s["layout_note_heads"] for s in sources)
    for source in sources:
        source.update(timing_share=source["timing_note_heads"] / total_t if total_t else 0.,
                      layout_share=source["layout_note_heads"] / total_c if total_c else 0.)
    if progress:
        progress(dict(completed=total, total=total, stage="complete", source=""))
    return dict(unit="weighted_note_heads_approximation", approximation=True,
                description="按真实起音音符头与分项权重估算；不是 token 数或训练 loss 权重占比",
                sources=sources, timing_note_heads=total_t, layout_note_heads=total_c, warnings=warnings)


class _ScanProgress:
    """Durable, throttled progress also works after refreshing the browser."""
    def __init__(self, root):
        self.path = Path(root) / 'scan_state.json'
        self.state = _read(self.path)
        self.started = self.stage_started = time.monotonic()
        self.stage = None
        self.last = -math.inf

    def __call__(self, value):
        now = time.monotonic()
        stage = value.get('stage', 'scanning')
        changed = stage != self.stage
        if changed:
            self.stage, self.stage_started = stage, now
        completed, total = int(value.get('completed', value.get('done', 0))), int(value.get('total', 0))
        if not changed and now - self.last < 1. and completed != total:
            return
        self.last = now
        elapsed = now - self.stage_started
        progress = dict(value, stage=stage, completed=completed, done=completed,
                        elapsed_sec=round(now - self.started, 1), stage_elapsed_sec=round(elapsed, 1))
        if completed >= 3 and elapsed >= 3 and total > completed:
            progress['eta_sec'] = round(elapsed * (total - completed) / completed)
        self.state.update(status='completed' if stage == 'complete' else 'running',
                          progress=progress, updated_at=_now(), pid=os.getpid())
        _write_json(self.path, self.state)
        print('candidate_scan ' + json.dumps(progress, ensure_ascii=False), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["scan", "build", "screen"])
    parser.add_argument("target")
    parser.add_argument("--workspace-root")
    parser.add_argument("--name")
    parser.add_argument("--reference-manifest", action="append", default=[])
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument('--workers', type=int, default=0, help='0: 自动最多 6 个独立检查进程；1: 串行')
    parser.add_argument("--difficulty-slots", nargs="+", type=int, default=[5, 6])
    parser.add_argument("--min-level", type=float, default=12.0)
    parser.add_argument("--max-level", type=float, default=15.0)
    parser.add_argument("--tap-on-slide-mode", default="mask_hard")
    parser.add_argument("--inner-muri-mode", default="allow")
    parser.add_argument("--max-offset-ms", type=float, default=35.0)
    parser.add_argument("--max-problem-fraction", type=float, default=0.35)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "screen":
        result = auto_screen_pool(
            args.target,
            difficulty_slots=tuple(args.difficulty_slots),
            min_level=args.min_level,
            max_level=args.max_level,
            tap_on_slide_mode=args.tap_on_slide_mode,
            inner_muri_mode=args.inner_muri_mode,
            max_offset_ms=args.max_offset_ms,
            max_problem_fraction=args.max_problem_fraction,
            dry_run=args.dry_run,
        )
        print(json.dumps(result, ensure_ascii=False))
        return 0
    import torch
    torch.set_num_threads(max(1, args.threads))
    workers = max(1, min(6, (os.cpu_count() or 2) // 2)) if args.workers == 0 else max(1, min(8, args.workers))
    if args.command == "build":
        if not args.workspace_root:
            parser.error("build requires --workspace-root")
        target = pool_identity(args.target, args.workspace_root)['pool_dir']
        report = _ScanProgress(target)
        report(dict(stage='starting', completed=0, total=0, workers=workers))
        exists = (Path(target) / 'pool.json').is_file()
        if not exists:
            create_pool(args.target, args.workspace_root, args.name, args.reference_manifest, progress=report)
    else:
        target = args.target
        if args.reference_manifest and (Path(target) / 'pool.json').is_file():
            meta = _read(Path(target) / 'pool.json')
            source_manifest = Path(meta['source_manifest']).resolve() if meta.get('source_manifest') else None
            meta['reference_manifests'] = [
                str(Path(p).resolve()) for p in args.reference_manifest
                if Path(p).is_file() and Path(p).resolve() != source_manifest
            ]
            _write_json(Path(target) / 'pool.json', meta)
        report = _ScanProgress(target)
        report(dict(stage='starting', completed=0, total=0, workers=workers))
        exists = True
    result = scan_pool(target, progress=report, workers=workers, synchronize=exists)
    report(dict(stage='complete', completed=result['counts']['charts'], total=result['counts']['charts'], counts=result['counts']))
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
