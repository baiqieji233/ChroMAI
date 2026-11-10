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
import time
import unicodedata
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from maichart.training.dataset_catalog import _absolutize_song_paths, _write_json
from maichart.training.quality_annotations import (
    chart_sha256, audio_sha256, iter_catalog, region_component_weights,
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


def _charts(manifest, payload=None):
    path = Path(manifest).resolve()
    payload = _read(path) if payload is None else payload
    for song in payload.get("songs", []):
        song = copy.deepcopy(song)
        _absolutize_song_paths(song, path, payload)
        for difficulty in song.get("difficulties", []):
            index = difficulty.get("difficulty_index", difficulty.get("index"))
            if index is not None:
                yield song, difficulty, int(index)


def _key(path, index):
    return str(Path(path).resolve()).casefold(), int(index)


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
    result = []
    for region in (record or {}).get("regions", []):
        if region.get("label") == "uncertain" or region.get("training_use") == "none":
            continue
        try:
            a, b = float(region["start_sec"]), float(region["end_sec"])
            t, c = region_component_weights(region)
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


def review_guard(manifest, chart_id, *, positive=True):
    manifest = Path(manifest).resolve()
    root = manifest.parent
    metadata = _read(root / "pool.json")
    if _read(root / 'scan_state.json').get('status') in {'queued', 'running'}:
        raise ValueError('候选库正在检查，请等待扫描完成后再保存评价')
    if metadata.get("schema") != SCHEMA or manifest != root / "training_manifest.json":
        raise ValueError("不是当前候选库清单")
    if not metadata.get("scan_complete"):
        raise ValueError("请先完成候选库扫描")
    row = next((r for r in iter_catalog(manifest) if r.chart_id == chart_id), None)
    if row is None:
        raise ValueError("谱面身份已变化，请重新扫描候选库")
    for song, difficulty, index in _charts(manifest):
        if _key(song["maidata_path"], index) != _key(row.source_path, row.difficulty_index):
            continue
        identity = difficulty.get("candidate_identity", {})
        try:
            valid = (not _stale(root, metadata, song, difficulty) and row.chart_sha256 == identity.get("chart_sha256")
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


def pool_status(pool_dir):
    root = _pool(pool_dir)
    metadata = _read(root / "pool.json")
    if metadata.get("schema") != SCHEMA:
        raise ValueError("候选库不存在或格式不兼容")
    manifest = root / "training_manifest.json"
    payload = _read(manifest)
    records = _records(manifest)
    record_index = {}
    for record in records:
        record_index.setdefault(_key(record.get("source_path", ""), record.get("difficulty_index", -1)), record)
    stat_cache = {}
    def stat_once(path):
        key = str(path)
        if key not in stat_cache:
            stat_cache[key] = _stat(path)
        return stat_cache[key]
    source_changed = metadata.get("source_manifest_stat") != stat_once(metadata.get("source_manifest"))
    counts = dict(charts=0, error=0, suspicious=0, ok=0, pending=0, stale=0, duplicate=len(metadata.get("excluded", [])))
    regions = []
    timing_heads = layout_heads = 0.
    for song, difficulty, index in _charts(manifest, payload):
        counts["charts"] += 1
        identity = difficulty.get("candidate_identity", {})
        stale = (source_changed or identity.get("source_stat") != stat_once(song.get("maidata_path")) or
                 identity.get("audio_stat") != stat_once(song.get("audio_path")))
        state = "stale" if stale else difficulty.get("candidate_status", "pending")
        counts[state] = counts.get(state, 0) + 1
        if stale:
            continue
        record = record_index.get(_key(song["maidata_path"], index))
        if (not record or record.get("chart_sha256") != identity.get("chart_sha256") or
                record.get("audio_sha256") != identity.get("audio_sha256")):
            continue
        valid_regions = _valid_regions(record)
        regions.extend(valid_regions)
        if (not metadata.get("scan_complete") or not valid_regions or
                difficulty.get("candidate_status") not in {"ok", "suspicious"}):
            continue
        try:
            weighted_regions = []
            for region in record.get("regions", []):
                t, c = region_component_weights(region) if region in valid_regions else (0., 0.)
                weighted_regions.append((float(region["start_sec"]), float(region["end_sec"]), max(0., t), max(0., c)))
            weight = max(0., float(record.get("weight", 1.)))
            for sec, count in _head_events(song["maidata_path"], index):
                region = next((r for r in reversed(weighted_regions) if r[0] <= sec < r[1]), None)
                if region:
                    timing_heads += region[2] * count * weight
                    layout_heads += region[3] * count * weight
        except (OSError, ValueError, UnicodeError):
            pass
    return dict(metadata, pool_dir=str(root), manifest=str(root / "training_manifest.json"), counts=counts,
                stale=bool(counts["stale"]), confirmed_regions=len(regions),
                reviewable_charts=counts["ok"] + counts["suspicious"] if metadata.get("scan_complete") else 0,
                supervision=dict(unit="weighted_note_heads_approximation", timing_note_heads=timing_heads,
                                 layout_note_heads=layout_heads),
                timing_regions=sum(region_component_weights(r)[0] > 0 for r in regions),
                layout_regions=sum(region_component_weights(r)[1] > 0 for r in regions))


def review_queue(pool_dir, offset=0, limit=1):
    root = _pool(pool_dir)
    metadata = _read(root / "pool.json")
    if not metadata.get("scan_complete"):
        return dict(items=[], total=0, offset=max(0, int(offset)), limit=max(1, int(limit)),
                    stale=pool_status(root)["stale"], next_offset=max(0, int(offset)), scan_complete=False)
    charts = {_key(s["maidata_path"], i): (s, d) for s, d, i in _charts(root / "training_manifest.json")}
    records = _records(root / "training_manifest.json")
    groups = {}
    for result in _read(root / "scan" / "results.json").get("charts", []):
        key = _key(result["source_path"], result["difficulty_index"])
        if key not in charts or not result.get("chart_id") or result.get("status") == "error":
            continue
        song, difficulty = charts[key]
        try:
            if _stale(root, metadata, song, difficulty):
                continue
        except OSError:
            continue
        record = next((r for r in records if r.get("chart_id") == result["chart_id"]), {})
        segments = []
        for segment in result.get("segments", []):
            if segment.get("queue_eligible") is False or segment.get("onsets", 0) < 2:
                continue
            a, b = segment["start_sec"], segment["end_sec"]
            covered = sorted((max(a, float(r["start_sec"])), min(b, float(r["end_sec"])))
                             for r in (record.get("candidate_reviews", []) + record.get("regions", []))
                             if float(r["end_sec"]) > a and float(r["start_sec"]) < b)
            total, end = 0., a
            for left, right in covered:
                total += max(0., right - max(end, left)); end = max(end, right)
            if total < .5 * (b - a):
                segments.append(segment)
        if not segments:
            continue
        rates = sorted(s.get("onset_rate", 0) for s in segments)
        median = rates[len(rates) // 2]
        representative = min(segments, key=lambda s: (abs(s.get("onset_rate", 0) - median), -(s.get("attack_support") or 0)))
        risk = max(segments, key=lambda s: (s.get("conflicts", 0), s.get("mixed_16_24", False), s.get("onset_rate", 0)))
        rng = random.Random(result["chart_id"])
        choices, seen = [], set()
        for segment, why in ((representative, "代表性片段"), (risk, "复杂或风险片段"), (rng.choice(segments), "随机抽查")):
            if segment["start_sec"] in seen:
                continue
            seen.add(segment["start_sec"])
            choices.append(dict(segment, chart_id=result["chart_id"], source_path=result["source_path"],
                                difficulty_index=result["difficulty_index"], title=result.get("title"), why=why,
                                issues=result.get("issues", []), context_start_sec=max(0, segment["start_sec"] - 4),
                                context_end_sec=min(result.get("audio_duration_sec", segment["end_sec"] + 4), segment["end_sec"] + 4)))
        try:
            level = float(str(result.get("level", "0")).replace("+", ".7"))
        except ValueError:
            level = 0
        bucket = (int(level // 3), int((result.get("stats", {}).get("bpm") or 120) // 40))
        groups.setdefault(bucket, []).append((result["chart_id"], choices))
    for rows in groups.values():
        rows.sort(key=lambda row: hashlib.sha256(row[0].encode()).hexdigest())
    items = []
    while any(groups.values()):
        for bucket in sorted(groups):
            if groups[bucket]:
                items.extend(groups[bucket].pop(0)[1])
    offset, limit = max(0, int(offset)), min(100, max(1, int(limit)))
    return dict(items=items[offset:offset + limit], total=len(items), offset=offset, limit=limit,
                stale=pool_status(root)["stale"],
                next_offset=offset + len(items[offset:offset + limit]))


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
    from maichart.notes import BASIC_COMPACT_CHORD_RE, _split_top_level_slashes
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
            compact = BASIC_COMPACT_CHORD_RE.fullmatch(raw)
            atoms = [button + compact['modifiers'] for button in compact['positions']] if compact else [raw]
            for atom in atoms:
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


def supervision_budget(manifests, *, _accepted=None, progress=None):
    """Estimate weighted heads; accepted owners are supplied by dataset composition.

    Standalone estimation keeps each source's occurrences, as before. Composition
    passes _accepted to count only the selected owner of duplicate charts.
    """
    sources, warnings = [], []
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
                weight = (max(0., float((record or {}).get("weight", 1.))) *
                          max(0., float((record or {}).get("unmarked_weight", 1.))))
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
            base = 0. if candidate else float((record or {}).get("unmarked_weight", 1.))
            weight = max(0., float((record or {}).get("weight", 1.)))
            valid_regions = _valid_regions(record) if candidate else None
            regions = []
            for region in (record or {}).get("regions", []):
                t, c = ((0., 0.) if candidate and region not in valid_regions
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
                            timing_note_heads=timing, layout_note_heads=layout))
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
                          progress=progress, updated_at=_now())
        _write_json(self.path, self.state)
        print('candidate_scan ' + json.dumps(progress, ensure_ascii=False), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["scan", "build"])
    parser.add_argument("target")
    parser.add_argument("--workspace-root")
    parser.add_argument("--name")
    parser.add_argument("--reference-manifest", action="append", default=[])
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument('--workers', type=int, default=0, help='0: 自动最多 4 个独立检查进程；1: 串行')
    args = parser.parse_args(argv)
    import torch
    torch.set_num_threads(max(1, args.threads))
    workers = max(1, min(4, (os.cpu_count() or 2) // 2)) if args.workers == 0 else max(1, min(8, args.workers))
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
        report = _ScanProgress(target)
        report(dict(stage='starting', completed=0, total=0, workers=workers))
        exists = True
    result = scan_pool(target, progress=report, workers=workers, synchronize=exists)
    report(dict(stage='complete', completed=result['counts']['charts'], total=result['counts']['charts'], counts=result['counts']))
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
