"""Reusable, content-addressed offline hand labels. Workers never load audio."""
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from functools import lru_cache
import hashlib
from itertools import zip_longest
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import zlib

import numpy as np

from maichart.event_timing import EventChart, EventTempo
from maichart import legacy_hands
from maichart.legacy_hands import HAND_FIELDS, MISSING, NO_RIVAL
from maichart.legacy_hands.batch import solver_revision, source_bytes
from maichart.training.reviewed_generation import library_file_lock

SCHEMA = 'head_edges_v2'
DEFAULT_LIMIT = 128
N_FIELDS = len(HAND_FIELDS)


@lru_cache(maxsize=1)
def revision():
    from maichart.legacy_hands import training_labels
    from maichart.generation.muri.data import REFERENCE_COMMIT
    h = hashlib.sha256((SCHEMA + solver_revision() + REFERENCE_COMMIT).encode())
    h.update(source_bytes(Path(legacy_hands.__file__)))
    h.update(source_bytes(Path(training_labels.__file__)))
    # The first v2 checkpoints hashed the package header with Windows line
    # endings. Keep that published cache contract for this exact fingerprint.
    canonical = h.hexdigest()
    if canonical == 'd6d7a49e5ac2a732d5ada89f8a44d65cc1c62390704c288e41410116dbc8ac78':
        return '6d5a9999ca940d1c9b911b1e0662b5639c8b3af61fe5fdc17fe1967dc2443c76'
    return canonical


def _chart(song):
    return EventChart(points=song.points, tempo_changes=[EventTempo(b,v) for b,v in zip(song.clock.beats,song.clock.bpms)],
                      first=float(song.clock.seconds[0]), end_beat=getattr(song,'authored_end_beat',None) or song.end_beat,
                      difficulty=song.difficulty)


def _key(chart):
    h = hashlib.sha256(repr((chart.first, str(chart.end_beat), chart.difficulty,
                              [(str(t.beat),float(t.bpm)) for t in chart.tempo_changes])).encode())
    for p in chart.points:
        h.update(f'\n{p.beat}\t{p.note_text}'.encode())
    return h.hexdigest()


def _load(path, key, count):
    try:
        raw = zlib.decompress(path.read_bytes())
        if raw[:32] == bytes.fromhex(key) and len(raw) == 32 + count*N_FIELDS:
            labels = np.frombuffer(raw, dtype=np.int8, offset=32).reshape(count, N_FIELDS)
            if np.all((labels == MISSING) | (labels == NO_RIVAL) | ((labels >= -2) & (labels <= 2))):
                return labels
    except (FileNotFoundError, zlib.error):
        pass
    return None


def _build(path, key, chart):
    from maichart.legacy_hands.training_labels import calculate
    with library_file_lock(path.with_suffix('.lock')):
        labels = _load(path,key,len(chart.points))
        if labels is not None:
            return False
        try:
            labels = calculate(chart)
        except ValueError as exc:
            if '上限' not in str(exc) and '超出预览范围' not in str(exc):
                raise
            labels = np.full((len(chart.points),N_FIELDS),MISSING,np.int8)
        packed = zlib.compress(bytes.fromhex(key) + labels.tobytes(), 1)
        path.parent.mkdir(parents=True,exist_ok=True)
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as f:
                tmp = Path(f.name)
                f.write(packed)
            os.replace(tmp,path)
        finally:
            if tmp is not None:
                tmp.unlink(missing_ok=True)
        return True


def _selection(dataset, limit):
    # Stable under manifest reorder/relocation. Give low-frequency roles and
    # difficulty slots a turn before taking a second chart from the same bucket.
    groups = defaultdict(list)
    by_source = defaultdict(list)
    for song in dataset.songs:
        source = song.chart_sha256 or _key(_chart(song))
        by_source[(source,song.difficulty)].append(song)
    for identity,songs in sorted(by_source.items()):
        role = tuple(sorted({s.learning_role for s in songs}))
        groups[role,identity[1]].append(songs)
    ordered = [songs for round_ in zip_longest(*(groups[g] for g in sorted(groups)))
               for songs in round_ if songs is not None]
    return ordered[:limit] if limit else ordered


def _work(fn,jobs,workers):
    if len(jobs)<2 or workers==1:
        for job in jobs:
            yield job,fn(*job)
        return
    iterator=iter(jobs)
    with ProcessPoolExecutor(max_workers=min(workers,len(jobs))) as pool:
        pending={}
        while True:
            while len(pending)<2*workers:
                job=next(iterator,None)
                if job is None:
                    break
                pending[pool.submit(fn,*job)]=job
            if not pending:
                break
            for future in wait(pending,return_when=FIRST_COMPLETED).done:
                yield pending.pop(future),future.result()


def _budget(limit,workers):
    if limit < 0 or workers < 1 or workers > 32:
        raise ValueError('手顺缓存谱面数须大于等于 0，并行进程须为 1 至 32')


def prepare(dataset, *, limit, workers, progress=None):
    _budget(limit,workers)
    started = time.perf_counter()
    rev = revision()
    root = Path(dataset.cache_dir) / 'hand_labels' / SCHEMA / rev
    selected = _selection(dataset,limit)
    groups = defaultdict(list)
    jobs = {}
    for copies in selected:
        for song in copies:
            chart = _chart(song)
            key = _key(chart)
            groups[key].append(song)
            jobs.setdefault(key,(root/key[:2]/(key+'.zlib'),key,chart))
    done = hits = built = empty = 0
    scored = np.zeros(N_FIELDS,np.int64)
    def attach(key, labels, cold):
        nonlocal done, hits, built, empty
        for song in groups[key]:
            song.hand_memory_labels = labels
        done += 1
        built += int(cold)
        hits += int(not cold)
        scored[:] += (labels != MISSING).sum(0)
        empty += int(np.all(labels == MISSING))
        if progress and (done == len(jobs) or done % 16 == 0):
            elapsed = time.perf_counter()-started
            progress(completed=done,total=len(jobs),hits=hits,built=built,seconds=round(elapsed,2))
    misses = []
    for path,key,chart in jobs.values():
        labels = _load(path,key,len(chart.points))
        if labels is None:
            misses.append((path,key,chart))
        else:
            attach(key,labels,False)
    def collect(job,cold):
        path,key,chart = job
        labels = _load(path,key,len(chart.points))
        if labels is None:
            raise OSError(f'手顺缓存未通过回读校验 {path}')
        attach(key,labels,cold)
    for job,cold in _work(_build,misses,workers):
        collect(job,cold)
    contract = dict(schema=SCHEMA,revision=rev,limit=limit,
                    selection_sha256=hashlib.sha256(''.join(sorted(jobs)).encode()).hexdigest())
    return contract,dict(charts=len(dataset.songs),unique_selected=len(jobs),hits=hits,built=built,
                         no_labels=empty,scored_by_field=scored.tolist(),workers=workers,
                         seconds=round(time.perf_counter()-started,3),cache=str(root))


def _prewarm_one(source,slot,raw_sha,dialect,root):
    from maichart.event_timing import parse_event_chart_text
    from maichart.maidata import decode_text_with_encoding
    from . import chart_dialect
    data=source.read_bytes()
    if hashlib.sha256(data).hexdigest()!=raw_sha:
        raise ValueError(f'扫描后谱面已改变，请重新运行 {source}')
    try:
        chart=parse_event_chart_text(decode_text_with_encoding(data)[0],slot)
        if dialect in chart_dialect.NORMALIZED_POLICIES:
            chart,_=chart_dialect.normalize_training_chart(chart,dialect)
    except (ValueError,OverflowError,ZeroDivisionError) as exc:
        return 'skipped',str(exc)
    key=_key(chart)
    path=root/key[:2]/(key+'.zlib')
    if _load(path,key,len(chart.points)) is not None:
        return 'hits',None
    return ('built' if _build(path,key,chart) else 'hits'),None


def main():
    """Prewarm a manifest using CPU workers; no audio, vocabulary or training windows."""
    import argparse
    import json
    from . import chart_dialect
    from maichart.training.dataset_catalog import _source_asset
    from maichart.training.event_dataset import numeric_level

    parser=argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--cache-dir',required=True)
    parser.add_argument('--limit',type=int,default=DEFAULT_LIMIT,help='0 computes all eligible charts')
    parser.add_argument('--workers',type=int,default=min(4,os.cpu_count() or 1))
    parser.add_argument('--min-level',type=float,default=0.)
    parser.add_argument('--max-level',type=float,default=16.)
    parser.add_argument('--chart-dialect',choices=(*chart_dialect.NORMALIZED_POLICIES,chart_dialect.LEGACY_STRICT_POLICY),
                        default=chart_dialect.POLICY)
    args=parser.parse_args()
    _budget(args.limit,args.workers)
    manifest=Path(args.manifest).resolve()
    payload=json.loads(manifest.read_text('utf-8-sig'))
    root=Path(args.cache_dir)/'hand_labels'/SCHEMA/revision()
    entries=[]; hashes={}
    for entry in payload['songs']:
        for d in entry.get('difficulties',()):
            if not d.get('has_chart',True):
                continue
            try:
                level=numeric_level(d.get('level_raw',d.get('level')))
            except ValueError:
                continue
            if not args.min_level<=level<=args.max_level:
                continue
            path=_source_asset(entry['maidata_path'],manifest,payload)
            if path not in hashes:
                hashes[path]=hashlib.sha256(path.read_bytes()).hexdigest()
            entries.append(SimpleNamespace(source_path=path,chart_sha256=hashes[path],
                difficulty=int(d.get('difficulty_index',d.get('index',0))),learning_role=d.get('learning_role',entry.get('learning_role','full'))))
    groups=_selection(SimpleNamespace(songs=entries),args.limit)
    jobs=[(g[0].source_path,g[0].difficulty,g[0].chart_sha256,args.chart_dialect,root) for g in groups]
    counts=dict(hits=0,built=0,skipped=0)
    started=time.perf_counter()
    for i,(job,(status,reason)) in enumerate(_work(_prewarm_one,jobs,args.workers),1):
        counts[status]+=1
        if reason:
            print(json.dumps(dict(source=str(job[0]),slot=job[1],reason=reason),ensure_ascii=False),flush=True)
        if i%16==0:
            print('hand_labels '+json.dumps(dict(completed=i,total=len(jobs),**counts)),flush=True)
    print(json.dumps(dict(unique_selected=len(jobs),**counts,seconds=round(time.perf_counter()-started,3),cache=str(root)),
                     ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
