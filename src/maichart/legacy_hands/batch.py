"""Batch old-player hand prediction. Run with --help for inputs and output format."""
import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import gzip
import hashlib
import json
import os
import platform
from pathlib import Path
import sqlite3
import sys
import time
from . import SWEEP_STYLES


# The first Windows CPython 3.12 release was trained with mixed checkout
# line endings. Keep its two existing cache IDs for exactly those contents
# and runtime. A change to an input file still produces a new revision.
_FIRST_RELEASE_CACHE_IDS = {
    'ba0988d3db7feafcad012429a06b87adfc72b5f3908d93eed998049b9a2b2bb8':
        '3ed58f32245f5a2ee4c6fc77568b4e3b7b3f45b705a65be8009d32b15d5e0fe6',
    '5b067c12928654873d4e5bf2521f927eca4572b68dcf4e1ac5ab78d62ca1cad9':
        '7de022284b52f86e843d73e11233958e0857cd8fa05f440dc4bd1c24d67e74c1',
}

def source_bytes(path):
    """Hash Python and JSON sources the same after a Git LF or CRLF checkout."""
    return path.read_bytes().replace(b'\r\n', b'\n')
def solver_revision(sweep_style='continuous'):
    if sweep_style not in SWEEP_STYLES:
        raise ValueError('未知的扫键手顺偏好')
    package = Path(__file__).resolve().parent
    root = package.parent
    files = [package / name for name in ('prediction.py', 'annotation.py', 'rim_stroke.py', 'tail_touch.py', 'player_motion.py', 'player_geometry.py',
                                         'player_kinematics.py', 'player_stance.py')]
    files.extend(root / name for name in (
        'event_timing.py', 'timing.py', 'duration.py', 'ir.py', 'notes.py', 'slides.py',
        'maidata.py', 'tempo_clock.py', 'generation/judgment_system.py',
        'generation/simai_compile.py', 'generation/muri/reference/provenance.json',
    ))
    files.extend((root / 'generation/muri').glob('*.py'))
    digest = hashlib.sha256(repr(('batch-v1', sweep_style, sys.implementation.cache_tag,
                                  sys.platform, platform.machine())).encode())
    for file in sorted(files):
        digest.update(file.relative_to(root).as_posix().encode())
        digest.update(hashlib.sha256(source_bytes(file)).digest())
    canonical = digest.hexdigest()
    return _FIRST_RELEASE_CACHE_IDS.get(canonical, canonical)


def calculate(data, slot, mode, sweep_style='continuous'):
    from maichart.event_timing import parse_event_chart_text
    from maichart.maidata import decode_text_with_encoding
    from .player_motion import simulate_parsed_chart
    from .prediction import predict_hands
    text, _ = decode_text_with_encoding(data)
    chart = parse_event_chart_text(text, slot, preserve_each_groups=True)
    if mode == 'labels':
        from .annotation import annotate
        result = annotate(chart)
    else:
        result = (predict_hands(chart, sweep_style=sweep_style) if mode == 'hands'
                  else simulate_parsed_chart(chart, sweep_style=sweep_style))
    encoded = json.dumps(result, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    return gzip.compress(encoded.encode('utf-8'), compresslevel=1, mtime=0)


def charts(roots):
    seen = set()
    for root in roots:
        for path in (root.rglob('maidata.txt') if root.is_dir() else (root,)):
            path = path.resolve()
            if path not in seen:
                seen.add(path)
                yield path


def run(roots, db, *, slot, mode, workers, sweep_style='continuous'):
    if mode not in ('hands', 'full', 'labels'):
        raise ValueError('计算模式须为 hands、labels 或 full')
    if workers < 1:
        raise ValueError('并发谱面数至少为 1')
    if os.name == 'nt' and workers > 61:
        raise ValueError('Windows 进程池最多支持 61 个工作进程')
    db.parent.mkdir(parents=True, exist_ok=True)
    rev = solver_revision('continuous' if mode == 'labels' else sweep_style)
    start = time.monotonic()
    done = reused = failed = 0
    last_report = start
    conn = sqlite3.connect(db, timeout=60)
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    queued = {}
    inflight = {}
    try:
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('CREATE TABLE IF NOT EXISTS chart_sources(path TEXT PRIMARY KEY, sha256 TEXT NOT NULL)')
        conn.execute('''CREATE TABLE IF NOT EXISTS hand_results(
            sha256 TEXT NOT NULL, revision TEXT NOT NULL, slot INTEGER NOT NULL,
            mode TEXT NOT NULL, payload BLOB NOT NULL, PRIMARY KEY (sha256, revision, slot, mode))''')

        def save(key, payload):
            with conn:
                conn.execute('INSERT OR IGNORE INTO hand_results VALUES(?,?,?,?,?)', (*key,payload))

        def collect():
            nonlocal done, failed, reused
            finished, _ = wait(queued, return_when=FIRST_COMPLETED)
            for future in finished:
                key = queued.pop(future)
                paths = inflight.pop(key)
                try:
                    payload = future.result()
                except Exception as exc:
                    failed += len(paths)
                    for path in paths:
                        print(f'{path} · {exc}', file=sys.stderr, flush=True)
                    continue
                save(key, payload)
                done += 1
                reused += len(paths)-1

        def progress():
            nonlocal last_report
            now = time.monotonic()
            if now-last_report >= 5.:
                print(f'已计算 {done} · 已复用 {reused} · 失败 {failed} · 处理中 {len(queued)}', flush=True)
                last_report = now

        for path in charts(roots):
            progress()
            try:
                data = path.read_bytes()
            except OSError as exc:
                failed += 1
                print(f'{path} · {exc}', file=sys.stderr)
                continue
            sha = hashlib.sha256(data).hexdigest()
            key = (sha, rev, slot, mode)
            with conn:
                conn.execute('INSERT INTO chart_sources VALUES(?,?) ON CONFLICT(path) DO UPDATE SET sha256=excluded.sha256 WHERE sha256!=excluded.sha256',
                             (str(path),sha))
            if conn.execute('SELECT 1 FROM hand_results WHERE sha256=? AND revision=? AND slot=? AND mode=?',key).fetchone():
                reused += 1
            elif key in inflight:
                inflight[key].append(path)
            elif pool is None:
                try:
                    payload = calculate(data,slot,mode,sweep_style)
                except Exception as exc:
                    failed += 1
                    print(f'{path} · {exc}',file=sys.stderr,flush=True)
                    continue
                save(key,payload)
                done += 1
            else:
                future = pool.submit(calculate,data,slot,mode,sweep_style)
                queued[future] = key
                inflight[key] = [path]
                if len(queued) >= 2*workers:
                    collect()
        while queued:
            collect()
            progress()
    finally:
        if pool:
            pool.shutdown(cancel_futures=True)
        conn.close()
    elapsed = time.monotonic()-start
    print(f'已计算 {done} · 已复用 {reused} · 失败 {failed} · 耗时 {elapsed:.2f} 秒',flush=True)
    return dict(computed=done,reused=reused,failed=failed,elapsed=elapsed,revision=rev)


def main(argv=None):
    p = argparse.ArgumentParser(description='按整谱并行计算旧版人体模拟手顺，已完成结果可在下次运行时复用。')
    p.add_argument('inputs', nargs='+', type=Path, help='谱面文件或包含 maidata.txt 的目录')
    p.add_argument('--output', required=True, type=Path, help='结果数据库（SQLite，payload 为 gzip 压缩的 JSON）')
    p.add_argument('--slot', type=int, default=5, help='难度槽，默认 5（MASTER）')
    p.add_argument('--workers', type=int, help='并发谱面数。自动按可用逻辑核数的一半选择，分手模式最多 8，完整模式最多 4')
    p.add_argument('--mode', choices=('hands','full','labels'), default='hands', help='hands 导出所选方案；labels 综合两种方案，导出手顺、置信等级与不确定项；full 计算姿态与诊断')
    p.add_argument('--sweep-style', choices=SWEEP_STYLES, default='continuous',
                   help='continuous 连扫优先（默认），relay 在另一只手能提前就位时偏好分段换手')
    args = p.parse_args(argv)
    cpus = getattr(os, 'process_cpu_count', os.cpu_count)() or 1
    automatic = min(8 if args.mode == 'hands' else 4, max(1, cpus // 2))
    workers = automatic if args.workers is None else args.workers
    print(f'难度槽 {args.slot} · 并发 {workers} · 输出 {args.output}', flush=True)
    result = run(args.inputs, args.output, slot=args.slot, mode=args.mode, workers=workers, sweep_style=args.sweep_style)
    if result['computed']+result['reused']+result['failed']==0:
        p.error('所选目录中没有 maidata.txt')
    return int(bool(result['failed']))


if __name__ == '__main__':
    raise SystemExit(main())
