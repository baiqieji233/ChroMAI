"""Pre-extract audio features in parallel before training starts.

Feature extraction is the single largest CPU cost in the pipeline: measured at
**2.71 s per song** (about 162 s of audio), which is 45 minutes for a 1000-song
corpus.  It happens lazily inside the training data loader, so the first epoch
pays it while the GPU waits.

The caches are content-addressed and idempotent, so the work parallelises
perfectly: each song is independent, extraction reads only the audio file, and
the result is a cache file named after the audio digest plus the feature
configuration.  A worker pool therefore only needs a list of audio paths.

Two safety rules make this sound:

* the destination of every write is unique per song, and
  ``_atomic_save_npz`` now uses a per-writer temporary name, so two processes
  can never interleave into one file;
* songs that already have a valid cache are skipped **without** re-analysing
  them, so re-running this (or running it after a partial run) does no work.

The feature configuration (sample rate, hop length) must match training exactly,
which is why the training entry point passes the dataset's own values.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Sequence

from maichart.generation.event_audio import (
    FEATURE_RATE_HZ,
    audio_cache_dir,
    event_audio_cache_path,
    extract_event_audio,
)

DEFAULT_WORKERS = max(1, min(8, (os.cpu_count() or 2) - 2))


def resolve_workers(value: str | int | None) -> int:
    """Translate a CLI/UI worker request into a process count (0 disables)."""

    if value is None:
        return DEFAULT_WORKERS
    if isinstance(value, int):
        return max(0, value)
    text = str(value).strip().lower()
    if text in {"", "auto", "default"}:
        return DEFAULT_WORKERS
    if text in {"0", "off", "none", "disable", "disabled"}:
        return 0
    try:
        return max(0, int(text))
    except ValueError:
        raise ValueError(f"进程数必须是整数、'auto' 或 'off'：{value!r}") from None


def _extract_one(task: tuple[str, str, int, int]) -> tuple[str, str, str]:
    """Worker entry point: return (path, status, message)."""

    audio_path, cache_dir, sample_rate, hop_length = task
    # One canonical location, shared with the training dataset: writing anywhere
    # else means the features are extracted again the moment training reads them.
    cache_root = audio_cache_dir(cache_dir)
    try:
        # Resolve the cache path first: an existing cache means this song needs
        # no analysis at all, and resolving is just one hash of the audio file.
        cache_path = event_audio_cache_path(
            audio_path, cache_root, sample_rate=sample_rate, hop_length=hop_length
        )
        if cache_path is not None and cache_path.exists():
            return audio_path, "cached", ""
        extract_event_audio(audio_path, cache_dir=cache_root, sample_rate=sample_rate,
                            hop_length=hop_length)
        return audio_path, "extracted", ""
    except Exception as exc:  # noqa: BLE001 - reported to the caller, never fatal
        return audio_path, "failed", f"{type(exc).__name__}: {exc}"


def audio_paths_from_manifest(manifest_path: str | Path) -> list[str]:
    """Every distinct audio file the manifest references (manifest order)."""

    source = Path(manifest_path).resolve()
    data = json.loads(source.read_text(encoding="utf-8"))
    base = source.parent
    seen: set[str] = set()
    result: list[str] = []
    for song in data.get("songs", []):
        value = song.get("audio_path")
        if not value:
            continue
        path = Path(str(value))
        if not path.is_absolute():
            path = (base / path).resolve()
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        result.append(key)
    return result


def dedupe_paths(paths: Iterable[str | Path]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in paths:
        key = str(Path(item))
        if key in seen:
            continue
        seen.add(key)
        result.append(key)
    return result


def prewarm_audio(
    paths: Sequence[str | Path],
    *,
    cache_dir: str | Path,
    sample_rate: int = 22050,
    hop_length: int = 441,
    workers: int | None = None,
    log: Any = None,
    progress_every: int = 25,
) -> dict[str, Any]:
    """Extract missing caches in parallel; returns counts and failures.

    ``cache_dir`` is the *dataset's* cache root (the value ``EventDataset``
    receives as ``cache_dir``); the canonical ``audio/`` subfolder is appended
    here.  Passing the audio folder itself nests a second ``audio/``, which
    leaves the dataset cold no matter how many songs were extracted.
    """

    worker_count = resolve_workers(workers)
    unique = dedupe_paths(paths)
    report: dict[str, Any] = {
        "songs": len(unique), "workers": worker_count,
        "cached": 0, "extracted": 0, "failed": 0, "failures": [],
        "seconds": 0.0,
    }
    if not unique:
        return report
    cache = str(Path(cache_dir))
    started = time.perf_counter()

    if worker_count <= 1:
        for index, path in enumerate(unique, start=1):
            _path, status, message = _extract_one((path, cache, int(sample_rate), int(hop_length)))
            report[status] = report.get(status, 0) + 1
            if status == "failed":
                report["failures"].append({"path": path, "error": message})
            if log and (index % progress_every == 0 or index == len(unique)):
                log(f"prewarm_audio_progress {index}/{len(unique)} extracted={report['extracted']} "
                    f"cached={report['cached']} failed={report['failed']}")
        report["seconds"] = time.perf_counter() - started
        return report

    tasks = [(path, cache, int(sample_rate), int(hop_length)) for path in unique]
    done = 0
    with ProcessPoolExecutor(max_workers=worker_count) as pool:
        futures = {pool.submit(_extract_one, task): task[0] for task in tasks}
        for future in as_completed(futures):
            path, status, message = future.result()
            done += 1
            report[status] = report.get(status, 0) + 1
            if status == "failed":
                report["failures"].append({"path": path, "error": message})
            if log and (done % progress_every == 0 or done == len(unique)):
                elapsed = time.perf_counter() - started
                remaining = len(unique) - done
                rate = done / elapsed if elapsed > 0 else 0.0
                eta = remaining / rate if rate > 0 else 0.0
                log(f"prewarm_audio_progress {done}/{len(unique)} extracted={report['extracted']} "
                    f"cached={report['cached']} failed={report['failed']} "
                    f"elapsed={elapsed:.0f}s eta={eta:.0f}s")
    report["seconds"] = time.perf_counter() - started
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pre-extract event audio features into the cache in parallel.",
    )
    parser.add_argument("manifest", help="training manifest (only audio paths are read)")
    parser.add_argument("--cache-dir", required=True,
                        help="dataset cache root (the trainer's --cache-dir); features go in its audio/ subfolder")
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--hop-length", type=int, default=441)
    parser.add_argument("--workers", default="auto",
                        help=f"process count, 'auto' (default {DEFAULT_WORKERS}) or 'off'")
    parser.add_argument("--out", help="write a JSON report here")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = audio_paths_from_manifest(args.manifest)
    if not paths:
        print("prewarm_audio: manifest references no audio files", file=sys.stderr)
        return 2
    workers = resolve_workers(args.workers)
    print(f"prewarm_audio songs={len(paths)} workers={workers} cache={args.cache_dir}", flush=True)
    report = prewarm_audio(
        paths, cache_dir=args.cache_dir, sample_rate=args.sample_rate,
        hop_length=args.hop_length, workers=workers, log=lambda text: print(text, flush=True),
    )
    per_song = (report["seconds"] / report["extracted"]) if report["extracted"] else 0.0
    print(f"prewarm_audio_done extracted={report['extracted']} cached={report['cached']} "
          f"failed={report['failed']} seconds={report['seconds']:.1f} "
          f"({per_song:.2f}s per extracted song, feature rate {FEATURE_RATE_HZ} Hz)", flush=True)
    for failure in report["failures"][:10]:
        print(f"prewarm_audio_failure {failure['path']}: {failure['error']}", file=sys.stderr)
    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    # A failed song is not fatal: training will surface it when that batch loads.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
