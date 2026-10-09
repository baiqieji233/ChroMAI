"""Parallel audio pre-extraction must be correct, idempotent and failure-safe.

Extraction is extracted out of the training loop because it costs ~1.4 s per song
and made the first epoch wait on the CPU.  The risks this pins down:

* the same song must never be analysed twice in one run, and already-cached
  songs must be skipped without re-analysis;
* a broken audio file must be reported, not raise out of a worker and kill the
  run;
* concurrent writers must not corrupt each other's cache (the npz temp name has
  to be unique per writer).
"""

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from maichart.generation.event_audio import extract_event_audio
from maichart.training.prewarm_audio import (
    DEFAULT_WORKERS,
    _extract_one,
    audio_paths_from_manifest,
    dedupe_paths,
    prewarm_audio,
    resolve_workers,
)


def _write_wav(path: Path, seconds: float = 0.6, sample_rate: int = 22050) -> None:
    samples = (np.sin(2 * np.pi * 220 * np.arange(int(seconds * sample_rate)) / sample_rate) * 8000)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.astype("<i2").tobytes())


def test_resolve_workers_accepts_auto_off_and_numbers():
    assert resolve_workers("auto") == DEFAULT_WORKERS
    assert resolve_workers(None) == DEFAULT_WORKERS
    assert resolve_workers("") == DEFAULT_WORKERS
    assert resolve_workers("off") == 0
    assert resolve_workers("0") == 0
    assert resolve_workers("4") == 4
    assert resolve_workers(3) == 3
    with pytest.raises(ValueError, match="进程数"):
        resolve_workers("many")
    assert DEFAULT_WORKERS >= 1


def test_dedupe_and_manifest_audio_paths(tmp_path):
    manifest = tmp_path / "training_manifest.json"
    manifest.write_text(json.dumps({
        "songs": [
            {"song_id": "a", "audio_path": "a/track.mp3"},
            {"song_id": "b", "audio_path": "b/track.mp3"},
            {"song_id": "c", "audio_path": "a/track.mp3"},  # duplicate file
            {"song_id": "d", "audio_path": None},           # no audio
        ]
    }), encoding="utf-8")
    paths = audio_paths_from_manifest(manifest)
    assert len(paths) == 2
    assert all(Path(path).is_absolute() for path in paths)
    assert dedupe_paths(["x", "x", "y"]) == ["x", "y"]


def test_prewarm_extracts_then_skips_cached_songs(tmp_path):
    audio = tmp_path / "song.wav"
    _write_wav(audio)
    cache = tmp_path / "cache"

    first = prewarm_audio([audio], cache_dir=cache, workers=1)
    assert first["extracted"] == 1 and first["cached"] == 0 and first["failed"] == 0
    assert first["songs"] == 1
    assert list((cache / "audio").glob("*.event_audio.json")), "a cache pair must be written"

    second = prewarm_audio([audio], cache_dir=cache, workers=1)
    assert second["extracted"] == 0 and second["cached"] == 1


def test_prewarm_writes_where_training_reads(tmp_path):
    """One cache location, or the features get extracted twice.

    The dataset reads ``<cache>/audio``; the gold/acceptance/prewarm paths used to
    write straight into ``<cache>``, so their features were never found again.
    """

    from maichart.generation.event_audio import audio_cache_dir, event_audio_cache_path

    audio = tmp_path / "song.wav"
    _write_wav(audio)
    cache = tmp_path / "cache"
    prewarm_audio([audio], cache_dir=cache, workers=1)
    expected = event_audio_cache_path(audio, audio_cache_dir(cache), sample_rate=22050,
                                      hop_length=441)
    assert expected is not None and expected.is_file()
    assert not list(cache.glob("*.event_audio.json")), "nothing may land in the cache root"


def test_trainer_hands_prewarm_the_cache_root(tmp_path):
    """Regression: the trainer passed ``<cache>/audio`` to ``prewarm_audio``.

    ``prewarm_audio`` appends ``audio/`` itself, so the trainer's extra segment
    nested a *second* folder: prewarm reported ``extracted=N`` into an orphan
    directory while ``EventDataset`` kept missing on ``<cache>/audio``.  Training
    then analysed every song inline, one ``event_audio_ready`` at a time, with
    the GPU idle -- and because prewarm's own "already cached?" probe reads the
    same wrong path, it called the corpus warm on every subsequent run.  Pinned
    without touching audio so it stays runnable anywhere.
    """

    from maichart.generation.event_audio import audio_cache_dir
    from maichart.training.train_events import prewarm_cache_root

    class _Dataset:
        cache_dir = tmp_path / "cache"

    dataset_reads = audio_cache_dir(_Dataset.cache_dir)
    assert dataset_reads == tmp_path / "cache" / "audio"
    # What prewarm will actually write, given the root the trainer hands it.
    assert audio_cache_dir(prewarm_cache_root(_Dataset())) == dataset_reads
    # The historical expression must not come back: it points somewhere else.
    assert audio_cache_dir(_Dataset.cache_dir / "audio") != dataset_reads


def test_prewarm_reports_failures_without_raising(tmp_path):
    missing = tmp_path / "nope.wav"
    report = prewarm_audio([missing], cache_dir=tmp_path / "cache", workers=1)
    assert report["failed"] == 1
    assert report["failures"] and "nope.wav" in report["failures"][0]["path"]


def test_extract_one_returns_status_tuples(tmp_path):
    audio = tmp_path / "song.wav"
    _write_wav(audio)
    cache = str(tmp_path / "cache")
    path, status, message = _extract_one((str(audio), cache, 22050, 441))
    assert status == "extracted" and message == ""
    # The dataset path must find exactly the same cache file.
    reloaded = extract_event_audio(audio, cache_dir=cache, sample_rate=22050, hop_length=441)
    assert reloaded.features.shape[1] == 98


def test_prewarm_uses_real_worker_processes(tmp_path):
    audios = []
    for index in range(3):
        path = tmp_path / f"song{index}.wav"
        _write_wav(path, seconds=0.4)
        audios.append(path)
    report = prewarm_audio(audios, cache_dir=tmp_path / "cache", workers=2)
    assert report["extracted"] == 3
    assert report["failed"] == 0
    assert report["workers"] == 2
    # A parallel run must leave no temporary files behind.
    assert not list((tmp_path / "cache").rglob("*.tmp*"))
    assert len(list((tmp_path / "cache" / "audio").glob("*.event_audio.json"))) == 3


def test_worker_pool_does_not_corrupt_a_shared_destination(tmp_path):
    """Two writers on the same song must both succeed and leave a valid cache."""

    audio = tmp_path / "song.wav"
    _write_wav(audio)
    cache = tmp_path / "cache"
    cache.mkdir()
    # Deliberately feed the same file twice to force two workers onto one target.
    report = prewarm_audio([audio, audio, audio], cache_dir=cache, workers=3)
    assert report["failed"] == 0
    assert report["songs"] == 1, "duplicate inputs collapse to one task"
    loaded = extract_event_audio(audio, cache_dir=cache)
    assert loaded.features.dtype == np.float32
    assert not list(cache.rglob("*.tmp*"))
