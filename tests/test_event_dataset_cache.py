"""The resident audio cache must be bounded, and spilling must stay correct.

Decoded features are ~3 MB per song, so an unbounded dict grows linearly with the
corpus (3000 songs would hold ~9 GB).  ``EventDataset`` now keeps an LRU bounded
by bytes; these tests pin its behaviour:

  * a resident song is reused (no reload),
  * exceeding the budget evicts the least-recently-used song,
  * an evicted song reloads and returns correct data,
  * a zero budget disables retention entirely,
  * ``None`` restores the old unbounded behaviour,
  * replaying a spilled song goes through its cache file instead of re-analysing
    the source audio.
"""

import json

import numpy as np
import pytest

from maichart.event_timing import parse_event_chart_text
from maichart.generation.event_audio import ACOUSTIC_DIM, EventAudio
from maichart.training import event_dataset as event_dataset_module
from maichart.training import prepare_events
from maichart.training.event_dataset import EventDataset, EventSample

FEATURE_FRAMES = 500  # 500 frames x ACOUSTIC_DIM columns is ~190 KB per song


def test_window_plan_uses_parsed_heads_and_recognizes_straight_slides(tmp_path):
    chart = parse_event_chart_text(
        "&title=Plan\n&wholebpm=120\n&first=0\n&lv_4=13\n"
        "&inote_4=(120){4}12,1-2[4:1],\n",
        4,
    )
    ref = EventSample("plan", tmp_path / "audio.mp3", chart, 13.0, tmp_path / "maidata.txt")
    plan = EventDataset._window_plan(ref, 0, 4)
    # The straight '-' slide is a parsed slide note, and the compact '12'
    # timing point contributes two playable heads to the density/chord target.
    assert plan[0].item() > 0.0
    assert plan[2].item() > 0.0
    assert plan[3].item() > 0.0
    assert plan[5].item() == pytest.approx(2.0 / 3.0)


def _write_chart(root, name: str, notes: str = "(134){4}1,2,3,4," * 8) -> None:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "maidata.txt").write_text(
        "&title=Test Song\n&artist=Tester\n&wholebpm=134\n&first=0\n"
        f"&lv_4=13\n&inote_4={notes}\n",
        encoding="utf-8",
    )
    (directory / "track.mp3").write_bytes(b"\x00" * 32)


def _fake_audio(_path, **_kwargs) -> EventAudio:
    return EventAudio(
        features=np.zeros((FEATURE_FRAMES, ACOUSTIC_DIM), dtype=np.float32),
        times=np.arange(FEATURE_FRAMES, dtype=np.float32) * 0.02,
        duration_sec=10.0,
        tempo_bpm=134.0,
        tempo_changes=[],
        metadata={},
    )


@pytest.fixture
def manifest(tmp_path):
    raw = tmp_path / "raw"
    for index in range(4):
        _write_chart(raw, f"song-{index}")
    output = tmp_path / "manifest.json"
    assert prepare_events.main([str(raw), "--cache-dir", str(tmp_path / "cache"), "-o", str(output)]) == 0
    return output, tmp_path / "cache"


def _dataset(manifest, cache, **kwargs) -> EventDataset:
    return EventDataset(manifest, cache_dir=cache, segment_beats=16, max_tokens=256,
                        eager_audio=False, **kwargs)


def test_bounded_cache_reuses_resident_songs(monkeypatch, manifest):
    calls = []

    def counting_extract(path, **kwargs):
        calls.append(str(path))
        return _fake_audio(path)

    monkeypatch.setattr(event_dataset_module, "extract_event_audio", counting_extract)
    dataset = _dataset(*manifest, audio_cache_bytes=10 * 1024 ** 2)
    ref = dataset.samples[0]

    first = dataset._load_audio(ref)
    second = dataset._load_audio(ref)
    assert first is second, "a resident song must be handed back without reloading"
    assert len(calls) == 1
    assert dataset.audio_cache_evictions == 0


def test_budget_evicts_least_recently_used_and_reloads_correctly(monkeypatch, manifest):
    per_song = FEATURE_FRAMES * ACOUSTIC_DIM * 4 + FEATURE_FRAMES * 4
    calls = []

    def counting_extract(path, **kwargs):
        calls.append(str(path))
        return _fake_audio(path)

    monkeypatch.setattr(event_dataset_module, "extract_event_audio", counting_extract)
    # Room for exactly two songs.
    dataset = _dataset(*manifest, audio_cache_bytes=per_song * 2 + 1)
    first, second, third = dataset.samples[:3]

    dataset._load_audio(first)
    dataset._load_audio(second)
    assert len(dataset.audio) == 2
    dataset._load_audio(third)  # must evict `first`

    assert len(dataset.audio) == 2, "resident songs must stay within the budget"
    assert dataset.audio_cache_evictions == 1
    assert str(first.audio_path) not in dataset.audio
    assert str(second.audio_path) in dataset.audio

    # The evicted song reloads and still yields the right shape.
    reloaded = dataset._load_audio(first)
    assert reloaded.features.shape == (FEATURE_FRAMES, ACOUSTIC_DIM)
    assert len(calls) == 4
    assert dataset._audio_bytes <= dataset.audio_cache_bytes


def test_zero_budget_disables_caching(monkeypatch, manifest):
    calls = []

    def counting_extract(path, **kwargs):
        calls.append(str(path))
        return _fake_audio(path)

    monkeypatch.setattr(event_dataset_module, "extract_event_audio", counting_extract)
    dataset = _dataset(*manifest, audio_cache_bytes=0)
    ref = dataset.samples[0]

    dataset._load_audio(ref)
    dataset._load_audio(ref)
    assert len(calls) == 2, "nothing may be retained when caching is disabled"
    assert len(dataset.audio) == 0
    assert dataset._audio_bytes == 0


def test_none_budget_keeps_the_old_unbounded_behaviour(monkeypatch, manifest):
    monkeypatch.setattr(event_dataset_module, "extract_event_audio", _fake_audio)
    dataset = _dataset(*manifest, audio_cache_bytes=None)
    for ref in dataset.samples:
        dataset._load_audio(ref)
    assert len(dataset.audio) == len(dataset.samples)
    assert dataset.audio_cache_evictions == 0


def test_spilled_song_reloads_from_its_cache_file_without_reanalysis(monkeypatch, tmp_path, manifest):
    """A spill reload must read the npz cache, not re-run the audio analysis."""

    cache_manifest, cache_dir = manifest
    audio_cache = cache_dir / "audio"
    audio_cache.mkdir(parents=True, exist_ok=True)
    written = []

    def writing_extract(path, **kwargs):
        # Emulate the real extractor: write a cache pair and count analyses.
        written.append(str(path))
        target = audio_cache / f"{len(written)}.event_audio.json"
        payload = {
            "schema": "maichart-event-audio-v1",
            "sample_rate": kwargs.get("sample_rate", 22050),
            "hop_length": kwargs.get("hop_length", 441),
            "feature_version": "event_audio_v1",
            "duration_sec": 10.0,
            "tempo_bpm": 134.0,
            "tempo_changes": [],
            "features": None,
        }
        npz = target.with_suffix(".npz")
        np.savez(npz, features=np.zeros((FEATURE_FRAMES, ACOUSTIC_DIM), dtype=np.float32),
                 times=np.arange(FEATURE_FRAMES, dtype=np.float32) * 0.02)
        payload["npz"] = npz.name
        target.write_text(json.dumps(payload), encoding="utf-8")
        return _fake_audio(path)

    monkeypatch.setattr(event_dataset_module, "extract_event_audio", writing_extract)

    dataset = _dataset(cache_manifest, cache_dir, audio_cache_bytes=1)
    ref = dataset.samples[0]
    dataset._load_audio(ref)
    analyses_after_first = len(written)
    dataset._load_audio(ref)  # evicted immediately by the 1-byte budget
    assert len(written) == analyses_after_first, "the spill must not re-analyse the audio"


def test_audio_cache_is_bounded_by_default():
    from maichart.training.event_dataset import DEFAULT_AUDIO_CACHE_BYTES

    assert 0 < DEFAULT_AUDIO_CACHE_BYTES <= 8 * 1024 ** 3


def test_audio_loads_are_logged_aggregated_by_default(monkeypatch, manifest, capsys):
    """One line per song floods the 300-line WebUI log during the first epoch.

    The default now reports every 25th load with the resident size and spill
    count, which still shows the loading curve without hiding the epoch lines.
    """

    monkeypatch.setattr(event_dataset_module, "extract_event_audio",
                        lambda path, **_kwargs: _fake_audio(path))
    # A zero-byte budget disables retention, so every call is a fresh load.
    dataset = _dataset(*manifest, audio_cache_bytes=0, log_audio_loads=False,
                       audio_cache_label="train")
    ref = dataset.samples[0]
    for _ in range(25):
        dataset._load_audio(ref)
    # Validation owns a separate EventDataset and starts its own counter at 1.
    validation = _dataset(*manifest, audio_cache_bytes=0, log_audio_loads=False,
                           audio_cache_label="validation")
    validation._load_audio(validation.samples[0])
    out = capsys.readouterr().out
    assert "event_audio_ready=" not in out, "per-song lines must be off by default"
    assert "event_audio_loads=1 " in out
    assert "event_audio_loads=25 " in out
    assert "dataset=train" in out and "dataset=validation" in out
    assert "resident=" in out and "spills=" in out
    assert "event_audio_loads=2 " not in out, "no line for every load"


def test_audio_loads_can_be_logged_per_song(monkeypatch, manifest, capsys):
    """``--log-audio-loads`` restores the per-song detail for debugging."""

    monkeypatch.setattr(event_dataset_module, "extract_event_audio",
                        lambda path, **_kwargs: _fake_audio(path))
    dataset = _dataset(*manifest, audio_cache_bytes=0, log_audio_loads=True)
    dataset._load_audio(dataset.samples[0])
    dataset._load_audio(dataset.samples[1])
    out = capsys.readouterr().out
    assert out.count("event_audio_ready=") == 2
    assert "event_audio_loads=" not in out


def test_cache_summary_tracks_loads_and_spills(monkeypatch, manifest):
    monkeypatch.setattr(event_dataset_module, "extract_event_audio",
                        lambda path, **_kwargs: _fake_audio(path))
    dataset = _dataset(*manifest, audio_cache_bytes=1)
    for ref in dataset.samples:
        dataset._load_audio(ref)
    summary = dataset.audio_cache_summary()
    assert "loads=" in summary and "spills=" in summary
    assert f"loads={len(dataset.samples)}" in summary
    assert "spills=0" not in summary, "a 1-byte budget must spill"


def test_trainer_reports_the_cache_every_epoch():
    """Thrash must be visible per epoch, not only once at startup."""

    import re
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "src/maichart/training/train_events.py").read_text(encoding="utf-8")
    assert "dataset_audio_cache epoch=" in source
    assert '"--log-audio-loads"' in source
    assert re.search(r'log_audio_loads=args\.log_audio_loads', source)
