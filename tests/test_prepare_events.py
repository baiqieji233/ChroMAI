"""The event preprocessor must emit a manifest the dataset and WebUI accept.

``maichart.training.prepare_events`` is the entry point the WebUI dispatches for
event_v1 preprocessing.  Without it the button failed outright, so these tests
pin the two fields the WebUI branches on and the paths the loader resolves.
"""

import json
from pathlib import Path

from maichart.training import prepare_events
from maichart.training.event_dataset import EventDataset, resolve_path


def _write_chart(root: Path, name: str, *, level: str, notes: str, audio: bool = True) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "maidata.txt").write_text(
        "&title=Test Song\n&artist=Tester\n&wholebpm=134\n&first=0\n"
        f"&lv_4={level}\n&inote_4={notes}\n",
        encoding="utf-8",
    )
    if audio:
        (directory / "track.mp3").write_bytes(b"\x00" * 32)
    return directory


def test_manifest_is_marked_as_an_event_dataset(tmp_path):
    raw = tmp_path / "raw"
    _write_chart(raw, "song-a", level="13+", notes="(134){4}1,2,3,4,\n")
    manifest = prepare_events.build_event_manifest(
        raw, cache_dir=tmp_path / "cache",
    )

    preprocessing = manifest["preprocessing"]
    # The WebUI decides whether a manifest is an event manifest from these two
    # fields alone; missing them makes it read an event dataset as legacy.
    assert preprocessing["target_schema"] == "event_v1"
    assert preprocessing["feature_profile"] in {"event_audio_v1", "event_audio_v2_rhythm"}
    assert manifest["schema"] == "maichart-training-manifest-v1"


def test_manifest_paths_resolve_after_writing(tmp_path):
    """Stored paths must resolve against the manifest, absolute or relative.

    A manifest written outside the raw root cannot express its source paths
    relatively, so it stores absolute paths; one written inside the raw root
    stores relative ones.  Both are valid as long as the loader resolves them.
    """

    raw = tmp_path / "raw"
    _write_chart(raw, "song-a", level="13", notes="(134){4}1,2,3,4," * 8 + "\n")
    output = tmp_path / "out" / "manifest.json"
    exit_code = prepare_events.main([
        str(raw), "--cache-dir", str(tmp_path / "cache"), "-o", str(output),
    ])
    assert exit_code == 0

    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["song_count"] == 1
    song = manifest["songs"][0]
    for key in ("maidata_path", "audio_path"):
        assert resolve_path(song[key], output.parent).is_file()


def test_manifest_inside_the_raw_root_stores_relative_paths(tmp_path):
    raw = tmp_path / "raw"
    _write_chart(raw, "song-a", level="13", notes="(134){4}1,2,3,4," * 8 + "\n")
    output = raw / "manifest.json"
    assert prepare_events.main([
        str(raw), "--cache-dir", str(tmp_path / "cache"), "-o", str(output),
    ]) == 0

    song = json.loads(output.read_text(encoding="utf-8"))["songs"][0]
    assert not Path(song["maidata_path"]).is_absolute()
    assert resolve_path(song["maidata_path"], output.parent).is_file()


def test_below_master_difficulties_are_reported_unusable(tmp_path):
    raw = tmp_path / "raw"
    directory = raw / "song-a"
    directory.mkdir(parents=True)
    # index 2 is Expert, which event training never uses.
    (directory / "maidata.txt").write_text(
        "&title=Test Song\n&wholebpm=134\n&lv_2=13\n&inote_2=(134){4}1,2,3,4,\n",
        encoding="utf-8",
    )
    (directory / "track.mp3").write_bytes(b"\x00" * 32)

    manifest = prepare_events.build_event_manifest(raw, cache_dir=tmp_path / "cache")
    difficulty = manifest["songs"][0]["difficulties"][0]
    assert difficulty["difficulty_index"] == 2
    assert difficulty["event_usable_for_training"] is False
    assert "difficulty_index_below_master" in difficulty["filter_reasons"]
    assert manifest["usable_difficulty_count"] == 0


def test_missing_audio_is_filtered_and_recorded(tmp_path):
    raw = tmp_path / "raw"
    _write_chart(raw, "song-a", level="13", notes="(134){4}1,2,3,4,\n", audio=False)
    manifest = prepare_events.build_event_manifest(raw, cache_dir=tmp_path / "cache")

    difficulty = manifest["songs"][0]["difficulties"][0]
    assert "audio_missing" in difficulty["filter_reasons"]
    assert manifest["songs"][0]["audio"]["readable"] is False


def test_generated_manifest_is_loadable_by_the_dataset(tmp_path):
    """End to end: build a manifest and let EventDataset consume it."""

    raw = tmp_path / "raw"
    # Four beats per cell at division 4, repeated so the chart has real points.
    _write_chart(raw, "song-a", level="13", notes="(134){4}1,2,3,4," * 8 + "\n")
    output = tmp_path / "manifest.json"
    assert prepare_events.main([
        str(raw), "--cache-dir", str(tmp_path / "cache"), "-o", str(output),
    ]) == 0

    dataset = EventDataset(output, cache_dir=tmp_path / "cache", segment_beats=16, eager_audio=False)
    assert len(dataset.samples) == 1
    assert dataset.segments
