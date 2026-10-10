"""Exercise contracts, real source loading, and quality snapshot isolation."""
import json
from pathlib import Path

import pytest

from maichart.training.dataset_catalog import compose_datasets, merge_training_manifests
from maichart.training.quality_annotations import chart_sha256, save_annotations
from test_multi_dataset_webui import make_source


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, data):
    Path(path).write_text(json.dumps(data), encoding="utf-8")


@pytest.mark.parametrize("field,changed", [("target_schema", "frame_v2"), ("feature_profile", "basic_v1"),
                                         ("sample_rate", 44100), ("hop_length", 512)])
def test_incompatible_preprocessing_is_rejected(tmp_path, field, changed):
    a, b = make_source(tmp_path / "a", "A"), make_source(tmp_path / "b", "B")
    data = read(b); data["preprocessing"][field] = changed; write(b, data)
    with pytest.raises(ValueError, match="contracts differ"):
        compose_datasets([a, b], tmp_path / "combined")


def test_source_root_paths_load_in_event_ctx_dataset(tmp_path):
    pytest.importorskip("torch")
    from maichart.training.event_ctx_dataset import EventCtxDataset
    a, b = make_source(tmp_path / "a", "A"), make_source(tmp_path / "b", "B")
    derived = a.parent / "manifests" / "selected.json"
    derived.parent.mkdir()
    data = read(a); data["source_root"] = ".."; write(derived, data)
    result = compose_datasets([derived, b], tmp_path / "combined", "event_ctx_v1")
    merged = read(result["manifest"])
    assert all(Path(song["maidata_path"]).is_file() and Path(song["audio_path"]).is_file() for song in merged["songs"])
    assert merged["preprocessing"]["target_schema"] == "event_v1"
    assert merged["preprocessing"]["feature_profile"] == "event_audio_v2_rhythm"
    assert merged["preprocessing"]["hop_length"] == 441
    dataset = EventCtxDataset(result["manifest"], cache_dir=result["cache_dir"], eager_audio=False,
                              context_beats=4, context_tokens=32, max_tokens=128, planner_profile="music_structure_v2")
    assert len(dataset.samples) == 2
    assert not any(item.get("reason") == "missing_source" for item in dataset.skipped)


def test_single_source_keeps_original_manifest_and_bytes(tmp_path):
    a = make_source(tmp_path / "a", "A")
    before = a.read_bytes()
    result = compose_datasets([a, a], tmp_path / "combined", "event_ctx_v1")
    assert result["manifest"] == str(a)
    assert result["raw_root"] == str(a.parent)
    assert a.read_bytes() == before
    assert not (tmp_path / "combined").exists()


def annotation(manifest, **extra):
    source = manifest.parent / "song" / "maidata.txt"
    return {"chart_id": "legacy", "song_id": "same-song-id", "source_path": str(source),
            "difficulty_index": 5, "chart_sha256": chart_sha256(source, 5),
            "label": "excellent", "weight": 2.0, "regions": [], **extra}


def test_quality_edit_creates_new_snapshot_and_preserves_old_one(tmp_path):
    a, b = make_source(tmp_path / "a", "A"), make_source(tmp_path / "b", "B")
    save_annotations(a, [annotation(a)])
    first = compose_datasets([a, b], tmp_path / "combined", "event_ctx_v1")
    before = Path(first["quality_annotations"]).read_bytes()
    save_annotations(a, [annotation(a, label="reduced", weight=.25)])
    second = compose_datasets([b, a], tmp_path / "combined", "event_ctx_v1")
    assert first["manifest"] != second["manifest"]
    assert Path(first["quality_annotations"]).read_bytes() == before
    assert read(second["quality_annotations"])["records"][0]["weight"] == .25
    repeated = compose_datasets([a, b], tmp_path / "combined", "event_ctx_v1")
    assert repeated["manifest"] == second["manifest"]
    assert read(repeated["manifest"])["songs"] == read(second["manifest"])["songs"]


def test_relative_correction_survives_real_quality_loading(tmp_path):
    pytest.importorskip("torch")
    from maichart.training.event_ctx_dataset import EventCtxDataset
    a, b = make_source(tmp_path / "a", "A"), make_source(tmp_path / "b", "B")
    correction = a.parent / "correction.txt"
    correction.write_bytes((a.parent / "song" / "maidata.txt").read_bytes())
    save_annotations(a, [annotation(a, corrected_path="correction.txt", corrected_difficulty=5,
                                   corrected_sha256=chart_sha256(correction, 5))])
    result = compose_datasets([a, b], tmp_path / "combined", "event_ctx_v1")
    records = read(result["quality_annotations"])["records"]
    assert records[0]["corrected_path"] == str(correction)
    dataset = EventCtxDataset(result["manifest"], cache_dir=result["cache_dir"], eager_audio=False,
                              quality_annotations=result["quality_annotations"], context_beats=4, context_tokens=32,
                              max_tokens=128, planner_profile="music_structure_v2")
    assert len(dataset.quality_records_by_sample) == 1
    assert 2.0 in dataset.segment_quality_weights


def test_shared_source_region_conflict_is_not_silently_chosen(tmp_path):
    a = make_source(tmp_path / "a", "A")
    other = tmp_path / "other" / "training_manifest.json"
    other.parent.mkdir()
    data = read(a); data["source_root"] = str(a.parent); write(other, data)
    save_annotations(a, [annotation(a)])
    save_annotations(other, [annotation(a, regions=[{"start_sec": 0, "end_sec": .5, "weight": .5}])])
    with pytest.raises(ValueError, match="冲突质量标注"):
        compose_datasets([a, other], tmp_path / "combined", "event_ctx_v1")


def test_stale_labels_are_reported_and_not_applied(tmp_path):
    a, b = make_source(tmp_path / "a", "A"), make_source(tmp_path / "b", "B")
    save_annotations(a, [annotation(a, chart_sha256="stale")])
    result = compose_datasets([a, b], tmp_path / "combined", "event_ctx_v1")
    assert result["summary"]["stale_quality_annotations"] == 1
    assert read(result["quality_annotations"])["records"] == []


def test_old_cache_reuse_and_snapshot_quality_are_separate(tmp_path):
    a, b = make_source(tmp_path / "a", "A"), make_source(tmp_path / "b", "B")
    (a.parent / "cache").mkdir()
    result = compose_datasets([a, b], tmp_path / "combined", "event_ctx_v1")
    assert result["cache_dir"] == str(a.parent / "cache")
    assert Path(result["quality_annotations"]).parent == Path(result["manifest"]).parent
    assert Path(result["quality_annotations"]).parent != a.parent
