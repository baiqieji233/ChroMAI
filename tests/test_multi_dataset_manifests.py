import json
from pathlib import Path

import pytest

from maichart.training.dataset_catalog import (
    assert_no_validation_overlap,
    catalog_datasets,
    compose_datasets,
    merge_training_manifests,
    validation_overlap,
)
from maichart.training.quality_annotations import chart_sha256


def _manifest(path: Path, root: Path, songs: list[dict]) -> Path:
    path.write_text(json.dumps({
        "schema": "maichart-training-manifest-v1",
        "source_root": str(root),
        "songs": songs,
    }, ensure_ascii=False), encoding="utf-8")
    return path


def _song(song_id: str, maidata: Path, index: int = 5) -> dict:
    return {"song_id": song_id, "maidata_path": str(maidata), "audio_path": str(maidata.with_suffix(".wav")),
            "difficulties": [{"difficulty_index": index, "level_raw": "13", "usable_for_training": True}]}


def test_merge_namespaces_sources_and_deduplicates_real_chart(tmp_path):
    root_a = tmp_path / "quality"; root_b = tmp_path / "full"
    root_a.mkdir(); root_b.mkdir()
    shared = root_a / "song" / "maidata.txt"; shared.parent.mkdir(); shared.write_text("x", encoding="utf-8")
    first = _manifest(root_a / "training_manifest.json", root_a, [_song("same", shared)])
    second = _manifest(root_b / "training_manifest.json", root_a, [_song("same", shared), _song("other", root_a / "other" / "maidata.txt")])
    out = tmp_path / "merged.json"
    result = merge_training_manifests([first, second], out)
    assert result["song_count"] == 2
    assert result["difficulty_count"] == 2
    assert all("source_dataset" in item for item in result["songs"])
    assert all("::" in item["song_id"] for item in result["songs"])
    assert len({(song["maidata_path"], song["difficulties"][0]["difficulty_index"]) for song in result["songs"]}) == 2


def test_catalog_reports_titles_and_annotation_sidecar(tmp_path):
    root = tmp_path / "curated"; song = root / "folder"; song.mkdir(parents=True)
    maidata = song / "maidata.txt"
    maidata.write_text("&title=Readable Title\n&wholebpm=120\n&lv_5=13\n&inote_5={4}1,2,3,4,", encoding="utf-8")
    manifest = _manifest(root / "training_manifest.json", root, [_song("id", maidata)])
    cards = catalog_datasets([manifest])
    assert cards[0]["dataset_id"].startswith("ds-")
    assert cards[0]["name"] == "curated"
    assert cards[0]["difficulty_count"] == 1
    result = compose_datasets([manifest], tmp_path / "out")
    assert Path(result["quality_annotations"]).parent == Path(result["manifest"]).parent


def test_multi_source_quality_records_are_carried_to_composed_manifest(tmp_path):
    root = tmp_path / "curated"; song = root / "folder"; song.mkdir(parents=True)
    maidata = song / "maidata.txt"
    maidata.write_text("&title=Readable Title\n&wholebpm=120\n&lv_5=13\n&inote_5={4}1,2,3,4,", encoding="utf-8")
    manifest = _manifest(root / "training_manifest.json", root, [_song("id", maidata)])
    (root / "quality_annotations.json").write_text(json.dumps({
        "schema": "maichart-quality-annotations-v1", "records": [{
            "chart_id": "legacy", "song_id": "id", "difficulty_index": 5,
            "source_path": str(maidata), "chart_sha256": chart_sha256(maidata, 5), "label": "excellent", "weight": 2,
        }]
    }), encoding="utf-8")
    other_root = tmp_path / "other"; other_root.mkdir()
    other = _manifest(other_root / "training_manifest.json", other_root, [_song("other", other_root / "other" / "maidata.txt")])
    result = compose_datasets([manifest, other], tmp_path / "out")
    annotations = json.loads(Path(result["quality_annotations"]).read_text(encoding="utf-8"))
    assert annotations["records"][0]["song_id"].startswith("ds-")
    assert annotations["records"][0]["source_dataset"]["name"] == "curated"


def test_validation_overlap_checks_source_paths_and_audio(tmp_path):
    root = tmp_path / "data"; root.mkdir()
    maidata = root / "song" / "maidata.txt"; maidata.parent.mkdir(); maidata.write_text("x", encoding="utf-8")
    train = _manifest(tmp_path / "train.json", root, [_song("a", maidata)])
    validation = _manifest(tmp_path / "val.json", root, [_song("different", maidata)])
    assert validation_overlap(train, validation)[0]["reason"] == "same_maidata_difficulty"
    with pytest.raises(ValueError, match="重叠"):
        assert_no_validation_overlap(train, validation)
