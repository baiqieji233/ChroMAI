"""Whole-source selection must cover hidden rows and every page."""
import json
from pathlib import Path

from maichart.training.quality_annotations import load_annotations
from maichart.webui.quality_api import QualityAPI


def source(root: Path, count: int) -> Path:
    songs = []
    for index in range(count):
        folder = root / f"song_{index:03d}"
        folder.mkdir(parents=True)
        (folder / "maidata.txt").write_text(
            f"&title=Song {index:03d}\n&wholebpm=120\n&lv_5=13\n&des=Global Designer\n&inote_5=(120){{4}}1,2,3,4,\n",
            encoding="utf-8")
        songs.append({"song_id": folder.name, "maidata_path": f"{folder.name}/maidata.txt",
                      "difficulties": [{"index": 5, "level": "13"}]})
    manifest = root / "training_manifest.json"
    manifest.write_text(json.dumps({"schema": "maichart-training-manifest-v1", "source_root": ".", "songs": songs}), encoding="utf-8")
    return manifest


def test_folder_selection_exceeds_page_limit_and_bulk_save_stays_in_source(tmp_path):
    a = source(tmp_path / "a", 503)
    b = source(tmp_path / "b", 1)
    api = QualityAPI()
    status, filtered = api.catalog({"manifest": str(a), "query": "not-in-this-folder", "label_filter": "negative", "page_size": 500})
    assert status == 200
    assert filtered["total"] == 0 and filtered["folder_total"] == 503
    status, selection = api.post("/api/quality/selection", {"manifest": str(a), "query": "not-in-this-folder", "page_size": 1})
    assert status == 200
    assert selection["scope"] == "current_dataset"
    assert selection["total"] == len(set(selection["chart_ids"])) == 503
    assert not (a.parent / "quality_annotations.json").exists()
    _, other = api.selection({"manifest": str(b)})
    assert not set(selection["chart_ids"]) & set(other["chart_ids"])
    status, saved = api.save({"manifest": str(a), "chart_ids": selection["chart_ids"], "label": "excellent", "weight": 2})
    assert status == 200 and saved["saved"] == 503
    records = load_annotations(a)["records"]
    assert len(records) == 503
    assert all(record["label"] == "excellent" and record["weight"] == 2 for record in records)
    assert not (b.parent / "quality_annotations.json").exists()


def test_empty_folder_selection_is_read_only(tmp_path):
    manifest = tmp_path / "training_manifest.json"
    manifest.write_text(json.dumps({"schema": "maichart-training-manifest-v1", "songs": []}), encoding="utf-8")
    status, selection = QualityAPI().selection({"manifest": str(manifest)})
    assert status == 200 and selection["chart_ids"] == [] and selection["total"] == 0
    assert not (tmp_path / "quality_annotations.json").exists()
