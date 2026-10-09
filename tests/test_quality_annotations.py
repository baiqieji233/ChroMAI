import json
from pathlib import Path

import pytest

from maichart.training.quality_annotations import (
    annotation_path,
    chart_sha256,
    iter_catalog,
    load_annotations,
    resolve_annotation,
    save_annotations,
)
from maichart.maidata import parse_maidata_file


def _manifest(tmp_path: Path) -> Path:
    song = tmp_path / "song"
    song.mkdir()
    (song / "maidata.txt").write_text(
        "&title=T\n&wholebpm=120\n&lv_5=13\n&des_5=D\n&inote_5=(120){4}1,2,3,4,\n",
        encoding="utf-8",
    )
    path = tmp_path / "training_manifest.json"
    path.write_text(json.dumps({"schema": "maichart-training-manifest-v1", "source_root": ".", "songs": [{
        "song_id": "song", "maidata_path": "song/maidata.txt", "audio_path": None,
        "difficulties": [{"index": 5, "level": "13", "designer": "D"}],
    }]}), encoding="utf-8")
    return path


def test_catalog_hash_changes_when_selected_slot_changes(tmp_path):
    manifest = _manifest(tmp_path)
    row = iter_catalog(manifest)[0]
    before = row.chart_sha256
    source = tmp_path / "song" / "maidata.txt"
    source.write_text(source.read_text(encoding="utf-8").replace("1,2,3,4", "1,2,3,4,5"), encoding="utf-8")
    assert chart_sha256(source, 5) != before


def test_atomic_schema_and_region_validation(tmp_path):
    manifest = _manifest(tmp_path)
    row = iter_catalog(manifest)[0]
    save_annotations(manifest, [{"chart_id": row.chart_id, "chart_sha256": row.chart_sha256,
        "label": "excellent", "weight": 2, "regions": [{"start_sec": 0, "end_sec": 1, "label": "flow", "weight": 1}]}])
    data = load_annotations(manifest)
    assert data["schema"] == "maichart-quality-annotations-v1"
    assert annotation_path(manifest).is_file()
    with pytest.raises(ValueError):
        save_annotations(manifest, [{"chart_id": row.chart_id, "label": "normal", "regions": [
            {"start_sec": 0, "end_sec": 2, "label": "a"}, {"start_sec": 1, "end_sec": 3, "label": "b"}]}])


def test_resolve_annotation_rejects_changed_content(tmp_path):
    manifest = _manifest(tmp_path)
    row = iter_catalog(manifest)[0]
    source = Path(row.source_path)
    chart = parse_maidata_file(source)
    record = {"chart_id": row.chart_id, "chart_sha256": row.chart_sha256, "label": "reduced", "weight": .25}
    assert resolve_annotation(record, source, chart, 5)
    source.write_text(source.read_text(encoding="utf-8").replace("1,2,3,4", "1,2,3,4,5"), encoding="utf-8")
    changed = parse_maidata_file(source)
    assert resolve_annotation(record, source, changed, 5) is None
