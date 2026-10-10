import json
from pathlib import Path

import pytest

from maichart.training.dataset_catalog import compose_datasets
from maichart.training.rhythm_report import analyze_training_manifest


def _write_manifest(path: Path, source_root: Path) -> Path:
    song_dir = source_root / "song"
    song_dir.mkdir(parents=True, exist_ok=True)
    maidata = song_dir / "maidata.txt"
    maidata.write_text(
        "&title=Rhythm Sample\n&wholebpm=120\n&lv_5=12\n&inote_5={16}1,2,3,4,E,",
        encoding="utf-8",
    )
    payload = {
        "schema": "maichart-training-manifest-v1",
        "source_root": str(source_root),
        "songs": [{
            "song_id": "rhythm-sample",
            "title": "Rhythm Sample",
            "maidata_path": "song/maidata.txt",
            "difficulties": [
                {"difficulty_index": 5, "level_raw": "12", "event_usable_for_training": True},
                {"difficulty_index": 4, "level_raw": "10", "event_usable_for_training": True},
            ],
        }],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_report_reads_composed_manifest_and_keeps_source_unchanged(tmp_path):
    source_root = tmp_path / "source"
    original_manifest = _write_manifest(source_root / "training_manifest.json", source_root)
    before = original_manifest.read_bytes()
    composed = compose_datasets([original_manifest], tmp_path / "composed")
    manifest_path = Path(composed["manifest"])
    report_path = tmp_path / "report.json"

    report = analyze_training_manifest(
        manifest_path,
        report_path,
        window_beats=1,
        difficulties={5},
        min_level=12,
        max_level=12,
    )

    assert original_manifest.read_bytes() == before
    assert report["summary"]["processed_difficulties"] == 1
    assert report["summary"]["filtered"] == {"difficulty": 1}
    assert report["summary"]["skipped"] == {}
    chart = report["charts"][0]
    assert Path(chart["source_path"]).is_absolute()
    assert chart["source_path"].endswith("song/maidata.txt") or chart["source_path"].endswith("song\\maidata.txt")
    first_window = report["window_examples"][0]
    assert first_window["start_beat"] == "0"
    assert first_window["end_beat"] == "1"
    assert first_window["start_sec"] == 0.0
    assert first_window["end_sec"] == 0.5
    assert report["per_level"]["12"]["chart_count"] == 1
    assert report["summary"]["fine_note_ratio"]["mean"] == 1.0
    assert json.loads(report_path.read_text(encoding="utf-8"))["schema"] == "maichart-rhythm-dataset-report-v1"


def test_report_records_chart_failures_and_limit(tmp_path):
    root = tmp_path / "dataset"
    manifest = _write_manifest(root / "training_manifest.json", root)
    report = analyze_training_manifest(manifest, limit=1, window_beats=4)

    assert report["summary"]["selected_difficulties"] == 1
    assert report["summary"]["limited_out"] == 1
    assert report["summary"]["processed_difficulties"] == 1

    all_report = analyze_training_manifest(
        manifest, tmp_path / "all.json", difficulties={4}, window_beats=4,
    )
    assert all_report["summary"]["skipped"] == {"chart_parse_or_validation_error": 1}


def test_report_rejects_output_equal_to_source_manifest(tmp_path):
    root = tmp_path / "dataset"
    manifest = _write_manifest(root / "training_manifest.json", root)
    before = manifest.read_bytes()

    with pytest.raises(ValueError, match="must not overwrite"):
        analyze_training_manifest(manifest, manifest)

    assert manifest.read_bytes() == before
