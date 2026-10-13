import json
from pathlib import Path

import pytest

from maichart.webui.quality_api import QualityAPI
from maichart.training.quality_annotations import chart_id, chart_sha256, iter_catalog, load_annotations, save_annotations


def test_quality_api_catalog_save_detail_preview(tmp_path):
    song = tmp_path / "song"
    song.mkdir()
    (song / "maidata.txt").write_text("&title=T\n&wholebpm=120\n&lv_5=13\n&inote_5=(120){4}1,2,3,4,\n", encoding="utf-8")
    (song / "track.wav").write_bytes(b"audio")
    manifest = tmp_path / "training_manifest.json"
    manifest.write_text(json.dumps({"schema": "maichart-training-manifest-v1", "source_root": ".", "songs": [{
        "song_id": "song", "maidata_path": "song/maidata.txt", "audio_path": "song/track.wav",
        "difficulties": [{"index": 5, "level": "13"}],
    }]}), encoding="utf-8")
    api = QualityAPI()
    status, data = api.catalog({"manifest": str(manifest), "page": 1, "page_size": 1,
                                "label_filter": ""})
    assert status == 200 and data["total"] == 1
    chart_id = data["rows"][0]["chart_id"]
    status, saved = api.save({"manifest": str(manifest), "chart_ids": [chart_id], "label": "negative", "reason": "待修正"})
    assert status == 200 and saved["saved"] == 1
    status, detail = api.detail({"manifest": str(manifest), "chart_id": chart_id})
    assert status == 200 and detail["slot"]["duration_sec"] is not None
    status, preview = api.preview({"manifest": str(manifest)})
    assert status == 200 and "training_note" in preview
    status, headers, path, byte_range = api.get_audio({"manifest": str(manifest), "chart_id": chart_id}, range_header="bytes=0-2")
    assert status == 206 and headers["Content-Range"].startswith("bytes 0-2/") and byte_range == (0, 2)
    corrected = tmp_path / "corrected.txt"
    corrected.write_text("&title=T\n&wholebpm=120\n&lv_5=13\n&inote_5=(120){4}1,3,2,4,\n", encoding="utf-8")
    status, saved = api.save({"manifest": str(manifest), "chart_ids": [chart_id],
                              "label": "negative", "corrected_path": str(corrected),
                              "corrected_difficulty": 5})
    assert status == 200 and saved["summary"]["preference_pairs"] == 1
    corrected.write_text(corrected.read_text(encoding="utf-8").replace("&title=T", "&title=T\n&first=-1"), encoding="utf-8")
    with pytest.raises(ValueError, match="first/audio_first|时长不一致"):
        api.save({"manifest": str(manifest), "chart_ids": [chart_id],
                  "label": "negative", "corrected_path": str(corrected), "corrected_difficulty": 5})


def test_quality_catalog_uses_maidata_title_and_stable_sorting(tmp_path):
    rows = [
        ("z-folder", "Z manifest title", "A maidata title", "14", "Designer Z"),
        ("a-folder", "A manifest title", "B maidata title", "12+", "Designer A"),
        ("m-folder", "M manifest title", "C maidata title", "13.4", "Designer M"),
    ]
    songs = []
    for folder, manifest_title, maidata_title, level, designer in rows:
        song = tmp_path / folder
        song.mkdir()
        (song / "maidata.txt").write_text(
            f"&title={maidata_title}\n&wholebpm=120\n&lv_5={level}\n&des_5={designer}\n"
            "&inote_5=(120){4}1,2,3,4,\n", encoding="utf-8")
        songs.append({"song_id": folder, "title": manifest_title,
                      "maidata_path": f"{folder}/maidata.txt",
                      "difficulties": [{"index": 5, "level": level, "designer": designer}]})
    manifest = tmp_path / "training_manifest.json"
    manifest.write_text(json.dumps({"schema": "maichart-training-manifest-v1", "source_root": ".",
                                    "source_dataset": "精选谱面", "songs": songs}), encoding="utf-8")
    api = QualityAPI()

    status, data = api.catalog({"manifest": str(manifest), "sort_by": "level", "sort_order": "asc", "page_size": 2})
    assert status == 200
    assert data["pages"] == 2 and data["page"] == 1
    assert [row["title"] for row in data["rows"]] == ["B maidata title", "C maidata title"]
    assert all(row["source_dataset"] == "精选谱面" for row in data["rows"])
    assert data["rows"][0]["folder_name"] == "a-folder"

    status, data = api.catalog({"manifest": str(manifest), "sort_by": "title", "sort_order": "desc",
                                "page": 99, "page_size": 2})
    assert status == 200 and data["page"] == 2
    assert [row["title"] for row in data["rows"]] == ["A maidata title"]

    status, data = api.catalog({"manifest": str(manifest), "sort_by": "designer", "sort_order": "asc", "page_size": 10})
    assert [row["designer"] for row in data["rows"]] == ["Designer A", "Designer M", "Designer Z"]


def test_quality_catalog_namespaces_same_named_dataset_paths(tmp_path):
    manifests = []
    for root_name in ("source_a", "source_b"):
        dataset = tmp_path / root_name / "同名精选集"
        dataset.mkdir(parents=True)
        (dataset / "song" ).mkdir()
        (dataset / "song" / "maidata.txt").write_text(
            "&title=Same Song\n&wholebpm=120\n&lv_5=13\n&inote_5=(120){4}1,2,3,4,\n",
            encoding="utf-8",
        )
        manifest = dataset / "training_manifest.json"
        manifest.write_text(json.dumps({"schema": "maichart-training-manifest-v1", "source_root": ".",
                                        "source_dataset": "同名精选集", "songs": [{
                                            "song_id": "same-song", "maidata_path": "song/maidata.txt",
                                            "difficulties": [{"index": 5, "level": "13"}],
                                        }]}), encoding="utf-8")
        manifests.append(manifest)

    api = QualityAPI()
    rows = []
    for manifest in manifests:
        status, data = api.catalog({"manifest": str(manifest), "page_size": 10})
        assert status == 200 and len(data["rows"]) == 1
        rows.append(data["rows"][0])
    assert rows[0]["source_dataset"] == rows[1]["source_dataset"] == "同名精选集"
    assert rows[0]["chart_id"] != rows[1]["chart_id"]
    assert rows[0]["source_dataset_id"] != rows[1]["source_dataset_id"]


def test_quality_catalog_designer_uses_slot_then_global_then_manifest_fallback(tmp_path):
    song = tmp_path / "song"
    song.mkdir()
    source = song / "maidata.txt"
    source.write_text(
        "&title=Fallbacks\n&wholebpm=120\n&des=Global Designer\n"
        "&lv_1=10\n&inote_1=(120){4}1,\n"
        "&lv_2=11\n&des_2=Slot Designer\n&inote_2=(120){4}2,\n"
        "&lv_3=12\n&des_3=   \n&inote_3=(120){4}3,\n"
        "&lv_4=13\n&inote_4=(120){4}4,\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "training_manifest.json"
    manifest.write_text(json.dumps({
        "schema": "maichart-training-manifest-v1", "source_root": ".", "songs": [{
            "song_id": "fallbacks", "maidata_path": "song/maidata.txt", "difficulties": [
                {"index": 1, "level": "10", "designer": "Manifest One"},
                {"index": 2, "level": "11", "designer": "Manifest Two"},
                {"index": 3, "level": "12", "designer": "Manifest Three"},
                {"index": 4, "level": "13", "designer": "Manifest Four"},
            ],
        }],
    }), encoding="utf-8")

    rows = {row.difficulty_index: row for row in iter_catalog(manifest)}
    assert rows[1].designer == "Global Designer"
    assert rows[2].designer == "Slot Designer"
    assert rows[3].designer == "Global Designer"
    assert rows[4].designer == "Global Designer"

    # With no global or slot metadata, the existing manifest fallback remains.
    source.write_text(source.read_text(encoding="utf-8").replace("&des=Global Designer\n", ""), encoding="utf-8")
    rows = {row.difficulty_index: row for row in iter_catalog(manifest)}
    assert rows[1].designer == "Manifest One"
    assert rows[2].designer == "Slot Designer"
    assert rows[3].designer == "Manifest Three"
    assert rows[4].designer == "Manifest Four"

    # Global display metadata must not alter chart identity/hash semantics.
    source.write_text(source.read_text(encoding="utf-8").replace("&title=Fallbacks\n", "&title=Fallbacks\n&des=Changed Global\n"), encoding="utf-8")
    assert chart_sha256(source, 1) == rows[1].chart_sha256


def test_quality_save_migrates_old_namespaced_record_without_duplicates(tmp_path):
    song = tmp_path / "dataset" / "song"
    song.mkdir(parents=True)
    source = song / "maidata.txt"
    source.write_text("&title=Migration\n&wholebpm=120\n&lv_5=13\n&inote_5=(120){4}1,2,3,4,\n", encoding="utf-8")
    manifest = song.parent / "training_manifest.json"
    manifest.write_text(json.dumps({"schema": "maichart-training-manifest-v1", "source_root": ".",
                                    "source_dataset": "精选集", "songs": [{
                                        "song_id": "migration-song", "maidata_path": "song/maidata.txt",
                                        "difficulties": [{"index": 5, "level": "13"}],
                                    }]}), encoding="utf-8")
    source_hash = chart_sha256(source, 5)
    # This is the namespace format used before the source-path-aware identity.
    old_id = chart_id("migration-song", 5, source_hash, None, "精选集")
    save_annotations(manifest, [{"chart_id": old_id, "song_id": "migration-song", "difficulty_index": 5,
                                 "source_path": str(source.resolve()), "chart_sha256": source_hash,
                                 "source_dataset": "精选集", "label": "excellent", "weight": 2.0,
                                 "reason": "保留这个理由", "regions": [{"start_sec": 0, "end_sec": 1,
                                                                       "label": "重点", "reason": "保留片段", "weight": 1.5}]}])
    api = QualityAPI()
    status, catalog = api.catalog({"manifest": str(manifest), "page_size": 10})
    assert status == 200
    current_id = catalog["rows"][0]["chart_id"]
    assert current_id != old_id
    status, result = api.save({"manifest": str(manifest), "chart_ids": [current_id],
                               "label": "reduced", "weight": 0.25, "reason": "更新后的理由"})
    assert status == 200 and result["saved"] == 1
    records = load_annotations(manifest)["records"]
    assert len(records) == 1
    assert records[0]["chart_id"] == current_id
    assert records[0]["label"] == "reduced"
    assert records[0]["weight"] == pytest.approx(0.25)
    assert records[0]["source_dataset"] == "精选集"
    assert records[0]["source_dataset_id"] == catalog["rows"][0]["source_dataset_id"]
    assert records[0]["folder_name"] == "song"
    assert records[0]["maidata_title"] == "Migration"
