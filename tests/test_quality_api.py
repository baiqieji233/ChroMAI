import json
from pathlib import Path

import pytest

from maichart.webui.quality_api import QualityAPI


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
    with pytest.raises(ValueError, match="first/audio_first"):
        api.save({"manifest": str(manifest), "chart_ids": [chart_id],
                  "label": "negative", "corrected_path": str(corrected), "corrected_difficulty": 5})
