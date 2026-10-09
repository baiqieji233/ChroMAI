"""Calibration keeps the training duration and labels auditable."""
import json

import pytest

from maichart.generation.conditions import GENERATION_DEFAULTS, RADAR_KEYS, STYLE_KEYS
from maichart.generation import preset_calibration as calibration


def test_weighted_quantile_and_monotonic_fit():
    assert calibration.weighted_quantile([.1, .8, .9], [10, 1, 1], .5) == .1
    assert calibration.weighted_quantile([.1, .8, .9], [10, 1, 1], .9) == .8
    assert calibration.isotonic([.2, .5, .3, .7], [1, 1, 3, 1]) == pytest.approx([.2, .35, .35, .7])
    with pytest.raises(ValueError):
        calibration.calibrate_levels([])


def test_collection_uses_real_audio_duration_and_keeps_approximate_labels_visible(tmp_path, monkeypatch):
    chart = tmp_path / "maidata.txt"
    chart.write_text("&title=Demo\n&first=0\n&lv_5=13.4\n&inote_5=(120){4}1,2,3,4,E\n", encoding="utf-8")
    (tmp_path / "audio.wav").write_bytes(b"audio header fixture")
    songs = []
    for index, level in enumerate(("13.4", "13+", "14")):
        source = tmp_path / f"chart_{index}.txt"
        source.write_bytes(chart.read_bytes())
        songs.append({"song_id": f"chart_{index}", "maidata_path": source.name, "audio_path": "audio.wav",
                      "difficulties": [{"difficulty_index": 5, "level_raw": level, "event_usable_for_training": True}]})
    songs.append({"song_id": "missing", "maidata_path": chart.name, "audio_path": "missing.wav",
                  "difficulties": [{"difficulty_index": 5, "level_raw": "13.4", "event_usable_for_training": True}]})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"songs": songs}), encoding="utf-8")
    monkeypatch.setattr(calibration, "_audio_duration", lambda _: 10)
    rows, audit = calibration.collect_chart_controls(manifest)
    assert len(rows) == 3
    assert rows[0]["controls"]["note_density"] == pytest.approx(4 / 10 / 16)
    assert rows[0]["audio_duration_sec"] == 10
    assert rows[0]["label_weight"] == 1
    assert rows[1]["level"] == 13.7
    assert rows[1]["label_kind"] == "display_plus"
    assert rows[2]["label_kind"] == "display_integer"
    assert rows[1]["label_weight"] == rows[2]["label_weight"] == .35
    assert rows[0]["controls"]["melody_following"] == .9
    assert audit["duration_source"] == "audio_header"
    assert audit["skipped"] == 1
    assert audit["skip_reasons"]["missing_source"] == 1


def test_levels_borrow_sparse_neighbors_and_preserve_control_bounds():
    rows = []
    for index, level in enumerate((12.1, 13.3, 13.4, 14.8)):
        values = {key: GENERATION_DEFAULTS[key] for key in RADAR_KEYS + STYLE_KEYS}
        values["note_density"] = [.15, .35, .3, .5][index]
        values["peak"] = [.2, .45, .4, .7][index]
        rows.append({"song_id": str(index), "level": level, "label_kind": "decimal", "controls": values})
    profiles = calibration.calibrate_levels(rows)
    assert [row["level"] for row in profiles] == [round(12 + index / 10, 1) for index in range(31)]
    assert profiles[0]["fallback"] is True
    assert profiles[0]["raw_count"] == 0
    assert profiles[0]["bandwidth"] == .45
    for key in ("note_density", "peak"):
        values = [row["radar"][key] for row in profiles]
        assert values == sorted(values)
    for row in profiles:
        assert row["style"]["chord_bias"] == row["radar"]["chord"]
        for key, value in {**row["radar"], **row["style"]}.items():
            assert 0 <= row["controls"][key]["p10"] <= value <= row["controls"][key]["p90"] <= 1


def test_calibration_cli_writes_complete_profile_and_audit(tmp_path, monkeypatch):
    rows = [{"song_id": str(level), "level": level, "label_kind": "decimal",
             "controls": {key: GENERATION_DEFAULTS[key] for key in RADAR_KEYS + STYLE_KEYS}}
            for level in (12.0, 13.5, 15.0)]
    manifest, output, audit_path = (tmp_path / name for name in ("manifest.json", "presets.json", "audit.json"))
    manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(calibration, "collect_chart_controls", lambda _: (rows, {"charts": 3, "duration_source": "audio_header"}))
    assert calibration.main(["--manifest", str(manifest), "--output", str(output), "--audit", str(audit_path)]) == 0
    profile = json.loads(output.read_text(encoding="utf-8"))
    assert profile["schema"] == calibration.SCHEMA
    assert len(profile["levels"]) == 31
    assert profile["source"]["duration_source"] == "audio_header"
    assert json.loads(audit_path.read_text(encoding="utf-8"))["charts"] == rows
