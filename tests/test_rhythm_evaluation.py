import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.run_rhythm_evaluation import build_fixed_panel, evaluate_paths


def _event_chart(*, first=-0.25, duration=4.5):
    event = {
        "points": [
            {"beat": "0", "note_text": "1"},
            {"beat": "1", "note_text": "1"},
            {"beat": "2", "note_text": "2"},
            {"beat": "4", "note_text": "1/2"},
            {"beat": "8", "note_text": "3"},
            {"beat": "10", "note_text": ""},
        ],
        "tempo_changes": [{"beat": "0", "bpm": 120}, {"beat": "4", "bpm": 60}],
        "first": first,
        "time_offset": 0.75,
        "level": "13",
        "title": "Tempo Sample",
        "end_beat": "10",
        "difficulty": 5,
    }
    return {
        "schema": "maichart-event-chart-v1",
        **event,
        "quality": {"duration_sec": duration},
    }


def _write_chart(path: Path, payload=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload or _event_chart()), encoding="utf-8")
    return path


def test_evaluation_uses_tempo_map_offset_and_reports_boundaries(tmp_path):
    chart_path = _write_chart(tmp_path / "outputs" / "song.chart.json")
    report = evaluate_paths([chart_path])
    chart = report["charts"][0]

    assert report["error_count"] == 0
    assert chart["timing"]["audio_first_sec"] == -0.25
    assert chart["timing"]["time_offset_sec"] == 0.75
    assert chart["timing"]["tempo_map"] == [
        {"beat": "0", "bpm": 120.0},
        {"beat": "4", "bpm": 60.0},
    ]
    assert chart["boundaries"]["notes_before_audio_start"] == 1
    assert chart["timing"]["audio_duration_sec"] is None
    assert chart["timing"]["duration_source"] is None
    assert chart["boundaries"]["notes_at_or_after_audio_end"] is None
    assert chart["per_second"]["windows"][0]["second"] == -1
    assert chart["per_second"]["windows"][0]["note_count"] == 1
    assert chart["distribution"]["note_count"] == 6
    assert chart["structure"]["repeated_event_signature_run"] == 2
    assert "aggregate quality score" in report["metric_policy"]


def test_evaluation_reports_fine_gaps_as_distinct_onset_samples(tmp_path):
    beats = ["0"]
    current = 0
    for index in range(128):
        current += 1 if index < 33 else 4
        beats.append(str(current / 4))
    payload = _event_chart(first=0.0)
    payload["points"] = [{"beat": beat, "note_text": "1"} for beat in beats]
    chart = evaluate_paths([_write_chart(tmp_path / "fine.chart.json", payload)])
    rhythm = chart["charts"][0]["distribution"]["onset_rhythm"]

    assert rhythm["fine_note_ratio"] == 33 / 128
    assert rhythm["gap_sample_count"] == 128
    assert rhythm["gap_histogram_beats"] == {"1/4": 33, "1": 95}
    assert rhythm["named_subdivision_gap_counts"] == {"12th": 0, "16th": 33, "24th": 0}
    assert rhythm["max_repeated_onset_type_pattern"] == {
        "cycles": 129, "onset_window_length": 1, "pattern": [["tap"]],
    }


def test_repeated_onset_type_pattern_tracks_multi_onset_cycles(tmp_path):
    payload = _event_chart(first=0.0)
    payload["points"] = [
        {"beat": str(beat), "note_text": note_text}
        for beat, note_text in enumerate(("1", "2h[4:1]", "1", "2h[4:1]"))
    ]
    rhythm = evaluate_paths([_write_chart(tmp_path / "types.chart.json", payload)])["charts"][0]["distribution"]["onset_rhythm"]

    assert rhythm["max_repeated_onset_type_pattern"] == {
        "cycles": 2, "onset_window_length": 2, "pattern": [["tap"], ["hold"]],
    }


def test_chartir_with_maidata_source_uses_real_chart_tempo_and_audio_first(tmp_path):
    source = (
        "&title=IR Source\n&wholebpm=120\n&first=0.25\n&lv_5=13\n"
        "&inote_5=(120){4}1,2,3,4,E,"
    )
    payload = {
        "metadata": {"title": "IR Source", "offset": 0.25, "raw": source},
        "difficulty": {"index": 5, "level": "13"},
        "timing": {"bpms": [{"beat": 0, "bpm": 120}]},
        "notes": [],
    }
    report = evaluate_paths([_write_chart(tmp_path / "cache" / "difficulty_5.chart_ir.json", payload)])

    assert report["error_count"] == 0
    chart = report["charts"][0]
    assert chart["identity"]["title"] == "IR Source"
    assert chart["timing"]["audio_first_sec"] == 0.25
    assert chart["distribution"]["note_count"] == 4
    assert chart["syntax"]["valid"] is True


def test_missing_end_beat_is_reported_unless_quality_song_beats_exists(tmp_path):
    payload = _event_chart()
    payload.pop("end_beat")
    missing = evaluate_paths([_write_chart(tmp_path / "missing.chart.json", payload)])
    row = missing["charts"][0]
    assert row["timing"]["end_beat"] is None
    assert row["timing"]["end_beat_source"] == "missing"
    assert row["boundaries"]["notes_at_or_after_declared_end_beat"] is None

    payload["quality"]["song_beats"] = 12
    recovered = evaluate_paths([_write_chart(tmp_path / "recovered.chart.json", payload)])
    row = recovered["charts"][0]
    assert row["timing"]["end_beat"] == "12"
    assert row["timing"]["end_beat_source"] == "quality.song_beats"
    assert row["boundaries"]["notes_at_or_after_declared_end_beat"] == 0


def test_reference_metrics_are_only_emitted_for_exact_filename_pairs(tmp_path):
    generated_dir = tmp_path / "generation"
    reference_dir = tmp_path / "reference"
    generated = _write_chart(generated_dir / "song.chart.json")
    _write_chart(reference_dir / "song.chart.json")
    report = evaluate_paths([generated_dir], reference_dir=reference_dir)

    pair = report["charts"][0]["reference_pair"]
    assert report["reference_pairing"]["matched"] == 1
    assert pair["status"] == "matched_unverified"
    assert pair["identity_status"] == "filename_only_unverified"
    assert pair["per_second_density_mae"] == 0.0
    assert pair["paired_metric_deltas"]["blank_fraction"] == 0.0
    assert pair["onset_alignment"]["matched_onsets"] == 5
    assert "reference intent may be incomplete" in pair["onset_alignment"]["interpretation"]

    empty_reference = tmp_path / "empty-reference"
    empty_reference.mkdir()
    unmatched = evaluate_paths([generated_dir], reference_dir=empty_reference)
    assert unmatched["reference_pairing"]["matched"] == 0
    assert unmatched["reference_pairing"]["missing"] == 1
    assert unmatched["charts"][0]["reference_pair"] == {
        "status": "missing", "key": "song.chart.json",
    }


def test_reference_hash_mismatch_is_incompatible_and_shadow_files_are_skipped(tmp_path):
    generated_dir = tmp_path / "generation"
    reference_dir = tmp_path / "reference"
    generated_payload = _event_chart(first=0.0)
    generated_payload["audio_sha256"] = "a" * 64
    reference_payload = _event_chart(first=0.0)
    reference_payload["audio_sha256"] = "b" * 64
    _write_chart(generated_dir / "song.chart.json", generated_payload)
    _write_chart(reference_dir / "song.chart.json", reference_payload)
    _write_chart(generated_dir / ".history" / "old" / "song.chart.json")
    _write_chart(generated_dir / "song.original.chart.json")

    report = evaluate_paths([generated_dir], reference_dir=reference_dir)

    assert report["input_count"] == 1
    assert report["reference_pairing"]["incompatible"] == 1
    assert report["reference_pairing"]["matched"] == 0
    assert report["charts"][0]["reference_pair"] == {
        "status": "incompatible",
        "identity_status": "audio_sha256_mismatch",
        "reference_path": str((reference_dir / "song.chart.json").resolve()),
    }


def test_stored_playability_evidence_is_exported_as_unverified(tmp_path):
    payload = _event_chart(first=0.0)
    payload["quality"]["playability"] = {
        "status": "partial", "notes": 12, "conflicts": 3, "hard": 1,
        "remaining_hard": 1, "repairs": [],
        "coverage": {"status": "partial", "unsupported": ["slide"]},
    }
    row = evaluate_paths([_write_chart(tmp_path / "playability.chart.json", payload)])["charts"][0]

    assert row["playability_evidence"] == {
        "status": "partial", "verified": False, "scope": None,
        "coverage": {"status": "partial", "unsupported": ["slide"]},
        "counts": {"notes": 12, "conflicts": 3, "hard": 1, "remaining_hard": 1},
        "repair_count": 0,
    }


def test_fixed_panel_uses_measured_traits_and_keeps_distinct_source_audio(tmp_path):
    root = tmp_path / "dataset"
    songs = []
    for song_id, pattern in (("sparse", "1,,,2,,,3,,,4,E"),
                             ("dense", "1,2,3,4,E")):
        folder = root / song_id
        folder.mkdir(parents=True)
        (folder / "maidata.txt").write_text(
            f"&title={song_id}\n&wholebpm=120\n&lv_5=13\n&inote_5=(120){{4}}{pattern}\n",
            encoding="utf-8",
        )
        (folder / "track.mp3").write_bytes(song_id.encode())
        songs.append({
            "song_id": song_id,
            "title": song_id,
            "maidata_path": f"{song_id}/maidata.txt",
            "audio_path": f"{song_id}/track.mp3",
            "difficulties": [{"difficulty_index": 5, "level_raw": "13",
                              "event_usable_for_training": True}],
        })
    manifest = root / "training_manifest.json"
    manifest.write_text(json.dumps({"songs": songs, "source_root": str(root)}), encoding="utf-8")

    panel = build_fixed_panel(manifest, size=2)

    assert panel["selected_song_count"] == 2
    assert {item["song_id"] for item in panel["songs"]} == {"sparse", "dense"}
    assert all(Path(item["audio_path"]).is_absolute() for item in panel["songs"])
    assert all("selection_basis" in item and "note_density_per_sec" in item for item in panel["songs"])
