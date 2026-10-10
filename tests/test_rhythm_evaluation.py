import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.run_rhythm_evaluation import build_fixed_panel, evaluate_paths


def test_tighter_alignment_exposes_shifted_fast_subdivisions(tmp_path):
    reference = _event_chart(first=0.0)
    reference["tempo_changes"] = [{"beat": "0", "bpm": 120}]
    reference["points"] = [{"beat": beat, "note_text": str(index + 1)}
                           for index, beat in enumerate(("0", "1/8", "1/4"))]
    generated = {**reference, "first": 0.04}
    _write_chart(tmp_path / "reference/song.chart.json", reference)
    _write_chart(tmp_path / "generated/song.chart.json", generated)
    report = evaluate_paths([tmp_path / "generated"], reference_dir=tmp_path / "reference")
    alignment = report["charts"][0]["reference_pair"]["onset_alignment_by_tolerance"]
    assert alignment["20ms"]["matched_onsets"] == 0
    assert alignment["35ms"]["matched_onsets"] == 2
    assert alignment["50ms"]["matched_onsets"] == 3


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


def _spatial_payload(points, *, tempo=120):
    payload = _event_chart(first=0.0)
    payload["tempo_changes"] = [{"beat": "0", "bpm": tempo}]
    payload["points"] = [
        {"beat": str(beat), "note_text": text}
        for beat, text in points
    ]
    payload["end_beat"] = str(max((beat for beat, _ in points), default=0) + 1)
    return payload


def _spatial_report(tmp_path, points):
    payload = _spatial_payload(points)
    row = evaluate_paths([_write_chart(tmp_path / "spatial.chart.json", payload)])[
        "charts"][0]
    return row["distribution"]["onset_rhythm"]["spatial_repetition"]


def test_spatial_repeat_detects_clockwise_and_counterclockwise_wrap(tmp_path):
    clockwise = _spatial_report(tmp_path / "cw", [(beat, str(key)) for beat, key in enumerate((1, 2, 3, 4))])
    counterclockwise = _spatial_report(tmp_path / "ccw", [(beat, str(key)) for beat, key in enumerate((1, 8, 7, 6))])

    assert clockwise["clockwise_adjacent"]["keys"] == [1, 2, 3, 4]
    assert clockwise["counterclockwise_adjacent"] is None
    assert counterclockwise["counterclockwise_adjacent"]["keys"] == [1, 8, 7, 6]
    assert counterclockwise["clockwise_adjacent"] is None
    for pattern in (clockwise["clockwise_adjacent"], counterclockwise["counterclockwise_adjacent"]):
        assert pattern["onset_count"] == 4
        assert pattern["start_beat"] == "0" and pattern["end_beat"] == "3"
        assert pattern["start_sec"] == pytest.approx(0.0)
        assert pattern["end_sec"] == pytest.approx(1.5)
        assert pattern["gap_range_beats"] == {"min": "1", "max": "1"}
        assert pattern["gap_range_sec"] == {"min": 0.5, "max": 0.5}


def test_spatial_repeat_detects_two_distinct_key_alternation(tmp_path):
    pattern = _spatial_report(
        tmp_path,
        [(beat, str(key)) for beat, key in enumerate((1, 2, 1, 2, 1))],
    )["alternating_two_keys"]

    assert pattern["keys"] == [1, 2]
    assert pattern["onset_count"] == 5
    assert pattern["start_beat"] == "0" and pattern["end_beat"] == "4"
    assert pattern["gap_range_beats"] == {"min": "1", "max": "1"}


@pytest.mark.parametrize("interrupt", ["2h[4:1]", "12", "A", "2-3[4:1]"])
def test_other_note_type_or_chord_interrupts_single_tap_patterns(tmp_path, interrupt):
    spatial = _spatial_report(
        tmp_path,
        [(0, "1"), (1, interrupt), (2, "1"), (3, "2"), (4, "1")],
    )

    assert spatial["clockwise_adjacent"]["onset_count"] == 2
    assert spatial["clockwise_adjacent"]["start_beat"] == "2"
    assert spatial["alternating_two_keys"]["onset_count"] == 3
    assert spatial["alternating_two_keys"]["start_beat"] == "2"


def test_spatial_repeat_breaks_when_gap_exceeds_one_beat_and_includes_break_taps(tmp_path):
    spatial = _spatial_report(
        tmp_path,
        [(0, "1b"), (1, "2"), (2, "3"), (4, "4"), (5, "5")],
    )

    assert spatial["break_tap_onset_count"] == 1
    assert spatial["clockwise_adjacent"]["keys"] == [1, 2, 3]
    assert spatial["clockwise_adjacent"]["onset_count"] == 3
    assert spatial["clockwise_adjacent"]["start_beat"] == "0"
    assert spatial["clockwise_adjacent"]["end_beat"] == "2"


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
    assert pair["spatial_repetition"]["generated"] == pair["spatial_repetition"]["reference"]
    assert pair["spatial_repetition"]["generated"]["single_tap_onset_count"] > 0
    assert pair["onset_alignment"]["matched_onsets"] == 5
    assert pair["onset_alignment_by_tolerance"]["50ms"] == pair["onset_alignment"]
    assert "reference intent may be incomplete" in pair["onset_alignment"]["interpretation"]

    empty_reference = tmp_path / "empty-reference"
    empty_reference.mkdir()
    unmatched = evaluate_paths([generated_dir], reference_dir=empty_reference)
    assert unmatched["reference_pairing"]["matched"] == 0
    assert unmatched["reference_pairing"]["missing"] == 1
    assert unmatched["charts"][0]["reference_pair"] == {
        "status": "missing", "key": "song.chart.json",
    }


def test_reference_pairing_accepts_relative_directories(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    payload = _event_chart(first=0.0)
    payload["audio_sha256"] = "a" * 64
    _write_chart(Path("generated") / "song.chart.json", payload)
    _write_chart(Path("references") / "song.chart.json", payload)

    report = evaluate_paths([Path("generated")], reference_dir=Path("references"))

    assert report["error_count"] == 0
    assert report["reference_pairing"]["verified_matched"] == 1
    assert report["reference_pairing"]["reference_dir"] == str(tmp_path / "references")


def test_explicit_chart_paths_keep_arm_relative_reference_pairing(tmp_path):
    generated_root = tmp_path / "generated"
    reference_root = tmp_path / "references"
    outputs = []
    for arm, audio_hash in (("v3_full", "a" * 64), ("v3_finetune", "b" * 64)):
        payload = _event_chart(first=0.0)
        payload["audio_sha256"] = audio_hash
        outputs.append(_write_chart(generated_root / arm / "song_00.chart.json", payload))
        _write_chart(reference_root / arm / "song_00.chart.json", payload)

    report = evaluate_paths(outputs, reference_dir=reference_root, pairing_root=generated_root)

    assert report["reference_pairing"]["matched"] == 2
    assert report["reference_pairing"]["verified_matched"] == 2
    assert [chart["reference_pair"]["reference_path"] for chart in report["charts"]] == [
        str((reference_root / "v3_finetune" / "song_00.chart.json").resolve()),
        str((reference_root / "v3_full" / "song_00.chart.json").resolve()),
    ]


def test_candidate_chart_files_are_discovered_and_paired(tmp_path):
    generated_root = tmp_path / "generated"
    reference_root = tmp_path / "references"
    _write_chart(generated_root / "v3_full" / "song.chart_candidate_001.json")
    _write_chart(reference_root / "v3_full" / "song.chart_candidate_001.json")

    report = evaluate_paths([generated_root], reference_dir=reference_root)

    assert report["input_count"] == 1
    assert report["reference_pairing"]["matched"] == 1
    assert report["charts"][0]["reference_pair"]["reference_path"] == str(
        (reference_root / "v3_full" / "song.chart_candidate_001.json").resolve()
    )


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
    assert all(len(item["audio_sha256"]) == 64 for item in panel["songs"])
    assert all(item["tempo_map"] and "audio_first_sec" in item for item in panel["songs"])


def test_fixed_panel_deduplicates_audio_content_across_song_ids(tmp_path):
    root = tmp_path / "dataset"
    songs = []
    for song_id, audio_bytes, pattern in (
        ("primary", b"same audio", "1,2,3,4,E"),
        ("alias", b"same audio", "1,,,2,,,3,,,4,E"),
        ("distinct", b"other audio", "1,2,,,3,4,E"),
    ):
        folder = root / song_id
        folder.mkdir(parents=True)
        (folder / "maidata.txt").write_text(
            f"&title={song_id}\n&wholebpm=120\n&lv_5=13\n&inote_5=(120){{4}}{pattern}\n",
            encoding="utf-8",
        )
        (folder / "track.mp3").write_bytes(audio_bytes)
        songs.append({
            "song_id": song_id,
            "maidata_path": f"{song_id}/maidata.txt",
            "audio_path": f"{song_id}/track.mp3",
            "difficulties": [{"difficulty_index": 5, "level_raw": "13",
                              "event_usable_for_training": True}],
        })
    manifest = root / "validation_manifest.json"
    manifest.write_text(json.dumps({"songs": songs, "source_root": str(root)}), encoding="utf-8")

    panel = build_fixed_panel(manifest)

    assert panel["candidate_chart_count"] == 2
    assert panel["selected_song_count"] == 2
    assert panel["skipped_counts"]["duplicate_audio_content"] == 1
    assert len({item["audio_sha256"] for item in panel["songs"]}) == 2
    assert len({item["song_id"] for item in panel["songs"]}) == 2
