"""Exercise the Python judge through the retained legacy import/API."""
from fractions import Fraction as F
import json
from pathlib import Path
import subprocess

import pytest

from maichart.event_timing import EventPoint as P, EventTempo as T
from maichart.generation import miacode_native as native
from maichart.generation.post_process import PostProcessConfig, detect_chart_conflicts, process_chart_points

TEMPO = [T(F(0), 150)]


def config(mode="none", **kwargs):
    return PostProcessConfig(True, detect_soft_conflicts=True, conflict_repair_mode=mode,
                             conflict_engine="miacode", **kwargs)


def test_real_core_reports_hold_tail_without_starting_any_process(monkeypatch):
    def no_process(*args, **kwargs):
        raise AssertionError("The embedded judge must not spawn any process")
    monkeypatch.setattr(subprocess, "Popen", no_process)
    report = native.analyze_inote("(120){4}1h[1:1],,,,1,E")
    assert report["ok"] and report["engine"] == "miacode_python"
    assert report["dynamic_dependency"] is False
    assert any(d["kind"] == "Overlap" and d["alert_level"] == "Muri" for d in report["diagnostics"])


def test_hold_tail_local_removes_only_the_affected_chord_child(capsys):
    points = [P(0, "1h[1:1]"), P(4, "1/2b/C")]
    result = process_chart_points(points, TEMPO, 0, config("local"))
    assert result.points[-1].note_text == "2b/C"
    assert points[-1].note_text == "1/2b/C"
    assert len(result.repairs_applied) == 1
    assert not [c for c in result.remaining_conflicts if c.severity == "hard"]
    assert 'playability_repair ' in capsys.readouterr().out
    assert result.to_dict()["remaining_details"] == []


def test_compact_chord_source_mapping_preserves_other_button():
    result = process_chart_points([P(0, "1h[1:1]"), P(4, "12b")], TEMPO, 0, config("local"))
    assert result.points[-1].note_text == "2b"
    assert result.repairs_applied[0]["note_index"] == 0


def test_real_slide_head_and_early_completion_are_muri_not_geometric_soft():
    points = [P(0, "3>1[4:1]"), P(1, "5<1[4:1]"), P(2, "1>5[4:1]"), P(3, "5<1[4:1]"), P(4, "1")]
    result = detect_chart_conflicts(points, TEMPO, 0, config())
    assert any(c.conflict_type == "miacode_TapOnSlide" and c.severity == "hard" for c in result.conflicts_detected)
    repaired = process_chart_points(points, TEMPO, 0, config("local"))
    assert repaired.repairs_applied
    assert sum(c.severity == "hard" for c in repaired.remaining_conflicts) < sum(
        c.severity == "hard" for c in result.conflicts_detected)
    assert repaired.coverage["in_process"]


def test_zero_threshold_removes_static_proximity_without_hiding_runtime_muri():
    chart = "(120){4}1-5[4:1],,{20},5,E"
    regular = native.analyze_inote(chart, tap_on_slide_threshold_ms=200)
    zero = native.analyze_inote(chart, tap_on_slide_threshold_ms=0)
    assert any(d["kind"] == "TapOnSlide" for d in regular["static_references"])
    assert not any(d["kind"] in {"TapOnSlide", "SlideHeadTap"} for d in zero["static_references"])
    assert zero["tap_on_slide_threshold_ms"] == 0
    actual = native.analyze_inote("(150){4}3>1[4:1],5<1[4:1],1>5[4:1],5<1[4:1],1,E",
                                  tap_on_slide_threshold_ms=0)
    assert any(d["kind"] == "TapOnSlide" and d["alert_level"] == "Muri" for d in actual["diagnostics"])


def test_sub_150_threshold_is_not_clamped_by_native_core():
    chart = "(120){4}1-5[4:1],,{20},5,E"
    smaller = native.analyze_inote(chart, tap_on_slide_threshold_ms=100)
    regular = native.analyze_inote(chart, tap_on_slide_threshold_ms=200)
    assert smaller["tap_on_slide_threshold_ms"] == 100
    assert len(smaller["static_references"]) < len(regular["static_references"])


@pytest.mark.parametrize("text", ["1/5/C", "B1/B2/B3/B4/B5/B6/B7/B8/C", "1-5[4:1]*-4[4:1]"])
def test_touch_groups_and_branch_heads_do_not_become_static_count_conflicts(text):
    result = detect_chart_conflicts([P(0, text)], TEMPO, 0, config())
    assert not result.coverage["parse_errors"]
    assert not any(c.conflict_type in {"miacode_Overlap", "same_pad_overlap", "hand_assignment_review"}
                   for c in result.conflicts_detected)


@pytest.mark.parametrize("text", ["1w5[4:1]", "1<8-4[4:1]", "8z4[2:1]"])
def test_native_covers_paths_not_simulated_by_static_checker(text):
    result = detect_chart_conflicts([P(0, text)], TEMPO, 0, config())
    assert result.coverage["status"] == "engine_reference"
    assert result.coverage["unsupported"] == []


def test_cross_window_retry_logs_trigger_and_preserves_context(capsys):
    from maichart.generation.event_cli import _DecodedWindow, _select_playable_window
    from maichart.generation.event_model import EventSequence

    prior = [_DecodedWindow(EventSequence([F(15)], ["1h[4:1]"]), F(0), F(16), True)]
    def decode(attempt):
        return EventSequence([F(0)], ["1/2" if attempt == 0 else "2"]), True, None, 0
    result = _select_playable_window(decode, prior, start=F(16), end=F(32),
        tempo_changes=TEMPO, first=0, config=config("local", conflict_retries=1), seed=42)
    assert result[0].note_texts == ["2"]
    assert len(result[4]["attempts"]) == 2
    assert prior[0].sequence.note_texts == ["1h[4:1]"]
    logs = capsys.readouterr().out
    for field in ("playability_check", "playability_issue", "playability_retry", "playability_window_done", "miacode_Overlap"):
        assert field in logs


def test_offsets_and_tempo_changes_preserve_native_source_mapping():
    tempo = [T(0, 120), T(4, 150)]
    points = [P(4, "1h[4:1]"), P(5, "1/3")]
    result = process_chart_points(points, tempo, -0.25, config("local"))
    assert result.points[-1].note_text == "3"
    assert result.repairs_applied[0]["time_sec"] == pytest.approx(2.15)


def test_native_reject_detect_only_and_hard_switch():
    points = [P(0, "1h[1:1]"), P(4, "1")]
    assert detect_chart_conflicts(points, TEMPO, 0, config("reject")).success is False
    assert detect_chart_conflicts(points, TEMPO, 0, config()).points == points
    assert not detect_chart_conflicts(points, TEMPO, 0, config(detect_hard_conflicts=False)).conflicts_detected


def test_duplicate_press_children_are_detected_without_silent_parser_loss():
    result = process_chart_points([P(0, "1/1/2")], TEMPO, 0, config("local"))
    assert result.repairs_applied
    assert result.points[0].note_text == "1/2"
    assert not result.coverage["parse_errors"]


@pytest.mark.parametrize("mode", ["none", "local", "reject"])
def test_cli_native_default_and_legacy_reference_migration(tmp_path, monkeypatch, capsys, mode):
    import numpy as np
    from maichart.generation import event_cli
    from maichart.generation.event_audio import EventAudio, ACOUSTIC_DIM
    from maichart.generation.event_model import EventSequence, EventVocabulary

    vocabulary = EventVocabulary.from_event_charts([], segment_beats=16)
    checkpoint = {"schema": "maichart-event-generator-checkpoint-v1", "epoch": 20,
                  "data_contract": {"target_schema": "event_v1", "segment_beats": 16}}
    monkeypatch.setattr(event_cli, "load_event_model", lambda *a: (object(), vocabulary, checkpoint))
    audio = EventAudio(np.zeros((151, ACOUSTIC_DIM), dtype=np.float32),
                       np.linspace(0, 3, 151, dtype=np.float32), 3.0, 120.0)
    monkeypatch.setattr(event_cli, "extract_event_audio", lambda *a, **kw: audio)
    monkeypatch.setattr(event_cli, "_decode", lambda *a, **kw:
                        EventSequence([F(0), F(4)], ["1h[1:1]", "1/2"]))
    output = tmp_path / "chart.txt"
    args = ["--checkpoint", "test.pt", "--audio", "test.wav", "--output", str(output),
            "--device", "cpu", "--bpm", "120", "--no-audio-bounds-trim", "--no-intensity-follow",
            "--enable-conflict-detection", "--conflict-repair-mode", mode]
    if mode == "reject":
        args += ["--runtime-reference-mode", "reject", "--runtime-reference-executable", "missing/MiaCode.exe"]
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: pytest.fail("spawned external process"))
    code = event_cli.main(args)
    report = json.loads(output.with_suffix(".quality.json").read_text(encoding="utf-8"))
    log = capsys.readouterr().out
    assert "playability_engine engine=miacode" in log
    assert "playability_final engine=miacode" in log
    if mode == "reject":
        assert code != 0 and not output.exists()
        assert report["generation_status"] == "rejected_by_playability"
    else:
        assert code == 0 and output.exists()
        if mode == "local":
            assert report["playability"]["repairs"]
            assert report["playability"]["remaining_hard"] == 0
            assert "playability_repair " in log
            assert output.with_suffix(".original.txt").is_file()
        else:
            assert report["playability"]["remaining_hard"] > 0
            assert not report["playability"]["repairs"]
