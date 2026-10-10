import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.render_rhythm_review import render_files, render_reports


def _chart(path, *, fine=None, notes=None, f1=None, empty=None):
    reference = {"status": "matched", "identity_status": "audio_sha256_match"}
    if f1 is not None:
        reference["onset_alignment"] = {"precision": f1, "recall": f1, "f1": f1}
    result = {
        "path": path,
        "identity": {"title": "测试|曲目", "level": "13+", "difficulty_index": 5},
        "distribution": {
            "note_count": notes,
            "type_ratios": {"tap": 0.8, "hold": 0.1, "slide": 0.1, "touch": 0.0, "touch_hold": 0.0},
            "onset_rhythm": {
                "distinct_onset_count": 10,
                "fine_note_ratio": fine,
                "gap_sample_count": 9,
                "gap_histogram_beats": {"1/2": 3, "1/3": 1, "1/4": 2, "1/6": 1},
                "spatial_repetition": {
                    "clockwise_adjacent": {"start_sec": 1.25, "end_sec": 2.75,
                                            "onset_count": 5, "keys": [1, 2, 3, 4, 5]},
                    "counterclockwise_adjacent": None,
                    "alternating_two_keys": {"start_sec": 4.0, "end_sec": 5.0,
                                              "onset_count": 4, "keys": [2, 7]},
                },
            },
        },
        "per_second": {"empty_seconds_within_audio": empty},
        "structure": {"longest_blank_run_sec": empty},
        "boundaries": {"notes_before_audio_start": 0, "notes_at_or_after_audio_end": None,
                       "notes_at_or_after_declared_end_beat": 0},
        "playability_evidence": {"status": "partial", "verified": False,
                                 "counts": {"hard": 0, "remaining_hard": 0}},
        "reference_pair": reference,
    }
    return result


def test_arm_summary_is_equal_chart_weighted_and_does_not_zero_fill_missing():
    report_path = Path("evaluation.json")
    report = {"schema": "rhythm-evaluation-v1", "input_count": 2, "evaluated_count": 2,
              "error_count": 0, "charts": [
                  _chart(r"E:\输出\generated\arm A\第一首.chart.json", fine=0.2, notes=100, f1=0.5, empty=2),
                  _chart(r"E:\输出\generated\arm A\第二首.chart.json", fine=0.8, notes=300, empty=None),
              ]}
    markdown = render_reports([(report_path, report)])
    assert "arm A" in markdown
    assert "Artifact completeness: not reported" in markdown
    assert "| 曲数 | 平均note数 | 平均起点数 | 细碎ratio | 最长连续空白秒 |" in markdown
    assert "50.000% (2/2)" in markdown
    assert "200 (2/2)" in markdown
    assert "2 (1/2)" in markdown
    assert "测试\\|曲目" in markdown
    assert "顺圈" in markdown and "1.250-2.750s" in markdown
    assert "逆圈" in markdown and "两键交互" in markdown
    assert "audio_sha256_match" in markdown


def test_missing_reference_metrics_and_static_evidence_are_not_presented_as_zero():
    chart = _chart(r"root\generated\partial\song.json", fine=None, notes=12, empty=None)
    chart.pop("playability_evidence")
    report = {"schema": "rhythm-evaluation-v1", "charts": [
        chart,
    ], "errors": [{"path": "failed song", "error": "no event chart"}], "error_count": 1}
    markdown = render_reports([(Path("report.json"), report)])
    assert "reference=matched" in markdown
    assert "precision | recall | F1" in markdown
    assert "未测量" in markdown
    assert "未完成记录" in markdown
    assert "no event chart" in markdown
    assert "unavailable" in markdown


def test_render_files_refuses_to_overwrite_any_input(tmp_path):
    source = tmp_path / "评估 报告.json"
    source.write_text(json.dumps({"charts": []}, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="must not overwrite"):
        render_files([source], source)


def test_json_input_with_utf8_bom_and_unicode_paths_renders(tmp_path):
    source = tmp_path / "评价.json"
    payload = {"charts": [_chart(str(tmp_path / "有 空格" / "generated" / "中性" / "歌.chart.json"),
                                 fine=0.25, notes=42, empty=1)]}
    source.write_text("\ufeff" + json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    output = tmp_path / "审阅 报告.md"
    markdown = render_files([source], output)
    assert output.read_text(encoding="utf-8") == markdown
    assert "中性" in markdown and "有 空格" in markdown and "歌.chart.json" in markdown


def test_nested_generated_run_and_arm_paths_use_immediate_chart_parent():
    report = {"charts": [_chart(
        r"E:\workspace\generated\v3_heldout_validation\v3_detailed\song_00.chart.json",
        fine=0.3, notes=8, empty=0)]}
    markdown = render_reports([(Path("nested.json"), report)])
    assert "### v3_detailed / 测试\\|曲目" in markdown


def test_missing_spatial_run_is_explicitly_not_a_qualifying_segment():
    report = {"charts": [_chart(r"root\generated\arm\song.json", fine=0.1, notes=2, empty=0)]}
    markdown = render_reports([(Path("report.json"), report)])
    assert "无符合条件片段" in markdown


def test_boundary_zero_without_audio_check_is_not_rendered_as_measured_zero():
    chart = _chart(r"root\generated\arm\song.json", fine=0.1, notes=2, empty=0)
    chart["boundaries"]["audio_end_check_available"] = False
    chart["boundaries"]["notes_at_or_after_audio_end"] = 0
    markdown = render_reports([(Path("report.json"), {"charts": [chart]})])
    assert "0 (1/1) / 未测量 (0/1) / 0 (1/1)" in markdown


def test_disabled_static_check_does_not_render_zero_hard_counts():
    chart = _chart(r"root\generated\arm\song.json", fine=0.1, notes=2, empty=0)
    chart["playability_evidence"] = {"status": "disabled", "verified": False,
                                     "counts": {"hard": 0, "conflicts": 0, "remaining_hard": 0}}
    markdown = render_reports([(Path("report.json"), {"charts": [chart]})])
    assert "disabled" in markdown
    assert "| disabled | 未测量 | 未测量 | 未测量 | False |" in markdown


def test_filename_only_reference_f1_is_labeled_unverified_in_review():
    chart = _chart(r"root\generated\arm\song.json", fine=0.1, notes=2, empty=0, f1=0.4)
    chart["reference_pair"]["identity_status"] = "filename_only_unverified"
    markdown = render_reports([(Path("report.json"), {"charts": [chart]})])
    assert "filename_only_unverified" in markdown
    assert "SHA-256 verified | 0/1" in markdown
    assert "Unverified/filename-only | 1/1" in markdown


def test_recorded_artifact_completeness_overrides_equal_evaluation_counts():
    report = {
        "input_count": 1, "evaluated_count": 1, "error_count": 0,
        "artifact_completeness": {
            "expected_count": 6, "produced_count": 3, "missing_count": 3,
            "unverifiable_count": 0,
        },
        "charts": [_chart(r"root\generated\arm\song.json", fine=0.1, notes=2, empty=0)],
    }
    markdown = render_reports([(Path("report.json"), report)])
    assert "Artifact completeness: partial: expected=6, produced=3, missing=3, unverifiable=0" in markdown
