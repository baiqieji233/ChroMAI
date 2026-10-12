from __future__ import annotations

import json
from pathlib import Path

import pytest

from maichart.event_timing import parse_event_chart_text
from maichart.training.chart_sanitization import (
    DEFAULT_CHART_FILTER_POLICY,
    LEGACY_STRICT_POLICY,
    sanitize_chart,
)
from maichart.training.event_dataset import EventDataset, inspect_chart
from maichart.training.prepare_events import build_event_manifest


def _chart(note_text: str):
    return parse_event_chart_text(
        "&title=Filter\n&wholebpm=120\n&first=0.125\n&lv_4=13\n"
        f"&inote_4={note_text}\n",
        4,
    )


def test_headless_star_atoms_are_removed_whole_and_timing_is_retained():
    original = _chart(
        "(120){4}1/8?-5[8:1]*q3[2:1],"
        "(180){4}2/8?qq4[2:1],E"
    )
    filtered = sanitize_chart(original)

    assert filtered.policy == DEFAULT_CHART_FILTER_POLICY
    assert [item["atom"] for item in filtered.removed] == [
        "8?-5[8:1]*q3[2:1]", "8?qq4[2:1]",
    ]
    assert [point.note_text for point in filtered.chart.points] == ["1", "2"]
    assert [point.beat for point in filtered.chart.points] == [point.beat for point in original.points]
    assert filtered.chart.tempo_changes == original.tempo_changes
    assert filtered.chart.end_beat == original.end_beat
    assert filtered.chart.first == original.first
    assert filtered.removed[0]["same_head_branches"] == 2
    inspect_chart(filtered.chart)


def test_legacy_policy_does_not_remove_headless_star_and_other_bad_syntax_stays_invalid():
    legacy = sanitize_chart(_chart("(120){4}8?-5[8:1],E"), LEGACY_STRICT_POLICY)
    assert legacy.removed == ()
    with pytest.raises(ValueError, match="syntax|Invalid event document"):
        inspect_chart(legacy.chart)

    immediate = sanitize_chart(_chart("(120){4}8!-5[8:1],E"))
    assert immediate.removed == ()
    with pytest.raises(ValueError):
        inspect_chart(immediate.chart)

    unknown = sanitize_chart(_chart("(120){4}1/Z, E"))
    assert unknown.removed == ()
    with pytest.raises(ValueError, match="syntax|Invalid event document"):
        inspect_chart(unknown.chart)

    empty_operand = sanitize_chart(_chart("(120){4}8?-5[8:1]/,E"))
    assert empty_operand.removed == ()
    with pytest.raises(ValueError):
        inspect_chart(empty_operand.chart)


def test_preprocessor_and_dataset_share_policy_and_preserve_source_identity(tmp_path: Path):
    raw = tmp_path / "raw"
    song = raw / "song-a"
    song.mkdir(parents=True)
    inote = "(120){4}" + "8?-5[8:1]*q3[2:1]/1," * 60 + "E"
    maidata = song / "maidata.txt"
    source_text = (
        "&title=Filter\n&wholebpm=120\n&first=0\n&lv_4=13\n"
        f"&inote_4={inote}\n"
    )
    maidata.write_text(source_text, encoding="utf-8")
    (song / "track.mp3").write_bytes(b"\x00" * 32)

    manifest = build_event_manifest(raw, cache_dir=tmp_path / "cache")
    assert manifest["preprocessing"]["chart_filter_policy"] == DEFAULT_CHART_FILTER_POLICY
    difficulty = manifest["songs"][0]["difficulties"][0]
    assert difficulty["event_usable_for_training"]
    assert difficulty["chart_filter"]["removed_count"] == 60
    assert difficulty["chart_filter"]["removed"][0] == {
        "beat": "0", "atom": "8?-5[8:1]*q3[2:1]", "same_head_branches": 2,
    }
    assert manifest["summary"]["chart_filter"] == {
        "policy": DEFAULT_CHART_FILTER_POLICY,
        "affected_chart_count": 1,
        "removed_atom_count": 60,
    }

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    dataset = EventDataset(manifest_path, cache_dir=tmp_path / "cache", eager_audio=False,
                           segment_beats=16, max_tokens=512)
    assert dataset.chart_filter_policy == DEFAULT_CHART_FILTER_POLICY
    assert len(dataset.samples) == 1
    assert dataset.chart_filter_audit[0]["removed_count"] == 60
    assert all("8?" not in point.note_text for point in dataset.samples[0].chart.points)
    assert maidata.read_text(encoding="utf-8") == source_text
