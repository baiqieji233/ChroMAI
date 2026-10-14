import hashlib
import json

import pytest

from maichart.form.resume_contract import (
    compare_resume_contract,
    contract_header,
    preflight_resume,
    resolve_resume_contract,
    validate_resume_cursor,
    verify_saved_chart_sources,
)


def _chart(source="song/maidata.txt", *, chart_hash="chart-a", audio_hash="audio-a",
           quality=None, weight=1.0, first_override=None, links=2, spans=4):
    return {
        "source": source,
        "difficulty": 5,
        "chart_sha256": chart_hash,
        "audio_sha256": audio_hash,
        "quality": quality,
        "weight": weight,
        "first_override": first_override,
        "links": links,
        "spans": spans,
    }


def _contract(charts=None):
    return {
        "manifest_sha256": "manifest-a",
        "meter_map": None,
        "max_region_seconds": 14.0,
        "charts": list(charts if charts is not None else [_chart()]),
    }


def test_exact_contract_is_compatible_and_strict_resume_is_allowed():
    saved = _contract()

    diff = compare_resume_contract(saved, saved.copy())

    assert diff["compatible"] is True
    assert diff["can_restart_epoch"] is True
    assert resolve_resume_contract(saved, saved, policy="strict") == diff


def test_added_chart_requires_explicit_restart_epoch_policy():
    saved = _contract()
    current = _contract([_chart(), _chart("new/maidata.txt", chart_hash="chart-b", audio_hash="audio-b")])

    diff = compare_resume_contract(saved, current)

    assert len(diff["added"]) == 1
    assert diff["removed"] == []
    assert diff["source_changes"] == []
    assert diff["can_restart_epoch"] is True
    with pytest.raises(ValueError, match="重跑当前轮|finetune"):
        resolve_resume_contract(saved, current, policy="strict")
    assert resolve_resume_contract(saved, current, policy="restart_epoch")["added"] == diff["added"]


def test_derived_link_change_can_restart_epoch_but_is_not_strict_resume():
    saved = _contract()
    changed = _chart(links=7)
    current = _contract([changed])

    diff = compare_resume_contract(saved, current)

    assert diff["changed"] == [{"source": "song/maidata.txt", "difficulty": 5, "fields": ["links"]}]
    assert diff["source_changes"] == []
    assert diff["can_restart_epoch"] is True
    with pytest.raises(ValueError, match="重跑当前轮|finetune"):
        resolve_resume_contract(saved, current, policy="strict")
    assert resolve_resume_contract(saved, current, policy="restart_epoch")["can_restart_epoch"] is True


def test_both_missing_audio_hashes_allow_exact_strict_but_not_derived_migration():
    saved = _contract([_chart(audio_hash=None)])
    exact = compare_resume_contract(saved, _contract([_chart(audio_hash=None)]))
    changed = compare_resume_contract(saved, _contract([_chart(audio_hash=None, links=9)]))

    assert exact["compatible"] is True
    assert exact["can_restart_epoch"] is False
    assert resolve_resume_contract(saved, _contract([_chart(audio_hash=None)]), policy="strict")["compatible"] is True
    assert changed["compatible"] is False
    assert changed["unverifiable_identities"]
    assert changed["can_restart_epoch"] is False
    with pytest.raises(ValueError, match="finetune"):
        resolve_resume_contract(saved, _contract([_chart(audio_hash=None, links=9)]), policy="restart_epoch")


def test_both_missing_chart_hashes_allow_exact_strict_but_not_derived_migration():
    saved = _contract([_chart(chart_hash=None)])
    exact = compare_resume_contract(saved, _contract([_chart(chart_hash=None)]))
    changed = compare_resume_contract(saved, _contract([_chart(chart_hash=None, links=9)]))

    assert exact["compatible"] is True
    assert exact["can_restart_epoch"] is False
    assert changed["unverifiable_identities"]
    assert changed["can_restart_epoch"] is False
    with pytest.raises(ValueError, match="finetune"):
        resolve_resume_contract(saved, _contract([_chart(chart_hash=None, links=9)]), policy="restart_epoch")


def test_added_chart_with_missing_identity_hash_cannot_restart_epoch():
    saved = _contract()
    added = _chart("new/maidata.txt", chart_hash=None, audio_hash=None)
    current = _contract([_chart(), added])

    diff = compare_resume_contract(saved, current)

    assert len(diff["added"]) == 1
    assert diff["unverifiable_identities"]
    assert diff["can_restart_epoch"] is False
    with pytest.raises(ValueError, match="finetune"):
        resolve_resume_contract(saved, current, policy="restart_epoch")


@pytest.mark.parametrize("side", ["saved", "current"])
def test_duplicate_chart_identity_cannot_restart_epoch(side):
    chart = _chart()
    duplicate = dict(chart)
    saved = _contract([chart, duplicate]) if side == "saved" else _contract([chart])
    current = _contract([chart]) if side == "saved" else _contract([chart, duplicate])

    diff = compare_resume_contract(saved, current)

    assert diff["duplicate_chart_identities"] is True
    assert diff["can_restart_epoch"] is False
    with pytest.raises(ValueError, match="finetune"):
        resolve_resume_contract(saved, current, policy="restart_epoch")


@pytest.mark.parametrize("change", [
    "removed",
    "source",
    "audio",
    "quality",
    "weight",
    "first",
    "missing_chart_hash",
    "missing_audio_hash",
])
def test_removed_or_source_identity_changes_cannot_restart_epoch(change):
    saved_chart = _chart()
    saved = _contract([saved_chart])
    if change == "removed":
        current = _contract([])
    else:
        current_chart = dict(saved_chart)
        if change == "source":
            current_chart["source"] = "changed/maidata.txt"
        elif change == "audio":
            current_chart["audio_sha256"] = "audio-changed"
        elif change == "quality":
            current_chart["quality"] = {"label": "excellent"}
        elif change == "weight":
            current_chart["weight"] = 2.0
        elif change == "first":
            current_chart["first_override"] = {"effective_first": 0.25}
        elif change == "missing_chart_hash":
            current_chart["chart_sha256"] = None
        elif change == "missing_audio_hash":
            current_chart["audio_sha256"] = None
        current = _contract([current_chart])

    diff = compare_resume_contract(saved, current)

    assert diff["can_restart_epoch"] is False
    with pytest.raises(ValueError, match="finetune"):
        resolve_resume_contract(saved, current, policy="restart_epoch")


@pytest.mark.parametrize("field", ["manifest_sha256", "meter_map", "max_region_seconds"])
def test_preflight_rejects_header_changes_before_preparation(field, tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"songs": []}), encoding="utf-8")
    saved = contract_header(manifest, None, 14.0)
    current = dict(saved)
    if field == "manifest_sha256":
        manifest.write_text(json.dumps({"songs": [{"song_id": "new"}]}), encoding="utf-8")
        current = contract_header(manifest, None, 14.0)
    elif field == "meter_map":
        current["meter_map"] = [{"beat": "0", "numerator": 3, "denominator": 4}]
    else:
        current["max_region_seconds"] = 12.0

    with pytest.raises(ValueError, match="数据合同已变化") as error:
        preflight_resume(saved, current)
    assert field in str(error.value)


def test_preflight_accepts_unchanged_header(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"songs": []}), encoding="utf-8")
    header = contract_header(manifest, None, 14.0)

    assert preflight_resume(header, dict(header)) is None


def test_verify_saved_chart_sources_rejects_changed_and_missing_files(tmp_path):
    source = tmp_path / "maidata.txt"
    source.write_text("original", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    saved = {"charts": [{"source": str(source), "chart_sha256": digest}]}

    assert verify_saved_chart_sources(saved) is None
    source.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="源文件变化或缺失"):
        verify_saved_chart_sources(saved)

    missing = tmp_path / "missing.txt"
    with pytest.raises(ValueError, match="源文件变化或缺失"):
        verify_saved_chart_sources({"charts": [{"source": str(missing), "chart_sha256": digest}]})


def test_missing_saved_identity_hash_is_not_silently_verified(tmp_path):
    source = tmp_path / "maidata.txt"
    source.write_text("original", encoding="utf-8")

    with pytest.raises(ValueError, match="hash|身份|缺失"):
        verify_saved_chart_sources({"charts": [{"source": str(source)}]})


def test_validate_resume_cursor_accepts_valid_phase_positions():
    validate_resume_cursor(
        {"epoch": 4, "phase": "blueprint", "completed": 2,
         "layout_total": 8, "plan_total": 5, "batch_size": 2, "max_steps": 0},
        epoch=3, layout_total=8, plan_total=5, batch_size=2, max_steps=0)
    validate_resume_cursor(
        {"epoch": 4, "phase": "layout_revision", "completed": 7,
         "layout_total": 8, "plan_total": 5, "batch_size": 2, "max_steps": 0},
        epoch=3, layout_total=8, plan_total=5, batch_size=2, max_steps=0)
    validate_resume_cursor(
        {"epoch": 3, "phase": "complete", "completed": 8,
         "layout_total": 8, "plan_total": 5, "batch_size": 2, "max_steps": 0},
        epoch=3, layout_total=8, plan_total=5, batch_size=2, max_steps=0)


def test_validate_resume_cursor_accepts_legacy_complete_cursor_without_optional_totals():
    validate_resume_cursor(
        {"epoch": 3, "phase": "complete", "completed": 8},
        epoch=3, layout_total=8, plan_total=5, batch_size=2, max_steps=0)


@pytest.mark.parametrize(
    "cursor, match",
    [
        ({"epoch": 4, "phase": "unknown", "completed": 1,
          "layout_total": 8, "plan_total": 5, "batch_size": 2, "max_steps": 0}, "阶段无效"),
        ({"epoch": 3, "phase": "blueprint", "completed": 1,
          "layout_total": 8, "plan_total": 5, "batch_size": 2, "max_steps": 0}, "epoch"),
        ({"epoch": 4, "phase": "blueprint", "completed": 6,
          "layout_total": 8, "plan_total": 5, "batch_size": 2, "max_steps": 0}, "completed"),
        ({"epoch": 4, "phase": "layout_revision", "completed": -1,
          "layout_total": 8, "plan_total": 5, "batch_size": 2, "max_steps": 0}, "completed"),
        ({"epoch": 4, "phase": "layout_revision", "completed": 2,
          "layout_total": 8, "plan_total": 5, "batch_size": 4, "max_steps": 0}, "batch_size"),
        ({"epoch": 3, "phase": "complete", "completed": 7,
          "layout_total": 8, "plan_total": 5, "batch_size": 2, "max_steps": 0}, "不完整"),
    ],
)
def test_validate_resume_cursor_rejects_invalid_positions(cursor, match):
    with pytest.raises(ValueError, match=match):
        validate_resume_cursor(cursor, epoch=3, layout_total=8, plan_total=5, batch_size=2, max_steps=0)
