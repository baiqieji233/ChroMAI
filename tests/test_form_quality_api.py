from copy import deepcopy

import pytest

from maichart.webui import form_api
from maichart.webui import server as webui
from test_webui import webui_api


def old_checkpoint(profile="legacy"):
    return {
        "family": "song_form_v2", "args": {"position_alignment_loss_weight": .2,
            "position_alignment_tolerance_ms": 25},
        "model_config": {"quality_profile": profile, "curve_bin_seconds": 1.25 if profile != "legacy" else 2.},
        "blueprint_config": {"context_seconds": 1.5 if profile != "legacy" else 0.},
    }


def resolve(payload=None, *, mode="fresh", family="song_form_v2", old=None):
    return form_api.resolve_quality_training_options(payload or {}, family=family, mode=mode, inspected=old)


def test_quality_api_optional_nulls_use_versioned_fresh_defaults():
    options = resolve({key: None for key in ("position_alignment_loss_weight", "position_alignment_tolerance_ms",
                                           "planner_context_seconds", "curve_bin_seconds")})
    assert options == {"quality_profile": "musical_evidence_v1", "position_alignment_loss_weight": .15,
                       "position_alignment_tolerance_ms": 15., "planner_context_seconds": 1., "curve_bin_seconds": 1.}
    legacy = resolve({"quality_profile": "legacy"})
    assert legacy["position_alignment_loss_weight"] == 0.
    assert legacy["planner_context_seconds"] == 0.
    assert legacy["curve_bin_seconds"] == 2.
    assert resolve(family="song_form_v1")["quality_profile"] == "legacy"


def test_resume_ignores_stale_ui_values_and_uses_saved_contract():
    old = old_checkpoint("musical_evidence_v1")
    before = deepcopy(old)
    options = resolve({"quality_profile": "legacy", "position_alignment_loss_weight": 9.,
                       "position_alignment_tolerance_ms": "stale", "planner_context_seconds": 20.,
                       "curve_bin_seconds": 5.}, mode="resume", old=old)
    assert options == {"quality_profile": "musical_evidence_v1", "position_alignment_loss_weight": .2,
                       "position_alignment_tolerance_ms": 25., "planner_context_seconds": 1.5, "curve_bin_seconds": 1.25}
    assert old == before
    settings = form_api.form_resume_settings(old)
    assert {key: settings[key] for key in options} == options


def test_finetune_requires_explicit_upgrade_and_prevents_downgrade():
    old = old_checkpoint()
    old["args"] = {}
    inherited = resolve(mode="finetune", old=old)
    assert inherited["quality_profile"] == "legacy"
    assert inherited["position_alignment_loss_weight"] == 0.
    upgraded = resolve({"quality_profile": "musical_evidence_v1"}, mode="finetune", old=old)
    assert upgraded["quality_profile"] == "musical_evidence_v1"
    assert upgraded["position_alignment_loss_weight"] == .15
    assert upgraded["planner_context_seconds"] == 1.
    assert upgraded["curve_bin_seconds"] == 1.
    with pytest.raises(ValueError):
        resolve({"quality_profile": "legacy"}, mode="finetune", old=old_checkpoint("musical_evidence_v1"))


@pytest.mark.parametrize(("key", "bad"), [
    ("quality_profile", "bogus"), ("position_alignment_loss_weight", True),
    ("position_alignment_loss_weight", float("nan")), ("planner_context_seconds", -1),
    ("curve_bin_seconds", 0), ("position_alignment_tolerance_ms", .5),
    ("position_alignment_tolerance_ms", 101),
])
def test_quality_api_rejects_invalid_values(key, bad):
    with pytest.raises(ValueError, match=key):
        resolve({key: bad})


def test_fresh_evidence_fields_reach_cli_and_effective_settings(webui_api, monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.json"; manifest.write_text("{}")
    calls = []
    monkeypatch.setattr(webui, "_run_subprocess_task", lambda task, args, **kwargs: calls.append((task, args)))
    status, result = webui_api("/api/form/train", {
        "manifest": str(manifest), "checkpoint_dir": str(tmp_path / "model"),
        "mode": "fresh", "model_family": "song_form_v2", "device": "cpu",
        "d_model": 32, "nhead": 4, "layers": 1,
        "quality_profile": "musical_evidence_v1", "position_alignment_loss_weight": .3,
        "layout_prepare_threads": 3,
        "position_alignment_tolerance_ms": 20., "planner_context_seconds": 1.5, "curve_bin_seconds": 1.25,
    })
    assert status == 200, result
    task, command = calls[0]
    from maichart.form.train import parser
    args = parser().parse_args(command[2:])
    for key, expected in {"quality_profile": "musical_evidence_v1", "position_alignment_loss_weight": .3,
                          "position_alignment_tolerance_ms": 20., "planner_context_seconds": 1.5,
                          "curve_bin_seconds": 1.25}.items():
        assert getattr(args, key) == expected
        assert task.result["effective_settings"][key] == expected
    assert args.layout_prepare_threads == 3
    assert task.result['effective_settings']['layout_prepare_threads'] == 3


def test_generation_preflight_forwards_meter_and_checkpoint_quality_to_curve_validator(webui_api, monkeypatch, tmp_path):
    from maichart.form import checkpoint, curve_validation
    audio = tmp_path / "song.wav"; audio.write_bytes(b"audio")
    model = tmp_path / "model.pt"; model.write_bytes(b"weights")
    inspected = {"family": "song_form_v2", "model_config": {"quality_profile": "musical_evidence_v1"}}
    monkeypatch.setattr(checkpoint, "load", lambda *_a, **_kw: (None, None, None, inspected))
    received = {}
    monkeypatch.setattr(curve_validation, "validate_saved_curve", lambda profile, **kw: (received.update(kw) or []))
    monkeypatch.setattr(webui, "_run_subprocess_task", lambda *a, **kw: None)
    curve = {"schema": "maichart-intensity-curve-v1", "audio_sha256": "a" * 64,
             "duration_sec": 10., "method": "learned_song_form_v2",
             "points": [{"time_sec": 0., "intensity": .5}, {"time_sec": 10., "intensity": .5}]}
    status, result = webui_api("/api/form/generate", {
        "audio": str(audio), "checkpoint": str(model), "output": str(tmp_path / "maidata.txt"),
        "model_family": "song_form_v2", "device": "cpu", "intensity_curve_enabled": True,
        "intensity_curve": curve, "meter_map": [{"beat": "0", "numerator": 3, "denominator": 4}],
    })
    assert status == 200, result
    assert received["meter_map"][0].numerator == 3
    assert received["quality_profile"] == "musical_evidence_v1"
