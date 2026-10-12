"""Visible training choices must survive WebUI -> CLI and continuation guards."""
import json

import pytest

from maichart.training.action_state_policy import resolve_action_state_profile
from maichart.webui import server as webui
from test_webui import webui_api


@pytest.mark.parametrize("mode,stored,requested,expected,status_expected", [
    ("fresh", "none", "sustain_state_v1", "sustain_state_v1", 200),
    ("finetune", "none", "sustain_state_v1", "sustain_state_v1", 200),
    ("resume", "sustain_state_v1", "auto", "sustain_state_v1", 200),
    ("resume", "none", "sustain_state_v1", None, 400),
])
def test_webui_forwards_state_profile_or_rejects_before_dispatch(
        webui_api, tmp_path, monkeypatch, mode, stored, requested, expected, status_expected):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": "maichart-training-manifest-v1", "songs": [],
        "preprocessing": {"target_schema": "event_v1", "feature_profile": "event_audio_v2_rhythm"}}))
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"test fixture")
    monkeypatch.setattr(webui, "_checkpoint_inspection", lambda *a, **kw: {
        "errors": [], "epoch": 2, "model_config": {"planner_profile": "music_structure_v3",
            "action_state_profile": stored}, "data_contract": {}})
    calls = []
    monkeypatch.setattr(webui, "_run_subprocess_task", lambda task, args, **kw: calls.append(args))
    status, result = webui_api("/api/train", {
        "manifest": str(manifest), "model_family": "event_ctx_v1", "training_mode": mode,
        "training_checkpoint": str(checkpoint), "checkpoint_dir": str(tmp_path / "output"),
        "action_state_profile": requested, "local_pooling_policy": "auto", "validation_manifest": None,
    })
    assert status == status_expected, result
    if expected is None:
        assert "finetune" in result["error"] and not calls
    else:
        from maichart.training.train_events import build_parser
        args = build_parser().parse_args(calls[0][2:])
        assert args.action_state_profile == expected
        assert args.local_pooling_policy == ("legacy_uniform" if mode == "resume" else "per_sample_v2")
        assert webui.TASKS[result["task_id"]].result["selection_metric"] == "training_loss"


def test_optional_state_inherits_and_cannot_silently_remove_trained_input():
    assert resolve_action_state_profile() == "none"
    assert resolve_action_state_profile(checkpoint={"model_config": {}}, mode="finetune") == "none"
    stored = {"model_config": {"action_state_profile": "sustain_state_v1"}}
    assert resolve_action_state_profile(checkpoint=stored, mode="finetune") == "sustain_state_v1"
    with pytest.raises(ValueError, match="保留该能力"):
        resolve_action_state_profile("none", checkpoint=stored, mode="finetune")
