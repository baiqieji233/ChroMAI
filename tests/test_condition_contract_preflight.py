"""Condition dropout must fail before dataset/audio setup, with exact UI terms."""
import json

import pytest

from maichart.training.event_checkpoints import compare_event_contract
from maichart.webui import server as webui
from test_webui import webui_api


def test_wrong_condition_dropout_fails_before_dispatch(webui_api, monkeypatch, tmp_path):
    manifest = tmp_path / "training_manifest.json"
    manifest.write_text(json.dumps({"schema": "maichart-training-manifest-v1", "songs": []}), encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"fixture")
    stored = webui._event_condition_parameters({"condition_dropout": .15})
    monkeypatch.setattr(webui, "_checkpoint_inspection", lambda *args, **kwargs: {
        "errors": [], "model_config": {"planner_profile": "legacy_v1"}, "data_contract": {"condition_params": stored}})
    calls = []
    monkeypatch.setattr(webui, "_run_subprocess_task", lambda task, args, **kwargs: calls.append(args))
    payload = {"model_family": "event_ctx_v1", "manifest": str(manifest), "training_mode": "finetune",
               "training_checkpoint": str(checkpoint), "dropout": .15, "context_dropout": .15}
    status, result = webui_api("/api/train", payload)
    assert status == 400
    assert "condition_dropout" in result["error"]
    assert "checkpoint=0.15" in result["error"] and "本次=0.0" in result["error"]
    assert not calls and not webui.TASKS
    status, result = webui_api("/api/train", {**payload, "condition_dropout": .15})
    assert status == 200, result
    from maichart.training.train_events import build_parser
    parsed = build_parser().parse_args(calls[0][2:])
    assert parsed.condition_dropout == .15
    assert parsed.dropout == .15 and parsed.context_dropout == .15


@pytest.mark.parametrize("field,value", [("condition_dropout", "nan"), ("condition_dropout", -.1),
                                         ("condition_mask_prob", 2), ("condition_default", "invalid")])
def test_invalid_condition_controls_are_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        webui._event_condition_parameters({field: value})


def test_cli_condition_mismatch_explains_three_independent_dropouts():
    stored = {"data_contract": {"condition_params": webui._event_condition_parameters({"condition_dropout": .15})}}
    current = {"condition_params": webui._event_condition_parameters({})}
    with pytest.raises(ValueError, match="条件随机丢弃") as error:
        compare_event_contract(stored, current, mode="init")
    assert "--condition-dropout" in str(error.value)
    assert "--context-dropout" in str(error.value)
