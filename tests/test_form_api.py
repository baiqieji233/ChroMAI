import json
from pathlib import Path

import pytest
import torch

from maichart.form import CHECKPOINT_SCHEMA, FAMILY
from maichart.form.blueprint import BlueprintConfig, SongBlueprint
from maichart.form.checkpoint import INPUT_CONTRACT, payload
from maichart.form.codec import RationalCodec
from maichart.form.model import FormConfig, FormTransformer
from maichart.webui import form_api
from maichart.webui.form_api import update_form_progress
from maichart.webui.server import TaskState


class _Response:
    def _json_response(self, value):
        return value


def _checkpoint(path, *, draft=1, revision=1, optimizer=1, blueprint=1):
    codec = RationalCodec()
    model = FormTransformer(FormConfig(vocab_size=len(codec), d_model=16, nhead=1,
                                       layers=1, dim_feedforward=32,
                                       max_tokens=32, max_memory_tokens=16))
    planner = SongBlueprint(BlueprintConfig(width=16, form_heads=4, form_layers=1,
                                            temporal_layers=1))
    data = payload(model, planner, codec, epoch=2, optimizer_steps=optimizer,
                   blueprint_steps=blueprint, task_training={"draft": draft, "revision": revision},
                   input_contract=INPUT_CONTRACT,
                   dataset_contract={"max_region_seconds": 14, "meter_map": None},
                   args={"lr": 0.0003, "limit_charts": 7},
                   optimizer_state_dict={"param_groups": [{"lr": 0.00025}]})
    # payload() supplies the dataclass config; keep the serialized shape explicit.
    data["schema"] = CHECKPOINT_SCHEMA
    data["family"] = FAMILY
    torch.save(data, path)


def test_form_progress_updates_epoch_fraction_and_keeps_unknown_prepare_indeterminate():
    task = TaskState(task_id="form-progress", kind="form_train")
    task.result["epoch_target"] = 4

    update_form_progress(task, "form_prepare {\"stage\": \"dataset\"}")
    assert task.result["phase"] == "prepare"
    assert task.result["progress_known"] is False
    assert task.progress == 0

    update_form_progress(
        task,
        "form_progress "
        + json.dumps({"epoch": 2, "phase": "layout", "overall_fraction": 0.5}),
    )
    assert task.result["progress_known"] is True
    assert task.result["phase"] == "layout"
    assert task.progress == 0.375


def test_form_progress_ignores_malformed_and_non_form_lines():
    task = TaskState(task_id="form-progress-invalid", kind="form_train")
    task.result["epoch_target"] = 3
    update_form_progress(task, "form_progress not-json")
    update_form_progress(task, "epoch=1 loss=2")
    assert task.result == {"epoch_target": 3}
    assert task.progress == 0


def test_form_progress_never_changes_legacy_task_state():
    task = TaskState(task_id="legacy", kind="train")
    update_form_progress(task, "form_epoch {\"epoch\": 2}")
    assert task.result == {}
    assert task.progress == 0


@pytest.mark.parametrize(("draft", "revision", "can_generate", "can_revise"),
                         [(1, 1, True, True), (0, 1, False, True), (1, 0, True, False)])
def test_inspect_form_checkpoint_reports_task_capabilities(tmp_path, draft, revision,
                                                            can_generate, can_revise):
    path = tmp_path / "form.pt"
    _checkpoint(path, draft=draft, revision=revision)
    result = form_api.inspect_form_checkpoint(path)
    assert result["valid"] is True
    assert result["model_family"] == FAMILY
    assert result["can_generate"] is can_generate
    assert result["can_revise"] is can_revise
    assert result["dataset_contract"]["max_region_seconds"] == 14
    assert result["resume_settings"]["lr"] == 0.00025
    assert result["resume_settings"]["limit_charts"] == 7


def test_resume_inherits_checkpoint_contract_before_validating_values(monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    checkpoint = tmp_path / "form.pt"
    _checkpoint(checkpoint)
    calls = []

    class Runtime:
        WEBUI_ROOT = tmp_path
        def _resolve_path(self, value): return Path(value)
        def _create_task(self, kind):
            task = TaskState(task_id="resume", kind=kind); calls.append(task); return task
        def _run_subprocess_task(self, task, args, **kwargs): calls.append(args)

    result = form_api.start_form_task(_Response(), "train", {
        "manifest": str(manifest), "checkpoint_dir": str(tmp_path / "out"),
        "mode": "resume", "init_checkpoint": str(checkpoint),
        "lr": 1.0, "limit_charts": 999999, "max_region_seconds": 20,
        "d_model": 15, "nhead": 64, "meter_map": [{"bad": "stale"}],
    }, runtime=Runtime())
    assert result == {"task_id": "resume"}
    args = calls[-1]
    assert args[args.index("--lr") + 1] == "0.00025"
    assert args[args.index("--limit-charts") + 1] == "7"
    assert args[args.index("--max-region-seconds") + 1] == "14.0"
    assert args[args.index("--d-model") + 1] == "16"
    assert args[args.index("--nhead") + 1] == "1"


def test_bad_manifest_does_not_create_queued_task(tmp_path):
    manifest = tmp_path / "bad.json"
    manifest.write_text("{", encoding="utf-8")
    created = []
    class Runtime:
        WEBUI_ROOT = tmp_path
        def _resolve_path(self, value): return Path(value)
        def _create_task(self, kind): created.append(kind); return TaskState(task_id="bad", kind=kind)
    with pytest.raises(json.JSONDecodeError):
        form_api.start_form_task(_Response(), "train", {
            "manifest": str(manifest), "checkpoint_dir": str(tmp_path / "out")
        }, runtime=Runtime())
    assert created == []


def test_inspect_rejects_old_and_corrupt_checkpoints(tmp_path):
    old = tmp_path / "old.pt"
    torch.save({"schema": "maichart-event-generator-checkpoint-v1",
                "model_state_dict": {"weight": torch.zeros(1)}}, old)
    result = form_api.inspect_form_checkpoint(old)
    assert result["valid"] is False and result["route"] == "/"
    corrupt = tmp_path / "corrupt.pt"
    torch.save({"schema": CHECKPOINT_SCHEMA, "family": FAMILY,
                "input_contract": INPUT_CONTRACT,
                "model_state_dict": {"weight": torch.zeros(1)}}, corrupt)
    result = form_api.inspect_form_checkpoint(corrupt)
    assert result["valid"] is False and result["errors"]


def test_start_form_task_uses_injected_runtime_task_registry(monkeypatch, tmp_path):
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"audio")
    created = []

    class Runtime:
        WEBUI_ROOT = tmp_path

        def _resolve_path(self, value):
            return Path(value)

        def _create_task(self, kind):
            task = TaskState(task_id="injected", kind=kind)
            created.append(task)
            return task

        def _run_subprocess_task(self, task, args, **kwargs):
            task.status = "queued"

    runtime = Runtime()
    result = form_api.start_form_task(_Response(), "analyze",
                                      {"audio": str(audio), "model_family": FAMILY},
                                      runtime=runtime)
    assert result == {"task_id": "injected"}
    assert created[0].kind == "form_analyze"
