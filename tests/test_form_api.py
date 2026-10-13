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


def test_resume_effective_settings_match_checkpoint_and_ignore_disabled_ui_values(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    checkpoint = tmp_path / "form.pt"
    _checkpoint(checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved["precision"] = "bf16"
    saved["args"].update({
        "batch_size": 3, "grad_accumulation": 4, "seed": 12,
        "weight_decay": 0.07, "condition_drop_prob": 0.3,
        "blueprint_loss_weight": 0.4, "load_curve_loss_weight": 0.6,
        "relation_loss_weight": 0.2, "quality_enabled": False,
        "max_steps": 9, "precision": "fp16",
    })
    torch.save(saved, checkpoint)
    calls = []

    class Runtime:
        WEBUI_ROOT = tmp_path
        def _resolve_path(self, value): return Path(value)
        def _create_task(self, kind):
            task = TaskState(task_id="resume-settings", kind=kind); calls.append(task); return task
        def _run_subprocess_task(self, task, args, **kwargs): calls.append(args)

    form_api.start_form_task(_Response(), "train", {
        "manifest": str(manifest), "checkpoint_dir": str(tmp_path / "out"),
        "mode": "resume", "init_checkpoint": str(checkpoint), "control_profile": "none",
        "batch_size": 0, "grad_accumulation": 0, "seed": -1, "weight_decay": -1,
        "condition_drop_prob": 2, "blueprint_loss_weight": -1,
        "load_curve_loss_weight": -1, "relation_loss_weight": -1,
        "quality_enabled": True, "max_steps": 1000001, "precision": "invalid",
    }, runtime=Runtime())
    task, args = calls
    effective = task.result["effective_settings"]
    expected = {"batch_size": 3, "grad_accumulation": 4, "seed": 12,
                "weight_decay": 0.07, "condition_drop_prob": 0.3,
                "blueprint_loss_weight": 0.4, "load_curve_loss_weight": 0.6,
                "relation_loss_weight": 0.2, "quality_enabled": False,
                "max_steps": 9, "precision": "bf16"}
    for key, value in expected.items():
        assert effective[key] == value
    for key, value in expected.items():
        if key == "quality_enabled":
            assert ("--quality-enabled" in args) is value
            assert ("--no-quality-enabled" in args) is (not value)
            continue
        flag = "--" + key.replace("_", "-")
        assert args[args.index(flag) + 1] == str(value)
    assert args[args.index("--lr") + 1] == "0.00025"


def test_train_forwards_song_form_runtime_controls_as_cli_flags(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    calls = []

    class Runtime:
        WEBUI_ROOT = tmp_path
        def _resolve_path(self, value): return Path(value)
        def _create_task(self, kind): return TaskState(task_id="runtime-controls", kind=kind)
        def _run_subprocess_task(self, task, args, **kwargs): calls.append(args)

    form_api.start_form_task(_Response(), "train", {
        "manifest": str(manifest), "checkpoint_dir": str(tmp_path / "out"),
        "precision": "bf16", "loader_workers": 2, "prefetch_factor": 3,
        "log_interval": 4.5, "checkpoint_interval_steps": 80,
        "audio_cache_gb": 1.5, "audio_cache_mode": "lru", "sample_cache_mb": 512,
    }, runtime=Runtime())
    args = calls[0]
    for flag, expected in (("--precision", "bf16"), ("--loader-workers", "2"),
                           ("--prefetch-factor", "3"), ("--log-interval", "4.5"),
                           ("--checkpoint-interval-steps", "80"),
                           ("--audio-cache-gb", "1.5"), ("--audio-cache-mode", "lru"),
                           ("--sample-cache-mb", "512")):
        assert args[args.index(flag) + 1] == expected


@pytest.mark.parametrize(("workers", "budget", "message"), [
    (1, 2, "worker=0"), (0, 0, "硬预算"),
])
def test_resident_audio_cache_rejects_unsafe_runtime_settings(tmp_path, workers, budget, message):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    created = []

    class Runtime:
        WEBUI_ROOT = tmp_path
        def _resolve_path(self, value): return Path(value)
        def _create_task(self, kind):
            created.append(kind)
            return TaskState(task_id="resident-rejected", kind=kind)

    with pytest.raises(ValueError, match=message):
        form_api.start_form_task(_Response(), "train", {
            "manifest": str(manifest), "checkpoint_dir": str(tmp_path / "out"),
            "audio_cache_mode": "resident", "loader_workers": workers,
            "audio_cache_gb": budget,
        }, runtime=Runtime())
    assert created == []


def test_form_progress_keeps_new_fields_and_checkpoint_events():
    task = TaskState(task_id="form-progress-fields", kind="form_train")
    detail = {"epoch": 1, "phase": "draft", "unit": "batch", "completed": 3,
              "total": 12, "overall_fraction": 0.25, "phase_fraction": 0.5,
              "optimizer_steps": 2, "blueprint_steps": 1, "elapsed_sec": 30,
              "phase_elapsed_sec": 12, "items_per_sec": 0.25, "eta_phase_sec": 24,
              "losses": {"draft": 0.5}, "timing": {"compute_sec": 2.0},
              "lr": 0.0002, "precision": "bf16", "memory": {"allocated_gb": 2.0}}
    update_form_progress(task, "form_progress " + json.dumps(detail))
    assert task.result["progress_detail"] == detail
    update_form_progress(task, 'form_checkpoint {"path":"form_last.pt","epoch":1,"phase":"draft","batch":3,"optimizer_steps":2,"resumable":true,"elapsed_sec":30}')
    assert task.result["last_checkpoint"]["path"] == "form_last.pt"
    assert task.result["progress_detail"]["optimizer_steps"] == 2


def test_form_audio_cache_event_is_exposed_as_current_progress():
    task = TaskState(task_id="form-audio-cache", kind="form_train")
    event = {"mode": "resident", "phase": "audio", "completed": 2, "total": 4,
             "bytes": 1024, "required_bytes": 2048, "budget_bytes": 4096,
             "unique_audio": 4, "saved_reads": 3}
    update_form_progress(task, "form_audio_cache " + json.dumps(event))
    assert task.result["phase"] == "audio_cache"
    assert task.result["progress_detail"] == event
    assert task.result["progress_known"] is True


@pytest.mark.parametrize("first", [0.5, -0.5])
def test_generation_forwards_first_and_bounded_onset_merge_ms(monkeypatch, tmp_path, first):
    checkpoint = tmp_path / "form_last.pt"
    audio = tmp_path / "song.wav"
    checkpoint.write_bytes(b"checkpoint")
    audio.write_bytes(b"audio")
    from maichart.form import checkpoint as form_checkpoint
    monkeypatch.setattr(form_checkpoint, "load", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(form_api.secrets, "randbelow", lambda upper: 123456)
    calls = []
    tasks = []

    class Runtime:
        WEBUI_ROOT = tmp_path
        def _resolve_path(self, value): return Path(value)
        def _create_task(self, kind):
            task = TaskState(task_id=f"onset-merge-{len(tasks)}", kind=kind)
            tasks.append(task)
            return task
        def _run_subprocess_task(self, task, args, **kwargs): calls.append(args)

    form_api.start_form_task(_Response(), "generate", {
        "checkpoint": str(checkpoint), "audio": str(audio),
        "output": str(tmp_path / "out.txt"), "onset_merge_ms": 55.6,
        "first": first,
        "planner_type_strength": 3, "style_balance_strength": 0,
        "rhythm_snap_enabled": True, "rhythm_snap_ms": 8,
        "rhythm_snap_divisions": [64, 48, 64],
    }, runtime=Runtime())
    args = calls[0]
    assert args[args.index("--first") + 1] == str(first)
    assert args[args.index("--device") + 1] == "cpu"
    assert args[args.index("--onset-merge-ms") + 1] == "55.6"
    assert args[args.index("--planner-type-strength") + 1] == "3.0"
    assert args[args.index("--style-balance-strength") + 1] == "0.0"
    assert args[args.index("--rhythm-snap-ms") + 1] == "8.0"
    assert args[args.index("--rhythm-snap-divisions") + 1] == "48,64"
    assert "--rhythm-snap" in args
    assert args[args.index("--seed") + 1] == "123456"
    assert tasks[0].result["seed"] == 123456
    assert tasks[0].result["effective_settings"]["seed"] == 123456
    assert tasks[0].result["effective_settings"]["temperature"] == 0.7
    assert tasks[0].result["effective_settings"]["planner_type_strength"] == 3
    assert tasks[0].result["effective_settings"]["style_balance_strength"] == 0
    fixed = form_api.start_form_task(_Response(), "generate", {
        "checkpoint": str(checkpoint), "audio": str(audio), "output": str(tmp_path / "fixed.txt"),
        "seed": 42,
    }, runtime=Runtime())
    assert fixed["task_id"] == "onset-merge-1"
    assert calls[1][calls[1].index("--seed") + 1] == "42"
    assert tasks[1].result["effective_settings"]["seed"] == 42
    assert calls[1][calls[1].index("--planner-type-strength") + 1] == "1.0"
    assert calls[1][calls[1].index("--style-balance-strength") + 1] == "1.0"
    assert tasks[1].result["effective_settings"]["planner_type_strength"] == 1.0
    assert tasks[1].result["effective_settings"]["style_balance_strength"] == 1.0
    form_api.start_form_task(_Response(), "generate", {
        "checkpoint": str(checkpoint), "audio": str(audio), "output": str(tmp_path / "zero.txt"),
        "seed": 0,
    }, runtime=Runtime())
    assert calls[2][calls[2].index("--seed") + 1] == "0"
    assert tasks[2].result["effective_settings"]["seed"] == 0
    with pytest.raises(ValueError, match="onset_merge_ms out of range"):
        form_api.start_form_task(_Response(), "generate", {
            "checkpoint": str(checkpoint), "audio": str(audio),
            "output": str(tmp_path / "bad.txt"), "onset_merge_ms": 251,
        }, runtime=Runtime())
    for invalid in ([4.0], [True], "4,769", []):
        with pytest.raises(ValueError):
            form_api.start_form_task(_Response(), "generate", {
                "checkpoint": str(checkpoint), "audio": str(audio),
                "output": str(tmp_path / "bad-divisions.txt"),
                "rhythm_snap_enabled": True, "rhythm_snap_divisions": invalid,
            }, runtime=Runtime())


@pytest.mark.parametrize("field", ["planner_type_strength", "style_balance_strength"])
@pytest.mark.parametrize("value", [-0.1, 3.1, float("nan"), True])
def test_generation_type_guidance_strengths_are_bounded_before_task_creation(monkeypatch, tmp_path, field, value):
    checkpoint = tmp_path / "form.pt"
    audio = tmp_path / "song.wav"
    checkpoint.write_bytes(b"checkpoint")
    audio.write_bytes(b"audio")
    from maichart.form import checkpoint as form_checkpoint
    monkeypatch.setattr(form_checkpoint, "load", lambda *args, **kwargs: None)
    created = []

    class Runtime:
        WEBUI_ROOT = tmp_path
        def _resolve_path(self, value): return Path(value)
        def _create_task(self, kind):
            created.append(kind)
            return TaskState(task_id="invalid-guidance", kind=kind)

    with pytest.raises(ValueError):
        form_api.start_form_task(_Response(), "generate", {
            "checkpoint": str(checkpoint), "audio": str(audio),
            "output": str(tmp_path / "out.txt"), field: value,
        }, runtime=Runtime())
    assert created == []


def test_revision_rejects_rhythm_snap_request_before_task_creation(tmp_path):
    created = []

    class Runtime:
        WEBUI_ROOT = tmp_path
        def _resolve_path(self, value): return Path(value)
        def _create_task(self, kind):
            created.append(kind)
            return TaskState(task_id="revision-rhythm-snap", kind=kind)

    with pytest.raises(ValueError, match="修订入口不支持"):
        form_api.start_form_task(_Response(), "revise", {
            "rhythm_snap_enabled": True,
        }, runtime=Runtime())
    assert created == []


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
