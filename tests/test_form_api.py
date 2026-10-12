import json

from maichart.webui.form_api import update_form_progress
from maichart.webui.server import TaskState


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
