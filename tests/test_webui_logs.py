"""Run logs must be persisted in full, not just the 300-line live ring.

The interface keeps only the last 300 lines for the live panel; a long training
run therefore lost everything interesting (and a crash lost everything at all).
Each task now appends its whole stream to ``webui_workspace/logs`` plus a sidecar
with status/argv/exit code, and ``/api/logs`` lists them for download.
"""

import json

import maichart.webui.server as webui
from maichart.webui.server import TaskState


def _task(tmp_path, monkeypatch, **kwargs):
    monkeypatch.setattr(webui, "LOG_ROOT", tmp_path / "logs")
    task = TaskState(task_id="t1", kind="train", **kwargs)
    return task


def test_log_file_keeps_every_line_while_memory_stays_a_ring(tmp_path, monkeypatch):
    task = _task(tmp_path, monkeypatch)
    path = webui._open_task_log(task, ["-m", "maichart.training.train_event_ctx", "--epochs", "70"])
    assert path is not None and path.is_file()
    for index in range(500):
        task.log(f"event_audio_loads={index} last=42 resident=0.10GiB spills=0")
    task.log("epoch=001 loss=1.234567 token_accuracy=0.5 val_loss=0.9")

    assert len(task.logs) == 300, "the live view stays bounded"
    assert task.logs[-1].endswith("epoch=001 loss=1.234567 token_accuracy=0.5 val_loss=0.9")

    text = path.read_text(encoding="utf-8")
    body = [line for line in text.splitlines() if line.startswith("[")]
    assert len(body) == 501, "the file keeps every line, including the epoch summary"
    assert "event_audio_loads=0 last=42" in text
    assert "event_audio_loads=499 last=42" in text
    assert "# argv=" in text and "train_event_ctx" in text


def test_finishing_a_task_writes_a_sidecar_the_history_list_reads(tmp_path, monkeypatch):
    # Downloads are only offered for files inside the served project root, so
    # pretend the temporary directory is the project.
    monkeypatch.setattr(webui, "PROJECT_ROOT", tmp_path)
    task = _task(tmp_path, monkeypatch)
    webui._open_task_log(task, ["-m", "maichart.training.prepare_events", "raw"])
    task.log("prepared=10 songs")
    webui._close_task_log(task, status="completed", exit_code=0)

    meta_path = task.log_path.with_suffix(".meta.json")
    assert meta_path.is_file()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["status"] == "completed" and meta["exit_code"] == 0
    assert meta["task_id"] == "t1" and meta["kind"] == "train"
    assert meta["log"] == str(task.log_path)
    assert meta["log_bytes"] > 0
    assert task.result["log_path"] == str(task.log_path)

    footer = task.log_path.read_text(encoding="utf-8")
    assert "# status=completed" in footer and "# exit_code=0" in footer

    entries = webui._persisted_logs()
    assert len(entries) == 1
    assert entries[0]["status"] == "completed"
    assert entries[0]["bytes"] > 0
    assert entries[0]["download"] and "/api/download?path=" in entries[0]["download"]


def test_logs_outside_the_project_are_listed_but_not_downloadable(tmp_path, monkeypatch):
    task = _task(tmp_path, monkeypatch)  # LOG_ROOT is tmp_path/logs, project root is elsewhere
    webui._open_task_log(task, ["-m", "x"])
    webui._close_task_log(task, status="completed", exit_code=0)
    entries = webui._persisted_logs()
    assert entries and entries[0]["log_path"].endswith(".log")
    assert entries[0]["download"] is None, "no download link outside the served root"


def test_failure_records_the_reason(tmp_path, monkeypatch):
    task = _task(tmp_path, monkeypatch)
    webui._open_task_log(task, ["-m", "boom"])
    task.log("Traceback: ValueError")
    webui._close_task_log(task, status="failed", exit_code=1, error="子进程退出码 1")

    meta = json.loads(task.log_path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == "failed" and meta["exit_code"] == 1
    assert "退出码 1" in meta["error"]
    assert "# error=子进程退出码 1" in task.log_path.read_text(encoding="utf-8")


def test_pruning_drops_whole_old_runs_and_keeps_the_newest(tmp_path, monkeypatch):
    monkeypatch.setattr(webui, "LOG_ROOT", tmp_path / "logs")
    for index in range(5):
        task = TaskState(task_id=f"t{index}", kind="train")
        webui._open_task_log(task, ["-m", f"run{index}"])
        task.log("x" * 100)
        webui._close_task_log(task, status="completed", exit_code=0)

    webui._prune_logs(max_files=2, max_bytes=10 ** 9)
    remaining = sorted(path.name for path in (tmp_path / "logs").glob("*.log"))
    assert len(remaining) == 2, remaining
    metas = sorted(path.name for path in (tmp_path / "logs").glob("*.meta.json"))
    assert len(metas) == 2, "a pruned run must lose its sidecar too"
    # The newest survives: its sidecar is still readable.
    assert webui._persisted_logs()


def test_logging_survives_an_unwritable_log_root(tmp_path, monkeypatch):
    """A broken log directory must not break the task itself."""

    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory", encoding="utf-8")
    monkeypatch.setattr(webui, "LOG_ROOT", blocker / "logs")
    task = TaskState(task_id="t9", kind="train")
    assert webui._open_task_log(task, ["-m", "x"]) is None
    assert "log_path_error" in task.result
    task.log("still works")  # no handle, no crash
    webui._close_task_log(task, status="completed", exit_code=0)
    assert len(task.logs) == 1 and task.logs[0].endswith("still works")
