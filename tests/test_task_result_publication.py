import maichart.webui.server as webui
from maichart.webui.server import TaskState


class _InlineThread:
    def __init__(self, *, target, **_kwargs):
        self.target = target

    def start(self):
        self.target()


class _FinishedProcess:
    stdout = ("child finished\n",)

    def __init__(self, code=0):
        self.code = code

    def wait(self):
        return self.code


def _run_inline(monkeypatch, task, collect, *, paths=(), code=0, close=None):
    process = _FinishedProcess(code)
    monkeypatch.setattr(webui.threading, "Thread", _InlineThread)
    monkeypatch.setattr(webui.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(webui, "_open_task_log", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(webui, "_collect_task_results", collect)
    monkeypatch.setattr(webui, "_close_task_log", close or (lambda *_args, **_kwargs: None))
    webui._run_subprocess_task(task, ["fake-child"], expected_epoch=None, result_paths=list(paths))


def test_task_stays_running_until_results_are_collected_and_then_publishes_complete(tmp_path, monkeypatch):
    task = TaskState(task_id="publication", kind="generate")
    output = tmp_path / "chart.txt"
    report = tmp_path / "chart.form.json"
    declared = [output, report]
    events = []

    def collect(current, paths):
        assert current.status == "running"
        assert current.to_dict()["status"] == "running"
        for path in declared:
            path.write_text("ready", encoding="utf-8")
        current.result["paths"] = [str(path) for path in declared]
        current.result["files"] = [path.name for path in declared]
        events.append("collected")

    def close(current, *, status, **_kwargs):
        events.append(("footer", status, current.status))

    process = _FinishedProcess()
    monkeypatch.setattr(webui.threading, "Thread", _InlineThread)
    monkeypatch.setattr(webui.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(webui, "_open_task_log", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(webui, "_collect_task_results", collect)
    monkeypatch.setattr(webui, "_close_task_log", close)

    webui._run_subprocess_task(task, ["fake-child"], expected_epoch=None, result_paths=declared)

    assert events == ["collected", ("footer", "completed", "running")]
    assert task.status == "completed"
    assert task.result["files"] == ["chart.txt", "chart.form.json"]
    assert "missing_artifacts" not in task.result


def test_missing_declared_results_are_recorded_without_failing_the_child(tmp_path, monkeypatch):
    task = TaskState(task_id="missing", kind="form_generate")
    present = tmp_path / "chart.txt"
    missing = tmp_path / "chart.form.json"
    present.write_text("ready", encoding="utf-8")

    def collect(current, _paths):
        assert current.status == "running"
        current.result["paths"] = [str(present)]
        current.result["files"] = [present.name]

    _run_inline(monkeypatch, task, collect, paths=[present, missing])

    assert task.status == "completed"
    assert task.message == "任务完成，但部分输出文件缺失"
    assert task.result["missing_artifacts"] == [str(missing)]
    assert any("artifact_warning" in line for line in task.logs)


def test_collection_exception_marks_task_failed_without_completed_footer(monkeypatch):
    task = TaskState(task_id="collect-error", kind="form_generate")
    footers = []

    def collect(_current, _paths):
        raise OSError("artifact scan failed")

    def close(_current, *, status, error=None, **_kwargs):
        footers.append((status, error))

    _run_inline(monkeypatch, task, collect, close=close)

    assert task.status == "failed"
    assert task.error == "artifact scan failed"
    assert footers == [("failed", "artifact scan failed")]
