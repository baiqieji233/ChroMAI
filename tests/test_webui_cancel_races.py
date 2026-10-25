import io
from types import SimpleNamespace

import pytest
import maichart.webui.server as webui
import maichart.webui.subprocess_control as control


@pytest.fixture
def deferred(monkeypatch, tmp_path):
    workers = []
    monkeypatch.setattr(webui.threading, "Thread", lambda *, target, **kw:
                        SimpleNamespace(start=lambda: workers.append(target)))
    monkeypatch.setattr(webui, "LOG_ROOT", tmp_path / "logs")
    monkeypatch.setattr(webui, "INTENSITY_ROOT", tmp_path / "curves")
    return workers


def handler():
    value = object.__new__(webui.WebUIHandler)
    value._json_response = lambda body: None
    return value


def cancel(task, monkeypatch):
    monkeypatch.setitem(webui.TASKS, task.task_id, task)
    handler()._cancel_task({"task_id": task.task_id})


def test_queued_cancel_never_spawns(deferred, monkeypatch):
    task = webui.TaskState("cancel-queued", "train")
    webui._run_subprocess_task(task, [], expected_epoch=None, result_paths=[])
    cancel(task, monkeypatch)
    assert task.status == "cancelling" and task.cancel_requested
    monkeypatch.setattr(webui.subprocess, "Popen", lambda *a, **k: pytest.fail("spawned"))
    deferred.pop()()
    assert task.status == "cancelled"


def test_cancel_during_popen_publication_terminates_and_does_not_complete(deferred, monkeypatch):
    task = webui.TaskState("cancel-launch", "train")
    process = SimpleNamespace(stdout=io.StringIO(""), poll=lambda: None, wait=lambda: -15)
    killed = []
    monkeypatch.setattr(control, "terminate_task_process", lambda value: killed.append(value))
    def popen(*args, **kwargs):
        cancel(task, monkeypatch)
        assert task.process is None
        return process
    monkeypatch.setattr(webui.subprocess, "Popen", popen)
    webui._run_subprocess_task(task, [], expected_epoch=None, result_paths=[])
    deferred.pop()()
    assert killed == [process]
    assert task.status == "cancelled" and task.process is None


@pytest.mark.parametrize("raises", [False, True])
def test_intensity_cancel_waits_for_uninterruptible_stage_and_wins(deferred, monkeypatch, tmp_path, raises):
    import maichart.generation.music_intensity as intensity
    task = webui.TaskState("cancel-intensity", "intensity")
    monkeypatch.setattr(webui, "_create_task", lambda kind: task)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"mock")
    def analyze(*args, **kwargs):
        cancel(task, monkeypatch)
        assert task.status == "cancelling"  # no claim that a Python thread was killed
        if raises:
            raise RuntimeError("stage stopped after cancellation")
        return {"method": "mock", "duration_sec": 1, "points": []}
    monkeypatch.setattr(intensity, "generate_intensity_profile", analyze)
    handler()._start_intensity_analysis({"audio": str(audio)})
    deferred.pop()()
    assert task.status == "cancelled"
    assert "intensity_profile" not in task.result
    assert not list((tmp_path / "curves").glob("*.json"))


def test_cancelling_preprocess_remains_busy_until_worker_stops(monkeypatch,tmp_path):
    manifest=tmp_path/'manifest.json'
    task=webui.TaskState('cancel-busy','preprocess',status='running',result={'manifest':str(manifest)})
    cancel(task,monkeypatch)
    assert task.status=='cancelling'
    pending=webui._manifest_pending_status(manifest)
    assert pending['state']=='processing' and not pending['ready']
    monkeypatch.setattr(webui,'_create_task',lambda kind:pytest.fail('cancel-pending task duplicated'))
    with pytest.raises(ValueError,match='正在预处理'):
        handler()._start_preprocess({'raw_root':str(tmp_path),'manifest':str(manifest)})
    with task.lock:
        webui._finish_cancelled_unlocked(task)
    assert webui._manifest_pending_status(manifest)['state']=='preprocess_failed'


def test_cancel_log_finalization_under_task_lock_does_not_deadlock(monkeypatch,tmp_path):
    import threading
    monkeypatch.setattr(webui,'LOG_ROOT',tmp_path/'logs')
    task=webui.TaskState('cancel-lock','preprocess',cancel_requested=True,status='cancelling')
    webui._open_task_log(task,['mock'])
    def finish():
        with task.lock:
            webui._finish_cancelled_unlocked(task)
        task.to_dict()
    thread=threading.Thread(target=finish,daemon=True)
    thread.start();thread.join(timeout=2)
    assert not thread.is_alive(), 'log closing must not reacquire the non-reentrant task lock'
    assert task.status=='cancelled' and task._handle is None
    assert 'status=cancelled' in task.log_path.read_text(encoding='utf-8')
