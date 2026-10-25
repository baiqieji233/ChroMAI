import io
from pathlib import Path
from types import SimpleNamespace

import pytest

from maichart.webui import server


@pytest.fixture
def worker_harness(monkeypatch, tmp_path):
    workers = []
    monkeypatch.setattr(server, 'TASKS', {})
    monkeypatch.setattr(server, 'PROJECT_ROOT', tmp_path)
    monkeypatch.setattr(server, 'LOG_ROOT', tmp_path / 'logs')
    monkeypatch.setattr(server.threading, 'Thread', lambda **kw:
                        SimpleNamespace(start=lambda: workers.append(kw['target'])))
    monkeypatch.setattr(server, '_collect_task_results', lambda *args: None)
    task = server._reserve_composer_training_task(tmp_path / 'model')
    return task, workers, tmp_path / 'model'


def test_log_initialization_exception_finalizes_worker_and_releases_directory(worker_harness, monkeypatch):
    task, workers, directory = worker_harness
    def fail(*args):
        raise RuntimeError('mock log initialization failure')
    monkeypatch.setattr(server, '_open_task_log', fail)
    monkeypatch.setattr(server.subprocess, 'Popen', lambda *a, **k: pytest.fail('must not launch'))
    server._run_subprocess_task(task, [], expected_epoch=None, result_paths=[])
    workers.pop()()
    assert task.status == 'failed' and task.process is None
    assert 'mock log initialization failure' in task.error
    server._reserve_composer_training_task(directory)


@pytest.mark.parametrize('stage', ['stat', 'sidecar', 'prune'])
@pytest.mark.parametrize('outcome', ['completed', 'failed', 'cancelled'])
def test_close_log_io_errors_preserve_real_worker_outcome(worker_harness, monkeypatch, stage, outcome):
    task, workers, directory = worker_harness
    code = 7 if outcome == 'failed' else 0
    process = SimpleNamespace(stdout=io.StringIO(''), poll=lambda: code, wait=lambda: code)
    def popen(*args, **kwargs):
        if outcome == 'cancelled':
            task.cancel_requested = True
        return process
    monkeypatch.setattr(server.subprocess, 'Popen', popen)
    def fail(*args, **kwargs):
        raise PermissionError(f'mock {stage} access denied')
    if stage == 'stat':
        def open_log(task, args):
            task.log_path = SimpleNamespace(is_file=lambda: True, stat=fail)
        monkeypatch.setattr(server, '_open_task_log', open_log)
    elif stage == 'sidecar':
        monkeypatch.setattr(Path, 'write_text', fail)
    else:
        monkeypatch.setattr(server, '_prune_logs', fail)
    server._run_subprocess_task(task, [], expected_epoch=None, result_paths=[])
    workers.pop()()
    assert task.status == outcome and task.process is None and task._handle is None
    assert task.result['log_close_error'] == f'mock {stage} access denied'
    if outcome == 'completed':
        assert task.error is None
    elif outcome == 'failed':
        assert '7' in task.error
    server._reserve_composer_training_task(directory)


def test_log_open_permission_failure_still_allows_success(worker_harness, monkeypatch):
    task, workers, _ = worker_harness
    def fail():
        raise PermissionError('mock log directory access denied')
    monkeypatch.setattr(server, '_logs_root', fail)
    monkeypatch.setattr(server.subprocess, 'Popen', lambda *a, **k:
                        SimpleNamespace(stdout=io.StringIO(''), poll=lambda: 0, wait=lambda: 0))
    server._run_subprocess_task(task, [], expected_epoch=None, result_paths=[])
    workers.pop()()
    assert task.status == 'completed' and task.error is None
    assert task.result['log_path_error'] == 'mock log directory access denied'
