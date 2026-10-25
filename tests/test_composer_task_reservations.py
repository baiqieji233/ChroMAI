import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from maichart.webui import composer_api, server
from maichart.webui import subprocess_control


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(server, 'TASKS', {})
    monkeypatch.setattr(server, 'PROJECT_ROOT', tmp_path)
    monkeypatch.setattr(server, '_run_subprocess_task', lambda *a, **k: None)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{}')
    payload = dict(manifest=str(manifest), checkpoint_dir=str(tmp_path / 'model'),
                   cache_dir=str(tmp_path / 'cache'))
    return payload, SimpleNamespace(_json_response=lambda body: None)


def test_concurrent_requests_reserve_same_directory_once(runtime):
    payload, handler = runtime
    barrier = threading.Barrier(2)
    def submit():
        barrier.wait(timeout=3)
        try:
            composer_api.start_train(handler, payload, runtime=server)
            return 'accepted'
        except ValueError as exc:
            assert 'checkpoint' in str(exc)
            return 'busy'
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(lambda _: submit(), range(2))) == ['accepted', 'busy']
    assert len(server.TASKS) == 1


@pytest.mark.parametrize('status', ['queued', 'running', 'cancelling'])
def test_canonical_directory_remains_reserved_until_terminal(runtime, status):
    payload, _ = runtime
    task = server._reserve_composer_training_task(Path('model'))
    task.status = status
    absolute = Path(payload['checkpoint_dir']) / '..' / 'model'
    with pytest.raises(ValueError, match='checkpoint'):
        server._reserve_composer_training_task(absolute)
    if os.name == 'nt':
        with pytest.raises(ValueError, match='checkpoint'):
            server._reserve_composer_training_task(Path(str(absolute).upper()))
    # Explicitly distinct outputs are permitted even while this task cancels.
    server._reserve_composer_training_task(Path('other-model'))
    for terminal in ('completed', 'failed', 'cancelled'):
        task.status = terminal
        task = server._reserve_composer_training_task(absolute)


def test_validation_precedes_reservation_and_launch_failure_releases(runtime, monkeypatch):
    payload, handler = runtime
    with pytest.raises(ValueError):
        composer_api.start_train(handler, dict(payload, epochs=0), runtime=server)
    assert not server.TASKS
    def fail(*args, **kwargs):
        raise RuntimeError('mock thread launch failed')
    monkeypatch.setattr(server, '_run_subprocess_task', fail)
    with pytest.raises(RuntimeError, match='mock thread'):
        composer_api.start_train(handler, payload, runtime=server)
    assert next(iter(server.TASKS.values())).status == 'failed'
    monkeypatch.setattr(server, '_run_subprocess_task', lambda *a, **k: None)
    composer_api.start_train(handler, payload, runtime=server)
    assert len(server.TASKS) == 2


def test_cancel_failure_is_visible_and_retryable_without_duplicate_pending(runtime, monkeypatch):
    _, handler = runtime
    replies = []
    handler._json_response = replies.append
    task = server._create_task('composer_train')
    task.status = 'running'
    task.process = SimpleNamespace(poll=lambda: None)
    entered, release = threading.Event(), threading.Event()
    calls = []
    def terminate(process):
        calls.append(process)
        entered.set()
        assert release.wait(timeout=3)
        raise RuntimeError('mock taskkill failure')
    monkeypatch.setattr(subprocess_control, 'terminate_task_process', terminate)
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(server.WebUIHandler._cancel_task, handler, {'task_id': task.task_id})
        assert entered.wait(timeout=3)
        server.WebUIHandler._cancel_task(handler, {'task_id': task.task_id})
        assert len(calls) == 1 and replies[-1]['cancel_pending']
        release.set()
        first.result(timeout=3)
    assert task.status == 'cancelling' and task.process.poll() is None
    assert not task.cancel_pending and task.cancel_error == 'mock taskkill failure'
    assert replies[-1]['cancel_retryable']
    assert '尚未确认停止' in replies[-1]['message']
    monkeypatch.setattr(subprocess_control, 'terminate_task_process', lambda p: calls.append(p))
    server.WebUIHandler._cancel_task(handler, {'task_id': task.task_id})
    assert len(calls) == 2 and task.cancel_error is None
    assert task.status == 'cancelling'  # only the worker can confirm exit
    with task.lock:
        server._finish_cancelled_unlocked(task)
    assert not task.to_dict()['cancel_retryable'] and task.cancel_error is None
