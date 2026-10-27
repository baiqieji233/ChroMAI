"""Runtime controls capture argv only; no training subprocess is launched."""
from types import SimpleNamespace

import pytest

from maichart.webui import composer_api


@pytest.mark.parametrize('mode,saved,requested,expected', [
    ('fresh', 8, None, 2), ('fresh', 8, 2, 2),
    ('resume', 8, None, 8), ('resume', None, None, 0),
    ('resume', 8, 2, 2), ('finetune', 8, 4, 4),
    ('finetune', 8, None, 2), ('fresh', None, 256, 256),
])
def test_cpu_threads_runtime_override(tmp_path, monkeypatch, mode, saved, requested, expected):
    from maichart.composer import checkpoint
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{}')
    parent = tmp_path / 'parent.pt'
    parent.touch()
    monkeypatch.setattr(checkpoint, 'read_payload', lambda _: {})
    monkeypatch.setattr(checkpoint, 'inspect_payload', lambda _: {
        'can_resume': True, 'can_init': True, 'model_config': {},
        'args': {} if saved is None else {'cpu_threads': saved}})
    calls = []
    task = SimpleNamespace(task_id='captured', result={})
    runtime = SimpleNamespace(_create_task=lambda _: task,
        _run_subprocess_task=lambda _, args, **kwargs: calls.append(args))
    payload = {'mode': mode, 'manifest': str(manifest),
        'checkpoint_dir': str(tmp_path / 'out'), 'cache_dir': str(tmp_path / 'cache'), 'checkpoint': str(parent)}
    if requested is not None:
        payload['cpu_threads'] = requested
    composer_api.start_train(SimpleNamespace(_json_response=lambda _: None), payload, runtime=runtime)
    assert calls[0][calls[0].index('--cpu-threads') + 1] == str(expected)
    assert task.result['effective_settings']['cpu_threads'] == expected


@pytest.mark.parametrize('value', [-1, 257, 2.5, True, None, 'invalid'])
def test_cpu_threads_rejects_before_task(tmp_path, value):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{}')
    runtime = SimpleNamespace(_create_task=lambda _: pytest.fail('created invalid task'))
    with pytest.raises(ValueError, match='cpu_threads'):
        composer_api.start_train(SimpleNamespace(), {'manifest': str(manifest),
            'checkpoint_dir': str(tmp_path / 'out'), 'cache_dir': str(tmp_path / 'cache'), 'cpu_threads': value}, runtime=runtime)
