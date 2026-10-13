from pathlib import Path
from types import SimpleNamespace

from maichart.webui import form_api
from maichart.webui.server import TaskState


def test_task_outputs_keep_the_filename_but_isolate_directories(tmp_path):
    requested = tmp_path / 'maidata.txt'
    first = form_api.isolated_form_output(requested, 'task-a')
    second = form_api.isolated_form_output(requested, 'task-b')
    assert first.name == second.name == 'maidata.txt'
    assert first.parent != second.parent
    assert first.with_suffix('.form.json') != second.with_suffix('.form.json')
    assert requested.parent == tmp_path


def test_api_uses_task_specific_outputs_even_for_matching_requested_paths(monkeypatch, tmp_path):
    from maichart.form import checkpoint
    monkeypatch.setattr(checkpoint, 'load', lambda *a, **kw: None)
    model = tmp_path / 'model.pt'; model.write_bytes(b'fixture')
    audio = tmp_path / 'track.wav'; audio.write_bytes(b'fixture')
    tasks, calls = [], []

    class Runtime:
        WEBUI_ROOT = tmp_path
        def _resolve_path(self, value): return Path(value)
        def _create_task(self, kind):
            task = TaskState(f'task-{len(tasks)}', kind)
            tasks.append(task)
            return task
        def _run_subprocess_task(self, task, args, **kwargs): calls.append(args)

    requested = tmp_path / 'maidata.txt'
    for client in ('client-a', 'client-b'):
        form_api.start_form_task(SimpleNamespace(_json_response=lambda value: value), 'generate',
            {'checkpoint': str(model), 'audio': str(audio),
             'output': str(tmp_path / client / 'maidata.txt'),
             'requested_output': str(requested), 'device': 'cpu'}, runtime=Runtime())
    actual = [Path(args[args.index('--output')+1]) for args in calls]
    assert actual == [tmp_path/'task-0'/'maidata.txt', tmp_path/'task-1'/'maidata.txt']
    for task, path in zip(tasks, actual):
        assert task.result['actual_output'] == str(path)
        assert task.result['requested_output'] == str(requested)
        assert task.result['artifacts'][1]['path'] == str(path.with_suffix('.form.json'))


def test_resume_inspection_settings_match_the_actual_inherited_arguments():
    checkpoint = {'args': {'batch_size': 2, 'grad_accumulation': 16,
                           'condition_drop_prob': .15, 'load_curve_loss_weight': 1.,
                           'quality_enabled': True, 'max_steps': 0, 'limit_charts': 0,
                           'lr': .001, 'precision': 'auto'},
                  'precision': 'bf16', 'model_config': {'dropout': .1},
                  'dataset_contract': {'max_region_seconds': 14., 'meter_map': []},
                  'optimizer_state_dict': {'param_groups': [{'lr': .0002}]}}
    settings = form_api.form_resume_settings(checkpoint)
    assert settings['batch_size'] == 2 and settings['grad_accumulation'] == 16
    assert settings['max_steps'] == 0 and settings['quality_enabled'] is True
    assert settings['condition_drop_prob'] == .15 and settings['load_curve_loss_weight'] == 1.
    assert settings['lr'] == .0002 and settings['precision'] == 'bf16'
    assert settings['dropout'] == .1 and settings['max_region_seconds'] == 14.
