from types import SimpleNamespace

import pytest

from maichart.webui import composer_api


def _payload(tmp_path):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{}')
    return {'manifest': str(manifest), 'checkpoint_dir': str(tmp_path / 'checkpoints'),
            'cache_dir': str(tmp_path / 'cache')}


def _runtime(calls):
    return SimpleNamespace(
        _create_task=lambda kind: SimpleNamespace(task_id='task', kind=kind, result={}),
        _run_subprocess_task=lambda task, args, **kwargs: calls.append((task, args)))


def test_backend_preserves_recovery_outcomes_and_weighted_loss():
    from maichart.webui.server import TaskState,_update_composer_progress
    task=TaskState('recovery','composer_train')
    _update_composer_progress(task,'composer_recovery {"optimizer_step":100,"loss":0.03,"outcomes":{"accepted":1,"illegal_rollout_proposal":1}}')
    assert task.result['composer_recovery']['outcomes']['accepted']==1
    assert task.result['composer_recovery']['loss']==.03


def test_fresh_api_forwards_grounded_defaults_and_effective_settings(tmp_path):
    calls = []
    composer_api.start_train(SimpleNamespace(_json_response=lambda _: None),
                             _payload(tmp_path), runtime=_runtime(calls))
    task, args = calls[0]
    expected = {'architecture_profile': 'audio_grounded_v3', 'dropout': .1, 'history_anchor_policy':'parsed_geometry_v3', 'lens_width': 64,
                'lens_chunk_size': 32, 'history_residual_bound': 1., 'history_dropout': .3,
                'ledger_dropout': .5, 'placement_weight': .15, 'planner_weight': .2,
                'recovery_interval': 100, 'recovery_samples': 2, 'recovery_steps': 2,
                'recovery_weight': .1, 'action_state_policy':'committed_causal_v2',
                'motif_memory_policy':'off','slide_factor_policy':'branch_joint_v2','motif_max_events':16,
                'touch_layout_policy':'open'}
    assert task.result['effective_settings'] == expected
    for key, value in expected.items():
        flag = '--' + key.replace('_', '-')
        assert args[args.index(flag) + 1] == str(value)


@pytest.mark.parametrize(('key', 'value'), [
    ('architecture_profile', 'unknown'), ('action_slots', 65), ('lens_width', 7), ('lens_width', 257),
    ('lens_chunk_size', 0), ('lens_chunk_size', 257), ('history_residual_bound', .09),
    ('history_residual_bound', 4.1), ('history_dropout', .91), ('ledger_dropout', -.1),
    ('placement_weight', 1.1), ('planner_weight', float('nan')), ('recovery_interval', -1),
    ('recovery_samples', 9), ('recovery_steps', 1), ('recovery_steps', 5), ('recovery_weight', 1.1),
])
def test_grounded_api_rejects_invalid_values_before_task_creation(tmp_path, key, value):
    payload = _payload(tmp_path)
    payload[key] = value
    runtime = SimpleNamespace(_create_task=lambda _: pytest.fail('Invalid input created a task'))
    with pytest.raises(ValueError):
        composer_api.start_train(SimpleNamespace(_json_response=lambda _: None), payload, runtime=runtime)


def test_resume_api_uses_inspected_parent_profile_and_grounded_settings(tmp_path, monkeypatch):
    from maichart.composer import checkpoint
    parent = tmp_path / 'parent.pt'
    parent.write_bytes(b'checkpoint')
    monkeypatch.setattr(checkpoint, 'read_payload', lambda _: {})
    monkeypatch.setattr(checkpoint, 'inspect_payload', lambda _: {
        'can_resume': True, 'model_config': {'architecture_profile': 'music_context_v2', 'action_slots': 9},
        'args': {'recovery_interval': 250, 'recovery_weight': .2}})
    payload = _payload(tmp_path)
    payload.update(mode='resume', resume=str(parent))
    calls = []
    composer_api.start_train(SimpleNamespace(_json_response=lambda _: None), payload, runtime=_runtime(calls))
    task, args = calls[0]
    assert args[args.index('--architecture-profile') + 1] == 'music_context_v2'
    assert args[args.index('--action-slots') + 1] == '9'
    assert task.result['effective_settings']['recovery_interval'] == 250
    assert task.result['effective_settings']['recovery_weight'] == .2
    assert task.result['settings_policy'] == 'resume_inherited'
    payload['recovery_weight'] = .4
    with pytest.raises(ValueError, match='resume inherits recovery_weight'):
        composer_api.start_train(SimpleNamespace(_json_response=lambda _: None), payload, runtime=_runtime([]))
    payload.pop('recovery_weight')
    payload['action_slots'] = 8
    with pytest.raises(ValueError, match='resume inherits action_slots'):
        composer_api.start_train(SimpleNamespace(_json_response=lambda _: None), payload, runtime=_runtime([]))


@pytest.mark.parametrize('value', [-.1, 1.1, float('nan'), float('inf')])
def test_generation_api_validates_history_strength_before_task_creation(tmp_path, value):
    audio = tmp_path / 'audio.wav'
    audio.write_bytes(b'audio')
    checkpoint = tmp_path / 'checkpoint.pt'
    checkpoint.write_bytes(b'checkpoint')
    runtime = SimpleNamespace(_create_task=lambda _: pytest.fail('Invalid strength created a task'))
    with pytest.raises(ValueError):
        composer_api.start_generate(SimpleNamespace(_json_response=lambda _: None),
            {'audio': str(audio), 'checkpoint': str(checkpoint), 'pointer_history_strength': value}, runtime=runtime)
