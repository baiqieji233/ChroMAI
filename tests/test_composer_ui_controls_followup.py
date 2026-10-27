"""CPU-only controls tests; subprocesses are captured and never launched."""
from types import SimpleNamespace

import pytest
import torch

from maichart.composer.action_guidance import action_logits_with_touch_guard
from maichart.composer.checkpoint import save, export_inference
from maichart.composer.model import ComposerConfig, build_model
from maichart.webui import composer_api


def _capture_runtime():
    tasks, calls, responses = [], [], []

    def create(kind):
        task = SimpleNamespace(task_id=f'fixture_{len(tasks)}', kind=kind, result={})
        tasks.append(task)
        return task

    runtime = SimpleNamespace(_create_task=create,
        _run_subprocess_task=lambda task, args, **kwargs: calls.append((task, args, kwargs)))
    return runtime, SimpleNamespace(_json_response=responses.append), tasks, calls, responses


def _generation_payload(tmp_path):
    audio = tmp_path/'audio.wav'
    checkpoint = tmp_path/'checkpoint.pt'
    audio.write_bytes(b'fixture: never decoded')
    checkpoint.write_bytes(b'fixture: never loaded')
    return {'audio': str(audio), 'checkpoint': str(checkpoint),
            'output': str(tmp_path/'maidata.txt'), 'seed': 0}


def _training_payload(tmp_path):
    manifest = tmp_path/'manifest.json'
    manifest.write_text('{}', encoding='utf8')
    return {'manifest': str(manifest), 'checkpoint_dir': str(tmp_path/'output'),
            'cache_dir': str(tmp_path/'cache')}


@pytest.fixture
def parent(tmp_path):
    manifest = tmp_path/'manifest.json'
    manifest.write_text('{}', encoding='utf8')
    model = build_model(ComposerConfig(atom_vocab_size=3, d_model=8, nhead=2,
        layers=1, dim_feedforward=16, dropout=.25, history_events=8,
        candidate_count=16, action_slots=2, head_timing_dim=99,
        architecture_profile='music_context_v2'))
    path = tmp_path/'parent.pt'
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    save(path, model, optimizer, epoch=1, args={'dropout': .1, 'lr': 2e-4},
         manifest=manifest, atom_catalog=['1', '2'], optimizer_steps=7)
    return path


@pytest.mark.parametrize('guard', [True, False])
@pytest.mark.parametrize('bound', [0., 2., 4.])
def test_generation_api_forwards_guard_and_bound_without_starting_process(tmp_path, guard, bound):
    from maichart.composer.generate import parser
    payload = _generation_payload(tmp_path)
    payload.update(touch_history_guard=guard, touch_history_bound=bound)
    runtime, handler, tasks, calls, responses = _capture_runtime()
    composer_api.start_generate(handler, payload, runtime=runtime)
    assert len(tasks) == len(calls) == len(responses) == 1
    task, args, _ = calls[0]
    flag = '--touch-history-guard' if guard else '--no-touch-history-guard'
    opposite = '--no-touch-history-guard' if guard else '--touch-history-guard'
    assert args.count(flag) == 1 and opposite not in args
    assert float(args[args.index('--touch-history-bound')+1]) == bound
    parsed = parser().parse_args(args[2:])
    assert parsed.touch_history_guard is guard
    assert parsed.touch_history_bound == bound
    assert task.result['effective_settings']['touch_history_guard'] is guard
    assert task.result['effective_settings']['touch_history_bound'] == bound
    assert task.result['generation_contract'] == composer_api.GENERATION_CONTRACT


@pytest.mark.parametrize('key,value', [
    ('touch_history_guard', 'false'), ('touch_history_guard', 0),
    ('touch_history_guard', 1), ('touch_history_guard', None),
    ('touch_history_bound', -.001), ('touch_history_bound', 4.001),
    ('touch_history_bound', float('nan')), ('touch_history_bound', float('inf')),
    ('touch_history_bound', float('-inf')), ('touch_history_bound', 'invalid'),
    ('touch_history_bound', None),
])
def test_generation_controls_reject_invalid_values_before_creating_task(tmp_path, key, value):
    payload = _generation_payload(tmp_path)
    payload[key] = value
    runtime = SimpleNamespace(_create_task=lambda _: pytest.fail('invalid controls created a task'))
    with pytest.raises(ValueError, match=key):
        composer_api.start_generate(SimpleNamespace(_json_response=lambda _: None),
                                   payload, runtime=runtime)


def test_wrapper_applies_requested_bound_to_same_candidate_and_prefix():
    full = torch.tensor([2., 0., 1., 8., 7.])
    audio = torch.tensor([3., 2., 1., -1., -2.])
    mask = torch.tensor([False, False, False, True, True])
    calls = []

    def action_slot(context, slot, selected_ids, **kwargs):
        calls.append((context, slot, tuple(selected_ids)))
        return full if context == 'full' else audio

    model = SimpleNamespace(action_slot=action_slot)
    outputs = []
    selected = [2]
    audio_odds = torch.logsumexp(audio[3:], 0)-torch.logsumexp(audio[:3], 0)
    for bound in (0., 2.):
        diagnostics = {}
        corrected = action_logits_with_touch_guard(model, 'full', 1, selected,
            audio_context='audio', enabled=True, touch_mask=mask,
            log_odds_bound=bound, diagnostics=diagnostics)
        assert diagnostics['applied']
        assert calls[-2:] == [('full', 1, (2,)), ('audio', 1, (2,))]
        odds = torch.logsumexp(corrected[3:], 0)-torch.logsumexp(corrected[:3], 0)
        torch.testing.assert_close(odds, audio_odds+bound)
        assert corrected.softmax(0)[0] > full.softmax(0)[0]
        torch.testing.assert_close(corrected[3]-corrected[4], full[3]-full[4])
        outputs.append(corrected)
    assert outputs[0].softmax(0)[3:].sum() < outputs[1].softmax(0)[3:].sum()
    assert selected == [2]
    before = len(calls)
    assert action_logits_with_touch_guard(model, 'full', 1, selected,
        audio_context='audio', enabled=False, log_odds_bound=0.) is full
    assert len(calls) == before+1


@pytest.mark.parametrize('dropout', [0., .3, 1.])
def test_fresh_api_forwards_requested_dropout(tmp_path, dropout):
    payload = _training_payload(tmp_path)
    payload.update(mode='fresh', dropout=dropout)
    runtime, handler, _, calls, _ = _capture_runtime()
    composer_api.start_train(handler, payload, runtime=runtime)
    task, args, _ = calls[0]
    assert float(args[args.index('--dropout')+1]) == dropout
    assert task.result['effective_settings']['dropout'] == dropout


@pytest.mark.parametrize('dropout', [-.001, 1.001, float('nan'), float('inf')])
def test_fresh_api_rejects_invalid_dropout_before_creating_task(tmp_path, dropout):
    payload = _training_payload(tmp_path)
    payload.update(mode='fresh', dropout=dropout)
    runtime = SimpleNamespace(_create_task=lambda _: pytest.fail('invalid dropout created a task'))
    with pytest.raises(ValueError, match='dropout'):
        composer_api.start_train(SimpleNamespace(_json_response=lambda _: None),
                                payload, runtime=runtime)


@pytest.mark.parametrize('mode', ['resume', 'finetune'])
@pytest.mark.parametrize('explicit', [False, True])
def test_continuation_api_uses_model_dropout_instead_of_stale_saved_args(tmp_path, parent, mode, explicit):
    payload = _training_payload(tmp_path)
    payload.update(mode=mode, checkpoint=str(parent))
    if explicit:
        payload['dropout'] = .25
    runtime, handler, _, calls, _ = _capture_runtime()
    composer_api.start_train(handler, payload, runtime=runtime)
    task, args, _ = calls[0]
    assert float(args[args.index('--dropout')+1]) == .25
    assert task.result['effective_settings']['dropout'] == .25


@pytest.mark.parametrize('mode', ['resume', 'finetune'])
def test_continuation_api_rejects_attempt_to_replace_parent_dropout(tmp_path, parent, mode):
    payload = _training_payload(tmp_path)
    payload.update(mode=mode, checkpoint=str(parent), dropout=.1)
    runtime = SimpleNamespace(_create_task=lambda _: pytest.fail('changed inherited dropout created a task'))
    with pytest.raises(ValueError, match='dropout'):
        composer_api.start_train(SimpleNamespace(_json_response=lambda _: None),
                                payload, runtime=runtime)


def test_inspection_reports_optimizer_lr_instead_of_stale_requested_lr(parent):
    responses = []
    composer_api.inspect(SimpleNamespace(_json_response=responses.append), {'checkpoint': str(parent)})
    result = responses[0]
    assert result['valid'] and result['can_resume']
    assert result['optimizer_lr'] == 3e-5
    assert result['args']['lr'] == 2e-4
    assert result['model_config']['dropout'] == .25
    assert result['args']['dropout'] == .1


def test_inspection_does_not_invent_optimizer_lr_for_inference_weights(tmp_path, parent):
    inference = tmp_path/'inference.pt'
    export_inference(parent, inference)
    responses = []
    composer_api.inspect(SimpleNamespace(_json_response=responses.append), {'checkpoint': str(inference)})
    result = responses[0]
    assert result['valid'] and result['can_generate'] and not result['can_resume']
    assert 'optimizer_lr' not in result
