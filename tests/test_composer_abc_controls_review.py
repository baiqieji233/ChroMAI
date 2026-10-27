"""Regression checks for the reviewed A/B/C control and masking contract."""
from types import SimpleNamespace

import pytest
import torch

from maichart.composer.checkpoint import load, migrate_audio_grounded, save
from maichart.composer.model import build_model, resolve_condition_mask_prob
from maichart.composer.train import parser
from maichart.webui import composer_api
from test_composer_audio_grounded_v3 import inputs
from test_composer_horizon_and_conserved_actions import ATOMS, make_config


def test_condition_mask_resolves_defaults_inheritance_and_new_stage_changes():
    assert resolve_condition_mask_prob() == 0.
    parent = {'condition_mask_prob': .35}
    assert resolve_condition_mask_prob(None, parent, resume=True) == .35
    assert resolve_condition_mask_prob(None, parent) == .35
    assert resolve_condition_mask_prob(.5, parent) == .5
    with pytest.raises(ValueError, match='Resume inherits'):
        resolve_condition_mask_prob(.5, parent, resume=True)
    options = parser().parse_args(['--manifest', 'unused.json', '--checkpoint-dir', 'unused',
                                   '--condition-mask-prob', '.35'])
    assert options.condition_mask_prob == .35


@pytest.mark.parametrize('invalid', [True, False, float('nan'), float('inf'), -.1, .91, 'invalid'])
def test_condition_mask_rejects_invalid_values(invalid):
    with pytest.raises(ValueError, match='condition_mask_prob'):
        resolve_condition_mask_prob(invalid)


def masked_model():
    return build_model(make_config(condition_mask_prob=.9, symbolic_dropout=0.,
        history_dropout=0., ledger_dropout=0.), ATOMS).train()


def test_condition_mask_hides_tail_from_all_internal_branches_but_keeps_level():
    torch.manual_seed(91)
    model = masked_model()
    batch = inputs()
    changed = {key: value.clone() if torch.is_tensor(value) else value for key, value in batch.items()}
    changed['condition'][:, 1:] = 1. - changed['condition'][:, 1:]
    seen = []
    hook = model.condition_projection.register_forward_pre_hook(
        lambda module, args: seen.append(args[0].detach().clone()))
    try:
        # Seed 0 masks both rows. All later dropout probabilities are zero.
        torch.manual_seed(0)
        original = model(**batch, decode_actions=False)
        torch.manual_seed(0)
        perturbed = model(**changed, decode_actions=False)
        assert torch.equal(seen[0], seen[1])
        assert bool((seen[0][:, 1:] == .5).all())
        torch.testing.assert_close(seen[0][:, 0], batch['condition'][:, 0], rtol=0, atol=0)
        for key in ('candidate_context', 'audio_action_context', 'candidate_logits',
                    'placement_logits', 'plan', 'workload', 'trailing_workload'):
            torch.testing.assert_close(original[key], perturbed[key], rtol=0, atol=0, msg=key)
        changed['condition'][:, 0] += .1
        torch.manual_seed(0)
        level_changed = model(**changed, decode_actions=False)
        assert not torch.allclose(original['candidate_context'], level_changed['candidate_context'])
    finally:
        hook.remove()


def test_condition_mask_is_training_only_and_explicit_plan_override_remains_evidence():
    model = masked_model().eval()
    batch = inputs()
    seen = []
    hook = model.condition_projection.register_forward_pre_hook(
        lambda module, args: seen.append(args[0].detach().clone()))
    try:
        with torch.no_grad():
            model(**batch, decode_actions=False)
        torch.testing.assert_close(seen[0], batch['condition'], rtol=0, atol=0)
    finally:
        hook.remove()
    model.train()
    override = torch.zeros(batch['planner_audio'].shape[:2] + (6,))
    torch.manual_seed(0)
    first = model(**batch, decode_actions=False, planner_override=override)
    torch.manual_seed(0)
    second = model(**batch, decode_actions=False, planner_override=override + 1.)
    assert not torch.allclose(first['candidate_context'], second['candidate_context'])


def test_condition_mask_finetune_and_checkpoint_roundtrip(tmp_path):
    base = build_model(make_config(), ATOMS)
    model = migrate_audio_grounded(base, ATOMS, condition_mask_prob=.35)
    assert model.config.condition_mask_prob == .35
    path = tmp_path / 'model.pt'
    save(path, model, None, epoch=1, args={'condition_mask_prob': .35},
         manifest=tmp_path / 'unused.json', atom_catalog=ATOMS)
    restored, payload = load(path)
    assert restored.config.condition_mask_prob == .35
    assert payload['args']['condition_mask_prob'] == .35


@pytest.fixture
def api_harness(tmp_path, monkeypatch):
    import maichart.composer.checkpoint as checkpoint
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{}')
    parent = tmp_path / 'parent.pt'
    parent.write_bytes(b'mock')
    inspection = {'can_resume': True, 'can_init': True,
                  'model_config': {'architecture_profile': 'audio_grounded_v3', 'condition_mask_prob': .35}, 'args': {}}
    monkeypatch.setattr(checkpoint, 'read_payload', lambda path: {})
    monkeypatch.setattr(checkpoint, 'inspect_payload', lambda payload: inspection)
    calls = []
    runtime = SimpleNamespace(_create_task=lambda kind: SimpleNamespace(task_id='mock', result={}),
        _run_subprocess_task=lambda task, args, **kwargs: calls.append((task, args)))
    handler = SimpleNamespace(_json_response=lambda payload: None)
    payload = {'manifest': str(manifest), 'checkpoint_dir': str(tmp_path / 'new'),
               'cache_dir': str(tmp_path / 'cache'), 'checkpoint': str(parent)}
    return payload, calls, handler, runtime


@pytest.mark.parametrize('mode,requested,expected', [
    ('fresh', None, 0.), ('fresh', .5, .5), ('resume', None, .35),
    ('finetune', None, .35), ('finetune', .1, .1)])
def test_condition_mask_api_reaches_cli_and_effective_settings(api_harness, mode, requested, expected):
    payload, calls, handler, runtime = api_harness
    changes = {} if requested is None else {'condition_mask_prob': requested}
    composer_api.start_train(handler, dict(payload, mode=mode, **changes), runtime=runtime)
    task, args = calls[-1]
    assert float(args[args.index('--condition-mask-prob') + 1]) == expected
    assert task.result['effective_settings']['condition_mask_prob'] == expected


def test_condition_mask_api_refuses_resume_changes_and_invalid_values(api_harness):
    payload, calls, handler, runtime = api_harness
    with pytest.raises(ValueError, match='Resume inherits'):
        composer_api.start_train(handler, dict(payload, mode='resume', condition_mask_prob=.5), runtime=runtime)
    for value in (True, -.1, .91, float('nan')):
        with pytest.raises(ValueError, match='condition_mask_prob'):
            composer_api.start_train(handler, dict(payload, mode='fresh', condition_mask_prob=value), runtime=runtime)
    assert calls == []
