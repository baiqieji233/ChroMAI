import pytest
import torch

from maichart.composer import ACTION_SAMPLING_POLICY, GROUNDED_TRAINING_SEMANTICS
from maichart.composer.checkpoint import inspect_payload, load, migrate_audio_grounded, save, validate
from maichart.composer.model import ComposerConfig, build_model


ATOMS = ['1', '2']
V10 = 'chart_composer_training_v10_touch_ring_recovery'


def _model(history_dropout=.3):
    model = build_model(ComposerConfig(atom_vocab_size=3, d_model=16, nhead=4,
        layers=1, dim_feedforward=32, dropout=0., history_events=8, action_slots=2,
        architecture_profile='audio_grounded_v3', head_timing_dim=99, lens_width=8,
        history_dropout=history_dropout, ledger_dropout=0., symbolic_dropout=0.), ATOMS)
    # These tests specify the legacy fully droppable history contract.
    # Fresh local-lead models are covered by test_composer_rhythm_review.py.
    model.local_lead_in = None
    return model


def _inputs():
    x = torch.randn(2, 3, 141)
    x[..., 120] = torch.tensor([.25, .5, 2.])
    return dict(candidate_features=x, history=torch.tensor([[[1, 0], [2, 0]], [[2, 0], [1, 0]]]),
        history_features=torch.randn(2, 2, 8), history_padding_mask=torch.zeros(2, 2, dtype=torch.bool),
        condition=torch.randn(2, 16), action_state=torch.randn(2, 50),
        head_timing=torch.rand(2, 3, 99), candidate_mask=torch.zeros(2, 3, dtype=torch.bool),
        workload_audio=torch.randn(2, 3, 104), interval_audio=torch.randn(2, 3, 16),
        audio_lens=torch.randn(2, 3, 31, 26), audio_lens_mask=torch.ones(2, 3, 31, dtype=torch.bool),
        planner_audio=torch.randn(2, 2, 104), planner_mask=torch.zeros(2, 2, dtype=torch.bool),
        planner_indices=torch.tensor([[[0, 1]] * 3] * 2), planner_mix=torch.full((2, 3), .5),
        symbolic_keep=torch.ones(2, 1), decode_actions=False)


def _mask_draw(monkeypatch, draw):
    # symbolic_keep is explicit and no ledger is supplied: this is only the
    # history event dropout draw, with other model randomness disabled.
    def sample(shape, **kwargs):
        assert tuple(shape) == (2, 2)
        return torch.tensor(draw, device=kwargs.get('device'))
    monkeypatch.setattr(torch, 'rand', sample)


def test_masked_tail_relation_uses_same_event_mask_as_history(monkeypatch):
    torch.manual_seed(17)
    model = _model().train()
    data = _inputs()
    captured = []
    handle = model.action_relation.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
    _mask_draw(monkeypatch, [[1., 0.], [0., 1.]])
    try:
        model(**data)
    finally:
        handle.remove()
    expected = model.action_factors.atom_head_lanes[data['history'][:, -1]].any(-2).float()
    expected[0] = 0.
    torch.testing.assert_close(captured[0], expected, rtol=0, atol=0)


@pytest.mark.parametrize('masked', [True, False])
def test_relation_gradient_is_zero_only_when_all_tail_events_are_masked(monkeypatch, masked):
    torch.manual_seed(18)
    model = _model().train()
    data = _inputs()
    _mask_draw(monkeypatch, [[1., 0. if masked else 1.]] * 2)
    result = model(**data)
    result['candidate_context'].square().sum().backward()
    gradient = model.action_relation.weight.grad
    assert gradient is not None
    assert bool(gradient.abs().sum() > 0) == (not masked)


@pytest.mark.parametrize('training,history_dropout', [(False, .3), (True, 0.)])
def test_eval_and_zero_dropout_preserve_unmasked_relation(training, history_dropout):
    torch.manual_seed(19)
    model = _model(history_dropout).train(training)
    data = _inputs()
    captured = []
    handle = model.action_relation.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
    try:
        model(**data)
    finally:
        handle.remove()
    expected = model.action_factors.atom_head_lanes[data['history'][:, -1]].any(-2).float()
    torch.testing.assert_close(captured[0], expected, rtol=0, atol=0)


def test_v10_load_and_explicit_finetune_remain_supported_but_resume_is_rejected(tmp_path, monkeypatch):
    model = _model()
    path = tmp_path / 'v11.pt'
    payload = save(path, model, torch.optim.AdamW(model.parameters()), epoch=1,
        args={}, manifest=tmp_path / 'manifest.json', atom_catalog=ATOMS)
    assert GROUNDED_TRAINING_SEMANTICS == 'chart_composer_training_v11_history_relation_mask'
    assert payload['training_semantics'] == GROUNDED_TRAINING_SEMANTICS
    assert inspect_payload(payload)['can_resume']
    payload['training_semantics'] = V10
    legacy_path = tmp_path / 'v10.pt'
    torch.save(payload, legacy_path)
    info = inspect_payload(payload)
    assert info['valid'] and info['can_generate'] and info['can_init']
    assert not info['can_resume']
    restored, legacy = load(legacy_path)
    assert legacy['training_semantics'] == V10
    upgraded = migrate_audio_grounded(restored, ATOMS, action_slots=2)
    for name, value in model.state_dict().items():
        if torch.is_tensor(value):
            torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
            torch.testing.assert_close(upgraded.state_dict()[name], value, rtol=0, atol=0)
        else:
            assert restored.state_dict()[name] == upgraded.state_dict()[name] == value
    from maichart.composer import train
    monkeypatch.setattr(train, 'ComposerDataset', lambda *args, **kwargs: pytest.fail('Resume reached data preparation'))
    with pytest.raises(ValueError, match='training contract changed'):
        train.main(['--resume', str(legacy_path), '--manifest', str(tmp_path / 'missing.json'),
                    '--checkpoint-dir', str(tmp_path / 'unused'), '--device', 'cpu'])


@pytest.mark.parametrize('semantics', [V10, GROUNDED_TRAINING_SEMANTICS])
@pytest.mark.parametrize('policy', [None, 'obsolete_policy'])
def test_v10_and_v11_both_require_touch_ring_sampling_policy(tmp_path, semantics, policy):
    payload = save(tmp_path / 'cp.pt', _model(), None, epoch=1, args={},
        manifest=tmp_path / 'manifest.json', atom_catalog=ATOMS)
    payload['training_semantics'] = semantics
    assert payload['action_sampling_policy'] == ACTION_SAMPLING_POLICY
    assert validate(payload)
    payload['action_sampling_policy'] = policy
    with pytest.raises(ValueError, match='action sampling policy mismatch'):
        validate(payload)
