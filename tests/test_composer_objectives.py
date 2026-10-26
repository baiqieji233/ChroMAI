import copy

import pytest
import torch
from torch.nn import functional as F

from maichart.composer.checkpoint import inspect_payload, load, save, validate
from maichart.composer.model import build_model
from maichart.composer.objectives import (JOINT_ACTION_OBJECTIVE, JOINT_TRAINING_SEMANTICS,
    RECOVERY_ACTION_OBJECTIVE, RECOVERY_TRAINING_SEMANTICS,
    LEGACY_ACTION_OBJECTIVE, action_objective, resolve_action_objective, recovery_objective)
from maichart.composer.train import _loss
from test_composer_audio_grounded_v3 import ATOMS, config, inputs


def test_legacy_objective_preserves_loss_and_gradients_exactly():
    torch.manual_seed(23)
    logits = torch.randn(3, 4, 5, requires_grad=True)
    targets = torch.tensor([[1, 0, 0, 0], [2, 3, 0, 0], [1, 2, 4, 0]])
    sample_weights = torch.tensor([1., .7, 1.5])
    per_slot = F.cross_entropy(logits.reshape(-1, 5), targets.reshape(-1), reduction='none').reshape(3, -1)
    active = torch.ones_like(targets, dtype=torch.bool)
    active[:, 1:] = targets[:, :-1].ne(0)
    weights = torch.where(targets.ne(0), 1., .35) * active.to(per_slot)
    expected = (((per_slot * weights).sum(-1) / weights.sum(-1).clamp_min(1e-6)) * sample_weights).sum() / sample_weights.sum().clamp_min(1e-6)
    actual, metrics = action_objective(logits, targets, sample_weights)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    left = torch.autograd.grad(actual, logits, retain_graph=True)[0]
    right = torch.autograd.grad(expected, logits)[0]
    torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert metrics['first_slot_ce'] > 0


def test_joint_nll_is_sequence_probability_and_stops_at_first_stop():
    probabilities = torch.tensor([[[.1, .8, .1], [.6, .2, .2], [.2, .3, .5], [.1, .1, .8]]])
    # Values after the first STOP are not part of this sequence.
    targets = torch.tensor([[1, 0, 2, 2]])
    actual, metrics = action_objective(probabilities.log(), targets, torch.ones(1), policy=JOINT_ACTION_OBJECTIVE)
    torch.testing.assert_close(actual, -torch.log(torch.tensor(.8 * .6)))
    torch.testing.assert_close(metrics['stop_ce'], -torch.log(torch.tensor(.6)))


def test_actual_training_loss_does_not_shift_a_true_half_stop_probability_under_joint_objective():
    gradients = {}
    for policy in (LEGACY_ACTION_OBJECTIVE, JOINT_ACTION_OBJECTIVE):
        stop = torch.tensor(0., requires_grad=True)
        shared = torch.stack([stop, stop.new_tensor(-30.), stop.new_tensor(0.)])
        row = torch.stack([stop.new_tensor([-30., 30., -30.]), shared, stop.new_tensor([30., -30., -30.])])
        output = {'action_logits': torch.stack([row, row]), 'candidate_logits': torch.zeros(2, 1),
                  'future': torch.zeros(2, 8), 'eos_logits': torch.zeros(2)}
        batch = {'action_target': torch.tensor([[1, 0, 0], [1, 2, 0]]),
                 'candidate_target': torch.zeros(2, dtype=torch.long), 'sample_weight': torch.ones(2),
                 'future_target': torch.zeros(2, 8), 'eos_target': torch.zeros(2)}
        loss, metrics = _loss(output, batch, 3, action_objective_policy=policy)
        gradients[policy] = float(torch.autograd.grad(loss, stop)[0])
        assert metrics['action_first_slot_ce'] < 1e-6
    assert gradients[LEGACY_ACTION_OBJECTIVE] > .04
    assert abs(gradients[JOINT_ACTION_OBJECTIVE]) < 1e-7


def test_old_checkpoint_keeps_legacy_and_objective_changes_require_finetune():
    assert resolve_action_objective('auto', {}, resume=True) == LEGACY_ACTION_OBJECTIVE
    assert resolve_action_objective('auto', {'action_objective_policy': JOINT_ACTION_OBJECTIVE}, resume=True) == JOINT_ACTION_OBJECTIVE
    with pytest.raises(ValueError, match='finetune'):
        resolve_action_objective(JOINT_ACTION_OBJECTIVE, {}, resume=True)
    with pytest.raises(ValueError, match='audio_grounded_v3'):
        resolve_action_objective(JOINT_ACTION_OBJECTIVE, architecture_profile='music_context_v2')


@pytest.mark.parametrize('policy,semantics', [(JOINT_ACTION_OBJECTIVE,JOINT_TRAINING_SEMANTICS),
    (RECOVERY_ACTION_OBJECTIVE,RECOVERY_TRAINING_SEMANTICS)])
def test_joint_checkpoint_contract_roundtrip_without_changing_generation_weights(tmp_path,policy,semantics):
    torch.manual_seed(8)
    model = build_model(config(), ATOMS).eval()
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / 'joint.pt'
    payload = save(path, model, optimizer, epoch=1,
                   args={'action_objective_policy': policy}, manifest='unused', atom_catalog=ATOMS)
    assert payload['training_semantics'] == semantics
    assert inspect_payload(payload)['can_resume'] is True
    restored, _ = load(path)
    restored.eval()
    data = inputs()
    with torch.no_grad():
        before, after = model(**data), restored(**data)
    for key in before:
        torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
    invalid = copy.copy(payload)
    invalid['args'] = {}
    with pytest.raises(ValueError, match='objective/training semantics'):
        validate(invalid)
    legacy_path = tmp_path / 'legacy.pt'
    legacy = save(legacy_path, model, optimizer, epoch=1, args={}, manifest='unused', atom_catalog=ATOMS)
    assert inspect_payload(legacy)['can_resume'] is True
    legacy['args'] = {'action_objective_policy': JOINT_ACTION_OBJECTIVE}
    with pytest.raises(ValueError, match='objective/training semantics'):
        validate(legacy)


def test_configuration_recovery_backpropagates_real_action_labels_and_preserves_legacy_pointer_only():
    batch = {'candidate_target': torch.tensor([0,1]), 'sample_weight': torch.tensor([1.,.7]),
             'pointer_weight': torch.ones(2), 'action_weight': torch.ones(2),
             'action_target': torch.tensor([[1,0],[2,0]])}
    torch.manual_seed(17)
    actions = torch.randn(2,2,3,requires_grad=True)
    pointer = torch.randn(2,3,requires_grad=True)
    out = {'candidate_logits': pointer, 'action_logits': actions}
    legacy, _ = recovery_objective(out,batch,policy=LEGACY_ACTION_OBJECTIVE)
    assert torch.autograd.grad(legacy,actions,allow_unused=True,retain_graph=True)[0] is None
    actual, metrics = recovery_objective(out,batch,policy=RECOVERY_ACTION_OBJECTIVE)
    gradients = torch.autograd.grad(actual,actions)[0]
    assert gradients[0,0,1] < 0 and gradients[1,0,2] < 0
    assert gradients[:,1,0].lt(0).all()  # Proper STOP after the authored actions.
    assert metrics['action_loss'] > 0
    torch.testing.assert_close(actual, .1*(metrics['pointer_loss']+metrics['action_loss']))


def test_joint_without_configuration_recovery_has_distinct_resume_contract():
    with pytest.raises(ValueError,match='finetune'):
        resolve_action_objective(RECOVERY_ACTION_OBJECTIVE,
            {'action_objective_policy':JOINT_ACTION_OBJECTIVE},resume=True)


def test_recovery_skips_unused_audio_classifier_without_changing_main_loss_or_gradients():
    torch.manual_seed(41)
    model=build_model(config(),ATOMS).eval()
    batch=inputs()
    outputs=[];gradients=[]
    for auxiliary in (True,False):
        model.zero_grad(set_to_none=True)
        out=model(**batch,audio_auxiliary=auxiliary)
        (out['action_logits'].square().mean()+out['candidate_logits'].square().mean()).backward()
        outputs.append(out)
        gradients.append({name:param.grad.clone() for name,param in model.named_parameters() if param.grad is not None})
    assert 'audio_action_logits' in outputs[0] and 'audio_action_logits' not in outputs[1]
    for key in ('candidate_logits','action_logits'):
        torch.testing.assert_close(outputs[0][key],outputs[1][key],rtol=0,atol=0)
    assert gradients[0].keys()==gradients[1].keys()
    for name in gradients[0]:
        torch.testing.assert_close(gradients[0][name],gradients[1][name],rtol=0,atol=0)
