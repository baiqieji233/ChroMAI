"""Runtime packing preserves teacher supervision; run in the unified acceptance pass."""
import copy
import pytest
import torch
from maichart.composer.model import build_model
from maichart.composer.objectives import action_objective, LEGACY_ACTION_OBJECTIVE, JOINT_ACTION_OBJECTIVE
from test_composer_audio_grounded_v3 import config, inputs, ATOMS


@pytest.mark.parametrize("policy", [LEGACY_ACTION_OBJECTIVE, JOINT_ACTION_OBJECTIVE])
@pytest.mark.parametrize("cross", ["off", "relative_heads_v1"])
def test_packed_heads_objectives_and_parameter_gradients(policy, cross):
    torch.manual_seed(91)
    dense=build_model(config(history_dropout=0., ledger_dropout=0., symbolic_dropout=0.,
        cross_event_relation_policy=cross), ATOMS).train()
    packed=copy.deepcopy(dense)
    data=inputs()
    targets=torch.tensor([[0,0,0],[1,2,0]])
    weights=torch.tensor([.3,1.7])
    outputs=[]; gradients=[]; metrics=[]
    for model, enabled in ((dense,False),(packed,True)):
        result=model(**data,candidate_indices=torch.tensor([1,2]),action_targets=targets,packed_actions=enabled)
        indices=result.get('action_slot_indices')
        loss=0
        all_metrics={}
        for head in ('action_logits','audio_action_logits'):
            value, diagnostic=action_objective(result[head], targets, weights,policy=policy,slot_indices=indices)
            loss=loss+value
            all_metrics[head]=diagnostic
        loss.backward()
        outputs.append(result);metrics.append(all_metrics)
        gradients.append({k:p.grad for k,p in model.named_parameters() if p.grad is not None})
    indices=outputs[1]['action_slot_indices']
    assert indices.tolist()==[[0,0],[1,0],[1,1],[1,2]]
    for head in ('action_logits','audio_action_logits'):
        assert outputs[1][head].shape==(4,5)
        torch.testing.assert_close(outputs[1][head],outputs[0][head][indices[:,0],indices[:,1]])
        for key in metrics[0][head]:
            torch.testing.assert_close(metrics[0][head][key],metrics[1][head][key])
    assert gradients[0].keys()==gradients[1].keys()
    for key in gradients[0]:
        torch.testing.assert_close(gradients[0][key],gradients[1][key],atol=2e-6,rtol=2e-4)


@pytest.mark.parametrize('mask_probability',[0.,.5,.9])
def test_planner_sharing_preserves_effective_condition_and_rng(mask_probability):
    torch.manual_seed(19)
    dense=build_model(config(condition_mask_prob=mask_probability,history_dropout=0.,
        ledger_dropout=0.,symbolic_dropout=0.),ATOMS).train()
    shared=copy.deepcopy(dense)
    data=inputs()
    for key in ('condition','planner_audio','planner_mask'):
        data[key][1]=data[key][0]
    targets=torch.tensor([[1,0,0],[2,3,0]])
    outputs=[]; gradients=[]; rng=[]
    for model, enabled in ((dense,False),(shared,True)):
        torch.manual_seed(27)
        extra={'planner_unique_indices':torch.tensor([0]),'planner_inverse':torch.tensor([0,0])} if enabled else {}
        result=model(**data,candidate_indices=torch.tensor([1,2]),action_targets=targets,**extra)
        result['plan'].square().sum().backward()
        outputs.append(result['plan']);rng.append(torch.get_rng_state())
        gradients.append({k:p.grad for k,p in model.planner.named_parameters()})
    torch.testing.assert_close(outputs[0],outputs[1])
    assert torch.equal(rng[0],rng[1])
    for key in gradients[0]:
        torch.testing.assert_close(gradients[0][key],gradients[1][key],atol=2e-6,rtol=2e-4)


def test_planner_distinct_conditions_remain_distinct_groups():
    model=build_model(config(condition_mask_prob=0.),ATOMS).eval()
    data=inputs()
    data['planner_audio'][1]=data['planner_audio'][0]
    result=model(**data,decode_actions=False,planner_unique_indices=torch.tensor([0,1]),
        planner_inverse=torch.tensor([0,1]))
    expected=model.planner(data['planner_audio'],data['condition'],data['planner_mask'])
    torch.testing.assert_close(result['plan'],expected)


@pytest.mark.parametrize('policy',[LEGACY_ACTION_OBJECTIVE,JOINT_ACTION_OBJECTIVE])
def test_full_loss_keeps_audio_auxiliary_and_recovery_support(policy):
    from maichart.composer.train import _loss
    from maichart.composer.objectives import recovery_objective, RECOVERY_ACTION_OBJECTIVE
    targets=torch.tensor([[1,0,0],[2,3,0]])
    indices=torch.tensor([[0,0],[0,1],[1,0],[1,1],[1,2]])
    logits=torch.randn(2,3,5,requires_grad=True)
    audio=torch.randn(2,3,5,requires_grad=True)
    common={'candidate_logits':torch.randn(2,2),'future':torch.zeros(2,8),'eos_logits':torch.zeros(2)}
    batch={'action_target':targets,'candidate_target':torch.tensor([0,1]),'sample_weight':torch.tensor([.5,1.5]),
        'action_weight':torch.tensor([1.,.4]),'pointer_weight':torch.ones(2),
        'eos_target':torch.zeros(2),'future_target':torch.zeros(2,8)}
    dense={**common,'action_logits':logits,'audio_action_logits':audio}
    packed={**common,'action_logits':logits[indices[:,0],indices[:,1]],
        'audio_action_logits':audio[indices[:,0],indices[:,1]],'action_slot_indices':indices}
    left,lmetrics=_loss(dense,batch,5,action_objective_policy=policy)
    right,rmetrics=_loss(packed,batch,5,action_objective_policy=policy)
    torch.testing.assert_close(left,right)
    for key in lmetrics:torch.testing.assert_close(lmetrics[key],rmetrics[key])
    for key in ('action_logits','audio_action_logits'):
        assert torch.autograd.grad(right,dense[key],retain_graph=True)[0].abs().sum()>0
    left,_=recovery_objective(dense,batch,policy=RECOVERY_ACTION_OBJECTIVE)
    right,_=recovery_objective(packed,batch,policy=RECOVERY_ACTION_OBJECTIVE)
    torch.testing.assert_close(left,right)


@pytest.mark.parametrize('policy',[LEGACY_ACTION_OBJECTIVE,JOINT_ACTION_OBJECTIVE])
def test_packing_keeps_legacy_diagnostics_after_noncanonical_stop(policy):
    model=build_model(config(history_dropout=0.,ledger_dropout=0.,symbolic_dropout=0.),ATOMS).eval()
    data=inputs()
    targets=torch.tensor([[0,2,3],[1,0,2]])
    weights=torch.tensor([.7,1.3])
    dense=model(**data,candidate_indices=torch.tensor([1,2]),action_targets=targets)
    packed=model(**data,candidate_indices=torch.tensor([1,2]),action_targets=targets,packed_actions=True)
    indices=packed['action_slot_indices']
    assert indices.tolist()==[[0,0],[0,2],[1,0],[1,1]]
    for head in ('action_logits','audio_action_logits'):
        left,lmetrics=action_objective(dense[head],targets,weights,policy=policy)
        right,rmetrics=action_objective(packed[head],targets,weights,policy=policy,slot_indices=indices)
        torch.testing.assert_close(left,right)
        for key in lmetrics:torch.testing.assert_close(lmetrics[key],rmetrics[key])


def test_cpu_slot_mapping_matches_fallback_with_cross_event_head():
    model=build_model(config(history_dropout=0.,ledger_dropout=0.,symbolic_dropout=0.,
        cross_event_relation_policy='relative_heads_v1'),ATOMS).eval()
    data=inputs()
    targets=torch.tensor([[1,0,0],[2,3,0]])
    kwargs=dict(candidate_indices=torch.tensor([1,2]),action_targets=targets,packed_actions=True)
    fallback=model(**data,**kwargs)
    mapped=model(**data,**kwargs,packed_slot_indices=torch.tensor([[0,0],[0,1],[1,0],[1,1],[1,2]]),
        packed_first_indices=torch.tensor([0,2]))
    for key in ('action_slot_indices','action_logits','audio_action_logits'):
        torch.testing.assert_close(fallback[key],mapped[key])
