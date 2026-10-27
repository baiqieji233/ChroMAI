import copy
import math
import pytest
import torch
from maichart.composer.learning_rate import learning_rate, resolve_plan, validate_plan
from maichart.composer.train import summarize_metrics
from maichart.composer.objectives import action_objective, JOINT_ACTION_OBJECTIVE


def test_old_checkpoint_resume_stays_constant_and_keeps_explicit_lr_override():
    for rate in (2e-4,3e-5):
        plan=resolve_plan(base_lr=rate,total_steps=800,resume=True)
        assert plan['kind']=='constant' and plan['warmup_steps']==0
        assert all(learning_rate(plan,i)==rate for i in (0,17,799,1200))
    with pytest.raises(ValueError,match='finetune'):
        resolve_plan(requested='cosine',base_lr=2e-4,total_steps=800,resume=True)


def test_cosine_warms_up_then_reaches_floor_and_does_not_restart_on_extension():
    plan=resolve_plan(requested='cosine',warmup_steps=4,min_lr_ratio=.1,base_lr=2e-4,total_steps=20)
    rates=[learning_rate(plan,i) for i in range(20)]
    assert rates[0]==pytest.approx(5e-5)
    assert rates[3]==pytest.approx(2e-4)
    assert rates[4]==pytest.approx(2e-4)
    assert rates[-1]==pytest.approx(2e-5)
    assert all(a>=b for a,b in zip(rates[4:],rates[5:]))
    assert learning_rate(plan,200)==pytest.approx(2e-5)
    restored=resolve_plan(base_lr=2e-4,total_steps=500,resume=True,saved=copy.deepcopy(plan))
    assert restored==plan
    assert [learning_rate(restored,i) for i in range(8,20)]==rates[8:]


@pytest.mark.parametrize('change',[dict(requested='constant'),dict(warmup_steps=0),dict(min_lr_ratio=.5),dict(base_lr=1e-5)])
def test_scheduled_resume_rejects_recipe_drift(change):
    saved=resolve_plan(requested='cosine',warmup_steps=3,base_lr=2e-4,total_steps=50)
    kwargs=dict(base_lr=2e-4,total_steps=60,resume=True,saved=saved)
    kwargs.update(change)
    with pytest.raises(ValueError,match='finetune'): resolve_plan(**kwargs)


@pytest.mark.parametrize('kwargs',[dict(warmup_steps=-1),dict(warmup_steps=20),dict(min_lr_ratio=-.1),dict(min_lr_ratio=1.1),dict(base_lr=float('nan'))])
def test_invalid_plan_is_rejected(kwargs):
    values=dict(base_lr=2e-4,total_steps=20);values.update(kwargs)
    with pytest.raises(ValueError): resolve_plan(**values)


def test_short_schedule_with_one_post_warmup_update_has_defined_endpoint():
    plan=resolve_plan(requested='cosine',warmup_steps=2,base_lr=1.,total_steps=3)
    assert learning_rate(plan,0)==.5
    assert learning_rate(plan,1)==1.
    assert learning_rate(plan,2)==pytest.approx(.1)


def test_automatic_warmup_does_not_consume_a_short_run():
    short=resolve_plan(requested='cosine',base_lr=1.,total_steps=40)
    assert short['warmup_steps']==2
    long=resolve_plan(requested='cosine',base_lr=1.,total_steps=74000)
    assert long['warmup_steps']==500


def test_conditional_continuation_ce_excludes_stop_and_pools_by_weight():
    logits=torch.tensor([[[0.,3.,-1.],[4.,0.,0.],[1.,0.,2.]],
                         [[0.,2.,1.],[0.,-1.,2.],[3.,0.,0.]]],requires_grad=True)
    target=torch.tensor([[1,0,0],[1,2,0]])
    _,metrics=action_objective(logits,target,torch.tensor([2.,.5]),policy=JOINT_ACTION_OBJECTIVE)
    expected=torch.nn.functional.cross_entropy(logits[1,1].unsqueeze(0),torch.tensor([2]))
    torch.testing.assert_close(metrics['later_nonstop_ce'],expected)
    assert metrics['later_nonstop_ce_weight']==.5
    assert metrics['first_nonstop_ce_weight']==2.5
    totals={'loss':torch.tensor(10.),'action_later_nonstop_ce':torch.tensor(6.),
            'action_later_nonstop_ce_sum':torch.tensor(12.),'action_later_nonstop_ce_weight':torch.tensor(3.)}
    summary=summarize_metrics(totals,2)
    assert summary['loss']==5.
    assert summary['action_later_nonstop_ce']==4.
    assert summary['action_later_nonstop_ce_weight']==3.
    assert 'action_later_nonstop_ce_sum' not in summary


def test_empty_continuation_mass_is_visible_and_has_zero_gradient():
    logits=torch.randn(2,2,4,requires_grad=True)
    _,metrics=action_objective(logits,torch.tensor([[1,0],[2,0]]),torch.ones(2),policy=JOINT_ACTION_OBJECTIVE)
    assert metrics['later_nonstop_ce']==0 and metrics['later_nonstop_ce_weight']==0
    gradient=torch.autograd.grad(metrics['later_nonstop_ce'],logits)[0]
    assert torch.count_nonzero(gradient)==0


def test_missing_scheduled_state_cannot_silently_resume_as_constant(tmp_path):
    from maichart.composer.checkpoint import save
    from maichart.composer.model import build_model
    from maichart.composer.train import main
    from test_composer_audio_grounded_v3 import ATOMS, config
    model=build_model(config(),ATOMS)
    path=tmp_path/'scheduled_without_state.pt'
    save(path,model,torch.optim.AdamW(model.parameters(),lr=2e-4),epoch=1,
        args={'lr':2e-4,'lr_schedule':'cosine','warmup_steps':4},manifest='unused',atom_catalog=ATOMS)
    with pytest.raises(ValueError,match='missing its learning-rate plan'):
        main(['--manifest','unused','--checkpoint-dir',str(tmp_path/'next'),'--resume',str(path),'--device','cpu'])
