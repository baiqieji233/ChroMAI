import math
import torch
from maichart.composer.event_survival import event_survival,normalize_survival
from test_composer_echo_stage2_integration import _echo_fixture


def test_survival_probability_conservation_and_grid_refinement():
    for times in ([1.,2.,4.],[.5,1.,1.5,2.,3.,4.]):
        sec=torch.tensor([times]);rate=torch.full_like(sec,.4)
        first,tail,width=event_survival(rate,sec,torch.ones_like(sec,dtype=torch.bool))
        torch.testing.assert_close(first.exp().sum()+tail.exp().sum(),torch.tensor(1.))
        torch.testing.assert_close(tail,torch.tensor([-1.6]))
        torch.testing.assert_close(width.sum(),torch.tensor(4.))
    first,tail,width=event_survival(rate,sec,torch.ones_like(sec,dtype=torch.bool),torch.tensor([2.]))
    torch.testing.assert_close(width.sum(),torch.tensor(2.))
    torch.testing.assert_close(tail,torch.tensor([-.8]))


def test_lower_rate_increases_rest_probability_and_zero_padding_is_ignored():
    sec=torch.tensor([[.25,.5,1.,0.]])
    valid=torch.tensor([[True,True,True,False]])
    _,busy,_=event_survival(torch.ones_like(sec)*4,sec,valid)
    _,rest,_=event_survival(torch.ones_like(sec)*.01,sec,valid)
    assert rest.exp()>busy.exp()
    first,tail,_=event_survival(torch.ones_like(sec),sec,torch.zeros_like(valid))
    assert tail.item()==0 and first.exp().sum()==0


def test_new_model_joint_timing_all_slot_load_and_symbolic_isolation():
    from maichart.composer.checkpoint import migrate_audio_grounded
    base,data=_echo_fixture('song_paired_v3')
    model=migrate_audio_grounded(base,base.action_factors.atom_catalog,
        timing_distribution_policy='survival_v1',button_load_policy='per_button_v1').eval()
    # All timing features use the actual same candidate origin.
    data['candidate_features'][...,121]=data['candidate_features'][...,120]*.5
    result=model(**data)
    torch.testing.assert_close(result['candidate_logits'].exp().sum(-1)+result['no_event_log_prob'].exp(),torch.ones(2))
    assert result['button_load_features'].shape==(2,8,10)
    assert model(**data,decode_actions=False)['button_load_features'].shape==(2,6,8,10)
    with torch.no_grad():model.button_load.output.weight.normal_(std=.1)
    dense=model(**data);packed=model(**data,packed_actions=True)
    rows,slots=packed['action_slot_indices'].unbind(-1)
    torch.testing.assert_close(packed['action_logits'],dense['action_logits'][rows,slots],atol=2e-5,rtol=2e-5)
    for row in range(2):
        for slot in (0,1):
            out=model.action_slot(dense['action_context'][row],slot,data['action_targets'][row,:slot].tolist(),
                previous_heads=dense['previous_heads'][row],echo_heads=dense['echo_heads'][row],
                button_load_features=dense['button_load_features'][row])
            torch.testing.assert_close(out,dense['action_logits'][row,slot],atol=2e-5,rtol=2e-5)
    hidden=model(**data,symbolic_keep=torch.zeros(2,1))
    assert not hidden['button_load_features'].any()
    assert not model.button_load(hidden['button_load_features'],torch.randn(2,6,16)).any()
    (-result['next_event_log_probs'][:,0].mean()).backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.workload_head.parameters())
