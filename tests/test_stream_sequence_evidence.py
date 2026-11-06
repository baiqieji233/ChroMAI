"""A whole authored configuration supplies joint evidence for one mode."""
import torch

from maichart.composer.stream_phrase_execution import draft_joint_sequence_loss


def batch(length=8):
    return dict(target_kind=torch.full((1,length),2),draft_departure_group=torch.zeros(1,length,dtype=torch.long),
        draft_landing_group=torch.zeros(1,length,dtype=torch.long),draft_loss_group_count=1,
        draft_loss_group_rows=torch.tensor([0]),draft_loss_group_positions=torch.tensor([0]),
        draft_loss_group_side=torch.tensor([0]),draft_loss_group_decisions=torch.tensor([length]),
        draft_loss_group_complete=torch.tensor([True]))


def test_persistent_choice_uses_full_phrase_evidence_not_per_event_average():
    b=batch();time=torch.zeros(1,8,2)
    mark=torch.tensor([.1,1.]).expand(1,8,2).clone().requires_grad_()
    prior=torch.zeros_like(time,requires_grad=True);w=torch.ones(1,8)
    loss,m=draft_joint_sequence_loss(time,mark,prior,prior,b,w,w,8.,objective_policy='sequence_evidence_v2')
    expected=-torch.logsumexp(torch.tensor([-.8,-8.])-torch.log(torch.tensor(2.)),0)/8.
    torch.testing.assert_close(loss,expected)
    loss.backward()
    assert mark.grad[...,0].sum()>500*mark.grad[...,1].sum()
    assert m['draft_phrase_mode_posterior_entropy']<.01


def test_sampling_importance_and_duplicate_seams_cannot_change_mode_evidence():
    b=batch();time=torch.rand(1,8,2);mark=torch.rand(1,8,2);p=torch.randn(1,8,2);w=torch.ones(1,8)
    reference,_=draft_joint_sequence_loss(time,mark,p,p,b,w,w,8.,objective_policy='sequence_evidence_v2')
    scaled,_=draft_joint_sequence_loss(time,mark,p,p,b,3*w,3*w,24.,objective_policy='sequence_evidence_v2')
    torch.testing.assert_close(reference,scaled)
    duplicate={**b,'target_kind':b['target_kind'].expand(2,-1),
        'draft_departure_group':b['draft_departure_group'].expand(2,-1),
        'draft_landing_group':b['draft_landing_group'].expand(2,-1)}
    repeated,_=draft_joint_sequence_loss(time.expand(2,-1,-1),mark.expand(2,-1,-1),p.expand(2,-1,-1),p.expand(2,-1,-1),
        duplicate,w.expand(2,-1)*.5,w.expand(2,-1)*.5,8.,objective_policy='sequence_evidence_v2')
    torch.testing.assert_close(reference,repeated)


def test_one_mode_and_committed_mode_preserve_original_data_budget():
    b=batch();w=torch.ones(1,8);time=torch.rand(1,8,1);mark=torch.rand_like(time);prior=torch.zeros_like(time)
    loss,_=draft_joint_sequence_loss(time,mark,prior,prior,b,w,w,8.,objective_policy='sequence_evidence_v2')
    torch.testing.assert_close(loss,(time.sum()+mark.sum())/8.,rtol=0,atol=0)
    b['draft_loss_group_mode']=torch.tensor([1])
    x=time.expand(-1,-1,2);y=mark.expand(-1,-1,2);p=torch.randn_like(x,requires_grad=True)
    loss,_=draft_joint_sequence_loss(x,y,p,p,b,w,w,8.,objective_policy='sequence_evidence_v2')
    loss.backward();assert p.grad.abs().sum()==0


def test_legacy_v12_checkpoint_does_not_silently_adopt_v13_objective():
    from maichart.composer.stream_checkpoint import training_objective_compatible
    old=dict(model_config=dict(draft_policy='beat_ordered_draft_v1',plan_kind_policy='unified_first_arrival_v5'),
        args=dict(training_objective_recipe=dict(version='stream_ordered_draft_first_arrival_v12')))
    assert training_objective_compatible(old)
    new={**old,'model_config':{**old['model_config'],'draft_objective_policy':'sequence_evidence_v2'}}
    assert not training_objective_compatible(new)
