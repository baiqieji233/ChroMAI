"""Causal input and actual first-step/choice gradients of the v14 contract."""
from collections import Counter
from dataclasses import replace
from fractions import Fraction

import torch

from maichart.composer.stream_data import collate_stream, StreamStep, WAIT, second_at_beat
from maichart.composer.stream_model import StreamComposer
from maichart.composer.stream_recovery import _configuration_recovery_row, _context
from maichart.composer.stream_song_profile import sampled_event_choice_credit
from test_stream_draft_integration import build
from test_stream_certified_phrase_recovery import make


def current_model(old,codec,**overrides):
    return StreamComposer(replace(old.config,plan_kind_policy='unified_first_arrival_v6',
        song_choice_objective_policy='event_credit_v1',recovery_policy='first_step_phrase_v6',
        draft_objective_policy='sequence_evidence_v2',dropout=0.,**overrides),codec)


def test_sampled_credit_expected_gradient_matches_executed_cost_without_actor_duplication():
    logits=torch.tensor([[.3,-.2]],requires_grad=True)
    probability=logits.softmax(-1).detach()
    cost=torch.tensor([1.,4.],requires_grad=True)
    expected=0.
    for choice in range(2):
        term,_=sampled_event_choice_credit(logits,torch.tensor([choice]),cost[choice:choice+1],
            torch.ones(1),torch.tensor([0]),1.,baseline=2.)
        assert term.item()==0.
        expected=expected+probability[0,choice]*term
    gradient=torch.autograd.grad(expected,logits)[0]
    reference=torch.autograd.grad((logits.softmax(-1)*cost.detach()).sum(),logits)[0]
    torch.testing.assert_close(gradient,reference)
    assert gradient[0,0]<0 and gradient[0,1]>0
    assert cost.grad is None
    term,metrics=sampled_event_choice_credit(logits,torch.tensor([1]),cost[1:],torch.ones(1),
        torch.tensor([0]),1.,committed=torch.tensor([1]))
    assert metrics['song_choice_credit_charts']==0
    assert torch.autograd.grad(term,logits)[0].abs().sum()==0


def test_actual_timing_reaches_shared_decoder_but_not_future_targets(monkeypatch):
    ds,old,_,_=build(monkeypatch);model=current_model(old,ds.codec).eval()
    batch=collate_stream([ds[1]])
    output=model(batch)
    target=batch['target_kind']
    time_loss=torch.nn.functional.cross_entropy(output['kind_logits'].flatten(0,1),target.flatten())
    gradient=torch.autograd.grad(time_loss,model.decoder.layers[0].linear1.weight)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum()>0
    changed=model({**batch,'candidate_action_state':torch.ones_like(batch['candidate_action_state']),
        'target_details':torch.zeros_like(batch['target_details']),'target_shape':torch.zeros_like(batch['target_shape'])})
    torch.testing.assert_close(output['execution_kind_logits'],changed['execution_kind_logits'],rtol=0,atol=0)
    torch.testing.assert_close(output['execution_delta_logits'],changed['execution_delta_logits'],rtol=0,atol=0)
    assert not hasattr(model.phrase_execution,'timing')


def test_first_wrong_prefix_step_has_type_gradient_without_spatial_labels(monkeypatch):
    ds,old,source,prefix,context,guard=make(monkeypatch)
    model=current_model(old,ds.codec,configuration_recovery_steps=1).eval()
    model.config.recovery_bridge_steps=0  # isolate the actual first decision
    row=_configuration_recovery_row(ds,model,0,prefix,source[2],guard,context,0,2,
        False,'time_only',Counter())
    batch=collate_stream([row]);output=model(batch)
    assert batch['configuration_family_mask'].any() and not batch['configuration_mask'][batch['loss_mask']].any()
    loss,metrics=model.loss(output,batch)
    route_grad=torch.autograd.grad(loss,output['execution_route_log_probs'],retain_graph=True)[0]
    assert route_grad[batch['configuration_family_mask']].abs().sum()>0
    phase_grad=torch.autograd.grad(loss,output['execution_phase_logits'],allow_unused=True,retain_graph=True)[0]
    assert phase_grad is None or phase_grad.abs().sum()==0
    assert batch['_literal_indices'].numel()==0 and not batch['_detail_groups']
    assert metrics['first_step_family_targets']==1


def test_actual_song_choice_receives_event_credit_with_profile_loss_disabled(monkeypatch):
    ds,old,_,_=build(monkeypatch)
    model=current_model(old,ds.codec,song_profile_weight=0.,deployment_condition_fraction=0.).train()
    batch=collate_stream([ds[1]])
    output=model(batch);prior=output['unified_profile_evidence']['prior_logits']
    loss,metrics=model.loss(output,batch)
    gradient=torch.autograd.grad(loss,prior)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum()>0
    assert metrics['song_choice_credit_charts']==1


def test_actual_timing_ignores_wait_subdivision_at_same_cursor(monkeypatch):
    ds,old,_,bank=build(monkeypatch);model=current_model(old,ds.codec).eval()
    prefix=list(ds.streams[0][:2]);start=prefix[-1].end;end=start+4
    def run(offsets):
        steps=list(prefix);previous=start
        for offset in offsets:
            following=start+offset
            steps.append(StreamStep(WAIT,previous,following,
                second_at_beat(ds.clocks[0],previous),second_at_beat(ds.clocks[0],following)))
            previous=following
        context=_context(ds,0,bank)
        with torch.no_grad():return model(context.batch(steps,end))
    a=run([Fraction(2),Fraction(4)])
    b=run([Fraction(1),Fraction(2),Fraction(4)])
    torch.testing.assert_close(a['execution_kind_logits'][:,-1],b['execution_kind_logits'][:,-1],rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(a['execution_delta_logits'][:,-1],b['execution_delta_logits'][:,-1],rtol=1e-5,atol=1e-6)

