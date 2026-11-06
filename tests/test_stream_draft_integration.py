"""Actual new-policy model/loss/state wiring, beyond isolated draft tests."""
from dataclasses import replace
import copy
import torch
from maichart.composer.stream_model import StreamComposer
from maichart.composer.stream_data import collate_stream
from test_stream_phrase_execution import execution_fixture


def build(monkeypatch):
    ds,old,song,bank=execution_fixture(monkeypatch)
    # Deployment completion needs feasible controls, unlike the old
    # execution-only fixture's impossible all-zero type locks.
    ds.conditions[0]=torch.tensor([13/15,.3,.4,.1,.05,.05,.2,.3,
        .7,.1,.1,.1,.3,.5,.2,.2])
    ds.draft_policy='beat_ordered_draft_v1';ds.unified_arrival=True
    config=replace(old.config,phrase_context_policy='shared_ordered_v4',phrase_execution_policy='component_modes_v2',
        detail_representation_policy='typed_components_v2',literal_output_policy='type_blocks_v1',
        draft_policy='beat_ordered_draft_v1',plan_kind_policy='unified_first_arrival_v5',
        song_profile_policy='masked_deployment_joint_v3',song_response_policy='unified_profile_response_v2',
        song_profile_components=4,song_response_modes=4,plan_weight=0.,phrase_loss_weight=0.,phrase_relation_weight=0.)
    return ds,StreamComposer(config,ds.codec),song,bank


def test_full_new_contract_has_finite_gradients_and_active_draft_interfaces(monkeypatch):
    ds,model,_,_=build(monkeypatch)
    batch=collate_stream([ds[1]])
    model.train();output=model(batch);loss,metrics=model.loss(output,batch);loss.backward()
    assert torch.isfinite(loss)
    assert metrics['plan_loss']==0 and metrics['phrase_loss']==0 and metrics['phrase_relation_loss']==0
    assert 'draft_joint_sequence_objective' in metrics
    for module in (model.draft_rate_projection,model.draft_action_projection,model.draft_detail_projection,model.phrase_context.draft):
        gradients=[p.grad for p in module.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert sum(float(g.abs().sum()) for g in gradients)>0


def test_new_timing_cannot_read_future_targets_or_native_landing_states(monkeypatch):
    ds,model,_,_=build(monkeypatch);model.eval()
    batch=collate_stream([ds[1]])
    with torch.no_grad():
        a=model(batch)
        changed={**batch,'candidate_action_state':torch.ones_like(batch['candidate_action_state']),
            'target_details':torch.zeros_like(batch['target_details']),
            'target_shape':torch.zeros_like(batch['target_shape'])}
        b=model(changed)
    torch.testing.assert_close(a['execution_kind_logits'],b['execution_kind_logits'],rtol=0,atol=0)
    torch.testing.assert_close(a['execution_delta_logits'],b['execution_delta_logits'],rtol=0,atol=0)


def test_recovery_training_keeps_the_executed_song_choice_and_controls(monkeypatch):
    ds,model,_,_=build(monkeypatch)
    batch=collate_stream([ds[1]])
    frozen=batch['condition'].clone()
    frozen[:,1:]=.2
    frozen[:,0]=.8
    known=torch.zeros_like(frozen,dtype=torch.bool);known[:,0]=True
    batch.update(committed_song_component=torch.tensor([2]),
        committed_execution_controls=frozen,committed_known_mask=known)
    model.train()
    for seed in (7,91):
        torch.manual_seed(seed)
        output=model(batch)
        assert output['selected_song_component'].tolist()==[2]
        torch.testing.assert_close(output['_causal_profile_condition'],frozen,rtol=0,atol=0)
        torch.testing.assert_close(output['unified_profile_evidence']['known_mask'],known)
        torch.testing.assert_close(output['unified_profile_evidence']['causal_condition'],frozen*known)
        loss,_=model.loss(output,batch)
        assert torch.isfinite(loss)
