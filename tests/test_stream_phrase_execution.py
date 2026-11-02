"""Final coordinated regression tests for whole-phrase learning contracts."""
import copy
import math
from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace

import pytest
import torch

from maichart.composer.stream_phrase_execution import whole_phrase_configuration_loss
from maichart.composer.stream_data import StreamDataset,StreamInferenceContext,collate_stream
from maichart.composer.stream_model import StreamComposer
from maichart.composer.stream_proposal import prepare_selected_candidate
from maichart.composer.event_set_codec import EventSetCodec
from maichart.event_timing import EventPoint
from test_stream_landing_integration import fixture as landing_fixture


def grouped_batch(length):
    return dict(training_phrase_group=torch.zeros(1,length,dtype=torch.long),
        training_phrase_position=torch.arange(length)[None],
        training_phrase_complete=torch.ones(1,length,dtype=torch.bool),
        training_phrase_group_weight=torch.tensor([float(length)]))


def test_one_mode_must_explain_whole_phrase_not_hop_per_event():
    scores=torch.tensor([[[.1,4.],[4.,.1]]],requires_grad=True)
    prior=torch.zeros_like(scores,requires_grad=True)
    weights=torch.ones(1,2)
    whole,metrics=whole_phrase_configuration_loss(scores,prior,grouped_batch(2),weights,2.)
    local,_=whole_phrase_configuration_loss(scores,prior,{},weights,2.)
    torch.testing.assert_close(whole,torch.tensor(2.05))
    assert whole>local and metrics['phrase_complete_groups']==1
    whole.backward()
    assert torch.isfinite(scores.grad).all() and torch.isfinite(prior.grad).all()
    assert prior.grad[:,1].abs().sum()==0


def test_wait_at_phrase_start_owns_prior_and_invalid_group_falls_back():
    scores=torch.tensor([[[0.,0.],[.2,2.],[.3,3.]]])
    prior=torch.tensor([[[math.log(.8),math.log(.2)],[-8.,8.],[-8.,8.]]])
    weights=torch.tensor([[1.,.5,1.]])
    batch=grouped_batch(3)
    actual,_=whole_phrase_configuration_loss(scores,prior,batch,weights,2.5)
    expected=-torch.logsumexp(prior[0,0]-torch.tensor([.4,4.]),0)/2.5
    torch.testing.assert_close(actual,expected)
    batch['training_phrase_complete'].zero_()
    fallback,_=whole_phrase_configuration_loss(scores,prior,batch,weights,2.5)
    local,_=whole_phrase_configuration_loss(scores,prior,{},weights,2.5)
    torch.testing.assert_close(fallback,local)


def test_all_recovery_replacement_rows_have_no_complete_groups():
    scores=torch.rand(2,3,4,requires_grad=True);prior=torch.randn(2,3,4,requires_grad=True)
    weights=torch.ones(2,3)
    batch=dict(training_phrase_group=torch.full((2,3),-1),training_phrase_complete=torch.zeros(2,3,dtype=torch.bool),
        training_phrase_position=torch.full((2,3),-1),training_phrase_group_weight=torch.empty(0))
    loss,_=whole_phrase_configuration_loss(scores,prior,batch,weights,6.)
    expected,_=whole_phrase_configuration_loss(scores,prior,{},weights,6.)
    torch.testing.assert_close(loss,expected)
    loss.backward();assert torch.isfinite(scores.grad).all()


def execution_fixture(monkeypatch):
    old,baseline,song,bank=landing_fixture(monkeypatch)
    texts=['1','2/6','3h[4:1]','4','5-1[8:1]','6','7/3','8','1h[8:1]','2','3/7','4']*2
    points=[EventPoint(Fraction(i),text) for i,text in enumerate(texts)]
    codec=EventSetCodec.fit(texts)
    base=SimpleNamespace(songs=[(song,points,13.,1.,None,torch.zeros(16),(),None)])
    ds=StreamDataset(base=base,codec=codec,lead_in=8,targets=4,audio_grid_seconds=.05,
        history_window_policy='block_aligned_v2',continuation_policy='pattern_runs_v2',
        phrase_context_policy='aligned_phrase_v3',candidate_state_policy='selected_native_v1',
        training_window_policy='phrase_complete_v1',max_phrase_events=64,phrase_seam_events=2,
        relation_policy='future_sets_v1')
    config=replace(baseline.config,event_family_policy='shared_continuation_v1',transition_policy='off',
        gap_continuation_policy='off',training_window_policy='phrase_complete_v1',
        phrase_execution_policy='persistent_modes_v1',detail_representation_policy='shared_components_v1',
        duration_audio_policy='interval_v1',duration_feature_policy='log_time_v2')
    model=StreamComposer(config,codec).eval()
    return ds,model,song,bank


def test_full_phrase_loss_backpropagates_execution_shared_details_and_relations(monkeypatch):
    ds,model,_,_=execution_fixture(monkeypatch)
    batch=collate_stream([ds[1]])
    output=model(batch)
    loss,metrics=model.loss(output,batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    for module in (model.phrase_execution.route,model.phrase_execution.relations[-1],model.shared_details):
        assert sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)>0
    torch.testing.assert_close(metrics['configuration_loss'],metrics['shape_loss']+metrics['phase_loss']+
        metrics['detail_loss']+metrics['phrase_configuration_adjustment'])
    assert metrics['phrase_complete_groups']>0


def test_future_relation_prediction_cannot_read_target_clock_or_chart(monkeypatch):
    ds,model,_,_=execution_fixture(monkeypatch)
    batch=collate_stream([ds[1]])
    changed=copy.deepcopy(batch)
    changed['target_second']+=.17
    changed['candidate_action_state']=torch.ones_like(changed['candidate_action_state'])
    changed['candidate_action_state_mask']=torch.zeros_like(changed['candidate_action_state_mask'])
    changed['target_relation'].normal_()
    changed['target_phrase_sequence_atoms'].zero_()
    with torch.no_grad():
        left=model(batch);right=model(changed)
    torch.testing.assert_close(left['future_relation_prediction'],right['future_relation_prediction'],rtol=0,atol=0)
    torch.testing.assert_close(left['kind_logits'],right['kind_logits'],rtol=0,atol=0)


def test_teacher_selected_mode_logits_equal_generation_selected_landing(monkeypatch):
    ds,model,song,bank=execution_fixture(monkeypatch)
    item=ds[1];batch=collate_stream([item])
    # The primary phrase uses exactly the same causal history as inference.
    end=ds.samples[1][2]
    position=end-1-int(item['step_index'][0])
    context=StreamInferenceContext(song,ds.codec,ds.delta_vocab,ds.conditions[0],lead_in=8,
        audio_bank=bank,audio_grid_seconds=.05,history_window_policy='block_aligned_v2',rollout_steps=4,
        continuation_policy='pattern_runs_v2',timing_output_policy=ds.timing_output_policy,
        phrase_context_policy='aligned_phrase_v3',candidate_state_policy='selected_native_v1',
        training_window_policy='phrase_complete_v1',max_phrase_events=64)
    for step in ds.streams[0][:end-1]:context.commit(step)
    raw=context.current_batch();target=ds.streams[0][end-1]
    with torch.no_grad():
        teacher=model(batch)
        cache=model.encode_audio(raw['song_audio'],raw['song_mask'],raw['song_seconds'],raw['song_beats'])
        output=model(raw,audio_cache=cache)
        selected,prepared=prepare_selected_candidate(model,context,raw,output,target.end,cache)
        actual,prepared=model.candidate_action_state(prepared,selected,cache,
            torch.full_like(selected['step_second'],target.end_second))
        proposal=model.execution_proposal(actual[0,-1],prepared,selected,2)
    # Training keeps finite dummy logits on unsupported families for masked
    # losses; generation keeps their exact -inf support. Probabilities agree.
    actual_routes=proposal['execution_route_log_probs']
    expected_routes=teacher['execution_route_log_probs'][0,position,2]
    supported=torch.isfinite(actual_routes)
    torch.testing.assert_close(actual_routes[supported],expected_routes[supported],rtol=1e-5,atol=1e-5)
    torch.testing.assert_close(actual_routes.softmax(-1),expected_routes.softmax(-1),rtol=1e-5,atol=1e-6)
    base=model.event_route_log_probs(actual,family_logits=model.family_logits_for_action(actual))
    executed=model.execution_outputs(actual,prepared,base)
    torch.testing.assert_close(proposal['execution_route_log_probs'],executed['routes'][0,-1,2],rtol=1e-5,atol=1e-5)
    assert teacher['execution_route_log_probs'].shape[-2]==4


def test_mode_contract_rejects_obsolete_independent_detail_and_keep_override(monkeypatch):
    _,model,_,_=execution_fixture(monkeypatch)
    with pytest.raises(ValueError,match='joint phase/detail'):replace(model.config,detail_policy='independent_v1')
    with pytest.raises(ValueError,match='keep/exit'):replace(model.config,transition_policy='music_exit_v3')
