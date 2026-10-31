from dataclasses import replace
import pytest
import torch
from maichart.composer.stream_model import StreamComposer
from test_composer_stream_model import fixture


def model_batch(policy='music_exit_v1'):
    original,batch=fixture(timing_output_policy='absolute_grid_v1',event_family_policy='audio_prior_transition_v3')
    # Include an actual repeated-template NOTE decision, rather than only
    # BOS/WAIT/literal-target positions whose keep groups are correctly empty.
    batch['history_kind'][0,0]=2
    batch['history_shape'][0,0]=batch['target_shape'][0,0]
    return StreamComposer(replace(original.config,transition_policy=policy,
        timing_grid_audio_policy='query_pool_v2'),original.codec).eval(),batch


@pytest.mark.parametrize('policy',['music_exit_v1','music_exit_v2','music_exit_v3'])
def test_new_model_joint_likelihood_has_finite_direct_transition_gradients(policy):
    model,batch=model_batch(policy)
    out=model(batch);loss,metrics=model.loss(out,batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(model.music_exit_head[2].weight.grad).all()
    assert model.music_exit_head[2].weight.grad.abs().sum()>0
    assert torch.isfinite(model.timing_grid.query_projection.weight.grad).all()
    assert all(torch.isfinite(v).all() for v in out.values() if torch.is_tensor(v) and v.is_floating_point())


@pytest.mark.parametrize('policy',['music_exit_v1','music_exit_v2','music_exit_v3'])
def test_time_decision_cannot_read_the_next_authored_time_or_action(policy):
    model,batch=model_batch(policy)
    before=model(batch)['delta_logits']
    altered=dict(batch,target_second=batch['target_second']+.125,
        target_shape=batch['target_shape'].roll(1,1))
    after=model(altered)['delta_logits']
    torch.testing.assert_close(before,after,rtol=0,atol=0)


def test_lookahead_music_and_run_age_reach_the_decision_without_target_labels():
    model,batch=model_batch()
    with torch.no_grad():model.music_exit_head[2].weight.normal_(0,.1)
    cache=model.encode_audio(batch['song_audio'],batch['song_mask'],batch['song_seconds'],batch['song_beats'])
    first=model.transition_logits_at(cache,batch['step_second'],batch['song_inverse'],batch)
    changed=dict(cache,delta_keys=cache['delta_keys'].flip(1))
    second=model.transition_logits_at(changed,batch['step_second'],batch['song_inverse'],batch)
    assert (first-second).abs().max()>1e-5
    longer=dict(batch,continuation_features=torch.ones(1,4,16)*4)
    third=model.transition_logits_at(cache,batch['step_second'],batch['song_inverse'],longer)
    assert (first-third).abs().max()>1e-5


def test_template_repartition_and_type_budget_use_the_same_family_marginal():
    model,batch=model_batch();out=model(batch)
    from maichart.composer.stream_timing import grouped_logsumexp
    expected=grouped_logsumexp(out['event_route_log_probs'],model.semantic_route_families-1,groups=31)
    torch.testing.assert_close(out['event_family_logits'],expected)
    loss,metrics=model.loss(out,batch)
    torch.testing.assert_close(metrics['event_family_loss'],.75*metrics['event_family_full_loss']+.25*metrics['event_family_audio_loss'])


def test_motion_support_respects_shape_and_complete_symmetric_orbits():
    model,_=model_batch()
    distribution=torch.zeros(1,3,8);distribution[0,0,[1,5]]=0.5
    keep=model.motion_keep_mask(torch.tensor([2]),distribution,torch.tensor([2]))
    assert keep.nonzero().tolist()==[[0,1],[0,5]]
    assert not model.motion_keep_mask(torch.tensor([3]),distribution,torch.tensor([2])).any()


def test_decision_sees_the_relative_direction_it_is_asked_to_continue():
    from maichart.composer.event_set_codec import EventSetCodec
    from maichart.composer.stream_model import StreamComposerConfig
    codec=EventSetCodec.fit(['1','2','8','1/5','1b/5'])
    model=StreamComposer(StreamComposerConfig(d_model=8,nhead=2,layers=1,dim_feedforward=16),codec)
    def symbols(texts):
        events=[codec.encode(text) for text in texts]
        width=max(len(e['atoms']) for e in events)
        atoms=torch.zeros(1,len(events),width,dtype=torch.long)
        for i,e in enumerate(events):atoms[0,i,:len(e['atoms'])]=torch.tensor(e['atoms'])
        return dict(history_kind=torch.full((1,len(events)),2),history_atoms=atoms,
            history_shape=torch.tensor([[e['shape'] for e in events]]),
            history_phase=torch.tensor([[e['phase'] for e in events]]),
            history_phase_mask=torch.tensor([[e['phase_mask'] for e in events]]))
    forward=model.transition_symbol_features(symbols(['1','2']))[0,1,:8]
    reverse=model.transition_symbol_features(symbols(['1','8']))[0,1,:8]
    assert forward.argmax()==1 and reverse.argmax()==7
    # Equal velocity with a different current position can require a different
    # exit. The classifier must see which physical pattern it is continuing.
    left=model.transition_symbol_features(symbols(['1','2']))[0,1]
    right=model.transition_symbol_features(symbols(['5','6']))[0,1]
    torch.testing.assert_close(left[:14],right[:14])
    assert not torch.equal(left[14:],right[14:])
    plain=model.transition_symbol_features(symbols(['1/5','1/5']))[0,1,:8]
    modified=model.transition_symbol_features(symbols(['1/5','1b/5']))[0,1,:8]
    torch.testing.assert_close(plain,modified)


@pytest.mark.parametrize('policy',['music_exit_v2','music_exit_v3'])
def test_zero_residual_preserves_parent_probabilities_and_known_tensor_weights(policy):
    from maichart.composer.stream_train import upgrade_learning_stage
    parent,batch=fixture(timing_output_policy='absolute_grid_v1',event_family_policy='audio_prior_transition_v3')
    repaired=upgrade_learning_stage(parent,dict(transition_policy=policy,timing_grid_audio_policy='query_pool_v2')).eval()
    before=parent(batch);after=repaired(batch)
    for key in ('delta_logits','phase_logits','shape_logits'):
        torch.testing.assert_close(before[key].softmax(-1),after[key].softmax(-1),atol=1e-6,rtol=1e-5)
    for key,value in parent.state_dict().items():torch.testing.assert_close(value,repaired.state_dict()[key],atol=0,rtol=0)


@pytest.mark.parametrize('policy',['music_exit_v2','music_exit_v3'])
def test_stationary_motion_has_a_learnable_exit_in_residual_policy(policy):
    model,_=model_batch(policy)
    distribution=torch.zeros(1,3,8);distribution[0,2,3]=1
    keep=model.motion_keep_mask(torch.tensor([1]),distribution,torch.tensor([1]))
    assert keep[0,3] and keep.sum()==model.shape_motion_stabilizer[1].sum()
    model.config=replace(model.config,transition_policy='music_exit_v1')
    assert not model.motion_keep_mask(torch.tensor([1]),distribution,torch.tensor([1])).any()


def test_absolute_classifier_cannot_be_reused_as_a_residual_on_stage_upgrade():
    from maichart.composer.stream_train import upgrade_learning_stage
    model,_=model_batch()
    with torch.no_grad():model.music_exit_head[2].weight.fill_(.7);model.music_exit_head[2].bias.fill_(.8)
    upgraded=upgrade_learning_stage(model,dict(transition_policy='music_exit_v2'))
    torch.testing.assert_close(model.music_exit_head[0].weight,upgraded.music_exit_head[0].weight)
    assert not upgraded.music_exit_head[2].weight.any() and not upgraded.music_exit_head[2].bias.any()


def _reference_batch():
    from maichart.composer.stream_model import BOS,NOTE,WAIT
    return dict(history_kind=torch.tensor([[BOS,NOTE,NOTE,WAIT,NOTE]]),
        step_mask=torch.ones(1,5,dtype=torch.bool),
        step_second=torch.tensor([[0.,1.,2.,4.,5.]],dtype=torch.float64))


def test_v3_cursor_uses_strict_prior_note_but_wait_uses_the_last_note():
    model,_=model_batch('music_exit_v3');batch=_reference_batch()
    previous=model.transition_reference_seconds(batch,batch['step_second'])
    torch.testing.assert_close(previous,torch.tensor([[0.,1.,1.,2.,2.]],dtype=torch.float64))
    assert (batch['step_second']-previous).tolist()==[[0.,0.,1.,2.,3.]]
    altered=dict(batch,target_second=batch['step_second']+100.,target_shape=torch.full((1,5),999))
    torch.testing.assert_close(model.transition_reference_seconds(altered,batch['step_second']),previous)


def test_v3_future_action_uses_latest_note_and_missing_reference_uses_current_music():
    model,_=model_batch('music_exit_v3');batch=_reference_batch();query=batch['step_second']+.5
    previous=model.transition_reference_seconds(batch,query)
    torch.testing.assert_close(previous,torch.tensor([[.5,1.,2.,2.,5.]],dtype=torch.float64))
    assert (query-previous).tolist()==[[0.,.5,.5,2.5,.5]]


@pytest.mark.parametrize('policy',['music_exit_v1','music_exit_v2'])
def test_saved_v1_v2_reference_contract_remains_unchanged(policy):
    model,_=model_batch(policy);batch=_reference_batch()
    previous=model.transition_reference_seconds(batch,batch['step_second'])
    torch.testing.assert_close(previous,torch.tensor([[0.,1.,2.,2.,5.]],dtype=torch.float64))


def test_v3_zero_correction_preserves_shared_base_gradients():
    from maichart.composer.stream_train import upgrade_learning_stage
    parent,batch=fixture(timing_output_policy='absolute_grid_v1',event_family_policy='audio_prior_transition_v3')
    repaired=upgrade_learning_stage(parent,dict(transition_policy='music_exit_v3',timing_grid_audio_policy='query_pool_v2')).eval()
    parent.loss(parent(batch),batch)[0].backward()
    repaired.loss(repaired(batch),batch)[0].backward()
    for name in ('delta_head.weight','shape_head.weight','phase_head.weight'):
        before=parent.get_parameter(name).grad;after=repaired.get_parameter(name).grad
        assert before is not None and after is not None
        torch.testing.assert_close(after,before,atol=3e-6,rtol=2e-5)
