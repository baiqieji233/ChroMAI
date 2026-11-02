from dataclasses import replace
import torch
from maichart.composer.stream_data import collate_stream
from maichart.composer.stream_model import StreamComposer
from maichart.composer.stream_train import upgrade_learning_stage
from test_stream_phrase_execution import execution_fixture


def upgraded_fixture(monkeypatch):
    ds,old,_,_=execution_fixture(monkeypatch)
    ds.phrase_context_policy='shared_ordered_v4'
    cfg=replace(old.config,phrase_context_policy='shared_ordered_v4',phrase_execution_policy='component_modes_v2',
        detail_representation_policy='typed_components_v2',song_profile_policy='masked_deployment_joint_v3',
        song_response_policy='unified_profile_response_v2',song_profile_components=4,song_response_modes=4)
    model=StreamComposer(cfg,old.codec)
    batch=collate_stream([ds[i] for i in range(len(ds))])
    return ds,model,batch


def test_full_new_objective_has_finite_shared_gradients_and_no_independent_song_head(monkeypatch):
    torch.set_num_threads(1)
    _,model,batch=upgraded_fixture(monkeypatch)
    out=model(batch)
    loss,metrics=model.loss(out,batch)
    assert not hasattr(model.song_response,'song')
    assert not hasattr(model.phrase_context,'audio')
    assert 'unified_prior_kl' in metrics and 'unified_event_nll' in metrics
    assert torch.isfinite(loss)
    loss.backward()
    for module in (model.song_profile,model.phrase_execution.timing,model.detail_mode_control):
        grads=[p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
    assert out['execution_delta_logits'].shape[-2]==model.config.phrase_modes


def test_unified_prior_and_phrase_choice_do_not_read_unknown_profile_labels(monkeypatch):
    torch.set_num_threads(1)
    _,model,batch=upgraded_fixture(monkeypatch);model.eval()
    left=model(batch)
    other={**batch,'condition':batch['condition'].clone()}
    other['condition'][:,1:]=.77
    right=model(other)
    torch.testing.assert_close(left['song_response_logits'],right['song_response_logits'],atol=0,rtol=0)
    torch.testing.assert_close(left['song_response_phrase_logits'],right['song_response_phrase_logits'],atol=0,rtol=0)


def test_zero_component_interface_preserves_literal_weights_and_feedback(monkeypatch):
    _,model,_=upgraded_fixture(monkeypatch)
    ids=torch.arange(model.codec.num_atoms)
    weights=model.literal_catalog_weights()
    torch.testing.assert_close(weights,model.literal_head.weight,atol=0,rtol=0)
    torch.testing.assert_close(model.literal_feedback(ids,weights),model.atom_embedding(ids),atol=0,rtol=0)


def test_identical_modes_cannot_erase_any_configuration_likelihood(monkeypatch):
    import maichart.composer.stream_phrase_execution as execution
    original=execution.whole_phrase_configuration_loss;captured=[]
    def capture(scores,*args,**kwargs):
        captured.append(scores.detach().clone())
        return original(scores,*args,**kwargs)
    monkeypatch.setattr(execution,'whole_phrase_configuration_loss',capture)
    _,model,batch=upgraded_fixture(monkeypatch);model.eval()
    with torch.no_grad():
        for layer in (model.phrase_execution.route,model.phrase_execution.lanes):
            layer.weight.zero_()
        model.phrase_execution.reference_gain.zero_()
    out=model(batch);_,metrics=model.loss(out,batch)
    expected=metrics['shape_loss']+metrics['phase_loss']+metrics['detail_loss']
    direct=(captured[0][...,0]*batch['loss_mask']*batch['sample_weight']*batch['configuration_mask']).sum()/batch['_loss_normalizer']
    torch.testing.assert_close(direct,expected,rtol=1e-5,atol=1e-5)
    torch.testing.assert_close(captured[0],captured[0][...,:1].expand_as(captured[0]),rtol=0,atol=0)
    torch.testing.assert_close(metrics['configuration_loss'],expected,rtol=1e-5,atol=1e-5)


def test_timing_modes_reweight_grid_without_reading_landing_targets(monkeypatch):
    _,model,batch=upgraded_fixture(monkeypatch);model.eval()
    with torch.no_grad():
        width=5+model.timing_grid.num_groups
        model.phrase_execution.timing.bias[width+5]=2.
    out=model(batch)
    assert not torch.equal(out['execution_delta_logits'][...,0,:],out['execution_delta_logits'][...,1,:])
    future={**batch,'target_phrase_features':batch['target_phrase_features']+1000}
    changed=model(future)
    torch.testing.assert_close(out['execution_delta_logits'],changed['execution_delta_logits'],atol=0,rtol=0)


def test_literal_components_cover_touch_and_absolute_slides_with_one_recurrent_pass(monkeypatch):
    from maichart.composer.event_set_codec import EventSetCodec
    from maichart.composer.stream_shared_details import typed_atom_components
    _,old,_,_=execution_fixture(monkeypatch)
    codec=EventSetCodec.fit(['1','2h[4:1]','A1','B2','C','Ch[4:1]','1>3[8:1]','1pp5[4:1]'])
    model=StreamComposer(replace(old.config,detail_representation_policy='typed_components_v2',
        phrase_execution_policy='component_modes_v2'),codec)
    for atom in ('A1','B2','C','Ch[4:1]','1>3[8:1]','1pp5[4:1]'):
        assert typed_atom_components(atom) is not None
    targets=torch.tensor([[codec._atoms['A1'],codec._atoms['B2'],0],
                          [codec._atoms['1>3[8:1]'],0,-1]])
    ctx=torch.randn(2,model.config.d_model,requires_grad=True)
    calls=[]
    handle=model.literal_cell.register_forward_hook(lambda *args:calls.append(1))
    model.literal_block_nll(ctx,targets,return_stop=True)
    reference_calls=len(calls);calls.clear()
    base,stop,modes=model.literal_block_nll(ctx,targets,return_stop=True,return_modes=True)
    handle.remove()
    assert len(calls)==reference_calls
    torch.testing.assert_close(modes,base[:,None].expand_as(modes))
    modes.sum().backward()
    assert model.literal_component_projection.weight.grad is not None
    assert torch.isfinite(model.literal_component_projection.weight.grad).all()
