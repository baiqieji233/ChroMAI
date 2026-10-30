"""New output vocabularies and priors are training-only and resume-stable."""
from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace
import torch

from maichart.composer.stream_data import StreamStep, NOTE
from maichart.composer.stream_train import _initialize_structured_heads, _validation_timing_support, upgrade_learning_stage
from test_composer_stream_model import fixture


def test_new_joint_model_has_finite_loss_and_direct_gradients():
    model,batch=fixture(timing_output_policy='absolute_grid_v1',event_family_policy='semantic_mass_v1')
    loss,metrics=model.loss(model(batch),batch)
    assert torch.isfinite(loss)
    loss.backward()
    for name in ('timing_grid.grid_head','event_family_head'):
        gradient=model.get_submodule(name).weight.grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum()>0
    torch.testing.assert_close(metrics['timing_grid_loss']+metrics['timing_within_grid_loss'],metrics['delta_loss'])
    torch.testing.assert_close(metrics['event_family_loss']+metrics['event_route_conditional_loss'],metrics['shape_loss'])


def test_structured_prior_uses_actual_weighted_training_events_and_resume_keeps_weights():
    model,_=fixture()
    model=upgrade_learning_stage(model,dict(timing_output_policy='absolute_grid_v1',event_family_policy='semantic_mass_v1'))
    texts=['1/5','A1/Cf']
    steps=[StreamStep(NOTE,Fraction(0),Fraction(1,2),0.,.5,texts[0]),
           StreamStep(NOTE,Fraction(1,2),Fraction(3,4),.5,.75,texts[1])]
    ds=SimpleNamespace(streams=[steps],encoded=[[(1,model.codec.encode(text)) for text in texts]],
        unknown_steps=[[False,False]],target_window_weights=[[1.,3.]],chart_weights=[2.])
    prior=_initialize_structured_heads(model,ds)
    assert prior['family_counts'][0]==2 and prior['family_counts'][7]==6
    assert prior['literal_counts'][0]==0 and prior['literal_counts'][7]==6
    assert Fraction(3,4) in model.timing_grid.phase_catalog
    before={name:parameter.detach().clone() for name,parameter in model.named_parameters()}
    saved=replace(model.config)
    restored=_initialize_structured_heads(model,ds,saved,resume=True)
    assert not restored['timing_prior_initialized'] and not restored['semantic_prior_initialized']
    for name,parameter in model.named_parameters():torch.testing.assert_close(parameter,before[name],rtol=0,atol=0)


def test_structured_stage_retains_every_parent_parameter():
    model,_=fixture()
    legacy=upgrade_learning_stage(model,dict(timing_output_policy='relative_delta_v1',event_family_policy='legacy_v1'))
    state={key:value.clone() for key,value in legacy.state_dict().items()}
    repaired=upgrade_learning_stage(legacy,dict(timing_output_policy='absolute_grid_v1',event_family_policy='semantic_mass_v1'))
    for key,value in state.items():torch.testing.assert_close(repaired.state_dict()[key],value,rtol=0,atol=0)


def test_dev_unknown_phase_is_reported_without_learning_it_or_erasing_history():
    model,_=fixture(timing_output_policy='absolute_grid_v1')
    phase_catalog=tuple(model.timing_grid.phase_catalog)
    step=StreamStep(NOTE,Fraction(0),Fraction(1,13),0.,1/13,'1')
    ds=SimpleNamespace(streams=[[step]],unknown_steps=[[False]],
        oov=dict(total_steps=1,masked_steps=0))
    _validation_timing_support(ds,model)
    assert ds.oov['absolute_phase_oov']==1 and ds.oov['known_steps']==0
    assert ds.unknown_steps==[[True]] and ds.streams[0][0] is step
    assert tuple(model.timing_grid.phase_catalog)==phase_catalog
