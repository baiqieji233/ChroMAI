"""Verify structural evidence reaches action and selected-prefix decoding."""
import copy
import pytest
import torch

from maichart.composer.model import build_model
from maichart.composer.checkpoint import migrate_audio_grounded, save, load
from test_composer_audio_grounded_v3 import config, inputs, ATOMS


@pytest.mark.parametrize('state_policy',['committed_causal_v2','selected_prefix_v3'])
def test_structure_changes_configuration_and_cross_event_route(state_policy):
    torch.manual_seed(54)
    model=build_model(config(structure_policy='metrical_context_v1',structure_dropout=0.,
        action_state_policy=state_policy,cross_event_relation_policy='relative_heads_v1'),ATOMS).eval()
    with torch.no_grad():
        model.structure_action_projection.weight.normal_(0,.2)
        model.structure_transition_projection.weight.normal_(0,.2)
        model.cross_event_relative_factors.weight.normal_(0,.1)
    data=inputs()
    data.update(candidate_indices=torch.tensor([1,2]),action_targets=torch.tensor([[1,2,0],[2,1,0]]),
        structure_features=torch.zeros(2,6,48),structure_transition=torch.zeros(2,6,48))
    if state_policy=='selected_prefix_v3':data['selected_action_state']=torch.randn(2,50)
    baseline=model(**data)
    changed=model(**{**data,'structure_features':torch.randn(2,6,48),
                    'structure_transition':torch.randn(2,6,48)})
    # Isolate action wiring by leaving the timing projection neutral.
    torch.testing.assert_close(baseline['candidate_logits'],changed['candidate_logits'])
    assert not torch.equal(baseline['action_logits'],changed['action_logits'])
    assert not torch.equal(baseline['audio_action_logits'],changed['audio_action_logits'])
    changed['action_logits'].square().mean().backward()
    for name in ('structure_action_projection','structure_transition_projection'):
        gradient=getattr(model,name).weight.grad
        assert gradient is not None and gradient.abs().sum()>0


def test_explicit_neutral_structure_migration_preserves_parent_and_versions_checkpoint(tmp_path):
    torch.manual_seed(21)
    parent=build_model(config(),ATOMS).eval()
    child=migrate_audio_grounded(parent,ATOMS,structure_policy='metrical_context_v1',
        structure_backend='native',structure_dropout=0.).eval()
    data=inputs()
    data.update(candidate_indices=torch.tensor([1,2]),action_targets=torch.tensor([[1,0,0],[2,0,0]]))
    baseline=parent(**data)
    upgraded=child(**data,structure_features=torch.randn(2,6,48))
    torch.testing.assert_close(baseline['action_logits'],upgraded['action_logits'],rtol=0,atol=0)
    path=tmp_path/'new.pt'
    save(path,child,None,epoch=1,args={},manifest=tmp_path/'manifest.json',atom_catalog=ATOMS)
    restored,payload=load(path)
    assert payload['training_semantics'].endswith('+structure_metrical_context_v1')
    assert restored.config.structure_policy=='metrical_context_v1'
    assert parent.config.structure_policy=='off'


def test_structure_enabled_model_refuses_missing_evidence():
    model=build_model(config(structure_policy='metrical_context_v1'),ATOMS)
    with pytest.raises(ValueError,match='structure features'):
        model(**inputs())


def test_checkpoint_refuses_silent_structure_algorithm_drift():
    with pytest.raises(ValueError,match='schema differs'):
        config(structure_policy='metrical_context_v1',structure_schema='a_different_feature_algorithm')
