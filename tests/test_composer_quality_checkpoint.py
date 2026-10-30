import torch
import pytest
from maichart.composer.checkpoint import save,load,inspect_payload,migrate_audio_grounded
from maichart.composer.model import build_model,resolve_composer_policy
from test_composer_audio_grounded_v3 import ATOMS,config


@pytest.mark.parametrize('policy',['cooperative_v1','cooperative_v2'])
def test_quality_checkpoint_roundtrip_and_recipe_are_explicit(tmp_path,policy):
    model=build_model(config(configuration_quality_policy=policy),ATOMS)
    path=tmp_path/'quality.pt'
    save(path,model,None,epoch=0,args={},manifest=tmp_path/'unused.json',atom_catalog=ATOMS)
    restored,payload=load(path)
    report=inspect_payload(payload)
    assert report['valid'] and report['configuration_quality_available']
    assert payload['configuration_quality_input_policy']=='native_lifecycle_paired_candidate_clock_phrase_v2'
    assert payload['training_objective_recipe']['configuration_quality']['gap_weight']==.1
    assert restored.config.configuration_quality_policy==policy
    recipe=payload['training_objective_recipe']['configuration_quality']
    assert recipe['first_slot_min_legal_mass']==(.01 if policy=='cooperative_v2' else 0.)
    for key,tensor in model.state_dict().items():
        if torch.is_tensor(tensor):
            torch.testing.assert_close(tensor,restored.state_dict()[key],rtol=0,atol=0)
        else:
            assert tensor==restored.state_dict()[key]


def test_explicit_quality_transfer_preserves_old_parameters_and_zeroes_new_residual():
    original=build_model(config(),ATOMS)
    upgraded=migrate_audio_grounded(original,ATOMS,configuration_quality_policy='cooperative_v1')
    assert upgraded is not original
    torch.testing.assert_close(upgraded.action_head.weight,original.action_head.weight,rtol=0,atol=0)
    assert not upgraded.quality_context.output.weight.detach().count_nonzero()
    assert resolve_composer_policy('configuration_quality_policy')=='song_paired_v3'


def test_first_quality_revision_transfers_neutrally_but_not_its_optimizer(tmp_path):
    model=build_model(config(configuration_quality_policy='cooperative_v1'),ATOMS)
    path=tmp_path/'old-quality.pt'
    save(path,model,torch.optim.AdamW(model.parameters()),epoch=0,args={},manifest=tmp_path/'unused.json',atom_catalog=ATOMS)
    payload=torch.load(path,weights_only=False)
    payload['configuration_quality_input_policy']='native_lifecycle_paired_anchors_phrase_v1'
    state=payload['model_state_dict']
    state.pop('cooperative_geometry.head_to_end_factors.weight')
    state.pop('quality_context.pair_time_query.weight')
    old_pair=state['quality_context.pair.0.weight'][:,:-6].clone()
    state['quality_context.pair.0.weight']=old_pair
    torch.save(payload,path)
    report=inspect_payload(payload)
    assert report['can_init'] and not report['can_resume']
    restored,_=load(path)
    assert not restored.cooperative_geometry.head_to_end_factors.weight.count_nonzero()
    assert not restored.quality_context.pair_time_query.weight.count_nonzero()
    assert not restored.quality_context.pair[0].weight[:,-6:].count_nonzero()
    torch.testing.assert_close(restored.quality_context.pair[0].weight[:,:-6],old_pair)
    with pytest.raises(ValueError,match='--init-checkpoint'):
        load(path,include_optimizer=True)
