import numpy as np
import pytest
import torch

from maichart.composer import structure,structure_backend
from maichart.composer.model import ComposerConfig,build_model,resolve_composer_policy
from maichart.composer.checkpoint import save,load,migrate_audio_grounded
from test_composer_structure import song
from test_composer_audio_grounded_v3 import inputs,ATOMS


def test_representatives_are_not_diluted_by_nearby_period_aliases():
    records=np.zeros((1,len(structure.PERIODS),8),np.float32)
    def index(period):return int(np.flatnonzero(np.isclose(structure.PERIODS,period))[0])
    records[0,index(4),0]=2.;records[0,index(8),0]=1.
    def selected_weights():
        order=structure._representative_order(records)
        strengths=records[0,np.maximum(order[0],0),0].copy();strengths[order[0]<0]=0.
        return order,structure._representative_weights(strengths)
    before,weights=selected_weights()
    for period in (3.75,4.25,7.75,8.5):records[0,index(period),0]=.9
    after,current=selected_weights()
    np.testing.assert_array_equal(before,after)
    np.testing.assert_array_equal(weights,current)
    np.testing.assert_allclose(weights,[.5,.25,0.,0.])
    assert current.sum()<1.
    np.testing.assert_array_equal(structure._representative_weights(np.zeros((2,4))),0.)


def test_tiny_channel_residue_does_not_receive_an_equal_harmonic_vote():
    grid=np.arange(0.,32.,.25)
    phase=np.exp(-2j*np.pi*grid[:,None]/np.array([3.,4.,8.]))
    accent=np.maximum(np.cos(2*np.pi*grid/4.),0.)
    primary=accent[:,None]
    first=structure._balanced_harmonics(phase,primary,primary.sum(0),2*primary.sum(0),len(grid))
    noise=np.full((len(grid),1),1e-6)
    augmented=np.column_stack((primary,noise))
    second=structure._balanced_harmonics(phase,augmented,augmented.sum(0),2*augmented.sum(0),len(grid))
    for a,b in zip(first,second):np.testing.assert_allclose(a,b,atol=1e-12)


def test_harmonic_mixture_does_not_depend_on_acoustic_channel_units():
    grid=np.arange(0.,32.,.25)
    phase=np.exp(-2j*np.pi*grid[:,None]/np.array([4.,8.]))
    residual=np.column_stack((np.maximum(np.cos(2*np.pi*grid/4.),0.),
                              np.maximum(np.cos(2*np.pi*grid/8.),0.)))
    energy=2*residual.sum(0)
    baseline=structure._balanced_harmonics(phase,residual,residual.sum(0),energy,len(grid))
    scaled=residual*np.array([1000.,1.])
    current=structure._balanced_harmonics(phase,scaled,scaled.sum(0),energy*np.array([1000.,1.]),len(grid))
    for a,b in zip(baseline,current):np.testing.assert_allclose(a,b,atol=1e-12)


def test_enhanced_features_expose_local_probabilities_and_cached_native_availability(tmp_path,monkeypatch):
    structure.clear_structure_caches();source=song()
    times=np.arange(2400)/50.
    probabilities=np.column_stack((np.full(len(times),.15),np.where(times%2.<.08,.8,.02)))
    logits=np.log(probabilities/(1-probabilities)).astype(np.float32)
    monkeypatch.setattr(structure_backend,'beat_this_frames',lambda *args,**kwargs:logits)
    harmonic_widths=[];original=structure._balanced_harmonics
    def capture(phase,residual,*args):
        harmonic_widths.append(residual.shape[1])
        return original(phase,residual,*args)
    monkeypatch.setattr(structure,'_balanced_harmonics',capture)
    kwargs=dict(policy=structure.ENHANCED_POLICY,backend='native',cache_dir=tmp_path)
    seconds=np.array([8.,8.5,-1.,100.]);beats=seconds*2
    actual=structure.structure_features(source,beats,seconds,**kwargs)
    assert actual.shape==(4,51) and np.isfinite(actual).all()
    assert harmonic_widths and set(harmonic_widths)=={4}  # three native + neural downbeat, no neural beat
    np.testing.assert_allclose(actual[:2,48],[.15,.15],atol=1e-6)
    np.testing.assert_allclose(actual[:2,49],[.8,.02],atol=1e-6)
    np.testing.assert_array_equal(actual[:,50],[1,1,0,0])
    np.testing.assert_array_equal(actual[2:,48:],0.)
    structure.clear_structure_caches()
    monkeypatch.setattr(structure_backend,'beat_this_frames',lambda *args,**kwargs:pytest.fail('raw analysis rerun on cache hit'))
    cached=structure.structure_features(source,beats,seconds,**kwargs)
    np.testing.assert_array_equal(cached,actual)


def test_native_enhanced_path_is_explicitly_missing_neural_evidence(tmp_path):
    structure.clear_structure_caches()
    source=song()
    features=structure.structure_features(source,[8.,12.],[4.,6.],policy=structure.ENHANCED_POLICY,
                                          backend='native',cache_dir=tmp_path)
    assert features.shape==(2,51) and np.isfinite(features).all()
    np.testing.assert_array_equal(features[:,48:],0.)
    legacy=structure.structure_fingerprint(source,tmp_path,'native')
    enhanced=structure.structure_fingerprint(source,tmp_path,'native',policy=structure.ENHANCED_POLICY)
    assert legacy!=enhanced


def test_new_input_contract_reaches_model_gradients_and_checkpoint(tmp_path):
    config=ComposerConfig(atom_vocab_size=len(ATOMS)+1,d_model=16,nhead=4,layers=1,dim_feedforward=32,
        dropout=0.,history_events=8,action_slots=3,architecture_profile='audio_grounded_v3',head_timing_dim=99,
        lens_width=8,lens_chunk_size=3,structure_policy=structure.ENHANCED_POLICY,structure_dropout=0.)
    model=build_model(config,ATOMS)
    data=inputs();data.update(candidate_indices=torch.tensor([1,2]),
        structure_features=torch.rand(2,6,51),structure_transition=torch.rand(2,6,51))
    output=model(**data)
    output['action_logits'].square().mean().backward()
    for name in ('structure_projection','structure_action_projection','structure_transition_projection'):
        assert getattr(model,name).weight.grad[:,48:].abs().sum()>0
    path=tmp_path/'enhanced.pt'
    save(path,model,None,epoch=1,args={},manifest='unused',atom_catalog=ATOMS)
    restored,payload=load(path)
    assert restored.config.structure_schema==structure.ENHANCED_SCHEMA
    assert payload['training_semantics'].endswith('+structure_metrical_context_v2')
    assert resolve_composer_policy('structure_policy')==structure.ENHANCED_POLICY
    assert resolve_composer_policy('structure_policy',parent_config={'structure_policy':'metrical_context_v1'},resume=True)=='metrical_context_v1'


def test_legacy_to_enhanced_transfer_uses_new_schema_and_neutral_structure_projection():
    config=ComposerConfig(atom_vocab_size=len(ATOMS)+1,d_model=16,nhead=4,layers=1,dim_feedforward=32,
        architecture_profile='audio_grounded_v3',head_timing_dim=99,lens_width=8,structure_policy='metrical_context_v1')
    old=build_model(config,ATOMS)
    child=migrate_audio_grounded(old,ATOMS,structure_policy=structure.ENHANCED_POLICY)
    assert child.config.structure_schema==structure.ENHANCED_SCHEMA
    assert child.structure_projection.weight.shape==(16,51)
    assert not torch.count_nonzero(child.structure_projection.weight)
    torch.testing.assert_close(child.action_head.weight,old.action_head.weight,rtol=0,atol=0)
    assert old.config.structure_schema==structure.SCHEMA
