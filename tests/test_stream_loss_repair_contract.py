import pytest
from maichart.composer import stream_checkpoint as checkpoint
from maichart.composer.stream_train import _validation_improvement,_retain_best_files


def test_changed_unified_objective_requires_new_stage_but_legacy_is_unaffected():
    old=dict(model_config={'song_response_policy':'unified_profile_response_v2'},
        args={'training_objective_recipe':{'version':'stream_unified_song_rhythm_configuration_v8'}})
    assert not checkpoint.training_objective_compatible(old)
    old['args']['training_objective_recipe']['version']=checkpoint.LEGACY_UNIFIED_OBJECTIVE_VERSION
    assert checkpoint.training_objective_compatible(old)
    old['model_config']['draft_objective_policy']='sequence_evidence_v2'
    assert not checkpoint.training_objective_compatible(old)
    old['args']['training_objective_recipe']['version']=checkpoint.CONFIGURATION_SEQUENCE_OBJECTIVE_VERSION
    assert checkpoint.training_objective_compatible(old)
    old['model_config']['song_choice_objective_policy']='event_credit_v1'
    assert not checkpoint.training_objective_compatible(old)
    old['args']['training_objective_recipe']['version']=checkpoint.UNIFIED_OBJECTIVE_VERSION
    assert checkpoint.training_objective_compatible(old)
    assert checkpoint.training_objective_compatible({'model_config':{'song_response_policy':'music_response_v1'}})


def test_best_point_uses_development_objective_and_survives_later_regression(tmp_path):
    prior=_validation_improvement({'metrics':{'unified_balanced_variational_objective':5.,'loss':100.}},None,1,10)
    assert prior['score']==5 and prior['optimizer_steps']==10
    (tmp_path/'composer_last.pt').write_bytes(b'full optimizer and cursor at step10')
    (tmp_path/'composer_inference.pt').write_bytes(b'weights at step10')
    _retain_best_files(tmp_path)
    (tmp_path/'composer_last.pt').write_bytes(b'newer worse state')
    (tmp_path/'composer_inference.pt').write_bytes(b'newer worse weights')
    assert _validation_improvement({'metrics':{'unified_balanced_variational_objective':6.,'loss':1.}},prior,2,20) is None
    assert (tmp_path/'composer_best.pt').read_bytes()==b'full optimizer and cursor at step10'
    assert (tmp_path/'composer_best_inference.pt').read_bytes()==b'weights at step10'
    assert _validation_improvement({'metrics':{'unified_balanced_variational_objective':float('nan')}},prior,3,30) is None
    improved=_validation_improvement({'metrics':{'unified_balanced_variational_objective':4.}},prior,3,30)
    assert improved['optimizer_steps']==30
