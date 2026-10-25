import pytest
import torch

from maichart.composer.model import (ComposerConfig,ChartComposer,build_model,
    resolve_composer_policy)
from maichart.composer.checkpoint import save,load,validate,migrate_music_context


def config(**kwargs):
    return ComposerConfig(atom_vocab_size=3,d_model=16,nhead=4,layers=1,
        dim_feedforward=32,history_events=4,candidate_count=4,action_slots=2,
        architecture_profile='music_context_v2',head_timing_dim=99,**kwargs)


@pytest.mark.parametrize('policy',[
    {'motif_memory_policy':'audio_recurrence_archive_v1'},
    {'action_state_policy':'candidate_prefix_v2'},
    {'slide_factor_policy':'branch_joint_v2'},
])
def test_unsupported_non_grounded_policies_fail_in_real_constructors(policy):
    cfg=config(**policy)
    with pytest.raises(ValueError,match='requires audio_grounded_v3'):build_model(cfg)
    with pytest.raises(ValueError,match='requires audio_grounded_v3'):ChartComposer(cfg)


@pytest.mark.parametrize(('flag','value'),[
    ('--motif-memory-policy','audio_recurrence_archive_v1'),
    ('--action-state-policy','candidate_prefix_v2'),
    ('--slide-factor-policy','branch_joint_v2'),
])
def test_cli_rejects_unsupported_policies_before_dataset(tmp_path,monkeypatch,flag,value):
    from maichart.composer import train
    monkeypatch.setattr(train,'ComposerDataset',lambda *a,**kw:pytest.fail('Dataset constructed before policy rejection'))
    with pytest.raises(ValueError,match='requires audio_grounded_v3'):
        train.main(['--manifest',str(tmp_path/'missing.json'),'--checkpoint-dir',str(tmp_path/'new'),
                    '--device','cpu','--architecture-profile','music_context_v2',flag,value])


def test_non_grounded_finetune_config_survives_save_load(tmp_path):
    parent=ChartComposer(config(action_state_policy='committed_legacy_v1',history_anchor_policy='legacy_digit_v1'))
    upgraded=migrate_music_context(parent,action_state_policy='committed_causal_v2',
                                  history_anchor_policy='parsed_geometry_v3')
    assert upgraded is not parent
    assert parent.config.action_state_policy=='committed_legacy_v1'
    for key,value in parent.state_dict().items():torch.testing.assert_close(value,upgraded.state_dict()[key])
    path=tmp_path/'composer.pt'
    save(path,upgraded,None,epoch=1,args={},manifest='unused',atom_catalog=['1','2'])
    loaded,_=load(path)
    assert loaded.config.action_state_policy=='committed_causal_v2'
    assert loaded.config.history_anchor_policy=='parsed_geometry_v3'
    assert migrate_music_context(loaded,action_state_policy='committed_causal_v2') is loaded


def test_grounded_can_build_all_context_features_through_shared_backbone():
    cfg=config(action_state_policy='candidate_prefix_v2',motif_memory_policy='audio_recurrence_archive_v1',
               slide_factor_policy='branch_joint_v2')
    cfg.architecture_profile='audio_grounded_v3'
    model=build_model(cfg,['1','2'])
    assert hasattr(model,'motif_memory')
    assert model.action_factors.slide_factor_policy=='branch_joint_v2'
    assert model.config.action_state_policy=='candidate_prefix_v2'


@pytest.mark.parametrize('requested',[[],{},None,1])
def test_policy_resolver_rejects_nonstring_with_value_error(requested):
    with pytest.raises(ValueError):resolve_composer_policy('action_state_policy',requested)


def test_fresh_factor_default_follows_effective_architecture():
    assert ComposerConfig(atom_vocab_size=3).slide_factor_policy=='marginal_v1'
    assert resolve_composer_policy('slide_factor_policy',architecture_profile='music_context_v2')=='marginal_v1'
    assert resolve_composer_policy('slide_factor_policy',architecture_profile='audio_grounded_v3')=='branch_joint_v2'


def test_api_rejects_unsupported_context_before_task_creation(tmp_path):
    from types import SimpleNamespace
    from maichart.webui import composer_api
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    runtime=SimpleNamespace(_create_task=lambda _:pytest.fail('Task created'))
    with pytest.raises(ValueError,match='requires audio_grounded_v3'):
        composer_api.start_train(SimpleNamespace(),{'manifest':str(manifest),
            'checkpoint_dir':str(tmp_path/'checkpoints'),'cache_dir':str(tmp_path/'cache'),'architecture_profile':'music_context_v2',
            'motif_memory_policy':'audio_recurrence_archive_v1'},runtime=runtime)


def test_checkpoint_validation_rejects_unsupported_non_grounded_context(tmp_path):
    path=tmp_path/'composer.pt'
    payload=save(path,ChartComposer(config()),None,epoch=1,args={},manifest='unused',atom_catalog=['1','2'])
    payload['model_config']['motif_memory_policy']='audio_recurrence_archive_v1'
    with pytest.raises(ValueError,match='requires audio_grounded_v3'):validate(payload)


def test_fresh_non_grounded_api_uses_marginal_factor_default(tmp_path):
    from types import SimpleNamespace
    from maichart.webui import composer_api
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    calls=[]
    runtime=SimpleNamespace(_create_task=lambda kind:SimpleNamespace(task_id='new',kind=kind,result={}),
        _run_subprocess_task=lambda task,args,**kw:calls.append((task,args)))
    composer_api.start_train(SimpleNamespace(_json_response=lambda _:None),
        {'manifest':str(manifest),'checkpoint_dir':str(tmp_path/'out'),'cache_dir':str(tmp_path/'cache'),
         'architecture_profile':'music_context_v2'},runtime=runtime)
    task,args=calls[0]
    assert task.result['effective_settings']['slide_factor_policy']=='marginal_v1'
    assert args[args.index('--slide-factor-policy')+1]=='marginal_v1'
