from types import SimpleNamespace

import pytest
import torch

from maichart.composer.model import ComposerConfig, build_model, resolve_history_anchor_policy
from maichart.composer.checkpoint import save, load, migrate_audio_grounded
from maichart.composer.train import main, parser
from maichart.webui import composer_api


def _parent(tmp_path):
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    atoms=['1','4','Ch[8:1]']
    model=build_model(ComposerConfig(atom_vocab_size=4,d_model=8,nhead=2,layers=1,
        dim_feedforward=16,history_events=8,candidate_count=16,action_slots=2,
        architecture_profile='audio_grounded_v3',head_timing_dim=99,lens_width=8,
        history_anchor_policy='legacy_digit_v1'),atoms)
    path=tmp_path/'parent.pt'
    payload=save(path,model,torch.optim.AdamW(model.parameters()),epoch=1,args={},
                 manifest=manifest,atom_catalog=atoms)
    payload['model_config'].pop('history_anchor_policy')
    torch.save(payload,path)
    return {'manifest':str(manifest),'checkpoint_dir':str(tmp_path/'next'),
            'cache_dir':str(tmp_path/'cache'),'checkpoint':str(path)}


def _request(payload):
    calls=[]
    runtime=SimpleNamespace(_create_task=lambda kind:SimpleNamespace(task_id='fixture',kind=kind,result={}),
        _run_subprocess_task=lambda task,args,**kwargs:calls.append((task,args)))
    composer_api.start_train(SimpleNamespace(_json_response=lambda _:None),payload,runtime=runtime)
    return calls[0]


def test_auto_resolution_preserves_existing_grounded_models():
    parent={'architecture_profile':'audio_grounded_v3'}
    assert resolve_history_anchor_policy()=='parsed_geometry_v3'
    assert resolve_history_anchor_policy('auto',parent)=='legacy_digit_v1'
    assert resolve_history_anchor_policy('auto',parent,resume=True)=='legacy_digit_v1'
    assert resolve_history_anchor_policy('parsed_heads_v2',parent)=='parsed_heads_v2'
    with pytest.raises(ValueError,match='Resume inherits'):
        resolve_history_anchor_policy('parsed_heads_v2',parent,resume=True)


@pytest.mark.parametrize('value',['unknown',None,False,[],{}])
def test_invalid_policy_is_not_silently_accepted(value):
    with pytest.raises(ValueError,match='history_anchor_policy'):
        resolve_history_anchor_policy(value)


@pytest.mark.parametrize('mode,policy,expected',[
    ('fresh','auto','parsed_geometry_v3'),('resume','auto','legacy_digit_v1'),
    ('finetune','auto','legacy_digit_v1'),('finetune','parsed_heads_v2','parsed_heads_v2')])
def test_api_forwards_resolved_policy_and_records_effective_value(tmp_path,mode,policy,expected):
    payload=_parent(tmp_path);payload.update(mode=mode,history_anchor_policy=policy)
    task,args=_request(payload)
    parsed=parser().parse_args(args[2:])
    assert parsed.history_anchor_policy==expected
    assert task.result['effective_settings']['history_anchor_policy']==expected


def test_resume_policy_change_rejected_before_task_creation_and_data_loading(tmp_path,monkeypatch):
    payload=_parent(tmp_path);payload.update(mode='resume',history_anchor_policy='parsed_heads_v2')
    runtime=SimpleNamespace(_create_task=lambda _:pytest.fail('incompatible policy created a task'))
    with pytest.raises(ValueError,match='Resume inherits'):
        composer_api.start_train(SimpleNamespace(_json_response=lambda _:None),payload,runtime=runtime)
    from maichart.composer import train
    monkeypatch.setattr(train,'ComposerDataset',lambda *args,**kw:pytest.fail('incompatible policy reached dataset loading'))
    with pytest.raises(ValueError,match='Resume inherits'):
        main(['--manifest',payload['manifest'],'--checkpoint-dir',payload['checkpoint_dir'],
              '--resume',payload['checkpoint'],'--history-anchor-policy','parsed_heads_v2','--device','cpu'])


def test_explicit_migration_changes_only_derived_history_representation(tmp_path):
    payload=_parent(tmp_path);model,old=load(payload['checkpoint'])
    weights={k:v.clone() for k,v in model.state_dict().items() if torch.is_tensor(v)}
    upgraded=migrate_audio_grounded(model,old['atom_catalog'],history_anchor_policy='parsed_heads_v2')
    assert model.config.history_anchor_policy=='legacy_digit_v1'
    assert upgraded.config.history_anchor_policy==upgraded.action_factors.history_anchor_policy=='parsed_heads_v2'
    for name,value in weights.items():
        torch.testing.assert_close(upgraded.state_dict()[name],value,rtol=0,atol=0)
    ids=torch.tensor([3])
    assert model.action_factors.history_anchor_embedding(ids,model.event_embedding).abs().sum()>0
    assert upgraded.action_factors.history_anchor_embedding(ids,upgraded.event_embedding).abs().sum()==0


def test_checkpoint_refuses_mismatched_policy_metadata_before_writing(tmp_path):
    payload=_parent(tmp_path);model,old=load(payload['checkpoint'])
    model.action_factors.set_history_anchor_policy('parsed_heads_v2')
    target=tmp_path/'do_not_write.pt'
    with pytest.raises(ValueError,match='Save history_anchor_policy differs'):
        save(target,model,None,epoch=1,args={},manifest=payload['manifest'],atom_catalog=old['atom_catalog'])
    assert not target.exists()
