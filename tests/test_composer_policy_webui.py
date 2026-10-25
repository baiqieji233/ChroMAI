from types import SimpleNamespace
import pytest
from maichart.webui import composer_api as api


@pytest.fixture
def harness(tmp_path, monkeypatch):
    import maichart.composer.checkpoint as checkpoint
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    parent=tmp_path/'parent.pt';parent.write_bytes(b'mock')
    inspection={'can_resume':True,'can_init':True,'model_config':{'architecture_profile':'audio_grounded_v3'},'args':{}}
    monkeypatch.setattr(checkpoint,'read_payload',lambda path: {})
    monkeypatch.setattr(checkpoint,'inspect_payload',lambda data: inspection)
    calls=[]
    runtime=SimpleNamespace(_create_task=lambda kind:SimpleNamespace(task_id='test',result={}),
        _run_subprocess_task=lambda task,args,**kw:calls.append((task,args)))
    handler=SimpleNamespace(_json_response=lambda data: None)
    payload={'manifest':str(manifest),'checkpoint_dir':str(tmp_path/'next'),'cache_dir':str(tmp_path/'cache'),'checkpoint':str(parent)}
    return payload,inspection,calls,handler,runtime


def value(args,key):return args[args.index('--'+key.replace('_','-'))+1]


@pytest.mark.parametrize('mode',['fresh','finetune','resume'])
def test_auto_policies_resolve_and_are_forwarded_without_training(harness,mode):
    payload,parent,calls,handler,runtime=harness
    api.start_train(handler,dict(payload,mode=mode),runtime=runtime)
    task,args=calls[0]
    expected={'action_state_policy':'committed_causal_v2' if mode=='fresh' else 'committed_legacy_v1',
              'motif_memory_policy':'off','slide_factor_policy':'branch_joint_v2' if mode=='fresh' else 'marginal_v1',
              'touch_layout_policy':'open' if mode=='fresh' else 'legacy_two_heads','motif_max_events':16}
    for key,want in expected.items():
        assert value(args,key)==str(want)
        assert task.result['effective_settings'][key]==want


def test_explicit_finetune_policies_forward_and_resume_rejects(harness):
    payload,parent,calls,handler,runtime=harness
    changes=dict(action_state_policy='candidate_prefix_v2',motif_memory_policy='audio_recurrence_archive_v1',
                 slide_factor_policy='branch_joint_v2',motif_max_events=24,touch_layout_policy='open')
    api.start_train(handler,dict(payload,mode='finetune',**changes),runtime=runtime)
    for key,want in changes.items():assert value(calls[0][1],key)==str(want)
    for key,want in changes.items():
        with pytest.raises(ValueError,match='[Rr]esume inherits'):
            api.start_train(handler,dict(payload,mode='resume',**{key:want}),runtime=runtime)
    assert len(calls)==1


def test_generation_new_options_reach_cli_and_echo(harness,tmp_path):
    payload,parent,calls,handler,runtime=harness
    audio=tmp_path/'audio.wav';audio.write_bytes(b'mock')
    api.start_generate(handler,dict(audio=str(audio),checkpoint=payload['checkpoint'],
        output=str(tmp_path/'out.txt'),touch_layout_policy='open',motif_strength=1.7),runtime=runtime)
    task,args=calls[0]
    assert value(args,'touch_layout_policy')=='open' and value(args,'motif_strength')=='1.7'
    assert task.result['effective_settings']['motif_strength']==1.7
    assert task.result['generation_contract']=='composer_generation_controls_v10_touch_layout'
    for changes in [{'motif_strength':2.1},{'touch_layout_policy':'auto'}]:
        with pytest.raises(ValueError):
            api.start_generate(handler,dict(audio=str(audio),checkpoint=payload['checkpoint'],**changes),runtime=runtime)
