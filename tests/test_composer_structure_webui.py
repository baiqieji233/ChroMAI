from types import SimpleNamespace
import pytest
from maichart.webui import composer_api as api


@pytest.fixture
def harness(tmp_path, monkeypatch):
    import maichart.composer.checkpoint as checkpoint
    manifest=tmp_path/'manifest.json'; manifest.write_text('{}')
    parent=tmp_path/'parent.pt'; parent.write_bytes(b'mock')
    inspection={'can_resume':True,'can_init':True,'model_config':{'architecture_profile':'audio_grounded_v3'},'args':{}}
    monkeypatch.setattr(checkpoint,'read_payload',lambda path: {})
    monkeypatch.setattr(checkpoint,'inspect_payload',lambda data: inspection)
    calls=[]
    runtime=SimpleNamespace(_create_task=lambda kind:SimpleNamespace(task_id='test',result={}),
        _run_subprocess_task=lambda task,args,**kw:calls.append((task,args)))
    handler=SimpleNamespace(_json_response=lambda data:None)
    payload={'manifest':str(manifest),'checkpoint_dir':str(tmp_path/'next'),
             'cache_dir':str(tmp_path/'cache'),'checkpoint':str(parent)}
    return payload, inspection, calls, handler, runtime


def option(args, name):
    i=args.index('--'+name.replace('_','-'))
    return args[i+1]


def test_fresh_and_legacy_checkpoint_defaults(harness):
    payload, parent, calls, handler, runtime=harness
    api.start_train(handler,dict(payload,mode='fresh'),runtime=runtime)
    assert option(calls[-1][1],'structure_policy')=='metrical_context_v2'
    assert option(calls[-1][1],'structure_backend')=='auto'
    assert option(calls[-1][1],'contact_policy')=='diagnostic_v1'
    api.start_train(handler,dict(payload,mode='resume'),runtime=runtime)
    assert option(calls[-1][1],'structure_policy')=='off'
    assert option(calls[-1][1],'contact_policy')=='off'


def test_resume_inherits_and_rejects_changes(harness):
    payload, parent, calls, handler, runtime=harness
    parent['model_config'].update(structure_policy='metrical_context_v1',
        structure_backend='native',structure_dropout=.2,contact_policy='diagnostic_v1')
    api.start_train(handler,dict(payload,mode='resume'),runtime=runtime)
    assert option(calls[-1][1],'structure_dropout')=='0.2'
    for change in ({'structure_backend':'beat_this'}, {'structure_dropout':.3},
                   {'structure_policy':'off'}, {'contact_policy':'off'}):
        with pytest.raises(ValueError,match='[Rr]esume inherits'):
            api.start_train(handler,dict(payload,mode='resume',**change),runtime=runtime)


def test_invalid_and_generation_policy_forwarding(harness,tmp_path):
    payload, parent, calls, handler, runtime=harness
    with pytest.raises(ValueError):
        api.start_train(handler,dict(payload,mode='fresh',structure_dropout=.91),runtime=runtime)
    audio=tmp_path/'audio.wav';audio.write_bytes(b'mock')
    api.start_generate(handler,dict(audio=str(audio),checkpoint=payload['checkpoint'],
        output=str(tmp_path/'out.txt'),contact_policy='diagnostic_v1'),runtime=runtime)
    assert option(calls[-1][1],'contact_policy')=='diagnostic_v1'
    assert calls[-1][0].result['effective_settings']['contact_policy']=='diagnostic_v1'
