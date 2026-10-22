import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from maichart.composer.generation_settings import resolve_seed,resolve_condition
from maichart.composer.generate import _sample_actions,_weight_fingerprint
from maichart.generation.event_cli import _NoteGrammar


def test_blank_seed_is_random_and_zero_remains_reproducible(monkeypatch):
    from maichart.composer import generation_settings as settings
    values=iter([17,23,41])
    monkeypatch.setattr(settings.secrets,'randbelow',lambda _:next(values))
    assert resolve_seed(None)==(17,True)
    assert resolve_seed('')==(23,True)
    assert resolve_seed('random')==(41,True)
    assert resolve_seed(0)==resolve_seed('0')==(0,False)
    for bad in (-1,1.2,True,2**63):
        with pytest.raises(ValueError):resolve_seed(bad)


def test_sparse_condition_file_overrides_only_explicit_fields(tmp_path):
    path=tmp_path/'style.json'
    path.write_text(json.dumps({'style':{'slide_bias':.7}}))
    condition,source=resolve_condition(14.2,{'level':14.2,'radar':{'note_density':.4},
        'style':{'tap_bias':.1,'hold_bias':.3}},path)
    assert source=='ui+file'
    assert condition.level==14.2
    assert condition.style['slide_bias']==.7 and condition.style['hold_bias']==.3
    assert condition.radar['note_density']==.4


def test_temperature_sampling_varies_with_seed_and_greedy_does_not():
    atoms=['1','2','3'];grammar=_NoteGrammar.from_atoms(atoms)
    logits=torch.zeros(1,4)
    def draw(seed,temperature):
        generator=torch.Generator().manual_seed(seed)
        return [_sample_actions(logits,atoms,grammar,temperature,generator) for _ in range(20)]
    assert draw(0,.8)==draw(0,.8)
    assert draw(0,.8)!=draw(1,.8)
    assert draw(0,0)==draw(1,0)
    # Impossible STOP must remain impossible even if every allowed logit is
    # below the old finite -10000 mask sentinel.
    assert _sample_actions(torch.tensor([[0.,-20000.,-30000.,-40000.]]),atoms,grammar,0,None)==['1']


def test_weight_identity_changes_only_with_weights():
    state={'a':torch.ones(3,4),'b':torch.zeros(2)}
    assert _weight_fingerprint(state)==_weight_fingerprint(dict(reversed(list(state.items()))))
    other={key:value.clone() for key,value in state.items()};other['a'][0,0]=2
    assert _weight_fingerprint(state)!=_weight_fingerprint(other)


def test_backend_generation_preserves_controls_and_isolates_outputs(tmp_path):
    from maichart.webui import composer_api as api
    audio=tmp_path/'audio.wav';audio.write_bytes(b'audio')
    checkpoint=tmp_path/'composer.pt';checkpoint.write_bytes(b'weights')
    calls=[];tasks=[];responses=[]
    def create(kind):
        task=SimpleNamespace(task_id=f'task_{len(tasks)}',kind=kind,result={})
        tasks.append(task);return task
    runtime=SimpleNamespace(_create_task=create,
        _run_subprocess_task=lambda task,args,**kw:calls.append((task,args,kw)))
    handler=SimpleNamespace(_json_response=responses.append)
    payload={'audio':str(audio),'checkpoint':str(checkpoint),'output':str(tmp_path/'maidata.txt'),
        'seed':0,'temperature':.91,'level':14.2,'condition':{'radar':{'note_density':.42},
        'style':{'hold_bias':.8}},'tempo_map':[{'beat':'0','bpm':150},{'beat':'8','bpm':180}]}
    api.start_generate(handler,payload,runtime=runtime)
    api.start_generate(handler,payload,runtime=runtime)
    first,args,kw=calls[0]
    assert args[args.index('--seed')+1]=='0'
    assert args[args.index('--temperature')+1]=='0.91'
    saved=json.loads(Path(args[args.index('--condition-file')+1]).read_text(encoding='utf8'))
    assert saved['radar']['note_density']==.42 and saved['style']['hold_bias']==.8
    assert json.loads(Path(args[args.index('--tempo-map')+1]).read_text())[1]['bpm']==180
    assert calls[0][0].result['output']!=calls[1][0].result['output']
    assert Path(first.result['output']).parent.name=='task_0'
    assert first.result['seed_randomized'] is False
    assert kw['result_paths'][0]==Path(first.result['output'])
    assert responses[0]['seed']==0


def test_backend_resolves_blank_seed_before_launch(tmp_path,monkeypatch):
    from maichart.webui import composer_api as api
    from maichart.composer import generation_settings as settings
    values=iter([432,876]);monkeypatch.setattr(settings.secrets,'randbelow',lambda _:next(values))
    audio=tmp_path/'a.wav';audio.write_bytes(b'audio')
    checkpoint=tmp_path/'cp.pt';checkpoint.write_bytes(b'checkpoint')
    calls=[]
    runtime=SimpleNamespace(_create_task=lambda kind:SimpleNamespace(task_id=str(len(calls)),kind=kind,result={}),
        _run_subprocess_task=lambda task,args,**kw:calls.append((task,args)))
    for seed in (None,''):
        api.start_generate(SimpleNamespace(_json_response=lambda x:None),
            {'audio':str(audio),'checkpoint':str(checkpoint),'output':str(tmp_path/'chart.txt'),'seed':seed},runtime=runtime)
    assert [call[0].result['seed'] for call in calls]==[432,876]
    assert all('--seed-randomized' in args for _,args in calls)


def test_collected_composer_report_exposes_actual_weight_and_output(tmp_path):
    from maichart.webui.server import TaskState,_collect_task_results
    path=tmp_path/'chart.composer.json'
    path.write_text(json.dumps({'seed':23,'weights_sha256':'a'*64,'events_sha256':'b'*64}))
    task=TaskState('fixture','composer_generate')
    _collect_task_results(task,[path])
    assert task.result['composer_report']['seed']==23


def test_checkpoint_cpu_snapshot_preserves_tensors_and_state_metadata():
    from collections import OrderedDict
    from maichart.composer.checkpoint import _cpu_snapshot
    tensor=torch.randn(3,4)
    state=OrderedDict(weight=tensor);state._metadata={'version':7}
    source={'model':state,'optimizer':{'states':[tensor,{'step':torch.tensor(3.)}]}}
    snapshot=_cpu_snapshot(source)
    assert snapshot['model'] is not state
    assert snapshot['model']['weight'] is tensor
    assert snapshot['model']._metadata==state._metadata
    assert snapshot['optimizer']['states'][0] is tensor
