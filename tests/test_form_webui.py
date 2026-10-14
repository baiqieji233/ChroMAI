import json
from pathlib import Path

from maichart.webui import server as webui
from maichart.webui import form_api
from test_webui import webui_api


def test_analysis_keeps_meter_separate_from_tempo_and_never_uses_segment_beats(webui_api,monkeypatch,tmp_path):
    audio=tmp_path/'song.wav';audio.write_bytes(b'fixture')
    calls=[]
    monkeypatch.setattr(webui,'_run_subprocess_task',lambda task,args,**kwargs:calls.append((task,args)))
    status,result=webui_api('/api/form/analyze',{'audio':str(audio),'bpm':180,'first':.25,
        'meter_map':[{'beat':'0','numerator':3,'denominator':4}],
        'tempo_map':[{'beat':'0','bpm':180},{'beat':'9','bpm':150}]})
    assert status==200,result
    task,args=calls[0]
    from maichart.form.cli import parser
    options=parser().parse_args(args[2:])
    assert options.operation=='analyze'
    assert json.load(open(options.meter_map,encoding='utf-8'))[0]['numerator']==3
    assert json.load(open(options.tempo_map,encoding='utf-8'))[-1]['beat']=='9'
    assert '--segment-beats' not in args
    assert task.result['experimental']


def test_training_dispatch_is_independent_and_short_run_controls_arrive(webui_api,monkeypatch,tmp_path):
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    calls=[]
    monkeypatch.setattr(webui,'_run_subprocess_task',lambda task,args,**kwargs:calls.append(args))
    status,result=webui_api('/api/form/train',{'manifest':str(manifest),'checkpoint_dir':str(tmp_path/'new'),
        'mode':'fresh','max_steps':2,'limit_charts':1,'epochs':1,'batch_size':1,'d_model':32,'nhead':2,'layers':1,
        'device':'cpu','lr':.0003,'prepare_workers':3})
    assert status==200,result
    from maichart.form.train import parser
    args=parser().parse_args(calls[0][2:])
    assert args.max_steps==2 and args.limit_charts==1 and args.lr==.0003 and args.prepare_workers==3
    assert calls[0][:2]==['-m','maichart.form.train']


def test_old_checkpoint_is_not_accepted_by_new_training(webui_api,tmp_path):
    import torch
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    checkpoint=tmp_path/'old.pt';torch.save({'schema':'maichart-event-generator-checkpoint-v1'},checkpoint)
    status,result=webui_api('/api/form/train',{'manifest':str(manifest),'checkpoint_dir':str(tmp_path/'new'),
        'mode':'finetune','init_checkpoint':str(checkpoint)})
    assert status==400 and 'old event checkpoints' in result['error']


def test_new_schema_is_identified_without_relabeling_legacy_models():
    from maichart.form import CHECKPOINT_SCHEMA,FAMILY
    from maichart.form.checkpoint import INPUT_CONTRACT
    from maichart.generation.model_compatibility import checkpoint_capabilities
    caps=checkpoint_capabilities({'schema':CHECKPOINT_SCHEMA,'family':FAMILY,'input_contract':INPUT_CONTRACT})
    assert caps['model_family']==FAMILY and caps['generation_entry']=='/form-lab'
    old=checkpoint_capabilities({'schema':'maichart-event-generator-checkpoint-v1',
        'data_contract':{'target_schema':'event_ctx_v1'}})
    assert old['model_family']=='event_ctx_v1'


def test_form_repair_log_event_is_exposed_on_task_result():
    class Task:
        kind='form_generate'
        result={}

    task=Task()
    form_api.update_form_progress(task,
        'form_repair {"checks":12,"elapsed_secs":10.2,"remaining_actionable_hard":19,"status":"partial_budget_exhausted"}')
    assert task.result['repair_summary']['remaining_actionable_hard']==19
    assert task.result['repair_summary']['status']=='partial_budget_exhausted'


def test_form_local_load_log_retains_head_and_event_peak_units():
    from types import SimpleNamespace
    task=SimpleNamespace(kind='form_generate',result={})
    form_api.update_form_progress(task,
        'form_local_load {"peak_all_head_nps":20,"peak_occupied_events_nps":14}')
    assert task.result['local_load_summary']['peak_all_head_nps']==20
    assert task.result['local_load_summary']['peak_occupied_events_nps']==14


def test_none_control_profile_label_matches_its_two_zeroed_training_weights(webui_api,monkeypatch,tmp_path):
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    calls=[]
    monkeypatch.setattr(webui,'_run_subprocess_task',lambda task,args,**kwargs:calls.append(args))
    status,result=webui_api('/api/form/train',{'manifest':str(manifest),'checkpoint_dir':str(tmp_path/'new'),
        'mode':'fresh','control_profile':'none','max_steps':1,'limit_charts':1,'epochs':1,'batch_size':1,
        'd_model':32,'nhead':2,'layers':1,'device':'cpu'})
    assert status==200,result
    from maichart.form.train import parser
    args=parser().parse_args(calls[0][2:])
    assert args.condition_drop_prob==0
    assert args.load_curve_loss_weight==0
    html=(Path(__file__).parents[1]/'src/maichart/webui/static/index.html').read_text(encoding='utf-8')
    assert '<option value="none">关闭条件省略与曲线预测损失</option>' in html
    assert 'Radar / Style 目标、曲线输入和布局训练仍保留' in html
