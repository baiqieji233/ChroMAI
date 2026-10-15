import json
from pathlib import Path
import pytest

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
        'device':'cpu','lr':.0003,'prepare_workers':3,'blueprint_batch_size':2,
        'blueprint_workers':1,'blueprint_prefetch':4})
    assert status==200,result
    from maichart.form.train import parser
    args=parser().parse_args(calls[0][2:])
    assert args.max_steps==2 and args.limit_charts==1 and args.lr==.0003 and args.prepare_workers==3
    assert args.blueprint_batch_size==2 and args.blueprint_workers==1 and args.blueprint_prefetch==4
    assert calls[0][:2]==['-m','maichart.form.train']


def test_blueprint_batch_size_blank_uses_checkpoint_aware_trainer_default(webui_api,monkeypatch,tmp_path):
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    calls=[]
    monkeypatch.setattr(webui,'_run_subprocess_task',lambda task,args,**kwargs:calls.append(args))
    status,result=webui_api('/api/form/train',{'manifest':str(manifest),'checkpoint_dir':str(tmp_path/'new'),
        'mode':'fresh','max_steps':1,'limit_charts':1,'epochs':1,'batch_size':1,
        'd_model':32,'nhead':2,'layers':1,'device':'cpu','blueprint_batch_size':None})
    assert status==200,result
    from maichart.form.train import parser
    args=parser().parse_args(calls[0][2:])
    assert args.blueprint_batch_size is None
    assert args.blueprint_workers==0 and args.blueprint_prefetch==8
    html=(Path(__file__).parents[1]/'src/maichart/webui/static/index.html').read_text(encoding='utf-8')
    assert '全曲规划批大小' in html and '规划准备线程' in html and '规划预取谱面数' in html


def test_rational_decoding_controls_reach_the_actual_cli(webui_api,monkeypatch,tmp_path):
    from types import SimpleNamespace
    from maichart.form import checkpoint
    audio=tmp_path/'audio.wav';audio.write_bytes(b'fixture')
    model_path=tmp_path/'model.pt';model_path.write_bytes(b'fixture')
    config=SimpleNamespace(max_tokens=2048,max_memory_tokens=640)
    monkeypatch.setattr(checkpoint,'load',lambda *_a,**_k:(SimpleNamespace(config=config),None,None,
        {'family':'song_form_v2','model_config':{},'dataset_contract':{'max_region_seconds':20}}))
    calls=[]
    monkeypatch.setattr(webui,'_run_subprocess_task',lambda task,args,**kwargs:calls.append(args))
    status,result=webui_api('/api/form/generate',{'audio':str(audio),'checkpoint':str(model_path),
        'model_family':'song_form_v2','output':str(tmp_path/'maidata.txt'),
        'position_complexity':.4,'position_beam_size':4,'early_eos_penalty':2})
    assert status==200,result
    from maichart.form.cli import parser
    args=parser().parse_args(calls[0][2:])
    assert args.position_complexity==.4 and args.position_beam_size==4 and args.early_eos_penalty==2


def test_rhythm_geometry_controls_reach_form_training_cli_with_fresh_auto_default(webui_api, monkeypatch, tmp_path):
    manifest = tmp_path / 'manifest.json'; manifest.write_text('{}')
    calls = []
    monkeypatch.setattr(webui, '_run_subprocess_task', lambda task, args, **kwargs: calls.append(args))
    status, result = webui_api('/api/form/train', {
        'manifest': str(manifest), 'checkpoint_dir': str(tmp_path / 'new'),
        'mode': 'fresh', 'model_family': 'song_form_v2', 'device': 'cpu',
        'rhythm_geometry_profile': 'auto', 'rhythm_loss_weight': None,
    })
    assert status == 200, result
    from maichart.form.train import parser
    args = parser().parse_args(calls[0][2:])
    assert args.rhythm_geometry_profile == 'exact_fraction_v1'
    assert args.rhythm_loss_weight == .1


def test_rhythm_geometry_rejects_invalid_ui_profile_before_dispatch(webui_api, monkeypatch, tmp_path):
    manifest = tmp_path / 'manifest.json'; manifest.write_text('{}')
    calls = []
    monkeypatch.setattr(webui, '_run_subprocess_task', lambda task, args, **kwargs: calls.append(args))
    status, result = webui_api('/api/form/train', {
        'manifest': str(manifest), 'checkpoint_dir': str(tmp_path / 'new'),
        'mode': 'fresh', 'model_family': 'song_form_v2', 'device': 'cpu',
        'rhythm_geometry_profile': 'grid_16',
    })
    assert status == 400
    assert 'rhythm_geometry_profile' in result['error']
    assert not calls


@pytest.mark.parametrize('workers', [0, 1, 4, 5, 8])
def test_blueprint_worker_bounds_match_cli_and_visible_control(webui_api, monkeypatch, tmp_path, workers):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{}')
    calls = []
    monkeypatch.setattr(webui, '_run_subprocess_task', lambda task, args, **kwargs: calls.append(args))
    status, result = webui_api('/api/form/train', {
        'manifest': str(manifest), 'checkpoint_dir': str(tmp_path / 'new'),
        'mode': 'fresh', 'model_family': 'song_form_v2', 'device': 'cpu',
        'blueprint_workers': workers,
    })
    if workers > 4:
        assert status == 400 and 'blueprint_workers' in result['error']
        assert not calls
    else:
        assert status == 200, result
        from maichart.form.train import parser, _validate
        args = parser().parse_args(calls[0][2:])
        _validate(args)
        assert args.blueprint_workers == workers
    html = (Path(__file__).parents[1] / 'src/maichart/webui/static/index.html').read_text(encoding='utf-8')
    assert 'id="formBlueprintWorkers" type="number" min="0" max="4"' in html


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
    assert caps['model_family']==FAMILY and caps['generation_entry']=='/'
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


def test_none_control_profile_label_describes_family_specific_training_weights(webui_api,monkeypatch,tmp_path):
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
    assert '<option value="none">关闭条件随机省略（v2 保留必要规划学习）</option>' in html
    assert 'Radar / Style 目标、曲线输入和布局训练仍保留' in html


def test_rational_none_control_profile_preserves_required_workload_planner(webui_api,monkeypatch,tmp_path):
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    calls=[]
    monkeypatch.setattr(webui,'_run_subprocess_task',lambda task,args,**kwargs:calls.append((task,args)))
    status,result=webui_api('/api/form/train',{'manifest':str(manifest),'checkpoint_dir':str(tmp_path/'new'),
        'model_family':'song_form_v2','mode':'fresh','control_profile':'none','max_steps':1,
        'limit_charts':1,'epochs':1,'batch_size':1,'d_model':32,'nhead':2,'layers':1,
        'device':'cpu','load_curve_loss_weight':.4})
    assert status==200,result
    from maichart.form.train import parser
    task,command=calls[0]
    args=parser().parse_args(command[2:])
    assert args.condition_drop_prob==0
    assert args.load_curve_loss_weight==.4
    assert task.result['effective_settings']['planner_objective_enabled'] is True
