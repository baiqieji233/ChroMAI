"""Independent experimental endpoints without changing legacy task contracts."""
from __future__ import annotations

from fractions import Fraction
import json
import math
from pathlib import Path

from maichart.form.timing import normalize_meter_map
from maichart.generation.timeline import normalize_tempo_map


def start_form_task(handler,operation,payload):
    from maichart.webui import server
    if operation not in {'analyze','train','generate','revise'}: raise ValueError('Unknown song-form operation')
    def number(key,default,minimum,maximum,integer=False):
        value=payload.get(key,default)
        if isinstance(value,bool): raise ValueError(f'{key} must be numeric')
        numeric=float(value)
        if not math.isfinite(numeric) or not minimum<=numeric<=maximum: raise ValueError(f'{key} out of range')
        if integer and not numeric.is_integer(): raise ValueError(f'{key} must be an integer')
        return int(numeric) if integer else numeric
    def file(key,required=True):
        text=str(payload.get(key) or '').strip()
        if not text:
            if required: raise ValueError(f'{key} is required')
            return None
        path=server._resolve_path(text)
        if not path.is_file(): raise ValueError(f'{key} file not found: {path}')
        return path
    device=str(payload.get('device','cpu'))
    if device not in {'cpu','cuda'}: raise ValueError('device must be cpu or cuda')
    region=number('max_region_seconds',14,2,20)
    meters=normalize_meter_map(payload.get('meter_map'))
    tempo=normalize_tempo_map(payload['tempo_map']) if payload.get('tempo_map') else None
    if operation=='train':
        manifest=file('manifest')
        mode=payload.get('mode','fresh')
        if mode not in {'fresh','resume','finetune'}: raise ValueError('Unknown training mode')
        initial=file('init_checkpoint',required=mode!='fresh') if mode!='fresh' else None
        if initial:
            from maichart.form.checkpoint import load
            load(initial)
        output_value=str(payload.get('checkpoint_dir') or '').strip()
        if not output_value: raise ValueError('checkpoint_dir is required')
        out=server._resolve_path(output_value)
        epochs=number('epochs',10,1,10000,True)
        values={"epochs":epochs,"batch_size":number('batch_size',2,1,512,True),
                'lr':number('lr',.0002,1e-8,1.),'max_steps':number('max_steps',0,0,1000000,True),
                'limit_charts':number('limit_charts',0,0,1000000,True),
                'd_model':number('d_model',256,16,2048,True),'layers':number('layers',4,1,32,True),
                'nhead':number('nhead',8,1,64,True),'max_tokens':number('max_tokens',2048,32,16384,True),
                'max_memory_tokens':number('max_memory_tokens',512,16,4096,True)}
        if values['d_model']%values['nhead']: raise ValueError('d_model must be divisible by nhead')
        args=['-m','maichart.form.train','--manifest',str(manifest),'--checkpoint-dir',str(out)]
        for key,value in values.items():args+=['--'+key.replace('_','-'),str(value)]
        if initial: args+=['--resume' if mode=='resume' else '--init-checkpoint',str(initial)]
        results=[out/'form_last.pt',out/'dataset_audit.json',out/'training_history.json']
        expected=None
    else:
        audio=file('audio')
        initial=file('checkpoint') if operation!='analyze' else None
        if initial:
            from maichart.form.checkpoint import load
            load(initial,inference=True,task='revision' if operation=='revise' else 'draft')
        bpm=number('bpm',120,1,1000);first=number('first',0,-3600,3600)
        args=['-m','maichart.form.cli',operation,'--audio',str(audio),'--bpm',str(bpm),'--first',str(first)]
        if initial: args+=['--checkpoint',str(initial)]
        output_value=str(payload.get('output') or '').strip()
        if operation!='analyze' and not output_value: raise ValueError('output path is required')
        out=server._resolve_path(output_value) if output_value else None
        if operation!='analyze':
            values={'level':number('level',13.3,1,15),'memory_strength':number('memory_strength',1,0,1),
                'onset_threshold':number('onset_threshold',.5,.001,.999),'temperature':number('temperature',.7,0,3),
                'seed':number('seed',0,0,2**31-1,True)}
            for key,value in values.items():args+=['--'+key.replace('_','-'),str(value)]
        if operation=='revise':
            source=file('maidata_path');start=Fraction(str(payload.get('editable_start','')));end=Fraction(str(payload.get('editable_end','')))
            if start<0 or end<=start: raise ValueError('Edit range must be increasing nonnegative beats')
            if out.resolve()==source.resolve(): raise ValueError('Revision output must preserve the source maidata file')
            args+=['--maidata',str(source),'--difficulty',str(number('difficulty_index',5,1,6,True)),
                   '--start',str(start),'--end',str(end)]
        if payload.get('repair'):
            args+=['--repair','--repair-checks',str(number('max_checks',24,1,1000,True)),
                   '--repair-edits',str(number('max_edits',4,0,100,True)),
                   '--repair-seconds',str(number('deadline_seconds',10,0,600))]
        expected=None
    task=server._create_task('form_'+operation)
    workspace=server.WEBUI_ROOT/'song_form'/task.task_id;workspace.mkdir(parents=True,exist_ok=True)
    if out is None:out=workspace/'analysis.json'
    if operation!='train':
        args+=['--output',str(out)]
        results=[out] if operation=='analyze' else [out,out.with_suffix('.form.json')]
    args+=['--device',device,'--max-region-seconds',str(region)]
    if meters:
        path=workspace/'meter_map.json';path.write_text(json.dumps([m.to_dict() for m in meters]),encoding='utf-8')
        args+=['--meter-map',str(path)]
    if tempo and operation!='train':
        path=workspace/'tempo_map.json';path.write_text(json.dumps([{'beat':str(t.beat),'bpm':t.bpm} for t in tempo]),encoding='utf-8')
        args+=['--tempo-map',str(path)]
    report=results[-1]
    task.result.update({'model_family':'song_form_v1','experimental':True,'report_path':str(report),
                        'artifacts':[{'path':str(p),'label':p.name} for p in results]})
    server._run_subprocess_task(task,args,expected_epoch=expected,result_paths=results)
    return handler._json_response({'task_id':task.task_id})
