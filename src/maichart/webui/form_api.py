"""Independent experimental endpoints without changing legacy task contracts."""
from __future__ import annotations

from fractions import Fraction
import json
import math
from pathlib import Path

from maichart.form.timing import normalize_meter_map
from maichart.generation.timeline import normalize_tempo_map
from maichart.form import FAMILY, CHECKPOINT_SCHEMA


def inspect_form_checkpoint(path):
    """Inspect this family's actual weights, without running training or inference."""
    from maichart.webui import server
    from maichart.generation.model_compatibility import checkpoint_capabilities
    payload = server._load_checkpoint_payload(path)
    caps = checkpoint_capabilities(payload)
    detected = caps['model_family']
    base = {'path':str(path), 'schema':payload.get('schema'), 'model_family':detected,
            'valid':False, 'can_train':False, 'can_generate':False, 'can_revise':False,
            'model_config':{}, 'args':{}, 'errors':[], 'route':'/form-lab'}
    if detected != FAMILY:
        base.update(route='/', errors=['该 checkpoint 不属于 song_form_v1；已有事件模型请回兼容模型工作台使用。'])
        return base
    try:
        from maichart.form.checkpoint import load
        # This also rejects partial/corrupt model weights and input contracts.
        model, planner, _, checked = load(path, device='cpu')
        trained = checked.get('task_training') or {}
        capabilities = checked.get('capabilities') or {}
        args = checked.get('args') or {}
        optimizer_state = checked.get('optimizer_state_dict') or {}
        param_groups = optimizer_state.get('param_groups') or []
        resume_settings = {
            'lr': (param_groups[0].get('lr') if param_groups else args.get('lr')),
            'limit_charts': args.get('limit_charts', 0),
            'max_region_seconds': (checked.get('dataset_contract') or {}).get('max_region_seconds'),
            'meter_map': (checked.get('dataset_contract') or {}).get('meter_map'),
        }
        plan_ready = int(checked.get('blueprint_steps',0)) > 0
        condition_cap = capabilities.get('draft') or {}
        curve_cap = capabilities.get('load_curve') or {}
        condition_ready = (condition_cap.get('contract') == 'form_condition_v2'
                           and int(condition_cap.get('steps', 0)) > 0
                           and checked.get('condition_supervision') == 'event_derived_radar_style_v1')
        curve_ready = curve_cap.get('contract') == 'load_curve_v1' and int(curve_cap.get('steps', 0)) > 0
        base.update(valid=True, can_train=True,
            can_generate=plan_ready and int(trained.get('draft',0))>0 and int(checked.get('optimizer_steps',0))>0,
            can_revise=plan_ready and int(trained.get('revision',0))>0 and int(checked.get('optimizer_steps',0))>0,
            epoch=checked.get('epoch',0), task_training=trained,
            blueprint_steps=checked.get('blueprint_steps',0), optimizer_steps=checked.get('optimizer_steps',0),
            model_config=checked['model_config'], args=checked.get('args') or {},
            dataset_contract=checked.get('dataset_contract') or {},
            resume_settings=resume_settings,
            condition_supervision=checked.get('condition_supervision'),
            capabilities=capabilities,
            condition_controls_trained=condition_ready,
            load_curve_trained=curve_ready,
            input_contract=checked['input_contract'],
            parameter_count=sum(p.numel() for p in model.parameters())+sum(p.numel() for p in planner.parameters()))
    except (ValueError, RuntimeError, KeyError, TypeError) as error:
        base['errors']=[str(error)]
    return base


def update_form_progress(task, line):
    """Consume structured counters; unknown preparation work has no fake percent."""
    if not task.kind.startswith('form_'):
        return
    prefix, _, body = line.partition(' ')
    if prefix not in {'form_prepare','form_progress','form_epoch','form_arrange','form_checkpoint','form_audio_cache'}:
        return
    try:
        value = json.loads(body)
    except (ValueError, TypeError):
        return
    if not isinstance(value,dict):
        return
    if prefix == 'form_prepare':
        task.result.update(phase='prepare',progress_known=False,progress_detail=value)
    elif prefix == 'form_progress':
        task.result.update(phase=value.get('phase'),progress_detail=value,progress_known=True)
        target = task.result.get('epoch_target')
        epoch = value.get('epoch')
        fraction = value.get('overall_fraction')
        if target and epoch is not None and isinstance(fraction,(float,int)):
            task.progress = min(.99,max(0.,(int(epoch)-1+float(fraction))/int(target)))
    elif prefix == 'form_epoch':
        task.result.update(last_epoch=value.get('epoch'),metrics=value,phase='saved',progress_known=True)
        if task.result.get('epoch_target'):
            task.progress=min(.99,float(value.get('epoch',0))/task.result['epoch_target'])
    elif prefix == 'form_checkpoint':
        task.result.update(last_checkpoint=value,phase='checkpoint',progress_detail=value,progress_known=True)
    elif prefix == 'form_audio_cache':
        task.result.update(phase='audio_cache',progress_detail=value,progress_known=True)
    else:
        task.result.update(phase='arrangement',progress_detail=value,progress_known=False)


def start_form_task(handler,operation,payload,*,runtime=None):
    if runtime is None:
        from maichart.webui import server
    else:
        # `python -m maichart.webui.server` runs as __main__. Importing the
        # canonical module again here creates a second, invisible task registry.
        server = runtime
    if operation not in {'inspect','analyze','train','generate','revise'}: raise ValueError('Unknown song-form operation')
    if payload.get('model_family', FAMILY) != FAMILY:
        raise ValueError('此入口只接受 song_form_v1；其他模型族使用原训练/生成入口。')
    old_controls = {'segment_beats','context_beats','context_tokens','audio_margin_beats',
                    'planner_profile','condition_mask_prob','condition_dropout','guidance_scale',
                    'slide_oversample','touch_oversample','fine_rhythm_oversample'}
    invalid = old_controls & payload.keys()
    if invalid:
        raise ValueError('song_form_v1 不使用这些旧模型参数：' + ', '.join(sorted(invalid)))
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
    if operation == 'inspect':
        return handler._json_response(inspect_form_checkpoint(file('checkpoint')))
    device=str(payload.get('device','auto'))
    if device == 'auto':
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device not in {'cpu','cuda'}: raise ValueError('device must be auto, cpu or cuda')
    if operation=='train':
        dataset_selection = None
        if payload.get('manifests') and hasattr(server, '_prepare_training_sources'):
            payload, dataset_selection = server._prepare_training_sources(payload)
        manifest=file('manifest')
        mode=payload.get('mode','fresh')
        if mode not in {'fresh','resume','finetune'}: raise ValueError('Unknown training mode')
        initial=file('init_checkpoint',required=mode!='fresh') if mode!='fresh' else None
        inspected = None
        if initial:
            from maichart.form.checkpoint import load
            _, _, _, inspected = load(initial)
        if inspected and mode == 'resume':
            # Resume is contract preserving. Inherit these before validating UI
            # fields because the browser may submit stale or disabled values.
            old_args = inspected.get('args') or {}
            old_contract = inspected.get('dataset_contract') or {}
            inherited = ('batch_size', 'grad_accumulation', 'seed', 'weight_decay',
                         'condition_drop_prob', 'blueprint_loss_weight',
                         'load_curve_loss_weight', 'relation_loss_weight',
                         'quality_enabled', 'max_steps')
            payload = {**payload, **{key: old_args[key] for key in inherited if key in old_args}}
            payload['precision'] = inspected.get('precision') or old_args.get('precision', 'fp32')
            if 'lr' in old_args:
                payload = {**payload, 'lr': old_args['lr']}
            payload = {**payload,
                       'limit_charts': old_args.get('limit_charts', 0),
                       'max_region_seconds': old_contract.get('max_region_seconds', 14),
                       'meter_map': old_contract.get('meter_map')}
            optimizer_state = inspected.get('optimizer_state_dict') or {}
            groups = optimizer_state.get('param_groups') or []
            if groups:
                payload['lr'] = groups[0].get('lr', payload.get('lr', .0002))
            config = inspected.get('model_config') or {}
            payload = {**payload, **{key: config[key] for key in
                       ('d_model', 'layers', 'nhead', 'max_tokens') if key in config},
                       'max_memory_tokens': config.get('max_memory_tokens', payload.get('max_memory_tokens', 512))}
        region=number('max_region_seconds',14,2,20)
        meters=normalize_meter_map(payload.get('meter_map'))
        tempo=None
        output_value=str(payload.get('checkpoint_dir') or '').strip()
        if not output_value: raise ValueError('checkpoint_dir is required')
        out=server._resolve_path(output_value)
        epochs=number('epochs',10,1,10000,True)
        values={"epochs":epochs,"batch_size":number('batch_size',2,1,512,True),
                'lr':number('lr',.0002,1e-8,1.),'max_steps':number('max_steps',0,0,1000000,True),
                'limit_charts':number('limit_charts',0,0,1000000,True),
                'prepare_workers':number('prepare_workers',1,0,64,True),
                'd_model':number('d_model',256,16,2048,True),'layers':number('layers',4,1,32,True),
                'nhead':number('nhead',8,1,64,True),'max_tokens':number('max_tokens',2048,32,16384,True),
                'max_memory_tokens':number('max_memory_tokens',512,16,4096,True),
                'dropout':number('dropout',.1,0,.99),
                'seed':number('seed',0,0,2**31-1,True),
                'weight_decay':number('weight_decay',.02,0,10),
                'grad_accumulation':number('grad_accumulation',1,1,1024,True),
                'save_every':number('save_every',0,0,1000000,True),
                'checkpoint_interval_steps':number('checkpoint_interval_steps',500,0,1000000,True),
                'loader_workers':number('loader_workers',0,0,64,True),
                'prefetch_factor':number('prefetch_factor',2,1,16,True),
                'log_interval':number('log_interval',10,0.1,3600),
                'audio_cache_gb':number('audio_cache_gb',1,0,1024),
                'audio_cache_mode':str(payload.get('audio_cache_mode','lru')).lower(),
                'sample_cache_mb':number('sample_cache_mb',256,0,1048576,True),
                'condition_drop_prob':number('condition_drop_prob',.1,0,1),
                'blueprint_loss_weight':number('blueprint_loss_weight',1.,0,100),
                'load_curve_loss_weight':number('load_curve_loss_weight',1.,0,100),
                'relation_loss_weight':number('relation_loss_weight',.1,0,100)}
        values['quality_enabled'] = bool(payload.get('quality_enabled', True))
        if values['audio_cache_mode'] not in {'lru','resident'}:
            raise ValueError('audio_cache_mode must be lru or resident')
        if values['audio_cache_mode'] == 'resident':
            if values['loader_workers'] != 0:
                raise ValueError('全量驻留内存要求数据加载 worker=0，以避免多进程复制音频缓存。')
            if values['audio_cache_gb'] <= 0:
                raise ValueError('全量驻留内存要求音频缓存上限大于 0 GB；该值是硬预算，必须足以容纳全部音频。')
        precision=str(payload.get('precision','auto')).lower()
        if precision not in {'auto','fp32','bf16','fp16'}: raise ValueError('precision must be auto, fp32, bf16 or fp16')
        values['precision']=precision
        if values['d_model']%values['nhead']: raise ValueError('d_model must be divisible by nhead')
        if inspected:
            config = inspected['model_config']
            for key in ('d_model','layers','nhead','max_tokens','max_memory_tokens','dropout'):
                values[key] = config[key]
        args=['-m','maichart.form.train','--manifest',str(manifest),'--checkpoint-dir',str(out)]
        for key,value in values.items():
            if key == 'quality_enabled':
                continue
            args+=['--'+key.replace('_','-'),str(value)]
        if mode != 'resume' and str(payload.get('control_profile','auto')) == 'none':
            values['condition_drop_prob'] = 0
            values['load_curve_loss_weight'] = 0
            args[args.index('--condition-drop-prob') + 1] = '0'
            args[args.index('--load-curve-loss-weight') + 1] = '0'
        args += ['--quality-enabled' if values['quality_enabled'] else '--no-quality-enabled']
        if initial: args+=['--resume' if mode=='resume' else '--init-checkpoint',str(initial)]
        results=[out/'form_last.pt',out/'dataset_audit.json',out/'training_history.json']
        start_epoch=int(inspected.get('epoch',0)) if inspected and mode=='resume' else 0
        expected=start_epoch+epochs
    else:
        region=number('max_region_seconds',14,2,20)
        meters=normalize_meter_map(payload.get('meter_map'))
        tempo=normalize_tempo_map(payload['tempo_map']) if payload.get('tempo_map') else None
        audio=file('audio')
        initial=file('checkpoint') if operation!='analyze' else None
        if initial:
            from maichart.form.checkpoint import load
            load(initial,inference=True,task='revision' if operation=='revise' else 'draft')
        bpm_value = payload.get('bpm')
        bpm = number('bpm',120,1,1000) if bpm_value not in {None, ''} else None
        first=number('first',0,-3600,3600)
        args=['-m','maichart.form.cli',operation,'--audio',str(audio)]
        if bpm is not None: args += ['--bpm',str(bpm)]
        args += ['--first',str(first)]
        if initial: args+=['--checkpoint',str(initial)]
        output_value=str(payload.get('output') or '').strip()
        if operation!='analyze' and not output_value: raise ValueError('output path is required')
        out=server._resolve_path(output_value) if output_value else None
        if operation!='analyze':
            values={'level':number('level',13.3,1,15),'memory_strength':number('memory_strength',1,0,1),
                'onset_threshold':number('onset_threshold',.5,.001,.999),'temperature':number('temperature',.7,0,3),
                'onset_merge_ms':number('onset_merge_ms',40,0,250),
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
    cache_path = None
    if payload.get('cache_dir'):
        cache_path = server._resolve_path(str(payload['cache_dir']))
    elif operation == 'train':
        from maichart.generation.event_audio import audio_cache_dir
        # Parse and validate before registering a task so malformed manifests
        # cannot leave an orphan queued task in the UI.
        manifest_data=json.loads(manifest.read_text(encoding='utf-8-sig'))
        cache=Path(str(manifest_data.get('cache_dir') or 'cache'))
        if not cache.is_absolute(): cache=manifest.parent/cache
        cache_path=audio_cache_dir(cache)
    task=server._create_task('form_'+operation)
    workspace=server.WEBUI_ROOT/'song_form'/task.task_id;workspace.mkdir(parents=True,exist_ok=True)
    if operation != 'train' and payload.get('intensity_curve_enabled') and payload.get('intensity_curve'):
        mode = str(payload.get('intensity_curve_mode', 'follow'))
        if mode not in {'follow', 'hard'}: raise ValueError('intensity_curve_mode must be follow or hard')
        curve_path = workspace/'intensity_curve.json'
        curve_path.write_text(json.dumps(payload['intensity_curve'], ensure_ascii=False), encoding='utf-8')
        args += ['--intensity-curve-file', str(curve_path),
                 '--intensity-curve-strength', str(number('intensity_curve_strength', .65, 0, 1)),
                 '--intensity-curve-span', str(number('intensity_curve_span', .5, 0, 1)),
                 '--intensity-curve-mode', mode]
    if out is None:out=workspace/'analysis.json'
    if operation!='train':
        args+=['--output',str(out)]
        results=[out] if operation=='analyze' else [out,out.with_suffix('.form.json')]
    args+=['--device',device,'--max-region-seconds',str(region)]
    if cache_path is not None:
        args+=['--cache-dir',str(cache_path)]
    if meters:
        path=workspace/'meter_map.json';path.write_text(json.dumps([m.to_dict() for m in meters]),encoding='utf-8')
        args+=['--meter-map',str(path)]
    if tempo and operation!='train':
        path=workspace/'tempo_map.json';path.write_text(json.dumps([{'beat':str(t.beat),'bpm':t.bpm} for t in tempo]),encoding='utf-8')
        args+=['--tempo-map',str(path)]
    if operation != 'analyze' and payload.get('condition'):
        condition_path=workspace/'condition.json'
        condition_path.write_text(json.dumps(payload['condition'],ensure_ascii=False),encoding='utf-8')
        args+=['--condition-file',str(condition_path)]
    report=results[-1]
    task.result.update({'model_family':'song_form_v1','experimental':True,'report_path':str(report),
                        'progress_known':False,
                        'artifacts':[{'path':str(p),'label':p.name} for p in results]})
    if operation=='train':
        task.result.update(epoch_target=expected,epoch_start=start_epoch,epochs_requested=epochs,
                           last_epoch=start_epoch,phase='prepare',effective_settings=values,
                           max_region_seconds=region,source_manifest=str(manifest))
        if dataset_selection is not None:
            task.result['dataset_selection'] = dataset_selection
    server._run_subprocess_task(task,args,expected_epoch=expected,result_paths=results)
    return handler._json_response({'task_id':task.task_id})
