"""Independent experimental endpoints without changing legacy task contracts."""
from __future__ import annotations

from fractions import Fraction
import json
import math
from pathlib import Path
import secrets

from maichart.form.timing import normalize_meter_map
from maichart.generation.timeline import normalize_tempo_map
from maichart.form import FAMILY, CHECKPOINT_SCHEMA, FORM_FAMILIES
from maichart.form.rhythm_snap import COMMON_DIVISIONS, normalize_divisions
from maichart.form.training_input_perturbation import QUALITY_TRAINING_DEFAULTS


RESUME_ARGUMENT_KEYS = ('batch_size', 'grad_accumulation', 'seed', 'weight_decay',
                       'condition_drop_prob', 'blueprint_loss_weight',
                       'load_curve_loss_weight', 'relation_loss_weight',
                       'quality_enabled', 'max_steps', 'blueprint_batch_size','event_decision_loss_weight',
                       'rhythm_loss_weight', *QUALITY_TRAINING_DEFAULTS)


def form_resume_settings(payload):
    args = payload.get('args') or {}
    contract = payload.get('dataset_contract') or {}
    groups = (payload.get('optimizer_state_dict') or {}).get('param_groups') or []
    values = {key: args[key] for key in RESUME_ARGUMENT_KEYS if key in args}
    values.update({key: args.get(key, default)
                   for key, default in QUALITY_TRAINING_DEFAULTS.items()})
    if payload.get('family') == 'song_form_v2':
        values['blueprint_batch_size'] = args.get('blueprint_batch_size') or 1
        values['event_decision_loss_weight'] = args.get('event_decision_loss_weight',0.)
    values.update(lr=groups[0].get('lr', args.get('lr')) if groups else args.get('lr'),
                  limit_charts=args.get('limit_charts', 0),
                  max_region_seconds=contract.get('max_region_seconds', 14),
                  meter_map=contract.get('meter_map'),
                  precision=payload.get('precision') or args.get('precision', 'fp32'))
    if 'dropout' in (payload.get('model_config') or {}):
        values['dropout'] = payload['model_config']['dropout']
    return values


def isolated_form_output(requested, task_id):
    """Keep each task's chart/report pair immutable while retaining filenames."""
    path = Path(requested)
    return path.parent / task_id / path.name


def resolve_rhythm_geometry_training_options(payload, *, family, mode, inspected):
    """Resolve browser intent without changing a saved resume contract."""
    requested = str(payload.get('rhythm_geometry_profile', 'auto')).lower()
    if requested not in {'auto', 'none', 'exact_fraction_v1'}:
        raise ValueError('rhythm_geometry_profile must be auto, none, or exact_fraction_v1')
    config = (inspected or {}).get('model_config') or {}
    stored_profile = str(config.get('rhythm_geometry_profile', 'none'))
    if stored_profile not in {'none', 'exact_fraction_v1'}:
        raise ValueError('checkpoint has an unknown rhythm geometry profile')
    saved = (inspected or {}).get('args') or {}
    if family != 'song_form_v2':
        profile = 'none'
    elif mode == 'resume':
        profile = stored_profile
    elif mode == 'finetune':
        # Old checkpoints migrate only after the user explicitly asks for it.
        profile = 'exact_fraction_v1' if requested == 'exact_fraction_v1' else stored_profile
    else:
        profile = 'exact_fraction_v1' if requested == 'auto' else requested

    raw_weight = payload.get('rhythm_loss_weight')
    if mode == 'resume':
        weight=saved.get('rhythm_loss_weight',0.)
    elif profile=='none':
        weight=0.
    elif raw_weight in {None, ''}:
        if mode == 'finetune' and profile == stored_profile:
            weight = saved.get('rhythm_loss_weight', 0.)
        else:
            weight = .1 if profile == 'exact_fraction_v1' else 0.
    else:
        if isinstance(raw_weight, bool):
            raise ValueError('rhythm_loss_weight must be numeric')
        weight = float(raw_weight)
    if not math.isfinite(weight) or not 0 <= weight <= 10:
        raise ValueError('rhythm_loss_weight must be between 0 and 10')
    return profile, weight


def inspect_form_checkpoint(path):
    """Inspect this family's actual weights, without running training or inference."""
    from maichart.webui import server
    from maichart.generation.model_compatibility import checkpoint_capabilities
    payload = server._load_checkpoint_payload(path)
    caps = checkpoint_capabilities(payload)
    detected = caps['model_family']
    base = {'path':str(path), 'schema':payload.get('schema'), 'model_family':detected,
            'valid':False, 'can_train':False, 'can_generate':False, 'can_revise':False,
            'model_config':{}, 'args':{}, 'errors':[], 'route':'/'}
    if detected not in FORM_FAMILIES:
        base.update(route='/', errors=['该 checkpoint 不属于 song-form；请在主页面切换对应旧事件模型族。'])
        return base
    try:
        from maichart.form.checkpoint import load,task_is_ready,planner_is_ready
        # This also rejects partial/corrupt model weights and input contracts.
        model, planner, _, checked = load(path, device='cpu')
        trained = checked.get('task_training') or {}
        capabilities = checked.get('capabilities') or {}
        args = checked.get('args') or {}
        resume_settings = form_resume_settings(checked)
        plan_ready = planner_is_ready(checked)
        condition_cap = capabilities.get('draft') or {}
        curve_cap = capabilities.get('load_curve') or {}
        condition_ready = (condition_cap.get('contract') == 'form_condition_v2'
                           and int(condition_cap.get('steps', 0)) > 0
                           and checked.get('condition_supervision') == 'event_derived_radar_style_v1')
        curve_ready = curve_cap.get('contract') == 'load_curve_v1' and int(curve_cap.get('steps', 0)) > 0
        base.update(valid=True, can_train=True,
            can_generate=task_is_ready(checked,'draft'),
            can_revise=task_is_ready(checked,'revision'),
            epoch=checked.get('epoch',0), task_training=trained,
            blueprint_steps=checked.get('blueprint_steps',0), optimizer_steps=checked.get('optimizer_steps',0),
            planner_trained=plan_ready,
            planner_effective_steps=checked.get('planner_effective_steps'),
            planner_training_status=('trained' if plan_ready else 'untrained_workload'),
            model_config=checked['model_config'], args=checked.get('args') or {},
            dataset_contract=checked.get('dataset_contract') or {},
            resume_settings=resume_settings,
            condition_supervision=checked.get('condition_supervision'),
            capabilities=capabilities,
            condition_controls_trained=condition_ready,
            load_curve_trained=curve_ready,
            input_contract=checked['input_contract'],
            rhythm_geometry_profile=checked['model_config'].get('rhythm_geometry_profile', 'none'),
            rhythm_geometry_available=detected == 'song_form_v2',
            parameter_count=sum(p.numel() for p in model.parameters())+sum(p.numel() for p in planner.parameters()))
        base['blueprint_config']=checked['blueprint_config']
        base['generation_capabilities']={'rational_event_positions':detected=='song_form_v2',
            'frame_onset_threshold':detected=='song_form_v1','planner_type_guidance':detected=='song_form_v1',
            'style_balance':True,'local_chord_guard':True,'conflict_detection':True}
    except (ValueError, RuntimeError, KeyError, TypeError) as error:
        base['errors']=[str(error)]
    return base


def update_form_progress(task, line):
    """Consume structured counters; unknown preparation work has no fake percent."""
    if not task.kind.startswith('form_'):
        return
    prefix, _, body = line.partition(' ')
    if prefix not in {'form_prepare','form_progress','form_epoch','form_arrange','form_checkpoint','form_audio_cache','form_repair','form_onsets','form_local_load','form_model_summary','form_rational_diagnostics'}:
        return
    try:
        value = json.loads(body)
    except (ValueError, TypeError):
        return
    if not isinstance(value,dict):
        return
    if prefix == 'form_rational_diagnostics':
        task.result['rational_diagnostics'] = value
    elif prefix == 'form_prepare':
        task.result.update(phase='prepare',progress_known=False,progress_detail=value)
    elif prefix == 'form_progress':
        task.result.update(phase=value.get('phase'),progress_detail=value,progress_known=True)
        if task.kind == 'form_train' and isinstance(value.get('losses'), dict):
            snapshots = task.result.setdefault('training_metrics', {})
            snapshots[value.get('phase', 'unknown')] = value
        target = task.result.get('epoch_target')
        epoch = value.get('epoch')
        fraction = value.get('overall_fraction')
        if target and epoch is not None and isinstance(fraction,(float,int)):
            task.progress = min(.99,max(0.,(int(epoch)-1+float(fraction))/int(target)))
    elif prefix == 'form_epoch':
        task.result.update(last_epoch=value.get('epoch'),metrics=value,phase='saved',progress_known=True)
        history = task.result.setdefault('metric_history', [])
        history[:] = [item for item in history if item.get('epoch') != value.get('epoch')]
        history.append(value)
        del history[:-20]
        if task.result.get('epoch_target'):
            task.progress=min(.99,float(value.get('epoch',0))/task.result['epoch_target'])
    elif prefix == 'form_checkpoint':
        task.result.update(last_checkpoint=value,phase='checkpoint',progress_detail=value,progress_known=True)
    elif prefix == 'form_audio_cache':
        task.result.update(phase='audio_cache',progress_detail=value,progress_known=True)
    elif prefix == 'form_repair':
        task.result.update(repair_summary=value)
    elif prefix == 'form_onsets':
        task.result.update(onset_summary=value)
    elif prefix == 'form_local_load':
        task.result.update(local_load_summary=value)
    elif prefix == 'form_model_summary':
        task.result.update(model_summary=value)
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
    if payload.get('model_family', FAMILY) not in FORM_FAMILIES:
        raise ValueError('此入口接受 song_form_v1 / song_form_v2；其他模型族使用原训练/生成入口。')
    rhythm_snap_value = payload.get('rhythm_snap_enabled', False)
    rhythm_snap_enabled = (rhythm_snap_value is True
                           or (type(rhythm_snap_value) is int and rhythm_snap_value == 1)
                           or (isinstance(rhythm_snap_value, str)
                               and rhythm_snap_value.lower() in {'true','on','yes'}))
    if operation == 'revise' and rhythm_snap_enabled:
        raise ValueError('近似到常见分音目前只支持 song_form_v1 自动生成，修订入口不支持。')
    old_controls = {'segment_beats','context_beats','context_tokens','audio_margin_beats',
                    'planner_profile','condition_mask_prob','condition_dropout','guidance_scale',
                    'slide_oversample','touch_oversample','fine_rhythm_oversample'}
    invalid = old_controls & payload.keys()
    if invalid:
        raise ValueError('song-form 不使用这些旧模型参数：' + ', '.join(sorted(invalid)))
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
        resume_data_policy=str(payload.get('resume_data_policy','strict'))
        if resume_data_policy not in {'strict','restart_epoch'}:
            raise ValueError('resume_data_policy must be strict or restart_epoch')
        if mode != 'resume' and resume_data_policy != 'strict':
            raise ValueError('迁移续训选项仅用于“从断点继续（resume）”。')
        initial=file('init_checkpoint',required=mode!='fresh') if mode!='fresh' else None
        inspected = None
        if initial:
            from maichart.form.checkpoint import load
            _, _, _, inspected = load(initial)
            if inspected['family']!=payload.get('model_family',FAMILY):
                raise ValueError('初始化模型属于 '+inspected['family']+'；请切换对应模型族续训，新有理数架构需从头训练。')
        if inspected and mode == 'resume':
            # Resume is contract preserving. Inherit these before validating UI
            # fields because the browser may submit stale or disabled values.
            old_args = inspected.get('args') or {}
            old_contract = inspected.get('dataset_contract') or {}
            payload = {**payload, **{key: old_args[key] for key in RESUME_ARGUMENT_KEYS if key in old_args}}
            payload.update({key: old_args.get(key, default)
                            for key, default in QUALITY_TRAINING_DEFAULTS.items()})
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
        if mode == 'resume':
            from maichart.form.resume_contract import contract_header, preflight_resume, verify_saved_chart_sources
            preflight_resume(inspected.get('dataset_contract') or {},
                             contract_header(manifest,(inspected.get('dataset_contract') or {}).get('meter_map'),region))
            verify_saved_chart_sources(inspected.get('dataset_contract') or {})
            if resume_data_policy == 'restart_epoch' and out.resolve() == initial.resolve().parent:
                raise ValueError('迁移续训请设置新的 checkpoint 输出目录，以保留来源模型。')
        epochs=number('epochs',10,1,10000,True)
        family=(inspected or {}).get('family',payload.get('model_family',FAMILY))
        rhythm_geometry_profile, rhythm_loss_weight = resolve_rhythm_geometry_training_options(
            payload, family=family, mode=mode, inspected=inspected)
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
                'relation_loss_weight':number('relation_loss_weight',.1,0,100),
                'rhythm_geometry_profile':rhythm_geometry_profile,
                'rhythm_loss_weight':rhythm_loss_weight,
                # Input perturbation and auxiliary objectives. API callers that
                # omit them keep the legacy off values; the main WebUI submits
                # its own recommended defaults explicitly.
                'perturb_onset_drop_prob':number('perturb_onset_drop_prob',0,0,0.5),
                'perturb_onset_add_prob':number('perturb_onset_add_prob',0,0,0.5),
                'perturb_onset_shift_ms':number('perturb_onset_shift_ms',0,0,50),
                'perturb_curve_noise_std':number('perturb_curve_noise_std',0,0,0.3),
                'perturb_curve_dropout_prob':number('perturb_curve_dropout_prob',0,0,0.5),
                'perturb_history_corruption_prob':number('perturb_history_corruption_prob',0,0,0.3),
                'perturb_apply_prob':number('perturb_apply_prob',1.0,0,1.0),
                'repetition_penalty_weight':number('repetition_penalty_weight',0,0,0.5),
                'diversity_loss_weight':number('diversity_loss_weight',0,0,0.3)}
        if values['repetition_penalty_weight']:
            if mode != 'resume':
                raise ValueError('重复惩罚损失尚未接入可训练目标，请保持 0。')
            # It never produced a gradient, so resuming with 0 is numerically identical.
            values['repetition_penalty_weight'] = 0.
        if payload.get('blueprint_batch_size') not in {None, ''}:
            values['blueprint_batch_size'] = number('blueprint_batch_size',4,1,16,True)
        values['blueprint_workers'] = number('blueprint_workers',0,0,4,True)
        values['blueprint_prefetch'] = number('blueprint_prefetch',8,1,64,True)
        if (inspected or {}).get('family',payload.get('model_family',FAMILY))=='song_form_v2':
            values['event_decision_loss_weight']=number('event_decision_loss_weight',.25,0,10)
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
        values['model_family']=family
        values['dim_feedforward']=(inspected or {}).get('model_config',{}).get('dim_feedforward',number('dim_feedforward',0,0,32768,True))
        values['planner_width']=(inspected or {}).get('blueprint_config',{}).get('width',number('planner_width',128,16,2048,True))
        if inspected:
            config = inspected['model_config']
            for key in ('d_model','layers','nhead','max_tokens','max_memory_tokens','dropout'):
                values[key] = config[key]
        else:
            from types import SimpleNamespace
            from maichart.form.train import fresh_model_configs
            fresh_model_configs(SimpleNamespace(**values))
        args=['-m','maichart.form.train','--manifest',str(manifest),'--checkpoint-dir',str(out)]
        for key,value in values.items():
            if key == 'quality_enabled':
                continue
            if value is None:
                continue
            args+=['--'+key.replace('_','-'),str(value)]
        if mode != 'resume' and str(payload.get('control_profile','auto')) == 'none':
            values['condition_drop_prob'] = 0
            args[args.index('--condition-drop-prob') + 1] = '0'
            # v2 always feeds its workload Planner into joint event decoding.
            # Disabling optional external controls cannot disable that core
            # supervision. An explicit zero still freezes a pretrained Planner.
            if family != 'song_form_v2':
                values['load_curve_loss_weight'] = 0
                args[args.index('--load-curve-loss-weight') + 1] = '0'
        if family == 'song_form_v2':
            values['planner_objective_enabled'] = (values['blueprint_loss_weight'] > 0
                                                   and values['load_curve_loss_weight'] > 0)
        args += ['--quality-enabled' if values['quality_enabled'] else '--no-quality-enabled']
        if mode == 'resume':
            args += ['--resume-data-policy',resume_data_policy]
            values['resume_data_policy']=resume_data_policy
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
        family=payload.get('model_family',FAMILY)
        if initial:
            from maichart.form import checkpoint as checkpoint_module
            imported_result = checkpoint_module.load(
                initial,inference=True,task='revision' if operation=='revise' else 'draft')
            if imported_result is not None:
                _,_,_,imported=imported_result
                family=imported['family']
        bpm_value = payload.get('bpm')
        bpm = number('bpm',120,1,1000) if bpm_value not in {None, ''} else None
        first=number('first',0,-3600,3600)
        args=['-m','maichart.form.cli',operation,'--audio',str(audio),'--model-family',family]
        if family=='song_form_v2' and operation=='generate':
            args+=['--position-temperature',str(number('position_temperature',.2,0,3)),
                   '--position-characters',str(number('position_characters',24,4,128,True)),
                   '--position-complexity',str(number('position_complexity',.2,0,2)),
                   '--position-beam-size',str(number('position_beam_size',1,1,8,True)),
                   '--early-eos-penalty',str(number('early_eos_penalty',0,0,8)),
                   '--max-events-per-span',str(number('max_events_per_span',0,0,512,True)),
                   '--rational-min-spacing-ms',str(number('rational_min_spacing_ms',2,0,50))]
        if bpm is not None: args += ['--bpm',str(bpm)]
        args += ['--first',str(first)]
        if initial: args+=['--checkpoint',str(initial)]
        output_value=str(payload.get('output') or '').strip()
        if operation!='analyze' and not output_value: raise ValueError('output path is required')
        out=server._resolve_path(output_value) if output_value else None
        requested_output = server._resolve_path(str(payload.get('requested_output') or output_value).strip()) if out else None
        if operation!='analyze':
            seed_value = payload.get('seed')
            if seed_value is None or seed_value == '':
                effective_seed = secrets.randbelow(2**31) if operation == 'generate' else 0
            else:
                effective_seed = number('seed',0,0,2**31-1,True)
            values={'level':number('level',13.3,1,15),'memory_strength':number('memory_strength',1,0,1),
                'onset_threshold':number('onset_threshold',.5,.001,.999),'temperature':number('temperature',.7,0,3),
                'onset_merge_ms':number('onset_merge_ms',40,0,250),
                'seed':effective_seed}
            for key,value in values.items():args+=['--'+key.replace('_','-'),str(value)]
            if operation == 'generate':
                planner_type_strength=number('planner_type_strength',1.0,0,3)
                style_balance_strength=number('style_balance_strength',1.0,0,3)
                local_chord_strength=number('local_chord_strength',1.0,0,3)
                rhythm_snap_mode=str(payload.get('rhythm_snap_mode','sequence'))
                if rhythm_snap_mode not in {'sequence','point'}:
                    raise ValueError('rhythm_snap_mode must be sequence or point')
                args += ['--planner-type-strength',str(planner_type_strength),
                         '--style-balance-strength',str(style_balance_strength),
                         '--local-chord-strength',str(local_chord_strength)]
                raw_divisions = payload.get('rhythm_snap_divisions', list(COMMON_DIVISIONS))
                empty_divisions = raw_divisions == [] or (isinstance(raw_divisions, str) and not raw_divisions.strip())
                if rhythm_snap_enabled and empty_divisions:
                    raise ValueError('启用节奏近似时至少选择一种候选分音。')
                divisions = COMMON_DIVISIONS if empty_divisions else normalize_divisions(raw_divisions)
                args += ['--rhythm-snap-ms', str(number('rhythm_snap_ms',15,0,50)),
                         '--rhythm-snap-source-ms', str(number('rhythm_snap_source_ms',30,0,50)),
                         '--rhythm-snap-divisions', ','.join(map(str, divisions))]
                if rhythm_snap_enabled:
                    args += ['--rhythm-snap', '--rhythm-snap-mode', rhythm_snap_mode]
        if operation=='revise':
            source=file('maidata_path');start=Fraction(str(payload.get('editable_start','')));end=Fraction(str(payload.get('editable_end','')))
            if start<0 or end<=start: raise ValueError('Edit range must be increasing nonnegative beats')
            if out.resolve()==source.resolve() or requested_output.resolve()==source.resolve():
                raise ValueError('Revision output must preserve the source maidata file')
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
    curve_to_forward = None
    curve_condition_warnings = []
    if operation not in {'train', 'analyze'} and payload.get('intensity_curve_enabled'):
        curve_to_forward = payload.get('intensity_curve')
        if not isinstance(curve_to_forward, dict) or not curve_to_forward:
            raise ValueError('已启用张力曲线，但曲线内容为空；请重新生成或关闭曲线。')
        if operation == 'revise':
            raise ValueError('song_form 张力曲线只用于 generate；revise 使用源谱面时钟且不应用曲线。')
        from maichart.generation.music_intensity import validate_intensity_profile
        curve_to_forward = validate_intensity_profile(curve_to_forward)
        if curve_to_forward.get('method') != ('learned_song_form_v2' if family=='song_form_v2' else 'learned_song_form_v1'):
            raise ValueError('曲线所属模型架构与当前 checkpoint 不同；请用当前模型生成曲线。')
        from maichart.form.curve_validation import validate_saved_curve
        from maichart.generation.event_cli import _file_identity
        from maichart.generation.conditions import GenerationCondition, GENERATION_DEFAULTS
        effective_condition = (GenerationCondition.from_dict(payload['condition'])
                               if payload.get('condition') else
                               GenerationCondition(level=values['level'], radar=GENERATION_DEFAULTS,
                                                   style=GENERATION_DEFAULTS))
        curve_condition_warnings = validate_saved_curve(
            curve_to_forward, audio_path=audio, checkpoint_path=initial,
            checkpoint_identity=_file_identity(initial), first=first,
            tempo_map=tempo, bpm_input=bpm, condition=effective_condition.to_dict())
    task=server._create_task('form_'+operation)
    if operation in {'generate', 'revise'}:
        out = isolated_form_output(requested_output, task.task_id)
        task.result.update(requested_output=str(requested_output), actual_output=str(out))
    if curve_condition_warnings:
        task.result['curve_condition_warnings'] = curve_condition_warnings
    workspace=server.WEBUI_ROOT/'song_form'/task.task_id;workspace.mkdir(parents=True,exist_ok=True)
    if curve_to_forward is not None:
        mode = str(payload.get('intensity_curve_mode', 'follow'))
        if mode not in {'follow', 'hard'}: raise ValueError('intensity_curve_mode must be follow or hard')
        curve_path = workspace/'intensity_curve.json'
        curve_path.write_text(json.dumps(curve_to_forward, ensure_ascii=False), encoding='utf-8')
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
    task.result.update({'model_family':family,'experimental':True,'report_path':str(report),
                        'progress_known':False,
                        'artifacts':[{'path':str(p),'label':p.name} for p in results]})
    if operation=='train':
        task.result.update(epoch_target=expected,epoch_start=start_epoch,epochs_requested=epochs,
                           last_epoch=start_epoch,phase='prepare',effective_settings=values,
                           checkpoint_dir=str(out),training_history_path=str(out/'training_history.json'),
                           training_model_summary_path=str(out/'training_model_summary.json'),
                           max_region_seconds=region,source_manifest=str(manifest))
        if dataset_selection is not None:
            task.result['dataset_selection'] = dataset_selection
    elif operation == 'generate':
        task.result.update(seed=values['seed'],effective_settings={'seed':values['seed'],'temperature':values['temperature'],
                            'planner_type_strength':planner_type_strength,
                            'style_balance_strength':style_balance_strength,
                            'local_chord_strength':local_chord_strength,
                            'rhythm_snap_enabled':rhythm_snap_enabled,
                            'rhythm_snap_mode':rhythm_snap_mode,
                            'rhythm_snap_ms':number('rhythm_snap_ms',15,0,50),
                            'rhythm_snap_source_ms':number('rhythm_snap_source_ms',30,0,50),
                            'rhythm_snap_divisions':list(divisions)})
    server._run_subprocess_task(task,args,expected_epoch=expected,result_paths=results)
    return handler._json_response({'task_id':task.task_id})
