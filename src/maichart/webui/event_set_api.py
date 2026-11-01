"""WebUI subprocess adapter for the independent event-set stream contract."""
from __future__ import annotations

import json, math, os, shutil, uuid

PROFILE = 'event_set_stream_v1'
SCHEMA = 'maichart-composer-event-set-checkpoint-v1'
FAMILY = 'chart_composer_v1'


def checkpoint_profile(path):
    from maichart.composer.checkpoint import read_payload
    try:
        data = read_payload(path)
    except Exception:
        return None
    return PROFILE if data.get('schema') == SCHEMA else (data.get('model_config') or {}).get('architecture_profile')


def start_train(handler, payload, *, runtime):
    from .composer_api import _path, _manifest, _int, _float, _arg, _device, _precision
    if payload.get('model_family') not in {None, FAMILY}: raise ValueError('Wrong Composer model family')
    selection = payload.get('dataset_selection')
    if 'manifests' in payload and selection is None and hasattr(runtime, '_prepare_training_sources'):
        original_cache = payload.get('cache_dir')
        payload, selection = runtime._prepare_training_sources({**payload, 'model_family': FAMILY})
        if original_cache:
            payload['cache_dir'] = original_cache
    mode = payload.get('mode', 'fresh')
    if mode not in {'fresh', 'resume', 'finetune'}:
        raise ValueError('Unknown training mode')
    manifest = _manifest(payload)
    directory = _path(payload.get('checkpoint_dir'), 'checkpoint-dir')
    cache = _path(payload.get('cache_dir') or 'cache/event_set_stream_v1', 'cache-dir')
    parent_data = None
    parent = None
    if mode != 'fresh':
        from maichart.composer.stream_checkpoint import read
        parent = _path(payload.get('resume') or payload.get('init_checkpoint') or payload.get('checkpoint'), 'checkpoint', exists=True)
        parent_data = read(parent)
        if parent_data.get('schema') != SCHEMA:
            raise ValueError('Event-set stream requires a new event-set checkpoint; old Composer weights are incompatible')
    payload = dict(payload)
    parent_args = (parent_data or {}).get('args') or {}
    parent_config = (parent_data or {}).get('model_config') or {}
    if parent_data:
        for key in ('d_model','layers','nhead','dim_feedforward','history_events','rollout_steps','dropout'):
            if key in parent_config: payload[key] = parent_config[key]
    if mode == 'resume':
        for key in ('batch_size','grad_accumulation','seed','symbolic_pretrain_epochs','max_steps',
                    'recovery_interval','recovery_samples','recovery_steps','recovery_weight','plan_weight',
                    'window_seconds','limit_charts','lr','warmup_steps','min_lr_ratio'):
            if key in parent_args: payload[key] = parent_args[key]
        # Reuse the effective split, rather than splitting a saved train subset again.
        if parent_args.get('manifest'):
            saved_manifest = _path(parent_args['manifest'], 'saved manifest', exists=True)
            saved_document = json.loads(saved_manifest.read_text(encoding='utf-8-sig'))
            source = (saved_document.get('split_metadata') or {}).get('source_manifest')
            accepted = {saved_manifest}
            if source: accepted.add(_path(source, 'source manifest'))
            if manifest not in accepted:
                raise ValueError('Resume requires its saved train manifest or original split source; use finetune for new data')
            source_hash = (saved_document.get('split_metadata') or {}).get('source_sha256')
            if source and manifest == _path(source, 'source manifest') and source_hash:
                from maichart.composer.evaluation import file_hash
                if file_hash(manifest) != source_hash:
                    raise ValueError('Resume split source changed; use finetune for new data')
            manifest = saved_manifest
        payload['validation_manifest'] = parent_args.get('validation_manifest')
    fraction = _float(payload, 'holdout_fraction', .05, minimum=0.)
    if fraction >= .5: raise ValueError('holdout_fraction must be less than 0.5')
    split_report = None
    needs_split = mode != 'resume' and not payload.get('validation_manifest') and bool(fraction)
    args = ['-m', 'maichart.composer.stream_train']
    for name, value in [('manifest', manifest), ('checkpoint-dir', directory), ('cache-dir', cache)]:
        _arg(args, '--' + name, value)
    integers = dict(epochs=20, batch_size=16, grad_accumulation=1, d_model=256, layers=4,
        nhead=8, dim_feedforward=768, history_events=64, rollout_steps=16,
        checkpoint_interval_steps=500, log_interval=25, max_steps=0, limit_charts=0,
        prepare_workers=1, cpu_threads=2, seed=0, recovery_interval=25,
        recovery_samples=2, recovery_steps=2, symbolic_pretrain_epochs=0,
        generation_eval_songs=2,validation_max_batches=32,validation_batch_size=4,
        generation_eval_interval_epochs=1,warmup_steps=100,prefetch_batches=2,sample_workers=1)
    settings = {'architecture_profile': PROFILE}
    settings['holdout_fraction'] = fraction
    settings['validation_manifest'] = payload.get('validation_manifest')
    zero = {'max_steps', 'limit_charts', 'prepare_workers', 'cpu_threads', 'seed',
            'recovery_interval', 'symbolic_pretrain_epochs', 'generation_eval_songs', 'checkpoint_interval_steps',
            'validation_max_batches','warmup_steps','prefetch_batches'}
    for key, default in integers.items():
        if parent_data and key in {'d_model','layers','nhead','dim_feedforward','history_events','rollout_steps'}:
            default=(parent_data.get('model_config') or {}).get(key,default)
        settings[key] = _int(payload, key, default, minimum=0 if key in zero else 1)
        _arg(args, '--' + key.replace('_', '-'), settings[key])
    if settings['sample_workers'] != 1:
        raise ValueError('Event-set shared song caches require sample_workers=1')
    if settings['prefetch_batches'] > 16:
        raise ValueError('prefetch_batches cannot exceed 16')
    for key, default in dict(lr=.0002, dropout=.1, window_seconds=8., recovery_weight=.1, plan_weight=.1,
                            audio_cache_gb=2.,min_lr_ratio=.1).items():
        settings[key] = _float(payload, key, default, minimum=0.)
        _arg(args, '--' + key.replace('_', '-'), settings[key])
    if settings['d_model'] % settings['nhead'] or settings['dropout'] > 1 or settings['window_seconds'] <= 0:
        raise ValueError('Invalid event-set model dimensions, dropout or window')
    if settings['min_lr_ratio'] > 1:
        raise ValueError('min_lr_ratio cannot exceed 1')
    schedule = payload.get('lr_schedule', 'cosine')
    if schedule == 'auto': schedule = 'cosine'
    if mode == 'resume': schedule = (parent_data.get('args') or {}).get('lr_schedule','constant')
    if schedule not in {'constant','cosine'}: raise ValueError('Unknown learning rate schedule')
    settings['lr_schedule'] = schedule
    _arg(args,'--lr-schedule',schedule)
    settings['precision'] = _precision(payload.get('precision') or 'bf16')
    settings['device'] = _device(payload.get('device'))
    _arg(args, '--precision', settings['precision']); _arg(args, '--device', settings['device'])
    if payload.get('validation_manifest'):
        _arg(args, '--validation-manifest', _path(payload['validation_manifest'], 'validation manifest', exists=True))
    if payload.get('generation_eval_seeds') is not None:
        seeds = str(payload['generation_eval_seeds']).split(',')
        if not seeds or any(not seed.strip().isascii() or not seed.strip().isdigit() for seed in seeds):
            raise ValueError('generation_eval_seeds requires comma-separated nonnegative integers')
        _arg(args, '--generation-eval-seeds', ','.join(str(int(seed)) for seed in seeds))
    if mode != 'fresh':
        grid_seconds = float((parent_data.get('model_config') or {}).get('audio_grid_seconds', .05))
        blank_seconds = float((parent_data.get('model_config') or {}).get('plan_blank_window_seconds', 0.))
        _arg(args, '--resume' if mode == 'resume' else '--init-checkpoint', parent)
    else:
        grid_seconds = _float(payload, 'audio_grid_seconds', .01, minimum=.005)
        blank_seconds = _float(payload, 'plan_blank_window_seconds', .5, minimum=0.)
    if not .005 <= grid_seconds <= .05:
        raise ValueError('audio_grid_seconds must be between .005 and .05')
    settings['audio_grid_seconds'] = grid_seconds
    _arg(args, '--audio-grid-seconds', grid_seconds)
    if not 0 <= blank_seconds < float('inf'): raise ValueError('Invalid blank window seconds')
    settings['plan_blank_window_seconds'] = blank_seconds
    _arg(args,'--plan-blank-window-seconds',blank_seconds)
    from types import SimpleNamespace
    from maichart.composer.stream_train import LEARNING_POLICIES, LEGACY_LEARNING_POLICIES, resolve_learning_policies
    from maichart.composer.stream_model import StreamComposerConfig
    learning_policy=payload.get('learning_policy','auto')
    if learning_policy not in {'auto','repaired','legacy','checkpoint'}:raise ValueError('Unknown learning policy')
    requested={key:payload.get(key, LEGACY_LEARNING_POLICIES[key] if learning_policy=='legacy' else 'auto') for key,choices in LEARNING_POLICIES.items()}
    requested['transition_exit_weight']=payload.get('transition_exit_weight',0. if learning_policy=='legacy' else None)
    requested['prefix_augmentation_rate']=payload.get('prefix_augmentation_rate',0. if learning_policy=='legacy' else None)
    requested['event_family_audio_weight']=payload.get('event_family_audio_weight',0. if learning_policy=='legacy' else None)
    requested['phrase_loss_weight']=payload.get('phrase_loss_weight')
    parent_config=StreamComposerConfig.from_dict(parent_data['model_config']) if parent_data else None
    if learning_policy=='checkpoint' and parent_config is None:
        raise ValueError('Checkpoint learning policy requires a parent checkpoint; use finetune or resume')
    policies=resolve_learning_policies(SimpleNamespace(**requested),parent_config,
        resume=mode=='resume' or learning_policy=='checkpoint')
    inherited_temperature=(parent_data.get('args') or {}).get('generation_eval_temperature',.65) if parent_data else .65
    temperature=payload.get('generation_eval_temperature')
    if temperature is None:temperature=inherited_temperature if mode=='resume' else .65
    temperature=float(temperature)
    if not math.isfinite(temperature) or temperature<0:raise ValueError('Invalid generation evaluation temperature')
    if mode=='resume' and temperature!=inherited_temperature:raise ValueError('Resume inherits generation_eval_temperature')
    settings['generation_eval_temperature']=temperature
    _arg(args,'--generation-eval-temperature',temperature)
    settings.update(policies)
    for key,value in policies.items():_arg(args,'--'+key.replace('_','-'),value)
    for key, default in dict(search_policy='phrase_v1', physical_policy='runtime_hard_v2',
            phrase_candidates=1, phrase_write_beats=8., phrase_commit_beats=4., phrase_step_budget=256,
            phrase_repair_budget=8, phrase_rollback_budget=4, phrase_time_budget_sec=30.,
            softcomfort_strength=1., audio_tracking_strength=0., pointer_temperature=None,
            action_temperature=None,min_same_head_ms=50.,touch_layout_policy='open',max_events=0).items():
        field = 'generation_eval_' + key
        value = parent_args.get(field, default) if mode == 'resume' else payload.get(field, default)
        settings[field] = value
        if value is not None: _arg(args, '--' + field.replace('_', '-'), value)
    from maichart.composer.training_lock import training_output_lock, training_request_lock
    reservation = training_request_lock(directory)
    reservation.__enter__()
    task = None
    try:
        # The request lock spans split preparation and the asynchronous child.
        # The trainer's own output lock also excludes a direct CLI training run.
        with training_output_lock(directory):
            if mode != 'resume':
                permitted = {'.composer-training.lock', '.composer-training-request.lock'}
                if any(path.name not in permitted for path in directory.iterdir()):
                    raise ValueError('Use a new empty checkpoint directory for fresh/finetune')
            reserve = getattr(runtime, '_reserve_composer_training_task', None)
            task = reserve(directory) if reserve else runtime._create_task('composer_train')
            if needs_split:
                from maichart.composer.evaluation import split_manifest, form_chart_eligibility
                staged = directory / ('.data_split-' + uuid.uuid4().hex)
                final = directory / 'data_split'
                try:
                    split_report = split_manifest(manifest, staged, fraction=fraction,
                        seed=_int(payload, 'seed', 0, minimum=0), cache_dir=cache,
                        chart_eligibility=form_chart_eligibility)
                    for entry in split_report['splits'].values():
                        entry['path'] = str(final / os.path.basename(entry['path']))
                    split_report['report_path'] = str(final / 'split_report.json')
                    (staged / 'split_report.json').write_text(json.dumps(
                        {key:value for key,value in split_report.items() if key != 'report_path'},
                        ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
                    os.replace(staged, final)
                finally:
                    if staged.exists(): shutil.rmtree(staged)
                manifest = _path(split_report['splits']['train']['path'], 'effective train manifest', exists=True)
                payload['validation_manifest'] = split_report['splits']['dev']['path']
                args[args.index('--manifest') + 1] = str(manifest)
                _arg(args, '--validation-manifest', _path(payload['validation_manifest'], 'validation manifest', exists=True))
                settings['validation_manifest'] = payload['validation_manifest']
        task.result.update(family=FAMILY, schema=SCHEMA, mode=mode, manifest=str(manifest),
            checkpoint_dir=str(directory), epoch_target=settings['epochs'], effective_settings=settings,
            settings_policy='resume_inherited' if mode == 'resume' else 'requested')
        if split_report is not None: task.result['data_split'] = split_report
        if selection is not None: task.result['dataset_selection'] = selection
        runtime._run_subprocess_task(task, args, expected_epoch=settings['epochs'],
            result_paths=[directory / 'composer_last.pt', directory / 'composer_inference.pt'],
            on_finish=lambda: reservation.__exit__(None, None, None))
    except Exception as exc:
        reservation.__exit__(None, None, None)
        if task is not None:
            task.status = 'failed'; task.error = str(exc)
        # A published data_split is experiment evidence. Keep it if launch
        # fails; the next fresh/finetune request must choose a new directory.
        raise
    handler._json_response({'task_id': task.task_id, 'family': FAMILY})


def start_generate(handler, payload, *, runtime):
    from .composer_api import _path, _int, _float, _arg, _device
    from maichart.composer.generation_settings import resolve_seed, resolve_condition
    from maichart.generation.timeline import serialize_tempo_map
    from maichart.form.timing import normalize_meter_map
    if payload.get('model_family') not in {None, FAMILY}: raise ValueError('Wrong Composer model family')
    audio = _path(payload.get('audio'), 'audio', exists=True)
    checkpoint = _path(payload.get('checkpoint'), 'checkpoint', exists=True)
    requested = _path(payload.get('output') or 'webui_workspace/generated/maidata.txt', 'output')
    seed, randomized = resolve_seed(payload.get('seed'))
    condition_file = _path(payload['condition_file'], 'condition-file', exists=True) if payload.get('condition_file') else None
    condition, source = resolve_condition(_float(payload, 'level', 13., minimum=1.), payload.get('condition'), condition_file)
    task = runtime._create_task('composer_generate')
    output = requested.parent / task.task_id / requested.name
    output.parent.mkdir(parents=True, exist_ok=True)
    args = ['-m', 'maichart.composer.stream_generate']
    settings = dict(architecture_profile=PROFILE, seed=seed, device=_device(payload.get('device')))
    for name, value in [('audio', audio), ('checkpoint', checkpoint), ('output', output), ('level', condition.level), ('seed', seed), ('device', settings['device'])]:
        _arg(args, '--' + name, value)
    temperature = _float(payload, 'temperature', .65, minimum=0.)
    for key, default in dict(first=0., temperature=temperature, pointer_temperature=temperature,
                              action_temperature=temperature, min_same_head_ms=50.).items():
        values=payload if payload.get(key) is not None else {**payload,key:default}
        settings[key] = _float(values, key, default, minimum=None if key == 'first' else 0.)
        _arg(args, '--' + key.replace('_', '-'), settings[key])
    _arg(args, '--max-events', _int(payload, 'max_events', 0, minimum=0))
    touch = payload.get('touch_layout_policy', 'open')
    if touch not in {'open', 'legacy_two_heads'}: raise ValueError('Unknown touch layout')
    _arg(args, '--touch-layout-policy', touch)
    settings['touch_layout_policy']=touch
    for key,default,choices in [('search_policy','phrase_v1',{'legacy','phrase_v1'}),
                                ('physical_policy','runtime_hard_v2',{'legacy','runtime_hard_v2'})]:
        value=payload.get(key,default)
        if value not in choices:raise ValueError('Unknown '+key)
        settings[key]=value;_arg(args,'--'+key.replace('_','-'),value)
    for key,default,minimum in [('phrase_candidates',1,1),('phrase_step_budget',256,1),
                               ('phrase_repair_budget',8,0),('phrase_rollback_budget',4,0)]:
        value=_int(payload,key,default,minimum=minimum)
        if key=='phrase_candidates' and value not in (1,2,4):raise ValueError('phrase_candidates must be 1,2,4')
        settings[key]=value;_arg(args,'--'+key.replace('_','-'),value)
    for key,default in [('phrase_write_beats',8.),('phrase_commit_beats',4.),('phrase_time_budget_sec',30.),
                        ('softcomfort_strength',1.),('audio_tracking_strength',0.)]:
        value=_float(payload,key,default,minimum=0.)
        if key in {'phrase_write_beats','phrase_commit_beats','phrase_time_budget_sec'} and value<=0:raise ValueError(key+' must be positive')
        if key=='audio_tracking_strength' and value>1 or key=='softcomfort_strength' and value>2:raise ValueError(key+' exceeds supported range')
        settings[key]=value;_arg(args,'--'+key.replace('_','-'),value)
    if settings['phrase_commit_beats']>settings['phrase_write_beats']:raise ValueError('phrase_commit_beats cannot exceed phrase_write_beats')
    if payload.get('bpm') not in (None, ''):
        bpm = _float(payload, 'bpm', 120., minimum=0.)
        if bpm <= 0: raise ValueError('bpm must be positive')
        _arg(args, '--bpm', bpm)
    snapshots = {'condition-file': condition.to_dict()}
    if payload.get('tempo_map') is not None: snapshots['tempo-map'] = serialize_tempo_map(payload['tempo_map'])
    meters = [meter.to_dict() for meter in normalize_meter_map(payload.get('meter_map'))]
    if meters: snapshots['meter-map'] = meters
    curve = None
    if payload.get('intensity_curve_enabled'):
        from maichart.generation.music_intensity import validate_intensity_profile
        snapshots['intensity-curve-file'] = validate_intensity_profile(payload.get('intensity_curve'))
    elif payload.get('intensity_curve_file'):
        curve = _path(payload['intensity_curve_file'], 'intensity curve', exists=True)
        _arg(args, '--intensity-curve-file', curve)
    for flag, value in snapshots.items():
        path = output.parent / (flag + '.json')
        path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        _arg(args, '--' + flag, path)
    if curve or 'intensity-curve-file' in snapshots:
        mode = payload.get('intensity_curve_mode', 'follow')
        if mode not in {'follow', 'soft', 'hard', 'strict'}: raise ValueError('Unknown curve mode')
        _arg(args, '--intensity-curve-mode', mode)
        _arg(args, '--intensity-curve-strength', _float(payload, 'intensity_curve_strength', .65, minimum=0.))
    task.result.update(family=FAMILY, checkpoint=str(checkpoint), output=str(output), requested_output=str(requested),
        seed=seed, seed_randomized=randomized, effective_settings={**settings, 'condition': condition.to_dict(), 'condition_source': source})
    runtime._run_subprocess_task(task, args, expected_epoch=None, result_paths=[output, output.with_suffix('.composer.json')])
    handler._json_response(dict(task_id=task.task_id, family=FAMILY, seed=seed, seed_randomized=randomized, output=str(output)))
