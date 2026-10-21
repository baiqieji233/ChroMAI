"""WebUI dispatcher for the independent chart_stream_v1 family."""
from __future__ import annotations

from pathlib import Path
import json
import math
import re
import secrets

from maichart.stream import FAMILY, CHECKPOINT_SCHEMA


def update_stream_progress(task,line):
    prefix,_,body=line.partition(' ')
    if prefix not in {'form_prepare','stream_prepare','stream_dataset','stream_runtime',
                      'stream_progress','stream_epoch','stream_checkpoint','stream_execution'}:
        return
    try:
        value=json.loads(body)
    except (ValueError,TypeError):
        return
    if not isinstance(value,dict):
        return
    if prefix in {'form_prepare','stream_prepare'}:
        task.result.update(phase='prepare',progress_known=False,progress_detail=value)
        done=value.get('completed',value.get('completed_songs',0))
        total=value.get('total',value.get('total_songs','?'))
        task.message=f'准备连续流训练数据 · {done}/{total} · 谱面 {value.get("charts", "?")} · 计算块 {value.get("chunks", "?")}'
        if value.get('stage')=='residency':
            task.message=(f'训练输入驻留 · {done}/{total} · 已载入 {value.get("loaded",done)} 项 · '
                f'{value.get("bytes",0)/1024**3:.2f} / {value.get("budget_bytes",0)/1024**3:.2f} GiB')
    elif prefix=='stream_dataset':
        task.result['stream_dataset']=value
        task.message=f'数据准备完成 · 谱面 {value.get("charts")} · 计算块 {value.get("chunks")}'
    elif prefix=='stream_runtime':
        task.result.update(stream_runtime=value,epoch_target=value.get('end_epoch'))
    elif prefix=='stream_execution':
        task.result.setdefault('stream_execution',{})[value.get('block','unknown')]=value
        task.message=f'训练执行准备 · {value.get("block")} · {value.get("reason")}'
    elif prefix=='stream_checkpoint':
        task.result['last_checkpoint']=value
        task.message=f'已保存 checkpoint · 更新 {value.get("optimizer_steps")} 步'
    else:
        task.result.update(phase='stream',progress_known=True,last_epoch=value.get('epoch'),stream_metrics=value)
        if prefix=='stream_progress':
            task.result['progress_detail']=value
            fraction=float(value['chunk'])/max(1,int(value['total']))
            task.message=(f'连续谱面训练 · epoch {value["epoch"]}/{task.result.get("epoch_target", "?")} · '
                f'batch {value["chunk"]}/{value["total"]} · loss {value["loss"]:.4f} · '
                f'token准确率 {value["token_accuracy"]:.1%} · {value.get("batches_per_sec",0):.2f} batch/秒')
            if 'recent_batches_per_sec' in value:
                task.message+=(f' · 近期 {value["recent_batches_per_sec"]:.2f} batch/秒'
                    f' · 主机计时：等数据 {value.get("data_wait_sec",0):.1f}s / 计算 {value.get("compute_host_sec",0):.1f}s')
            if 'sample_cuda_forward_backward_ms' in value:
                task.message+=(f' · 抽样 batch CUDA 流：传输 {value.get("sample_cuda_transfer_ms",0):.1f}ms'
                    f' / 前后向 {value["sample_cuda_forward_backward_ms"]:.1f}ms')
            execution=value.get('execution') or {}
            if execution.get('captured'):
                task.message+=' · CUDA Graph '+','.join(execution['captured'])
            memory=value.get('cuda_memory') or {}
            if memory:
                task.message+=f' · 显存 {memory.get("allocated_gib",0):.2f} GiB（峰值 {memory.get("peak_allocated_gib",0):.2f}）'
        else:
            fraction=1.
            history=task.result.setdefault('stream_metric_history',[])
            history.append(value); del history[:-80]
            task.message=f'第 {value["epoch"]} 轮已保存 · loss {value["loss"]:.4f}'
        runtime=task.result.get('stream_runtime',{})
        start=int(runtime.get('start_epoch',1))
        count=max(1,int(runtime.get('end_epoch',value['epoch']))-start+1)
        task.progress=min(.99,max(0.,(int(value['epoch'])-start+fraction)/count))


def inspect(path):
    import torch
    payload=torch.load(path,map_location='cpu',weights_only=False)
    return inspect_payload(payload)


def inspect_payload(payload):
    from maichart.stream.data import TRAINING_SEMANTICS
    from maichart.generation.model_compatibility import checkpoint_capabilities
    capabilities = checkpoint_capabilities(payload)
    valid=bool(capabilities.get('valid'))
    errors=[]
    try:
        from maichart.stream.checkpoint import validate_contract
        validate_contract(payload)
    except (ValueError,KeyError,TypeError) as exc:
        valid=False
        errors.append(str(exc))
    from maichart.stream.checkpoint import TRAINING_OBJECTIVE_CONTRACT
    current = ((payload.get('args') or {}).get('training_semantics') == TRAINING_SEMANTICS
               and payload.get('training_objective_contract')==TRAINING_OBJECTIVE_CONTRACT)
    can_resume = bool(valid and current and payload.get('optimizer_state_dict') and payload.get('dataset_fingerprint'))
    warnings=[]
    if not current:
        warnings.append('训练合同已更新为 v14（统一因果间隔遮罩与多假设双手特征）；旧权重仍可生成，请用 finetune 和新输出目录迁移，不能直接 resume。')
    elif not can_resume:
        warnings.append('此权重缺少续训所需的优化器状态或数据身份；请选择 finetune 和新输出目录。')
    if can_resume and payload.get('training_position') and payload['training_position'].get('complete') is not True:
        warnings.append('此权重保存在未完成轮次；resume 保留优化器状态并重跑该轮，避免跳过剩余目标。')
    return {'valid':valid,'model_family':FAMILY,'schema':payload.get('schema'),
            'model_config':payload.get('model_config'), 'epoch':payload.get('epoch'),
            'args':payload.get('args',{}), 'input_contract':payload.get('input_contract'),
            'optimizer_steps':payload.get('optimizer_steps',0),
            'training_position':payload.get('training_position',{}),
            'can_resume':can_resume, 'can_init':valid,
            'warnings':warnings,
            'errors':errors,
            'generation_capabilities':capabilities}


def start(handler, operation, payload, *, runtime):
    if operation == 'inspect':
        path=runtime._required_path(payload.get('checkpoint') or payload.get('checkpoint_path'),'chart_stream checkpoint',must_exist=True)
        return handler._json_response(inspect(path))
    if payload.get('model_family',FAMILY) != FAMILY:
        raise ValueError('chart stream endpoint requires model_family=chart_stream_v1')
    if operation == 'train':
        dataset_selection=None
        if payload.get('manifests'):
            payload,dataset_selection=runtime._prepare_training_sources(payload)
        manifest=runtime._required_path(payload.get('manifest'),'manifest',must_exist=True)
        output=runtime._resolve_path(payload.get('checkpoint_dir') or 'checkpoints/chart_stream_v1')
        mode=payload.get('mode','fresh'); checkpoint=payload.get('init_checkpoint') or payload.get('training_checkpoint')
        if mode not in {'fresh', 'resume', 'finetune'}:
            raise ValueError('训练模式必须是 fresh、resume 或 finetune')
        if mode == 'resume' and checkpoint and not inspect(checkpoint).get('can_resume'):
            raise ValueError('训练监督合同已更新；请使用 finetune 和新输出目录，不能直接 resume 旧训练。')
        args=['-m','maichart.stream.train','--manifest',str(manifest),'--checkpoint-dir',str(output),
              '--epochs',str(payload.get('epochs',10)),'--batch-size',str(payload.get('batch_size',1)),
              '--grad-accumulation',str(payload.get('grad_accumulation',1)),'--d-model',str(payload.get('d_model',256)),
              '--layers',str(payload.get('layers',4)),'--nhead',str(payload.get('nhead',8)),
              '--dim-feedforward',str(payload.get('dim_feedforward',768)),'--max-tokens',str(payload.get('max_tokens',1024)),
              '--dropout',str(payload.get('dropout',.1)),'--seed',str(payload.get('seed',0)),
              '--max-history-tokens',str(payload.get('max_history_tokens',256)),'--max-stream-memory',str(payload.get('max_stream_memory',32)),
              '--lr',str(payload.get('lr',2e-4)),'--device',str(payload.get('device','auto')),
              '--precision',str(payload.get('precision','auto')),'--cache-dir',str(payload.get('cache_dir','cache/chart_stream_v1')),
               '--plan-loss-weight',str(payload.get('plan_loss_weight',.15)),
               '--curve-slope-loss-weight',str(payload.get('curve_slope_loss_weight',.05)),
              '--event-decision-loss-weight',str(payload.get('event_decision_loss_weight',.25)),
              '--weight-decay',str(payload.get('weight_decay',.01)),
              '--audio-cache-gb',str(payload.get('audio_cache_gb',8)),
              '--prepare-workers',str(payload.get('prepare_workers',4)),
              '--max-region-seconds',str(payload.get('max_region_seconds',15)),
              '--limit-charts',str(payload.get('limit_charts',0)),
              '--log-interval',str(payload.get('log_interval',25)),
              '--checkpoint-interval-steps',str(payload.get('checkpoint_interval_steps',500))]
        for key,default in (('grounding_profile','auto'),('detail_audio_nodes',8192),
                ('local_audio_neighbors',32),('local_audio_seconds',2.),('archive_events',4096),
                ('archive_retrievals',8),('archive_min_similarity',.5),('history_note_dropout',.15),
                ('prefix_note_dropout',.1),('alignment_loss_weight',.15),('alignment_tolerance_ms',15.),
                ('event_balance_loss_weight',.2),('prefetch_batches',2),('data_workers',4),('cpu_threads',0),
                ('input_cache_mode','mapped'),('structure_cache_mb',256),('optimizer_implementation','auto'),
                ('attention_backend','auto'),('motor_profile','auto'),('hand_backend','auto'),
                ('execution_mode','auto'),('cuda_graph_budget_gb',2.),
                ('input_residency','auto'),('resident_cache_gb',0.),('resident_memory_reserve_gb',0.),
                ('cache_validation_records',0),('continuation_profile','auto'),('context_replay_tokens',256),
                ('prefix_mask_span_events',8),('cycle_exit_loss_weight',.25),('symmetry_augmentation','mirror_lr')):
            args += ['--'+key.replace('_','-'),str(payload.get(key,default))]
        if type(payload.get('quality_enabled',True)) is not bool:
            raise ValueError('quality_enabled 必须是布尔值')
        args += ['--quality-enabled' if payload.get('quality_enabled',True) else '--no-quality-enabled']
        if mode == 'resume':
            if not checkpoint: raise ValueError('resume requires checkpoint path')
            args += ['--resume',str(runtime._required_path(checkpoint,'resume checkpoint',must_exist=True))]
        elif mode == 'finetune':
            if not checkpoint: raise ValueError('finetune requires checkpoint path')
            args += ['--init-checkpoint',str(runtime._required_path(checkpoint,'init checkpoint',must_exist=True))]
        from maichart.stream.train import parser as training_parser, validate_training_args, validate_output_directory
        try:
            parsed=training_parser().parse_args(args[2:])
        except SystemExit as exc:
            raise ValueError('连续流训练参数无效，请检查整数、数值范围与关联输入方案') from exc
        validate_training_args(parsed)
        validate_output_directory(parsed)
        task=runtime._create_task('chart_stream_train')
        report=output/'stream_last.pt'
        runtime._run_subprocess_task(task,args,expected_epoch=None,result_paths=[report])
        task.result.update({'model_family':FAMILY,'checkpoint_dir':str(output),'experimental':True,
                           'dataset_selection':dataset_selection,'effective_manifest':str(manifest),
                           'quality_enabled':parsed.quality_enabled})
        return handler._json_response({'task_id':task.task_id})
    if operation == 'generate':
        from maichart.stream.generate import parser as generation_parser
        from maichart.stream.decoding import validate_options
        from maichart.form.timing import normalize_meter_map
        from maichart.generation.timeline import serialize_tempo_map
        audio=runtime._required_path(payload.get('audio'),'audio',must_exist=True)
        checkpoint=runtime._required_path(payload.get('checkpoint'),'chart_stream checkpoint',must_exist=True)
        requested_output=runtime._resolve_path(payload.get('output') or str(runtime.GENERATED_ROOT/'chart_stream_maidata.txt'))
        device = str(payload.get('device') or 'auto').strip().lower()
        if device not in {'auto', 'cpu', 'cuda'} and not re.fullmatch(r'cuda:\d+', device):
            raise ValueError('生成设备必须为 auto、cpu 或 cuda[:编号]')
        raw_seed = payload.get('seed')
        if raw_seed is None or raw_seed == '':
            seed = secrets.randbits(31)
        else:
            if isinstance(raw_seed, bool):
                raise ValueError('seed 必须为非负整数')
            try:
                number = float(raw_seed)
                if not math.isfinite(number) or number != int(raw_seed):
                    raise ValueError('seed 必须为非负整数')
                seed = int(raw_seed)
            except (TypeError, OverflowError, ValueError) as exc:
                raise ValueError('seed 必须为非负整数') from exc
        args=['-m','maichart.stream.generate','--audio',str(audio),'--checkpoint',str(checkpoint),
              '--output',str(requested_output),'--first',str(payload.get('first',0)),
              '--level',str(payload.get('level',13)),'--temperature',str(payload.get('temperature',.7)),
              '--device',device, '--seed',str(seed)]
        if payload.get('bpm') not in {None, ''}:
            args += ['--bpm', str(payload['bpm'])]
        for key, default in (('position_temperature', .15), ('max_events', 0),
                             ('eos_temperature',1.),('position_audio_strength',.25),('repetition_strength',.6),
                             ('repetition_structure_strength',.6),('configuration_exploration_temperature',1.),
                             ('repetition_min_seconds',4.),('repetition_min_beats',8.),
                             ('position_beam_size', 4), ('position_complexity', .35),
                             ('position_complexity_floor',8),('learned_ending_strength',1.),
                             ('constraint_min_probability',.05),('event_reconsider_attempts',4),
                             ('position_interval_strength', 1.), ('unplanned_gap_penalty_strength', .5),
                             ('rhythm_diversity_strength', .6),
                             ('early_eos_penalty', 0.),
                             ('max_denominator', 384), ('max_interval_denominator', 96),
                             ('min_spacing_ms', 20.)):
            args += ['--'+key.replace('_', '-'), str(payload.get(key, default))]
        args += ['--style-balance-strength', str(payload.get('style_balance_strength', 0.)),
                 '--configuration-candidates',str(payload.get('configuration_candidates',4)),
                 '--local-chord-strength', str(payload.get('local_chord_strength', 0.)),
                 '--physical-feasibility-strength', str(payload.get('physical_feasibility_strength', 0.)),
                 '--section-type-strength', str(payload.get('section_type_strength', 1.)),
                 '--plan-intent-strength', str(payload.get('plan_intent_strength', 1.)),
                 '--max-button-heads', str(payload.get('max_button_heads', 2)),
                 '--hold-protection', str(payload.get('hold_protection', 'on')),
                 '--reference-check-attempts', str(payload.get('reference_check_attempts',8)),
                 '--trim-exact-silence',str(payload.get('trim_exact_silence','on'))]
        if payload.get('generation_end_sec') is not None:
            args += ['--generation-end-sec',str(payload['generation_end_sec'])]
        if payload.get('sustain_end_sec') is not None:
            args += ['--sustain-end-sec',str(payload['sustain_end_sec'])]
        try:
            options = generation_parser().parse_args(args[2:])
            validate_options(options)
        except SystemExit as exc:
            raise ValueError('连续生成参数无效，请检查数值与整数范围') from exc
        tempo = serialize_tempo_map(payload['tempo_map']) if payload.get('tempo_map') is not None else None
        meters = [m.to_dict() for m in normalize_meter_map(payload.get('meter_map'))]
        task=runtime._create_task('chart_stream_generate')
        output=requested_output.parent/task.task_id/requested_output.name
        args[args.index('--output')+1]=str(output)
        task_directory=runtime.WEBUI_ROOT/'plans'/task.task_id
        condition = payload.get('condition')
        for flag, name, value in (('--condition-file','condition.json',condition),
                                  ('--tempo-map','tempo.json',tempo), ('--meter-map','meter.json',meters)):
            if value is not None and value != []:
                task_directory.mkdir(parents=True, exist_ok=True)
                file_path=task_directory/name
                file_path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding='utf-8')
                args += [flag,str(file_path)]
        options = generation_parser().parse_args(args[2:])
        task.result.update({'model_family':FAMILY,'experimental':True,'output':str(output),
                            'requested_output':str(requested_output),'actual_output':str(output),
                            'seed':seed,'device':device,
                            'effective_settings':{**vars(options),'output':str(output)}})
        runtime._run_subprocess_task(task,args,expected_epoch=None,result_paths=[output,output.with_suffix('.stream.json')])
        return handler._json_response({'task_id':task.task_id})
    raise ValueError('unknown chart stream operation')
