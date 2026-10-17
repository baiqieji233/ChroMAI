"""WebUI dispatcher for the independent chart_stream_v1 family."""
from __future__ import annotations

from pathlib import Path
import json
import math
import re
import secrets

from maichart.stream import FAMILY, CHECKPOINT_SCHEMA


def inspect(path):
    import torch
    payload=torch.load(path,map_location='cpu',weights_only=False)
    valid=payload.get('schema')==CHECKPOINT_SCHEMA and payload.get('family')==FAMILY
    return {'valid':valid,'model_family':FAMILY,'schema':payload.get('schema'),
            'model_config':payload.get('model_config'), 'epoch':payload.get('epoch'),
            'args':payload.get('args',{}), 'input_contract':payload.get('input_contract'),
            'generation_capabilities':{'continuous_stream':valid,'rational_event_positions':valid,
                                       'sustained_action_memory':valid}}


def start(handler, operation, payload, *, runtime):
    if operation == 'inspect':
        path=runtime._required_path(payload.get('checkpoint') or payload.get('checkpoint_path'),'chart_stream checkpoint',must_exist=True)
        return handler._json_response(inspect(path))
    if payload.get('model_family',FAMILY) != FAMILY:
        raise ValueError('chart stream endpoint requires model_family=chart_stream_v1')
    if operation == 'train':
        manifest=runtime._required_path(payload.get('manifest'),'manifest',must_exist=True)
        output=runtime._resolve_path(payload.get('checkpoint_dir') or 'checkpoints/chart_stream_v1')
        mode=payload.get('mode','fresh'); checkpoint=payload.get('init_checkpoint') or payload.get('training_checkpoint')
        args=['-m','maichart.stream.train','--manifest',str(manifest),'--checkpoint-dir',str(output),
              '--epochs',str(payload.get('epochs',10)),'--batch-size',str(payload.get('batch_size',1)),
              '--grad-accumulation',str(payload.get('grad_accumulation',1)),'--d-model',str(payload.get('d_model',256)),
              '--layers',str(payload.get('layers',4)),'--nhead',str(payload.get('nhead',8)),
              '--dim-feedforward',str(payload.get('dim_feedforward',768)),'--max-tokens',str(payload.get('max_tokens',1024)),
              '--max-history-tokens',str(payload.get('max_history_tokens',256)),'--max-stream-memory',str(payload.get('max_stream_memory',32)),
              '--lr',str(payload.get('lr',2e-4)),'--device',str(payload.get('device','auto')),
              '--precision',str(payload.get('precision','auto')),'--cache-dir',str(payload.get('cache_dir','cache/chart_stream_v1')),
              '--plan-loss-weight',str(payload.get('plan_loss_weight',.15)),
              '--weight-decay',str(payload.get('weight_decay',.01)),
              '--audio-cache-gb',str(payload.get('audio_cache_gb',8)),
              '--prepare-workers',str(payload.get('prepare_workers',4)),
              '--max-region-seconds',str(payload.get('max_region_seconds',15)),
              '--limit-charts',str(payload.get('limit_charts',0)),
              '--log-interval',str(payload.get('log_interval',25)),
              '--checkpoint-interval-steps',str(payload.get('checkpoint_interval_steps',500))]
        if mode == 'resume':
            if not checkpoint: raise ValueError('resume requires checkpoint path')
            args += ['--resume',str(runtime._required_path(checkpoint,'resume checkpoint',must_exist=True))]
        elif mode == 'finetune':
            if not checkpoint: raise ValueError('finetune requires checkpoint path')
            args += ['--init-checkpoint',str(runtime._required_path(checkpoint,'init checkpoint',must_exist=True))]
        task=runtime._create_task('chart_stream_train')
        report=output/'stream_last.pt'
        runtime._run_subprocess_task(task,args,expected_epoch=None,result_paths=[report])
        task.result.update({'model_family':FAMILY,'checkpoint_dir':str(output),'experimental':True})
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
                             ('position_beam_size', 4), ('position_complexity', .35),
                             ('position_interval_strength', 1.),
                             ('early_eos_penalty', 0.),
                             ('max_denominator', 384), ('max_interval_denominator', 96),
                             ('min_spacing_ms', 20.)):
            args += ['--'+key.replace('_', '-'), str(payload.get(key, default))]
        args += ['--style-balance-strength', str(payload.get('style_balance_strength', .5)),
                 '--local-chord-strength', str(payload.get('local_chord_strength', 0.)),
                 '--physical-feasibility-strength', str(payload.get('physical_feasibility_strength', 0.))]
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
