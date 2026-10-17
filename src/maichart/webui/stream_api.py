"""WebUI dispatcher for the independent chart_stream_v1 family."""
from __future__ import annotations

from pathlib import Path
import json

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
        path=runtime._required_path(payload.get('checkpoint'),'chart_stream checkpoint',must_exist=True)
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
              '--precision',str(payload.get('precision','auto')),'--cache-dir',str(payload.get('cache_dir','cache/chart_stream_v1'))]
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
        audio=runtime._required_path(payload.get('audio'),'audio',must_exist=True)
        checkpoint=runtime._required_path(payload.get('checkpoint'),'chart_stream checkpoint',must_exist=True)
        output=runtime._resolve_path(payload.get('output') or str(runtime.GENERATED_ROOT/'chart_stream_maidata.txt'))
        args=['-m','maichart.stream.generate','--audio',str(audio),'--checkpoint',str(checkpoint),
              '--output',str(output),'--bpm',str(payload.get('bpm',120)),'--first',str(payload.get('first',0)),
              '--level',str(payload.get('level',13)),'--temperature',str(payload.get('temperature',.7)),
              '--device',str(payload.get('device','cpu'))]
        task=runtime._create_task('chart_stream_generate')
        runtime._run_subprocess_task(task,args,expected_epoch=None,result_paths=[output,output.with_suffix('.stream.json')])
        task.result.update({'model_family':FAMILY,'experimental':True,'output':str(output)})
        return handler._json_response({'task_id':task.task_id})
    raise ValueError('unknown chart stream operation')
