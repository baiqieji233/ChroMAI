"""Several independent charts with a single frozen model and audio preparation."""
from __future__ import annotations

from copy import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import traceback
import uuid

from .defaults import MAX_BATCH_COUNT


def _digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def _write(path, report):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    temporary.replace(path)


def _emit(report, index):
    print('composer_v2_batch '+json.dumps(dict(index=index,count=report['count'],
        completed=report['completed'],failed=report['failed'],status=report['status']),ensure_ascii=False),flush=True)


def generate_batch(args):
    # Imports stay here so the manifest helpers and workbench stay lightweight.
    import torch
    from .checkpoint import read, inspect_payload
    from .codec import ComponentCodec
    from .model import ComposerV2, ComposerV2Config
    from .generate import generate, prepare_inputs

    if not 1<=args.count<=MAX_BATCH_COUNT:
        raise ValueError(f'count must be in 1..{MAX_BATCH_COUNT}')
    if int(args.cpu_threads)<1:raise ValueError('cpu_threads must be positive')
    torch.set_num_threads(int(args.cpu_threads))
    checkpoint=Path(args.checkpoint).resolve()
    digest=_digest(checkpoint)
    payload=read(checkpoint)
    checked=inspect_payload(payload)
    if not checked['can_generate']:raise ValueError(checked['error'])
    device=args.device
    if device=='auto':device='cuda' if torch.cuda.is_available() else 'cpu'
    if device=='cuda':
        from maichart.webui.startup_check import require_cuda
        require_cuda(torch)
    config=ComposerV2Config.from_dict(checked['model_config'])
    model=ComposerV2(config,ComponentCodec.from_dict(payload['codec']))
    model.load_state_dict(payload['model_state_dict'],strict=True)
    model=model.to(device).eval()
    # Detect concurrent checkpoint replacement before labelling the batch.
    if _digest(checkpoint)!=digest:
        raise ValueError('检查点在批量载入期间发生变化，请重新提交')
    prepared=prepare_inputs(args,config,device)
    target=Path(args.output).resolve()
    manifest=Path(args.batch_manifest).resolve() if args.batch_manifest else (
        target.parent/(target.stem+'_batch_'+uuid.uuid4().hex[:12])/'batch.composer_v2_batch.json')
    # A manifest uniquely owns its directory and never overwrites an old batch.
    manifest.parent.mkdir(parents=True,exist_ok=True)
    if manifest.exists():raise ValueError('批量结果已存在，请选择新的目录')
    report=dict(schema='composer-v2-batch-v1',status='running',count=int(args.count),
        completed=0,failed=0,checkpoint=str(checkpoint),checkpoint_sha256=digest,
        optimizer_steps=checked['optimizer_steps'],audio=str(Path(args.audio).resolve()),
        created_at=datetime.now().isoformat(timespec='seconds'),charts=[])
    _write(manifest,report)
    _emit(report,1)
    for index in range(args.count):
        item=copy(args)
        item.count=1
        item.seed=int(args.seed)+index
        item.output=str(manifest.parent/f'{index+1:02d}_seed_{item.seed}'/target.name)
        entry=dict(index=index+1,seed=item.seed,output=item.output)
        _emit(report,index+1)
        try:
            metadata=generate(item,model_override=model,payload_override=payload,prepared_override=prepared)
            output=Path(item.output)
            entry.update(status='completed',report=str(output.with_suffix('.composer_v2.json')),
                events=metadata.get('events'),sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                accepted_with_conflicts_count=metadata.get('accepted_with_conflicts_count',0))
            report['completed']+=1
        except Exception as exc:
            traceback.print_exc()
            entry.update(status='failed',error=str(exc))
            diagnostic=Path(item.output).with_suffix('.failure.json')
            if diagnostic.is_file():entry['diagnostic']=str(diagnostic)
            report['failed']+=1
        report['charts'].append(entry)
        _write(manifest,report)
    report['status']='completed' if not report['failed'] else ('partial' if report['completed'] else 'failed')
    _write(manifest,report)
    _emit(report,args.count)
    if not report['completed']:
        raise RuntimeError(f'全部 {args.count} 张谱面生成失败；详见 {manifest}')
    return report
