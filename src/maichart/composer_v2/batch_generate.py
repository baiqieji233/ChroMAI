"""Bounded independent chart generation with one shared audio preparation."""
from __future__ import annotations

from copy import copy
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import tempfile
import time
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
        completed=report['completed'],failed=report['failed'],status=report['status'],
        parallel=report.get('parallel',1),active=report.get('active',[]),
        queued=max(0,report['count']-report['completed']-report['failed']-len(report.get('active',[])))),ensure_ascii=False),flush=True)


def _failed(item,index,error):
    entry=dict(index=index,seed=item.seed,output=item.output,status='failed',error=str(error))
    diagnostic=Path(item.output).with_suffix('.failure.json')
    if diagnostic.is_file():entry['diagnostic']=str(diagnostic)
    log=Path(item.output).with_suffix('.generation.log')
    if log.is_file():entry['log']=str(log)
    return entry


def _generate_one(item,index,**overrides):
    from .generate import generate
    started=datetime.now().isoformat(timespec='milliseconds');stamp=time.perf_counter()
    try:
        metadata=generate(item,**overrides)
        output=Path(item.output)
        entry=dict(index=index,seed=item.seed,output=item.output,status='completed',
            report=str(output.with_suffix('.composer_v2.json')),events=metadata.get('events'),
            sha256=_digest(output),accepted_with_conflicts_count=metadata.get('accepted_with_conflicts_count',0))
        audit=metadata.get('candidate_probability_audit') or {}
        if audit.get('path'):entry['candidate_audit']=audit['path']
    except Exception as exc:
        traceback.print_exc();entry=_failed(item,index,exc)
    entry.update(worker_pid=os.getpid(),started_at=started,
                 finished_at=datetime.now().isoformat(timespec='milliseconds'),elapsed_sec=time.perf_counter()-stamp)
    return entry


_WORKER_STATE=None


def _parallel_chart(item,index,snapshot,digest,timing):
    """One process owns its model, caches and random generators for the batch."""
    global _WORKER_STATE
    output=Path(item.output);output.parent.mkdir(parents=True,exist_ok=True)
    log=output.with_suffix('.generation.log')
    with log.open('w',encoding='utf-8',buffering=1) as handle,redirect_stdout(handle),redirect_stderr(handle):
        try:
            if _WORKER_STATE is None:
                import torch
                from .checkpoint import read,inspect_payload
                from .codec import ComponentCodec
                from .model import ComposerV2,ComposerV2Config
                from .audio import AudioFeatureStore
                from .inference_runtime import configure_threads
                configure_threads(item.cpu_threads,item.device,getattr(item,'cpu_parallelism',1))
                if _digest(snapshot)!=digest:raise ValueError('批量检查点快照已变化')
                payload=read(snapshot);checked=inspect_payload(payload)
                if not checked['can_generate']:raise ValueError(checked['error'])
                config=ComposerV2Config.from_dict(checked['model_config'])
                model=ComposerV2(config,ComponentCodec.from_dict(payload['codec']))
                model.load_state_dict(payload['model_state_dict'],strict=True)
                model=model.to(item.device).eval()
                # The parent has already completed MERT/native extraction and
                # timing. Workers only map the existing feature bank on CPU.
                store=AudioFeatureStore(item.cache_dir,backend=config.audio_backend,
                                       model_path=config.mert_model,device='cpu')
                try:audio=store.load(Path(item.audio).resolve())
                finally:store.close()
                _WORKER_STATE=(model,payload,(*timing,audio))
            model,payload,prepared=_WORKER_STATE
            entry=_generate_one(item,index,model_override=model,payload_override=payload,prepared_override=prepared)
        except Exception as exc:
            traceback.print_exc();entry=_failed(item,index,exc)
        if entry['status']=='failed':
            import torch
            if _WORKER_STATE is not None:
                _WORKER_STATE[0]._song_cache.clear();_WORKER_STATE[0]._structured_weight_cache.clear()
            if torch.cuda.is_initialized():torch.cuda.empty_cache()
    entry['log']=str(log)
    return entry


def _record(report,entry,manifest):
    report['charts'].append(entry);report['charts'].sort(key=lambda row:row['index'])
    report['completed' if entry['status']=='completed' else 'failed']+=1
    _write(manifest,report);_emit(report,entry['index'])


def _terminate_pool(pool):
    # Python 3.12 has no public terminate_workers(). These are exclusively this
    # coordinator's children; normal WebUI cancellation also kills its task tree.
    processes=list((pool._processes or {}).values())
    for process in processes:
        if process.is_alive():process.terminate()
    for process in processes:
        process.join(timeout=2)
        if process.is_alive():process.kill();process.join()


def _run_parallel(items,report,manifest,snapshot,digest,timing):
    pool=ProcessPoolExecutor(max_workers=report['parallel'],mp_context=multiprocessing.get_context('spawn'))
    pending={};cursor=0
    try:
        while pending or cursor<len(items):
            while len(pending)<report['parallel'] and cursor<len(items):
                item=items[cursor];index=cursor+1;cursor+=1
                try:pending[pool.submit(_parallel_chart,item,index,str(snapshot),digest,timing)]=(index,item)
                except Exception as exc:
                    # A broken worker pool must produce a finite, complete batch
                    # report rather than leave unsubmitted charts waiting forever.
                    for remaining in range(index-1,len(items)):
                        _record(report,_failed(items[remaining],remaining+1,exc),manifest)
                    cursor=len(items);break
            report['active']=sorted(index for index,_ in pending.values())
            _write(manifest,report);_emit(report,report['active'][0] if pending else report['count'])
            if not pending:break
            done,_=wait(pending,return_when=FIRST_COMPLETED)
            for future in done:
                index,item=pending.pop(future)
                try:entry=future.result()
                except Exception as exc:entry=_failed(item,index,exc)
                report['active']=sorted(index for index,_ in pending.values())
                _record(report,entry,manifest)
    except BaseException:
        _terminate_pool(pool);raise
    finally:pool.shutdown(wait=True,cancel_futures=True)


def generate_batch(args):
    # Imports stay here so the manifest helpers and workbench stay lightweight.
    import torch
    from .checkpoint import read, inspect_payload
    from .codec import ComponentCodec
    from .model import ComposerV2, ComposerV2Config
    from .generate import prepare_inputs

    if not 1<=args.count<=MAX_BATCH_COUNT:
        raise ValueError(f'count must be in 1..{MAX_BATCH_COUNT}')
    parallel=getattr(args,'parallel',1)
    if isinstance(parallel,bool) or not isinstance(parallel,int) or not 1<=parallel<=MAX_BATCH_COUNT:
        raise ValueError(f'parallel must be in 1..{MAX_BATCH_COUNT}')
    parallel=min(parallel,args.count)
    checkpoint=Path(args.checkpoint).resolve()
    digest=_digest(checkpoint)
    payload=read(checkpoint)
    checked=inspect_payload(payload)
    if not checked['can_generate']:raise ValueError(checked['error'])
    device=args.device
    if device=='auto':device='cuda' if torch.cuda.is_available() else 'cpu'
    from .inference_runtime import configure_threads
    cpu_runtime=configure_threads(args.cpu_threads,device,parallel)
    if device=='cuda':
        from maichart.webui.startup_check import require_cuda
        require_cuda(torch)
    config=ComposerV2Config.from_dict(checked['model_config'])
    # Detect concurrent checkpoint replacement before labelling the batch.
    if _digest(checkpoint)!=digest:
        raise ValueError('检查点在批量载入期间发生变化，请重新提交')
    target=Path(args.output).resolve()
    manifest=Path(args.batch_manifest).resolve() if args.batch_manifest else (
        target.parent/(target.stem+'_batch_'+uuid.uuid4().hex[:12])/'batch.composer_v2_batch.json')
    # A manifest uniquely owns its directory and never overwrites an old batch.
    manifest.parent.mkdir(parents=True,exist_ok=True)
    if manifest.exists():raise ValueError('批量结果已存在，请选择新的目录')
    report=dict(schema='composer-v2-batch-v1',status='preparing',count=int(args.count),parallel=parallel,
        requested_parallel=getattr(args,'parallel',1),active=[],device=device,cpu_runtime=cpu_runtime,
        completed=0,failed=0,checkpoint=str(checkpoint),checkpoint_sha256=digest,
        optimizer_steps=checked['optimizer_steps'],audio=str(Path(args.audio).resolve()),
        created_at=datetime.now().isoformat(timespec='seconds'),charts=[])
    _write(manifest,report)
    _emit(report,1)
    items=[]
    for index in range(args.count):
        item=copy(args)
        item.count=1;item.parallel=1;item.device=device;item.cpu_parallelism=parallel;item.checkpoint_sha256=digest
        item.seed=int(args.seed)+index
        item.output=str(manifest.parent/f'{index+1:02d}_seed_{item.seed}'/target.name)
        items.append(item)
    try:
        if parallel==1:
            model=ComposerV2(config,ComponentCodec.from_dict(payload['codec']))
            model.load_state_dict(payload['model_state_dict'],strict=True);model=model.to(device).eval()
            prepared=prepare_inputs(args,config,device)
            report['status']='running'
            for index,item in enumerate(items,1):
                report['active']=[index];_write(manifest,report);_emit(report,index)
                entry=_generate_one(item,index,model_override=model,payload_override=payload,prepared_override=prepared)
                report['active']=[];_record(report,entry,manifest)
        else:
            # Freeze before audio preparation: an active trainer may atomically
            # replace the public checkpoint while this batch is running.
            with tempfile.TemporaryDirectory(prefix='.batch-runtime-',dir=manifest.parent) as temporary:
                snapshot=Path(temporary)/'checkpoint.pt';shutil.copyfile(checkpoint,snapshot)
                if _digest(snapshot)!=digest:raise ValueError('检查点在批量冻结期间发生变化，请重新提交')
                del payload
                prepared=prepare_inputs(args,config,device)
                prepared[3].summary(160)
                timing=prepared[:3];del prepared
                if torch.cuda.is_initialized():torch.cuda.empty_cache()
                report['status']='running';_run_parallel(items,report,manifest,snapshot,digest,timing)
    except BaseException as exc:
        report.update(status='cancelled' if isinstance(exc,(KeyboardInterrupt,SystemExit)) else 'failed',
                      error=str(exc),active=[])
        _write(manifest,report);_emit(report,report['completed']+report['failed']);raise
    report['active']=[]
    report['status']='completed' if not report['failed'] else ('partial' if report['completed'] else 'failed')
    _write(manifest,report)
    _emit(report,args.count)
    if not report['completed']:
        raise RuntimeError(f'全部 {args.count} 张谱面生成失败；详见 {manifest}')
    return report
