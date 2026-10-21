"""Matched real-v14 eager pipeline before/after efficiency-only changes.

Loads a saved hand implementation and dataset methods into a private module.
It does not load, save or change user checkpoints. Cold and warm measurements
include the exact causal hand inputs, concurrent prefetch, AMP, accumulation
and optimizer updates. Use a small representative manifest/--limit-charts.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import gc
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

import torch
from torch.utils.data import DataLoader,RandomSampler

from maichart.form.codec import RationalCodec
from maichart.stream import data as data_module,hand_tracking
from maichart.stream.model import ChartStreamTransformer,StreamConfig
from maichart.stream.prefetch import StreamBatchPrefetch
from maichart.stream.runtime import attention_backend,training_cpu_budget
from maichart.stream.train import _loss,_finish_accumulation
from maichart.stream.training_batch import pack_batch,transfer_batch


def load_method(path,name):
    tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='StreamDataset')
    node=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name==name)
    namespace=dict(vars(data_module))
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    return namespace[name]


def load_hands(path):
    spec=importlib.util.spec_from_file_location('_frozen_v14_hands',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True);parser.add_argument('--cache-dir',required=True)
    parser.add_argument('--snapshot',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--accumulation',type=int,default=2)
    parser.add_argument('--limit-charts',type=int,default=32)
    parser.add_argument('--batches',type=int,default=16)
    args=parser.parse_args()
    device=torch.device('cuda:0');torch.cuda.set_per_process_memory_fraction(.72,device)
    baseline=Path(args.snapshot)/'src/maichart/stream'
    original_hands=hand_tracking.HandHistory
    original_query=data_module.StreamDataset._mapped_query
    frozen_hands=load_hands(baseline/'hand_tracking.py').HandHistory
    frozen_query=load_method(baseline/'data.py','_mapped_query')
    from maichart.stream.hand_tracking import prepare_training_backend
    native=prepare_training_backend('auto')
    if native['effective']=='native':
        from maichart.stream.native_hand_tracking import NativeHandHistory
        optimized_hands=NativeHandHistory
    else:optimized_hands=original_hands
    config=StreamConfig(len(RationalCodec()),d_model=384,nhead=8,layers=6,dim_feedforward=1536,
        dropout=.1,grounding_profile='causal_audio_v1',motor_profile='causal_hands_v1')
    results={'device':torch.cuda.get_device_name(),'torch':torch.__version__,
        'batch_size':args.batch_size,'accumulation':args.accumulation,'limit_charts':args.limit_charts,
        'hand_backend':native,'modes':{}}
    with training_cpu_budget(2,device),attention_backend('auto',device) as policy:
        results['attention_policy']=policy
        dataset=data_module.StreamDataset(args.manifest,cache_dir=args.cache_dir,max_tokens=1024,
            max_history_tokens=256,grounding_config=config,preparation_workers=2,audio_cache_gb=1,
            limit_charts=args.limit_charts,history_note_dropout=.15,prefix_note_dropout=.1,
            prefix_mask_span_events=8)
        dataset.set_epoch(1)
        rng=torch.Generator().manual_seed(99)
        indices=torch.randint(len(dataset),(args.batch_size*args.batches,),generator=rng).tolist()
        # Both sides start with the same trusted deterministic disk artifacts.
        # Cache building during one side would not be a matched compute test.
        for index in dict.fromkeys(indices):dataset[index]
        reference=None
        try:
            for mode,history,query in (('before',frozen_hands,frozen_query),('after',optimized_hands,original_query)):
                hand_tracking.HandHistory=history;data_module.StreamDataset._mapped_query=query
                dataset._hand_histories={}
                dataset.hand_backend='python' if mode=='before' else native['effective']
                dataset._audio_queries.clear();dataset.query_bytes=dataset.query_hits=dataset.query_misses=0
                dataset._structures.clear();dataset.structure_bytes=dataset.structure_hits=dataset.structure_misses=0
                dataset.query_prepare_seconds.clear();dataset.query_prepare_counts.clear()
                torch.cuda.reset_peak_memory_stats(device)
                torch.manual_seed(71)
                model=ChartStreamTransformer(config).to(device).train()
                optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,fused=True)
                scaler=torch.amp.GradScaler('cuda',enabled=False);pending=0
                # Warm GPU without building any real hand history.
                from benchmark_stream_runtime import batch as synthetic_batch
                synthetic=synthetic_batch(config,args.batch_size,device)
                def step(tensors):
                    nonlocal pending
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        output=model(tensors);loss,_=_loss(output,tensors,cycle_exit_weight=.25)
                    (loss/args.accumulation).backward();pending+=1
                    if pending==args.accumulation:
                        _finish_accumulation(model,optimizer,scaler,pending,args.accumulation);pending=0
                for _ in range(args.accumulation*2):step(synthetic)
                del synthetic;torch.cuda.synchronize()
                rounds={}
                for phase in ('cold','warm'):
                    loader=DataLoader(dataset,batch_size=args.batch_size,num_workers=0,
                        collate_fn=data_module.collate_stream,sampler=indices)
                    started=time.perf_counter()
                    with StreamBatchPrefetch(loader,depth=2,workers=4,prepare_batch=pack_batch) as batches:
                        for packed in batches:step(transfer_batch(packed,device))
                        torch.cuda.synchronize();stats=batches.stats()
                    elapsed=time.perf_counter()-started
                    rounds[phase]={'seconds':elapsed,'batches_per_second':args.batches/elapsed,
                        'samples_per_second':args.batch_size*args.batches/elapsed,'prefetch':stats}
                check=data_module.collate_stream([dataset[i] for i in indices[:args.batch_size]])
                if reference is None:reference=check
                else:
                    for key,value in check.items():
                        if torch.is_tensor(value):torch.testing.assert_close(value,reference[key],rtol=0,atol=0,msg=key)
                        else:assert value==reference[key],key
                frozen=transfer_batch(pack_batch(check),device)
                values=[]
                for _ in range(4):
                    started=time.perf_counter();step(frozen);torch.cuda.synchronize();values.append(time.perf_counter()-started)
                rounds['compute_only_median_seconds']=statistics.median(values)
                rounds['peak_allocated_gib']=torch.cuda.max_memory_allocated()/1024**3
                rounds['cache']=dataset.cache_stats()
                results['modes'][mode]=rounds
                print('pipeline_result '+json.dumps({mode:rounds},ensure_ascii=False),flush=True)
                del step,frozen,check,model,optimizer,scaler
                gc.collect();torch.cuda.empty_cache()
        finally:
            hand_tracking.HandHistory=original_hands;data_module.StreamDataset._mapped_query=original_query
    results['input_tensors_identical']=True
    Path(args.output).write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({key:value for key,value in results.items() if key!='modes'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
