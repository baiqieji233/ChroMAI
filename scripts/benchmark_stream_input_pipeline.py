"""Compare a saved source snapshot to the current real threaded pipeline."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import statistics
import time

import torch
from torch.utils.data import DataLoader

from maichart.form.codec import RationalCodec
from maichart.stream import data as data_module,symmetry
from maichart.stream.model import ChartStreamTransformer,StreamConfig
from maichart.stream.prefetch import StreamBatchPrefetch
from maichart.stream.runtime import attention_backend,training_cpu_budget
from maichart.stream.train import _loss,_finish_accumulation
from maichart.stream.training_batch import pack_batch,transfer_batch


def snapshot_function(path,name,module):
    tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    namespace=dict(vars(module))
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    return namespace[name]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--cache-dir',required=True)
    p.add_argument('--snapshot',required=True);p.add_argument('--output',required=True)
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--batches',type=int,default=24)
    args=p.parse_args()
    device=torch.device('cuda:0');torch.cuda.set_per_process_memory_fraction(.72,device)
    config=StreamConfig(len(RationalCodec()),d_model=384,nhead=8,layers=6,
                       dim_feedforward=1536,dropout=.1,grounding_profile='causal_audio_v1')
    old_kinematics=snapshot_function(Path(args.snapshot)/'stream/symmetry.py','kinematic_rows',symmetry)
    old_collate=snapshot_function(Path(args.snapshot)/'stream/data.py','collate_stream',data_module)
    new_kinematics=symmetry.kinematic_rows
    report={}
    with training_cpu_budget(2,device),attention_backend('auto',device):
        dataset=data_module.StreamDataset(args.manifest,cache_dir=args.cache_dir,max_tokens=1024,
            max_history_tokens=256,grounding_config=config,preparation_workers=2,audio_cache_gb=2,limit_charts=8)
        indices=[min(len(dataset)-1,int((i+.5)*len(dataset)/args.batch_size)) for i in range(args.batch_size)]
        reference=None
        for mode,kinematics,collate in (('before',old_kinematics,old_collate),('after',new_kinematics,data_module.collate_stream)):
            symmetry.kinematic_rows=kinematics
            for index in indices:dataset[index]
            times=[]
            for _ in range(3):
                start=time.perf_counter();batch=collate([dataset[i] for i in indices]);times.append(time.perf_counter()-start)
            if reference is None:reference=batch
            else:
                for key in batch:
                    if torch.is_tensor(batch[key]):torch.testing.assert_close(batch[key],reference[key],rtol=0,atol=0,msg=key)
            torch.manual_seed(71)
            model=ChartStreamTransformer(config).cuda().train()
            optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,fused=True)
            scaler=torch.amp.GradScaler('cuda',enabled=False);pending=0
            def step(batch):
                nonlocal pending
                tensors=transfer_batch(batch,device)
                with torch.autocast('cuda',dtype=torch.bfloat16):output=model(tensors);loss,_=_loss(output,tensors,cycle_exit_weight=.25)
                (loss/4).backward();pending+=1
                if pending==4:_finish_accumulation(model,optimizer,scaler,pending,4);pending=0
            for _ in range(4):step(pack_batch(batch))
            torch.cuda.synchronize()
            loader=DataLoader(dataset,batch_size=args.batch_size,num_workers=0,collate_fn=collate,
                              sampler=indices*args.batches)
            start=time.perf_counter()
            with StreamBatchPrefetch(loader,depth=2,workers=4,prepare_batch=pack_batch) as source:
                for value in source:step(value)
                torch.cuda.synchronize();stats=source.stats()
            elapsed=time.perf_counter()-start
            report[mode]={'cpu_batch_seconds':times,'cpu_median_seconds':statistics.median(times),
                          'pipeline_seconds':elapsed,'batches_per_second':args.batches/elapsed,'prefetch':stats}
            del optimizer,model,step
        symmetry.kinematic_rows=new_kinematics
    Path(args.output).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
