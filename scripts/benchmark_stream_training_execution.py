"""Bounded eager/graph comparison including optimizer and accumulation.

Optional --manifest/--cache-dir selects a representative real batch from the
existing dataset, without loading or changing a checkpoint. This is a speed
probe, never a musical-quality or complete-epoch measurement.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import statistics
import time

import torch

from benchmark_stream_runtime import batch as synthetic_batch
from maichart.form.codec import RationalCodec
from maichart.stream.model import ChartStreamTransformer,StreamConfig
from maichart.stream.runtime import attention_backend,training_cpu_budget
from maichart.stream.train import _loss,_finish_accumulation
from maichart.stream.training_batch import pack_batch,transfer_batch
from maichart.stream.training_graph import install_training_graphs,MetricAccumulator


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--manifest')
    p.add_argument('--cache-dir',default='cache/chart_stream_v1')
    p.add_argument('--limit-charts',type=int,default=8)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--steps',type=int,default=12)
    p.add_argument('--graph-budget-gb',type=float,default=2.)
    args=p.parse_args()
    device=torch.device('cuda:0')
    torch.cuda.set_per_process_memory_fraction(.72,device)
    config=StreamConfig(len(RationalCodec()),d_model=384,nhead=8,layers=6,
                       dim_feedforward=1536,dropout=.1,grounding_profile='causal_audio_v1')
    with training_cpu_budget(2,device),attention_backend('auto',device) as policy:
        if args.manifest:
            from maichart.stream.data import StreamDataset,collate_stream
            dataset=StreamDataset(args.manifest,cache_dir=args.cache_dir,max_tokens=config.max_tokens,
                max_history_tokens=config.max_history_tokens,grounding_config=config,
                preparation_workers=2,audio_cache_gb=1,limit_charts=args.limit_charts)
            indices=[min(len(dataset)-1,int((i+.5)*len(dataset)/args.batch_size)) for i in range(args.batch_size)]
            data=transfer_batch(pack_batch(collate_stream([dataset[i] for i in indices])),device)
            source={'kind':'real_batch','indices':indices,'charts':len(dataset.songs)}
        else:
            data=synthetic_batch(config,args.batch_size,device)
            source={'kind':'synthetic'}
        result={'device':torch.cuda.get_device_name(),'torch':torch.__version__,
                'batch_size':args.batch_size,'source':source,'attention_policy':policy,'modes':{}}
        for mode in ('eager','cuda_graph'):
            torch.manual_seed(71)
            model=ChartStreamTransformer(config).to(device).train()
            optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,fused=True)
            scaler=torch.amp.GradScaler('cuda',enabled=False)
            runner=install_training_graphs(model,mode=mode,batch_size=args.batch_size,
                max_tokens=config.max_tokens,max_audio_tokens=config.max_audio_tokens,budget_gb=args.graph_budget_gb)
            runner.prepare(model,'bf16')
            optimizer.zero_grad(set_to_none=True)
            metrics=MetricAccumulator()
            pending=0
            def step():
                nonlocal pending
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    output=model(data);loss,values=_loss(output,data,cycle_exit_weight=.25)
                (loss/4).backward();metrics.update(values);pending+=1
                if pending==4:
                    _finish_accumulation(model,optimizer,scaler,pending,4);pending=0
            for _ in range(4):step()
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            times=[]
            for _ in range(3):
                start=time.perf_counter()
                for _ in range(args.steps):step()
                torch.cuda.synchronize()
                times.append((time.perf_counter()-start)/args.steps)
            result['modes'][mode]={'seconds_per_batch':times,'median_seconds':statistics.median(times),
                'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3,'execution':runner.stats()}
            # Scope teardown avoids counting both models/graph pools together.
            object.__delattr__(model,'_training_graph_runner')
            for _,wrapper in runner.slots.values():wrapper.forward=type(wrapper).forward.__get__(wrapper)
            del step,runner,optimizer,scaler,model,metrics
            gc.collect();torch.cuda.empty_cache()
        Path(args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
