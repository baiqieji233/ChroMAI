"""Synthetic full-loss ABBA for integrated survival and shared button context.

Random initial weights, full read-only v26 atom catalog, no optimizer steps or
checkpoint writes. Measures compute and memory, never chart quality. Defaults:
B64, BF16, two warmup steps and five measured forward/backward steps per round.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path
import statistics

import torch

from benchmark_composer_compact import make_config, make_v3_batch
from benchmark_composer_quality import measure
from maichart.composer.checkpoint import read_payload
from maichart.composer.model import build_model


def make_integrated_batch(config, batch_size, device, notes_per_event=2, grid_cells=240):
    data, labels=make_v3_batch(config,batch_size,device,notes_per_event)
    b,k=batch_size,grid_cells
    # Four seconds of causal budget history plus an eight-second search window.
    edges=torch.linspace(0.,12.,k+1,device=device,dtype=torch.float64)[None].expand(b,-1)
    centers=(edges[:,:-1]+edges[:,1:])*.5
    plan_seconds=data['quality_inputs']['quality_plan_clock'][...,1].contiguous()
    # Nearest plan location, expressed through the same pair/mix interface as
    # real grid inputs. Zero mix intentionally chooses this nearest location.
    nearest=(centers[...,None]-plan_seconds[:,None,:]).abs().argmin(-1)
    data.update(
        survival_audio=torch.rand(b,k,104,device=device),
        survival_edges=edges,
        survival_candidate_seconds=data['candidate_features'][...,121].double()+4.,
        survival_start_seconds=torch.full((b,),4.,device=device,dtype=torch.float64),
        survival_prefix_counts=torch.zeros(b,k,2,device=device),
        survival_plan_indices=torch.stack((nearest,nearest),-1),
        survival_plan_mix=torch.zeros(b,k,device=device))
    labels.update(
        survival_count_target=torch.randint(0,3,(b,k),device=device).float(),
        survival_count_weight=torch.ones(b,k,device=device),
        survival_widths=(edges[:,1:]-edges[:,:-1]).float())
    # Keep original workload/empty-gap/count labels: _loss selects the grid
    # Poisson branch only when the model actually returns survival_grid_rates.
    return data,labels


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',help='Read only atom_catalog; defaults to v26 inference, then last')
    parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--notes-per-event',type=int,default=2)
    parser.add_argument('--grid-cells',type=int,default=240)
    parser.add_argument('--warmup',type=int,default=2)
    parser.add_argument('--steps',type=int,default=5)
    parser.add_argument('--seed',type=int,default=123)
    parser.add_argument('--device',default='auto')
    parser.add_argument('--precision',choices=('bf16','fp32'),default='bf16')
    parser.add_argument('--cpu-threads',type=int,default=2)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if min(args.batch_size,args.steps,args.cpu_threads,args.grid_cells)<1 or args.warmup<0:
        parser.error('batch-size, steps, cpu-threads and grid-cells must be positive; warmup nonnegative')
    if not 1<=args.notes_per_event<=7:parser.error('notes-per-event must be in 1..7')
    root=Path(__file__).resolve().parents[1]
    checkpoint=(Path(args.checkpoint).resolve() if args.checkpoint else next(
        (p for p in (root/'checkpoints/prism_composer_v26'/name
                     for name in ('composer_inference.pt','composer_last.pt')) if p.is_file()),None))
    if checkpoint is None:parser.error('No v26 inference/last checkpoint; supply --checkpoint')
    payload=read_payload(checkpoint)
    atoms=list(payload['atom_catalog'])
    del payload
    gc.collect()
    torch.set_num_threads(args.cpu_threads)
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device)
    if device.type=='cuda' and args.precision=='bf16' and not torch.cuda.is_bf16_supported():
        parser.error('Selected CUDA device does not support BF16; use --precision fp32')
    base=make_config(len(atoms)+1,'compact_grouped_v3')
    configs={'baseline':base,'integrated':replace(base,timing_distribution_policy='survival_integrated_v2',
                                                 button_load_policy='per_button_context_v2',fine_timing_policy='relative_phrase_v1',
                                                 density_feedback_policy='causal_count_debt_v1')}
    torch.manual_seed(args.seed)
    data,labels=make_integrated_batch(base,args.batch_size,device,args.notes_per_event,args.grid_cells)
    rounds=[]
    for number,variant in enumerate(('baseline','integrated','integrated','baseline'),1):
        torch.manual_seed(args.seed+1)
        model=build_model(configs[variant],atoms).to(device).train()
        parameters=sum(p.numel() for p in model.parameters())
        print(f'Round {number}/4: {variant}; B={args.batch_size}; parameters={parameters}',flush=True)
        result=measure(model,data,labels,args,device)
        rounds.append(dict(round=number,variant=variant,parameter_count=parameters,**result))
        print(json.dumps(rounds[-1]),flush=True)
        del model
        gc.collect()
        if device.type=='cuda':
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
    variants={}
    for variant in configs:
        rows=[row for row in rounds if row['variant']==variant]
        variants[variant]={key:(None if rows[0][key] is None else
            max(row[key] for row in rows) if 'peak' in key else statistics.median(row[key] for row in rows))
            for key in rows[0] if key not in ('round','variant')}
    before,after=variants['baseline'],variants['integrated']
    report=dict(
        scope='Synthetic full training-loss forward/backward; excludes preparation, transfer, optimizer, recovery and generation',
        interpretation='Random weights and synthetic labels; compute/memory only, not musical quality or end-to-end throughput.',
        checkpoint_use='Read-only atom_catalog only; no weights/optimizer restored or checkpoint writes',
        catalog_source=str(checkpoint),vocab_size=len(atoms)+1,
        initialization='Same reset seed each round; additional parameter shapes may shift random draws between variants',
        order='ABBA',seed=args.seed,torch_version=torch.__version__,device=str(device),
        device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
        precision=args.precision if device.type=='cuda' else 'fp32',batch_size=args.batch_size,
        warmup=args.warmup,steps=args.steps,notes_per_event=args.notes_per_event,
        packed_slots_per_event=args.notes_per_event+1,grid_cells=args.grid_cells,grid_duration_seconds=12.,
        configs={key:value.to_dict() for key,value in configs.items()},
        memory_note='One model resident per round; shared batch includes new grid in both variants. Peaks include model, inputs, activations and gradients.',
        variants=variants,rounds=rounds,integrated_wall_ratio=after['wall_step_ms']/before['wall_step_ms'],
        integrated_gpu_step_ratio=(after['forward_ms_median']+after['backward_ms_median'])/
                                  (before['forward_ms_median']+before['backward_ms_median']))
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
