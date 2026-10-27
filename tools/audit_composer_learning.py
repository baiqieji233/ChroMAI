"""Bounded CPU paired probes; never update or overwrite the supplied checkpoint."""
from __future__ import annotations
import argparse
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import torch
from maichart.composer.checkpoint import load, save
from maichart.composer.objectives import action_objective
from maichart.composer.train import _forward_batch, _loss, summarize_metrics


def evaluate(model, batches, policy):
    totals={}
    model.eval()
    with torch.no_grad():
        for batch in batches:
            out=_forward_batch(model,batch)
            _,metrics=_loss(out,batch,model.config.atom_vocab_size,audio_pointer_weight=.15,
                placement_weight=.15,planner_weight=.2,action_objective_policy=policy)
            for key,value in metrics.items(): totals[key]=totals.get(key,0.)+value
    return summarize_metrics(totals,len(batches))


def gradient_interference(model,batches,policy,maximum):
    groups={
        'shared_classifier':[(n,p) for n,p in model.named_parameters() if n.startswith(('action_head.','slot_fusion.','candidate_fusion.'))],
        'shared_audio':[(n,p) for n,p in model.named_parameters() if n.startswith(('audio_projection.','audio_encoder.','audio_fusion.'))],
    }
    selected=[(name,p) for values in groups.values() for name,p in values]
    parameters=[p for _,p in selected]
    rows=[];model.eval()
    for index,batch in enumerate(batches[:maximum]):
        out=_forward_batch(model,batch)
        primary,_=_loss(out,batch,model.config.atom_vocab_size,audio_pointer_weight=0.,action_objective_policy=policy)
        weights=batch['sample_weight']*batch.get('transition_weight',torch.ones_like(batch['sample_weight']))*batch.get('action_weight',torch.ones_like(batch['sample_weight']))
        auxiliary,_=action_objective(out['audio_action_logits'],batch['action_target'],weights,policy=policy)
        first=torch.autograd.grad(primary,parameters,retain_graph=True,allow_unused=True)
        second=torch.autograd.grad(auxiliary,parameters,allow_unused=True)
        offset=0;row={'batch':index}
        for name,values in groups.items():
            a2=b2=dot=0.
            for a,b in zip(first[offset:offset+len(values)],second[offset:offset+len(values)]):
                if a is not None: a2+=float(a.square().sum())
                if b is not None: b2+=float(b.square().sum())
                if a is not None and b is not None: dot+=float((a*b).sum())
            offset+=len(values)
            row[name]={'cosine':dot/math.sqrt(max(a2*b2,1e-30)),
                'primary_norm':math.sqrt(a2),'unweighted_aux_norm':math.sqrt(b2),
                'weighted_aux_to_primary_norm_at_015':.15*math.sqrt(b2/max(a2,1e-30))}
        rows.append(row)
        del out,first,second,primary,auxiliary
        print('gradient_probe '+json.dumps(row),flush=True)
    return rows


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--batches',required=True)
    p.add_argument('--output-dir',required=True);p.add_argument('--updates',type=int,default=16)
    p.add_argument('--gradient-batches',type=int,default=3);p.add_argument('--cpu-threads',type=int,default=1)
    a=p.parse_args(argv)
    if a.updates<0 or a.gradient_batches<0 or a.cpu_threads<1: p.error('Invalid probe limits')
    checkpoint=Path(a.checkpoint).resolve();out=Path(a.output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    if checkpoint.parent==out: p.error('Use a separate output directory from the baseline snapshot')
    torch.set_num_threads(a.cpu_threads)
    pack=torch.load(a.batches,map_location='cpu',weights_only=False)
    for key in ('train','probe'):
        if not pack.get(key): p.error('Pack must have nonempty train/probe batches')
    digest=hashlib.file_digest(checkpoint.open('rb'),'sha256').hexdigest()
    model,payload=load(checkpoint,device='cpu')
    policy=payload['args']['action_objective_policy']
    result={'checkpoint':str(checkpoint),'sha256':digest,'optimizer_steps':payload['optimizer_steps'],
        'input_config':payload['model_config'],'updates_per_arm':a.updates,'seed':20261027,
        'scope':'paired short CPU updates on frozen batches; probe songs disjoint from short updates, not necessarily unseen by parent; recovery disabled in every arm; no musical-quality acceptance',
        'gradient_scope':'main objective versus audio action auxiliary only; excludes audio pointer gradients',
        'baseline':evaluate(model,pack['probe'],policy),
        'gradient_probes':gradient_interference(model,pack['train'],policy,a.gradient_batches),'arms':{}}
    path=out/'result.json';path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    del model,payload;gc.collect()
    for name,lr,aux_weight in [('baseline',2e-4,.15),('lower_lr',5e-5,.15),('lower_audio',2e-4,.05)]:
        if not a.updates: break
        started=time.perf_counter();torch.manual_seed(20261027);random.seed(20261027)
        model,payload=load(checkpoint,device='cpu',include_optimizer=True)
        if not payload.get('optimizer_state_dict'): raise ValueError('Paired updates require saved Adam moments')
        optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01)
        optimizer.load_state_dict(payload.pop('optimizer_state_dict'))
        for group in optimizer.param_groups: group.update(lr=lr,fused=False,foreach=False)
        # Reset after model construction/load; all arms see identical dropout RNG.
        torch.manual_seed(20261027);model.train();trace=[]
        for step in range(a.updates):
            optimizer.zero_grad(set_to_none=True)
            batch=pack['train'][step%len(pack['train'])]
            output=_forward_batch(model,batch)
            loss,metrics=_loss(output,batch,model.config.atom_vocab_size,audio_pointer_weight=aux_weight,action_objective_policy=policy)
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm) or not torch.isfinite(loss): raise FloatingPointError('Probe became nonfinite')
            optimizer.step();trace.append({'update':step+1,'loss':float(loss.detach()),'grad_norm':float(norm)})
            if (step+1)%4==0: print('paired_update '+json.dumps({'arm':name,**trace[-1]}),flush=True)
        score=evaluate(model,pack['probe'],policy)
        result['arms'][name]={'lr':lr,'audio_weight':aux_weight,'metrics_common_weight_015':score,'seconds':time.perf_counter()-started,'trace':trace}
        path.write_text(json.dumps(result,indent=2),encoding='utf-8')
        save(out/(name+'_diagnostic.pt'),model,None,epoch=0,args={**payload['args'],'diagnostic_only':True},
             manifest=payload['manifest'],atom_catalog=payload['atom_catalog'],optimizer_steps=payload['optimizer_steps']+a.updates)
        print('paired_result '+json.dumps({'arm':name,'metrics':score}),flush=True)
        del model,payload,optimizer,output,loss,metrics;gc.collect()
    if hashlib.file_digest(checkpoint.open('rb'),'sha256').hexdigest()!=digest: raise RuntimeError('Baseline was modified')
    print('audit_complete '+str(path),flush=True)
    return 0


if __name__=='__main__': raise SystemExit(main())
