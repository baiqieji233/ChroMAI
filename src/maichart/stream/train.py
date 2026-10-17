"""Train the independent continuous whole-song chart stream family."""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import math
import random

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from maichart.form.quality_profile import MUSICAL_EVIDENCE
from .data import StreamDataset, collate_stream
from .model import ChartStreamTransformer, StreamConfig
from .checkpoint import save, load


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True); p.add_argument('--checkpoint-dir',required=True)
    p.add_argument('--cache-dir',default='cache/chart_stream_v1'); p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--batch-size',type=int,default=1); p.add_argument('--grad-accumulation',type=int,default=1)
    p.add_argument('--d-model',type=int,default=256); p.add_argument('--layers',type=int,default=4)
    p.add_argument('--nhead',type=int,default=8); p.add_argument('--dim-feedforward',type=int,default=768)
    p.add_argument('--dropout',type=float,default=.1); p.add_argument('--max-tokens',type=int,default=1024)
    p.add_argument('--max-history-tokens',type=int,default=256); p.add_argument('--max-stream-memory',type=int,default=32)
    p.add_argument('--lr',type=float,default=2e-4); p.add_argument('--weight-decay',type=float,default=.01)
    p.add_argument('--device',default='auto'); p.add_argument('--precision',choices=('auto','fp32','bf16','fp16'),default='auto')
    p.add_argument('--audio-cache-gb',type=float,default=8.); p.add_argument('--prepare-workers',type=int,default=4)
    p.add_argument('--limit-charts',type=int,default=0); p.add_argument('--resume'); p.add_argument('--init-checkpoint')
    p.add_argument('--log-interval',type=int,default=25); p.add_argument('--seed',type=int,default=0)
    p.add_argument('--max-region-seconds',type=float,default=15.)
    p.add_argument('--plan-loss-weight',type=float,default=.15)
    return p


def _precision(value,device):
    if value == 'auto':
        return 'bf16' if device.type == 'cuda' and torch.cuda.is_bf16_supported() else 'fp32'
    if device.type != 'cuda' and value != 'fp32': raise ValueError('CPU stream training requires fp32')
    if value == 'bf16' and not torch.cuda.is_bf16_supported(): raise ValueError('bf16 unsupported')
    return value


def _loss(output,batch,plan_weight=.15):
    logits=output['logits'].float(); target=batch['target']; mask=~batch['target_padding_mask']
    ce=F.cross_entropy(logits.transpose(1,2),target,ignore_index=0,reduction='none')
    token=(ce*mask).sum()/mask.sum().clamp_min(1)
    plan_mask=batch['plan_mask'] & ~batch['plan_mask_padding_mask']
    plan=F.smooth_l1_loss(output['plan'].float(),batch['plan_labels'].float(),reduction='none').mean(-1)
    plan=(plan*plan_mask).sum()/plan_mask.sum().clamp_min(1)
    return token + plan_weight*plan, {'token_loss':token.detach(),'plan_loss':plan.detach(),
                               'loss':(token+plan_weight*plan).detach(),
                               'token_accuracy':((logits.argmax(-1)==target)&mask).sum().float()/mask.sum().clamp_min(1)}


def main(argv=None):
    args=parser().parse_args(argv)
    device=torch.device('cuda' if args.device=='auto' and torch.cuda.is_available() else ('cpu' if args.device=='auto' else args.device))
    precision=_precision(args.precision,device); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    old=None
    if args.resume or args.init_checkpoint:
        model,old=load(args.resume or args.init_checkpoint,device='cpu')
        if args.init_checkpoint and not args.resume:
            optimizer=None
        else: optimizer_state=old.get('optimizer_state_dict')
        args.d_model=model.config.d_model; args.layers=model.config.layers; args.nhead=model.config.nhead
        args.dim_feedforward=model.config.dim_feedforward; args.dropout=model.config.dropout
        args.max_tokens=model.config.max_tokens; args.max_history_tokens=model.config.max_history_tokens
        args.max_stream_memory=model.config.max_stream_memory
    dataset=StreamDataset(args.manifest,cache_dir=args.cache_dir,max_tokens=args.max_tokens,
        max_history_tokens=args.max_history_tokens,max_region_seconds=args.max_region_seconds,
        preparation_workers=args.prepare_workers,audio_cache_gb=args.audio_cache_gb,limit_charts=args.limit_charts)
    if old is None:
        model=ChartStreamTransformer(StreamConfig(vocab_size=len(dataset.codec),d_model=args.d_model,
            nhead=args.nhead,layers=args.layers,dim_feedforward=args.dim_feedforward,dropout=args.dropout,
            max_tokens=args.max_tokens,max_history_tokens=args.max_history_tokens,max_stream_memory=args.max_stream_memory))
    model.to(device); model.train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    if args.resume and old.get('optimizer_state_dict'): optimizer.load_state_dict(old['optimizer_state_dict'])
    loader=DataLoader(dataset,batch_size=args.batch_size,shuffle=True,collate_fn=collate_stream,num_workers=0)
    out=Path(args.checkpoint_dir); out.mkdir(parents=True,exist_ok=True)
    atom_catalog=[]
    for record in dataset.base.songs:
        for point in record[1]:
            # Keep the same atomic split used by the legacy event vocabulary.
            atom_catalog.extend(str(point.note_text).split('/'))
    global_step=0
    start_epoch = int(old.get('epoch', 0)) if args.resume and old else 0
    for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
        totals={}; optimizer.zero_grad(set_to_none=True); pending=0
        for index,batch in enumerate(loader,1):
            tensors={k:v.to(device,non_blocking=device.type=='cuda') if torch.is_tensor(v) else v for k,v in batch.items()}
            with torch.autocast('cuda',dtype=torch.bfloat16 if precision=='bf16' else torch.float16,enabled=precision in {'bf16','fp16'}):
                output=model(tensors); loss,metrics=_loss(output,tensors,args.plan_loss_weight)
            (loss/args.grad_accumulation).backward(); pending+=1
            if pending>=args.grad_accumulation or index==len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step(); optimizer.zero_grad(set_to_none=True); pending=0; global_step+=1
            for key,value in metrics.items(): totals[key]=totals.get(key,0.)+float(value)
            if index%args.log_interval==0 or index==len(loader):
                average={k:v/index for k,v in totals.items()}
                print('stream_progress '+json.dumps({'family':'chart_stream_v1','epoch':epoch,'chunk':index,'total':len(loader),'optimizer_steps':global_step,**average}),flush=True)
        save(out/'stream_last.pt',model,optimizer,epoch=epoch,args=vars(args),manifest=args.manifest,metrics=totals,atom_catalog=atom_catalog)
    return 0


if __name__=='__main__': raise SystemExit(main())
