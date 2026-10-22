"""Train chart_composer_v1 on structured authored events."""
from __future__ import annotations
import argparse,json,math,random,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from . import FAMILY,TRAINING_SEMANTICS
from .data import ComposerDataset,collate_composer
from .model import ChartComposer,ComposerConfig
from .checkpoint import save,load


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--checkpoint-dir',required=True)
    p.add_argument('--cache-dir',default='cache/chart_composer_v1');p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--batch-size',type=int,default=4);p.add_argument('--lr',type=float,default=2e-4)
    p.add_argument('--d-model',type=int,default=192);p.add_argument('--layers',type=int,default=3)
    p.add_argument('--nhead',type=int,default=6);p.add_argument('--dim-feedforward',type=int,default=576)
    p.add_argument('--dropout',type=float,default=.1);p.add_argument('--history-events',type=int,default=128)
    p.add_argument('--candidate-count',type=int,default=96);p.add_argument('--action-slots',type=int,default=8)
    p.add_argument('--device',default='auto');p.add_argument('--precision',choices=('auto','fp32','bf16','fp16'),default='auto')
    p.add_argument('--seed',type=int,default=0);p.add_argument('--limit-charts',type=int,default=0)
    p.add_argument('--prepare-workers',type=int,default=4);p.add_argument('--audio-cache-gb',type=float,default=4.)
    p.add_argument('--checkpoint-interval-steps',type=int,default=500);p.add_argument('--resume');p.add_argument('--init-checkpoint')
    return p


def _device(value):
    if value=='auto':return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(value)


def _precision(value,device):
    if value=='auto':return 'bf16' if device.type=='cuda' and torch.cuda.is_bf16_supported() else 'fp32'
    if device.type!='cuda' and value!='fp32':raise ValueError('CPU composer training requires fp32')
    return value


def _loss(output,batch,atom_vocab_size):
    pointer=F.cross_entropy(output['candidate_logits'],batch['candidate_target'])
    target=batch['candidate_target']
    action_logits=output['action_logits'][torch.arange(target.size(0),device=target.device),target]
    future=output['future'][torch.arange(target.size(0),device=target.device),target]
    eos=output['eos_logits'][torch.arange(target.size(0),device=target.device),target]
    action_per_slot=F.cross_entropy(action_logits.reshape(-1,atom_vocab_size),
        batch['action_target'].reshape(-1),reduction='none').reshape(action_logits.size(0),-1)
    slot_weights=torch.ones(action_per_slot.size(1),device=action_per_slot.device)
    if slot_weights.numel()>2:
        slot_weights[2:]=.25
    action=(action_per_slot*slot_weights[None]).sum()/slot_weights.sum().clamp_min(1e-6)/action_per_slot.size(0)
    eos_target=batch['eos_target']
    positives=eos_target.sum().clamp_min(1.)
    negatives=(eos_target.numel()-eos_target.sum()).clamp_min(1.)
    eos_loss=F.binary_cross_entropy_with_logits(eos,eos_target,
        pos_weight=(negatives/positives).clamp(1.,64.))
    future_loss=F.smooth_l1_loss(future,batch['future_target'])
    total=pointer+.6*action+.05*eos_loss+.2*future_loss
    return total,{'loss':total.detach(),
        'pointer_loss':pointer.detach(),'action_loss':action.detach(),'eos_loss':eos_loss.detach(),
        'future_loss':future_loss.detach(),
        'pointer_accuracy':(output['candidate_logits'].argmax(-1)==batch['candidate_target']).float().mean().detach()}


def main(argv=None):
    args=parser().parse_args(argv);device=_device(args.device);precision=_precision(args.precision,device)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    dataset=ComposerDataset(args.manifest,cache_dir=args.cache_dir,candidate_count=args.candidate_count,
        history_events=args.history_events,action_slots=args.action_slots,preparation_workers=args.prepare_workers,
        audio_cache_gb=args.audio_cache_gb,limit_charts=args.limit_charts,seed=args.seed)
    old=None
    if args.resume or args.init_checkpoint:
        model,old=load(args.resume or args.init_checkpoint,device='cpu')
        if args.resume and old.get('training_semantics')!=TRAINING_SEMANTICS:
            raise ValueError('Composer training contract changed; use --init-checkpoint with a new output directory')
        if sorted(old.get('atom_catalog') or []) != sorted(dataset.codec_atoms):
            raise ValueError('Composer atom catalog differs from checkpoint; use a matching manifest or retrain a new model')
        if old.get('dataset_fingerprint') and args.resume and old['dataset_fingerprint'] != dataset.fingerprint:
            raise ValueError('Composer resume dataset fingerprint changed; use finetune with a new output directory')
        for key in ('d_model','dim_feedforward','layers','nhead','history_events','candidate_count','action_slots'):
            if int(getattr(model.config,key)) != int(getattr(args,key)):
                raise ValueError(f'Composer checkpoint {key}={getattr(model.config,key)} differs from requested {getattr(args,key)}')
    else:
        model=ChartComposer(ComposerConfig(atom_vocab_size=len(dataset.codec_atoms)+1,d_model=args.d_model,
            nhead=args.nhead,layers=args.layers,dim_feedforward=args.dim_feedforward,dropout=args.dropout,
            history_events=args.history_events,candidate_count=args.candidate_count,action_slots=args.action_slots))
    model.to(device).train()
    loader=DataLoader(dataset,batch_size=args.batch_size,shuffle=True,collate_fn=collate_composer,
                      generator=torch.Generator().manual_seed(args.seed))
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    if args.resume and old and old.get('optimizer_state_dict'):optimizer.load_state_dict(old['optimizer_state_dict'])
    out=Path(args.checkpoint_dir);out.mkdir(parents=True,exist_ok=True)
    global_step=int(old.get('optimizer_steps',0)) if args.resume and old else 0
    start_epoch=(int(old.get('epoch',0))+1) if args.resume and old else 1
    end_epoch=(int(old.get('epoch',0))+args.epochs) if args.resume and old else args.epochs
    scaler=(torch.amp.GradScaler('cuda') if device.type=='cuda' and precision=='fp16' else None)
    print('composer_runtime '+json.dumps({'family':FAMILY,'training_semantics':TRAINING_SEMANTICS,
        'device':str(device),'precision':precision,'samples':len(dataset),'parameters':sum(p.numel() for p in model.parameters()),
        'candidate_count':args.candidate_count,'action_vocab_size':len(dataset.codec_atoms)+1,
        'end_epoch':end_epoch,'candidate_misses':dataset.candidate_misses,
        'candidate_total':dataset.candidate_total,
        'candidate_miss_examples':dataset.candidate_miss_examples}),flush=True)
    for epoch in range(start_epoch,end_epoch+1):
        dataset.set_epoch(epoch);started=time.monotonic(); totals={};
        for batch_index,batch in enumerate(loader,1):
            tensors={key:(value.to(device) if torch.is_tensor(value) else value) for key,value in batch.items()}
            with torch.autocast('cuda',dtype=torch.bfloat16 if precision=='bf16' else torch.float16,
                                enabled=device.type=='cuda' and precision in {'bf16','fp16'}):
                output=model(tensors['candidate_features'],tensors['history'],tensors['condition'],
                    action_state=tensors['action_state'],history_features=tensors['history_features'],
                    candidate_mask=tensors['candidate_padding_mask'],
                    history_padding_mask=tensors['history_padding_mask'])
                loss,metrics=_loss(output,tensors,model.config.atom_vocab_size)
            if scaler is not None:
                scaler.scale(loss).backward();scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.);scaler.step(optimizer);scaler.update()
            else:
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            optimizer.zero_grad(set_to_none=True);global_step+=1
            for key,value in metrics.items():totals[key]=totals.get(key,0.)+float(value)
            if batch_index%25==0 or batch_index==len(loader):
                avg={key:value/max(1,batch_index) for key,value in totals.items()}
                print('composer_progress '+json.dumps({'epoch':epoch,'batch':batch_index,'total':len(loader),
                    'optimizer_steps':global_step,'elapsed_sec':time.monotonic()-started,**avg}),flush=True)
            if args.checkpoint_interval_steps and global_step%args.checkpoint_interval_steps==0:
                save(out/'composer_last.pt',model,optimizer,epoch=epoch,args=vars(args),manifest=args.manifest,
                     metrics={key:value/max(1,batch_index) for key,value in totals.items()},optimizer_steps=global_step,
                     dataset_fingerprint=dataset.fingerprint,
                     atom_catalog=dataset.codec_atoms)
                print('composer_checkpoint '+json.dumps({'epoch':epoch,'optimizer_steps':global_step,'path':str(out/'composer_last.pt')}),flush=True)
        save(out/'composer_last.pt',model,optimizer,epoch=epoch,args=vars(args),manifest=args.manifest,
             metrics={key:value/max(1,len(loader)) for key,value in totals.items()},optimizer_steps=global_step,
             dataset_fingerprint=dataset.fingerprint,
             atom_catalog=dataset.codec_atoms)
        print('composer_epoch '+json.dumps({'epoch':epoch,'optimizer_steps':global_step,'elapsed_sec':time.monotonic()-started}),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
