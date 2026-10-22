"""Train chart_composer_v1 on structured authored events."""
from __future__ import annotations
import argparse,json,math,random,time
from contextlib import nullcontext
from itertools import islice
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from maichart.form.layout_prefetch import LayoutBatchPrefetch

from . import FAMILY,TRAINING_SEMANTICS
from .data import ComposerDataset,collate_composer
from .model import ChartComposer,ComposerConfig
from .checkpoint import save,load


class RemainingBatches:
    """Skip consumed indices without loading their audio/history again."""
    def __init__(self,batches,skip): self.batches,self.skip=batches,skip
    def __iter__(self): return islice(iter(self.batches),self.skip,None)
    def __len__(self): return max(0,len(self.batches)-self.skip)


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
    p.add_argument('--grad-accumulation',type=int,default=1)
    p.add_argument('--prefetch-batches',type=int,choices=range(0,17),default=2)
    p.add_argument('--log-interval',type=int,default=25)
    p.add_argument('--max-steps',type=int,default=0,help='maximum batches per epoch; 0 uses every batch')
    p.add_argument('--optimizer-implementation',choices=('auto','standard','foreach','fused'),default='auto')
    p.add_argument('--export-inference',action=argparse.BooleanOptionalAction,default=True)
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
    if output['action_logits'].ndim==3:
        action_logits,future,eos=output['action_logits'],output['future'],output['eos_logits']
    else:
        rows=torch.arange(target.size(0),device=target.device)
        action_logits=output['action_logits'][rows,target]
        future=output['future'][rows,target]
        eos=output['eos_logits'][rows,target]
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
    if args.batch_size<1 or args.grad_accumulation<1 or args.log_interval<1 or args.max_steps<0:
        raise ValueError('batch_size, grad_accumulation and log_interval must be positive; max_steps must be nonnegative')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    dataset=ComposerDataset(args.manifest,cache_dir=args.cache_dir,candidate_count=args.candidate_count,
        history_events=args.history_events,action_slots=args.action_slots,preparation_workers=args.prepare_workers,
        audio_cache_gb=args.audio_cache_gb,limit_charts=args.limit_charts,seed=args.seed)
    old=None
    if args.resume or args.init_checkpoint:
        model,old=load(args.resume or args.init_checkpoint,device='cpu',include_optimizer=bool(args.resume))
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
    loader_generator=torch.Generator().manual_seed(args.seed)
    implementation=args.optimizer_implementation
    if implementation=='auto': implementation='fused' if device.type=='cuda' else 'standard'
    if implementation=='fused' and device.type!='cuda':
        raise ValueError('fused optimizer requires CUDA; choose auto or standard for CPU')
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01,
        **({'fused':True} if implementation=='fused' else {'foreach':True} if implementation=='foreach' else {}))
    if args.resume:
        if not old or old.get('optimizer_state_dict') is None:
            raise ValueError('Inference-only weights cannot resume; choose finetune')
        optimizer.load_state_dict(old.pop('optimizer_state_dict'))
        for group in optimizer.param_groups:
            group['fused']=implementation=='fused';group['foreach']=implementation=='foreach'
    out=Path(args.checkpoint_dir);out.mkdir(parents=True,exist_ok=True)
    global_step=int(old.get('optimizer_steps',0)) if args.resume and old else 0
    resumed_state=(old.get('training_state') or {}) if args.resume and old else {}
    partial_resume=bool(resumed_state and not resumed_state.get('epoch_complete',True))
    if args.resume and old:
        for key in ('batch_size','grad_accumulation','max_steps'):
            stored=old.get('args',{}).get(key,1 if key=='grad_accumulation' else 0)
            if partial_resume and stored!=getattr(args,key):
                raise ValueError(f'Mid-epoch resume requires the saved {key}={stored}; use finetune to change it')
    start_epoch=(int(old.get('epoch',0))+(0 if partial_resume else 1)) if args.resume and old else 1
    end_epoch=start_epoch+args.epochs-1
    scaler=(torch.amp.GradScaler('cuda') if device.type=='cuda' and precision=='fp16' else None)
    if resumed_state:
        if resumed_state.get('loader_generator_state') is not None:
            loader_generator.set_state(resumed_state['loader_generator_state'])
        if resumed_state.get('torch_rng_state') is not None:
            torch.set_rng_state(resumed_state['torch_rng_state'])
        if device.type=='cuda' and resumed_state.get('cuda_rng_state') is not None:
            torch.cuda.set_rng_state(resumed_state['cuda_rng_state'],device)
        if scaler is not None and resumed_state.get('scaler'):
            scaler.load_state_dict(resumed_state['scaler'])
    parameters=sum(p.numel() for p in model.parameters())
    run_started=time.monotonic()
    timing=dict(data_sec=0.,transfer_sec=0.,compute_sec=0.,save_sec=0.)
    print('composer_runtime '+json.dumps({'family':FAMILY,'training_semantics':TRAINING_SEMANTICS,
        'device':str(device),'precision':precision,'samples':len(dataset),'parameters':parameters,
        'model_bytes':sum(p.numel()*p.element_size() for p in model.parameters()),
        'action_head_parameters':sum(p.numel() for p in model.action_head.parameters()),
        'selected_candidate_projection':True,'optimizer_implementation':implementation,
        'batch_size':args.batch_size,'grad_accumulation':args.grad_accumulation,
        'start_epoch':start_epoch,'prefetch_batches':args.prefetch_batches,
        'candidate_count':args.candidate_count,'action_vocab_size':len(dataset.codec_atoms)+1,
        'end_epoch':end_epoch,'candidate_misses':dataset.candidate_misses,
        'candidate_total':dataset.candidate_total,
        'candidate_miss_examples':dataset.candidate_miss_examples,
        'candidate_recall':1.-dataset.candidate_misses/max(1,dataset.candidate_total)}),flush=True)
    for epoch in range(start_epoch,end_epoch+1):
        dataset.set_epoch(epoch);started=time.monotonic();totals={};seen=0
        skip=int(resumed_state.get('batch_in_epoch',0)) if partial_resume and epoch==start_epoch else 0
        epoch_loader_state=loader_generator.get_state().clone()
        sampler=torch.utils.data.RandomSampler(dataset,generator=loader_generator)
        batch_sampler=torch.utils.data.BatchSampler(sampler,args.batch_size,drop_last=False)
        all_batches=len(batch_sampler)
        loader=DataLoader(dataset,batch_sampler=RemainingBatches(batch_sampler,skip),
            collate_fn=collate_composer,generator=loader_generator,pin_memory=device.type=='cuda')
        total_batches=min(all_batches,args.max_steps) if args.max_steps else all_batches
        source=LayoutBatchPrefetch(loader,depth=args.prefetch_batches) if args.prefetch_batches else nullcontext(iter(loader))
        optimizer.zero_grad(set_to_none=True)
        with source as batches:
            for batch_index in range(skip+1,total_batches+1):
                stamp=time.perf_counter();batch=next(batches)
                timing['data_sec']+=time.perf_counter()-stamp
                stamp=time.perf_counter()
                tensors={key:value.to(device,non_blocking=device.type=='cuda')
                         for key,value in batch.items() if torch.is_tensor(value)
                         and key not in {'song_index','event_index'}}
                timing['transfer_sec']+=time.perf_counter()-stamp
                stamp=time.perf_counter()
                with torch.autocast('cuda',dtype=torch.bfloat16 if precision=='bf16' else torch.float16,
                                    enabled=device.type=='cuda' and precision in {'bf16','fp16'}):
                    output=model(tensors['candidate_features'],tensors['history'],tensors['condition'],
                        action_state=tensors['action_state'],history_features=tensors['history_features'],
                        candidate_mask=tensors['candidate_padding_mask'],
                        history_padding_mask=tensors['history_padding_mask'],
                        candidate_indices=tensors['candidate_target'])
                    loss,metrics=_loss(output,tensors,model.config.atom_vocab_size)
                window_start=((batch_index-1)//args.grad_accumulation)*args.grad_accumulation
                window_batches=min(args.grad_accumulation,total_batches-window_start)
                backward_loss=loss/window_batches
                if scaler is not None: scaler.scale(backward_loss).backward()
                else: backward_loss.backward()
                update=(batch_index%args.grad_accumulation==0 or batch_index==total_batches)
                if update:
                    if scaler is not None: scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                    if scaler is not None: scaler.step(optimizer);scaler.update()
                    else: optimizer.step()
                    optimizer.zero_grad(set_to_none=True);global_step+=1
                timing['compute_sec']+=time.perf_counter()-stamp
                seen+=int(tensors['candidate_target'].numel())
                for key,value in metrics.items():
                    totals[key]=totals[key]+value.float() if key in totals else value.float().clone()
                if update and args.checkpoint_interval_steps and global_step%args.checkpoint_interval_steps==0:
                    stamp=time.perf_counter()
                    save(out/'composer_last.pt',model,optimizer,epoch=epoch,args=vars(args),manifest=args.manifest,
                         metrics={key:float(value)/(batch_index-skip) for key,value in totals.items()},optimizer_steps=global_step,
                         dataset_fingerprint=dataset.fingerprint,atom_catalog=dataset.codec_atoms,
                         training_state={'epoch_complete':batch_index==total_batches,'batch_in_epoch':batch_index,
                            'loader_generator_state':loader_generator.get_state() if batch_index==total_batches else epoch_loader_state,
                            'torch_rng_state':torch.get_rng_state(),
                            'cuda_rng_state':torch.cuda.get_rng_state(device) if device.type=='cuda' else None,
                            'scaler':scaler.state_dict() if scaler is not None else None})
                    timing['save_sec']+=time.perf_counter()-stamp
                    print('composer_checkpoint '+json.dumps({'epoch':epoch,'optimizer_steps':global_step,
                        'path':str(out/'composer_last.pt'),'save_sec':timing['save_sec']}),flush=True)
                if batch_index%args.log_interval==0 or batch_index in {1,total_batches}:
                    avg={key:float(value)/(batch_index-skip) for key,value in totals.items()}
                    elapsed=time.monotonic()-started
                    memory=({'allocated_gib':torch.cuda.memory_allocated(device)/1024**3,
                        'reserved_gib':torch.cuda.memory_reserved(device)/1024**3,
                        'peak_allocated_gib':torch.cuda.max_memory_allocated(device)/1024**3} if device.type=='cuda' else {})
                    print('composer_progress '+json.dumps({'epoch':epoch,'batch':batch_index,'total':total_batches,
                        'optimizer_steps':global_step,'elapsed_sec':time.monotonic()-run_started,
                        'phase_elapsed_sec':elapsed,'batches_per_sec':(batch_index-skip)/max(elapsed,1e-6),
                        'samples_per_sec':seen/max(elapsed,1e-6),'eta_sec':(total_batches-batch_index)*elapsed/(batch_index-skip),
                        'metric_batches':batch_index-skip,'resume_skipped_batches':skip,
                        'timing':dict(timing),'cache':dataset.cache_stats(),'cuda_memory':memory,**avg}),flush=True)
        stamp=time.perf_counter()
        common=dict(epoch=epoch,args=vars(args),manifest=args.manifest,
            metrics={key:float(value)/max(1,total_batches-skip) for key,value in totals.items()},optimizer_steps=global_step,
            dataset_fingerprint=dataset.fingerprint,atom_catalog=dataset.codec_atoms,
            training_state={'epoch_complete':True,'batch_in_epoch':total_batches,
                'loader_generator_state':loader_generator.get_state(),'torch_rng_state':torch.get_rng_state(),
                'cuda_rng_state':torch.cuda.get_rng_state(device) if device.type=='cuda' else None,
                'scaler':scaler.state_dict() if scaler is not None else None})
        save(out/'composer_last.pt',model,optimizer,**common)
        if args.export_inference: save(out/'composer_inference.pt',model,None,**common)
        timing['save_sec']+=time.perf_counter()-stamp
        print('composer_epoch '+json.dumps({'epoch':epoch,'optimizer_steps':global_step,
            'elapsed_sec':time.monotonic()-started,'timing':dict(timing)}),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
