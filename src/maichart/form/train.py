"""Independent draft/reconstruction + whole-song blueprint training entry."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F

from maichart.form.data import FormDataset, collate_form, form_audio
from maichart.form.model import FormConfig, FormTransformer
from maichart.form import checkpoint
from maichart.generation.conditions import GenerationCondition, GENERATION_DEFAULTS


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--checkpoint-dir',required=True)
    p.add_argument('--cache-dir',default='cache/song_form');p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--device',default='cpu');p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--lr',type=float,default=0.0002);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--max-steps',type=int,default=0);p.add_argument('--limit-charts',type=int,default=0)
    p.add_argument('--d-model',type=int,default=256);p.add_argument('--layers',type=int,default=4)
    p.add_argument('--nhead',type=int,default=8);p.add_argument('--max-tokens',type=int,default=2048)
    p.add_argument('--dropout',type=float,default=0.1)
    p.add_argument('--weight-decay',type=float,default=0.01)
    p.add_argument('--grad-accumulation',type=int,default=1)
    p.add_argument('--save-every',type=int,default=0)
    p.add_argument('--condition-drop-prob',type=float,default=0.1)
    p.add_argument('--blueprint-loss-weight',type=float,default=1.0)
    p.add_argument('--load-curve-loss-weight',type=float,default=1.0)
    p.add_argument('--relation-loss-weight',type=float,default=0.1)
    p.add_argument('--quality-enabled',dest='quality_enabled',action='store_true',default=True)
    p.add_argument('--no-quality-enabled',dest='quality_enabled',action='store_false')
    p.add_argument('--max-memory-tokens',type=int,default=512);p.add_argument('--max-region-seconds',type=float,default=14.)
    p.add_argument('--meter-map',help='Explicit same meter map for this run; per-song music_structure.json can override')
    p.add_argument('--init-checkpoint');p.add_argument('--resume')
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    if args.epochs<1 or args.batch_size<1 or args.lr<=0 or args.max_steps<0 or args.grad_accumulation<1 or args.weight_decay<0 or not 0<=args.condition_drop_prob<=1: raise ValueError('Invalid training limits')
    if args.resume and args.init_checkpoint: raise ValueError('Choose resume or finetune, not both')
    meters=json.loads(Path(args.meter_map).read_text(encoding='utf-8')) if args.meter_map else None
    device=torch.device(args.device)
    old=None
    if args.resume or args.init_checkpoint:
        model,blueprint,codec,old=checkpoint.load(args.resume or args.init_checkpoint,device=device)
        args.max_tokens=model.config.max_tokens;args.max_memory_tokens=model.config.max_memory_tokens
        if args.resume:
            saved_args=old.get('args',{})
            # Resume is an exact continuation: optimizer and data-affecting
            # controls come from the checkpoint rather than a stale UI form.
            for name in ('seed','dropout','weight_decay','grad_accumulation','condition_drop_prob',
                         'blueprint_loss_weight','load_curve_loss_weight','relation_loss_weight',
                         'quality_enabled','max_region_seconds','limit_charts'):
                if name in saved_args and hasattr(args,name):
                    setattr(args,name,saved_args[name])
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    print('form_prepare '+json.dumps({'stage':'dataset','manifest':str(Path(args.manifest).resolve()),
        'limit_charts':args.limit_charts,'max_steps':args.max_steps}),flush=True)
    dataset=FormDataset(args.manifest,cache_dir=args.cache_dir,max_tokens=args.max_tokens,
        max_memory_tokens=args.max_memory_tokens,meters=meters,seed=args.seed,limit_charts=args.limit_charts,
        max_region_seconds=args.max_region_seconds,quality_enabled=args.quality_enabled)
    source_digest=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()
    contract={"manifest_sha256":source_digest,"meter_map":meters,"max_region_seconds":args.max_region_seconds,
              "charts":dataset.audit['charts']}
    if args.resume and old.get('dataset_contract')!=contract: raise ValueError('Resume data changed; use --init-checkpoint')
    from maichart.form.blueprint import BlueprintConfig, SongBlueprint, build_blueprint_targets, compute_blueprint_loss
    if old is None:
        codec=dataset.codec
        model=FormTransformer(FormConfig(vocab_size=len(codec),d_model=args.d_model,nhead=args.nhead,dropout=args.dropout,
            layers=args.layers,dim_feedforward=4*args.d_model,max_tokens=args.max_tokens,
            max_memory_tokens=args.max_memory_tokens)).to(device)
        blueprint=SongBlueprint(BlueprintConfig(width=max(16,args.d_model//2))).to(device)
    parameters=list(model.parameters())+list(blueprint.parameters())
    optimizer=torch.optim.AdamW(parameters,lr=args.lr,weight_decay=args.weight_decay)
    start=old.get('epoch',0) if args.resume else 0
    steps=old.get('optimizer_steps',0) if old else 0
    plan_steps=old.get('blueprint_steps',0) if old else 0
    task_training=dict(old.get('task_training',{})) if old else {'draft':0,'revision':0}
    if args.resume:
        optimizer.load_state_dict(old['optimizer_state_dict'])
        rng=old.get('rng_state',{})
        if rng:
            random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch'].cpu())
            if device.type=='cuda' and rng.get('cuda'):torch.cuda.set_rng_state_all(rng['cuda'])
    output=Path(args.checkpoint_dir);output.mkdir(parents=True,exist_ok=True)
    if old and Path(args.resume or args.init_checkpoint).resolve().parent==output.resolve() and args.init_checkpoint:
        raise ValueError('Finetune uses a separate output directory to preserve the source model')
    (output/'dataset_audit.json').write_text(json.dumps(dataset.audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'form_training charts={len(dataset.songs)} examples={len(dataset)} device={device} parameters={sum(p.numel() for p in parameters)}',flush=True)
    loader=DataLoader(dataset,batch_size=args.batch_size,shuffle=True,collate_fn=collate_form)
    first_items={}
    for i,item in enumerate(dataset.items):first_items.setdefault(item[0],i)
    history=[]
    for epoch in range(start+1,start+args.epochs+1):
        dataset.seed=args.seed+epoch*100003
        model.train();blueprint.train();losses=[];plan_losses=[]
        plan_total=min(len(dataset.songs),args.max_steps) if args.max_steps else len(dataset.songs)
        layout_total=min(len(loader),args.max_steps) if args.max_steps else len(loader)
        last_progress=0.
        def progress(phase,completed,total,force=False):
            nonlocal last_progress
            now=time.monotonic()
            if not force and completed not in (1,total) and now-last_progress<10:return
            overall=(completed if phase=='blueprint' else plan_total+completed)/max(1,plan_total+layout_total)
            print('form_progress '+json.dumps({'epoch':epoch,'phase':phase,'completed':completed,
                'total':total,'overall_fraction':overall}),flush=True)
            last_progress=now
        # Planner sees the entire song before event layout. No 16-beat resets.
        for song_index,(song,points,level,weight,_,condition,*_) in enumerate(dataset.songs):
            if args.max_steps and song_index>=args.max_steps: break
            if song_index not in first_items:
                print(f'form_skip_blueprint chart={song_index} reason=no_usable_configuration_examples',flush=True)
                continue
            audio=form_audio(song,song.spans[0].start,song.spans[-1].end).unsqueeze(0).to(device)
            from maichart.generation.event_audio import segment_frame_times
            from maichart.event_timing import seconds_at_beat
            stamps=segment_frame_times(seconds_at_beat(song.spans[0].start,song.tempo,song.first),seconds_at_beat(song.spans[-1].end,song.tempo,song.first))
            form=dataset[first_items[song_index]]['form'].unsqueeze(0).to(device)
            condition=condition.to_vector(device=device).unsqueeze(0)
            targets=build_blueprint_targets(points,song.tempo,song.first,stamps)
            try:
                from maichart.form.curve import resample_load_curve
                curve_times,curve_values=dataset.songs[song_index][0].load_curve_times_sec,dataset.songs[song_index][0].load_curve
                target_curve=torch.from_numpy(resample_load_curve(curve_times,curve_values,np.asarray(stamps)-float(song.first)))
                targets['load_curve_target']=target_curve
                targets['load_curve_mask']=torch.ones_like(target_curve,dtype=torch.bool)
            except (ImportError,TypeError,ValueError):
                pass
            targets={key:value.unsqueeze(0).to(device) for key,value in targets.items()}
            optimizer.zero_grad(set_to_none=True)
            prediction=blueprint(audio,form,condition)
            loss=compute_blueprint_loss(prediction,targets,load_curve_weight=args.load_curve_loss_weight)['loss']*weight*args.blueprint_loss_weight
            if not torch.isfinite(loss): raise ValueError('Nonfinite full-song planner loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(parameters,1.);optimizer.step();plan_steps+=1
            plan_losses.append(float(loss.detach()))
            progress('blueprint',song_index+1,plan_total)
        for batch_index,batch in enumerate(loader):
            if args.max_steps and batch_index>=args.max_steps: break
            batch={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
            if args.condition_drop_prob > 0:
                batch['condition_null'] = (torch.rand(batch['condition'].size(0), device=device) < args.condition_drop_prob)
            optimizer.zero_grad(set_to_none=True)
            prediction=model(batch)
            ce=F.cross_entropy(prediction['logits'].transpose(1,2),batch['target'],ignore_index=0,reduction='none')
            valid=batch['target']!=0
            token_loss=((ce*valid).sum(1)/valid.sum(1).clamp_min(1)*batch['weight']).mean()
            relation_loss=F.binary_cross_entropy_with_logits(prediction['relation_logits'],batch['relation_target'],reduction='none').mean(1)
            mask=batch['relation_valid'].float()
            loss=token_loss+args.relation_loss_weight*(relation_loss*mask).sum()/mask.sum().clamp_min(1)
            if not torch.isfinite(loss): raise ValueError('Nonfinite form loss')
            (loss/args.grad_accumulation).backward()
            if (batch_index+1)%args.grad_accumulation==0 or batch_index+1==layout_total:
                torch.nn.utils.clip_grad_norm_(parameters,1.);optimizer.step();optimizer.zero_grad(set_to_none=True);steps+=1
            losses.append(float(loss.detach()))
            for task_id,name in ((0,'draft'),(1,'revision')):
                task_training[name]=task_training.get(name,0)+int((batch['task']==task_id).sum())
            progress('layout_revision',batch_index+1,layout_total)
        metrics={'epoch':epoch,'draft_revision_loss':float(np.mean(losses)),'whole_song_plan_loss':float(np.mean(plan_losses)),
                 'optimizer_steps':steps,'blueprint_steps':plan_steps,'task_training':dict(task_training)}
        history.append(metrics)
        atoms=sorted({atom for _,points,*_ in dataset.songs for p in points for atom in _atoms(p.note_text)}
                     | set(old.get('atom_catalog',[]) if old else []))
        saved=checkpoint.payload(model,blueprint,codec,epoch=epoch,optimizer_steps=steps,blueprint_steps=plan_steps,
            optimizer_state_dict=optimizer.state_dict(),dataset_contract=contract,args=vars(args),metrics=metrics,
            atom_catalog=atoms,task_training=dict(task_training),capabilities={'draft':{'steps':task_training.get('draft',0),'contract':'form_condition_v2'},'revision':{'steps':task_training.get('revision',0),'contract':'form_condition_v2'},'load_curve':{'steps':plan_steps,'contract':'load_curve_v1'}},condition_supervision='event_derived_radar_style_v1',
            supervision='authored_draft_and_masked_reconstruction_not_aesthetic_reward',
            rng_state={'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
                       'cuda':torch.cuda.get_rng_state_all() if device.type=='cuda' else None})
        temporary=output/'form_last.tmp';torch.save(saved,temporary);temporary.replace(output/'form_last.pt')
        if args.save_every and epoch % args.save_every == 0:
            torch.save(saved, output / f'form_epoch_{epoch:03d}.pt')
        (output/'training_history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        print('form_epoch '+json.dumps(metrics),flush=True)
    return 0


def _atoms(text):
    from maichart.notes import _split_top_level_slashes
    return _split_top_level_slashes(text)


if __name__=='__main__': raise SystemExit(main())
