"""Train the independent continuous whole-song chart stream family."""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import math
import random
import time
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from maichart.form.quality_profile import MUSICAL_EVIDENCE
from .data import StreamDataset, collate_stream, TRAINING_SEMANTICS
from .model import ChartStreamTransformer, StreamConfig
from .checkpoint import save, load, upgrade_grounding
from maichart.generation.event_cli import _NoteGrammar
from maichart.form.codec import RationalCodec
from maichart.form.output_guard import training_output_guard
from .runtime import resolve_device,training_cpu_budget,attention_backend,CudaBatchTimer
from .training_graph import install_training_graphs,MetricAccumulator


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
    p.add_argument('--curve-slope-loss-weight',type=float,default=.05)
    p.add_argument('--event-decision-loss-weight',type=float,default=.25)
    p.add_argument('--checkpoint-interval-steps',type=int,default=500)
    p.add_argument('--quality-enabled',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--grounding-profile',choices=('auto','none','causal_audio_v1'),default='auto')
    p.add_argument('--detail-audio-nodes',type=int,default=8192)
    p.add_argument('--local-audio-neighbors',type=int,default=32)
    p.add_argument('--local-audio-seconds',type=float,default=2.)
    p.add_argument('--archive-events',type=int,default=4096)
    p.add_argument('--archive-retrievals',type=int,default=8)
    p.add_argument('--archive-min-similarity',type=float,default=.5)
    p.add_argument('--history-note-dropout',type=float,default=.15)
    p.add_argument('--prefix-note-dropout',type=float,default=.1)
    p.add_argument('--alignment-loss-weight',type=float,default=.15)
    p.add_argument('--alignment-tolerance-ms',type=float,default=15.)
    p.add_argument('--event-balance-loss-weight',type=float,default=.2)
    p.add_argument('--continuation-profile',choices=('auto','reset_v1','overlap_v1'),default='auto')
    p.add_argument('--context-replay-tokens',type=int,default=256)
    p.add_argument('--prefix-mask-span-events',type=int,default=8)
    p.add_argument('--cycle-exit-loss-weight',type=float,default=.25)
    p.add_argument('--symmetry-augmentation',choices=('none','mirror_lr','mirror4'),default='mirror_lr')
    p.add_argument('--prefetch-batches',type=int,default=2)
    p.add_argument('--data-workers',type=int,default=4)
    p.add_argument('--cpu-threads',type=int,default=0)
    p.add_argument('--input-cache-mode',choices=('mapped','legacy'),default='mapped')
    p.add_argument('--structure-cache-mb',type=int,default=256)
    p.add_argument('--optimizer-implementation',choices=('auto','standard','fused'),default='auto')
    p.add_argument('--attention-backend',choices=('auto','cudnn','efficient'),default='auto')
    p.add_argument('--execution-mode',choices=('auto','eager','cuda_graph'),default='auto')
    p.add_argument('--cuda-graph-budget-gb',type=float,default=2.)
    p.add_argument('--input-residency',choices=('auto','mapped','resident'),default='auto')
    p.add_argument('--resident-cache-gb',type=float,default=0.)
    p.add_argument('--resident-memory-reserve-gb',type=float,default=0.)
    p.add_argument('--cache-validation-records',type=int,default=0)
    return p


def _precision(value,device):
    if value == 'auto':
        return 'bf16' if device.type == 'cuda' and torch.cuda.is_bf16_supported() else 'fp32'
    if device.type != 'cuda' and value != 'fp32': raise ValueError('CPU stream training requires fp32')
    if value == 'bf16' and not torch.cuda.is_bf16_supported(): raise ValueError('bf16 unsupported')
    return value


_CODEC = RationalCodec()


def _loss(output,batch,plan_weight=.15,event_weight=.25,alignment_weight=.15,event_balance_weight=.2,
          cycle_exit_weight=0.,curve_slope_weight=.05):
    logits=output['logits'].float(); target=batch['target']; mask=~batch['target_padding_mask']
    local = batch.get('target_quality', torch.ones_like(target, dtype=torch.float32))
    quality = batch.get('quality_weight', logits.new_ones(logits.size(0)))[:, None]
    mask = mask & (local > 0)
    def reduce(values, valid, local_weight):
        # Quality is a gradient multiplier, including batch_size=1. Never
        # borrow another objective's denominator or divide by a zero weight.
        return (values * valid * local_weight * quality).sum()/valid.sum().clamp_min(1)
    ce=F.cross_entropy(logits.reshape(-1,logits.size(-1)),target.reshape(-1),ignore_index=0,reduction='none').reshape_as(target)
    codec=_CODEC
    from .objectives import event_balanced_loss,alignment_loss
    token=reduce(ce, mask, local)
    plan_mask=batch['plan_mask'] & ~batch['plan_mask_padding_mask'] & mask
    plan_labels=batch['plan_labels'].float()
    if output['plan'].shape != plan_labels.shape:
        raise ValueError(f"stream plan shape mismatch: output={tuple(output['plan'].shape)} "
                         f"labels={tuple(plan_labels.shape)}")
    plan=F.smooth_l1_loss(output['plan'].float(),plan_labels,reduction='none').mean(-1)
    plan=reduce(plan, plan_mask, local)
    curve_mask=~batch['curve_target_padding_mask'] & ~batch['audio_padding_mask']
    curve_target=batch['curve_target'].to(output['curve'])
    curve_prediction=output['curve'].float().sigmoid()
    curve=F.smooth_l1_loss(curve_prediction,curve_target.float(),reduction='none').mean(-1)
    curve_quality=batch.get('curve_quality', torch.ones_like(curve))
    if 'curve_time_weight' in batch:
        time_weight=batch['curve_time_weight']*curve_mask*(curve_quality>0)
        curve=((curve*time_weight*curve_quality).sum(1)/time_weight.sum(1).clamp_min(1e-9)
               *quality[:,0]*batch['curve_example_weight']).mean()
    else:curve=reduce(curve, curve_mask & (curve_quality > 0), curve_quality)
    curve_slope=curve_prediction.sum()*0.
    if curve_slope_weight and curve_prediction.size(1)>1:
        pair_mask=(curve_mask[:,:-1]&curve_mask[:,1:]&
                   (curve_quality[:,:-1]>0)&(curve_quality[:,1:]>0))
        pair_quality=torch.minimum(curve_quality[:,:-1],curve_quality[:,1:])
        pair_loss=F.smooth_l1_loss(curve_prediction[:,1:]-curve_prediction[:,:-1],
                                   curve_target[:,1:]-curve_target[:,:-1],reduction='none').mean(-1)
        pair_weight=pair_quality*pair_mask
        curve_slope=(pair_loss*pair_weight*quality).sum()/pair_weight.sum().clamp_min(1.)
    decision_mask=mask & ((batch['decoder_input']==codec.bos_id) | (batch['decoder_input']==codec.end_id))
    decision_mask &= (target==codec.at_id) | (target==codec.eos_id)
    choices=torch.stack((logits[...,codec.at_id],logits[...,codec.eos_id]),-1)
    decision_values=F.cross_entropy(choices.reshape(-1,2),(target==codec.eos_id).long().reshape(-1),reduction='none').reshape_as(target)
    decision=reduce(decision_values,decision_mask,local)
    prediction=logits.argmax(-1)
    eos_mask=mask & (target==codec.eos_id)
    position_mask=mask & (target>=codec.ids['P:0']) & (target<=codec.ids['P:/'])
    note_mask=mask & (target>=codec.ids['N:1'])
    balanced,extra=event_balanced_loss(ce,logits,batch,codec,predicted=prediction)
    alignment=logits.sum()*0
    if 'alignment_event_logits' in output:
        alignment,metrics=alignment_loss(output,batch);extra.update(metrics)
    exit_mask=batch.get('cycle_exit_mask',torch.zeros_like(mask)) & mask
    # This is an extra weight on authored exit tokens in the main objective.
    # Share its valid-token denominator: averaging only over sparse exits
    # would multiply each marked token's gradient by valid_count/exit_count.
    cycle_exit=(ce*exit_mask*local*quality).sum()/mask.sum().clamp_min(1)
    total=(token + plan_weight*plan + .2*curve + curve_slope_weight*curve_slope
           + event_weight*decision + event_balance_weight*balanced + alignment_weight*alignment
           + cycle_exit_weight*cycle_exit)
    return total, {**extra,'token_loss':token.detach(),'plan_loss':plan.detach(),'curve_loss':curve.detach(),
                               'event_balance_loss':balanced.detach(),'alignment_loss':alignment.detach(),
                               'event_decision_loss':decision.detach(),
                               'cycle_exit_loss':cycle_exit.detach(),
                               'curve_slope_loss':curve_slope.detach(),
                               '_cycle_exit_count':exit_mask.sum().detach(),
                               '_cycle_exit_correct':((prediction==target)&exit_mask).sum().detach(),
                               'loss':total.detach(),
                               'token_accuracy':((prediction==target)&mask).sum().float()/mask.sum().clamp_min(1),
                               '_eos_count':eos_mask.sum().detach(),
                               '_eos_correct':((prediction==target)&eos_mask).sum().detach(),
                               '_eos_loss_sum':(ce*eos_mask).sum().detach(),
                               '_position_count':position_mask.sum().detach(),
                               '_position_correct':((prediction==target)&position_mask).sum().detach(),
                               '_note_count':note_mask.sum().detach(),
                               '_note_correct':((prediction==target)&note_mask).sum().detach()}


def _finish_accumulation(model, optimizer, scaler, pending, accumulation):
    scaler.unscale_(optimizer)
    # A short final group must average its actual batches, not be weakened by
    # the configured group size (for example two batches divided by eight).
    if pending != accumulation:
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(accumulation/pending)
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    execution=getattr(model,'_training_graph_runner',None)
    optimizer.zero_grad(set_to_none=not bool(execution and execution.slots))
    return scaler.get_scale() >= before


def _mean_metrics(totals, batches):
    if isinstance(totals,MetricAccumulator):totals=totals.as_dict()
    if not totals:
        return {}
    values=torch.stack([torch.as_tensor(value) for value in totals.values()]).detach().float().cpu().tolist()
    raw=dict(zip(totals,values))
    result={key:value/max(1,batches) for key,value in raw.items() if not key.startswith('_')}
    for prefix in ('eos','position','note','whole_position','whole_configuration','whole_event','alignment','cycle_exit'):
        count=raw.get('_'+prefix+'_count')
        if count is not None:
            result[prefix+'_targets']=int(count)
            result[prefix+'_accuracy']=raw['_'+prefix+'_correct']/count if count else None
    if '_eos_count' in raw:
        result['eos_token_loss']=raw['_eos_loss_sum']/raw['_eos_count'] if raw['_eos_count'] else None
    return result


def _resume_start_epoch(checkpoint):
    epoch=int(checkpoint.get('epoch',0))
    position=checkpoint.get('training_position') or {}
    # A periodic checkpoint is saved inside its current epoch. Until sampler
    # and RNG state are persisted, replay that unfinished epoch explicitly;
    # starting at epoch+1 would silently omit its remaining targets and EOS.
    return epoch if position.get('complete') is True else max(0,epoch-1)


def validate_training_args(args):
    if getattr(args,'attention_backend','auto')!='auto' and str(args.device)=='cpu':
        raise ValueError('CPU 注意力计算后端必须选择自动')
    if getattr(args,'execution_mode','auto')=='cuda_graph' and str(args.device)=='cpu':
        raise ValueError('CUDA Graph 训练执行方式需要 CUDA，CPU 请选择自动或普通执行')
    if not math.isfinite(getattr(args,'cuda_graph_budget_gb',2.)) or not 0<=getattr(args,'cuda_graph_budget_gb',2.)<=32:
        raise ValueError('CUDA Graph 额外显存预算必须为 0–32 GiB')
    if getattr(args,'symmetry_augmentation','mirror_lr') not in {'none','mirror_lr','mirror4'}:
        raise ValueError('symmetry_augmentation 必须为 none、mirror_lr 或 mirror4')
    if not 1<=args.context_replay_tokens<=65536:raise ValueError('context_replay_tokens 必须为1–65536')
    if not 1<=args.prefix_mask_span_events<=64:raise ValueError('prefix_mask_span_events 必须为1–64')
    if not math.isfinite(args.cycle_exit_loss_weight) or not 0<=args.cycle_exit_loss_weight<=1:
        raise ValueError('cycle_exit_loss_weight 必须为0–1')
    for name,maximum in (('plan_loss_weight',2.),('curve_slope_loss_weight',1.),
                         ('event_decision_loss_weight',10.),('weight_decay',1.)):
        value=getattr(args,name)
        if not math.isfinite(value) or not 0 <= value <= maximum:
            raise ValueError(f'{name} 必须是 0–{maximum:g} 的有限数值')
    for name in ('epochs','batch_size','grad_accumulation','log_interval','d_model','layers','nhead',
                 'dim_feedforward','max_tokens','max_history_tokens','max_stream_memory'):
        if getattr(args,name) < 1:
            raise ValueError(f'{name} 必须是正整数')
    for name in ('prepare_workers','limit_charts','checkpoint_interval_steps','seed'):
        if getattr(args,name) < 0:
            raise ValueError(f'{name} 必须是非负整数')
    for name in ('lr','max_region_seconds','audio_cache_gb'):
        value=getattr(args,name)
        if not math.isfinite(value) or value < 0 or (name != 'audio_cache_gb' and value == 0):
            raise ValueError(f'{name} 超出有效范围')
    if args.seed >= 2**32:
        raise ValueError('seed 必须小于 2**32')
    if args.max_region_seconds < 2:
        raise ValueError('max_region_seconds 必须至少为 2 秒，与音频准备器最小区域一致')
    if not math.isfinite(args.dropout) or not 0 <= args.dropout < 1:
        raise ValueError('dropout 必须是 [0,1) 的有限数值')
    if args.d_model % args.nhead:
        raise ValueError('d_model 必须能被 nhead 整除')
    if args.resume and args.init_checkpoint:
        raise ValueError('resume 和 init-checkpoint 不能同时使用')
    for key in ('detail_audio_nodes','local_audio_neighbors','archive_events','archive_retrievals'):
        value=getattr(args,key)
        if not 1<=value<=65536:raise ValueError(f'{key} 必须是1–65536整数')
    for key in ('history_note_dropout','prefix_note_dropout'):
        value=getattr(args,key)
        if not math.isfinite(value) or not 0<=value<=.8:raise ValueError(f'{key} 必须为0–0.8')
    for key in ('alignment_loss_weight','event_balance_loss_weight'):
        value=getattr(args,key)
        if not math.isfinite(value) or not 0<=value<=3:raise ValueError(f'{key} 必须为0–3')
    if not math.isfinite(args.local_audio_seconds) or not .05<=args.local_audio_seconds<=30:
        raise ValueError('local_audio_seconds 必须为0.05–30')
    if not math.isfinite(args.archive_min_similarity) or not -1<=args.archive_min_similarity<=1:
        raise ValueError('archive_min_similarity 必须为-1–1')
    if not math.isfinite(args.alignment_tolerance_ms) or not 1<=args.alignment_tolerance_ms<=100:
        raise ValueError('alignment_tolerance_ms 必须为1–100')
    if not 0<=args.prefetch_batches<=16:raise ValueError('prefetch_batches 必须为0–16')
    if not 1<=args.data_workers<=8:raise ValueError('data_workers 必须为1–8')
    if not 0<=args.cpu_threads<=64:raise ValueError('cpu_threads 必须为0–64')
    if not 0<=args.structure_cache_mb<=8192:raise ValueError('structure_cache_mb 必须为0–8192')
    for name in ('resident_cache_gb','resident_memory_reserve_gb'):
        if not math.isfinite(getattr(args,name)) or not 0<=getattr(args,name)<=1024:
            raise ValueError(f'{name} 必须为0–1024的有限数值')
    if not 0<=args.cache_validation_records<=1_000_000:
        raise ValueError('cache_validation_records 必须为0–1000000')


def validate_output_directory(args):
    out=Path(args.checkpoint_dir).resolve()
    existing=out.is_dir() and next(out.glob('*.pt'),None) is not None
    if existing and not (args.resume and Path(args.resume).resolve().parent == out):
        raise ValueError('输出目录已有权重；fresh / finetune 请使用新输出目录，避免覆盖原模型。')
    if args.init_checkpoint and Path(args.init_checkpoint).resolve().parent == out:
        raise ValueError('finetune 必须使用新输出目录。')


def _validate_resume_data(old, dataset):
    if old.get('dataset_fingerprint') != dataset.fingerprint:
        raise ValueError('Resume data changed（谱面、音频、标注或上下文合同变化）；请使用 finetune / --init-checkpoint 和新输出目录。')


def main(argv=None):
    args=parser().parse_args(argv)
    validate_training_args(args)
    with training_output_guard(args.checkpoint_dir):
        validate_output_directory(args)
        with training_cpu_budget(args.cpu_threads,resolve_device(args.device)) as budget:
            args.cpu_budget=budget
            with attention_backend(args.attention_backend,resolve_device(args.device)) as policy:
                args.attention_policy=policy
                return _train(args)


def _migrate_plan_semantics(model, checkpoint):
    """Reset absolute-phase rows only when importing a pre-v12 contract."""
    semantics=(checkpoint.get('args') or {}).get('training_semantics','')
    prefix='chart_stream_training_v'
    version=str(semantics).removeprefix(prefix).split('_',1)[0]
    if str(semantics).startswith(prefix) and version.isdigit() and int(version)>=12:
        return False
    # Before v12, channels 1:3 were absolute beat phase. From v12 onward
    # they are relative motion, including v13's loss-normalization update.
    with torch.no_grad():
        model.plan_head.weight[1:3].zero_(); model.plan_head.bias[1:3].zero_()
        if hasattr(model,'grounding'):
            model.grounding.intent_projection.weight[:,1:3].zero_()
    return True


def _train(args):
    args.training_semantics = TRAINING_SEMANTICS
    device=resolve_device(args.device)
    precision=_precision(args.precision,device); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    old=None
    semantic_migration=False
    if args.resume or args.init_checkpoint:
        model,old=load(args.resume or args.init_checkpoint,device='cpu')
        if args.resume and old.get('plan_dim_migration'):
            raise ValueError('plan_dim=8 checkpoint migration cannot resume its old optimizer; use --init-checkpoint with a new output directory.')
        if args.resume and (old.get('args') or {}).get('training_semantics') != TRAINING_SEMANTICS:
            raise ValueError('训练合同已更新为有界循环退出监督；旧训练请使用 finetune / --init-checkpoint 和新输出目录。')
        if args.resume:
            for name in ('event_decision_loss_weight','plan_loss_weight','curve_slope_loss_weight',
                         'alignment_loss_weight','event_balance_loss_weight',
                         'history_note_dropout','prefix_note_dropout','prefix_mask_span_events','cycle_exit_loss_weight',
                         'symmetry_augmentation'):
                setattr(args,name,(old.get('args') or {}).get(name,getattr(args,name)))
            if not old.get('optimizer_state_dict'):
                raise ValueError('Resume checkpoint 缺少 optimizer 状态；请使用 finetune。')
            groups=old['optimizer_state_dict']['param_groups']
            args.lr=float(groups[0]['lr']); args.weight_decay=float(groups[0]['weight_decay'])
        args.d_model=model.config.d_model; args.layers=model.config.layers; args.nhead=model.config.nhead
        args.dim_feedforward=model.config.dim_feedforward; args.dropout=model.config.dropout
        args.max_tokens=model.config.max_tokens; args.max_history_tokens=model.config.max_history_tokens
        args.max_stream_memory=model.config.max_stream_memory
        if args.init_checkpoint:
            semantic_migration=_migrate_plan_semantics(model,old)
    grounding_keys=('grounding_profile','detail_audio_nodes','local_audio_neighbors','local_audio_seconds',
                    'archive_events','archive_retrievals','archive_min_similarity','alignment_tolerance_ms',
                    'continuation_profile','context_replay_tokens')
    if args.resume:
        for key in grounding_keys:setattr(args,key,getattr(model.config,key))
    else:
        if args.grounding_profile=='auto':args.grounding_profile=model.config.grounding_profile if old is not None else 'causal_audio_v1'
        if args.continuation_profile=='auto':
            args.continuation_profile=(model.config.continuation_profile if old is not None
                else 'overlap_v1' if args.grounding_profile!='none' else 'reset_v1')
        if old is not None:
            if model.config.grounding_profile!='none' and args.grounding_profile=='none':
                raise ValueError('已有音频接地模块不能在finetune中静默删除')
            model=upgrade_grounding(model,**{key:getattr(args,key) for key in grounding_keys})
    if args.grounding_profile=='none':
        args.history_note_dropout=args.prefix_note_dropout=args.alignment_loss_weight=0.
    config=(model.config if old is not None else StreamConfig(vocab_size=len(RationalCodec()),d_model=args.d_model,
        nhead=args.nhead,layers=args.layers,dim_feedforward=args.dim_feedforward,dropout=args.dropout,
        max_tokens=args.max_tokens,max_history_tokens=args.max_history_tokens,max_stream_memory=args.max_stream_memory,
        **{key:getattr(args,key) for key in grounding_keys}))
    dataset=StreamDataset(args.manifest,cache_dir=args.cache_dir,max_tokens=args.max_tokens,
        max_history_tokens=args.max_history_tokens,max_region_seconds=args.max_region_seconds,
        preparation_workers=args.prepare_workers,audio_cache_gb=args.audio_cache_gb,limit_charts=args.limit_charts,
        quality_enabled=args.quality_enabled,grounding_config=config,prefix_mask_span_events=args.prefix_mask_span_events,
        history_note_dropout=args.history_note_dropout,prefix_note_dropout=args.prefix_note_dropout,seed=args.seed,
        input_cache_mode=args.input_cache_mode,structure_cache_mb=args.structure_cache_mb,
        symmetry_augmentation=getattr(args,'symmetry_augmentation','none'))
    if not len(dataset):
        raise ValueError('没有可监督的连续流训练样本；请检查质量排除标注和数据集。')
    if args.resume:
        _validate_resume_data(old,dataset)
    print('stream_dataset '+json.dumps({'training_semantics': TRAINING_SEMANTICS,
        'charts': len(dataset.songs), 'chunks': len(dataset),
        'targets': sum(len(chunk.target) for chunk in dataset.chunks),
        'eos_targets': sum(chunk.target.count(dataset.codec.eos_id) for chunk in dataset.chunks),
        'supervised_tokens': sum(sum(w>0 for w in chunk.target_quality) for chunk in dataset.chunks),
        'dataset_fingerprint': dataset.fingerprint}),flush=True)
    if args.input_residency in {'auto','resident'} and args.input_cache_mode=='mapped':
        warmup=dataset.warm_input_cache(
            report=lambda **value:print('stream_prepare '+json.dumps(value),flush=True))
        print('stream_prepare '+json.dumps({'stage':'warmup_complete',**warmup}),flush=True)
    residency=dataset.prepare_residency(policy=args.input_residency,
        budget_bytes=int(args.resident_cache_gb*1024**3) if args.resident_cache_gb else None,
        reserve_bytes=int(args.resident_memory_reserve_gb*1024**3),
        validation_records=args.cache_validation_records,
        report=lambda **value:print('stream_prepare '+json.dumps(value),flush=True))
    if old is None:
        model=ChartStreamTransformer(config)
    model.to(device); model.train()
    fused=args.optimizer_implementation=='fused' or (args.optimizer_implementation=='auto' and device.type=='cuda')
    if fused and device.type!='cuda':raise ValueError('融合优化器需要CUDA；CPU请选择auto或standard')
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay,fused=fused)
    scaler=torch.amp.GradScaler('cuda',enabled=precision=='fp16')
    if args.resume and old.get('optimizer_state_dict'): optimizer.load_state_dict(old['optimizer_state_dict'])
    for group in optimizer.param_groups:group['fused']=fused;group['foreach']=False if fused else None
    if args.resume and old.get('scaler_state_dict') and scaler.is_enabled():
        scaler.load_state_dict(old['scaler_state_dict'])
    # Release CPU optimizer/model copies before training. Adam's live state is
    # owned by optimizer now; the initial finetune payload is no longer needed.
    if old is not None:
        old.pop('optimizer_state_dict',None); old.pop('model_state_dict',None)
    execution=install_training_graphs(model,mode=args.execution_mode,batch_size=args.batch_size,
        max_tokens=config.max_tokens,max_audio_tokens=config.max_audio_tokens,
        budget_gb=args.cuda_graph_budget_gb)
    execution.prepare(model,precision)
    loader=DataLoader(dataset,batch_size=args.batch_size,shuffle=True,collate_fn=collate_stream,
                      num_workers=0,pin_memory=device.type=='cuda' and not args.prefetch_batches,
                      generator=torch.Generator().manual_seed(args.seed))
    out=Path(args.checkpoint_dir); out.mkdir(parents=True,exist_ok=True)
    atom_catalog=[]
    for record in dataset.base.songs:
        for point in record[1]:
            # Keep the same atomic split used by the legacy event vocabulary.
            atom_catalog.extend(_NoteGrammar._split_slashes(str(point.note_text)))
    # Mirrored cells are training targets too; generation must be able to emit them.
    atom_catalog.extend(dataset.augmented_atoms())
    global_step=int(old.get('optimizer_steps', 0)) if args.resume and old else 0
    start_epoch = _resume_start_epoch(old) if args.resume and old else 0
    if args.resume and old and start_epoch < int(old.get('epoch',0)):
        print('stream_resume '+json.dumps({'policy':'restart_incomplete_epoch',
            'epoch':start_epoch+1,'saved_position':old.get('training_position'),
            'optimizer_state_retained':True}),flush=True)
    print('stream_runtime '+json.dumps({'training_semantics':TRAINING_SEMANTICS,'device':str(device),
        'precision':precision,'lr':args.lr,'weight_decay':args.weight_decay,'batch_size':args.batch_size,
        'grad_accumulation':args.grad_accumulation,'event_decision_loss_weight':args.event_decision_loss_weight,
        'plan_loss_weight':args.plan_loss_weight,'parameters':sum(p.numel() for p in model.parameters()),
        'audio_cache_budget_gb':args.audio_cache_gb,'quality_enabled':args.quality_enabled,'start_epoch':start_epoch+1,
        'grounding_profile':args.grounding_profile,'detail_audio_nodes':args.detail_audio_nodes,
        'curve_slope_loss_weight':args.curve_slope_loss_weight,
        'alignment_loss_weight':args.alignment_loss_weight,'event_balance_loss_weight':args.event_balance_loss_weight,
        'history_note_dropout':args.history_note_dropout,'prefix_note_dropout':args.prefix_note_dropout,
        'continuation_profile':config.continuation_profile,'context_replay_tokens_effective':dataset.replay_tokens,
        'prefix_mask_span_events':args.prefix_mask_span_events,'cycle_exit_loss_weight':args.cycle_exit_loss_weight,
        'symmetry_augmentation':args.symmetry_augmentation,
        'semantic_migration':semantic_migration,
        'prefetch_batches':args.prefetch_batches,'cache_query_fraction':.75 if dataset.grounded else .25,
        'data_workers':args.data_workers if args.prefetch_batches else 1,
        'input_cache_mode':args.input_cache_mode,'structure_cache_mb':args.structure_cache_mb,
        'optimizer_implementation':'fused' if fused else 'standard',
        'input_transfer':'pinned_dtype_slabs' if device.type=='cuda' and args.prefetch_batches else 'per_field',
        'attention_policy':getattr(args,'attention_policy',{'policy':'pytorch_default'}),
        'execution_mode':args.execution_mode,'cuda_graph_budget_gb':args.cuda_graph_budget_gb,
        'input_residency':residency,
        'cpu_budget':args.cpu_budget,
        'end_epoch':start_epoch+args.epochs}),flush=True)
    for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
        dataset.set_epoch(epoch)
        totals=MetricAccumulator(); optimizer.zero_grad(set_to_none=not bool(execution.slots)); pending=0
        started=time.monotonic()
        waited=transfer=compute=saving=0.;previous_end=started
        last_log_time=started;last_log_batch=0
        from .prefetch import StreamBatchPrefetch
        from .training_batch import transfer_batch,pack_batch
        loader.generator.manual_seed(args.seed+epoch)
        batch_source=(StreamBatchPrefetch(loader,depth=args.prefetch_batches,
            workers=args.data_workers,prepare_batch=pack_batch if device.type=='cuda' else None)
            if args.prefetch_batches else nullcontext(loader))
        with batch_source as batches:
            for index,batch in enumerate(batches,1):
                data_ready=time.monotonic();waited+=data_ready-previous_end
                sample_timer=(CudaBatchTimer() if device.type=='cuda' and
                    (index%args.log_interval==0 or index==len(loader)) else None)
                tensors=transfer_batch(batch,device)
                if sample_timer:sample_timer.transferred()
                transferred=time.monotonic();transfer+=transferred-data_ready
                with torch.autocast('cuda',dtype=torch.bfloat16 if precision=='bf16' else torch.float16,enabled=precision in {'bf16','fp16'}):
                    output=model(tensors); loss,metrics=_loss(output,tensors,args.plan_loss_weight,args.event_decision_loss_weight,
                        args.alignment_loss_weight,args.event_balance_loss_weight,args.cycle_exit_loss_weight,
                        args.curve_slope_loss_weight)
                scaler.scale(loss/args.grad_accumulation).backward(); pending+=1
                if sample_timer:sample_timer.computed()
                totals.update(metrics)
                if pending>=args.grad_accumulation or index==len(loader):
                    stepped = _finish_accumulation(model,optimizer,scaler,pending,args.grad_accumulation)
                    pending=0
                    if stepped: global_step+=1
                    compute+=time.monotonic()-transferred
                    if stepped and args.checkpoint_interval_steps and global_step % args.checkpoint_interval_steps == 0:
                        save_started=time.monotonic()
                        state = {'epoch':epoch, 'chunk':index, 'total_chunks':len(loader)}
                        save(out/'stream_last.pt', model, optimizer, epoch=epoch, args=vars(args),
                             manifest=args.manifest, metrics=_mean_metrics(totals,index), atom_catalog=atom_catalog,
                             optimizer_steps=global_step, training_position=state, scaler=scaler,
                             dataset_fingerprint=dataset.fingerprint,aliases=(out/f'stream_step_{global_step:08d}.pt',))
                        print('stream_checkpoint '+json.dumps({'epoch':epoch,'optimizer_steps':global_step,
                            'path':str(out/f'stream_step_{global_step:08d}.pt')}),flush=True)
                        saving+=time.monotonic()-save_started
                else:compute+=time.monotonic()-transferred
                if index%args.log_interval==0 or index==len(loader):
                    average=_mean_metrics(totals,index)
                    now=time.monotonic();elapsed=now-started
                    recent=(index-last_log_batch)/max(now-last_log_time,1e-6)
                    last_log_time,last_log_batch=now,index
                    print('stream_progress '+json.dumps({'family':'chart_stream_v1','epoch':epoch,'chunk':index,
                        'total':len(loader),'optimizer_steps':global_step,'elapsed_sec':elapsed,
                        'batches_per_sec':index/max(elapsed,1e-6),'recent_batches_per_sec':recent,
                        'data_wait_sec':waited,'transfer_host_sec':transfer,'compute_host_sec':compute,'save_sec':saving,
                        'timing_scope':'host_wall_not_cuda_kernel',
                        'execution':execution.stats(),
                        'cuda_memory':({'allocated_gib':torch.cuda.memory_allocated(device)/1024**3,
                            'reserved_gib':torch.cuda.memory_reserved(device)/1024**3,
                            'peak_allocated_gib':torch.cuda.max_memory_allocated(device)/1024**3}
                            if device.type=='cuda' else None),
                        **(sample_timer.metrics() if sample_timer else {}),
                        'prefetch':batches.stats() if args.prefetch_batches else {'mode':'synchronous'},
                        'cache':dataset.cache_stats(),**average}),flush=True)
                previous_end=time.monotonic()
        save(out/'stream_last.pt',model,optimizer,epoch=epoch,args=vars(args),manifest=args.manifest,
             metrics=_mean_metrics(totals,len(loader)),atom_catalog=atom_catalog,optimizer_steps=global_step,
             training_position={'epoch':epoch, 'chunk':len(loader), 'total_chunks':len(loader), 'complete':True},
             scaler=scaler,dataset_fingerprint=dataset.fingerprint)
        print('stream_epoch '+json.dumps({'epoch':epoch,'optimizer_steps':global_step,
            'elapsed_sec':time.monotonic()-started,**_mean_metrics(totals,len(loader))}),flush=True)
    return 0


if __name__=='__main__': raise SystemExit(main())
