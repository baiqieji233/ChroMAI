"""Teacher-forced Composer v2 training with exact update-boundary resume."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import uuid
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from .checkpoint import FAMILY,SCHEMA,read,save,save_epoch_snapshot,inspect_payload
from .codec import ComponentCodec
from .data import (AUDIO_LOCAL_BATCH_ORDER, GLOBAL_BATCH_ORDER, ComposerV2Dataset,
                   collate_composer_v2, LengthBucketBatchSampler)
from .conditions import LEVEL_ONLY,RADAR_STYLE,POLICIES,condition_contract,policy_from_config
from . import chart_dialect, timing_recovery
from .defaults import AUDIO_WRITING_POLICIES
from .defaults import (
    FINETUNE_SCOPE_CHOICES, FUTURE_CONFIGURATION_CHOICES, FUTURE_CONFIGURATION_POLICY, configuration_view_fraction,
    FUTURE_CONFIGURATION_WEIGHT,
    MUSICAL_CONTEXT_CHOICES, MUSICAL_CONTEXT_PROBABILITY, MUSICAL_RELATION_CHOICES,
    MUSICAL_RELATION_FRACTION, resolve_auxiliary_policy, resolve_finetune_scope, resolve_musical_context,
)


def emit(kind,**values):
    print("composer_v2_"+kind+" "+json.dumps(values,ensure_ascii=False,allow_nan=False),flush=True)

def parser():
    from .defaults import TRAIN_DEFAULTS
    result=argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest",required=True)
    result.add_argument("--validation-manifest")
    result.add_argument('--quality-enabled', action=argparse.BooleanOptionalAction, default=True,
                        help='Read saved chart/region weights; required for reviewed-only sources')
    result.add_argument("--checkpoint-dir",required=True)
    result.add_argument("--cache-dir",default="cache/chart_composer_v2")
    parents=result.add_mutually_exclusive_group()
    parents.add_argument("--resume");parents.add_argument("--init-checkpoint")
    for key,value in TRAIN_DEFAULTS.items():
        if key in ('audio_layers','song_layers','music_context'):continue
        result.add_argument("--"+key.replace("_","-"),default=value,type=type(value))
    result.add_argument("--mert-model")
    from .difficulty import SOURCES
    result.add_argument('--difficulty-source', choices=SOURCES, default='auto',
                        help='Use the same declared or fitted level for filtering and model conditioning')
    result.add_argument('--min-level', type=float)
    result.add_argument('--max-level', type=float)
    result.add_argument('--min-declared-level', type=float, help='Minimum declared level for fitted scoring; 0 disables this check.')
    result.add_argument('--difficulty-slots', type=int, nargs='+', help='Difficulty slots to include (4, 5, 6).')
    result.add_argument('--open-mdp-dir')
    result.add_argument('--chartgate-dir')
    from .batching import CHOICES as BATCH_LAYOUTS
    result.add_argument('--batch-layout',choices=BATCH_LAYOUTS,default='auto')
    result.add_argument('--audio-layers',type=int,default=None,help='Fresh: 4; parent: inherit. Init may append identity blocks.')
    result.add_argument('--song-layers',type=int,default=None,help='Fresh: 2; parent: inherit.')
    from .music_policy import CHOICES as MUSIC_CHOICES
    result.add_argument('--music-context',choices=MUSIC_CHOICES,default='auto')
    result.add_argument('--musical-context', choices=MUSICAL_CONTEXT_CHOICES, default='auto',
                        help='Train with music-conditioned history views; auto enables fresh/finetune and inherits resume')
    result.add_argument('--musical-context-probability', type=float,
                        help=f'Proportion relying on music with reduced older chart context, 0 to 1 (default {MUSICAL_CONTEXT_PROBABILITY}; resume inherits)')
    result.add_argument('--musical-relation-policy', choices=MUSICAL_RELATION_CHOICES, default='auto',
                        help='Supervise chart events against musical relationships; resume inherits the saved policy')
    result.add_argument('--musical-relation-fraction', type=float,
                        help=f'Maximum relative training supervision on musical relationship views (default {MUSICAL_RELATION_FRACTION})')
    result.add_argument('--future-configuration-policy', choices=FUTURE_CONFIGURATION_CHOICES, default='auto',
                        help=f'Auto uses {FUTURE_CONFIGURATION_POLICY} for fresh training and continuous-event fine-tuning; exact resume inherits the checkpoint')
    result.add_argument('--future-configuration-weight', type=float,
                        help=f'Weight of the ordered future chart objective (default {FUTURE_CONFIGURATION_WEIGHT})')
    result.add_argument('--finetune-scope', choices=FINETUNE_SCOPE_CHOICES, default='auto',
                        help='Train all parameters or only music-reading parameters of an initialized checkpoint')
    result.add_argument('--sequence-policy',choices=('window_v1','continuous_events_v1'),default=None)
    result.add_argument('--refresh-events',type=int,default=16)
    result.add_argument('--training-window-events', type=int,
                        help='Maximum authored target events per training window; defaults to refresh-events. '
                             'Generation, validation and timing recovery keep their original window limits.')
    result.add_argument('--local-audio-block-beats',type=int,choices=(1,2),default=None,
                        help='Local audio encoding block width in beats; omitted inherits the checkpoint or uses 1.')
    result.add_argument('--timing-recovery',choices=timing_recovery.CHOICES,default='auto')
    from .onset_policy import CHOICES as ONSET_CHOICES
    result.add_argument('--onset-readout',choices=ONSET_CHOICES,default='auto',
                        help='Phase-only acoustic evidence; auto enables fresh/continuous finetune and inherits exact resume')
    from .rhythm_structure import CHOICES as STRUCTURE_CHOICES
    result.add_argument('--rhythm-structure',choices=STRUCTURE_CHOICES,default='auto',
                        help='Sub-beat phase geometry and aligned trusted references; auto enables fresh/continuous finetune and inherits exact resume')
    from .intensity import CHOICES as CURVE_CHOICES
    result.add_argument('--intensity-curve',choices=CURVE_CHOICES,default='auto',
                        help='Train low-frequency event/object rate curves; resume inherits its parent policy')
    result.add_argument('--reviewed-reference-policy',choices=('auto','approved_only','off'),default='auto',
                        help='Allow [REF] on unmarked_weight=0 charts only when every referenced event is approved; resume inherits parent contract')
    result.add_argument('--auto-quality-policy',choices=('auto','on','off'),default='auto',
                        help='Merge auto_quality_annotations.json for continuous events; manual labels win and resume inherits the parent contract')
    result.add_argument("--condition-policy", choices=POLICIES, default=None,
        help="Fresh/init default radar_style_v1; resume inherits the checkpoint contract")
    return result

def _hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()


def resolve_training_policy(requested, parent_config=None, *, resume=False):
    """Keep resume semantics exact; allow only explicit init policy migration."""
    parent_policy = policy_from_config(parent_config) if parent_config is not None else None
    policy = requested or (parent_policy if resume else RADAR_STYLE)
    condition_contract(policy)
    migration = parent_policy is not None and policy != parent_policy
    if migration and (resume or (parent_policy, policy) != (LEVEL_ONLY, RADAR_STYLE)):
        raise ValueError('Condition policy migration requires init-checkpoint from level_only_v1 to radar_style_v1')
    return policy, migration


def configure_trainable_parameters(model, scope):
    trainable_names = []
    trainable_parameters = []
    for name, parameter in model.named_parameters():
        path = name.split('.')
        decoder_reading = (len(path) >= 4 and path[0] == 'decoder'
                           and (path[2] in ('local_attention', 'song_attention') or
                                path[2] == 'norms' and path[3] in ('1', '2')))
        music_projection = path[0] in ('paired_music', 'entry_preview', 'future_configuration_head', 'local_writing')
        trainable = scope == 'all' or decoder_reading or music_projection
        parameter.requires_grad_(trainable)
        if trainable:
            trainable_names.append(name)
            trainable_parameters.append(parameter)
    return trainable_names, trainable_parameters


def rng_state():
    state=np.random.get_state()
    result=dict(python=random.getstate(),numpy=dict(kind=state[0],keys=state[1].tolist(),position=state[2],has_gauss=state[3],cached_gaussian=state[4]),
        torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    if hasattr(torch,'xpu') and torch.xpu.is_initialized():result['xpu']=torch.xpu.get_rng_state_all()
    return result


def resolve_zero_unmarked_policy(parent_contract=None, *, resume=False):
    if not resume:
        return 'explicit_good_v1'
    saved = parent_contract or {}
    return saved.get('zero_unmarked_policy', 'explicit_good_v1'
        if saved.get('reviewed_reference_policy', 'off') != 'off' else 'legacy_weights_v1')


def resolve_auto_quality_policy(requested, manifest_path, parent_contract=None, *, resume=False,
                                quality_enabled=True, sequence_policy='continuous_events_v1'):
    if requested not in ('auto', 'on', 'off'):
        raise ValueError('Unknown auto_quality_policy')
    saved = (parent_contract or {}).get('auto_quality_policy', 'off')
    if resume:
        if requested != 'auto' and ('auto_v1' if requested == 'on' else 'off') != saved:
            raise ValueError('Changing auto_quality_policy requires --init-checkpoint finetune, not --resume')
        if saved == 'off':
            return 'off'
        if saved != 'auto_v1':
            raise ValueError(f'Unknown saved auto_quality_policy: {saved}')
    elif requested == 'off':
        return 'off'
    if not quality_enabled or sequence_policy != 'continuous_events_v1':
        if resume or requested == 'on':
            raise ValueError('自动质检标注需要启用质量加权的 Composer v2 连续事件训练')
        return 'off'
    from maichart.training.quality_annotations import resolve_auto_annotations
    path, _ = resolve_auto_annotations(manifest_path)
    if path is not None:
        return 'auto_v1'
    if resume:
        raise ValueError('续训缺少原自动质检标注，请恢复原文件或使用 --init-checkpoint')
    if requested == 'on':
        raise ValueError('未找到与训练清单或原始划分来源匹配的 auto_quality_annotations.json')
    return 'off'

def restore_rng(value):
    random.setstate(value["python"])
    state=value["numpy"]
    np.random.set_state((state["kind"],np.asarray(state["keys"],dtype=np.uint32),state["position"],state["has_gauss"],state["cached_gaussian"]))
    torch.set_rng_state(value["torch"])
    if value.get("cuda"):
        if not torch.cuda.is_available() or len(value["cuda"])!=torch.cuda.device_count():
            raise ValueError("Exact resume requires the saved CUDA RNG device topology")
        torch.cuda.set_rng_state_all(value["cuda"])
    if value.get('xpu'):
        if not hasattr(torch,'xpu') or not torch.xpu.is_available() or len(value['xpu'])!=torch.xpu.device_count():
            raise ValueError('Exact resume requires the saved Intel XPU RNG device topology')
        torch.xpu.set_rng_state_all(value['xpu'])

def _identity(dataset):
    value=dict(manifest_sha256=dataset.manifest_sha256,charts=dataset.audit["charts"],
        audio=[dict(song_id=s.song_id,difficulty=s.difficulty,identity=s.audio.identity) for s in dataset.songs],
        codec_sha256=_hash(dataset.codec.to_dict()),windows=len(dataset))
    excluded = dataset.audit.get('vocabulary_exclusions', ())
    if excluded:
        value['vocabulary_excluded_chart_indices'] = [row['chart_index'] for row in excluded]
    if dataset.condition_policy != LEVEL_ONLY:
        value['condition_contract'] = dataset.condition_contract
    if getattr(dataset, 'auto_quality_policy', 'off') == 'auto_v1':
        value['auto_annotations_sha256'] = dataset.auto_annotations_sha256
    return dict(sha256=_hash(value),**value)


def _same_data_identity(saved, current):
    """Compare legacy identities after resolving only their source paths."""
    from maichart.portable_paths import material_path

    if saved is None or current is None:
        return saved is current
    if set(saved) != set(current):
        return False
    for split in saved:
        old, new = saved[split], current[split]
        if old is None or new is None:
            if old is not new:
                return False
            continue
        old_content = {key: value for key, value in old.items() if key != 'sha256'}
        new_content = {key: value for key, value in new.items() if key != 'sha256'}
        if old.get('sha256') != _hash(old_content) or new.get('sha256') != _hash(new_content):
            return False
        old_content = dict(old_content)
        old_content['charts'] = [dict(chart, source=str(material_path(chart['source']).resolve()))
                                 for chart in old_content['charts']]
        if old_content != new_content:
            return False
    return True

def _batches(loader,skip=0,*,pending=None):
    # Dataset fetching is deterministic; one outstanding CPU batch bounds RAM.
    sampler = getattr(loader, 'batch_sampler', None)
    from .batching import FullInputBatchSampler
    can_seek = isinstance(sampler, (LengthBucketBatchSampler, FullInputBatchSampler))
    start = max(0, skip) if can_seek else 0
    if can_seek:
        previous_start = sampler.start_batch
        sampler.start_batch = start
    try:
        iterator = iter(loader)
        sentinel = object()
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="composer-v2-data") as pool:
            future = pool.submit(next, iterator, sentinel)
            index = start
            while True:
                batch = future.result()
                if batch is sentinel:
                    return
                future = pool.submit(next, iterator, sentinel)
                if pending is not None:pending[:]=[future]
                if index >= skip:
                    yield index, batch
                index += 1
    finally:
        if pending is not None:pending.clear()
        if can_seek:
            sampler.start_batch = previous_start

def _device_batch(batch,device):
    if device.type=='xpu' and 'bank_semantic_right' in batch:
        # Preserve the CPU double -> float contract without depending on
        # the client's Intel FP64 kernel support or scheduling extra GPU work.
        batch=dict(batch)
        fraction=batch.pop('bank_semantic_weight').double()[:,None]
        right=batch.pop('bank_semantic_right').float().double()
        batch['bank_semantic']=(batch['bank_semantic'].float().double()*(1.-fraction)+right*fraction).float()
    return {key:value.to(device,non_blocking=True) if isinstance(value,torch.Tensor) else value for key,value in batch.items()}

def _loader(dataset,batch_size,seed,shuffle,device,*,order=GLOBAL_BATCH_ORDER,
            batch_layout='target_only_v1',accumulation=1,source_batch_size=None):
    from .batching import FULL_INPUT, FullInputBatchSampler
    if shuffle and batch_layout == FULL_INPUT:
        sampler=FullInputBatchSampler(dataset,batch_size,accumulation,
            source_batch_size=source_batch_size or batch_size,seed=seed,order=order)
        loader_dataset=sampler.prepared
    else:
        sampler=LengthBucketBatchSampler(dataset,batch_size,seed=seed,shuffle=shuffle,order=order)
        loader_dataset=dataset
    generator=torch.Generator().manual_seed(seed)
    from .stream_data import StreamDataset,collate_stream
    return DataLoader(loader_dataset,batch_sampler=sampler,collate_fn=collate_stream if isinstance(dataset,StreamDataset) else collate_composer_v2,
        num_workers=0,pin_memory=device.type in ('cuda','xpu'),generator=generator),sampler


def _validation_loader(dataset,batch_size,seed,device,maximum,policy):
    from .validation_sampling import POLICY,LEGACY,balanced_window_indices,coverage_report,ValidationBatchSampler
    if policy not in (POLICY,LEGACY):raise ValueError('Unknown validation sampling contract')
    loader,sampler=_loader(dataset,batch_size,seed,False,device)
    if policy==POLICY and maximum and maximum*batch_size<len(dataset):
        indices=balanced_window_indices(dataset,maximum*batch_size)
        from .stream_data import StreamDataset,collate_stream
        loader=DataLoader(dataset,batch_sampler=ValidationBatchSampler(dataset,indices,batch_size),
            collate_fn=collate_stream if isinstance(dataset,StreamDataset) else collate_composer_v2,
            num_workers=0,pin_memory=device.type=='cuda',generator=torch.Generator().manual_seed(seed))
    else:
        batches=list(sampler)
        indices=[index for batch in (batches[:maximum] if maximum else batches) for index in batch]
    loader.validation_coverage=coverage_report(dataset,indices,policy,maximum_batches=maximum)
    return loader

def _sync(device):
    if device.type=="cuda":torch.cuda.synchronize(device)
    elif device.type=='xpu':torch.xpu.synchronize(device)

def _autocast(device,precision):
    if precision=="fp32":return nullcontext()
    return torch.autocast(device_type=device.type,dtype=torch.bfloat16 if precision=="bf16" else torch.float16)

def _cpu_budget(batch,time_tokens=None):
    active=batch["labels"].ne(-100)
    weights=batch.get("token_weights",active.float()).float()*active
    weights=weights*batch.get("sample_weight",weights.new_ones(weights.size(0)))[:,None]
    mass=float(weights.sum())
    timing=time_tokens[batch["labels"].clamp_min(0)]&active if time_tokens is not None else active
    relation_fractions=batch.get('musical_relation_extra_fraction')
    relation_mass=float((weights.sum(1)*relation_fractions).sum()) if relation_fractions is not None else 0.
    return int(active.sum()),mass,float((weights*timing).sum()),float((weights*(active&~timing)).sum()),relation_mass

def _phase_budget(batch,phase_tokens):
    active=batch['labels'].ne(-100)
    weights=batch.get('token_weights',active.float()).float()*active
    weights=weights*batch.get('sample_weight',weights.new_ones(weights.size(0)))[:,None]
    return float((weights*phase_tokens[batch['labels'].clamp_min(0)]).sum())


def _validate(model,loader,device,precision,maximum):
    model.validation_phase_loss=None
    model.validation_time_loss=None
    model.validation_nonphase_time_loss=None
    model.validation_curve_loss=None
    model.validation_curve_mae=None
    model.validation_future_configuration_loss=None
    model.validation_coverage=getattr(loader,'validation_coverage',None)
    if loader is None:return None
    phase_numerator=torch.zeros((),device=device);phase_mass_sum=0.
    phase_tokens=model.phase_targets.detach().cpu()
    time_tokens=model.time_tokens.detach().cpu()
    time_numerator=torch.zeros((),device=device);time_mass_sum=0.
    model.eval();numerator=torch.zeros((),device=device);mass_sum=0.
    finite=torch.ones((),device=device,dtype=torch.bool)
    curve_numerator=torch.zeros((),device=device)
    future_configuration_numerator=torch.zeros((),device=device)
    curve_error=torch.zeros(2,device=device);curve_exposure=torch.zeros(2,device=device)
    with torch.inference_mode():
        for index,batch in _batches(loader):
            if maximum and index>=maximum:break
            if model.config.condition_policy==RADAR_STYLE:
                # Select checkpoints for the default music-led generation path,
                # not a validation path supplied with the answer chart's statistics.
                known=torch.zeros_like(batch['condition'],dtype=torch.bool)
                known[:,0]=True;batch={**batch,'condition_known':known}
            _,mass,time_mass,_,_=_cpu_budget(batch,time_tokens)
            phase_mass=_phase_budget(batch,phase_tokens)
            if not math.isfinite(mass) or mass<0:raise ValueError("Invalid validation token weight budget")
            batch=_device_batch(batch,device)
            with _autocast(device,precision):
                output=model(batch);loss,metrics=model.loss(output,batch)
            finite=finite&torch.isfinite(loss)
            numerator+=metrics['token_loss']*max(1.,mass);mass_sum+=mass
            curve_numerator+=metrics['curve_loss']*max(1.,mass)
            if model.config.future_configuration_policy != 'off':
                future_configuration_numerator+=metrics['future_configuration_loss']*max(1.,mass)
            if hasattr(model,'intensity'):
                widths=batch['curve_widths'].float()
                # Rate MAE is independent of token CE and measured in real units.
                quality=batch.get('curve_quality_weights',torch.ones_like(batch['curve_counts'])).float()
                sample_quality=batch.get('sample_weight',torch.ones(widths.size(0),device=device)).float()
                exposure=widths[...,None]*quality*sample_quality[:,None,None]
                curve_error+=((output['curve_rates']-batch['curve_counts']/widths[...,None].clamp_min(1e-6)).abs()*exposure).sum((0,1))
                curve_exposure+=exposure.sum((0,1))
            phase_numerator+=metrics["phase_loss"]*max(1.,phase_mass);phase_mass_sum+=phase_mass
            time_numerator+=metrics['time_loss']*max(1.,time_mass);time_mass_sum+=time_mass
    if not bool(finite):raise RuntimeError("Nonfinite validation loss")
    if mass_sum<=0:raise ValueError("Validation has no positive quality-weighted target tokens")
    result=float(numerator)/max(1.,mass_sum)
    model.validation_phase_loss=float(phase_numerator)/phase_mass_sum if phase_mass_sum else None
    model.validation_time_loss=float(time_numerator)/time_mass_sum if time_mass_sum else None
    nonphase_mass=time_mass_sum-phase_mass_sum
    model.validation_nonphase_time_loss=(float(time_numerator-phase_numerator)/nonphase_mass
                                        if nonphase_mass>0 else None)
    if model.config.future_configuration_policy != 'off' and model.config.future_configuration_weight > 0:
        model.validation_future_configuration_loss=float(future_configuration_numerator)/max(1.,mass_sum)
    if hasattr(model,'intensity'):
        model.validation_curve_loss=float(curve_numerator)/max(1.,mass_sum)
        values=(curve_error/curve_exposure.clamp_min(1e-6)).cpu().tolist()
        model.validation_curve_mae=[value if exposure>0 else None for value,exposure in zip(values,curve_exposure.cpu().tolist())]
    model.train()
    return result


def _validate_checkpoint(model,loader,device,precision,maximum,*,pending=None):
    """Inspect fixed holdout windows without changing the next training RNG draw."""
    # Train/dev share one bounded audio bank cache. Finish the outstanding
    # training fetch before a second loader can read or evict its mappings.
    if pending:pending[0].result()
    saved_rng=rng_state();was_training=model.training
    try:
        return _validate(model,loader,device,precision,maximum)
    finally:
        restore_rng(saved_rng)
        model.train(was_training)

def _apply_cuda_budget(budget_gib, uses_cuda):
    if not uses_cuda or budget_gib == 0:
        return 0.0
    if not torch.cuda.is_available():
        raise ValueError('CUDA 显存预算需要可用的 CUDA 设备')
    index = torch.cuda.current_device()
    total_gib = torch.cuda.get_device_properties(index).total_memory / 2**30
    if budget_gib > total_gib:
        raise ValueError(f'CUDA 显存预算 {budget_gib:g} GiB 超出设备容量 {total_gib:.2f} GiB')
    torch.cuda.set_per_process_memory_fraction(budget_gib / total_gib, index)
    return budget_gib


def _extend_vocabulary_state(model, state, parent_size):
    if model.codec.vocab_size == parent_size:
        return state
    expanded = dict(state)
    initialized = model.state_dict()
    for name in ('embedding.weight', 'output_bias'):
        original = state[name]
        target = initialized[name]
        if original.shape != (parent_size, *target.shape[1:]):
            raise ValueError(f'Invalid checkpoint vocabulary tensor shape for {name}: {original.shape}')
        expanded[name] = torch.cat((original, target[parent_size:].to(original)))
    return expanded


def train(args):
    from .model import ComposerV2Config,ComposerV2
    from .audio import AudioFeatureStore

    args.cuda_memory_budget_gb = getattr(args, 'cuda_memory_budget_gb', 0.0)
    args.audio_cache_gb = getattr(args, 'audio_cache_gb', 0.25)
    args.save_every = getattr(args, 'save_every', 5)
    args.early_stopping_patience=getattr(args,'early_stopping_patience',5)
    for key in ('cuda_memory_budget_gb', 'audio_cache_gb'):
        value = getattr(args, key)
        if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1024:
            raise ValueError(f'{key} 必须在 0 至 1024 GiB 之间')
    if args.cpu_threads<1:raise ValueError("cpu_threads must be positive")
    from .training_memory import POLICIES as MEMORY_POLICIES, PROJECTION_PACKING_CHOICES
    if getattr(args,'activation_checkpointing','auto') not in MEMORY_POLICIES:
        raise ValueError('Unknown activation checkpointing policy')
    packing = getattr(args, 'attention_projection_packing', 'auto')
    if packing not in ('auto', *PROJECTION_PACKING_CHOICES):
        raise ValueError('Unknown attention projection packing policy')
    flash = getattr(args, 'decoder_flash_attention', 'off')
    if flash not in ('off', 'on'):
        raise ValueError('Unknown decoder FlashAttention policy')
    torch.set_num_threads(args.cpu_threads)
    if min(args.epochs,args.batch_size,args.grad_accumulation,args.log_interval,args.checkpoint_interval_steps)<=0:raise ValueError("epochs/batch/accumulation/log/save intervals must be positive")
    if args.lr<=0 or not 0<=args.min_lr_ratio<=1 or not 0<=args.dropout<1:raise ValueError("Invalid optimizer/dropout settings")
    if min(args.warmup_steps,args.validation_max_batches,args.seed,args.max_steps,args.limit_charts,args.save_every,args.early_stopping_patience)<0:raise ValueError("Training limits, seed and save_every must be nonnegative")
    from .acceleration import AUDIO_DEVICES
    if args.device not in {"auto","cpu","cuda","xpu"} or args.prepare_device not in AUDIO_DEVICES:
        raise ValueError('Invalid training/preparation device')
    if args.lr_schedule not in {"cosine","constant"}:raise ValueError("Unknown lr_schedule")
    if args.audio_backend not in {"mert","native"}:raise ValueError("audio_backend must be mert or explicit native")
    directory=Path(args.checkpoint_dir).resolve()
    parent_path=Path(args.resume or args.init_checkpoint).resolve() if args.resume or args.init_checkpoint else None
    source_directory=parent_path.parent if parent_path else None
    artifacts=[directory/name for name in ("composer_v2_last.pt","composer_v2_best.pt","composer_v2_inference.pt")]
    requested_training_window = getattr(args, 'training_window_events', None)
    if (args.init_checkpoint and requested_training_window not in (None, args.refresh_events)
            and (directory == source_directory or any(path.exists() for path in artifacts))):
        raise ValueError('Changing training-window-events requires a new checkpoint directory; retain the parent')
    if parent_path in artifacts:
        archive=directory/"resume_sources"/(time.strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:10]+"_"+parent_path.name)
        archive.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(parent_path,archive)
        parent_path=archive
        emit("checkpoint",stage="parent_preserved",path=str(archive))
    if not parent_path and any(path.exists() for path in artifacts):raise ValueError("Fresh output directory already contains Composer v2 weights; choose a new directory")
    parent=read(parent_path) if parent_path else None
    if packing == 'auto':
        packing = ((parent or {}).get('args') or {}).get('attention_projection_packing', 'off')
    args.attention_projection_packing = 'off' if packing == 'auto' else packing
    if parent:
        checked=inspect_payload(parent)
        if not checked["valid"] or (args.resume and not checked["can_resume"]):raise ValueError("Parent checkpoint cannot initialize this training mode: "+str(checked.get("error")))
    parent_policy = policy_from_config(parent.get('model_config') or parent.get('config')) if parent else None
    parent_sequence=(parent.get('model_config') or parent.get('config')).get('sequence_policy','window_v1') if parent else None
    args.sequence_policy=getattr(args,'sequence_policy',None) or (parent_sequence if parent else 'continuous_events_v1')
    parent_contract = (parent.get('training_contract') or {}) if parent else {}
    args.musical_context_policy, args.musical_context_probability = resolve_musical_context(
        getattr(args, 'musical_context', 'auto'), getattr(args, 'musical_context_probability', None),
        sequence_policy=args.sequence_policy, parent_contract=parent_contract, resume=bool(args.resume))
    parent_model = (parent.get('model_config') or parent.get('config')) if parent else {}
    args.musical_relation_policy, args.musical_relation_fraction = resolve_auxiliary_policy(
        getattr(args, 'musical_relation_policy', 'auto'), getattr(args, 'musical_relation_fraction', None),
        feature='musical_relation', choices=MUSICAL_RELATION_CHOICES, default=MUSICAL_RELATION_FRACTION,
        sequence_policy=args.sequence_policy, saved_config=parent_contract, resume=bool(args.resume))
    args.future_configuration_policy, args.future_configuration_weight = resolve_auxiliary_policy(
        getattr(args, 'future_configuration_policy', 'auto'), getattr(args, 'future_configuration_weight', None),
        feature='future_configuration', choices=FUTURE_CONFIGURATION_CHOICES, default=FUTURE_CONFIGURATION_WEIGHT,
        sequence_policy=args.sequence_policy, saved_config=parent_model, resume=bool(args.resume))
    args.finetune_scope = resolve_finetune_scope(getattr(args, 'finetune_scope', 'auto'), parent_contract,
                                                resume=bool(args.resume), initialized=bool(args.init_checkpoint))
    if args.finetune_scope == 'music_reading' and args.sequence_policy != 'continuous_events_v1':
        raise ValueError('Music-reading fine-tuning requires continuous event training')
    future_migration = bool(parent and args.future_configuration_policy !=
                            parent_model.get('future_configuration_policy', 'off'))
    learning_contract_changed = bool(args.init_checkpoint and (
        args.musical_relation_policy != parent_contract.get('musical_relation_policy', 'off') or
        args.musical_relation_fraction != parent_contract.get('musical_relation_fraction', MUSICAL_RELATION_FRACTION) or
        args.future_configuration_weight != parent_model.get('future_configuration_weight', FUTURE_CONFIGURATION_WEIGHT) or
        future_migration or args.finetune_scope != 'all' or
        args.finetune_scope != parent_contract.get('finetune_scope', 'all')))
    if learning_contract_changed and (directory == source_directory or any(path.exists() for path in artifacts)):
        raise ValueError('Changing musical relation, ordered future, or fine-tuning scope requires a new checkpoint directory')
    recovery_policy=timing_recovery.resolve(getattr(args,'timing_recovery','auto'),sequence_policy=args.sequence_policy,
        saved=parent_contract.get('timing_recovery','off'),resume=bool(args.resume))
    if (args.init_checkpoint and recovery_policy!=parent_contract.get('timing_recovery','off')
            and (directory==source_directory or any(path.exists() for path in artifacts))):
        raise ValueError('更换前文纠错策略请使用新的微调输出目录')
    args.timing_recovery=recovery_policy
    reviewed_ref=getattr(args,'reviewed_reference_policy','auto')
    saved_reviewed_ref=parent['training_contract'].get('reviewed_reference_policy','off') if parent and parent.get('training_contract') else 'off'
    if reviewed_ref=='auto':
        if args.resume:
            reviewed_ref=saved_reviewed_ref
        elif args.sequence_policy=='continuous_events_v1':
            reviewed_ref='approved_only'
        else:
            reviewed_ref='off'
    if reviewed_ref not in ('approved_only','off'):raise ValueError('Unknown reviewed reference policy')
    if args.resume and reviewed_ref!=saved_reviewed_ref:
        raise ValueError('Changing reviewed_reference_policy requires --init-checkpoint finetune, not --resume')
    if reviewed_ref!='off' and args.sequence_policy!='continuous_events_v1':
        raise ValueError('Reviewed reference policy requires continuous event training')
    args.reviewed_reference_policy=reviewed_ref
    # Reference availability and annotation approval are independent. Only an
    # actual old resume retains the old weight-only supervision semantics.
    if args.sequence_policy == 'continuous_events_v1':
        saved_window = parent_contract.get('training_window_events', args.refresh_events)
        if args.resume and requested_training_window not in (None, saved_window):
            raise ValueError('Changing training-window-events requires --init-checkpoint, not --resume')
        default_window = saved_window if args.resume else args.refresh_events
        args.training_window_events = default_window if requested_training_window is None else requested_training_window
        if type(args.training_window_events) is not int or args.training_window_events < 1:
            raise ValueError('training-window-events must be a positive integer')
    elif requested_training_window is not None:
        raise ValueError('training-window-events requires continuous event training')
    else:
        args.training_window_events = None
    from .difficulty import FITTED_SOURCES, locate_scorer, make_scorer, resolve_selection, scorer_path_key
    difficulty = resolve_selection(getattr(args, 'difficulty_source', 'auto'),
        getattr(args, 'min_level', None), getattr(args, 'max_level', None),
        min_declared_level=getattr(args, 'min_declared_level', None),
        difficulty_slots=getattr(args, 'difficulty_slots', None),
        parent=parent_contract, resume=bool(args.resume))
    args.difficulty_source, args.min_level, args.max_level = difficulty.source, difficulty.min_level, difficulty.max_level
    args.min_declared_level = difficulty.min_declared_level
    args.difficulty_slots = list(difficulty.difficulty_slots)
    scorer = None
    if difficulty.source in FITTED_SOURCES:
        key = scorer_path_key(difficulty.source)
        saved_directory = (parent.get('args') or {}).get(key) if args.resume else None
        root = locate_scorer(difficulty.source, args.cache_dir, getattr(args,key,None), saved_directory)
        setattr(args,key,str(root))
        scorer = make_scorer(difficulty.source, root, args.cache_dir)
    difficulty_contract = difficulty.contract(scorer)
    if args.resume and difficulty_contract != parent_contract.get('difficulty_selection'):
        raise ValueError('断点续训需要沿用检查点的估分版本；切换版本可用新目录微调。')
    zero_unmarked_policy = resolve_zero_unmarked_policy(parent_contract,resume=bool(args.resume))
    reference_quality_policy = (parent_contract.get('reference_quality_policy','legacy_v1')
                                if args.resume else 'event_mask_v1')
    auto_quality_policy = resolve_auto_quality_policy(getattr(args,'auto_quality_policy','auto'),
        args.manifest, parent_contract, resume=bool(args.resume),
        quality_enabled=getattr(args,'quality_enabled',True), sequence_policy=args.sequence_policy)
    if parent and args.sequence_policy!=parent_sequence:raise ValueError('Continuous time language requires fresh training in a new directory')
    args.condition_policy, migration = resolve_training_policy(getattr(args, 'condition_policy', None),
        (parent.get('model_config') or parent.get('config')) if parent else None, resume=bool(args.resume))
    from .onset_policy import resolve_policy as resolve_onset_policy
    args.onset_readout_policy, onset_migration = resolve_onset_policy(getattr(args,'onset_readout','auto'),
        (parent.get('model_config') or parent.get('config')) if parent else None, resume=bool(args.resume))
    from .rhythm_structure import resolve_policy as resolve_structure_policy
    args.structure_policy, structure_migration = resolve_structure_policy(getattr(args,'rhythm_structure','auto'),
        (parent.get('model_config') or parent.get('config')) if parent else None, resume=bool(args.resume))
    from .intensity import resolve_policy as resolve_curve_policy
    args.intensity_policy,curve_migration=resolve_curve_policy(getattr(args,'intensity_curve','auto'),
        (parent.get('model_config') or parent.get('config')) if parent else None,resume=bool(args.resume))
    # Existing checkpoints keep their original attention function. Fresh runs
    # use bounded Q/K scores; loading weights must never silently change this.
    args.attention_normalization = parent_model.get('attention_normalization', 'off') if parent else 'qk_rms_v1'
    from .music_policy import resolve_policy as resolve_music_policy
    args.music_context_policy,music_migration=resolve_music_policy(getattr(args,'music_context','auto'),
        parent_model if parent else None,resume=bool(args.resume))
    for key,fresh in (('audio_layers',4),('song_layers',2)):
        requested=getattr(args,key,None)
        setattr(args,key,(parent_model.get(key,2) if parent else fresh) if requested is None else requested)
    depth_migration=bool(parent and args.audio_layers!=parent_model.get('audio_layers',2))
    if depth_migration and (args.resume or args.audio_layers<parent_model.get('audio_layers',2)):
        raise ValueError('Audio encoder expansion requires init-checkpoint and a greater audio_layers')
    from .rhythm_learning import POLICY as LEGACY_LEARNING,PROPER_POLICY
    args.rhythm_learning_policy=(parent['training_contract'].get('rhythm_learning',LEGACY_LEARNING)
                                if args.resume else PROPER_POLICY)
    if args.intensity_policy!='off' and args.sequence_policy!='continuous_events_v1':
        raise ValueError('Intensity curves require continuous event training')
    if args.structure_policy!='off' and args.sequence_policy!='continuous_events_v1':
        raise ValueError('Rhythm structure requires continuous event training')
    if migration or onset_migration or structure_migration or curve_migration or music_migration or depth_migration:
        if directory == source_directory or any(path.exists() for path in artifacts):
            raise ValueError('Condition/onset/rhythm-structure migration requires a new checkpoint directory; retain the parent')
    from .acceleration import training_device
    requested_device=(parent.get('training_contract',{}).get('device_type',args.device)
                      if args.resume and args.device=='auto' else args.device)
    device=torch.device(training_device(requested_device))
    if device.type=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA requested but unavailable")
    requested_precision=(parent.get('training_contract',{}).get('precision',args.precision)
                         if args.resume and args.precision=='auto' else args.precision)
    precision=("bf16" if device.type=='xpu' or (device.type=="cuda" and torch.cuda.is_bf16_supported()) else "fp32") if requested_precision=="auto" else requested_precision
    if precision not in {"bf16","fp16","fp32"}:raise ValueError("Unknown precision")
    if precision=="bf16" and device.type=="cuda" and not torch.cuda.is_bf16_supported():raise RuntimeError("BF16 requested but unsupported; choose precision explicitly")
    if precision=="fp16" and device.type!="cuda":raise ValueError("FP16 training requires CUDA")
    from .acceleration import audio_runtime
    prepare_device=audio_runtime(args.prepare_device,'cuda' if torch.cuda.is_available() else 'cpu')
    cuda_budget = _apply_cuda_budget(args.cuda_memory_budget_gb,
        device.type == 'cuda' or (args.audio_backend == 'mert' and prepare_device == 'cuda'))
    resource_budgets = dict(cuda_memory_budget_gb=cuda_budget, audio_cache_gb=args.audio_cache_gb)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    inherited_audio_block = parent_model.get('local_audio_block_beats', 1)
    requested_audio_block = getattr(args, 'local_audio_block_beats', None)
    args.local_audio_block_beats = inherited_audio_block if requested_audio_block is None else requested_audio_block
    audio_block_migration = bool(parent and args.local_audio_block_beats != inherited_audio_block)
    if audio_block_migration and (args.resume or directory == source_directory or any(p.exists() for p in artifacts)):
        raise ValueError('Changing local audio sharing requires init-checkpoint in a new directory')
    config=ComposerV2Config.from_dict({key:getattr(args,key) for key in ("d_model","layers","nhead","dim_feedforward","dropout","window_beats","history_beats","future_beats","ticks_per_beat","max_tokens","audio_backend","mert_model","condition_policy","sequence_policy","refresh_events","onset_readout_policy","structure_policy","intensity_policy","audio_layers","song_layers","music_context_policy","local_audio_block_beats","attention_normalization","future_configuration_policy","future_configuration_weight")})
    parent_config = ComposerV2Config.from_dict(parent.get('model_config') or parent.get('config')).to_dict() if parent else None
    comparable_config = dict(config.to_dict())
    if migration: comparable_config['condition_policy'] = parent_policy
    if onset_migration: comparable_config['onset_readout_policy'] = parent_config['onset_readout_policy']
    if structure_migration: comparable_config['structure_policy'] = parent_config['structure_policy']
    if curve_migration: comparable_config['intensity_policy'] = parent_config['intensity_policy']
    if music_migration: comparable_config['music_context_policy'] = parent_config['music_context_policy']
    if depth_migration: comparable_config['audio_layers'] = parent_config['audio_layers']
    if audio_block_migration:
        comparable_config['local_audio_block_beats'] = inherited_audio_block
    if args.init_checkpoint:
        comparable_config['future_configuration_policy'] = parent_config['future_configuration_policy']
        comparable_config['future_configuration_weight'] = parent_config['future_configuration_weight']
    if parent and comparable_config != parent_config:
        raise ValueError("Parent model/audio/token contract differs; use its exact configuration or fresh training")
    codec=ComponentCodec.from_dict(parent["codec"]) if parent else None
    dialect_policy = (parent['training_contract'].get('chart_dialect', chart_dialect.LEGACY_STRICT_POLICY)
                      if args.resume else chart_dialect.POLICY)
    if args.validation_manifest:
        from maichart.composer.evaluation import assert_unseen
        assert_unseen(args.manifest,args.validation_manifest,cache_dir=args.cache_dir)
    emit("runtime",stage="preparation",audio_backend=args.audio_backend,prepare_device=prepare_device,manifest=args.manifest,
         **resource_budgets)
    audio_store = AudioFeatureStore(args.cache_dir, args.audio_backend, args.mert_model,
        prepare_device, max_mapped_bytes=int(args.audio_cache_gb * 2**30))
    options=dict(cache_dir=args.cache_dir,codec=codec,window_beats=args.window_beats,history_beats=args.history_beats,
        future_beats=args.future_beats,ticks_per_beat=args.ticks_per_beat,max_tokens=args.max_tokens,
        audio_backend=args.audio_backend,mert_model=args.mert_model,prepare_device=prepare_device,limit_charts=args.limit_charts,
        condition_policy=args.condition_policy,audio_store=audio_store,dialect_policy=dialect_policy,
        quality_enabled=getattr(args,'quality_enabled',True),auto_quality_policy=auto_quality_policy,
        difficulty_scorer=scorer, **difficulty.dataset_options())
    dataset_class=ComposerV2Dataset
    if args.sequence_policy=='continuous_events_v1':
        from .stream_data import StreamDataset
        dataset_class=StreamDataset;options['refresh_events']=args.refresh_events
        options['local_audio_block_beats'] = args.local_audio_block_beats
        options.update(timing_recovery_policy=recovery_policy,augmentation_seed=args.seed,
                       onset_readout_policy=args.onset_readout_policy,structure_policy=args.structure_policy,
                       intensity_policy=args.intensity_policy,rhythm_learning_policy=args.rhythm_learning_policy,
                       music_context_policy=args.music_context_policy,
                       future_configuration_policy=args.future_configuration_policy,
                       reviewed_reference_policy=reviewed_ref,zero_unmarked_policy=zero_unmarked_policy,
                       reference_quality_policy=reference_quality_policy)
    train_options=dict(options)
    if args.sequence_policy=='continuous_events_v1':
        train_options.update(reference_augmentation=True, training_window_events=args.training_window_events,
            musical_context_policy=args.musical_context_policy,
            musical_context_probability=args.musical_context_probability,
            musical_relation_policy=args.musical_relation_policy,
            musical_relation_fraction=args.musical_relation_fraction)
    with scorer if scorer is not None else nullcontext():
        from .codec import VocabularyCoverageError
        try:
            dataset=dataset_class(args.manifest,extend_vocabulary=bool(args.init_checkpoint),**train_options)
        except VocabularyCoverageError as error:
            if not args.resume:
                raise
            raise ValueError(f'{error}。断点续训使用原训练素材和词表。'
                             '要使用扩充后的训练素材，请选「微调」并使用新检查点目录。') from error
        if auto_quality_policy == 'auto_v1' and dataset.auto_annotations_sha256 is None:
            raise ValueError('准备训练数据时未找到匹配的自动质检标注')
        vocabulary_changed = codec is not None and dataset.codec is not codec
        if vocabulary_changed and (directory == source_directory or any(path.exists() for path in artifacts)):
            raise ValueError('Vocabulary expansion requires a new checkpoint directory; retain the parent')
        parent_vocab_size = codec.vocab_size if codec is not None else None
        codec=dataset.codec
        validation_options={**options,"codec":codec,"limit_charts":0,"validate_fixed_vocabulary":True}
        if args.sequence_policy=='continuous_events_v1':
            validation_options.update(timing_recovery_policy='off', musical_context_policy='off',
                                      auxiliary_validation=args.future_configuration_policy != 'off')
        validation=dataset_class(args.validation_manifest,**validation_options) if args.validation_manifest else None
    if scorer is not None:
        emit('difficulty_fit', scorer=scorer.identity,cache_hits=scorer.hits,computed=scorer.computed,
             train_charts=len(dataset.songs), outside_range=sum(
                 row['reason']=='fitted_level_out_of_range' for row in dataset.audit['filtered']),
             unsupported_slot=sum(row['reason']=='difficulty_slot_unsupported' for row in dataset.audit['filtered']))
    identity=dict(train=_identity(dataset),validation=_identity(validation) if validation is not None else None)
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    if args.resume:
        batch_order = parent['training_contract']['sampler']
    else:
        batch_order = AUDIO_LOCAL_BATCH_ORDER if args.sequence_policy == 'continuous_events_v1' else GLOBAL_BATCH_ORDER
    from .batching import FULL_INPUT, resolve_layout, compatible_contract, resume_cursor
    batch_layout,source_batch_size,validation_batch_size=resolve_layout(
        getattr(args,'batch_layout','auto'),args.batch_size,args.grad_accumulation,parent if args.resume else None)
    args.batch_layout=batch_layout
    loader,sampler=_loader(dataset,args.batch_size,args.seed,True,device,order=batch_order,
        batch_layout=batch_layout,accumulation=args.grad_accumulation,source_batch_size=source_batch_size)
    from .validation_sampling import POLICY as VALIDATION_POLICY,LEGACY as LEGACY_VALIDATION,coverage_report
    validation_policy=parent['training_contract'].get('validation_sampling',LEGACY_VALIDATION) if args.resume else VALIDATION_POLICY
    validation_loader=_validation_loader(validation,validation_batch_size,args.seed,device,args.validation_max_batches,validation_policy) if validation else None
    validation_coverage = getattr(validation_loader, 'validation_coverage', None)
    if validation is not None and validation_loader is None:
        validation_coverage = coverage_report(validation, [], validation_policy, maximum_batches=args.validation_max_batches)
    if validation_coverage and validation_coverage['vocabulary_excluded_chart_records']:
        emit('validation_vocabulary', status='partial' if validation_loader is not None else 'unavailable',
             excluded_charts=validation_coverage['vocabulary_excluded_chart_records'],
             usable_charts=validation_coverage['total_chart_records'])
    updates_per_epoch=math.ceil(len(loader)/args.grad_accumulation)
    total_updates=updates_per_epoch*args.epochs
    contract=dict(batch_size=args.batch_size,grad_accumulation=args.grad_accumulation,seed=args.seed,
        lr=args.lr,lr_schedule=args.lr_schedule,warmup_steps=args.warmup_steps,min_lr_ratio=args.min_lr_ratio,
        epochs=args.epochs,total_updates=total_updates,precision=precision,device_type=device.type,
        window_policy="stable_beat_window_components_v2",sampler=batch_order,
        loss_normalization="quality_weighted_token_mass_per_update_v1",cpu_threads=args.cpu_threads)
    if not getattr(args,'quality_enabled',True): contract['quality_enabled']=False
    if batch_layout == FULL_INPUT:
        contract.update(batch_layout=batch_layout,batch_source_size=source_batch_size,
                        validation_batch_size=validation_batch_size)
    if validation is not None and validation_policy!=LEGACY_VALIDATION:
        contract.update(validation_sampling=validation_policy,validation_sampling_budget=args.validation_max_batches)
    if args.condition_policy != LEVEL_ONLY:
        contract['condition_contract'] = condition_contract(args.condition_policy)
    if args.sequence_policy=='continuous_events_v1':
        contract.update(window_policy='causal_event_slices_no_boundary_eos_v1',refresh_events=args.refresh_events,
            acoustic_policy='query_integer_beat_tiles_v1')
        if args.training_window_events != args.refresh_events:
            contract['training_window_events'] = args.training_window_events
        if args.local_audio_block_beats != 1:
            contract['acoustic_policy'] = 'query_two_beat_shared_tiles_v1'
    if recovery_policy!='off':contract['timing_recovery']=recovery_policy
    if args.musical_context_policy != 'off':
        contract.update(musical_context_policy=args.musical_context_policy,
                        musical_context_probability=args.musical_context_probability)
    if args.musical_relation_policy != 'off':
        contract.update(musical_relation_policy=args.musical_relation_policy,
                        musical_relation_fraction=args.musical_relation_fraction)
    if args.future_configuration_policy != 'off':
        contract.update(future_configuration_policy=args.future_configuration_policy,
                        future_configuration_weight=args.future_configuration_weight,
                        future_configuration_train_view_fraction=configuration_view_fraction(args.future_configuration_policy))
    if args.finetune_scope != 'all':
        contract['finetune_scope'] = args.finetune_scope
    if reviewed_ref!='off':contract['reviewed_reference_policy']=reviewed_ref
    if args.sequence_policy=='continuous_events_v1' and (not args.resume or 'zero_unmarked_policy' in parent_contract):
        contract['zero_unmarked_policy']=zero_unmarked_policy
    if args.sequence_policy=='continuous_events_v1' and (not args.resume or 'reference_quality_policy' in parent_contract):
        contract['reference_quality_policy']=reference_quality_policy
    if args.onset_readout_policy != 'off':contract['onset_readout_policy']=args.onset_readout_policy
    if args.structure_policy != 'off':contract['structure_policy']=args.structure_policy
    if args.intensity_policy!='off':contract['intensity_policy']=args.intensity_policy
    if args.structure_policy=='aligned_rhythm_v2':
        contract['rhythm_learning']=args.rhythm_learning_policy
    if dialect_policy != chart_dialect.LEGACY_STRICT_POLICY:
        contract['chart_dialect'] = dialect_policy
    if args.attention_normalization != 'off':
        contract['attention_normalization'] = args.attention_normalization
    if auto_quality_policy != 'off':
        contract['auto_quality_policy'] = auto_quality_policy
    if difficulty_contract is not None:
        contract['difficulty_selection'] = difficulty_contract
    if args.resume and (not _same_data_identity(parent["data_identity"],identity)
                        or not compatible_contract(parent["training_contract"],contract)):
        raise ValueError("Exact resume data/optimizer/schedule contract changed; retain the parent and use compatible finetune in a new directory")
    model=ComposerV2(config,codec).to(device)
    from .training_memory import configure as configure_memory, configure_projection_packing, configure_decoder_flash_attention, cuda_stats
    memory_policy=configure_memory(model,getattr(args,'activation_checkpointing','auto'),device)
    projection_policy=configure_projection_packing(model,args.attention_projection_packing,device)
    flash_policy=configure_decoder_flash_attention(model,flash,device)
    if parent:
        parent_state = parent['model_state_dict']
        if vocabulary_changed:
            parent_state = _extend_vocabulary_state(model, parent_state, parent_vocab_size)
        if migration or onset_migration or structure_migration or curve_migration or music_migration or depth_migration or future_migration:
            expected_missing = {'condition_missing.weight'} if migration else set()
            if music_migration:
                expected_missing.update(k for k in model.state_dict() if k not in parent_state and
                    (k=='song_detail.weight' or k.endswith('.relative_beat') or k.startswith(('paired_music.','entry_preview.'))))
            if depth_migration:
                for index in range(parent_config['audio_layers'],config.audio_layers):
                    block=model.audio_blocks[index]
                    # Both residual branches are exactly zero at initialization.
                    for projection in (block.attention.out,block.ff[-1]):
                        torch.nn.init.zeros_(projection.weight);torch.nn.init.zeros_(projection.bias)
                    expected_missing.update(k for k in model.state_dict() if k.startswith(f'audio_blocks.{index}.'))
            if onset_migration:
                if args.onset_readout_policy != 'off':
                    expected_missing.update(k for k in model.state_dict() if k.startswith('onset_readout.'))
                else:
                    parent_state = {k:v for k,v in parent_state.items() if not k.startswith('onset_readout.')}
            if structure_migration:
                from .rhythm_structure import migration_state
                parent_state, missing_structure = migration_state(model,parent_state)
                expected_missing.update(missing_structure)
            if curve_migration:
                if args.intensity_policy!='off':
                    expected_missing.update(k for k in model.state_dict() if k.startswith('intensity.'))
                else:parent_state={k:v for k,v in parent_state.items() if not k.startswith('intensity.')}
            if future_migration:
                from .local_writing import migrate_state
                parent_state, writing_missing = migrate_state(model, parent_state)
                expected_missing.update(writing_missing)
            loaded = model.load_state_dict(parent_state, strict=False)
            if set(loaded.missing_keys) != expected_missing or loaded.unexpected_keys:
                raise ValueError('Unexpected state keys during explicit migration: ' + str(loaded))
            emit('runtime',stage='model_migration',parent=str(parent_path),
                source_policy=parent_policy,condition_policy=args.condition_policy,
                onset_readout_policy=args.onset_readout_policy,structure_policy=args.structure_policy,intensity_policy=args.intensity_policy,
                music_context_policy=args.music_context_policy,audio_layers=config.audio_layers,song_layers=config.song_layers,
                future_configuration_policy=args.future_configuration_policy,
                zero_initialization=not future_migration or args.future_configuration_policy in ('off',*AUDIO_WRITING_POLICIES))
        else:
            model.load_state_dict(parent_state,strict=True)
    if hasattr(model,'onset_readout') and (not parent or onset_migration):
        model.onset_readout.fit_normalization(dataset.songs)
    trainable_names, trainable_parameters = configure_trainable_parameters(model, args.finetune_scope)
    trainable_count = sum(parameter.numel() for parameter in trainable_parameters)
    parameter_audit = dict(scope=args.finetune_scope, trainable_parameters=trainable_count,
                           total_parameters=sum(parameter.numel() for parameter in model.parameters()))
    if args.finetune_scope != 'all':
        parameter_audit['trainable_parameter_names'] = trainable_names
    optimizer=torch.optim.AdamW(trainable_parameters,lr=args.lr,weight_decay=.01)
    scaler=torch.amp.GradScaler(device.type if device.type=='xpu' else 'cuda',enabled=precision=="fp16")
    state=dict(epoch=0,batch_cursor=0,optimizer_steps=0,update_attempts=0,best_loss=None,last_loss=None,epoch_loss_sum=0.,epoch_weight_sum=0.)
    if args.resume:
        optimizer.load_state_dict(parent["optimizer_state_dict"])
        state.update(parent["training_state"])
        if parent['training_contract'] != contract:
            previous_cursor=state['batch_cursor']
            state['batch_cursor']=resume_cursor(state,parent['training_contract'],contract,len(loader))
            emit('runtime',stage='batch_layout_migration',previous_batch=previous_cursor,
                 resumed_batch=state['batch_cursor'],effective_batch=args.batch_size*args.grad_accumulation,
                 optimizer_steps=state['optimizer_steps'],batch_layout=batch_layout)
        if parent.get("scaler_state_dict"):scaler.load_state_dict(parent["scaler_state_dict"])
        restore_rng(parent["rng_state"])
    directory.mkdir(parents=True,exist_ok=True)
    for split in (dataset, validation):
        if split is None:
            continue
        levels = [song.level for song in split.songs]
        split.audit['condition_level'] = dict(source=difficulty.source, prepared_charts=len(levels),
            min_level=min(levels, default=None), max_level=max(levels, default=None))
    (directory/'dataset_audit.json').write_text(json.dumps(
        dict(train=dataset.audit,validation=validation.audit if validation is not None else None),
        ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    if args.resume and not artifacts[1].exists():
        original_best=source_directory/"composer_v2_best.pt"
        matching=False
        if original_best.is_file():
            best_payload=read(original_best)
            matching=(_same_data_identity(best_payload.get("data_identity"),identity) and best_payload.get("training_contract")==contract
                and best_payload.get("family")==FAMILY and best_payload.get("schema")==SCHEMA)
            del best_payload
        if matching:shutil.copy2(original_best,artifacts[1])
        else:
            state["best_loss"]=None
            emit("runtime",stage="best_selection_reset",reason="Previous best weights unavailable; best selection restarts while optimizer and cursor resume exactly")
    best_basis=("validation_automatic_conditions_quality_weighted_token_ce" if args.condition_policy==RADAR_STYLE
        else "validation_quality_weighted_token_ce") if validation_loader is not None else (
            "training_quality_weighted_token_ce_validation_unavailable" if validation is not None
            else "training_quality_weighted_token_ce_no_validation")
    epoch_loss_gpu=torch.tensor(float(state.get("epoch_loss_sum",0.)),device=device)
    epoch_mass=float(state.get("epoch_weight_sum",0.))
    running_mass=0.
    running_relation_mass=0.
    def supervision_audit():
        fraction=running_relation_mass/running_mass if running_mass else 0.
        return dict(token_weight_sum=running_mass, musical_relation_extra_mass=running_relation_mass,
                    musical_relation_extra_fraction=fraction)
    def flush_epoch():
        state["epoch_loss_sum"]=float(epoch_loss_gpu)
        state["epoch_weight_sum"]=epoch_mass
        if epoch_mass:state["last_loss"]=state["epoch_loss_sum"]/max(1.,epoch_mass)
    def checkpoint(improved=False):
        save_started = time.perf_counter()
        flush_epoch()
        payload=dict(schema=SCHEMA,family=FAMILY,model_config=config.to_dict(),config=config.to_dict(),
            codec=codec.to_dict(),model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),
            scaler_state_dict=scaler.state_dict(),rng_state=rng_state(),training_state=dict(state),
            data_identity=identity,training_contract=contract,args=vars(args),best_basis=best_basis,
            parameter_audit=parameter_audit,
            supervision_audit=supervision_audit(),
            condition_contract=condition_contract(args.condition_policy),
            selection_loss=state["best_loss"] if state["best_loss"] is not None else state["last_loss"],
            best_provisional=state["best_loss"] is None)
        save(artifacts[0],payload)
        if improved or not artifacts[1].exists():save(artifacts[1],payload)
        inference={key:value for key,value in payload.items() if key not in ("optimizer_state_dict","scaler_state_dict","rng_state")}
        inference["inference_only"]=True
        save(artifacts[2],inference)
        emit("checkpoint",path=str(artifacts[0]),epoch=state["epoch"],batch_cursor=state["batch_cursor"],optimizer_steps=state["optimizer_steps"],best_basis=best_basis,
             save_sec=time.perf_counter()-save_started)
    emit("runtime",stage="training",device=str(device),precision=precision,audio_backend=args.audio_backend,
        windows=len(dataset),validation_windows=len(validation) if validation else 0,model_config=config.to_dict(),
        epoch_target=args.epochs,batches_per_epoch=len(loader),resume_epoch=int(state['epoch']),
        resume_batch_cursor=int(state['batch_cursor']),
        trainable_parameters=trainable_count,parameter_audit=parameter_audit,effective_batch=args.batch_size*args.grad_accumulation,
         train_charts=len(dataset.songs),validation_charts=len(validation.songs) if validation else 0,
         preparation=dataset.audit.get('preparation', {}), batch_order=batch_order,batch_layout=batch_layout,
         batch_source_size=source_batch_size,validation_batch_size=validation_batch_size,**resource_budgets,
         timing_recovery=dataset.audit.get('timing_recovery'),
         musical_context=dataset.audit.get('musical_context'),
         musical_context_policy=args.musical_context_policy,
         musical_context_probability=args.musical_context_probability,
         musical_relation=dataset.audit.get('musical_relation'),
         musical_relation_policy=args.musical_relation_policy,
         musical_relation_fraction=args.musical_relation_fraction,
         future_configuration=dataset.audit.get('future_configuration'),
         future_configuration_policy=args.future_configuration_policy,
         future_configuration_weight=args.future_configuration_weight,
         future_configuration_train_view_fraction=configuration_view_fraction(args.future_configuration_policy),
         training_window_events=args.training_window_events,
         rhythm_structure=dataset.audit.get('rhythm_structure'),
         intensity_policy=args.intensity_policy,
         validation_coverage=validation_coverage,
        best_basis=best_basis,activation_checkpointing=memory_policy,attention_projection_packing=projection_policy,
        decoder_flash_attention=flash_policy,biased_attention_backward="fp32_math_recompute_v1",
        attention_normalization=config.attention_normalization)
    model.train();optimizer.zero_grad(set_to_none=True)
    began=time.perf_counter();previous=began;data_sec=0.;compute_sec=0.;seen_samples=0;seen_tokens=0
    input_tokens=0;input_slots=0
    running=torch.zeros(8,device=device);running_time_mass=0.;running_component_mass=0.;running_phase_mass=0.
    replay_keys=0
    gesture_pairs_seen=0
    group_metrics=torch.zeros(7,device=device);group_finite=torch.ones((),device=device,dtype=torch.bool)
    group_mass=0.;group_time_mass=0.;group_component_mass=0.;group_phase_mass=0.;group_relation_mass=0.;stop=False
    time_tokens=model.time_tokens.detach().cpu();phase_tokens=model.phase_targets.detach().cpu()
    if device.type=="cuda":torch.cuda.reset_peak_memory_stats(device)
    elif device.type=='xpu':torch.xpu.reset_peak_memory_stats(device)
    for epoch in range(int(state["epoch"]),args.epochs):
        sampler.set_epoch(epoch)
        cursor=int(state["batch_cursor"]) if epoch==state["epoch"] else 0
        epoch_started=time.perf_counter()
        last_position=epoch_started-5.
        emit('progress',stage='epoch_start',epoch=epoch+1,batch=cursor,total=len(loader),
             optimizer_steps=state['optimizer_steps'])
        epoch_loss_gpu=torch.tensor(float(state.get("epoch_loss_sum",0.)),device=device)
        epoch_mass=float(state.get("epoch_weight_sum",0.))
        pending=[]
        for batch_index,batch in _batches(loader,cursor,pending=pending):
            now=time.perf_counter();data_sec+=now-previous
            tokens,mass,time_mass,component_mass,relation_mass=_cpu_budget(batch,time_tokens)
            input_tokens+=int(batch['token_mask'].sum())
            input_slots+=batch['token_mask'].numel()
            phase_mass=_phase_budget(batch,phase_tokens)
            cpu_valid=math.isfinite(mass) and mass>=0
            # Keep the unnormalized numerator numerically modest for FP16;
            # divide by the complete update's quality-weighted token budget below.
            budget=max(1.,mass) if cpu_valid else 1.
            batch=_device_batch(batch,device)
            if 'gesture_weights' in batch:
                gesture_pairs_seen+=batch['gesture_weights'].numel()
            with _autocast(device,precision):
                loss,metrics=model.loss(model(batch),batch)
            scaler.scale(loss*(budget/1024.)).backward()
            if 'layout_replay' in batch:
                replay_keys+=len(batch['layout_replay']['coordinates'])
                running[7]+=metrics['layout_replay_changed']
            group_finite=group_finite&torch.isfinite(loss.detach())&cpu_valid
            group_metrics+=torch.stack((metrics['token_loss']*budget,
                metrics["time_loss"]*max(1.,time_mass),metrics["component_loss"]*max(1.,component_mass),metrics["phase_loss"]*max(1.,phase_mass),
                metrics['curve_loss']*budget,loss.detach()*budget,metrics['future_configuration_loss']*budget))
            group_mass+=mass if cpu_valid else 0.
            group_time_mass+=time_mass if cpu_valid else 0.
            group_component_mass+=component_mass if cpu_valid else 0.
            group_phase_mass+=phase_mass if cpu_valid else 0.
            group_relation_mass+=relation_mass if cpu_valid else 0.
            seen_samples+=len(batch["input_ids"]);seen_tokens+=tokens
            boundary=(batch_index+1)%args.grad_accumulation==0 or batch_index+1==len(loader)
            if boundary:
                scaler.unscale_(optimizer)
                for parameter in trainable_parameters:
                    if parameter.grad is not None:parameter.grad.mul_(1024./max(1.,group_mass))
                norm=torch.nn.utils.clip_grad_norm_(trainable_parameters,1.)
                finite=bool(group_finite&torch.isfinite(norm)&(group_mass>0))
                attempt=int(state["update_attempts"])
                if args.warmup_steps and attempt<args.warmup_steps:factor=(attempt+1)/args.warmup_steps
                else:
                    progress=min(1.,max(0.,(attempt-args.warmup_steps)/max(1,total_updates-args.warmup_steps)))
                    factor=args.min_lr_ratio+(1-args.min_lr_ratio)*.5*(1+math.cos(math.pi*progress)) if args.lr_schedule=="cosine" else 1.
                for group in optimizer.param_groups:group["lr"]=args.lr*factor
                if finite:
                    scaler.step(optimizer);state["optimizer_steps"]+=1
                    epoch_loss_gpu+=group_metrics[0];epoch_mass+=group_mass
                    running[:7]+=group_metrics;running_mass+=group_mass
                    running_relation_mass+=group_relation_mass
                    running_time_mass+=group_time_mass;running_component_mass+=group_component_mass;running_phase_mass+=group_phase_mass
                else:emit("progress",stage="nonfinite_update_skipped",epoch=epoch,batch=batch_index+1,optimizer_steps=state["optimizer_steps"])
                scaler.update();optimizer.zero_grad(set_to_none=True)
                group_metrics.zero_();group_finite.fill_(True)
                group_mass=0.;group_time_mass=0.;group_component_mass=0.;group_phase_mass=0.;group_relation_mass=0.
                state.update(epoch=epoch,batch_cursor=batch_index+1,update_attempts=attempt+1)
            ended=time.perf_counter();compute_sec+=ended-now
            if boundary and ended-last_position>=5.:
                emit('position',stage='training',epoch=epoch+1,batch=batch_index+1,total=len(loader),
                     optimizer_steps=state['optimizer_steps'],update_attempts=state['update_attempts'],
                     epoch_elapsed_sec=ended-epoch_started,epoch_start_batch=cursor,
                     epoch_processed_batches=batch_index+1-cursor)
                last_position=ended
            if boundary and (state["update_attempts"]%args.log_interval==0 or batch_index+1==len(loader)):
                _sync(device)
                values=running.cpu().tolist();flush_epoch();elapsed=time.perf_counter()-began
                emit("progress",stage="training",epoch=epoch+1,batch=batch_index+1,total=len(loader),
                    epoch_elapsed_sec=time.perf_counter()-epoch_started,epoch_start_batch=cursor,
                    epoch_processed_batches=batch_index+1-cursor,update_attempts=state['update_attempts'],
                    optimizer_steps=state["optimizer_steps"],loss=values[0]/max(1.,running_mass),
                    loss_scope="run_quality_weighted_token_mean",
                    curve_loss=values[4]/max(1.,running_mass),objective_loss=values[5]/max(1.,running_mass),
                    future_configuration_loss=values[6]/max(1.,running_mass),
                    layout_replay_keys=replay_keys,layout_replay_changed=int(values[7]),
                    gesture_pairs_seen=gesture_pairs_seen,
                    token_count=seen_tokens,
                    tokens_per_sec=seen_tokens/max(elapsed,1e-9),samples_per_sec=seen_samples/max(elapsed,1e-9),
                    input_padding_fraction=1.-input_tokens/max(input_slots,1),batch_layout=batch_layout,
                    time_loss=values[1]/max(1.,running_time_mass),component_loss=values[2]/max(1.,running_component_mass),phase_loss=values[3]/max(1.,running_phase_mass),
                    data_sec=data_sec,compute_sec=compute_sec,compute_measurement="host_submission_wall_including_update_boundary_wait",
                    prefetch={"consumer_wait_sec":data_sec,"max_batches":1},elapsed_sec=elapsed,lr=optimizer.param_groups[0]["lr"],
                    audio_banks=dataset.audio_store.mapping_stats(),grad_norm=float(norm.detach()),**cuda_stats(device),
                    gpu_peak_bytes=(torch.cuda.max_memory_allocated(device) if device.type=="cuda" else
                                    torch.xpu.max_memory_allocated(device) if device.type=='xpu' else 0),
                    **supervision_audit())
            if boundary and finite and state["optimizer_steps"]%args.checkpoint_interval_steps==0:
                improved=False
                # Do not wait an entire multi-hour epoch to detect regression.
                # Epoch-end validation below already covers its final update.
                if validation_loader is not None and batch_index+1<len(loader):
                    validation_started=time.perf_counter()
                    emit('progress',stage='validation',epoch=epoch+1,total=len(validation_loader),
                         optimizer_steps=state['optimizer_steps'],scope='checkpoint',started_unix=time.time())
                    score=_validate_checkpoint(model,validation_loader,device,precision,args.validation_max_batches,pending=pending)
                    improved=state['best_loss'] is None or score<state['best_loss']
                    if improved:
                        state['best_loss']=score
                        state['validation_improved_in_epoch']=True
                    emit('progress',stage='checkpoint_validation',epoch=epoch+1,batch=batch_index+1,total=len(loader),
                         optimizer_steps=state['optimizer_steps'],validation_loss=score,
                         validation_time_loss=model.validation_time_loss,
                         validation_nonphase_time_loss=model.validation_nonphase_time_loss,
                         validation_phase_loss=model.validation_phase_loss,
                         validation_curve_loss=model.validation_curve_loss,
                         validation_future_configuration_loss=model.validation_future_configuration_loss,
                         validation_coverage=validation_coverage,improved=improved,
                         validation_sec=time.perf_counter()-validation_started,loss_scope='fixed_holdout')
                checkpoint(improved)
            if boundary and args.max_steps and state["update_attempts"]>=args.max_steps:
                stop=True;break
            previous=time.perf_counter()
        if state["batch_cursor"]>=len(loader):
            validation_started = time.perf_counter()
            epoch_train_sec=validation_started-epoch_started
            if validation_loader is not None:
                emit('progress',stage='validation',epoch=epoch+1,total=len(validation_loader),
                     epoch_train_sec=epoch_train_sec,started_unix=time.time())
            validation_loss=_validate(model,validation_loader,device,precision,args.validation_max_batches)
            validation_sec = time.perf_counter() - validation_started if validation_loader is not None else 0.
            flush_epoch()
            score=validation_loss if validation_loss is not None else state["last_loss"]
            improved=score is not None and (state["best_loss"] is None or score<state["best_loss"])
            if improved:state["best_loss"]=score
            if validation_loss is not None:
                epoch_improved=improved or state.get('validation_improved_in_epoch',False)
                state['bad_validation_epochs']=0 if epoch_improved else int(state.get('bad_validation_epochs',0))+1
            emit("progress",stage="epoch_complete",epoch=epoch+1,total=args.epochs,loss=score,
                epoch_train_sec=epoch_train_sec,epoch_start_batch=cursor,
                epoch_processed_batches=len(loader)-cursor,epoch_total_batches=len(loader),
                epoch_train_loss=state["last_loss"],validation_loss=validation_loss,
                validation_time_loss=getattr(model,"validation_time_loss",None),
                validation_nonphase_time_loss=getattr(model,"validation_nonphase_time_loss",None),
                validation_phase_loss=getattr(model,"validation_phase_loss",None),
                validation_curve_loss=getattr(model,'validation_curve_loss',None),
                validation_curve_mae=getattr(model,'validation_curve_mae',None),
                validation_future_configuration_loss=getattr(model,'validation_future_configuration_loss',None),
                validation_coverage=validation_coverage,
                 validation_sec=validation_sec,
                loss_scope="epoch_validation" if validation_loss is not None else "epoch_training",
                best_basis=best_basis,optimizer_steps=state["optimizer_steps"],**supervision_audit())
            state.update(epoch=epoch+1,batch_cursor=0,epoch_loss_sum=0.,epoch_weight_sum=0.,validation_improved_in_epoch=False)
            # Checkpoint's next-epoch accumulator must agree with its zero cursor.
            epoch_loss_gpu.zero_();epoch_mass=0.
            checkpoint(improved)
            snapshot_started=time.perf_counter()
            snapshot=save_epoch_snapshot(artifacts[0],state["epoch"],args.save_every)
            if snapshot is not None:
                emit("checkpoint",stage="epoch_snapshot",path=str(snapshot),epoch=state["epoch"],
                     batch_cursor=0,optimizer_steps=state["optimizer_steps"],
                     save_sec=time.perf_counter()-snapshot_started)
            if (validation_loss is not None and args.early_stopping_patience and
                    state.get('bad_validation_epochs',0)>=args.early_stopping_patience):
                emit('runtime',stage='early_stopped',epoch=state['epoch'],best_loss=state['best_loss'],
                     reason='validation_no_improvement',patience=args.early_stopping_patience)
                stop=True
        elif stop:checkpoint()
        previous=time.perf_counter()
        if stop:break
    if not artifacts[0].exists():checkpoint()
    return state

def main(argv=None):
    args=parser().parse_args(argv)
    try:return train(args)
    except Exception as error:
        emit("runtime",stage="failed",error=str(error),audio_backend=args.audio_backend)
        raise

if __name__=="__main__":main()
