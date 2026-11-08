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
from .checkpoint import FAMILY,SCHEMA,read,save,inspect_payload
from .codec import ComponentCodec
from .data import (AUDIO_LOCAL_BATCH_ORDER, GLOBAL_BATCH_ORDER, ComposerV2Dataset,
                   collate_composer_v2, LengthBucketBatchSampler)
from .conditions import LEVEL_ONLY,RADAR_STYLE,POLICIES,condition_contract,policy_from_config
from . import chart_dialect, timing_recovery


def emit(kind,**values):
    print("composer_v2_"+kind+" "+json.dumps(values,ensure_ascii=False,allow_nan=False),flush=True)

def parser():
    from .defaults import TRAIN_DEFAULTS
    result=argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest",required=True)
    result.add_argument("--validation-manifest")
    result.add_argument("--checkpoint-dir",required=True)
    result.add_argument("--cache-dir",default="cache/chart_composer_v2")
    parents=result.add_mutually_exclusive_group()
    parents.add_argument("--resume");parents.add_argument("--init-checkpoint")
    for key,value in TRAIN_DEFAULTS.items():
        result.add_argument("--"+key.replace("_","-"),default=value,type=type(value))
    result.add_argument("--mert-model")
    result.add_argument('--sequence-policy',choices=('window_v1','continuous_events_v1'),default=None)
    result.add_argument('--refresh-events',type=int,default=16)
    result.add_argument('--timing-recovery',choices=timing_recovery.CHOICES,default='auto')
    from .onset_policy import CHOICES as ONSET_CHOICES
    result.add_argument('--onset-readout',choices=ONSET_CHOICES,default='auto',
                        help='Phase-only acoustic evidence; auto enables fresh/continuous finetune and inherits exact resume')
    from .rhythm_structure import CHOICES as STRUCTURE_CHOICES
    result.add_argument('--rhythm-structure',choices=STRUCTURE_CHOICES,default='auto',
                        help='Sub-beat phase geometry and aligned trusted references; auto enables fresh/continuous finetune and inherits exact resume')
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

def rng_state():
    state=np.random.get_state()
    return dict(python=random.getstate(),numpy=dict(kind=state[0],keys=state[1].tolist(),position=state[2],has_gauss=state[3],cached_gaussian=state[4]),
        torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])

def restore_rng(value):
    random.setstate(value["python"])
    state=value["numpy"]
    np.random.set_state((state["kind"],np.asarray(state["keys"],dtype=np.uint32),state["position"],state["has_gauss"],state["cached_gaussian"]))
    torch.set_rng_state(value["torch"])
    if value.get("cuda"):
        if not torch.cuda.is_available() or len(value["cuda"])!=torch.cuda.device_count():
            raise ValueError("Exact resume requires the saved CUDA RNG device topology")
        torch.cuda.set_rng_state_all(value["cuda"])

def _identity(dataset):
    value=dict(manifest_sha256=dataset.manifest_sha256,charts=dataset.audit["charts"],
        audio=[dict(song_id=s.song_id,difficulty=s.difficulty,identity=s.audio.identity) for s in dataset.songs],
        codec_sha256=_hash(dataset.codec.to_dict()),windows=len(dataset))
    if dataset.condition_policy != LEVEL_ONLY:
        value['condition_contract'] = dataset.condition_contract
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

def _batches(loader,skip=0):
    # Dataset fetching is deterministic; one outstanding CPU batch bounds RAM.
    sampler = getattr(loader, 'batch_sampler', None)
    can_seek = isinstance(sampler, LengthBucketBatchSampler)
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
                if index >= skip:
                    yield index, batch
                index += 1
    finally:
        if can_seek:
            sampler.start_batch = previous_start

def _device_batch(batch,device):
    return {key:value.to(device,non_blocking=True) if isinstance(value,torch.Tensor) else value for key,value in batch.items()}

def _loader(dataset,batch_size,seed,shuffle,device,*,order=GLOBAL_BATCH_ORDER):
    sampler=LengthBucketBatchSampler(dataset,batch_size,seed=seed,shuffle=shuffle,order=order)
    generator=torch.Generator().manual_seed(seed)
    from .stream_data import StreamDataset,collate_stream
    return DataLoader(dataset,batch_sampler=sampler,collate_fn=collate_stream if isinstance(dataset,StreamDataset) else collate_composer_v2,
        num_workers=0,pin_memory=device.type=="cuda",generator=generator),sampler

def _sync(device):
    if device.type=="cuda":torch.cuda.synchronize(device)

def _autocast(device,precision):
    if precision=="fp32":return nullcontext()
    return torch.autocast(device_type=device.type,dtype=torch.bfloat16 if precision=="bf16" else torch.float16)

def _cpu_budget(batch,time_tokens=None):
    active=batch["labels"].ne(-100)
    weights=batch.get("token_weights",active.float()).float()*active
    weights=weights*batch.get("sample_weight",weights.new_ones(weights.size(0)))[:,None]
    mass=float(weights.sum())
    timing=time_tokens[batch["labels"].clamp_min(0)]&active if time_tokens is not None else active
    return int(active.sum()),mass,float((weights*timing).sum()),float((weights*(active&~timing)).sum())

def _phase_budget(batch,phase_tokens):
    active=batch['labels'].ne(-100)
    weights=batch.get('token_weights',active.float()).float()*active
    weights=weights*batch.get('sample_weight',weights.new_ones(weights.size(0)))[:,None]
    return float((weights*phase_tokens[batch['labels'].clamp_min(0)]).sum())


def _validate(model,loader,device,precision,maximum):
    model.validation_phase_loss=None
    if loader is None:return None
    phase_numerator=torch.zeros((),device=device);phase_mass_sum=0.
    phase_tokens=model.phase_targets.detach().cpu()
    model.eval();numerator=torch.zeros((),device=device);mass_sum=0.
    finite=torch.ones((),device=device,dtype=torch.bool)
    with torch.inference_mode():
        for index,batch in _batches(loader):
            if maximum and index>=maximum:break
            if model.config.condition_policy==RADAR_STYLE:
                # Select checkpoints for the default music-led generation path,
                # not a validation path supplied with the answer chart's statistics.
                known=torch.zeros_like(batch['condition'],dtype=torch.bool)
                known[:,0]=True;batch={**batch,'condition_known':known}
            _,mass,_,_=_cpu_budget(batch)
            phase_mass=_phase_budget(batch,phase_tokens)
            if not math.isfinite(mass) or mass<0:raise ValueError("Invalid validation token weight budget")
            batch=_device_batch(batch,device)
            with _autocast(device,precision):
                loss,metrics=model.loss(model(batch),batch)
            finite=finite&torch.isfinite(loss)
            numerator+=loss.detach()*max(1.,mass);mass_sum+=mass
            phase_numerator+=metrics["phase_loss"]*phase_mass;phase_mass_sum+=phase_mass
    if not bool(finite):raise RuntimeError("Nonfinite validation loss")
    if mass_sum<=0:raise ValueError("Validation has no positive quality-weighted target tokens")
    result=float(numerator)/max(1.,mass_sum)
    model.validation_phase_loss=float(phase_numerator)/phase_mass_sum if phase_mass_sum else None
    model.train()
    return result

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
    for key in ('cuda_memory_budget_gb', 'audio_cache_gb'):
        value = getattr(args, key)
        if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1024:
            raise ValueError(f'{key} 必须在 0 至 1024 GiB 之间')
    if args.cpu_threads<1:raise ValueError("cpu_threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    if min(args.epochs,args.batch_size,args.grad_accumulation,args.log_interval,args.checkpoint_interval_steps)<=0:raise ValueError("epochs/batch/accumulation/log/save intervals must be positive")
    if args.lr<=0 or not 0<=args.min_lr_ratio<=1 or not 0<=args.dropout<1:raise ValueError("Invalid optimizer/dropout settings")
    if min(args.warmup_steps,args.validation_max_batches,args.seed,args.max_steps,args.limit_charts)<0:raise ValueError("Training limits and seed must be nonnegative")
    if args.device not in {"auto","cpu","cuda"} or args.prepare_device not in {"auto","cpu","cuda"}:raise ValueError("Device must be auto/cpu/cuda")
    if args.lr_schedule not in {"cosine","constant"}:raise ValueError("Unknown lr_schedule")
    if args.audio_backend not in {"mert","native"}:raise ValueError("audio_backend must be mert or explicit native")
    directory=Path(args.checkpoint_dir).resolve()
    parent_path=Path(args.resume or args.init_checkpoint).resolve() if args.resume or args.init_checkpoint else None
    source_directory=parent_path.parent if parent_path else None
    artifacts=[directory/name for name in ("composer_v2_last.pt","composer_v2_best.pt","composer_v2_inference.pt")]
    if parent_path in artifacts:
        archive=directory/"resume_sources"/(time.strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:10]+"_"+parent_path.name)
        archive.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(parent_path,archive)
        parent_path=archive
        emit("checkpoint",stage="parent_preserved",path=str(archive))
    if not parent_path and any(path.exists() for path in artifacts):raise ValueError("Fresh output directory already contains Composer v2 weights; choose a new directory")
    parent=read(parent_path) if parent_path else None
    if parent:
        checked=inspect_payload(parent)
        if not checked["valid"] or (args.resume and not checked["can_resume"]):raise ValueError("Parent checkpoint cannot initialize this training mode: "+str(checked.get("error")))
    parent_policy = policy_from_config(parent.get('model_config') or parent.get('config')) if parent else None
    parent_sequence=(parent.get('model_config') or parent.get('config')).get('sequence_policy','window_v1') if parent else None
    args.sequence_policy=getattr(args,'sequence_policy',None) or (parent_sequence if parent else 'continuous_events_v1')
    recovery_policy=getattr(args,'timing_recovery','auto')
    if recovery_policy=='auto':
        if args.resume:
            recovery_policy=parent['training_contract'].get('timing_recovery','off')
        elif args.sequence_policy=='continuous_events_v1':
            recovery_policy=timing_recovery.FRESH_DEFAULT
        else:
            recovery_policy='off'
    if recovery_policy not in ('off',timing_recovery.POLICY):raise ValueError('Unknown timing recovery policy')
    if recovery_policy!='off' and args.sequence_policy!='continuous_events_v1':
        raise ValueError('Timing recovery requires continuous event training')
    args.timing_recovery=recovery_policy
    if parent and args.sequence_policy!=parent_sequence:raise ValueError('Continuous time language requires fresh training in a new directory')
    args.condition_policy, migration = resolve_training_policy(getattr(args, 'condition_policy', None),
        (parent.get('model_config') or parent.get('config')) if parent else None, resume=bool(args.resume))
    from .onset_policy import resolve_policy as resolve_onset_policy
    args.onset_readout_policy, onset_migration = resolve_onset_policy(getattr(args,'onset_readout','auto'),
        (parent.get('model_config') or parent.get('config')) if parent else None, resume=bool(args.resume))
    from .rhythm_structure import resolve_policy as resolve_structure_policy
    args.structure_policy, structure_migration = resolve_structure_policy(getattr(args,'rhythm_structure','auto'),
        (parent.get('model_config') or parent.get('config')) if parent else None, resume=bool(args.resume))
    if args.structure_policy!='off' and args.sequence_policy!='continuous_events_v1':
        raise ValueError('Rhythm structure requires continuous event training')
    if migration or onset_migration or structure_migration:
        if directory == source_directory or any(path.exists() for path in artifacts):
            raise ValueError('Condition/onset/rhythm-structure migration requires a new checkpoint directory; retain the parent')
    device=torch.device("cuda" if args.device=="auto" and torch.cuda.is_available() else "cpu" if args.device=="auto" else args.device)
    if device.type=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA requested but unavailable")
    precision=("bf16" if device.type=="cuda" and torch.cuda.is_bf16_supported() else "fp32") if args.precision=="auto" else args.precision
    if precision not in {"bf16","fp16","fp32"}:raise ValueError("Unknown precision")
    if precision=="bf16" and device.type=="cuda" and not torch.cuda.is_bf16_supported():raise RuntimeError("BF16 requested but unsupported; choose precision explicitly")
    if precision=="fp16" and device.type!="cuda":raise ValueError("FP16 training requires CUDA")
    prepare_device=("cuda" if torch.cuda.is_available() else "cpu") if args.prepare_device=="auto" else args.prepare_device
    cuda_budget = _apply_cuda_budget(args.cuda_memory_budget_gb,
        device.type == 'cuda' or (args.audio_backend == 'mert' and prepare_device == 'cuda'))
    resource_budgets = dict(cuda_memory_budget_gb=cuda_budget, audio_cache_gb=args.audio_cache_gb)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    config=ComposerV2Config.from_dict({key:getattr(args,key) for key in ("d_model","layers","nhead","dim_feedforward","dropout","window_beats","history_beats","future_beats","ticks_per_beat","max_tokens","audio_backend","mert_model","condition_policy","sequence_policy","refresh_events","onset_readout_policy","structure_policy")})
    parent_config = ComposerV2Config.from_dict(parent.get('model_config') or parent.get('config')).to_dict() if parent else None
    comparable_config = dict(config.to_dict())
    if migration: comparable_config['condition_policy'] = parent_policy
    if onset_migration: comparable_config['onset_readout_policy'] = parent_config['onset_readout_policy']
    if structure_migration: comparable_config['structure_policy'] = parent_config['structure_policy']
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
        condition_policy=args.condition_policy,audio_store=audio_store,dialect_policy=dialect_policy)
    dataset_class=ComposerV2Dataset
    if args.sequence_policy=='continuous_events_v1':
        from .stream_data import StreamDataset
        dataset_class=StreamDataset;options['refresh_events']=args.refresh_events
        options.update(timing_recovery_policy=recovery_policy,augmentation_seed=args.seed,
                       onset_readout_policy=args.onset_readout_policy,structure_policy=args.structure_policy)
    train_options=dict(options)
    if args.sequence_policy=='continuous_events_v1':train_options['reference_augmentation']=True
    dataset=dataset_class(args.manifest,extend_vocabulary=bool(args.init_checkpoint),**train_options)
    vocabulary_changed = codec is not None and dataset.codec is not codec
    if vocabulary_changed and (directory == source_directory or any(path.exists() for path in artifacts)):
        raise ValueError('Vocabulary expansion requires a new checkpoint directory; retain the parent')
    parent_vocab_size = codec.vocab_size if codec is not None else None
    codec=dataset.codec
    validation_options={**options,"codec":codec,"limit_charts":0}
    if args.sequence_policy=='continuous_events_v1':validation_options['timing_recovery_policy']='off'
    validation=dataset_class(args.validation_manifest,**validation_options) if args.validation_manifest else None
    identity=dict(train=_identity(dataset),validation=_identity(validation) if validation else None)
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    if args.resume:
        batch_order = parent['training_contract']['sampler']
    else:
        batch_order = AUDIO_LOCAL_BATCH_ORDER if args.sequence_policy == 'continuous_events_v1' else GLOBAL_BATCH_ORDER
    loader,sampler=_loader(dataset,args.batch_size,args.seed,True,device,order=batch_order)
    validation_loader=_loader(validation,args.batch_size,args.seed,False,device)[0] if validation else None
    updates_per_epoch=math.ceil(len(loader)/args.grad_accumulation)
    total_updates=updates_per_epoch*args.epochs
    contract=dict(batch_size=args.batch_size,grad_accumulation=args.grad_accumulation,seed=args.seed,
        lr=args.lr,lr_schedule=args.lr_schedule,warmup_steps=args.warmup_steps,min_lr_ratio=args.min_lr_ratio,
        epochs=args.epochs,total_updates=total_updates,precision=precision,device_type=device.type,
        window_policy="stable_beat_window_components_v2",sampler=batch_order,
        loss_normalization="quality_weighted_token_mass_per_update_v1",cpu_threads=args.cpu_threads)
    if args.condition_policy != LEVEL_ONLY:
        contract['condition_contract'] = condition_contract(args.condition_policy)
    if args.sequence_policy=='continuous_events_v1':
        contract.update(window_policy='causal_event_slices_no_boundary_eos_v1',refresh_events=args.refresh_events,
            acoustic_policy='query_integer_beat_tiles_v1')
    if recovery_policy!='off':contract['timing_recovery']=recovery_policy
    if args.onset_readout_policy != 'off':contract['onset_readout_policy']=args.onset_readout_policy
    if args.structure_policy != 'off':contract['structure_policy']=args.structure_policy
    if args.structure_policy=='aligned_rhythm_v2':
        from .rhythm_learning import POLICY as LEARNING_POLICY
        contract['rhythm_learning']=LEARNING_POLICY
    if dialect_policy != chart_dialect.LEGACY_STRICT_POLICY:
        contract['chart_dialect'] = dialect_policy
    if args.resume and (not _same_data_identity(parent["data_identity"],identity) or parent["training_contract"]!=contract):
        raise ValueError("Exact resume data/optimizer/schedule contract changed; retain the parent and use compatible finetune in a new directory")
    model=ComposerV2(config,codec).to(device)
    if parent:
        parent_state = parent['model_state_dict']
        if vocabulary_changed:
            parent_state = _extend_vocabulary_state(model, parent_state, parent_vocab_size)
        if migration or onset_migration or structure_migration:
            expected_missing = {'condition_missing.weight'} if migration else set()
            if onset_migration:
                if args.onset_readout_policy != 'off':
                    expected_missing.update(k for k in model.state_dict() if k.startswith('onset_readout.'))
                else:
                    parent_state = {k:v for k,v in parent_state.items() if not k.startswith('onset_readout.')}
            if structure_migration:
                from .rhythm_structure import migration_state
                parent_state, missing_structure = migration_state(model,parent_state)
                expected_missing.update(missing_structure)
            loaded = model.load_state_dict(parent_state, strict=False)
            if set(loaded.missing_keys) != expected_missing or loaded.unexpected_keys:
                raise ValueError('Unexpected state keys during explicit migration: ' + str(loaded))
            emit('runtime',stage='model_migration',parent=str(parent_path),
                source_policy=parent_policy,condition_policy=args.condition_policy,
                onset_readout_policy=args.onset_readout_policy,structure_policy=args.structure_policy,
                zero_initialization=True)
        else:
            model.load_state_dict(parent_state,strict=True)
    if hasattr(model,'onset_readout') and (not parent or onset_migration):
        model.onset_readout.fit_normalization(dataset.songs)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    scaler=torch.amp.GradScaler("cuda",enabled=precision=="fp16")
    state=dict(epoch=0,batch_cursor=0,optimizer_steps=0,update_attempts=0,best_loss=None,last_loss=None,epoch_loss_sum=0.,epoch_weight_sum=0.)
    if args.resume:
        optimizer.load_state_dict(parent["optimizer_state_dict"])
        state.update(parent["training_state"])
        if parent.get("scaler_state_dict"):scaler.load_state_dict(parent["scaler_state_dict"])
        restore_rng(parent["rng_state"])
    directory.mkdir(parents=True,exist_ok=True)
    (directory/'dataset_audit.json').write_text(json.dumps(
        dict(train=dataset.audit,validation=validation.audit if validation else None),
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
        else "validation_quality_weighted_token_ce") if validation else "training_quality_weighted_token_ce_no_validation"
    epoch_loss_gpu=torch.tensor(float(state.get("epoch_loss_sum",0.)),device=device)
    epoch_mass=float(state.get("epoch_weight_sum",0.))
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
        trainable_parameters=sum(p.numel() for p in model.parameters()),effective_batch=args.batch_size*args.grad_accumulation,
         train_charts=len(dataset.songs),validation_charts=len(validation.songs) if validation else 0,
         preparation=dataset.audit.get('preparation', {}), batch_order=batch_order, **resource_budgets,
         timing_recovery=dataset.audit.get('timing_recovery'),
         rhythm_structure=dataset.audit.get('rhythm_structure'),
        best_basis=best_basis)
    model.train();optimizer.zero_grad(set_to_none=True)
    began=time.perf_counter();previous=began;data_sec=0.;compute_sec=0.;seen_samples=0;seen_tokens=0
    running=torch.zeros(4,device=device);running_mass=0.;running_time_mass=0.;running_component_mass=0.;running_phase_mass=0.
    group_metrics=torch.zeros(4,device=device);group_finite=torch.ones((),device=device,dtype=torch.bool)
    group_mass=0.;group_time_mass=0.;group_component_mass=0.;group_phase_mass=0.;stop=False
    time_tokens=model.time_tokens.detach().cpu();phase_tokens=model.phase_targets.detach().cpu()
    if device.type=="cuda":torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(int(state["epoch"]),args.epochs):
        sampler.set_epoch(epoch)
        cursor=int(state["batch_cursor"]) if epoch==state["epoch"] else 0
        epoch_loss_gpu=torch.tensor(float(state.get("epoch_loss_sum",0.)),device=device)
        epoch_mass=float(state.get("epoch_weight_sum",0.))
        for batch_index,batch in _batches(loader,cursor):
            now=time.perf_counter();data_sec+=now-previous
            tokens,mass,time_mass,component_mass=_cpu_budget(batch,time_tokens)
            phase_mass=_phase_budget(batch,phase_tokens)
            cpu_valid=math.isfinite(mass) and mass>=0
            # Keep the unnormalized numerator numerically modest for FP16;
            # divide by the complete update's quality-weighted token budget below.
            budget=max(1.,mass) if cpu_valid else 1.
            batch=_device_batch(batch,device)
            with _autocast(device,precision):
                loss,metrics=model.loss(model(batch),batch)
            scaler.scale(loss*(budget/1024.)).backward()
            group_finite=group_finite&torch.isfinite(loss.detach())&cpu_valid
            group_metrics+=torch.stack((loss.detach()*budget,
                metrics["time_loss"]*max(1.,time_mass),metrics["component_loss"]*max(1.,component_mass),metrics["phase_loss"]*phase_mass))
            group_mass+=mass if cpu_valid else 0.
            group_time_mass+=time_mass if cpu_valid else 0.
            group_component_mass+=component_mass if cpu_valid else 0.
            group_phase_mass+=phase_mass if cpu_valid else 0.
            seen_samples+=len(batch["input_ids"]);seen_tokens+=tokens
            boundary=(batch_index+1)%args.grad_accumulation==0 or batch_index+1==len(loader)
            if boundary:
                scaler.unscale_(optimizer)
                for parameter in model.parameters():
                    if parameter.grad is not None:parameter.grad.mul_(1024./max(1.,group_mass))
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
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
                    running+=group_metrics;running_mass+=group_mass
                    running_time_mass+=group_time_mass;running_component_mass+=group_component_mass;running_phase_mass+=group_phase_mass
                else:emit("progress",stage="nonfinite_update_skipped",epoch=epoch,batch=batch_index+1,optimizer_steps=state["optimizer_steps"])
                scaler.update();optimizer.zero_grad(set_to_none=True)
                group_metrics.zero_();group_finite.fill_(True)
                group_mass=0.;group_time_mass=0.;group_component_mass=0.;group_phase_mass=0.
                state.update(epoch=epoch,batch_cursor=batch_index+1,update_attempts=attempt+1)
            ended=time.perf_counter();compute_sec+=ended-now
            if boundary and (state["update_attempts"]%args.log_interval==0 or batch_index+1==len(loader)):
                _sync(device)
                values=running.cpu().tolist();flush_epoch();elapsed=time.perf_counter()-began
                emit("progress",stage="training",epoch=epoch+1,batch=batch_index+1,total=len(loader),
                    optimizer_steps=state["optimizer_steps"],loss=values[0]/max(1.,running_mass),
                    loss_scope="run_quality_weighted_token_mean",
                    token_count=seen_tokens,token_weight_sum=running_mass,
                    tokens_per_sec=seen_tokens/max(elapsed,1e-9),samples_per_sec=seen_samples/max(elapsed,1e-9),
                    time_loss=values[1]/max(1.,running_time_mass),component_loss=values[2]/max(1.,running_component_mass),phase_loss=values[3]/max(1.,running_phase_mass),
                    data_sec=data_sec,compute_sec=compute_sec,compute_measurement="host_submission_wall_including_update_boundary_wait",
                    prefetch={"consumer_wait_sec":data_sec,"max_batches":1},elapsed_sec=elapsed,lr=optimizer.param_groups[0]["lr"],
                    audio_banks=dataset.audio_store.mapping_stats(),
                    gpu_peak_bytes=torch.cuda.max_memory_allocated(device) if device.type=="cuda" else 0)
            if boundary and finite and state["optimizer_steps"]%args.checkpoint_interval_steps==0:checkpoint()
            if boundary and args.max_steps and state["update_attempts"]>=args.max_steps:
                stop=True;break
            previous=time.perf_counter()
        if state["batch_cursor"]>=len(loader):
            validation_started = time.perf_counter()
            validation_loss=_validate(model,validation_loader,device,precision,args.validation_max_batches)
            validation_sec = time.perf_counter() - validation_started if validation_loader is not None else 0.
            flush_epoch()
            score=validation_loss if validation_loss is not None else state["last_loss"]
            improved=score is not None and (state["best_loss"] is None or score<state["best_loss"])
            if improved:state["best_loss"]=score
            emit("progress",stage="epoch_complete",epoch=epoch+1,total=args.epochs,loss=score,
                epoch_train_loss=state["last_loss"],validation_loss=validation_loss,
                validation_phase_loss=getattr(model,"validation_phase_loss",None),
                 validation_sec=validation_sec,
                loss_scope="epoch_validation" if validation_loss is not None else "epoch_training",
                best_basis=best_basis,optimizer_steps=state["optimizer_steps"])
            state.update(epoch=epoch+1,batch_cursor=0,epoch_loss_sum=0.,epoch_weight_sum=0.)
            # Checkpoint's next-epoch accumulator must agree with its zero cursor.
            epoch_loss_gpu.zero_();epoch_mass=0.
            checkpoint(improved)
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
