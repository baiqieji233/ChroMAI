"""Train the complete causal event-set Composer pipeline."""
from __future__ import annotations
import argparse, hashlib, json, math, random, time
from contextlib import contextmanager
from dataclasses import asdict, replace
from fractions import Fraction
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Sampler
from .stream_data import StreamDataset,DeltaVocabulary,collate_stream
from .stream_model import StreamComposer,StreamComposerConfig,to_device_batch
from . import stream_checkpoint as checkpoint
from .training_lock import training_output_lock
from .metric_totals import MetricTotals
from maichart.form.layout_prefetch import LayoutBatchPrefetch

LEARNING_POLICIES = {
    'transition_policy': ('music_exit_v3','music_exit_v2','music_exit_v1','off'),
    'decoder_policy': ('prenorm_independent_v2','postnorm_v1'),
    'window_balance_policy': ('sqrt_window_v2','equal_window_v1'),
    'timing_output_policy': ('absolute_grid_v1','relative_delta_v1'),
    'timing_grid_audio_policy': ('query_pool_v3','query_pool_v2','candidate_pool_v1','off'),
    'timing_group_policy': ('phase_gap_v2','phase_only_v1'),
    'literal_output_policy': ('type_blocks_v1','flat_family_v1'),
    'event_family_policy': ('audio_prior_transition_v3','audio_semantic_v2','semantic_mass_v1','legacy_v1'),
    'prefix_augmentation_input_policy': ('musical_loop_recovery_v7','musical_loop_recovery_v8','musical_loop_recovery_v6','musical_loop_recovery_v5','phase_recovery_v4','causal_window_config_time_first_decision_v3'),
    'history_window_policy': ('block_aligned_v2','rolling_legacy_v1'),
    'continuation_policy': ('pattern_runs_v2','typed_runs_v1','off'),
    'sensor_load_policy': ('separate_touch_v1','button_only_v1'),
    'gap_continuation_policy': ('probability_v2','additive_v1'),
    'motion_continuation_policy': ('context_motion_v3','motion_distinct_v2','motion_continuation_v1','off'),
    'lane_load_policy': ('direct_buttons_v1','off'),
    'recovery_policy': ('greedy_mixed_v4','greedy_mixed_v3','mixed_exit_v2','persistent_v1'),
}

LEGACY_LEARNING_POLICIES = dict(transition_policy='off',decoder_policy='postnorm_v1',window_balance_policy='equal_window_v1',timing_output_policy='relative_delta_v1',timing_grid_audio_policy='off',timing_group_policy='phase_only_v1',literal_output_policy='flat_family_v1',event_family_policy='legacy_v1',
    prefix_augmentation_input_policy='causal_window_config_time_first_decision_v3',
    history_window_policy='rolling_legacy_v1',continuation_policy='off',
    sensor_load_policy='button_only_v1',gap_continuation_policy='additive_v1',
    motion_continuation_policy='motion_continuation_v1',lane_load_policy='off',recovery_policy='persistent_v1')


def resolve_learning_policies(requested, parent_config=None, *, resume=False):
    result={}
    for key,choices in LEARNING_POLICIES.items():
        value=getattr(requested,key,'auto')
        inherited=getattr(parent_config,key) if parent_config is not None else None
        if value=='auto':value=inherited if resume else choices[0]
        if value not in choices:raise ValueError(f'Unsupported {key}')
        if resume and value!=inherited:raise ValueError(f'Resume inherits {key}; use finetune')
        result[key]=value
    weight=getattr(requested,'transition_exit_weight',None)
    inherited=getattr(parent_config,'transition_exit_weight',0.) if parent_config is not None else None
    if weight is None:weight=inherited if resume else .5
    if not math.isfinite(weight) or weight<0:raise ValueError('Transition exit weight must be finite and nonnegative')
    if resume and weight!=inherited:raise ValueError('Resume inherits transition_exit_weight; use finetune')
    rate=getattr(requested,'prefix_augmentation_rate',None)
    inherited=getattr(parent_config,'prefix_augmentation_rate',0.) if parent_config is not None else None
    if rate is None:rate=inherited if resume else .15
    if not math.isfinite(rate) or not 0<=rate<=1:raise ValueError('Prefix augmentation rate must be finite in [0, 1]')
    if resume and rate!=inherited:raise ValueError('Resume inherits prefix_augmentation_rate; use finetune')
    audio_weight=getattr(requested,'event_family_audio_weight',None)
    inherited=getattr(parent_config,'event_family_audio_weight',0.) if parent_config is not None else 0.
    if audio_weight is None:audio_weight=inherited if resume else .25
    if not math.isfinite(audio_weight) or not 0<=audio_weight<=1:
        raise ValueError('Event family audio weight must be in [0,1]')
    if resume and audio_weight!=inherited:raise ValueError('Resume inherits event_family_audio_weight')
    return {**result,'transition_exit_weight':weight,'prefix_augmentation_rate':rate,'event_family_audio_weight':audio_weight}


def upgrade_learning_stage(model, policies, device='cpu'):
    # New output interfaces start at zero; the small musical exit MLP keeps
    # random hidden features. Temporary draws must not shift dropout RNG.
    with torch.random.fork_rng(devices=[]):
        upgraded=StreamComposer(replace(model.config,**policies),model.codec).to(device)
    source=model.state_dict(); target=upgraded.state_dict()
    allowed={'touch_load_projection.weight','touch_load_projection.bias',
             'continuation_projection.weight','continuation_projection.bias',
             'keep_gap_audio.weight','keep_gap_audio.bias','lane_load_projection.weight',
             'action_continuation_projection.weight','kind_continuation_projection.weight',
             'gap_continuation_projection.weight','motion_context_projection.weight'}
    new_prefixes=('timing_grid.','history_gap_projection.','event_family_head.',
                  'literal_route_head.','literal_family_embedding.','family_audio_projection.',
                  'family_history_projection.','family_audio_norm.','audio_event_family_head.',
                  'family_continuity_head.','family_rhythm_projection.','decoder.norm.',
                  'literal_stop_head.','literal_type_embedding.','music_exit_head.')
    allowed.update(key for key in set(source)|set(target) if key.startswith(new_prefixes))
    missing=set(target)-set(source); unexpected=set(source)-set(target)
    # Explicit baseline policies may remove only these optional projection modules.
    if missing-allowed or unexpected-allowed:
        raise ValueError(f'Unsafe stage migration: missing={sorted(missing)}, unexpected={sorted(unexpected)}')
    for key in missing:
        if key=='decoder.norm.weight':
            if not torch.equal(target[key],torch.ones_like(target[key])):raise ValueError('Decoder final normalization must start at unit gain')
            continue
        if not key.startswith(new_prefixes) and torch.count_nonzero(target[key]).item():
            raise ValueError(f'New parameter is not zero initialized: {key}')
    transferred={key:value for key,value in source.items() if key in target}
    if ((model.config.transition_policy=='music_exit_v1' and upgraded.config.transition_policy=='music_exit_v2')
            or (model.config.transition_policy in ('music_exit_v1','music_exit_v2')
                and upgraded.config.transition_policy=='music_exit_v3')):
        # Absolute outputs and the old cursor-note reference are not a new
        # residual. Retain features, but restart its correction at zero.
        for key in ('music_exit_head.2.weight','music_exit_head.2.bias'):transferred.pop(key,None)
    for key,value in list(transferred.items()):
        if value.shape==target[key].shape:continue
        if key=='continuation_projection.weight' and value.ndim==2 and value.shape[1]==8 and target[key].shape==(value.shape[0],16):
            expanded=torch.zeros_like(target[key]);expanded[:,:8]=value
            transferred[key]=expanded
        else:raise ValueError(f'Unsafe stage migration shape: {key}: {tuple(value.shape)} -> {tuple(target[key].shape)}')
    upgraded.load_state_dict(transferred,strict=False)
    return upgraded


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('manifest','checkpoint-dir'):p.add_argument('--'+key,required=True)
    p.add_argument('--cache-dir',default='cache/chart_composer_v1')
    p.add_argument('--validation-manifest');p.add_argument('--resume');p.add_argument('--init-checkpoint')
    for key,default in dict(epochs=20,batch_size=16,grad_accumulation=1,d_model=256,layers=4,nhead=8,
        dim_feedforward=768,history_events=64,rollout_steps=16,checkpoint_interval_steps=500,
        log_interval=25,max_steps=0,limit_charts=0,prepare_workers=1,cpu_threads=2,
        recovery_interval=25,recovery_samples=2,recovery_steps=2,symbolic_pretrain_epochs=0,
        validation_max_batches=32,validation_batch_size=4,generation_eval_songs=2,
        generation_eval_interval_epochs=1,seed=0,warmup_steps=100,prefetch_batches=2,sample_workers=1).items():
        p.add_argument('--'+key.replace('_','-'),type=int,default=default)
    for key,default in dict(lr=2e-4,dropout=.1,window_seconds=8.,recovery_weight=.1,
        plan_weight=.1,audio_cache_gb=2.,min_lr_ratio=.1).items():
        p.add_argument('--'+key.replace('_','-'),type=float,default=default)
    for key in ('d_model','layers','nhead','dim_feedforward','history_events','rollout_steps'):p.set_defaults(**{key:None})
    p.add_argument('--lr-schedule',choices=('constant','cosine'),default='cosine')
    p.add_argument('--generation-eval-seeds',default='0,1')
    p.add_argument('--audio-grid-seconds',type=float,default=None,
                   help='Fresh default 0.01 seconds; continuation inherits its checkpoint grid')
    p.add_argument('--plan-blank-window-seconds',type=float,default=None,
                   help='Fresh default 0.5-second centered blank label; continuation inherits checkpoint')
    for key,choices in LEARNING_POLICIES.items():p.add_argument('--'+key.replace('_','-'),choices=('auto',*choices),default='auto')
    p.add_argument('--transition-exit-weight',type=float,default=None)
    p.add_argument('--prefix-augmentation-rate',type=float,default=None)
    p.add_argument('--event-family-audio-weight',type=float,default=None)
    p.add_argument('--generation-eval-temperature',type=float,default=None)
    p.add_argument('--device',default='auto');p.add_argument('--precision',choices=('auto','bf16','fp16','fp32'),default='bf16')
    return p

def _emit(kind,**values):print(kind+' '+json.dumps(values,ensure_ascii=False),flush=True)

def _dataset(args,*,codec=None,delta_vocab=None,validation=False):
    return StreamDataset(args.validation_manifest if validation else args.manifest,
        cache_dir=args.cache_dir,codec=codec,delta_vocab=delta_vocab,lead_in=args.history_events,
        targets=args.rollout_steps,window_seconds=args.window_seconds,
        preparation_workers=args.prepare_workers,audio_cache_gb=args.audio_cache_gb,
        audio_grid_seconds=args.audio_grid_seconds,
        plan_blank_window_seconds=args.plan_blank_window_seconds,
        timing_output_policy=getattr(args,'timing_output_policy','relative_delta_v1'),
        window_balance_policy=getattr(args,'window_balance_policy','equal_window_v1'),
        history_window_policy=('block_aligned_v2' if getattr(args,'history_window_policy','auto')=='auto' else args.history_window_policy),
        continuation_policy=('pattern_runs_v2' if getattr(args,'continuation_policy','auto')=='auto' else args.continuation_policy),
        limit_charts=args.limit_charts,unknown_policy='mask' if validation else 'error')

class _LocalBatches(Sampler):
    def __init__(self,ds,batch_size,seed,epoch,skip=0):
        self.ds,self.bs,self.seed,self.epoch,self.skip=ds,batch_size,seed,epoch,skip
    def __len__(self):return max(0,math.ceil(len(self.ds)/self.bs)-self.skip)
    def __iter__(self):
        # Shuffle whole songs then time windows: each row is visited once and
        # adjacent rows share an immutable audio bank within the collator.
        g=torch.Generator().manual_seed(self.seed+self.epoch*1000003)
        groups={}
        for i,sample in enumerate(self.ds.samples):
            song=sample[0] if isinstance(sample,(tuple,list)) else sample.song_index
            groups.setdefault(song,[]).append(i)
        keys=list(groups);order=torch.randperm(len(keys),generator=g).tolist()
        indices=[]
        for k in order:
            rows=groups[keys[k]]
            perm=torch.randperm(len(rows),generator=g).tolist()
            indices.extend(rows[i] for i in perm)
        for start in range(self.skip*self.bs,len(indices),self.bs):yield indices[start:start+self.bs]

class _TrainingCollator:
    """Prepare augmented CPU rows in the ordered producer, before one collate."""
    def __init__(self,dataset,args,epoch,skip=0):
        self.dataset=dataset
        self.rate=getattr(args,'prefix_augmentation_rate',0.)
        self.input_policy=getattr(args,'prefix_augmentation_input_policy','causal_window_config_time_first_decision_v3')
        self.seed=args.seed+epoch*1000003
        self.batch_number=skip

    def __call__(self,items):
        self.batch_number+=1
        diagnostics={}
        if self.rate:
            from .stream_augmentation import augment_stream_items
            items,diagnostics=augment_stream_items(self.dataset,items,rate=self.rate,
                seed=self.seed+self.batch_number,input_policy=self.input_policy)
        batch=collate_stream(items)
        batch['_augmentation']=diagnostics
        return batch


def _loader(ds,args,epoch,skip=0,validation=False):
    # DataLoader iterator creation must not consume the model/dropout RNG on
    # resume, where the number of loader constructions differs from a full run.
    generator=torch.Generator().manual_seed(args.seed+epoch*1000003+1729)
    if validation:return DataLoader(ds,batch_size=args.validation_batch_size,collate_fn=collate_stream,num_workers=0,generator=generator)
    return DataLoader(ds,batch_sampler=_LocalBatches(ds,args.batch_size,args.seed,epoch,skip),
        collate_fn=_TrainingCollator(ds,args,epoch,skip),num_workers=0,pin_memory=args.device.startswith('cuda'),generator=generator)

@contextmanager
def _prepared_batches(loader,args):
    if args.prefetch_batches:
        with LayoutBatchPrefetch(loader,depth=args.prefetch_batches,sample_workers=1) as prepared:
            yield prepared
    else:
        yield iter(loader)

def _averages(totals,n):
    return {key:value/max(n,1) for key,value in totals.as_floats().items()}

def _exit_coverage(batch):
    kinds=batch['target_kind']
    mask=batch.get('transition_exit_mask',torch.zeros_like(kinds,dtype=torch.bool)).bool()
    mask=mask & batch.get('transition_exit_labels',torch.zeros_like(mask)).bool()
    mask=mask & batch['loss_mask'].bool() & batch['sample_weight'].gt(0)
    result={'exit_'+name+'_targets':(mask & kinds.eq(kind)).sum().detach()
            for name,kind in (('note',2),('wait',3),('eos',4))}
    features=batch.get('continuation_features')
    if features is not None and features.shape[-1]==16:
        # Report bins only; these thresholds never alter supervision.
        counts=features[...,::2].float().expm1().amax(-1)
        result['exit_short_run_targets']=(mask & counts.ge(2) & counts.lt(8)).sum().detach()
        result['exit_long_run_targets']=(mask & counts.ge(8)).sum().detach()
    return result


def _record_recovery_metrics(window_metrics,weighted_recovery,window_batches):
    # A recovery backward is added once to the averaged accumulation objective.
    # Metrics store sums over microbatches, so replicate its scalar contribution
    # W times before the eventual /W; base_loss remains the authored objective.
    contribution=weighted_recovery.detach()*window_batches
    window_metrics.add({'loss':contribution,'recovery_loss':contribution})

def _cuda_memory(device):
    if not str(device).startswith('cuda'):return {}
    scale=1024**3
    return {'allocated_gib':torch.cuda.memory_allocated(device)/scale,
            'reserved_gib':torch.cuda.memory_reserved(device)/scale,
            'peak_allocated_gib':torch.cuda.max_memory_allocated(device)/scale}

def _validate_resume_cursor(state,batches,accumulation):
    cursor=int(state.get('batch_in_epoch',0))
    if cursor<0 or cursor>batches:
        raise ValueError('Resume batch cursor is outside this epoch')
    if not state.get('epoch_complete') and cursor!=batches and cursor%accumulation:
        raise ValueError('Resume requires a complete optimizer accumulation window')
    return cursor

@contextmanager
def _preserve_rng(device):
    python,numpy=random.getstate(),np.random.get_state()
    devices=[torch.device(device).index if torch.device(device).index is not None else torch.cuda.current_device()] if str(device).startswith('cuda') else []
    try:
        with torch.random.fork_rng(devices=devices):yield
    finally:
        random.setstate(python);np.random.set_state(numpy)

def _fingerprint(args,ds,*,legacy=False,pre_blank=False):
    h=hashlib.sha256(Path(args.manifest).read_bytes())
    h.update(json.dumps(ds.codec.to_dict(),sort_keys=True).encode())
    h.update(json.dumps(ds.delta_vocab.to_dict(),sort_keys=True).encode())
    h.update(str((args.history_events,args.rollout_steps,args.window_seconds)).encode())
    if not legacy:h.update(str(float(args.audio_grid_seconds)).encode())
    if not legacy and not pre_blank:h.update(str(float(getattr(args,'plan_blank_window_seconds',.5))).encode())
    policies=(getattr(args,'history_window_policy','rolling_legacy_v1'),getattr(args,'continuation_policy','off'))
    if policies!=('rolling_legacy_v1','off'):h.update(json.dumps(policies).encode())
    if getattr(args,'prefix_augmentation_rate',0.):
        # Configuration/time reconstruction and native whole-window selection
        # define training examples. Changing that recipe requires a new stage.
        h.update(getattr(args,'prefix_augmentation_input_policy','causal_window_config_time_first_decision_v3').encode())
    timing=getattr(args,'timing_output_policy','relative_delta_v1')
    family=getattr(args,'event_family_policy','legacy_v1')
    if timing!='relative_delta_v1' or family!='legacy_v1':
        h.update(json.dumps((timing,family,getattr(args,'timing_grid_audio_policy','off'),
            getattr(args,'event_family_audio_weight',0.),getattr(ds,'timing_phase_catalog',None))).encode())
    if getattr(args,'window_balance_policy','equal_window_v1')!='equal_window_v1':h.update(args.window_balance_policy.encode())
    if getattr(args,'decoder_policy','postnorm_v1')!='postnorm_v1':h.update(args.decoder_policy.encode())
    if getattr(args,'timing_group_policy','phase_only_v1')!='phase_only_v1':h.update(args.timing_group_policy.encode())
    if getattr(args,'literal_output_policy','flat_family_v1')!='flat_family_v1':h.update(args.literal_output_policy.encode())
    if getattr(args,'transition_policy','off')!='off':h.update(args.transition_policy.encode())
    return h.hexdigest()

def _resolve_audio_grid(requested,parent_config=None):
    inherited=parent_config.audio_grid_seconds if parent_config is not None else None
    value=(inherited if inherited is not None else .01) if requested is None else float(requested)
    if not math.isfinite(value) or not .005<=value<=.05:raise ValueError('Audio grid seconds must be positive and finite in [0.005, 0.05]')
    if inherited is not None and value!=inherited:
        raise ValueError('Continuation inherits checkpoint audio_grid_seconds; use fresh training for a different grid')
    return value

def _condition_defaults(ds):
    from maichart.generation.conditions import GenerationCondition,RADAR_KEYS,STYLE_KEYS
    v=torch.stack(ds.conditions).median(0).values.tolist()
    return GenerationCondition(level=v[0]*15,radar=dict(zip(RADAR_KEYS,v[1:8])),style=dict(zip(STYLE_KEYS,v[8:]))).to_dict()


def _initialize_structured_heads(model,ds,parent_config=None,*,resume=False):
    """Fit finite output vocabularies and priors from the training split only.

    This is one CPU pass over already encoded labels, never audio or another
    model forward. Resume retains both its vocabulary and its learned priors.
    """
    from .stream_data import NOTE
    from .stream_timing import phase_group
    absolute=model.config.timing_output_policy=='absolute_grid_v1'
    semantic=model.config.event_family_policy!='legacy_v1'
    init_timing=absolute and not resume and (parent_config is None or parent_config.timing_output_policy!='absolute_grid_v1'
        or parent_config.timing_group_policy!=model.config.timing_group_policy)
    init_family=semantic and not resume and (parent_config is None or parent_config.event_family_policy!=model.config.event_family_policy)
    init_literal=semantic and model.config.literal_output_policy=='type_blocks_v1' and not resume and (
        parent_config is None or parent_config.literal_output_policy!='type_blocks_v1')
    phases={Fraction(value)%1 for value in model.config.delta_beats} if absolute else set()
    grid_counts=torch.zeros(11,dtype=torch.float64)
    bucket_counts=torch.zeros(22,dtype=torch.float64)
    literal_blocks=torch.zeros(5,dtype=torch.float64);literal_extras=torch.zeros(5,dtype=torch.float64)
    family_counts=torch.zeros(31,dtype=torch.float64);literal_counts=torch.zeros(31,dtype=torch.float64)
    atom_families=model.semantic_atom_families.detach().cpu().tolist() if semantic else None
    for song,stream in enumerate(ds.streams):
        encoded=ds.encoded[song];unknown=ds.unknown_steps[song]
        for position,step in enumerate(stream):
            if step.kind!=NOTE:continue
            phase=step.end%1
            if absolute:phases.add(phase)
            if not (init_timing or init_family or init_literal) or unknown[position]:continue
            mass=float(step.weight)*ds.target_window_weights[song][position]*ds.chart_weights[song]
            if mass<=0:continue
            if init_timing:
                group=phase_group(phase)
                grid_counts[group]+=mass
                bucket_counts[group+11*int(step.end-step.start>=1)]+=mass
            if init_family or init_literal:
                event=encoded[position][1]
                if event is None:continue
                family=0
                for atom in event['atoms']:family|=int(atom_families[atom])
                if not 1<=family<=31:raise ValueError('Encoded event has no semantic family')
                if init_family:
                    family_counts[family-1]+=mass
                    if event['shape']==0:literal_counts[family-1]+=mass
                if init_literal and event['shape']==0:
                    for index in range(5):
                        count=sum(bool(int(atom_families[atom])&(1<<index)) for atom in event['atoms'])
                        if count:literal_blocks[index]+=mass;literal_extras[index]+=(count-1)*mass
    if absolute:
        if resume and not phases.issubset(set(model.timing_grid.phase_catalog)):
            raise ValueError('Resume requires its saved absolute timing catalog')
        if not resume:model.configure_timing_phases(sorted(phases))
        ds.timing_phase_catalog=model.config.timing_phase_catalog
    if init_timing:
        with torch.no_grad():
            prior=(grid_counts+.5)/(grid_counts.sum()+.5*len(grid_counts))
            model.timing_grid.grid_head.bias.copy_(prior.log().to(model.timing_grid.grid_head.bias))
            if hasattr(model.timing_grid,'bucket_head'):
                bucket_prior=(bucket_counts+.5)/(bucket_counts.sum()+.5*len(bucket_counts))
                model.timing_grid.bucket_head.bias.copy_((bucket_prior.log()-prior.log().repeat(2)).to(model.timing_grid.bucket_head.bias))
    if init_family:model.initialize_event_family_prior(family_counts,literal_counts)
    if init_literal:model.initialize_literal_stop_prior(literal_blocks,literal_extras)
    return dict(scope='weighted_authored_training_labels_only',timing_prior_initialized=init_timing,
        semantic_prior_initialized=init_family,timing_phases=len(phases),
        grid_counts=grid_counts.tolist() if init_timing else None,
        bucket_counts=bucket_counts.tolist() if init_timing else None,
        literal_block_counts=literal_blocks.tolist() if init_literal else None,
        literal_extra_counts=literal_extras.tolist() if init_literal else None,
        family_counts=family_counts.tolist() if init_family else None,
        literal_counts=literal_counts.tolist() if init_family else None)

def _validation_timing_support(ds,model):
    """Report unseen dev phases without fitting output vocabulary to dev truth."""
    if model.config.timing_output_policy!='absolute_grid_v1':return
    from .stream_data import NOTE
    phases=set(model.timing_grid.phase_catalog)
    missing=0
    for song,stream in enumerate(ds.streams):
        for index,step in enumerate(stream):
            if step.kind==NOTE and step.end%1 not in phases:
                missing+=1
                if not ds.unknown_steps[song][index]:
                    ds.unknown_steps[song][index]=True
                    ds.oov['masked_steps']+=1
    ds.oov['absolute_phase_oov']=missing
    ds.oov['known_steps']=ds.oov['total_steps']-ds.oov['masked_steps']
    ds.oov['known_fraction']=ds.oov['known_steps']/max(1,ds.oov['total_steps'])


def evaluate(model,ds,args):
    was=model.training;model.eval();totals=MetricTotals();n=0
    try:
        with torch.no_grad():
            for i,b in enumerate(_loader(ds,args,0,validation=True)):
                if args.validation_max_batches and i>=args.validation_max_batches:break
                b=to_device_batch(b,args.device)
                with _autocast(args):out=model(b);_,metrics=model.loss(out,b)
                totals.add(metrics)
                n+=1
    finally:model.train(was)
    return dict(metrics=_averages(totals,n),batches=n,oov=ds.oov,
        condition_scope='reference-derived requested controls',scope='authored_prefix_not_musical_quality')

def _autocast(args):
    return torch.autocast('cuda',dtype=torch.bfloat16 if args.precision=='bf16' else torch.float16,
        enabled=args.device.startswith('cuda') and args.precision in ('bf16','fp16'))

def _train(args):
    from .learning_rate import resolve_plan,learning_rate
    from .stream_recovery import recovery_batch,recovery_state,restore_recovery_state
    from .train import _forward_finite_flags,_model_finite_items,_assert_model_finite
    if args.resume and args.init_checkpoint:raise ValueError('Choose resume or init-checkpoint')
    if min(args.epochs,args.batch_size,args.grad_accumulation)<1:raise ValueError('Invalid positive training controls')
    if not 0<=args.prefetch_batches<=16 or args.sample_workers!=1:
        raise ValueError('Stream prefetch requires depth 0..16 and one shared dataset worker')
    if args.log_interval<1 or args.generation_eval_interval_epochs<1 or args.symbolic_pretrain_epochs<0:
        raise ValueError('Invalid logging/evaluation/stage interval')
    args.device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    if args.precision=='auto':args.precision='bf16' if args.device.startswith('cuda') else 'fp32'
    if args.precision=='fp16':raise ValueError('Use BF16 or FP32 for the event-set trainer')
    torch.set_num_threads(args.cpu_threads);random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    directory=Path(args.checkpoint_dir);directory.mkdir(parents=True,exist_ok=True)
    if not args.resume and (directory/'composer_last.pt').exists():raise ValueError('Use a new directory for fresh/finetune')
    old=None
    if args.resume or args.init_checkpoint:model,old=checkpoint.load(args.resume or args.init_checkpoint,args.device,include_optimizer=bool(args.resume))
    for key,default in dict(d_model=256,layers=4,nhead=8,dim_feedforward=768,history_events=64,rollout_steps=16).items():
        if getattr(args,key) is None:setattr(args,key,getattr(model.config,key) if old else default)
    if args.rollout_steps<1 or args.history_events<1:raise ValueError('Invalid history/rollout controls')
    learning_policies=resolve_learning_policies(args,model.config if old else None,resume=bool(args.resume))
    for key,value in learning_policies.items():setattr(args,key,value)
    inherited_temperature=((old or {}).get('args') or {}).get('generation_eval_temperature',.65)
    if args.generation_eval_temperature is None:args.generation_eval_temperature=inherited_temperature if args.resume else .65
    if not math.isfinite(args.generation_eval_temperature) or args.generation_eval_temperature<0:
        raise ValueError('Generation evaluation temperature must be finite and nonnegative')
    if args.resume and args.generation_eval_temperature!=inherited_temperature:
        raise ValueError('Resume inherits generation_eval_temperature; use finetune')
    if args.init_checkpoint:model=upgrade_learning_stage(model,learning_policies,args.device)
    args.audio_grid_seconds=_resolve_audio_grid(args.audio_grid_seconds,model.config if old else None)
    inherited_blank=model.config.plan_blank_window_seconds if old else .5
    if args.plan_blank_window_seconds is not None and (not math.isfinite(args.plan_blank_window_seconds) or args.plan_blank_window_seconds<0):
        raise ValueError('Plan blank window must be finite and nonnegative')
    if old and args.plan_blank_window_seconds is not None and args.plan_blank_window_seconds!=inherited_blank:
        raise ValueError('Continuation inherits checkpoint plan_blank_window_seconds')
    args.plan_blank_window_seconds=inherited_blank if args.plan_blank_window_seconds is None else args.plan_blank_window_seconds
    ds=_dataset(args,codec=model.codec if old else None,
        delta_vocab=DeltaVocabulary.from_dict(old['delta_vocab']) if old else None)
    if not len(ds):raise ValueError('No supervised stream samples')
    if old is None:
        config=StreamComposerConfig(**learning_policies,d_model=args.d_model,nhead=args.nhead,layers=args.layers,
            dim_feedforward=args.dim_feedforward,dropout=args.dropout,history_events=args.history_events,
            rollout_steps=args.rollout_steps,delta_beats=('0',*map(str,ds.delta_vocab.values)),
            audio_grid_seconds=args.audio_grid_seconds,
            plan_blank_window_seconds=args.plan_blank_window_seconds,
            max_event_atoms=max(33,max(a for a,d in ds.widths)),plan_weight=args.plan_weight,recovery_weight=args.recovery_weight)
        model=StreamComposer(config,ds.codec).to(args.device)
    else:
        for name in ('history_events','rollout_steps','d_model','layers','nhead','dim_feedforward'):
            if getattr(args,name)!=getattr(model.config,name):raise ValueError(f'Continuation requires saved {name}')
        if args.init_checkpoint:
            model.config.plan_weight=args.plan_weight;model.config.recovery_weight=args.recovery_weight
    output_initialization=_initialize_structured_heads(model,ds,
        StreamComposerConfig.from_dict(old['model_config']) if old else None,resume=bool(args.resume))
    args.training_objective_recipe=dict(version='stream_musical_transition_residual_budget_v5',
        transition_policy=model.config.transition_policy,
        decoder_policy=model.config.decoder_policy,window_balance_policy=model.config.window_balance_policy,
        timing_output_policy=model.config.timing_output_policy,
        timing_grid_audio_policy=model.config.timing_grid_audio_policy,
        timing_query_interaction=('elementwise_query_candidate_32d'
            if model.config.timing_grid_audio_policy=='query_pool_v3' else 'scalar_query_candidate'
            if model.config.timing_grid_audio_policy=='query_pool_v2' else 'candidate_audio_mean_or_off'),
        timing_query_residual_gain=(8. if model.config.timing_grid_audio_policy=='query_pool_v3' else 1.
            if model.config.timing_grid_audio_policy=='query_pool_v2' else 0.),
        timing_query_compatibility=('retain_v2_scalar_projection_and_add_zero_vector_readout'
            if model.config.timing_grid_audio_policy=='query_pool_v3' else 'policy_defined'),
        timing_group_policy=model.config.timing_group_policy,literal_output_policy=model.config.literal_output_policy,
        event_family_policy=model.config.event_family_policy,
        family_audio_budget=(model.config.event_family_audio_weight if model.config.event_family_policy in ('audio_semantic_v2','audio_prior_transition_v3') else 0.),
        family_full_budget=(1-model.config.event_family_audio_weight if model.config.event_family_policy in ('audio_semantic_v2','audio_prior_transition_v3') else 1.),
        family_mixture_gradient=('audio_prior_detached_when_audio_budget_positive' if model.config.event_family_policy=='audio_prior_transition_v3' else 'joint'),
        conditional_geometry=('joint_phase_detail_with_learned_motion_exit'
            if model.config.transition_policy in ('music_exit_v1','music_exit_v2','music_exit_v3') else 'unchanged_joint_phase_detail'),
        transition_exit='same_budget_as_main_configuration',
        transition_factorization=('parent_keep_exit_log_odds_plus_musical_residual'
            if model.config.transition_policy in ('music_exit_v2','music_exit_v3') else
            'learned_keep_exit_time_motion_template_same_nll_budget'
            if model.config.transition_policy=='music_exit_v1' else 'legacy_joint_reweighting'),
        transition_residual_gain=(8. if model.config.transition_policy in ('music_exit_v2','music_exit_v3') else 1.
            if model.config.transition_policy=='music_exit_v1' else 0.),
        time_music_reference=('strict_prior_note_at_cursor'
            if model.config.transition_policy=='music_exit_v3' else 'checkpoint_policy_defined'),
        history_recovery=model.config.recovery_policy,
        recovery_long_sources=('alternating_persistent_8_steps_and_constructed_authored_exit'
            if model.config.recovery_policy=='greedy_mixed_v4' else 'policy_defined'),
        counterfactual_span=('variable_event_count_loops_32_64_128_with_first_trusted_authored_target'
            if args.prefix_augmentation_input_policy in ('musical_loop_recovery_v7','musical_loop_recovery_v8') else 'all_loop_modes_up_to_history_events'
            if args.prefix_augmentation_input_policy in ('musical_loop_recovery_v5','musical_loop_recovery_v6')
            else 'short_legacy_loops_with_policy_defined_phase_edits'),
        counterfactual_budget=('preserve_source_row_total_sample_weight'
            if args.prefix_augmentation_input_policy in ('musical_loop_recovery_v7','musical_loop_recovery_v8') else 'policy_defined'),
        counterfactual_selection=('authored_exits3to1positivecontrols_within_existing_rate'
            if args.prefix_augmentation_input_policy=='musical_loop_recovery_v8' else 'within_existing_prefix_augmentation_rate'
            if args.prefix_augmentation_input_policy=='musical_loop_recovery_v7' else 'policy_defined'),
        time_target='exact_authored_onset',prefix_input=args.prefix_augmentation_input_policy)
    dev=_dataset(args,codec=ds.codec,delta_vocab=ds.delta_vocab,validation=True) if args.validation_manifest else None
    if dev is not None:_validation_timing_support(dev,model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01,
        **({'fused':True} if args.device.startswith('cuda') else {}))
    state=(old or {}).get('training_state') or {};fingerprint=_fingerprint(args,ds)
    steps=0;start_epoch=1;skip=0
    if args.resume:
        if not old.get('optimizer_state_dict') or not state:
            raise ValueError('Resume requires a complete optimizer/RNG/cursor checkpoint')
        if state.get('dataset_fingerprint')!=fingerprint:
            legacy_grid=('audio_grid_seconds' not in old['model_config'] and args.audio_grid_seconds==.05
                         and state.get('dataset_fingerprint')==_fingerprint(args,ds,legacy=True))
            pre_blank=('plan_blank_window_seconds' not in old['model_config'] and args.plan_blank_window_seconds==0.
                       and state.get('dataset_fingerprint')==_fingerprint(args,ds,pre_blank=True))
            if not (legacy_grid or pre_blank):raise ValueError('Resume data/codec contract changed')
        optimizer.load_state_dict(old['optimizer_state_dict']);steps=int(old['optimizer_steps'])
        start_epoch=int(old['epoch'])+int(state.get('epoch_complete',False));skip=0 if state.get('epoch_complete') else int(state['batch_in_epoch'])
        if int(old['args']['batch_size'])!=args.batch_size or int(old['args']['grad_accumulation'])!=args.grad_accumulation:
            raise ValueError('Resume keeps batch size and accumulation')
        for name in ('seed','symbolic_pretrain_epochs','max_steps','recovery_interval','recovery_samples','recovery_steps','recovery_weight','plan_weight'):
            if old['args'].get(name,getattr(args,name))!=getattr(args,name):
                raise ValueError(f'Resume inherits {name}; use finetune for objective changes')
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng'])
        if args.device.startswith('cuda') and state.get('cuda_rng') is not None:torch.cuda.set_rng_state(state['cuda_rng'],device=args.device)
        restore_recovery_state(model,state.get('recovery_episodes'))
    batches_per_epoch=math.ceil(len(ds)/args.batch_size)
    if args.max_steps:batches_per_epoch=min(batches_per_epoch,args.max_steps)
    if args.resume:
        _validate_resume_cursor(state,batches_per_epoch,args.grad_accumulation)
        if not state.get('epoch_complete') and skip==batches_per_epoch:
            start_epoch+=1;skip=0
    warmup=(state['lr_schedule']['warmup_steps'] if args.resume and state.get('lr_schedule') else
            min(args.warmup_steps,max(0,math.ceil(batches_per_epoch/args.grad_accumulation)*args.epochs-1)))
    lr_plan=resolve_plan(requested=args.lr_schedule,warmup_steps=warmup,
        min_lr_ratio=args.min_lr_ratio,base_lr=args.lr,total_steps=math.ceil(batches_per_epoch/args.grad_accumulation)*args.epochs,
        resume=bool(args.resume),saved=state.get('lr_schedule'))
    defaults=(old or {}).get('condition_defaults') or _condition_defaults(ds)
    _emit('composer_runtime',family=checkpoint.FAMILY,architecture_profile=checkpoint.PROFILE,
        training_semantics='causal_event_set_stream_v1',samples=len(ds),batch_size=args.batch_size,
        grad_accumulation=args.grad_accumulation,history_events=args.history_events,rollout_steps=args.rollout_steps,
        parameters=sum(p.numel() for p in model.parameters()),codec_coverage=ds.codec.coverage,
        effective_lr=optimizer.param_groups[0]['lr'],start_epoch=start_epoch,end_epoch=args.epochs,
        resume_sample_cursor=skip*args.batch_size,precision=args.precision,device=args.device,
        epoch_target=args.epochs,learning_rate_schedule=lr_plan,audio_grid_seconds=args.audio_grid_seconds,
        plan_target_units=model.config.plan_target_units,plan_loss_normalization=model.config.plan_loss_normalization,
        plan_blank_window_seconds=model.config.plan_blank_window_seconds,
        clock_policy=model.config.clock_policy,timing_audio_policy=model.config.timing_audio_policy,
        detail_policy=model.config.detail_policy,motion_continuation_policy=model.config.motion_continuation_policy,
        attention_policy=model.config.attention_policy,learning_policies=learning_policies,
        generation_eval_temperature=args.generation_eval_temperature,
        prefix_augmentation_input_policy=args.prefix_augmentation_input_policy,
        output_initialization=output_initialization,
        training_objective_recipe=args.training_objective_recipe,
        parent_optimizer_steps=int((old or {}).get('optimizer_steps',0)) if args.init_checkpoint else None)
    if args.device.startswith('cuda'):torch.cuda.reset_peak_memory_stats(args.device)
    finite_items=_model_finite_items(model)
    _assert_model_finite(model,finite_items)
    def persist(epoch,batch,complete,metrics):
        s=dict(epoch_complete=complete,batch_in_epoch=batch,dataset_fingerprint=fingerprint,
            python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(device=args.device) if args.device.startswith('cuda') else None,
            lr_schedule=lr_plan,recovery_episodes=recovery_state(model),learning_policies=learning_policies,
            parent_stage={'checkpoint':str(args.init_checkpoint),'optimizer_steps':int(old.get('optimizer_steps',0)),
                          'learning_policies':{key:getattr(StreamComposerConfig.from_dict(old['model_config']),key) for key in learning_policies}} if args.init_checkpoint else state.get('parent_stage'))
        kwargs=dict(epoch=epoch,optimizer_steps=steps,args=vars(args),metrics=metrics,
            delta_vocab=ds.delta_vocab,condition_defaults=defaults)
        checkpoint.save(directory/'composer_last.pt',model,optimizer,state=s,**kwargs)
        checkpoint.save(directory/'composer_inference.pt',model,**kwargs)
        _emit('composer_checkpoint',epoch=epoch,optimizer_steps=steps,path=str(directory/'composer_last.pt'))
    for epoch in range(start_epoch,args.epochs+1):
        model.train();totals=MetricTotals();n=0;supervised=0;sequences=0;started=time.perf_counter();timing=dict(data_sec=0.,transfer_sec=0.,compute_sec=0.)
        optimizer.zero_grad(set_to_none=True)
        symbolic=epoch<=args.symbolic_pretrain_epochs
        # The prior epoch's producer is already joined before changing its view.
        ds.set_symbolic_only(symbolic)
        _emit('composer_stage',epoch=epoch,stage='symbolic' if symbolic else 'audio_conditioned',
              audio_conditioned=not symbolic,symbolic_pretrain_epochs=args.symbolic_pretrain_epochs)
        window_metrics=MetricTotals();window_batches=0
        window_finite=torch.ones(3,dtype=torch.bool,device=args.device)
        skipped_windows=0;last_norm=None;augmentation_totals={}
        with _prepared_batches(_loader(ds,args,epoch,skip),args) as loader:
            for bi in range(skip+1,batches_per_epoch+1):
                tick=time.perf_counter();batch=next(loader)
                for key,value in batch.get('_augmentation',{}).items():
                    augmentation_totals[key]=augmentation_totals.get(key,0)+value
                timing['data_sec']+=time.perf_counter()-tick
                supervised+=int((batch['loss_mask'] & batch['sample_weight'].gt(0)).sum())
                sequences+=batch['history_kind'].size(0)
                tick=time.perf_counter();b=to_device_batch(batch,args.device,non_blocking=args.device.startswith('cuda'))
                timing['transfer_sec']+=time.perf_counter()-tick;tick=time.perf_counter()
                window_start=((bi-1)//args.grad_accumulation)*args.grad_accumulation
                divisor=min(args.grad_accumulation,batches_per_epoch-window_start)
                with _autocast(args):out=model(b,symbolic_only=symbolic);loss,metrics=model.loss(out,b)
                backward_loss=loss/divisor
                window_finite=window_finite & _forward_finite_flags(out,loss,backward_loss)
                backward_loss.backward()
                window_metrics.add(metrics);window_metrics.add(_exit_coverage(b));window_batches+=1
                update=bi%args.grad_accumulation==0 or bi==batches_per_epoch
                updated=False
                if update:
                    episode_backup=None
                    if not symbolic and args.recovery_interval and (steps+1)%args.recovery_interval==0 and args.recovery_weight:
                        episode_backup=recovery_state(model)
                        recovered,diagnostics=recovery_batch(ds,model,args.device,steps,samples=args.recovery_samples,steps=args.recovery_steps)
                        if recovered is not None:
                            # A recovery row has one target, not K authored ones.
                            recovered['_loss_normalizer']=recovered['target_kind'].size(0)
                            recovered=to_device_batch(recovered,args.device,non_blocking=args.device.startswith('cuda'))
                            with _autocast(args):rout=model(recovered);rloss,_=model.loss(rout,recovered)
                            weighted_recovery=args.recovery_weight*rloss
                            window_finite=window_finite & _forward_finite_flags(rout,rloss,weighted_recovery)
                            weighted_recovery.backward()
                            _record_recovery_metrics(window_metrics,weighted_recovery,window_batches)
                        _emit('composer_recovery',optimizer_step=steps+1,outcomes=diagnostics)
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=False)
                    # One host transfer checks outputs, both losses and the
                    # entire accumulated gradient before optimizer mutation.
                    health=torch.cat((window_finite,torch.isfinite(norm.detach()).reshape(1))).cpu().tolist()
                    if all(health):
                        for group in optimizer.param_groups:group['lr']=learning_rate(lr_plan,steps)
                        optimizer.step();steps+=1;updated=True
                        _assert_model_finite(model,finite_items)
                        totals.add(window_metrics);n+=window_batches;last_norm=norm.detach()
                    else:
                        if episode_backup is not None:restore_recovery_state(model,episode_backup)
                        skipped_windows+=1
                        _emit('composer_warning',epoch=epoch,batch=bi,message='non-finite accumulation window skipped',
                              finite=dict(zip(('outputs','loss','backward_loss','grad_norm'),health)))
                    optimizer.zero_grad(set_to_none=True)
                    window_finite=torch.ones(3,dtype=torch.bool,device=args.device)
                    window_metrics=MetricTotals();window_batches=0
                timing['compute_sec']+=time.perf_counter()-tick
                checkpoint_due=updated and args.checkpoint_interval_steps and steps%args.checkpoint_interval_steps==0
                report_due=bi==skip+1 or bi%args.log_interval==0 or bi==batches_per_epoch
                if checkpoint_due or report_due:
                    avg=_averages(totals,n)
                if checkpoint_due:persist(epoch,bi,bi==batches_per_epoch,avg)
                if report_due:
                    elapsed=time.perf_counter()-started
                    _emit('composer_progress',epoch=epoch,batch=bi,total=batches_per_epoch,optimizer_steps=steps,
                        elapsed_sec=elapsed,samples_per_sec=sequences/max(elapsed,1e-6),
                        supervised_steps_per_sec=supervised/max(elapsed,1e-6),
                        supervised_steps=supervised,metric_batches=n,timing=timing,skipped_windows=skipped_windows,
                        prefetch=loader.stats() if hasattr(loader,'stats') else {'mode':'off'},
                        cache=ds.cache_stats() if hasattr(ds,'cache_stats') else {},
                        exit_coverage={key:value for key,value in totals.as_floats().items() if key.startswith('exit_')},
                        augmentation=augmentation_totals,
                        cuda_memory=_cuda_memory(args.device),
                        last_gradient_norm=float(last_norm) if last_norm is not None else None,
                        learning_rate=optimizer.param_groups[0]['lr'],stage='symbolic' if symbolic else 'audio_conditioned',**avg)
        avg=_averages(totals,n)
        skip=0
        if dev is not None:
            validation=evaluate(model,dev,args);validation.update(epoch=epoch,optimizer_steps=steps)
            (directory/'validation_latest.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding='utf-8')
            _emit('composer_validation',**validation)
        persist(epoch,batches_per_epoch,True,avg)
        _emit('composer_epoch',epoch=epoch,optimizer_steps=steps,elapsed_sec=time.perf_counter()-started,
            exit_coverage={key:value for key,value in totals.as_floats().items() if key.startswith('exit_')},
            augmentation=augmentation_totals,**avg)
        if dev is not None and args.generation_eval_songs and epoch%args.generation_eval_interval_epochs==0:
            from .stream_generate import evaluate_panel
            with _preserve_rng(args.device):
                panel=evaluate_panel(directory/'composer_inference.pt',dev,directory/f'generation_eval_epoch_{epoch}',
                    count=args.generation_eval_songs,seeds=[int(s) for s in args.generation_eval_seeds.split(',')],device=args.device,
                    temperature=args.generation_eval_temperature)
            _emit('composer_generation_validation',**panel)
    return 0

def main(argv=None):
    args=parser().parse_args(argv)
    with training_output_lock(args.checkpoint_dir):return _train(args)

if __name__=='__main__':raise SystemExit(main())
