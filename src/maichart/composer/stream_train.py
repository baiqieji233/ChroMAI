"""Train the complete causal event-set Composer pipeline."""
from __future__ import annotations
import argparse, hashlib, json, math, random, time
from contextlib import contextmanager
from dataclasses import asdict, replace
from fractions import Fraction
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Sampler,Subset,Dataset
from .stream_data import StreamDataset,DeltaVocabulary,collate_stream
from .stream_model import StreamComposer,StreamComposerConfig,to_device_batch
from . import stream_checkpoint as checkpoint
from .training_lock import training_output_lock
from .metric_totals import MetricTotals
from maichart.form.layout_prefetch import LayoutBatchPrefetch

LEARNING_POLICIES = {
    'song_profile_policy': ('masked_deployment_joint_v3','masked_ordered_joint_v2','masked_joint_v1','off'),
    'song_response_policy': ('unified_profile_response_v2','music_response_v1','off'),
    'training_window_policy': ('phrase_complete_v1','block_v1'),
    'phrase_execution_policy': ('component_modes_v2','persistent_modes_v1','off'),
    'detail_representation_policy': ('typed_components_v2','shared_components_v1','private_v1'),
    'duration_feature_policy': ('log_time_v2','legacy_v1'),
    'phrase_context_policy': ('shared_ordered_v4','aligned_phrase_v3','ordered_phrase_v2','causal_phrase_v1','off'),
    'template_policy': ('family_preserving_v1','cross_family_v1'),
    'candidate_state_policy': ('selected_native_v1','cursor_v1'),
    'duration_audio_policy': ('interval_v1','off'),
    'physical_policy': ('runtime_hard_v2','legacy_report_v1'),
    'transition_policy': ('off','music_exit_v3','music_exit_v2','music_exit_v1'),
    'decoder_policy': ('prenorm_independent_v2','postnorm_v1'),
    'window_balance_policy': ('sqrt_window_v2','equal_window_v1'),
    'timing_output_policy': ('absolute_grid_v1','relative_delta_v1'),
    'timing_grid_audio_policy': ('query_pool_v3','query_pool_v2','candidate_pool_v1','off'),
    'timing_group_policy': ('phase_gap_v2','phase_only_v1'),
    'literal_output_policy': ('type_blocks_v1','flat_family_v1'),
    'event_family_policy': ('shared_continuation_v1','audio_state_phrase_v4','audio_prior_transition_v3','audio_semantic_v2','semantic_mass_v1','legacy_v1'),
    'recovery_target_policy': ('causal_correspondence_v1','authored_next_v1'),
    'prefix_augmentation_input_policy': ('musical_loop_recovery_v7','musical_loop_recovery_v8','musical_loop_recovery_v6','musical_loop_recovery_v5','phase_recovery_v4','causal_window_config_time_first_decision_v3'),
    'history_window_policy': ('block_aligned_v2','rolling_legacy_v1'),
    'continuation_policy': ('pattern_runs_v2','typed_runs_v1','off'),
    'sensor_load_policy': ('separate_touch_v1','button_only_v1'),
    'gap_continuation_policy': ('off','probability_v2','additive_v1'),
    'motion_continuation_policy': ('context_motion_v3','motion_distinct_v2','motion_continuation_v1','off'),
    'lane_load_policy': ('direct_buttons_v1','off'),
    'recovery_policy': ('greedy_mixed_v4','greedy_mixed_v3','mixed_exit_v2','persistent_v1'),
}

LEGACY_LEARNING_POLICIES = dict(phrase_context_policy='off',template_policy='cross_family_v1',candidate_state_policy='cursor_v1',duration_audio_policy='off',physical_policy='legacy_report_v1',transition_policy='off',decoder_policy='postnorm_v1',window_balance_policy='equal_window_v1',timing_output_policy='relative_delta_v1',timing_grid_audio_policy='off',timing_group_policy='phase_only_v1',literal_output_policy='flat_family_v1',event_family_policy='legacy_v1',
    prefix_augmentation_input_policy='causal_window_config_time_first_decision_v3',
    history_window_policy='rolling_legacy_v1',continuation_policy='off',
    sensor_load_policy='button_only_v1',gap_continuation_policy='additive_v1',
    motion_continuation_policy='motion_continuation_v1',lane_load_policy='off',recovery_policy='persistent_v1',
    recovery_target_policy='authored_next_v1',training_window_policy='block_v1',phrase_execution_policy='off',
    detail_representation_policy='private_v1',duration_feature_policy='legacy_v1',song_profile_policy='off',song_response_policy='off')


def resolve_learning_policies(requested, parent_config=None, *, resume=False):
    result={}
    for key,choices in LEARNING_POLICIES.items():
        value=getattr(requested,key,'auto')
        inherited=getattr(parent_config,key,LEGACY_LEARNING_POLICIES.get(key)) if parent_config is not None else None
        inherit_auto=resume
        if value=='auto':value=inherited if inherit_auto else choices[0]
        if value not in choices:raise ValueError(f'Unsupported {key}')
        if resume and value!=inherited:raise ValueError(f'Resume inherits {key}; use finetune')
        result[key]=value
    if (not resume and getattr(requested,'song_response_policy','auto')=='auto'
            and (result['phrase_execution_policy'] not in ('persistent_modes_v1','component_modes_v2') or result['phrase_context_policy'] not in ('aligned_phrase_v3','shared_ordered_v4'))):
        result['song_response_policy']='off'
    weight=getattr(requested,'transition_exit_weight',None)
    inherited=getattr(parent_config,'transition_exit_weight',0.) if parent_config is not None else None
    if weight is None:weight=inherited if resume else 0.
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
    phrase_weight=getattr(requested,'phrase_loss_weight',None)
    saved_phrase_weight=getattr(parent_config,'phrase_loss_weight',.05) if parent_config is not None else .05
    if phrase_weight is None:phrase_weight=saved_phrase_weight if resume else .05
    if not math.isfinite(phrase_weight) or phrase_weight<0:raise ValueError('Phrase loss weight must be finite and nonnegative')
    if resume and phrase_weight!=saved_phrase_weight:raise ValueError('Resume inherits phrase_loss_weight')
    for key,default in dict(max_phrase_events=64,phrase_seam_events=2,phrase_modes=4,phrase_relation_weight=.1).items():
        inherited=getattr(parent_config,key,default) if parent_config is not None else default
        value=getattr(requested,key,None)
        if value is None:value=inherited
        if not math.isfinite(float(value)) or value<0:raise ValueError(f'Invalid {key}')
        if key!='phrase_relation_weight' and (int(value)!=value or (key=='max_phrase_events' and value<1)):
            raise ValueError(f'Invalid {key}')
        if key=='phrase_modes' and value not in (2,4,8):raise ValueError('phrase_modes must be 2, 4 or 8')
        if key=='phrase_seam_events' and value>16:raise ValueError('phrase_seam_events must be between 0 and 16')
        if resume and value!=inherited:raise ValueError(f'Resume inherits {key}; use finetune')
        result[key]=value
    for key,default in dict(song_profile_components=4,song_profile_weight=.1,song_response_modes=4,song_response_group_size=4).items():
        inherited=getattr(parent_config,key,default) if parent_config is not None else default
        value=getattr(requested,key,None)
        if value is None:value=inherited if resume else default
        if not math.isfinite(float(value)) or value<0 or (key!='song_profile_weight' and (int(value)!=value or value<1)):
            raise ValueError(f'Invalid {key}')
        if resume and value!=inherited:raise ValueError(f'Resume inherits {key}; use finetune')
        result[key]=value
    contexts=('aligned_phrase_v3','shared_ordered_v4')
    executions=('persistent_modes_v1','component_modes_v2')
    if result['song_response_policy']!='off' and (result['phrase_execution_policy'] not in executions or result['phrase_context_policy'] not in contexts):
        raise ValueError('music_response_v1 requires persistent_modes_v1/component_modes_v2 and aligned_phrase_v3/shared_ordered_v4')
    if result['phrase_execution_policy'] in executions:
        if result['phrase_context_policy'] not in contexts or result['event_family_policy']!='shared_continuation_v1':
            raise ValueError('Phrase execution requires aligned_phrase_v3/shared_ordered_v4 and shared_continuation_v1')
        if result['training_window_policy']!='phrase_complete_v1':
            raise ValueError('Phrase execution requires phrase_complete_v1 training windows')
    if result['phrase_execution_policy']=='component_modes_v2' and (result['detail_representation_policy']!='typed_components_v2' or result['literal_output_policy']!='type_blocks_v1' or result['timing_output_policy']!='absolute_grid_v1'):
        raise ValueError('component_modes_v2 requires typed_components_v2, type_blocks_v1 and absolute_grid_v1')
    if result['song_response_policy']=='unified_profile_response_v2':
        if result['song_profile_policy']!='masked_deployment_joint_v3':
            raise ValueError('Unified response requires masked_deployment_joint_v3')
        if result['song_profile_components']!=result['song_response_modes']:
            raise ValueError('Unified profile and response require the same latent count')
    return {**result,'transition_exit_weight':weight,'prefix_augmentation_rate':rate,'event_family_audio_weight':audio_weight,
            'phrase_loss_weight':phrase_weight}


def upgrade_learning_stage(model, policies, device='cpu'):
    policies={**policies,'detail_mode_parameterization':policies.get('detail_mode_parameterization','centered_v2')}
    # New output interfaces start at zero; the small musical exit MLP keeps
    # random hidden features. Temporary draws must not shift dropout RNG.
    with torch.random.fork_rng(devices=[]):
        upgraded=StreamComposer(replace(model.config,**policies),model.codec).to(device)
    source=model.state_dict(); target=upgraded.state_dict()
    allowed={'touch_load_projection.weight','touch_load_projection.bias',
             'continuation_projection.weight','continuation_projection.bias',
             'keep_gap_audio.weight','keep_gap_audio.bias','lane_load_projection.weight',
             'action_continuation_projection.weight','kind_continuation_projection.weight',
             'gap_continuation_projection.weight','motion_context_projection.weight','candidate_state_projection.weight'}
    new_prefixes=('timing_grid.','history_gap_projection.','event_family_head.',
                  'literal_route_head.','literal_family_embedding.','family_audio_projection.',
                  'family_history_projection.','family_audio_norm.','audio_event_family_head.',
                  'family_continuity_head.','family_rhythm_projection.','family_state_phrase.','decoder.norm.',
                  'literal_stop_head.','literal_type_embedding.','music_exit_head.','phrase_context.','duration_audio.',
                  'shared_details.','phrase_execution.','song_profile.','song_response.',
                  'detail_mode_control.','literal_component_projection.')
    if model.config.detail_representation_policy!=upgraded.config.detail_representation_policy:
        allowed.update(key for key in set(source)|set(target) if key.startswith('detail_embeddings.'))
    allowed.update(key for key in set(source)|set(target) if key.startswith(new_prefixes))
    missing=set(target)-set(source); unexpected=set(source)-set(target)
    # Explicit baseline policies may remove only these optional projection modules.
    if missing-allowed or unexpected-allowed:
        raise ValueError(f'Unsafe stage migration: missing={sorted(missing)}, unexpected={sorted(unexpected)}')
    for key in missing:
        if key in ('detail_mode_control.choices','detail_mode_control.stop','literal_component_projection.weight'):
            if torch.count_nonzero(target[key]).item():raise ValueError(f'New component interface must be zero initialized: {key}')
        if key.startswith('detail_embeddings.') and model.config.detail_representation_policy!=upgraded.config.detail_representation_policy:
            continue
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
        elif (key.startswith('shared_details.component_embeddings.') and key.endswith('.weight')
              and model.config.detail_representation_policy=='shared_components_v1'
              and upgraded.config.detail_representation_policy=='typed_components_v2'
              and value.ndim==2 and target[key].ndim==2
              and target[key].shape[0]>=value.shape[0] and target[key].shape[1]==value.shape[1]):
            expanded=target[key].clone();expanded[:value.shape[0]]=value
            transferred[key]=expanded
        else:raise ValueError(f'Unsafe stage migration shape: {key}: {tuple(value.shape)} -> {tuple(target[key].shape)}')
    upgraded.load_state_dict(transferred,strict=False)
    if (getattr(model.config,'detail_mode_parameterization','legacy_v1')=='legacy_v1'
            and upgraded.config.detail_mode_parameterization=='centered_v2'
            and hasattr(model,'detail_mode_control') and hasattr(upgraded,'detail_mode_control')):
        if model.config.detail_representation_policy!=upgraded.config.detail_representation_policy:
            raise ValueError('Center legacy detail modes before changing their candidate feature representation')
        upgraded.absorb_legacy_detail_mode_mean()
    return upgraded


BATCH_ORDER_POLICIES=('song_local_v1','interleaved_microbatches_v2')


def resolve_batch_order_policy(requested='auto',old_args=None,*,resume=False):
    inherited=(old_args or {}).get('batch_order_policy','song_local_v1')
    value=(inherited if resume else 'interleaved_microbatches_v2') if requested=='auto' else requested
    if value not in BATCH_ORDER_POLICIES:raise ValueError('Unsupported batch_order_policy')
    if resume and value!=inherited:raise ValueError('Resume inherits batch_order_policy; use finetune')
    return value


class _StreamArgumentParser(argparse.ArgumentParser):
    def parse_args(self,args=None,namespace=None):
        import sys
        tokens=list(sys.argv[1:] if args is None else args)
        parsed=super().parse_args(tokens,namespace)
        parsed._explicit_batch_controls=tuple(name for name in ('batch_size','grad_accumulation','lr','audio_cache_gb')
            if any(token=='--'+name.replace('_','-') or token.startswith('--'+name.replace('_','-')+'=') for token in tokens))
        return parsed


def parser():
    p=_StreamArgumentParser(description=__doc__)

    for key in ('manifest','checkpoint-dir'):p.add_argument('--'+key,required=True)
    p.add_argument('--cache-dir',default='cache/chart_composer_v1')
    p.add_argument('--rebatch-resume',action='store_true',help='Explicitly remap a complete resume cursor while retaining effective batch and sample order')
    p.add_argument('--batch-order-policy',choices=('auto',*BATCH_ORDER_POLICIES),default='auto')
    p.add_argument('--data-workers',type=int,choices=(0,1),default=0)
    p.add_argument('--detail-compute',choices=('serial','packed'),default='serial',
        help='Batch equivalent detail recurrence across shapes; serial is a diagnostic reference')
    p.add_argument('--validation-manifest');p.add_argument('--resume');p.add_argument('--init-checkpoint')
    for key,default in dict(epochs=50,batch_size=16,grad_accumulation=4,d_model=256,layers=4,nhead=8,
        dim_feedforward=768,history_events=64,rollout_steps=16,checkpoint_interval_steps=500,
        log_interval=25,max_steps=0,limit_charts=0,prepare_workers=1,cpu_threads=2,
        recovery_interval=25,recovery_samples=2,recovery_steps=2,symbolic_pretrain_epochs=0,
        validation_max_batches=32,validation_batch_size=4,generation_eval_songs=2,
        generation_eval_interval_epochs=1,seed=0,warmup_steps=0,prefetch_batches=2,sample_workers=1).items():
        p.add_argument('--'+key.replace('_','-'),type=int,default=default)
    for key,default in dict(lr=.0002,dropout=.1,window_seconds=8.,recovery_weight=.1,
        plan_weight=.1,audio_cache_gb=1.,min_lr_ratio=.1).items():
        p.add_argument('--'+key.replace('_','-'),type=float,default=default)
    for key in ('d_model','layers','nhead','dim_feedforward','history_events','rollout_steps','dropout'):p.set_defaults(**{key:None})
    p.add_argument('--lr-schedule',choices=('constant','cosine'),default='constant')
    p.add_argument('--reset-learning-rate',action='store_true',
                   help='Explicitly replace a resumed LR plan while retaining optimizer/RNG/data progress')
    p.add_argument('--timing-compute',choices=('supervised','dense'),default='supervised',
                   help='Equivalent training timing execution; dense is the performance reference')
    p.add_argument('--stop-after-updates',type=int,default=0,
                   help='Save and exit after this many additional successful updates; 0 is unlimited')
    p.add_argument('--generation-eval-seeds',default='0,1')
    p.add_argument('--audio-grid-seconds',type=float,default=None,
                   help='Fresh default 0.01 seconds; continuation inherits its checkpoint grid')
    p.add_argument('--plan-blank-window-seconds',type=float,default=None,
                   help='Fresh default 0.5-second centered blank label; continuation inherits checkpoint')
    for key,choices in LEARNING_POLICIES.items():p.add_argument('--'+key.replace('_','-'),choices=('auto',*choices),default='auto')
    p.add_argument('--transition-exit-weight',type=float,default=None)
    p.add_argument('--prefix-augmentation-rate',type=float,default=None)
    p.add_argument('--event-family-audio-weight',type=float,default=None)
    p.add_argument('--phrase-loss-weight',type=float,default=None)
    for key in ('max_phrase_events','phrase_seam_events','phrase_modes','song_profile_components','song_response_modes','song_response_group_size'):
        p.add_argument('--'+key.replace('_','-'),type=int,default=None)
    p.add_argument('--phrase-relation-weight',type=float,default=None)
    p.add_argument('--song-profile-weight',type=float,default=None)
    p.add_argument('--generation-eval-temperature',type=float,default=None)
    from .stream_evaluation_settings import add_arguments
    add_arguments(p)
    p.add_argument('--device',default='auto');p.add_argument('--precision',choices=('auto','bf16','fp16','fp32'),default='bf16')
    return p

def _emit(kind,**values):print(kind+' '+json.dumps(values,ensure_ascii=False),flush=True)

def _dataset(args,*,codec=None,delta_vocab=None,validation=False):
    dataset=StreamDataset(args.validation_manifest if validation else args.manifest,
        cache_dir=args.cache_dir,codec=codec,delta_vocab=delta_vocab,lead_in=args.history_events,
        targets=args.rollout_steps,window_seconds=args.window_seconds,
        preparation_workers=args.prepare_workers,audio_cache_gb=args.audio_cache_gb,
        audio_grid_seconds=args.audio_grid_seconds,
        plan_blank_window_seconds=args.plan_blank_window_seconds,
        timing_output_policy=getattr(args,'timing_output_policy','relative_delta_v1'),
        window_balance_policy=getattr(args,'window_balance_policy','equal_window_v1'),
        history_window_policy=('block_aligned_v2' if getattr(args,'history_window_policy','auto')=='auto' else args.history_window_policy),
        continuation_policy=('pattern_runs_v2' if getattr(args,'continuation_policy','auto')=='auto' else args.continuation_policy),
        phrase_context_policy=getattr(args,'phrase_context_policy','off'),
        candidate_state_policy=getattr(args,'candidate_state_policy','cursor_v1'),
        training_window_policy=getattr(args,'training_window_policy','block_v1'),
        max_phrase_events=getattr(args,'max_phrase_events',64),phrase_seam_events=getattr(args,'phrase_seam_events',2),
        song_response_group_size=getattr(args,'song_response_group_size',4),
        relation_policy='future_sets_v1' if getattr(args,'phrase_execution_policy','off') in ('persistent_modes_v1','component_modes_v2') else 'off',
        limit_charts=args.limit_charts,unknown_policy='mask' if validation else 'error')
    dataset.physical_policy=getattr(args,'physical_policy','legacy_report_v1')
    dataset.recovery_target_policy=getattr(args,'recovery_target_policy','authored_next_v1')
    dataset._batch_order_policy=getattr(args,'batch_order_policy','song_local_v1')
    return dataset

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
            shuffled=[rows[i] for i in perm]
            if getattr(self.ds,'training_window_policy','block_v1')=='phrase_complete_v1':
                # Bounded sortish pools retain song locality and stochastic
                # batches, while grouping similar causal sequence lengths.
                # Fixed sample counts keep batch/optimizer cursor contracts.
                order_bs=int(getattr(self.ds,'_sample_order_batch_size',self.bs))
                span=max(order_bs,order_bs*16)
                packed=[]
                for start in range(0,len(shuffled),span):
                    pool=shuffled[start:start+span]
                    pool.sort(key=lambda index:self.ds.samples[index][2]-self.ds.samples[index][1])
                    chunks=[pool[j:j+order_bs] for j in range(0,len(pool),order_bs)]
                    chunk_order=torch.randperm(len(chunks),generator=g).tolist()
                    packed.extend(index for j in chunk_order for index in chunks[j])
                shuffled=packed
            indices.extend(shuffled)
        batches=[indices[start:start+self.bs] for start in range(0,len(indices),self.bs)]
        policy=getattr(self.ds,'_batch_order_policy','song_local_v1')
        if policy=='interleaved_microbatches_v2':
            full=[batch for batch in batches if len(batch)==self.bs]
            tail=[batch for batch in batches if len(batch)<self.bs]
            order_generator=torch.Generator().manual_seed(self.seed+self.epoch*1000003+0x494E5445)
            order=torch.randperm(len(full),generator=order_generator).tolist()
            batches=[full[index] for index in order]+tail
        elif policy!='song_local_v1':raise ValueError('Unsupported dataset batch order policy')
        yield from batches[self.skip:]

def batch_order_diagnostics(ds,args,epoch,*,max_updates=100):
    """Inspect optimizer-window chart diversity using sample metadata only."""
    accumulation=max(1,int(getattr(args,'grad_accumulation',1)))
    values=[];charts=set();microbatches=0
    for indices in _LocalBatches(ds,args.batch_size,args.seed,epoch):
        charts.update(int(ds.samples[index][0]) for index in indices)
        microbatches+=1
        if microbatches%accumulation==0:
            values.append(len(charts));charts.clear()
            if len(values)>=max_updates:break
    if charts and len(values)<max_updates:values.append(len(charts))
    return dict(policy=getattr(ds,'_batch_order_policy','song_local_v1'),epoch=epoch,
        scope='CPU_sample_metadata_only_first_optimizer_windows',
        sampled_optimizer_steps=len(values),unique_charts_per_optimizer_step=values,
        mean_unique_charts=float(np.mean(values)) if values else 0.,
        min_unique_charts=min(values,default=0),max_unique_charts=max(values,default=0))


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
        original_items=items
        if self.rate:
            from .stream_augmentation import augment_stream_items
            items,diagnostics=augment_stream_items(self.dataset,items,rate=self.rate,
                seed=self.seed+self.batch_number,input_policy=self.input_policy)
        batch=collate_stream(items)
        batch['augmented_row_mask']=torch.tensor([item is not original for item,original in zip(items,original_items)],dtype=torch.bool)
        batch['_augmentation']=diagnostics
        return batch


class _ProcessBatches(Sampler):
    def __init__(self,ds,args,epoch,skip):
        self.ds=ds;self.bs=args.batch_size;self.seed=args.seed
        self.reset(epoch,skip)
    def reset(self,epoch,skip):
        self.epoch=epoch;self.skip=skip;self.symbolic_only=bool(getattr(self.ds,'symbolic_only',False))
    def __len__(self):return len(_LocalBatches(self.ds,self.bs,self.seed,self.epoch,self.skip))
    def __iter__(self):
        for number,indices in enumerate(_LocalBatches(self.ds,self.bs,self.seed,self.epoch,self.skip),self.skip+1):
            yield [(index,self.epoch,number,self.symbolic_only) for index in indices]


class _ProcessRows(Dataset):
    def __init__(self,dataset):self.dataset=dataset
    def __len__(self):return len(self.dataset)
    def __getitem__(self,key):
        index,epoch,number,symbolic=key
        started=time.perf_counter()
        self.dataset.set_symbolic_only(symbolic)
        item=self.dataset[index]
        item['_process_cursor']=(epoch,number)
        item['_process_row_sec']=time.perf_counter()-started
        return item


class _ProcessCollator:
    def __init__(self,dataset,args):self.dataset=dataset;self.args=args
    def __call__(self,items):
        started=time.perf_counter()
        epoch,number=items[0]['_process_cursor']
        if any(item['_process_cursor']!=(epoch,number) for item in items):
            raise ValueError('Mixed process batch cursors')
        row_seconds=sum(item.pop('_process_row_sec') for item in items)
        for item in items:item.pop('_process_cursor')
        output=_TrainingCollator(self.dataset,self.args,epoch,skip=number-1)(items)
        output['_process_stats']={'prepare_sec':row_seconds+time.perf_counter()-started,
            'cache_scope':'worker_process','cache':self.dataset.cache_stats() if hasattr(self.dataset,'cache_stats') else {}}
        return output


def _process_worker_init(worker_id):
    torch.set_num_threads(1)


def _shutdown_process_loader(ds):
    loader=getattr(ds,'_process_loader',None)
    iterator=getattr(loader,'_iterator',None)
    if iterator is not None:iterator._shutdown_workers()
    for name in ('_process_loader','_process_sampler','_process_loader_key'):
        ds.__dict__.pop(name,None)


class _ProcessPrepared:
    def __init__(self,loader):
        started=time.perf_counter();self.source=iter(loader)
        self.wait=time.perf_counter()-started;self.prepare=0.;self.count=0;self.cache={}
        self.depth=loader.prefetch_factor
        if not getattr(loader,'_parent_cache_released',False):
            dataset=loader.dataset.dataset
            if hasattr(dataset,'release_hot_caches'):dataset.release_hot_caches(include_audio=True)
            loader._parent_cache_released=True
    def __iter__(self):return self
    def __next__(self):
        started=time.perf_counter()
        try:batch=next(self.source)
        finally:self.wait+=time.perf_counter()-started
        stats=batch.pop('_process_stats',{})
        self.prepare+=stats.get('prepare_sec',0.);self.count+=1;self.cache=stats.get('cache',{})
        return batch
    def stats(self):
        return dict(mode='persistent_process',depth=self.depth,sample_workers=1,prepared_batches=self.count,
            prepare_sec=round(self.prepare,3),consumer_wait_sec=round(self.wait,3),cache_scope='worker_process',cache=self.cache)


def _loader(ds,args,epoch,skip=0,validation=False):
    # Ordered archives are shared across rows and departure/landing queries.
    # Collation owns their physical packing; standalone dataset rows remain eager.
    ds.defer_phrase_sequence_packing=True
    ds._batch_order_policy=getattr(args,'batch_order_policy','song_local_v1')
    # Fixed per-row denominator keeps the final short batch from overweighting
    # its charts in the whole-song profile objective.
    ds._profile_reference_batch_size=args.validation_batch_size if validation else args.batch_size
    # DataLoader iterator creation must not consume the model/dropout RNG on
    # resume, where the number of loader constructions differs from a full run.
    generator=torch.Generator().manual_seed(args.seed+epoch*1000003+1729)
    if validation:
        budget=getattr(args,'validation_max_batches',0)*args.validation_batch_size
        if budget and len(ds)>budget:
            # Validation rows are song-local in the dataset. A prefix-only
            # batch limit would repeatedly measure just the first few songs.
            selected=np.linspace(0,len(ds)-1,budget,dtype=np.int64).tolist()
            ds=Subset(ds,selected)
        return DataLoader(ds,batch_size=args.validation_batch_size,collate_fn=collate_stream,num_workers=0,generator=generator)
    if getattr(args,'data_workers',0):
        key=(args.batch_size,args.seed,getattr(args,'prefix_augmentation_rate',0.),
             getattr(args,'prefix_augmentation_input_policy','causal_window_config_time_first_decision_v3'),
             args.device,min(args.prefetch_batches or 2,4),ds._batch_order_policy)
        if getattr(ds,'_process_loader_key',None)!=key:
            _shutdown_process_loader(ds)
            sampler=_ProcessBatches(ds,args,epoch,skip)
            loader=DataLoader(_ProcessRows(ds),batch_sampler=sampler,collate_fn=_ProcessCollator(ds,args),
                num_workers=1,persistent_workers=True,multiprocessing_context='spawn',pin_memory=True,
                prefetch_factor=min(args.prefetch_batches or 2,4),worker_init_fn=_process_worker_init,generator=generator)
            # StreamDataset serialization excludes parent loader/sampler state.
            ds._process_loader=loader;ds._process_sampler=sampler;ds._process_loader_key=key
        else:ds._process_sampler.reset(epoch,skip)
        return ds._process_loader
    return DataLoader(ds,batch_sampler=_LocalBatches(ds,args.batch_size,args.seed,epoch,skip),
        collate_fn=_TrainingCollator(ds,args,epoch,skip),num_workers=0,pin_memory=args.device.startswith('cuda'),generator=generator)

@contextmanager
def _prepared_batches(loader,args):
    if getattr(loader,'num_workers',0):
        yield _ProcessPrepared(loader)
    elif args.prefetch_batches:
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

def _resume_batch_mapping(old_args,state,args,sample_count):
    old_bs=int(old_args['batch_size']);old_acc=int(old_args['grad_accumulation'])
    old_batches=math.ceil(sample_count/old_bs)
    if int(old_args.get('max_steps',0)):old_batches=min(old_batches,int(old_args['max_steps']))
    _validate_resume_cursor(state,old_batches,old_acc)
    policy=resolve_batch_order_policy(getattr(args,'batch_order_policy','auto'),old_args,resume=True)
    if policy=='interleaved_microbatches_v2' and old_bs!=args.batch_size:
        raise ValueError('interleaved_microbatches_v2 cannot rebatch: the shuffled microbatch unit would change')
    changed=(old_bs,old_acc)!=(args.batch_size,args.grad_accumulation)
    if not changed:return int(state['batch_in_epoch']) if not state.get('epoch_complete') else 0,None
    if not getattr(args,'rebatch_resume',False):raise ValueError('Resume batch changes require --rebatch-resume')
    if old_bs*old_acc!=args.batch_size*args.grad_accumulation:
        raise ValueError('Rebatch resume requires unchanged effective batch size')
    if int(old_args.get('max_steps',0)) or args.max_steps:
        raise ValueError('Rebatch resume requires max_steps=0 to preserve epoch coverage')
    consumed=0 if state.get('epoch_complete') else min(int(state['batch_in_epoch'])*old_bs,sample_count)
    if consumed==sample_count:
        cursor=math.ceil(sample_count/args.batch_size)
    else:
        if consumed%args.batch_size:raise ValueError('Resume sample cursor is not divisible by new batch size')
        cursor=consumed//args.batch_size
        if cursor%args.grad_accumulation:raise ValueError('Rebatch cursor must remain at a complete optimizer boundary')
    return cursor,dict(previous_batch_size=old_bs,previous_grad_accumulation=old_acc,
        batch_size=args.batch_size,grad_accumulation=args.grad_accumulation,
        previous_batch_cursor=int(state['batch_in_epoch']),batch_cursor=cursor,
        consumed_samples=consumed,epoch_complete=bool(state.get('epoch_complete',False)),
        sample_order_batch_size=int(old_args.get('sample_order_batch_size',old_bs)))


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
    if getattr(args,'batch_order_policy','song_local_v1')=='interleaved_microbatches_v2':
        h.update(b'batch_order:interleaved_microbatches_v2')
    if getattr(args,'training_window_policy','block_v1')=='phrase_complete_v1':
        h.update(b'causal_phrase_packed_blocks_v2')
    h.update(json.dumps(ds.codec.to_dict(),sort_keys=True).encode())
    h.update(json.dumps(ds.delta_vocab.to_dict(),sort_keys=True).encode())
    h.update(str((args.history_events,args.rollout_steps,args.window_seconds)).encode())
    for key,default in dict(training_window_policy='block_v1',phrase_execution_policy='off',
                            detail_representation_policy='private_v1',duration_feature_policy='legacy_v1').items():
        value=getattr(args,key,default)
        if value!=default:h.update((key+':'+value).encode())
    if getattr(args,'training_window_policy','block_v1')!='block_v1':
        h.update(json.dumps(('song_sortish_v1',getattr(args,'max_phrase_events',64),getattr(args,'phrase_seam_events',2))).encode())
    if getattr(args,'phrase_execution_policy','off')!='off':
        h.update(json.dumps((getattr(args,'phrase_modes',4),getattr(args,'phrase_relation_weight',.1))).encode())
    if getattr(args,'song_profile_policy','off')!='off':
        h.update(json.dumps(('song_profile',args.song_profile_policy,args.song_profile_components,args.song_profile_weight)).encode())
    if getattr(args,'song_response_policy','off')!='off':
        h.update(json.dumps(('song_response',args.song_response_policy,args.song_response_modes,args.song_response_group_size)).encode())
    if not legacy:h.update(str(float(args.audio_grid_seconds)).encode())
    if not legacy and not pre_blank:h.update(str(float(getattr(args,'plan_blank_window_seconds',.5))).encode())
    policies=(getattr(args,'history_window_policy','rolling_legacy_v1'),getattr(args,'continuation_policy','off'))
    if policies!=('rolling_legacy_v1','off'):h.update(json.dumps(policies).encode())
    if getattr(args,'prefix_augmentation_rate',0.):
        # Configuration/time reconstruction and native whole-window selection
        # define training examples. Changing that recipe requires a new stage.
        h.update(getattr(args,'prefix_augmentation_input_policy','causal_window_config_time_first_decision_v3').encode())
    target_policy=getattr(args,'recovery_target_policy','authored_next_v1')
    if target_policy!='authored_next_v1':h.update(target_policy.encode())
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
    for key,default in (('phrase_context_policy','off'),('duration_audio_policy','off'),('physical_policy','legacy_report_v1')):
        value=getattr(args,key,default)
        if value!=default:h.update((key+':'+value).encode())
    if getattr(args,'phrase_context_policy','off')!='off':h.update(str(getattr(args,'phrase_loss_weight',.05)).encode())
    for key,default in (('template_policy','cross_family_v1'),('candidate_state_policy','cursor_v1')):
        value=getattr(args,key,default)
        if value!=default:h.update((key+':'+value).encode())
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
    preserve_family_prior=(parent_config is not None
        and parent_config.event_family_policy=='audio_prior_transition_v3'
        and model.config.event_family_policy=='audio_state_phrase_v4')
    init_family=(semantic and not resume and not preserve_family_prior
        and (parent_config is None or parent_config.event_family_policy!=model.config.event_family_policy))
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
        condition_scope='reference-derived requested controls',scope='authored_prefix_not_musical_quality',
        selection='deterministic_spread_across_all_samples' if args.validation_max_batches else 'all_samples')

def _autocast(args):
    return torch.autocast('cuda',dtype=torch.bfloat16 if args.precision=='bf16' else torch.float16,
        enabled=args.device.startswith('cuda') and args.precision in ('bf16','fp16'))

def _inherit_resume_batch_defaults(args, old):
    explicit=getattr(args,'_explicit_batch_controls',None)
    if explicit is not None:
        for name in ('batch_size','grad_accumulation','lr','audio_cache_gb'):
            if name not in explicit and name in old.get('args',{}):setattr(args,name,old['args'][name])


def _validation_improvement(validation,previous,epoch,steps):
    metrics=validation.get('metrics',{})
    field=('unified_balanced_variational_objective' if 'unified_balanced_variational_objective' in metrics
           else 'main_prediction_loss')
    score=float(metrics.get(field,float('inf')))
    if not math.isfinite(score):return None
    if previous and previous.get('field')==field and score>=previous['score']:return None
    return dict(field=field,score=score,epoch=int(epoch),optimizer_steps=int(steps),
        scope='development_authored_prefix_objective_not_playtest_quality')


def _retain_best_files(directory):
    import os,shutil
    for source,target in (('composer_last.pt','composer_best.pt'),('composer_inference.pt','composer_best_inference.pt')):
        path=Path(directory)/target;temporary=path.with_suffix('.tmp')
        shutil.copyfile(Path(directory)/source,temporary)
        os.replace(temporary,path)


def _train(args):
    from .learning_rate import resolve_plan,learning_rate
    from .stream_recovery import recovery_batch,recovery_state,restore_recovery_state
    if getattr(args,'reset_learning_rate',False) and not args.resume:
        raise ValueError('--reset-learning-rate requires --resume')
    if getattr(args,'stop_after_updates',0)<0:
        raise ValueError('--stop-after-updates must be nonnegative')
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
    if args.cpu_threads<0:raise ValueError('CPU threads must be nonnegative; 0 selects automatic')
    args.effective_cpu_threads=(args.cpu_threads if args.cpu_threads else
        min(2,torch.get_num_threads()) if args.device.startswith('cuda') else torch.get_num_threads())
    torch.set_num_threads(args.effective_cpu_threads);random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    directory=Path(args.checkpoint_dir);directory.mkdir(parents=True,exist_ok=True)
    if not args.resume and (directory/'composer_last.pt').exists():raise ValueError('Use a new directory for fresh/finetune')
    old=None
    if args.resume or args.init_checkpoint:model,old=checkpoint.load(args.resume or args.init_checkpoint,args.device,include_optimizer=bool(args.resume))
    if args.resume:
        if not checkpoint.training_objective_compatible(old):
            raise ValueError('Unified training objective changed: use --init-checkpoint in a new directory instead of --resume')
        # CLI omissions inherit the checkpoint; explicit changes still pass
        # through the complete-boundary rebatch validation below.
        _inherit_resume_batch_defaults(args,old)
        saved=old.get('training_state') or {}
        next_epoch=int(old.get('epoch',0))+int(saved.get('epoch_complete',False))
        if next_epoch>args.epochs:
            raise ValueError(f'epochs is the total target, not added epochs; next epoch is {next_epoch}, requested target is {args.epochs}')
    for key,default in dict(d_model=256,layers=4,nhead=8,dim_feedforward=768,history_events=64,rollout_steps=16,dropout=.1).items():
        if getattr(args,key) is None:setattr(args,key,getattr(model.config,key) if old else default)
    if args.rollout_steps<1 or args.history_events<1:raise ValueError('Invalid history/rollout controls')
    learning_policies=resolve_learning_policies(args,model.config if old else None,resume=bool(args.resume))
    for key,value in learning_policies.items():setattr(args,key,value)
    args.batch_order_policy=resolve_batch_order_policy(getattr(args,'batch_order_policy','auto'),(old or {}).get('args'),resume=bool(args.resume))
    inherited_temperature=((old or {}).get('args') or {}).get('generation_eval_temperature',.65)
    if args.generation_eval_temperature is None:args.generation_eval_temperature=inherited_temperature if args.resume else .65
    if not math.isfinite(args.generation_eval_temperature) or args.generation_eval_temperature<0:
        raise ValueError('Generation evaluation temperature must be finite and nonnegative')
    if args.resume and args.generation_eval_temperature!=inherited_temperature:
        raise ValueError('Resume inherits generation_eval_temperature; use finetune')
    from .stream_evaluation_settings import from_training_args
    generation_settings=from_training_args(args)
    if args.init_checkpoint:model=upgrade_learning_stage(model,learning_policies,args.device)
    args.audio_grid_seconds=_resolve_audio_grid(args.audio_grid_seconds,model.config if old else None)
    inherited_blank=model.config.plan_blank_window_seconds if old else .5
    if args.plan_blank_window_seconds is not None and (not math.isfinite(args.plan_blank_window_seconds) or args.plan_blank_window_seconds<0):
        raise ValueError('Plan blank window must be finite and nonnegative')
    if old and args.plan_blank_window_seconds is not None and args.plan_blank_window_seconds!=inherited_blank:
        raise ValueError('Continuation inherits checkpoint plan_blank_window_seconds')
    args.plan_blank_window_seconds=inherited_blank if args.plan_blank_window_seconds is None else args.plan_blank_window_seconds
    if args.validation_manifest:
        from .evaluation import assert_unseen
        assert_unseen(args.manifest,args.validation_manifest,cache_dir=args.cache_dir)
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
        for name in ('history_events','rollout_steps','d_model','layers','nhead','dim_feedforward','dropout'):
            if getattr(args,name)!=getattr(model.config,name):raise ValueError(f'Continuation requires saved {name}')
        if args.init_checkpoint:
            model.config.plan_weight=args.plan_weight;model.config.recovery_weight=args.recovery_weight
    output_initialization=_initialize_structured_heads(model,ds,
        StreamComposerConfig.from_dict(old['model_config']) if old else None,resume=bool(args.resume))
    args.training_objective_recipe=dict(version=(checkpoint.UNIFIED_OBJECTIVE_VERSION if model.config.song_response_policy=='unified_profile_response_v2' else
        ('stream_ordered_song_profile_v7' if model.config.song_profile_policy=='masked_ordered_joint_v2' else 'stream_song_response_joint_v6' if model.config.song_response_policy!='off' or model.config.song_profile_policy!='off' else 'stream_persistent_phrase_joint_v5' if model.config.phrase_execution_policy in ('persistent_modes_v1','component_modes_v2') else 'stream_shared_causal_correspondence_v4' if model.config.event_family_policy=='shared_continuation_v1'
        else 'stream_aligned_phrase_joint_v3' if model.config.phrase_context_policy=='aligned_phrase_v3' else 'stream_ordered_phrase_joint_v2'
        if model.config.phrase_context_policy in ('ordered_phrase_v2','aligned_phrase_v3','shared_ordered_v4') else 'stream_causal_phrase_joint_v1')),
        song_profile_policy=model.config.song_profile_policy,song_profile_components=model.config.song_profile_components,song_profile_weight=model.config.song_profile_weight,
        song_response_policy=model.config.song_response_policy,song_response_modes=model.config.song_response_modes,song_response_group_size=model.config.song_response_group_size,
        song_profile_objective=('unified_z_profile_evidence_once_chart_budget_CRPS_50_percent_only_level_v3' if model.config.song_profile_policy=='masked_deployment_joint_v3' else 'masked_joint_nll_plus_ordinal_crps_equal_chart_mass_v2' if model.config.song_profile_policy=='masked_ordered_joint_v2' else 'masked_mixture_categorical_129_bins_preprojection_unique_charts_per_batch'),
        plan_metric_contract='unchanged_poisson_nll_with_saturated_floor_and_excess_diagnostics',
        song_response_objective=('balanced_hierarchical_variational_risk_consistent_phrase_and_chart_evidence_budgets' if model.config.song_response_policy=='unified_profile_response_v2' else 'same_chart_distant_complete_phrase_subgroups_nested_composite_likelihood_prior_only'),
        time_configuration_budget=('unified_event_nll_once' if model.config.song_response_policy=='unified_profile_response_v2' else 'separate_time_and_configuration_joint_terms'),
        song_style_runtime='audio_condition_choice_once_immutable_across_prefix_replay',
        phrase_context_policy=model.config.phrase_context_policy,
        training_window_policy=model.config.training_window_policy,max_phrase_events=model.config.max_phrase_events,
        phrase_seam_events=model.config.phrase_seam_events,phrase_execution_policy=model.config.phrase_execution_policy,
        phrase_modes=model.config.phrase_modes,phrase_relation_weight=model.config.phrase_relation_weight,
        detail_representation_policy=model.config.detail_representation_policy,duration_feature_policy=model.config.duration_feature_policy,
        detail_mode_parameterization=model.config.detail_mode_parameterization,
        relation_policy='future_sets_v1' if model.config.phrase_execution_policy in ('persistent_modes_v1','component_modes_v2') else 'off',
        phrase_configuration_objective=('whole_phrase_mean_event_risk_with_prior_KL_scaled_by_phrase_weight_mass' if model.config.song_response_policy=='unified_profile_response_v2' else 'whole_trusted_phrase_mode_marginal_replaces_configuration_nll_local_recovery_mixture'
            if model.config.phrase_execution_policy in ('persistent_modes_v1','component_modes_v2') else 'local_configuration_nll'),
        configuration_loss_scope=('sampled_chart_joint_event_nll_one_time_configuration_budget' if model.config.song_response_policy=='unified_profile_response_v2' else 'whole_phrase_weighted_composite_likelihood_complete_trusted_groups_else_local_mixture'
            if model.config.phrase_execution_policy in ('persistent_modes_v1','component_modes_v2') else 'base_head_weighted_joint_configuration_nll'),
        configuration_base_diagnostics='shape_loss_plus_phase_loss_plus_detail_loss_base_heads_only',
        unified_event_nll_field='unified_event_nll' if model.config.song_response_policy=='unified_profile_response_v2' else None,
        unified_event_nll_scope=('legacy_metric_alias_for_balanced_conditional_phrase_risk_including_inner_mode_KL' if model.config.song_response_policy=='unified_profile_response_v2' else None),
        configuration_adjustment_identity=('unified_event_joint_replaces_separate_time_and_configuration_budget' if model.config.song_response_policy=='unified_profile_response_v2' else 'configuration_loss_equals_shape_plus_phase_plus_detail_plus_phrase_configuration_adjustment'),
        configuration_adjustment_budget=('single_unified_event_nll_no_duplicate_separate_time_or_configuration' if model.config.song_response_policy=='unified_profile_response_v2' else 'replacement_difference_not_additional_configuration_cross_entropy'),
        phrase_group_weighting='sample_weight_inside_mode_sequence_sum_actual_effective_target_budget_normalizer',
        phrase_mode_details=('shared_recurrence_once_mode_component_biases_in_same_joint_likelihood' if model.config.phrase_execution_policy=='component_modes_v2' else 'shared_conditional_detail_scores_computed_once_without_mode_axis'),
        effective_target_budget='sum_cpu_loss_normalizer_multiplicity_corrected_primary_and_seam_target_exposure',
        effective_target_budget_scope='consumed_authored_batches_this_epoch_execution_since_resume_including_discarded_updates_excluding_recovery',
        effective_targets_per_sec_scope='same_execution_training_loop_wall_time_including_data_transfer_compute_and_in_loop_checkpoint_io_excluding_epoch_end_validation',
        batch_order=args.batch_order_policy,
        phrase_history_contract='causal_phrase_packed_blocks_v2',
        relation_query_contract='departure_decoder_hidden_no_selected_target_clock',
        phrase_loss_weight=model.config.phrase_loss_weight,
        phrase_target='ordered_segment_features_labels_only',
        ordered_phrase_target=('configuration_only_ordered_encoder_cosine_detached_symmetric_match_joint'
            if model.config.phrase_context_policy in ('ordered_phrase_v2','aligned_phrase_v3','shared_ordered_v4') else 'off'),
        phrase_budget=('half_coarse_statistics_half_ordered_cosine_and_multi_positive_match'
            if model.config.phrase_context_policy in ('ordered_phrase_v2','aligned_phrase_v3','shared_ordered_v4') else 'coarse_statistics'),
        ordered_memory_max_events=32 if model.config.phrase_context_policy in ('ordered_phrase_v2','aligned_phrase_v3','shared_ordered_v4') else 0,
        phrase_loss_normalization='sum_weighted_features_over_batch_and_features_query_budget_preserved',
        phrase_input='audio_anchor_and_completed_prefix_references',
        audio_tracking_training='25_percent_rows_uniform_strength_otherwise_zero',
        duration_audio_policy=model.config.duration_audio_policy,
        physical_policy=model.config.physical_policy,
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
        family_audio_budget=(model.config.event_family_audio_weight if model.config.event_family_policy in ('audio_semantic_v2','audio_prior_transition_v3','audio_state_phrase_v4','shared_continuation_v1') else 0.),
        family_full_budget=(1-model.config.event_family_audio_weight if model.config.event_family_policy in ('audio_semantic_v2','audio_prior_transition_v3','audio_state_phrase_v4','shared_continuation_v1') else 1.),
        family_mixture_gradient=('audio_prior_detached_context_residual_live' if model.config.event_family_policy=='audio_state_phrase_v4' else
            'audio_prior_detached_when_audio_budget_positive' if model.config.event_family_policy=='audio_prior_transition_v3' else 'joint'),
        family_context_inputs=('shared_causal_history_landing_music_phrase_progress_native_state'
            if model.config.event_family_policy=='shared_continuation_v1' else
            'music_phrase_intent_candidate_state_gap' if model.config.event_family_policy=='audio_state_phrase_v4' else 'policy_default'),
        conditional_geometry=('joint_phase_detail_with_learned_motion_exit'
            if model.config.transition_policy in ('music_exit_v1','music_exit_v2','music_exit_v3') else 'unchanged_joint_phase_detail'),
        transition_exit='additional_weighted_joint_nll_on_exit_targets',
        template_policy=model.config.template_policy,
        candidate_state_policy=model.config.candidate_state_policy,
        transition_factorization=('parent_keep_exit_log_odds_plus_musical_residual'
            if model.config.transition_policy in ('music_exit_v2','music_exit_v3') else
            'learned_keep_exit_time_motion_template_same_nll_budget'
            if model.config.transition_policy=='music_exit_v1' else 'direct_joint_without_keep_exit_override'),
        transition_residual_gain=(8. if model.config.transition_policy in ('music_exit_v2','music_exit_v3') else 1.
            if model.config.transition_policy=='music_exit_v1' else 0.),
        time_music_reference=('strict_prior_note_at_cursor'
            if model.config.transition_policy=='music_exit_v3' else 'checkpoint_policy_defined'),
        history_recovery=model.config.recovery_policy,
        recovery_target_policy=model.config.recovery_target_policy,
        configuration_target=('authored_or_certified_causal_geometry_correspondence'
            if model.config.recovery_target_policy=='causal_correspondence_v1' else 'next_reachable_authored_event'),
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
    if dev is not None:
        _validation_timing_support(dev,model)
        trusted=sum(1 for song,stream in enumerate(dev.streams) for index,step in enumerate(stream)
                    if step.weight>0 and dev.chart_weights[song]>0 and not dev.unknown_steps[song][index])
        if not len(dev) or not trusted:
            raise ValueError('Validation set has no trusted supervised targets; choose eligible independent songs or explicitly disable validation')
        _emit('composer_validation_support',samples=len(dev),trusted_targets=trusted,oov=dev.oov)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01,
        **({'fused':True} if args.device.startswith('cuda') else {}))
    from .stream_data_identity import dataset_content_identity
    content_identity=dataset_content_identity(args.manifest,ds,cache_dir=args.cache_dir)
    state=(old or {}).get('training_state') or {};fingerprint=_fingerprint(args,ds)
    args.sample_order_batch_size=int((old or {}).get('args',{}).get('sample_order_batch_size',
        (old or {}).get('args',{}).get('batch_size',args.batch_size))) if args.resume else args.batch_size
    ds._sample_order_batch_size=args.sample_order_batch_size
    ds._batch_order_policy=args.batch_order_policy
    rebatch_changes=list(state.get('rebatch_changes') or [])
    rebatch_mapping=None
    steps=0;start_epoch=1;skip=0
    if args.resume:
        if not old.get('optimizer_state_dict') or not state:
            raise ValueError('Resume requires a complete optimizer/RNG/cursor checkpoint')
        saved_content=state.get('dataset_content_identity') or {}
        if saved_content.get('schema')!=content_identity['schema'] or saved_content.get('fingerprint')!=content_identity['fingerprint']:
            raise ValueError('Resume requires an unchanged, content-verified dataset; this checkpoint lacks matching material identity. Use finetune in a new directory.')
        if state.get('dataset_fingerprint')!=fingerprint:
            legacy_grid=('audio_grid_seconds' not in old['model_config'] and args.audio_grid_seconds==.05
                         and state.get('dataset_fingerprint')==_fingerprint(args,ds,legacy=True))
            pre_blank=('plan_blank_window_seconds' not in old['model_config'] and args.plan_blank_window_seconds==0.
                       and state.get('dataset_fingerprint')==_fingerprint(args,ds,pre_blank=True))
            if not (legacy_grid or pre_blank):raise ValueError('Resume data/codec contract changed')
        optimizer.load_state_dict(old['optimizer_state_dict']);steps=int(old['optimizer_steps'])
        start_epoch=int(old['epoch'])+int(state.get('epoch_complete',False));skip=0 if state.get('epoch_complete') else int(state['batch_in_epoch'])
        skip,rebatch_mapping=_resume_batch_mapping(old['args'],state,args,len(ds))
        if rebatch_mapping is not None:
            rebatch_mapping.update(optimizer_step=steps,epoch=int(old['epoch']))
            rebatch_changes.append(rebatch_mapping)
            _emit('composer_rebatch_resume',**rebatch_mapping)
        for name in ('seed','symbolic_pretrain_epochs','max_steps','recovery_interval','recovery_samples','recovery_steps','recovery_weight','plan_weight'):
            if old['args'].get(name,getattr(args,name))!=getattr(args,name):
                raise ValueError(f'Resume inherits {name}; use finetune for objective changes')
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng'])
        if args.device.startswith('cuda') and state.get('cuda_rng') is not None:torch.cuda.set_rng_state(state['cuda_rng'],device=args.device)
        restore_recovery_state(model,state.get('recovery_episodes'))
    if old is not None:
        # load() materializes the complete checkpoint on CPU before copying
        # model/optimizer tensors to the training device. Only metadata is
        # needed below; retaining these dictionaries wastes host RAM throughout
        # every epoch (especially with a separate data-preparation process).
        old.pop('model_state_dict',None)
        old.pop('optimizer_state_dict',None)
    batches_per_epoch=math.ceil(len(ds)/args.batch_size)
    if args.max_steps:batches_per_epoch=min(batches_per_epoch,args.max_steps)
    if args.resume:
        mapped_state={**state,'batch_in_epoch':skip}
        _validate_resume_cursor(mapped_state,batches_per_epoch,args.grad_accumulation)
        if not state.get('epoch_complete') and skip==batches_per_epoch:
            start_epoch+=1;skip=0
    reset_lr=bool(getattr(args,'reset_learning_rate',False))
    warmup=(state['lr_schedule']['warmup_steps'] if args.resume and state.get('lr_schedule') and not reset_lr else
            min(args.warmup_steps,max(0,math.ceil(batches_per_epoch/args.grad_accumulation)*args.epochs-1)))
    lr_plan=resolve_plan(requested=args.lr_schedule,warmup_steps=warmup,
        min_lr_ratio=args.min_lr_ratio,base_lr=args.lr,total_steps=math.ceil(batches_per_epoch/args.grad_accumulation)*args.epochs,
        resume=bool(args.resume) and not reset_lr,saved=state.get('lr_schedule'))
    lr_changes=list(state.get('learning_rate_changes') or [])
    if reset_lr:
        lr_changes.append(dict(optimizer_step=steps,previous=state.get('lr_schedule'),current=lr_plan,
            reason='explicit_resume_learning_rate_reset'))
        for group in optimizer.param_groups:group['lr']=learning_rate(lr_plan,steps)
        _emit('composer_learning_rate_changed',**lr_changes[-1])
    model._sparse_timing_enabled=getattr(args,'timing_compute','supervised')=='supervised'
    model._packed_detail_enabled=getattr(args,'detail_compute','serial')=='packed'
    stop_at_update=steps+args.stop_after_updates if getattr(args,'stop_after_updates',0) else None
    defaults=(old or {}).get('condition_defaults') or _condition_defaults(ds)
    _emit('composer_runtime',family=checkpoint.FAMILY,architecture_profile=checkpoint.PROFILE,
        training_semantics='causal_event_set_stream_v1',samples=len(ds),batch_size=args.batch_size,
        grad_accumulation=args.grad_accumulation,sample_order_batch_size=args.sample_order_batch_size,history_events=args.history_events,rollout_steps=args.rollout_steps,
        parameters=sum(p.numel() for p in model.parameters()),codec_coverage=ds.codec.coverage,
        effective_lr=optimizer.param_groups[0]['lr'],start_epoch=start_epoch,end_epoch=args.epochs,
        resume_sample_cursor=skip*args.batch_size,precision=args.precision,device=args.device,
        cpu_threads=args.effective_cpu_threads,validation_enabled=dev is not None,
        data_workers=getattr(args,'data_workers',0),
        detail_compute=getattr(args,'detail_compute','serial'),
        timing_compute=getattr(args,'timing_compute','supervised'),stop_at_update=stop_at_update,
        validation_manifest=args.validation_manifest,generation_eval_settings=generation_settings,
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
    best_validation=state.get('best_validation')
    def persist(epoch,batch,complete,metrics):
        s=dict(epoch_complete=complete,batch_in_epoch=batch,dataset_fingerprint=fingerprint,
            best_validation=best_validation,
            dataset_content_identity=content_identity,
            python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(device=args.device) if args.device.startswith('cuda') else None,
            lr_schedule=lr_plan,learning_rate_changes=lr_changes,rebatch_changes=rebatch_changes,recovery_episodes=recovery_state(model),learning_policies=learning_policies,
            parent_stage={'checkpoint':str(args.init_checkpoint),'optimizer_steps':int(old.get('optimizer_steps',0)),
                          'learning_policies':{key:getattr(StreamComposerConfig.from_dict(old['model_config']),key) for key in learning_policies}} if args.init_checkpoint else state.get('parent_stage'))
        kwargs=dict(epoch=epoch,optimizer_steps=steps,args=vars(args),metrics=metrics,
            delta_vocab=ds.delta_vocab,condition_defaults=defaults)
        checkpoint.save(directory/'composer_last.pt',model,optimizer,state=s,**kwargs)
        checkpoint.save(directory/'composer_inference.pt',model,**kwargs)
        _emit('composer_checkpoint',epoch=epoch,optimizer_steps=steps,path=str(directory/'composer_last.pt'))
    if reset_lr or rebatch_mapping is not None:
        # Make an explicitly changed plan durable before another update. A
        # later ordinary resume must not silently recover the old schedule.
        persist(start_epoch,skip,False,(old or {}).get('metrics',{}))
    if not args.resume and dev is not None:
        with _preserve_rng(args.device):initial_validation=evaluate(model,dev,args)
        best_validation=_validation_improvement(initial_validation,None,0,steps)
        if best_validation:
            persist(0,0,True,initial_validation['metrics'])
            _retain_best_files(directory)
            _emit('composer_best_checkpoint',**best_validation,path=str(directory/'composer_best_inference.pt'))
    for epoch in range(start_epoch,args.epochs+1):
        model.train();totals=MetricTotals();n=0;supervised=0;sequences=0;effective_target_budget=0.;started=time.perf_counter();timing=dict(data_sec=0.,transfer_sec=0.,compute_sec=0.)
        optimizer.zero_grad(set_to_none=True)
        symbolic=epoch<=args.symbolic_pretrain_epochs
        # The prior epoch's producer is already joined before changing its view.
        ds.set_symbolic_only(symbolic)
        _emit('composer_stage',epoch=epoch,stage='symbolic' if symbolic else 'audio_conditioned',
              audio_conditioned=not symbolic,symbolic_pretrain_epochs=args.symbolic_pretrain_epochs)
        _emit('composer_batch_order',**batch_order_diagnostics(ds,args,epoch,max_updates=100))
        window_chart_ids=set();last_optimizer_unique_charts=0;optimizer_chart_counts=[]
        window_metrics=MetricTotals();window_batches=0
        window_finite=torch.ones(3,dtype=torch.bool,device=args.device)
        skipped_windows=0;last_norm=None;augmentation_totals={}
        timing_queries=0;timing_queries_total=0
        with _prepared_batches(_loader(ds,args,epoch,skip),args) as loader:
            for bi in range(skip+1,batches_per_epoch+1):
                tick=time.perf_counter();batch=next(loader)
                for key,value in batch.get('_augmentation',{}).items():
                    augmentation_totals[key]=augmentation_totals.get(key,0)+value
                timing['data_sec']+=time.perf_counter()-tick
                if 'training_chart_id' in batch:
                    window_chart_ids.update(int(chart) for chart in batch['training_chart_id'].tolist() if int(chart)>=0)
                else:
                    window_chart_ids.update(int(ds.samples[index][0]) for index in batch.get('_sample_indices',[]) if index is not None)
                supervised+=int((batch['loss_mask'] & batch['sample_weight'].gt(0)).sum())
                # CPU collation has already corrected variable phrase lengths
                # and repeated seams. Count consumed budget at the same point
                # as supervised steps, even when a later update is discarded.
                effective_target_budget+=float(batch.get('_loss_normalizer',batch['history_kind'].size(0)*model.config.rollout_steps))
                sequences+=batch['history_kind'].size(0)
                tick=time.perf_counter();b=to_device_batch(batch,args.device,non_blocking=args.device.startswith('cuda'))
                timing['transfer_sec']+=time.perf_counter()-tick;tick=time.perf_counter()
                window_start=((bi-1)//args.grad_accumulation)*args.grad_accumulation
                divisor=min(args.grad_accumulation,batches_per_epoch-window_start)
                if not symbolic and model.config.phrase_context_policy!='off':
                    # Train both the default disabled setting and the optional
                    # acoustic continuity control. Keep one strength per row,
                    # never independently resample it at each event.
                    strength=torch.rand(b['history_kind'].size(0),device=args.device)
                    enabled=torch.rand_like(strength)<.25
                    b['audio_tracking_strength']=strength*enabled
                with _autocast(args):out=model(b,symbolic_only=symbolic);loss,metrics=model.loss(out,b)
                timing_queries+=int(out.get('timing_query_evaluated',b['history_kind'].numel()))
                timing_queries_total+=int(out.get('timing_query_total',b['history_kind'].numel()))
                backward_loss=loss/divisor
                window_finite=window_finite & _forward_finite_flags(out,loss,backward_loss)
                backward_loss.backward()
                combined_metrics=dict(metrics)
                for name,value in _exit_coverage(b).items():
                    combined_metrics[name]=combined_metrics[name]+value if name in combined_metrics else value
                window_metrics.add(combined_metrics);window_batches+=1
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
                        last_optimizer_unique_charts=len(window_chart_ids)
                        optimizer_chart_counts.append(last_optimizer_unique_charts)
                        _assert_model_finite(model,finite_items)
                        totals.add(window_metrics);n+=window_batches;last_norm=norm.detach()
                    else:
                        if episode_backup is not None:restore_recovery_state(model,episode_backup)
                        skipped_windows+=1
                        _emit('composer_warning',epoch=epoch,batch=bi,message='non-finite accumulation window skipped',
                              finite=dict(zip(('outputs','loss','backward_loss','grad_norm'),health)))
                    optimizer.zero_grad(set_to_none=True)
                    window_finite=torch.ones(3,dtype=torch.bool,device=args.device)
                    window_chart_ids.clear()
                    window_metrics=MetricTotals();window_batches=0
                timing['compute_sec']+=time.perf_counter()-tick
                stopping=updated and stop_at_update is not None and steps>=stop_at_update
                checkpoint_due=updated and (stopping or (args.checkpoint_interval_steps and steps%args.checkpoint_interval_steps==0))
                report_due=stopping or bi==skip+1 or bi%args.log_interval==0 or bi==batches_per_epoch
                if checkpoint_due or report_due:
                    avg=_averages(totals,n)
                if checkpoint_due:persist(epoch,bi,bi==batches_per_epoch,avg)
                if report_due:
                    elapsed=time.perf_counter()-started
                    _emit('composer_progress',epoch=epoch,batch=bi,total=batches_per_epoch,optimizer_steps=steps,
                        elapsed_sec=elapsed,samples_per_sec=sequences/max(elapsed,1e-6),
                        supervised_steps_per_sec=supervised/max(elapsed,1e-6),
                        supervised_steps=supervised,metric_batches=n,timing=timing,skipped_windows=skipped_windows,
                        optimizer_chart_diversity=dict(policy=args.batch_order_policy,last=last_optimizer_unique_charts,
                            first_100_optimizer_steps=optimizer_chart_counts[:100],
                            mean=float(np.mean(optimizer_chart_counts)) if optimizer_chart_counts else 0.,
                            min=min(optimizer_chart_counts,default=0),max=max(optimizer_chart_counts,default=0)),
                        effective_target_budget=effective_target_budget,
                        effective_targets_per_sec=effective_target_budget/max(elapsed,1e-6),
                        effective_target_budget_scope='consumed_authored_batches_current_epoch_execution_since_resume',
                        prefetch=loader.stats() if hasattr(loader,'stats') else {'mode':'off'},
                        cache=(loader.cache if isinstance(loader,_ProcessPrepared) else ds.cache_stats() if hasattr(ds,'cache_stats') else {}),
                        cache_scope='worker_process' if isinstance(loader,_ProcessPrepared) else 'training_process',
                        exit_coverage={key:value for key,value in totals.as_floats().items() if key.startswith('exit_')},
                        augmentation=augmentation_totals,
                        timing_queries=dict(evaluated=timing_queries,total=timing_queries_total),
                        cuda_memory=_cuda_memory(args.device),
                        last_gradient_norm=float(last_norm) if last_norm is not None else None,
                        learning_rate=optimizer.param_groups[0]['lr'],stage='symbolic' if symbolic else 'audio_conditioned',**avg)
                if stopping:
                    _emit('composer_bounded_stop',epoch=epoch,batch=bi,optimizer_steps=steps,
                          reason='requested_additional_update_budget',checkpoint=str(directory/'composer_last.pt'))
                    _shutdown_process_loader(ds)
                    return 0
        avg=_averages(totals,n)
        training_elapsed=time.perf_counter()-started
        skip=0
        improvement=None
        if dev is not None:
            validation=evaluate(model,dev,args);validation.update(epoch=epoch,optimizer_steps=steps)
            (directory/'validation_latest.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding='utf-8')
            _emit('composer_validation',**validation)
            improvement=_validation_improvement(validation,best_validation,epoch,steps)
            if improvement:best_validation=improvement
        persist(epoch,batches_per_epoch,True,avg)
        if improvement:
            _retain_best_files(directory)
            _emit('composer_best_checkpoint',**best_validation,path=str(directory/'composer_best_inference.pt'))
        _emit('composer_epoch',epoch=epoch,optimizer_steps=steps,elapsed_sec=time.perf_counter()-started,
            effective_target_budget=effective_target_budget,
            effective_targets_per_sec=effective_target_budget/max(training_elapsed,1e-6),
            effective_target_budget_elapsed_sec=training_elapsed,
            effective_target_budget_scope='consumed_authored_batches_current_epoch_execution_since_resume',
            exit_coverage={key:value for key,value in totals.as_floats().items() if key.startswith('exit_')},
            augmentation=augmentation_totals,**avg)
        if dev is not None and args.generation_eval_songs and epoch%args.generation_eval_interval_epochs==0:
            from .stream_generate import evaluate_panel
            with _preserve_rng(args.device):
                panel=evaluate_panel(directory/'composer_inference.pt',dev,directory/f'generation_eval_epoch_{epoch}',
                    count=args.generation_eval_songs,seeds=[int(s) for s in args.generation_eval_seeds.split(',')],device=args.device,
                    temperature=args.generation_eval_temperature,generation_settings=generation_settings)
            _emit('composer_generation_validation',**panel)
    _shutdown_process_loader(ds)
    return 0

def main(argv=None):
    args=parser().parse_args(argv)
    with training_output_lock(args.checkpoint_dir):return _train(args)

if __name__=='__main__':raise SystemExit(main())
