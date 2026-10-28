"""Train chart_composer_v1 on structured authored events."""
from __future__ import annotations
import argparse,json,math,os,random,time
from collections import deque
from contextlib import contextmanager,nullcontext
from itertools import islice
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from maichart.form.layout_prefetch import LayoutBatchPrefetch
from maichart.form.cuda_memory import CudaCacheGuard
from maichart.form.training_runtime import memory_snapshot

from . import FAMILY,TRAINING_SEMANTICS,GROUNDED_TRAINING_SEMANTICS
from .data import ComposerDataset,collate_composer
from .candidates import COMPOSER_AUDIO_FEATURE_POLICY
from .model import (ChartComposer,ComposerConfig,build_model,GROUNDED_PROFILE,WORKLOAD_PROFILES,
                    resolve_history_anchor_policy,resolve_composer_policy,COMPOSER_POLICY_FIELDS,HISTORY_ANCHOR_POLICIES)
from .events import HEAD_TIMING_DIM,head_timing_dim_for_policy
from .checkpoint import save,load,migrate_music_context,migrate_audio_grounded,structure_training_semantics
from .runtime_memory import GIB,host_memory,memory_gib
from .objectives import (ACTION_OBJECTIVES, LEGACY_ACTION_OBJECTIVE, RECOVERY_ACTION_OBJECTIVE, action_objective, recovery_objective,
                         resolve_action_objective, training_semantics_for_objective)
from .sampling_contract import RECOVERY_SAMPLING_POLICIES,resolve_recovery_sampling,recovery_sampling_settings
from . import ACTION_SAMPLING_POLICY,LEGACY_ACTION_SAMPLING_POLICY
from .training_defaults import TRAINING_DEFAULTS as DEFAULTS
from .learning_rate import resolve_plan as resolve_lr_plan, learning_rate


class RemainingBatches:
    """Skip consumed indices without loading their audio/history again."""
    def __init__(self,batches,skip,stop=None): self.batches,self.skip,self.stop=batches,skip,stop
    def __iter__(self): return islice(iter(self.batches),self.skip,self.stop)
    def __len__(self): return max(0,min(len(self.batches),self.stop if self.stop is not None else len(self.batches))-self.skip)


class SongLocalBatchSampler:
    """Shuffle samples while keeping a small active set of source songs.

    Global sample shuffling maximizes mmap page churn: nearly every batch asks
    for unrelated audio banks. This sampler shuffles song order and each
    song's events, then interleaves a few active songs per batch. It preserves
    every sample exactly once per epoch while allowing the bank/context caches
    to reuse nearby files.
    """
    def __init__(self, dataset, batch_size, generator, locality=4):
        self.batch_size=int(batch_size); self.generator=generator
        self.locality=max(1,int(locality))
        grouped={}
        for index,sample in enumerate(dataset.samples):
            grouped.setdefault(sample.song_index,[]).append(index)
        self.groups=list(grouped.values())

    def __len__(self):
        total=sum(len(group) for group in self.groups)
        return (total+self.batch_size-1)//self.batch_size

    def __iter__(self):
        if not self.groups:return
        group_order=torch.randperm(len(self.groups),generator=self.generator).tolist()
        states=[]
        for gid in group_order:
            source=self.groups[gid]
            order=torch.randperm(len(source),generator=self.generator).tolist()
            states.append({'values':[source[i] for i in order],'pos':0})
        active=deque()
        next_group=0
        while next_group<len(states) and len(active)<self.locality:
            active.append(states[next_group]); next_group+=1
        while active:
            batch=[]
            for state in list(active):
                if len(batch)>=self.batch_size: break
                left=len(state['values'])-state['pos']
                take=min(max(1,self.batch_size//max(1,self.locality)),left,self.batch_size-len(batch))
                batch.extend(state['values'][state['pos']:state['pos']+take])
                state['pos']+=take
            while len(batch)<self.batch_size and active:
                state=active[0]
                left=len(state['values'])-state['pos']
                if left<=0:
                    active.popleft()
                    if next_group<len(states):
                        active.append(states[next_group]);next_group+=1
                    continue
                take=min(self.batch_size-len(batch),left)
                batch.extend(state['values'][state['pos']:state['pos']+take])
                state['pos']+=take
            for state in list(active):
                if state['pos']>=len(state['values']):
                    active.remove(state)
                    if next_group<len(states):
                        active.append(states[next_group]);next_group+=1
            if batch:yield batch


class RebatchedSampler:
    """Regroup an existing epoch's exact sample sequence without resampling."""
    def __init__(self,source,batch_size,total_samples):
        self.source,self.batch_size,self.total_samples=source,int(batch_size),int(total_samples)
        if self.batch_size<1:raise ValueError('Rebatch size must be positive')

    def __len__(self):
        return (self.total_samples+self.batch_size-1)//self.batch_size

    def __iter__(self):
        batch=[]
        for original in self.source:
            for index in original:
                batch.append(index)
                if len(batch)==self.batch_size:
                    yield batch;batch=[]
        if batch:yield batch


def resume_batch_layout(args,old,total_samples):
    """Resolve saved epoch sample order and safe explicit runtime rebatching.

    No iterator or RNG is consumed here. Reconstruct the original sampler only
    after DataLoader has consumed its ordinary generator base seed.
    """
    state=(old or {}).get('training_state') or {}
    if not args.resume or state.get('epoch_complete',True):
        return 0,args.batch_size,0
    saved=(old or {}).get('args') or {}
    previous_size=int(saved.get('batch_size',args.batch_size))
    previous_accum=int(saved.get('grad_accumulation',1))
    previous_batch=int(state.get('batch_in_epoch',0))
    cursor=int(state.get('sample_in_epoch',min(previous_batch*previous_size,total_samples)))
    sampler_size=int(state.get('sampler_batch_size',previous_size))
    if not 0<=cursor<total_samples or sampler_size<1 or cursor!=previous_batch*previous_size:
        raise ValueError('Mid-epoch resume has an inconsistent sample cursor')
    changed=previous_size!=args.batch_size or previous_accum!=args.grad_accumulation
    if changed:
        if not args.rebatch_resume:
            raise ValueError('Mid-epoch resume requires the saved batch_size/grad_accumulation; use --rebatch-resume explicitly')
        if args.max_steps or saved.get('max_steps',0):
            raise ValueError('Rebatch resume requires full epochs (max_steps=0)')
        if previous_size*previous_accum!=args.batch_size*args.grad_accumulation:
            raise ValueError('Rebatch resume must preserve the effective batch size')
        if previous_batch%previous_accum or cursor%args.batch_size or (cursor//args.batch_size)%args.grad_accumulation:
            raise ValueError('Rebatch resume requires a complete optimizer and sample boundary')
    if (changed or sampler_size!=previous_size) and resume_loader_rng_state(state) is None:
        raise ValueError('Rebatch resume requires the saved epoch loader RNG state')
    return cursor//args.batch_size,sampler_size,cursor


def resume_loader_rng_state(state):
    value=state.get('loader_generator_state')
    if isinstance(value,dict):
        size=value.get('sampler_batch_size',0)
        if (value.get('schema')!='composer_epoch_sampler_v1' or not isinstance(size,int) or size<1
                or size!=state.get('sampler_batch_size') or not torch.is_tensor(value.get('state'))):
            raise ValueError('Unsupported or inconsistent Composer epoch sampler state')
        value=value['state']
    if value is not None and (not torch.is_tensor(value) or value.dtype!=torch.uint8 or value.ndim!=1):
        raise ValueError('Composer loader RNG state must be a byte tensor')
    return value


def checkpoint_loader_rng_state(value,sampler_size,batch_size,epoch_complete):
    # A legacy reader would silently reconstruct the wrong sampler at 64.
    # Its set_state(dict) instead fails before updates; generation drops this
    # training-only state. Completed epochs remain legacy-compatible tensors.
    return ({'schema':'composer_epoch_sampler_v1','sampler_batch_size':sampler_size,'state':value}
            if not epoch_complete and sampler_size!=batch_size else value)


def _cuda_failure_diagnostics(error,device):
    """Read host state first; a failed CUDA context need not accept more work."""
    info={'error':str(error),'host_memory':memory_gib(host_memory()),
          'torch_version':torch.__version__,'cuda_build':torch.version.cuda}
    try:
        info['cuda_memory']={'allocated_gib':torch.cuda.memory_allocated(device)/1024**3,
            'reserved_gib':torch.cuda.memory_reserved(device)/1024**3}
    except RuntimeError:
        pass
    print('composer_cuda_failure '+json.dumps(info),flush=True)
    return RuntimeError('Composer CUDA execution failed. See composer_cuda_failure for host commit and CUDA memory. '
        'On Windows, free GPU memory alone does not rule out host memory pressure; close unused memory-heavy '
        'processes or increase available system commit before restarting. No automatic precision/batch change was applied.')


@contextmanager
def _cuda_execution(device):
    try:
        yield
    except RuntimeError as error:
        if device.type=='cuda' and ('CUDA' in str(error) or 'CUBLAS' in str(error)):
            raise _cuda_failure_diagnostics(error,device) from error
        raise


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--checkpoint-dir',required=True)
    p.add_argument('--cache-dir',default='cache/chart_composer_v1');p.add_argument('--epochs',type=int,default=DEFAULTS['epochs'])
    p.add_argument('--validation-manifest')
    p.add_argument('--holdout-fraction',type=float,default=.05,
        help='fresh runs: group-disjoint holdout fraction; 0 explicitly disables automatic split')
    p.add_argument('--validation-max-batches',type=int,default=32,
        help='deterministic authored-history evaluation per epoch; 0 evaluates all held-out batches')
    p.add_argument('--validation-batch-size',type=int,default=8)
    p.add_argument('--generation-eval-songs',type=int,default=2,
        help='structure-enabled runs: fixed held-out songs per epoch; 0 disables closed-loop evaluation')
    p.add_argument('--generation-eval-interval-epochs',type=int,default=1)
    p.add_argument('--generation-eval-seeds',default='0,1')
    p.add_argument('--structure-backend',choices=('auto','native','beat_this'),default='auto')
    p.add_argument('--structure-dropout',type=float,default=.15)
    p.add_argument('--condition-mask-prob',type=float,default=None,
        help='hide target-derived condition controls coherently, retaining level; fresh default 0, continuation inherits')
    p.add_argument('--batch-size',type=int,default=DEFAULTS['batch_size']);p.add_argument('--lr',type=float,default=DEFAULTS['lr'])
    p.add_argument('--lr-schedule',choices=('auto','constant','cosine'),default='auto',
        help='auto preserves constant fresh/legacy training and inherits a resumed schedule')
    p.add_argument('--warmup-steps',type=int,default=None,
        help='successful optimizer updates; cosine defaults to at most 500, constant to 0')
    p.add_argument('--min-lr-ratio',type=float,default=None,
        help='cosine final/base rate; defaults to 0.1 (constant: 1.0)')
    p.add_argument('--d-model',type=int,default=DEFAULTS['d_model']);p.add_argument('--layers',type=int,default=DEFAULTS['layers'])
    p.add_argument('--nhead',type=int,default=DEFAULTS['nhead']);p.add_argument('--dim-feedforward',type=int,default=DEFAULTS['dim_feedforward'])
    p.add_argument('--dropout',type=float,default=.1);p.add_argument('--history-events',type=int,default=128)
    p.add_argument('--candidate-count',type=int,default=96);p.add_argument('--action-slots',type=int,default=8)
    p.add_argument('--device',default='auto');p.add_argument('--precision',choices=('auto','fp32','bf16','fp16'),default='auto')
    p.add_argument('--seed',type=int,default=0);p.add_argument('--limit-charts',type=int,default=0)
    p.add_argument('--prepare-workers',type=int,default=4);p.add_argument('--audio-cache-gb',type=float,default=4.)
    p.add_argument('--checkpoint-interval-steps',type=int,default=500)
    p.add_argument('--archive-checkpoints',choices=('auto','off','geometric'),default='auto',
        help='keep immutable inference snapshots at interval * 1,2,4,8... for regression comparisons')
    continuation=p.add_mutually_exclusive_group()
    continuation.add_argument('--resume');continuation.add_argument('--init-checkpoint')
    p.add_argument('--rebatch-resume',action='store_true',
        help='explicitly change resume microbatch/accumulation while preserving effective batch and current epoch sample sequence')
    p.add_argument('--grad-accumulation',type=int,default=DEFAULTS['grad_accumulation'])
    p.add_argument('--prefetch-batches',type=int,choices=range(0,17),default=2)
    p.add_argument('--sample-workers',type=int,choices=range(1,9),default=DEFAULTS['sample_workers'],
                   help='parallel CPU sample preparation threads; CUDA work stays on the main thread')
    p.add_argument('--cpu-threads',type=int,default=DEFAULTS['cpu_threads'],
                   help='PyTorch CPU threads, 0 keeps environment settings; CUDA runs can benchmark 2-4')
    p.add_argument('--sample-locality',type=int,choices=range(1,9),default=4,
                   help='songs interleaved per batch; fewer improves locality but increases same-song gradient correlation')
    p.add_argument('--log-interval',type=int,default=25)
    p.add_argument('--max-steps',type=int,default=0,help='maximum batches per epoch; 0 uses every batch')
    p.add_argument('--optimizer-implementation',choices=('auto','standard','foreach','fused'),default='auto')
    p.add_argument('--export-inference',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--symbolic-dropout',type=float,default=.15,
                   help='probability of coherently hiding chart history/state/ledger, retaining audio and targets')
    p.add_argument('--audio-pointer-weight',type=float,default=.15,
        help='audio-only auxiliary weight: next-position plus action supervision in audio_grounded_v3')
    p.add_argument('--action-objective-policy',choices=('auto',*sorted(ACTION_OBJECTIVES)),default='auto',
        help='auto uses joint configuration recovery for fresh grounded models and inherits existing parents')
    p.add_argument('--recovery-sampling-policy',choices=('auto',*sorted(RECOVERY_SAMPLING_POLICIES)),default='auto',
        help='auto inherits on resume; new stages use grouped sampling and retained generated histories with recorded temperatures')
    p.add_argument('--recovery-pointer-temperature',type=float,default=1.)
    p.add_argument('--recovery-action-temperature',type=float,default=.85)
    p.add_argument('--profile-steps',type=int,default=0,
                   help='record CUDA H2D/forward/backward/update timings for first N consumed batches; 0 disables')
    p.add_argument('--architecture-profile',choices=('music_context_v2',GROUNDED_PROFILE),default=GROUNDED_PROFILE)
    p.add_argument('--history-anchor-policy',choices=('auto',*sorted(HISTORY_ANCHOR_POLICIES)),default='auto',
                   help='auto inherits an existing model policy; fresh models use parser-backed angular history anchors')
    for name,(choices,_,_) in COMPOSER_POLICY_FIELDS.items():
        p.add_argument('--'+name.replace('_','-'),choices=('auto',*sorted(choices)),default='auto')
    p.add_argument('--motif-max-events',type=int,default=16)
    p.add_argument('--touch-layout-policy',choices=('auto','open','legacy_two_heads'),default='auto')
    p.add_argument('--lens-width',type=int,default=64)
    p.add_argument('--lens-chunk-size',type=int,default=32)
    p.add_argument('--history-residual-bound',type=float,default=1.)
    p.add_argument('--history-memory-rms-bound',type=float,default=8.,
                   help='per-event history-memory RMS ceiling for rms_cap_v1; existing models inherit their value')
    p.add_argument('--history-dropout',type=float,default=.3)
    p.add_argument('--ledger-dropout',type=float,default=.5)
    p.add_argument('--placement-weight',type=float,default=.15)
    p.add_argument('--planner-weight',type=float,default=.2)
    p.add_argument('--recovery-interval',type=int,default=DEFAULTS['recovery_interval'],help='optimizer updates between short recovery batches; 0 disables')
    p.add_argument('--recovery-samples',type=int,default=2)
    p.add_argument('--recovery-steps',type=int,default=2,choices=(2,3,4))
    p.add_argument('--recovery-weight',type=float,default=.1)
    return p


def _device(value):
    if value=='auto':return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(value)


def _precision(value,device):
    if value=='auto':return 'bf16' if device.type=='cuda' and torch.cuda.is_bf16_supported() else 'fp32'
    if device.type!='cuda' and value!='fp32':raise ValueError('CPU composer training requires fp32')
    return value


def _loss(output,batch,atom_vocab_size, *, audio_pointer_weight=.15,placement_weight=.15,planner_weight=.2,
          action_objective_policy=LEGACY_ACTION_OBJECTIVE):
    weights=batch.get('sample_weight',torch.ones_like(batch['eos_target']))
    transitions=batch.get('transition_weight',torch.ones_like(weights))
    def reduce(values, factors=weights):
        return (values * factors).sum()/factors.sum().clamp_min(1e-6)
    pointer_weights=weights*transitions*batch.get('pointer_weight',torch.ones_like(weights))
    pointer=reduce(F.cross_entropy(output['candidate_logits'],batch['candidate_target'],reduction='none'),
                   pointer_weights)
    target=batch['candidate_target']
    if output['action_logits'].ndim==3 or 'action_slot_indices' in output:
        action_logits,future,eos=output['action_logits'],output['future'],output['eos_logits']
    else:
        rows=torch.arange(target.size(0),device=target.device)
        action_logits=output['action_logits'][rows,target]
        future=output['future'][rows,target]
        eos=output['eos_logits'][rows,target]
    action_targets=batch['action_target']
    action_weights=weights*transitions*batch.get('action_weight',torch.ones_like(weights))
    if action_logits.size(-1)!=atom_vocab_size:
        raise ValueError('Action logits do not match atom_vocab_size')
    action,action_metrics=action_objective(action_logits,action_targets,action_weights,policy=action_objective_policy,
        slot_indices=output.get('action_slot_indices'))
    audio_action_loss=pointer.new_zeros(())
    audio_action_metrics={}
    if 'audio_action_logits' in output:
        audio_action_logits=output['audio_action_logits']
        if audio_action_logits.ndim>3:
            rows=torch.arange(target.size(0),device=target.device)
            audio_action_logits=audio_action_logits[rows,target]
        audio_action_loss,audio_action_metrics=action_objective(
            audio_action_logits,action_targets,action_weights,policy=action_objective_policy,
        slot_indices=output.get('action_slot_indices'))
    eos_target=batch['eos_target']
    eos_weights=weights*batch.get('eos_weight',torch.ones_like(weights))
    positives=(eos_target*eos_weights).sum().clamp_min(1.)
    negatives=((1.-eos_target)*eos_weights).sum().clamp_min(1.)
    eos_loss=reduce(F.binary_cross_entropy_with_logits(eos,eos_target,
        pos_weight=(negatives/positives).clamp(1.,64.),reduction='none'),eos_weights)
    future_weights=weights[:,None]*batch.get('future_weight',torch.ones_like(future))
    future_loss=(F.smooth_l1_loss(future,batch['future_target'],reduction='none')*future_weights).sum()/future_weights.sum().clamp_min(1e-6)
    workload_loss=pointer.new_zeros(())
    curve_slope_loss=pointer.new_zeros(())
    workload_rate_mae=pointer.new_zeros(())
    audio_pointer_loss=pointer.new_zeros(())
    trailing_loss=pointer.new_zeros(())
    if 'audio_candidate_logits' in output:
        audio_pointer_loss=reduce(F.cross_entropy(output['audio_candidate_logits'], target, reduction='none'), pointer_weights)
    if 'workload' in output and 'workload_target' in batch:
        prediction=output['workload'].float()
        reference=batch['workload_target']
        mask=batch['workload_weight']*weights[:,None]
        residual=F.smooth_l1_loss(prediction,reference,reduction='none').mean(-1)
        workload_loss=(residual*mask).sum()/mask.sum().clamp_min(1e-6)
        workload_rate_mae=((prediction[...,:2]-reference[...,:2]).abs().mean(-1)*mask).sum()/mask.sum().clamp_min(1e-6)
        # Match authored changes, rather than penalizing every change as noise.
        slope_prediction,slope_reference,slope_mask=prediction,reference,mask
        slope_seconds=batch['workload_seconds']
        count=output['candidate_logits'].size(1)
        if 'audio_candidate_logits' in output and prediction.size(1)>count:
            # Uniform-only v2 probes should remain neighbours for change
            # supervision; zero-weight candidate rows must not split them.
            slope_prediction,slope_reference,slope_mask=prediction[:,count:],reference[:,count:],mask[:,count:]
            slope_seconds=slope_seconds[:,count:]
        order=slope_seconds.argsort(-1)
        ordered_prediction=slope_prediction.gather(1,order[...,None].expand(-1,-1,5))
        ordered_reference=slope_reference.gather(1,order[...,None].expand(-1,-1,5))
        ordered_mask=slope_mask.gather(1,order)
        pair_weight=torch.minimum(ordered_mask[:,1:],ordered_mask[:,:-1])
        changes=F.smooth_l1_loss(torch.diff(ordered_prediction,dim=1),
            torch.diff(ordered_reference,dim=1),reduction='none').mean(-1)
        curve_slope_loss=(changes*pair_weight).sum()/pair_weight.sum().clamp_min(1e-6)
    if 'trailing_workload' in output and 'trailing_target' in batch:
        mask=batch['trailing_weight']*weights[:,None,None]
        errors=F.smooth_l1_loss(output['trailing_workload'].float(),batch['trailing_target'],reduction='none')
        trailing_loss=(errors*mask).sum()/mask.sum().clamp_min(1e-6)
    placement_loss=pointer.new_zeros(())
    plan_loss=pointer.new_zeros(())
    if 'placement_logits' in output:
        mask=batch['placement_weight']*weights[:,None,None]
        errors=F.binary_cross_entropy_with_logits(output['placement_logits'].float(),batch['placement_target'],reduction='none')
        placement_loss=(errors*mask).sum()/mask.sum().clamp_min(1e-6)
    if 'plan' in output:
        mask=batch['planner_weight']*weights[:,None]
        errors=F.smooth_l1_loss(output['plan'].float(),batch['planner_target'],reduction='none').mean(-1)
        plan_loss=(errors*mask).sum()/mask.sum().clamp_min(1e-6)
    total=pointer+action+.05*eos_loss+.2*future_loss+.4*workload_loss+.05*curve_slope_loss+audio_pointer_weight*(audio_pointer_loss+audio_action_loss)+.2*trailing_loss+placement_weight*placement_loss+planner_weight*plan_loss
    return total,{'loss':total.detach(),
        **{'action_'+name:value.detach() for name,value in action_metrics.items()},
        **{'audio_action_'+name:value.detach() for name,value in audio_action_metrics.items()},
        'pointer_loss':pointer.detach(),'action_loss':action.detach(),'eos_loss':eos_loss.detach(),
        'future_loss':future_loss.detach(),'workload_loss':workload_loss.detach(),
        'curve_slope_loss':curve_slope_loss.detach(),'workload_rate_mae':workload_rate_mae.detach(),
        'audio_pointer_loss':audio_pointer_loss.detach(),'audio_action_loss':audio_action_loss.detach(),
        'trailing_workload_loss':trailing_loss.detach(),
        'placement_loss':placement_loss.detach(),'planner_loss':plan_loss.detach(),
        'pointer_accuracy':(output['candidate_logits'].argmax(-1)==batch['candidate_target']).float().mean().detach()}


def summarize_metrics(totals, batches):
    """Keep historical batch means; pool conditional CEs by decision weight."""
    summary={key:float(value)/max(1,batches) for key,value in totals.items()
             if not key.endswith(('_ce_sum','_ce_weight'))}
    for key,numerator in totals.items():
        if not key.endswith('_ce_sum'): continue
        metric=key[:-4]
        mass=float(totals.get(metric+'_weight',0.))
        summary[metric]=float(numerator)/max(mass,1e-6)
        summary[metric+'_weight']=mass
    return summary


def _nonfinite_tensors(mapping,*,healthy_fast_path=False):
    items=[(key,value) for key,value in mapping.items() if torch.is_tensor(value)]
    if not items:return []
    # Infinity norms avoid the per-element isfinite temporaries and several
    # small kernels per parameter. Unlike L2, this cannot overflow on large
    # finite values. Keep the original predicate for unsupported inputs.
    flags=[True]*len(items)
    groups={}
    pending={}
    with torch.no_grad():
        for index,(_,value) in enumerate(items):
            if value.numel()==0:continue
            if value.dtype in (torch.bool,torch.uint8,torch.int8,torch.int16,torch.int32,torch.int64):continue
            if (value.device.type=='cuda' and value.layout==torch.strided
                    and value.dtype in (torch.float16,torch.bfloat16,torch.float32,torch.float64)):
                groups.setdefault((value.device,value.dtype),[]).append((index,value))
            else:
                pending.setdefault(value.device,[]).append((index,torch.isfinite(value).all()))
        for (device,_),group in groups.items():
            try:
                norms=torch._foreach_norm([value for _,value in group],float('inf'))
                checks=torch.isfinite(torch.stack(norms)).unbind()
            except (RuntimeError,NotImplementedError):
                checks=[torch.isfinite(value).all() for _,value in group]
            pending.setdefault(device,[]).extend((index,check) for (index,_),check in zip(group,checks))
        for checks in pending.values():
            packed=torch.stack([check for _,check in checks])
            if healthy_fast_path and bool(packed.all()):
                continue
            values=packed.cpu().tolist()
            for (index,_),finite in zip(checks,values):flags[index]=finite
    return [key for (key,_),finite in zip(items,flags) if not finite]


_LOSS_FINITE_KEY=object()
_BACKWARD_FINITE_KEY=object()


def _forward_finite_check(output,loss,backward_loss):
    """One healthy-path host decision per device for outputs and both losses.

    Detailed offending keys are fetched only on failure. All original output,
    loss and backward-loss checks remain active before backward is allowed.
    """
    checks=dict(output)
    checks[_LOSS_FINITE_KEY]=loss
    checks[_BACKWARD_FINITE_KEY]=backward_loss
    bad=_nonfinite_tensors(checks,healthy_fast_path=True)
    return ([key for key in bad if key is not _LOSS_FINITE_KEY and key is not _BACKWARD_FINITE_KEY],
            _LOSS_FINITE_KEY not in bad,_BACKWARD_FINITE_KEY not in bad)


def _forward_finite_flags(output,loss,backward_loss):
    """Device-only status; keep unused outputs covered without a host wait."""
    with torch.no_grad():
        checks=[]
        groups={}
        for value in output.values():
            if not torch.is_tensor(value) or not (value.is_floating_point() or value.is_complex()):continue
            if value.numel()==0:continue
            value=value.detach()
            if value.layout==torch.strided and not value.is_complex():
                groups.setdefault((value.device,value.dtype),[]).append(value)
            else:checks.append(torch.isfinite(value).all().to(loss.device))
        for values in groups.values():
            try: norms=torch._foreach_norm(values,float('inf'))
            except (RuntimeError,NotImplementedError):
                checks.extend(torch.isfinite(value).all().to(loss.device) for value in values)
            else:checks.append(torch.isfinite(torch.stack(norms)).all().to(loss.device))
        output_finite=torch.stack(checks).all() if checks else torch.ones((),dtype=torch.bool,device=loss.device)
        return torch.stack((output_finite,torch.isfinite(loss.detach()).all(),
                            torch.isfinite(backward_loss.detach()).all()))


def _model_finite_items(model):
    # Build after device placement. Registered tensors are stable throughout an
    # optimizer run; state_dict construction traverses the full Python module tree.
    return {name:value for name,value in list(model.named_parameters())+list(model.named_buffers())
            if value.is_floating_point() or value.is_complex()}


def _window_update_health(window_finite,grad_norm,model,episode_backup=None):
    """One host read at the update boundary, with transactional episode state."""
    health=torch.cat((window_finite,torch.isfinite(grad_norm.detach()).reshape(1))).cpu().tolist()
    if not all(health) and episode_backup is not None:
        from .recovery import restore_recovery_episodes
        restore_recovery_episodes(model,episode_backup)
    return health


def _assert_model_finite(model,items=None):
    bad = _nonfinite_tensors(model.state_dict() if items is None else items,healthy_fast_path=True)
    if bad:
        raise FloatingPointError('Composer model became non-finite: ' + ', '.join(bad[:8]))


def _forward_batch(model,tensors,*,decode_actions=True,packed_actions=False,**overrides):
    extra={key:tensors[key] for key in ('head_timing','workload_audio','ledger','interval_audio',
        'trailing_audio','audio_lens','audio_lens_mask','planner_audio','planner_mask','planner_indices','planner_mix',
        'motif_ids','motif_relative_beats','motif_mask','selected_action_state','structure_features','structure_transition',
        'planner_unique_indices','planner_inverse','packed_slot_indices','packed_first_indices') if key in tensors}
    if getattr(model.config,'architecture_profile',None)=='audio_grounded_v3':extra['packed_actions']=packed_actions
    extra.update(overrides)
    return model(tensors['candidate_features'],tensors['history'],tensors['condition'],
        action_state=tensors['action_state'],history_features=tensors['history_features'],
        candidate_mask=tensors['candidate_padding_mask'],history_padding_mask=tensors['history_padding_mask'],
        candidate_indices=tensors['candidate_target'],action_targets=tensors['action_target'],decode_actions=decode_actions,**extra)


def recovery_batch(dataset,model,args,device,global_step):
    from .recovery import RecoveryBuilder
    from .generate import (_sample_actions,_head_metadata,_active_constraints,_truncate_probabilities,
                           _touch_head_sets,_candidate_has_capacity,_grounded_pointer_probabilities,_candidate_playability)
    from .action_guidance import action_logits_with_touch_guard
    from .guard import first_rejected_prefix
    from .events import event_metadata
    from maichart.generation.event_cli import _NoteGrammar
    from maichart.event_timing import EventPoint
    atoms=dataset.codec_atoms;grammar=dataset.action_grammar
    touch_policy=getattr(args,'touch_layout_policy','legacy_two_heads')
    metadata=_head_metadata(atoms,grammar,touch_policy)
    from .contacts import ContactBudget
    contact_policy=getattr(args,'contact_policy','off')
    contact_budget=ContactBudget(grammar) if contact_policy!='off' else None
    touch_heads=_touch_head_sets(atoms,grammar)
    policy=getattr(args,'recovery_sampling_policy',ACTION_SAMPLING_POLICY)
    settings=recovery_sampling_settings(policy,getattr(args,'recovery_pointer_temperature',1.),
                                        getattr(args,'recovery_action_temperature',.85))
    aligned=settings['pointer_sampler']=='grouped'
    touch_mask=torch.tensor([False]+[atom[0] in 'ABCDE' for atom in atoms],device=device,dtype=torch.bool)
    rng=torch.Generator(device=device).manual_seed(args.seed+global_step+191)
    planner_cache={}
    def propose(item,index):
        candidates=item['_candidate_beats']
        constraints=[_active_constraints(item['_action_index'],beat,item['_tempo'],item['_first'],
            clock=item['_clock'],touch_layout_policy=touch_policy) for beat in candidates]
        playable=[not bool(item['candidate_mask'][i]) and _candidate_has_capacity(held,blocked,touch_policy,touch_heads)
                  for i,(held,blocked,_) in enumerate(constraints)]
        if aligned:
            previous=item['_prefix_points'][-1].beat if item['_prefix_points'] else None
            playable=_candidate_playability(candidates,playable,previous,dataset.candidate_policy)
        # Recovery features have a fixed padded candidate width; physical
        # proposals contain only actual candidates near the audio boundary.
        physical_mask=torch.ones_like(item['candidate_mask'])
        physical_mask[:len(candidates)]=torch.tensor([not value for value in playable])
        item['candidate_mask']=item['candidate_mask'] | physical_mask
        if bool(item['candidate_mask'].all()):return None
        clean={k:v for k,v in item.items() if not k.startswith('_')}
        batch=collate_composer([clean])
        tensors={k:v.to(device) for k,v in batch.items() if torch.is_tensor(v)}
        song_id=int(item['song_index'])
        overrides={}
        if model.config.architecture_profile==GROUNDED_PROFILE and 'planner_audio' in tensors:
            if song_id not in planner_cache:
                planner_cache[song_id]=model.planner(tensors['planner_audio'],tensors['condition'],tensors['planner_mask'])
            overrides['planner_override']=planner_cache[song_id]
        result=_forward_batch(model,tensors,decode_actions=False,**overrides)
        logits=result['candidate_logits'][0].float().masked_fill(tensors['candidate_padding_mask'][0],float('-inf'))
        if not torch.isfinite(logits).any():return None
        guard=item['_proposal_guard']
        for _ in range(min(8,len(candidates))):
            if not torch.isfinite(logits).any():break
            pointer_probs=(_grounded_pointer_probabilities(logits,result['candidate_bins'][0],settings['pointer_temperature'])
                           if aligned else _truncate_probabilities(logits.softmax(-1),top_p=.92,min_p=.05,max_k=16))
            chosen=int(torch.multinomial(pointer_probs,1,generator=rng))
            beat=candidates[chosen]
            cand_sec=float(item['_clock'].at(beat))
            context=result['candidate_context'][0,chosen]
            audio_ctx=result['audio_action_context'][0,chosen] if 'audio_action_context' in result else None
            decode_extra={}
            if 'previous_heads' in result:decode_extra['previous_heads']=result['previous_heads']
            if model.config.action_state_policy=='selected_prefix_v3':
                from .context_features import selected_action_state
                decode_extra.update(selected_action_state=selected_action_state(item['_action_index'],len(item['_prefix_points']),beat)[None].to(device),
                    state_base=result['selected_state_base'],state_keep=result['selected_state_keep'])
            if '_motif_index' in item:
                from .context_features import motif_tensors
                memory=motif_tensors(item['_motif_index'],beat,len(item['_prefix_points']),
                    model.config.history_events,model.config.motif_max_events)
                decode_extra.update({k:v[None].to(device) for k,v in memory.items()})
            if decode_extra:
                decoded=model.decode_candidate(result['candidate_context'],torch.tensor([chosen],device=device),
                    project_actions=False,**decode_extra)
                context=decoded['action_context'][0]
            occupied,blocked,_=constraints[chosen]
            cooldown_blocked=set(blocked)|{pad for pad,prior in guard.last_head_seconds.items()
                if cand_sec-prior<guard.min_same_head_sec-1e-9}
            blocked_first=set()
            blocked_prefixes=set()
            previous_heads=result['previous_heads'][0] if 'previous_heads' in result else None
            guidance={}
            def decode_slot(slot,prefix,legal_mask):
                ids=[dataset.atom_to_id[x] for x in prefix]
                if aligned:
                    return action_logits_with_touch_guard(model,context,slot,ids,audio_context=audio_ctx,
                        enabled=settings['touch_history_guard'],touch_mask=touch_mask,diagnostics=guidance,previous_heads=previous_heads,
                        log_odds_bound=settings['touch_history_bound'],legal_mask=legal_mask,
                        legacy=not settings.get('touch_legal_mask',False))
                extra={} if previous_heads is None else {'previous_heads':previous_heads}
                return model.action_slot(context,slot,ids,audio_context=audio_ctx,**extra)
            for _ in range(6):
                guidance.clear()
                selected=_sample_actions(context.new_zeros(model.config.action_slots,len(atoms)+1),atoms,grammar,settings['action_temperature'],rng,
                    slot_mask_decoder=decode_slot,
                    contact_budget=contact_budget,contact_policy=contact_policy,
                    occupied=occupied,blocked=cooldown_blocked,blocked_first=blocked_first,head_metadata=metadata,
                    canonical_order=True,touch_layout_policy=touch_policy,blocked_prefixes=blocked_prefixes,
                    macro_group_ids=getattr(getattr(model,'conserved_action_head',None),'atom_group_id',None))
                if selected:
                    point=EventPoint(candidates[chosen],'/'.join(selected))
                    accepted=(guard.accepts(point,protect_reported_hard=True) if guidance.get('applied') else guard.accepts(point))
                    if accepted:return point
                    if aligned:
                        blocked_prefixes.add(first_rejected_prefix(guard,beat,selected,
                            protect_reported_hard=bool(guidance.get('applied'))))
                    else:blocked_first.add(dataset.atom_to_id[selected[0]])
                    if guard.last_reason=='repeated_head_too_soon':
                        for head in event_metadata('/'.join(selected)).heads:
                            pad=f'A{head}' if touch_policy=='legacy_two_heads' and str(head).isdigit() else str(head)
                            prior=guard.last_head_seconds.get(pad)
                            if not aligned or (prior is not None and cand_sec-prior<guard.min_same_head_sec-1e-9):
                                cooldown_blocked.add(pad)
            logits[chosen]=float('-inf')
        return None
    indices=np.random.default_rng(args.seed+global_step+991).integers(0,len(dataset),size=args.recovery_samples)
    was_training=model.training
    model.eval()
    try:
        with torch.no_grad():
            horizon=max(4,((min(dataset.history_events,128)+args.recovery_steps-1)//args.recovery_steps)*args.recovery_steps)
            builder=RecoveryBuilder(dataset,model,device,propose=propose,
                supervise_actions=getattr(args,'action_objective_policy',LEGACY_ACTION_OBJECTIVE)==RECOVERY_ACTION_OBJECTIVE,
                maximum_generated_events=horizon if aligned else 4)
            if not aligned:return builder.batch_from_indices(indices,steps=args.recovery_steps)
            batch,outcomes,episodes,details=builder.batch_from_episodes(indices,getattr(model,'_recovery_episodes',[]),steps=args.recovery_steps)
            model._recovery_episodes=episodes;model._recovery_diagnostics=details
            return batch,outcomes
    finally:model.train(was_training)


def main(argv=None):
    import sys
    arguments=list(sys.argv[1:] if argv is None else argv)
    explicit_flags={token.split('=',1)[0] for token in arguments if token.startswith('--')}
    args=parser().parse_args(arguments)
    from .training_lock import training_output_lock
    with training_output_lock(args.checkpoint_dir):
        return _train(args,explicit_flags)


def _train(args,explicit_flags):
    import sys
    device=_device(args.device);precision=_precision(args.precision,device)
    if not 0<=args.cpu_threads<=256:raise ValueError('cpu_threads must be in [0,256]')
    if args.rebatch_resume and (not args.resume or args.max_steps):
        raise ValueError('--rebatch-resume requires --resume and full epochs (max_steps=0)')
    if args.epochs<1 or args.batch_size<1 or args.grad_accumulation<1 or args.log_interval<1 or args.max_steps<0 or args.profile_steps<0:
        raise ValueError('epochs, batch_size, grad_accumulation and log_interval must be positive; max_steps/profile_steps must be nonnegative')
    if not math.isfinite(args.lr) or args.lr<=0:
        raise ValueError('lr must be positive and finite')
    if args.warmup_steps is not None and args.warmup_steps<0:
        raise ValueError('warmup_steps must be nonnegative')
    if args.min_lr_ratio is not None and (not math.isfinite(args.min_lr_ratio) or not 0<=args.min_lr_ratio<=1):
        raise ValueError('min_lr_ratio must be in [0,1]')
    if not math.isfinite(args.dropout) or not 0<=args.dropout<=1:
        raise ValueError('dropout must be between 0 and 1')
    if not math.isfinite(args.symbolic_dropout) or not 0<=args.symbolic_dropout<=.5:
        raise ValueError('symbolic_dropout must be between 0 and .5')
    if not math.isfinite(args.audio_pointer_weight) or not 0<=args.audio_pointer_weight<=1:
        raise ValueError('audio_pointer_weight must be between 0 and 1')
    if args.recovery_interval<0 or not 1<=args.recovery_samples<=8:
        raise ValueError('Invalid recovery interval/sample count')
    if not 8<=args.lens_width<=256 or not 1<=args.lens_chunk_size<=256:
        raise ValueError('Lens width/chunk must be in [8,256] / [1,256]')
    for key in ('placement_weight','planner_weight','recovery_weight'):
        if not math.isfinite(getattr(args,key)) or not 0<=getattr(args,key)<=1:raise ValueError(f'{key} must be in [0,1]')
    destination=Path(args.checkpoint_dir).resolve()
    if args.init_checkpoint and Path(args.init_checkpoint).resolve().parent==destination:
        raise ValueError('Finetune/migration requires a new output directory; preserve the parent checkpoint')
    if not args.resume:
        existing=next((destination/name for name in ('composer_last.pt','composer_inference.pt')
                       if (destination/name).exists()),None)
        if existing is not None:
            raise ValueError(f'Fresh/finetune output already contains {existing.name}; choose a new directory')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    old=None;model=None
    parent_anchor_config=None
    if args.resume or args.init_checkpoint:
        model,old=load(args.resume or args.init_checkpoint,device='cpu',include_optimizer=bool(args.resume))
        parent_recipe=old.get('args') or {}
        if (args.resume and (parent_recipe.get('lr_schedule')=='cosine' or parent_recipe.get('warmup_steps',0)>0)
                and not (old.get('training_state') or {}).get('lr_schedule')):
            raise ValueError('Scheduled checkpoint is missing its learning-rate plan; use finetune with a new directory')
        parent_anchor_config=model.config.to_dict()
        args.history_anchor_policy=resolve_history_anchor_policy(args.history_anchor_policy,parent_anchor_config,resume=bool(args.resume))
        if '--dropout' in explicit_flags and args.dropout!=model.config.dropout:
            raise ValueError('dropout inherits the parent model; use fresh training to change it')
        args.dropout=float(model.config.dropout)
        # Both continuation modes keep the parent's backbone. CLI defaults
        # must not silently request a different model during explicit finetune.
        for key in ('d_model','dim_feedforward','layers','nhead','history_events','candidate_count','action_slots'):
            if '--'+key.replace('_','-') not in explicit_flags:
                setattr(args,key,int(getattr(model.config,key)))
        if args.resume:
            args.architecture_profile=model.config.architecture_profile
            if '--cpu-threads' not in explicit_flags:
                args.cpu_threads=int((old.get('args') or {}).get('cpu_threads',0))
            # Resume is an exact continuation: restore the saved contract
            # before dataset construction so CLI defaults do not conflict.
            for key in ('sample_locality','symbolic_dropout','audio_pointer_weight','placement_weight','planner_weight',
                        'recovery_interval','recovery_samples','recovery_steps','recovery_weight','history_dropout',
                        'ledger_dropout','history_residual_bound','lens_width','lens_chunk_size',
                        'recovery_pointer_temperature','recovery_action_temperature'):
                if key in (old.get('args') or {}) and '--'+key.replace('_','-') not in explicit_flags:
                    setattr(args,key,old['args'][key])
        if args.resume:
            from .checkpoint import motif_training_compatible
            if not motif_training_compatible(old):
                raise ValueError('Motif input/dropout contract changed; use --init-checkpoint with a new output directory')
    from .model import resolve_condition_mask_prob
    args.condition_mask_prob=resolve_condition_mask_prob(getattr(args,'condition_mask_prob',None),
        parent_anchor_config,resume=bool(args.resume))
    args.action_objective_policy=resolve_action_objective(getattr(args,'action_objective_policy','auto'),
        old.get('args',{}) if old else None,resume=bool(args.resume),architecture_profile=args.architecture_profile)
    args.recovery_sampling_policy=resolve_recovery_sampling(args.recovery_sampling_policy,old,
        resume=bool(args.resume),architecture_profile=args.architecture_profile)
    recovery_sampling_settings(args.recovery_sampling_policy,args.recovery_pointer_temperature,args.recovery_action_temperature)
    if not (args.resume or args.init_checkpoint):
        args.history_anchor_policy=resolve_history_anchor_policy(args.history_anchor_policy)
    for name in COMPOSER_POLICY_FIELDS:
        setattr(args,name,resolve_composer_policy(name,getattr(args,name),parent_anchor_config,resume=bool(args.resume),architecture_profile=args.architecture_profile))
    semantics=structure_training_semantics(
        training_semantics_for_objective(args.architecture_profile,args.action_objective_policy),args.structure_policy)
    if args.resume and old.get('training_semantics')!=semantics:
        raise ValueError('Composer training contract changed; use --init-checkpoint with a new output directory')
    if args.resume:
        parent_backend=parent_anchor_config.get('structure_backend','native')
        if args.structure_backend not in {'auto',parent_backend}:
            raise ValueError('Resume inherits structure_backend; use a new training directory')
        args.structure_backend=parent_backend
        inherited_drop=float(parent_anchor_config.get('structure_dropout',.15))
        if '--structure-dropout' in explicit_flags and args.structure_dropout!=inherited_drop:
            raise ValueError('Resume inherits structure_dropout')
        args.structure_dropout=inherited_drop
    elif args.structure_policy in {'metrical_context_v1','metrical_context_v2'}:
        from .structure import resolve_structure_backend
        args.structure_backend=resolve_structure_backend(args.structure_backend)
    else:args.structure_backend='native'
    from .structure_backend import backend_identity
    args.structure_backend_identity=(backend_identity(args.structure_backend,refresh=True) if args.structure_policy!='off' else None)
    if args.resume and args.structure_policy in {'metrical_context_v1','metrical_context_v2'}:
        expected=parent_anchor_config.get('structure_backend_identity') or model.config.structure_backend_identity
        if expected!=args.structure_backend_identity:
            raise ValueError('Resume requires the exact saved structure backend identity')
    if not math.isfinite(args.structure_dropout) or not 0<=args.structure_dropout<=.9:
        raise ValueError('structure_dropout must be in [0,.9]')
    if args.archive_checkpoints=='auto':
        args.archive_checkpoints=((old.get('args') or {}).get('archive_checkpoints','off') if args.resume
            else 'geometric' if args.structure_policy!='off' else 'off')
    if parent_anchor_config and '--motif-max-events' not in explicit_flags:
        args.motif_max_events=int(parent_anchor_config.get('motif_max_events',16))
    if args.resume and args.motif_max_events!=int(parent_anchor_config.get('motif_max_events',16)):
        raise ValueError('Resume inherits motif_max_events; use explicit finetune')
    inherited_rms=float((parent_anchor_config or {}).get('history_memory_rms_bound',8.))
    if parent_anchor_config and '--history-memory-rms-bound' not in explicit_flags:
        args.history_memory_rms_bound=inherited_rms
    if not math.isfinite(args.history_memory_rms_bound) or not 1<=args.history_memory_rms_bound<=64:
        raise ValueError('history_memory_rms_bound must be in [1,64]')
    if args.resume and args.history_memory_rms_bound!=inherited_rms:
        raise ValueError('Resume inherits history_memory_rms_bound; use explicit finetune')
    if not 1<=args.motif_max_events<=32:raise ValueError('motif_max_events must be in [1,32]')
    inherited_touch=(old.get('args',{}).get('touch_layout_policy','legacy_two_heads') if old else 'open')
    if args.touch_layout_policy=='auto':args.touch_layout_policy=inherited_touch
    if args.resume and args.touch_layout_policy!=inherited_touch:
        raise ValueError('Resume inherits touch_layout_policy; use explicit finetune')
    from .model import validate_supported_context_policies
    validate_supported_context_policies(vars(args))
    if not 0<=args.cpu_threads<=256:raise ValueError('cpu_threads must be in [0,256]')
    if args.cpu_threads:
        torch.set_num_threads(args.cpu_threads)
    if not 0<=args.holdout_fraction<.5 or args.validation_max_batches<0 or args.validation_batch_size<1:
        raise ValueError('Invalid validation settings')
    if args.generation_eval_songs<0 or args.generation_eval_interval_epochs<1:
        raise ValueError('Invalid generation evaluation settings')
    generation_seeds=tuple(int(s.strip()) for s in args.generation_eval_seeds.split(',') if s.strip())
    if not generation_seeds:raise ValueError('generation-eval-seeds must include at least one integer')
    split_report=None
    if args.resume:
        saved_args=old.get('args') or {}
        for key in ('manifest','validation_manifest','holdout_fraction','source_manifest'):
            if key in saved_args:
                flag='--'+key.replace('_','-')
                if key=='manifest' and Path(args.manifest).resolve() not in {
                        Path(saved_args['manifest']).resolve(),Path(saved_args.get('source_manifest',saved_args['manifest'])).resolve()}:
                    raise ValueError('Resume requires the recorded training or source manifest')
                if flag not in explicit_flags or key=='manifest':setattr(args,key,saved_args[key])
    elif not args.validation_manifest and args.holdout_fraction and Path(args.manifest).is_file():
        from .evaluation import split_manifest
        source_data=json.loads(Path(args.manifest).read_text(encoding='utf-8-sig'))
        if len(source_data.get('songs',[]))>=3:
            args.source_manifest=args.manifest
            split_report=split_manifest(args.manifest,Path(args.checkpoint_dir)/'data_split',
                                        fraction=args.holdout_fraction,seed=args.seed)
            args.manifest=split_report['splits']['train']['path']
            args.validation_manifest=split_report['splits']['dev']['path']
            print('composer_data_split '+json.dumps(split_report),flush=True)
        else:print('composer_validation_disabled '+json.dumps({'reason':'fewer_than_three_source_songs'}),flush=True)
    if args.validation_manifest:
        from .evaluation import assert_unseen
        assert_unseen(args.manifest,args.validation_manifest)
    dataset=ComposerDataset(args.manifest,cache_dir=args.cache_dir,candidate_count=args.candidate_count,
        history_events=args.history_events,action_slots=args.action_slots,preparation_workers=args.prepare_workers,
        audio_cache_gb=args.audio_cache_gb,limit_charts=args.limit_charts,seed=args.seed,architecture_profile=args.architecture_profile,
        action_state_policy=args.action_state_policy,motif_memory_policy=args.motif_memory_policy,
        motif_max_events=args.motif_max_events,head_timing_policy=args.head_timing_policy,
        structure_policy=args.structure_policy,structure_backend=args.structure_backend)
    dataset.touch_layout_policy=args.touch_layout_policy
    dataset.contact_policy=args.contact_policy
    dataset.defer_audio_queries=device.type=='cuda' and args.architecture_profile==GROUNDED_PROFILE
    from .device_audio import AudioQueryMaterializer
    audio_materializer=AudioQueryMaterializer()
    args.action_slots_requested = dataset.action_slots_requested
    args.action_slots = dataset.action_slots
    if args.resume and model.config.action_slots != args.action_slots:
        raise ValueError(f'Composer dataset requires action_slots={args.action_slots}, but checkpoint has '
            f'{model.config.action_slots}; use --init-checkpoint with a new output directory for explicit finetune')
    if args.resume or args.init_checkpoint:
        if model.config.architecture_profile not in WORKLOAD_PROFILES:
            raise ValueError('Composer workload/conditional-slot architecture requires fresh training in a new directory')
        if args.resume and model.config.architecture_profile not in {'music_context_v2',GROUNDED_PROFILE}:
            raise ValueError('Older Composer architecture requires explicit finetune migration to a new directory')
        if args.init_checkpoint:
            model=(migrate_audio_grounded(model,dataset.codec_atoms,action_slots=args.action_slots,lens_width=args.lens_width,
                lens_chunk_size=args.lens_chunk_size,history_residual_bound=args.history_residual_bound,
                history_dropout=args.history_dropout,ledger_dropout=args.ledger_dropout,history_anchor_policy=args.history_anchor_policy,
                history_memory_rms_bound=args.history_memory_rms_bound,condition_mask_prob=args.condition_mask_prob,
                motif_max_events=args.motif_max_events,**{name:getattr(args,name) for name in COMPOSER_POLICY_FIELDS})
                if args.architecture_profile==GROUNDED_PROFILE else migrate_music_context(model,action_slots=args.action_slots,history_anchor_policy=args.history_anchor_policy,
                    history_memory_rms_bound=args.history_memory_rms_bound,condition_mask_prob=args.condition_mask_prob,
                    motif_max_events=args.motif_max_events,**{name:getattr(args,name) for name in COMPOSER_POLICY_FIELDS}))
            model.config.symbolic_dropout=args.symbolic_dropout
        if list(old.get('atom_catalog') or []) != list(dataset.codec_atoms):
            raise ValueError('Composer atom catalog differs from checkpoint; use a matching manifest or retrain a new model')
        if old.get('dataset_fingerprint') and args.resume and old['dataset_fingerprint'] != dataset.fingerprint:
            raise ValueError('Composer resume dataset fingerprint changed; use finetune with a new output directory')
        for key in ('d_model','dim_feedforward','layers','nhead','history_events','candidate_count','action_slots'):
            if int(getattr(model.config,key)) != int(getattr(args,key)):
                raise ValueError(f'Composer checkpoint {key}={getattr(model.config,key)} differs from requested {getattr(args,key)}')
    else:
        model=build_model(ComposerConfig(atom_vocab_size=len(dataset.codec_atoms)+1,d_model=args.d_model,
            nhead=args.nhead,layers=args.layers,dim_feedforward=args.dim_feedforward,dropout=args.dropout,
            history_events=args.history_events,candidate_count=args.candidate_count,action_slots=args.action_slots,
            architecture_profile=args.architecture_profile,symbolic_dropout=args.symbolic_dropout,
            head_timing_dim=head_timing_dim_for_policy(args.head_timing_policy),lens_width=args.lens_width,lens_chunk_size=args.lens_chunk_size,
            history_residual_bound=args.history_residual_bound,history_dropout=args.history_dropout,
            ledger_dropout=args.ledger_dropout,history_anchor_policy=args.history_anchor_policy,
            history_memory_rms_bound=args.history_memory_rms_bound,condition_mask_prob=args.condition_mask_prob,
            structure_backend=args.structure_backend,structure_dropout=args.structure_dropout,
            structure_backend_identity=args.structure_backend_identity,
            motif_max_events=args.motif_max_events,**{name:getattr(args,name) for name in COMPOSER_POLICY_FIELDS}),dataset.codec_atoms)
    if args.init_checkpoint:
        model.config.structure_backend=args.structure_backend
        model.config.structure_dropout=args.structure_dropout
        model.config.structure_backend_identity=args.structure_backend_identity
    validation_dataset=None
    if args.validation_manifest:
        validation_dataset=ComposerDataset(args.validation_manifest,cache_dir=args.cache_dir,
            candidate_count=args.candidate_count,history_events=args.history_events,action_slots=args.action_slots,
            preparation_workers=args.prepare_workers,audio_cache_gb=min(args.audio_cache_gb,1.),
            architecture_profile=args.architecture_profile,action_state_policy=args.action_state_policy,
            motif_memory_policy=args.motif_memory_policy,motif_max_events=args.motif_max_events,
            head_timing_policy=args.head_timing_policy,structure_policy=args.structure_policy,
            structure_backend=args.structure_backend,atom_catalog=dataset.codec_atoms)
    model.to(device).train()
    model_finite_items=_model_finite_items(model)
    from .recovery import restore_recovery_episodes,recovery_episode_state
    restore_recovery_episodes(model,(old.get('training_state') or {}).get('recovery_episodes') if args.resume else None)
    cuda_cache_guard=CudaCacheGuard(device,0.)
    if model.config.architecture_profile==GROUNDED_PROFILE:
        for key in ('lens_width','lens_chunk_size','history_residual_bound','history_dropout','ledger_dropout','history_memory_rms_bound'):
            if getattr(model.config,key)!=getattr(args,key):
                raise ValueError(f'Effective model {key} differs from requested value; use explicit finetune')
    loader_generator=torch.Generator().manual_seed(args.seed)
    implementation=args.optimizer_implementation
    if implementation=='auto': implementation='fused' if device.type=='cuda' else 'standard'
    if implementation=='fused' and device.type!='cuda':
        raise ValueError('fused optimizer requires CUDA; choose auto or standard for CPU')
    migrated_from_older = bool(args.init_checkpoint and old
        and (old.get('model_config') or {}).get('architecture_profile') not in {'music_context_v2', GROUNDED_PROFILE})
    saved_opt_groups = ((old.get('optimizer_state_dict') or {}).get('param_groups') or []) if args.resume and old else []
    saved_lr = (float(saved_opt_groups[0]['lr']) if saved_opt_groups and 'lr' in saved_opt_groups[0]
                else float((old.get('args') or {}).get('lr', args.lr)) if args.resume and old else None)
    effective_lr = (saved_lr if (args.resume and saved_lr is not None and '--lr' not in explicit_flags)
                    else min(float(args.lr), 5e-5) if migrated_from_older else float(args.lr))
    saved_lr_plan=((old.get('training_state') or {}).get('lr_schedule') if args.resume and old else None)
    if saved_lr_plan and '--lr' not in explicit_flags:
        effective_lr=float(saved_lr_plan['base_lr'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=effective_lr,weight_decay=.01,
        **({'fused':True} if implementation=='fused' else {'foreach':True} if implementation=='foreach' else {}))
    if args.resume:
        if not old or old.get('optimizer_state_dict') is None:
            raise ValueError('Inference-only weights cannot resume; choose finetune')
        optimizer.load_state_dict(old.pop('optimizer_state_dict'))
        for group in optimizer.param_groups:
            group['lr']=effective_lr
            group['fused']=implementation=='fused';group['foreach']=implementation=='foreach'
    out=Path(args.checkpoint_dir);out.mkdir(parents=True,exist_ok=True)
    global_step=int(old.get('optimizer_steps',0)) if args.resume and old else 0
    resumed_state=(old.get('training_state') or {}) if args.resume and old else {}
    partial_resume=bool(resumed_state and not resumed_state.get('epoch_complete',True))
    resume_skip,resume_sampler_size,resume_cursor=resume_batch_layout(args,old,len(dataset))
    planned_batches=(len(dataset)+args.batch_size-1)//args.batch_size
    if args.max_steps: planned_batches=min(planned_batches,args.max_steps)
    remaining_updates=(planned_batches-resume_skip+args.grad_accumulation-1)//args.grad_accumulation
    remaining_updates+=(args.epochs-1)*((planned_batches+args.grad_accumulation-1)//args.grad_accumulation)
    lr_plan=resolve_lr_plan(requested=args.lr_schedule,warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,base_lr=effective_lr,total_steps=max(1,global_step+remaining_updates),
        resume=bool(args.resume),saved=saved_lr_plan)
    args.lr_schedule=lr_plan['kind'];args.warmup_steps=lr_plan['warmup_steps'];args.min_lr_ratio=lr_plan['min_lr_ratio']
    args.lr=effective_lr
    for group in optimizer.param_groups: group['lr']=learning_rate(lr_plan,global_step)
    print('composer_learning_rate '+json.dumps(lr_plan),flush=True)
    if args.resume and old:
        for key in ('max_steps',):
            stored=old.get('args',{}).get(key,1 if key=='grad_accumulation' else 0)
            if partial_resume and stored!=getattr(args,key):
                raise ValueError(f'Mid-epoch resume requires the saved {key}={stored}; use finetune to change it')
        for key in ('sample_locality','symbolic_dropout','audio_pointer_weight','placement_weight','planner_weight',
                    'recovery_interval','recovery_samples','recovery_steps','recovery_weight','history_dropout',
                    'ledger_dropout','history_residual_bound','lens_width','lens_chunk_size',
                    'recovery_pointer_temperature','recovery_action_temperature'):
            if old.get('args',{}).get(key,getattr(args,key))!=getattr(args,key):
                raise ValueError(f'Resume requires saved {key}; use finetune to change the training policy')
    start_epoch=(int(old.get('epoch',0))+(0 if partial_resume else 1)) if args.resume and old else 1
    end_epoch=start_epoch+args.epochs-1
    scaler=(torch.amp.GradScaler('cuda') if device.type=='cuda' and precision=='fp16' else None)
    if resumed_state:
        if resumed_state.get('loader_generator_state') is not None:
            loader_generator.set_state(resume_loader_rng_state(resumed_state))
        if resumed_state.get('torch_rng_state') is not None:
            torch.set_rng_state(resumed_state['torch_rng_state'])
        if device.type=='cuda' and resumed_state.get('cuda_rng_state') is not None:
            torch.cuda.set_rng_state(resumed_state['cuda_rng_state'],device)
        if scaler is not None and resumed_state.get('scaler'):
            scaler.load_state_dict(resumed_state['scaler'])
    parameters=sum(p.numel() for p in model.parameters())
    initial_host_memory=host_memory()
    if initial_host_memory is not None and (initial_host_memory['free_commit']<4*GIB
                                           or initial_host_memory['free_physical']<2*GIB):
        print('composer_warning '+json.dumps({'message':
            'Low Windows host memory headroom: the audio cache may shrink. Close unused memory-heavy processes; '
            'available GPU memory alone is insufficient to diagnose allocation failures.',
            'host_memory':memory_gib(initial_host_memory)}),flush=True)
    run_started=time.monotonic()
    timing=dict(data_sec=0.,transfer_sec=0.,compute_sec=0.,save_sec=0.)
    gpu_profiles=[]; gpu_timing=dict(samples=0,h2d_ms=0.,forward_ms=0.,backward_ms=0.,update_ms=0.)
    print('composer_runtime '+json.dumps({'family':FAMILY,'training_semantics':semantics,
        'action_objective_policy':args.action_objective_policy,
        'recovery_sampling_policy':args.recovery_sampling_policy,
        'recovery_sampling_settings':recovery_sampling_settings(args.recovery_sampling_policy,args.recovery_pointer_temperature,args.recovery_action_temperature),
        'recovery_pointer_temperature':args.recovery_pointer_temperature,
        'recovery_action_temperature':args.recovery_action_temperature,
        'device':str(device),'precision':precision,'effective_lr':effective_lr,'learning_rate_schedule':lr_plan,
        'samples':len(dataset),'parameters':parameters,
        'model_bytes':sum(p.numel()*p.element_size() for p in model.parameters()),
        'action_head_parameters':sum(p.numel() for p in model.action_head.parameters()),
        'selected_candidate_projection':True,'optimizer_implementation':implementation,
        'action_projection':'active_slots_v1' if args.architecture_profile==GROUNDED_PROFILE else 'dense',
        'audio_query_execution':'device_banks_v1' if dataset.defer_audio_queries else 'cpu',
        'audio_device_cache_budget_bytes':audio_materializer.budget_bytes if dataset.defer_audio_queries else 0,
        'planner_batch_sharing':'same_source_and_effective_condition_v1',
        'audio_bank_mode':'lossless_mmap_v1','audio_feature_policy':COMPOSER_AUDIO_FEATURE_POLICY,
        'batch_size':args.batch_size,'grad_accumulation':args.grad_accumulation,
        'effective_batch_size':args.batch_size*args.grad_accumulation,
        'forward_finite_check':'packed_outputs_and_losses_v1',
        'start_epoch':start_epoch,'prefetch_batches':args.prefetch_batches,
        'sample_workers':args.sample_workers,'sample_locality':args.sample_locality,
        'cuda_cache_guard':cuda_cache_guard.policy(),
        'cuda_allocator_config':os.environ.get('PYTORCH_CUDA_ALLOC_CONF'),
        'resume_sample_cursor':resume_cursor,'resume_sampler_batch_size':resume_sampler_size,
        'rebatch_resume':args.rebatch_resume,
        'torch_cpu_threads':torch.get_num_threads(),'python_switch_interval_sec':sys.getswitchinterval(),
        'architecture_profile':model.config.architecture_profile,'symbolic_dropout':model.config.symbolic_dropout,
        'history_anchor_policy':model.config.history_anchor_policy,
        'head_timing_policy':model.config.head_timing_policy,
        'cross_event_relation_policy':model.config.cross_event_relation_policy,
        'action_state_policy':model.config.action_state_policy,'motif_memory_policy':model.config.motif_memory_policy,
        'slide_factor_policy':model.config.slide_factor_policy,'motif_max_events':model.config.motif_max_events,
        'history_memory_policy':model.config.history_memory_policy,
        'history_attention_policy':model.config.history_attention_policy,
        'history_memory_rms_bound':model.config.history_memory_rms_bound,
        'touch_layout_policy':args.touch_layout_policy,
        'contact_policy':args.contact_policy,'structure_policy':args.structure_policy,
        'future_horizon_policy':model.config.future_horizon_policy,
        'action_head_policy':model.config.action_head_policy,
        'condition_mask_prob':model.config.condition_mask_prob,
        'structure_backend':args.structure_backend,'structure_dropout':args.structure_dropout,
        'structure_schema':model.config.structure_schema,
        'structure_backend_identity':model.config.structure_backend_identity,
        'validation_manifest':args.validation_manifest,'holdout_fraction':args.holdout_fraction,
        'generation_eval_songs':args.generation_eval_songs,
        'dropout':model.config.dropout,
        **{key:getattr(args,key) for key in ('lens_width','lens_chunk_size','history_residual_bound',
            'history_dropout','ledger_dropout','placement_weight','planner_weight','recovery_interval',
            'recovery_samples','recovery_steps','recovery_weight')},
        'audio_pointer_weight':args.audio_pointer_weight,'candidate_miss_categories':getattr(dataset,'candidate_miss_categories',{}),
        'candidate_count':args.candidate_count,'action_vocab_size':len(dataset.codec_atoms)+1,
        'action_slots':args.action_slots,'action_slots_requested':dataset.action_slots_requested,
        'max_required_action_slots':dataset.max_required_action_slots,
        'host_memory':memory_gib(initial_host_memory),
        'end_epoch':end_epoch,'candidate_misses':dataset.candidate_misses,
        'phase_filter_misses':getattr(dataset,'phase_filter_misses',0),
        'candidate_total':dataset.candidate_total,
        'candidate_miss_examples':dataset.candidate_miss_examples,
        'candidate_recall':1.-dataset.candidate_misses/max(1,dataset.candidate_total)}),flush=True)
    for epoch in range(start_epoch,end_epoch+1):
        dataset.set_epoch(epoch);started=time.monotonic();totals={};seen=0;metric_batches=0;window_valid=True
        recent_totals={};recent_batches=0;optimized_total=0.;optimized_updates=0;window_objective=0.
        last_gradient_norm=None
        continuing=partial_resume and epoch==start_epoch
        skip=resume_skip if continuing else 0
        sampler_size=resume_sampler_size if continuing else args.batch_size
        epoch_loader_state=loader_generator.get_state().clone()
        batch_sampler=SongLocalBatchSampler(dataset,sampler_size,loader_generator,args.sample_locality)
        if sampler_size!=args.batch_size:
            batch_sampler=RebatchedSampler(batch_sampler,args.batch_size,len(dataset))
        all_batches=len(batch_sampler)
        total_batches=min(all_batches,args.max_steps) if args.max_steps else all_batches
        loader=DataLoader(dataset,batch_sampler=RemainingBatches(batch_sampler,skip,total_batches),
            collate_fn=collate_composer,generator=loader_generator,pin_memory=device.type=='cuda')
        source=LayoutBatchPrefetch(loader,depth=args.prefetch_batches,sample_workers=args.sample_workers) if args.prefetch_batches else nullcontext(iter(loader))
        optimizer.zero_grad(set_to_none=True)
        with _cuda_execution(device), source as batches:
            for batch_index in range(skip+1,total_batches+1):
                stamp=time.perf_counter();batch=next(batches)
                timing['data_sec']+=time.perf_counter()-stamp
                if (batch_index-1)%args.grad_accumulation==0:
                    window_valid=True;window_objective=0.;window_finite=None;window_metrics={};window_metric_batches=0
                if not window_valid: continue
                events=[torch.cuda.Event(enable_timing=True) for _ in range(5)] if device.type=='cuda' and len(gpu_profiles)+gpu_timing['samples']<args.profile_steps else None
                if events: events[0].record()
                stamp=time.perf_counter()
                tensors={key:value.to(device,non_blocking=device.type=='cuda')
                         for key,value in batch.items() if torch.is_tensor(value)
                         and key not in {'song_index','event_index'}}
                audio_materializer.materialize(batch,tensors)
                timing['transfer_sec']+=time.perf_counter()-stamp
                if events: events[1].record()
                stamp=time.perf_counter()
                with torch.autocast('cuda',dtype=torch.bfloat16 if precision=='bf16' else torch.float16,
                                    enabled=device.type=='cuda' and precision in {'bf16','fp16'}):
                    output=_forward_batch(model,tensors,packed_actions=True,audio_auxiliary=args.audio_pointer_weight>0)
                    loss,metrics=_loss(output,tensors,model.config.atom_vocab_size,audio_pointer_weight=args.audio_pointer_weight,
                        placement_weight=args.placement_weight,planner_weight=args.planner_weight,
                        action_objective_policy=args.action_objective_policy)
                window_start=((batch_index-1)//args.grad_accumulation)*args.grad_accumulation
                window_batches=min(args.grad_accumulation,total_batches-window_start)
                backward_loss=loss/window_batches
                finite_flags=_forward_finite_flags(output,loss,backward_loss)
                window_finite=finite_flags if window_finite is None else window_finite & finite_flags
                if events: events[2].record()
                if scaler is not None: scaler.scale(backward_loss).backward()
                else: backward_loss.backward()
                window_objective=window_objective+backward_loss.detach()
                for key,value in metrics.items():
                    value=value.detach().float()
                    window_metrics[key]=window_metrics[key]+value if key in window_metrics else value.clone()
                window_metric_batches+=1
                if events: events[3].record()
                update=(batch_index%args.grad_accumulation==0 or batch_index==total_batches)
                if update:
                    episode_backup=None
                    if (model.config.architecture_profile==GROUNDED_PROFILE and args.recovery_interval and args.recovery_weight>0
                            and (global_step+1)%args.recovery_interval==0):
                        recovery_started=time.perf_counter()
                        episode_backup=recovery_episode_state(model)
                        recovered,diagnostics=recovery_batch(dataset,model,args,device,global_step)
                        recovery_sampled=time.perf_counter()
                        recovery_value=None
                        recovery_parts={}
                        configuration_samples=0
                        if recovered is not None:
                            recovered={k:v.to(device) for k,v in recovered.items() if torch.is_tensor(v)}
                            with torch.autocast('cuda',dtype=torch.bfloat16 if precision=='bf16' else torch.float16,
                                    enabled=device.type=='cuda' and precision in {'bf16','fp16'}):
                                recovery_output=_forward_batch(model,recovered,
                                    packed_actions=True,
                                    decode_actions=args.action_objective_policy==RECOVERY_ACTION_OBJECTIVE,
                                    audio_auxiliary=False)
                                recovery_loss,recovery_metrics=recovery_objective(recovery_output,recovered,
                                    policy=args.action_objective_policy,weight=args.recovery_weight)
                            window_finite=window_finite & _forward_finite_flags(recovery_output,recovery_loss,recovery_loss)
                            if torch.isfinite(recovery_loss):
                                recovery_value=float(recovery_loss.detach())
                                window_objective=window_objective+recovery_loss.detach()
                                recovery_parts={name:float(value) for name,value in recovery_metrics.items()}
                                if scaler is not None:scaler.scale(recovery_loss).backward()
                                else:recovery_loss.backward()
                                if args.action_objective_policy==RECOVERY_ACTION_OBJECTIVE:
                                    configuration_samples=int(recovered['candidate_target'].numel())
                        print('composer_recovery '+json.dumps({'optimizer_step':global_step+1,
                            'seconds':time.perf_counter()-recovery_started,'outcomes':diagnostics,
                            'sampling_seconds':recovery_sampled-recovery_started,
                            'learning_seconds':time.perf_counter()-recovery_sampled,
                            'loss':recovery_value,'loss_includes_weight':True,
                            'components_unweighted':recovery_parts,
                            'episode':getattr(model,'_recovery_diagnostics',{}),
                            'configuration_samples':configuration_samples}),flush=True)
                    if scaler is not None: scaler.unscale_(optimizer)
                    # The global norm detects NaN/Inf across all gradients;
                    # no per-parameter Python scalar checks or mutation needed.
                    grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=False)
                    last_gradient_norm=grad_norm.detach()
                    update_health=_window_update_health(window_finite,grad_norm,model,episode_backup)
                    if not all(update_health):
                        print('composer_warning '+json.dumps({'message':'non-finite accumulation window skipped; checkpoint remains valid',
                            'epoch':epoch,'batch':batch_index,
                            'finite':dict(zip(('outputs','loss','backward_loss','grad_norm'),update_health))}),flush=True)
                        optimizer.zero_grad(set_to_none=True)
                        if scaler is not None: scaler.update()
                        continue
                    for group in optimizer.param_groups: group['lr']=learning_rate(lr_plan,global_step)
                    if scaler is not None: scaler.step(optimizer);scaler.update()
                    else: optimizer.step()
                    # AdamW can still create non-finite parameters even when
                    # the gradient was finite (notably after BF16 overflow in
                    # an optimizer state).  Refuse to write such a checkpoint;
                    # the previous checkpoint remains the valid recovery point.
                    _assert_model_finite(model,model_finite_items)
                    optimizer.zero_grad(set_to_none=True);global_step+=1
                    optimized_total=optimized_total+window_objective;optimized_updates+=1
                    cache_started=time.perf_counter()
                    cache_event=cuda_cache_guard.check()
                    if cache_event is not None:
                        print('composer_cuda_cache_trim '+json.dumps({'optimizer_steps':global_step,
                            'seconds':time.perf_counter()-cache_started,**cache_event}),flush=True)
                if events:
                    events[4].record();gpu_profiles.append(events)
                timing['compute_sec']+=time.perf_counter()-stamp
                seen+=int(tensors['candidate_target'].numel())
                if update:
                    # A rejected accumulation window contributes neither updates
                    # nor NaN metrics to later progress/checkpoint summaries.
                    metric_batches+=window_metric_batches
                    for key,value in window_metrics.items():
                        totals[key]=totals[key]+value if key in totals else value.clone()
                        recent_totals[key]=recent_totals[key]+value if key in recent_totals else value.clone()
                    recent_batches+=window_metric_batches
                if update and args.checkpoint_interval_steps and global_step%args.checkpoint_interval_steps==0:
                    _assert_model_finite(model)
                    stamp=time.perf_counter()
                    save(out/'composer_last.pt',model,optimizer,epoch=epoch,args=vars(args),manifest=args.manifest,
                         metrics=summarize_metrics(totals,metric_batches),optimizer_steps=global_step,
                         dataset_fingerprint=dataset.fingerprint,atom_catalog=dataset.codec_atoms,
                         training_state={'epoch_complete':batch_index==total_batches,'batch_in_epoch':batch_index,
                            'lr_schedule':lr_plan,
                            'recovery_episodes':recovery_episode_state(model),
                            'sample_in_epoch':min(batch_index*args.batch_size,len(dataset)),
                            'sampler_batch_size':sampler_size,
                            'loader_generator_state':checkpoint_loader_rng_state(
                                loader_generator.get_state() if batch_index==total_batches else epoch_loader_state,
                                sampler_size,args.batch_size,batch_index==total_batches),
                            'torch_rng_state':torch.get_rng_state(),
                            'cuda_rng_state':torch.cuda.get_rng_state(device) if device.type=='cuda' else None,
                            'scaler':scaler.state_dict() if scaler is not None else None})
                    timing['save_sec']+=time.perf_counter()-stamp
                    print('composer_checkpoint '+json.dumps({'epoch':epoch,'optimizer_steps':global_step,
                        'path':str(out/'composer_last.pt'),'save_sec':timing['save_sec']}),flush=True)
                    quotient=global_step//args.checkpoint_interval_steps
                    if args.archive_checkpoints=='geometric' and quotient and not (quotient & (quotient-1)):
                        archive=out/'snapshots'/f'composer_step_{global_step:08d}.pt'
                        if not archive.exists():
                            save(archive,model,None,epoch=epoch,args=vars(args),manifest=args.manifest,
                                metrics=summarize_metrics(totals,metric_batches),optimizer_steps=global_step,
                                dataset_fingerprint=dataset.fingerprint,atom_catalog=dataset.codec_atoms)
                            print('composer_snapshot '+json.dumps({'optimizer_steps':global_step,
                                'path':str(archive),'kind':'inference'}),flush=True)
                if batch_index%args.log_interval==0 or batch_index in {1,total_batches}:
                    for recorded in gpu_profiles:
                        recorded[-1].synchronize()
                        for i,key in enumerate(('h2d_ms','forward_ms','backward_ms','update_ms')):
                            gpu_timing[key]+=recorded[i].elapsed_time(recorded[i+1])
                        gpu_timing['samples']+=1
                    gpu_profiles.clear()
                    avg=summarize_metrics(totals,metric_batches)
                    elapsed=time.monotonic()-started
                    memory=memory_snapshot(device) if device.type=='cuda' else {}
                    if device.type=='cuda':
                        memory['inactive_split_gib']=torch.cuda.memory_stats(device).get('inactive_split_bytes.all.current',0)/1024**3
                    print('composer_progress '+json.dumps({'epoch':epoch,'batch':batch_index,'total':total_batches,
                        'optimizer_steps':global_step,'elapsed_sec':time.monotonic()-run_started,
                        'phase_elapsed_sec':elapsed,'batches_per_sec':(batch_index-skip)/max(elapsed,1e-6),
                        'samples_per_sec':seen/max(elapsed,1e-6),'eta_sec':(total_batches-batch_index)*elapsed/(batch_index-skip),
                        'metric_batches':metric_batches,'resume_skipped_batches':skip,
                        'recent_metrics':summarize_metrics(recent_totals,recent_batches),
                        'recent_metric_batches':recent_batches,'learning_rate':optimizer.param_groups[0]['lr'],
                        'optimized_loss':float(optimized_total)/max(1,optimized_updates),
                        'optimized_updates':optimized_updates,
                        'last_gradient_norm':float(last_gradient_norm) if last_gradient_norm is not None else None,
                        'timing':dict(timing),'cache':dataset.cache_stats(),'cuda_memory':memory,
                        'gpu_timing':dict(gpu_timing),
                        'cuda_cache_trim_count':cuda_cache_guard.trim_count,
                        'prefetch':batches.stats() if hasattr(batches,'stats') else {'mode':'synchronous'},
                        'host_memory':memory_gib(host_memory()),**avg}),flush=True)
                    recent_totals={};recent_batches=0
        stamp=time.perf_counter()
        if validation_dataset is not None:
            from .evaluation import evaluate_teacher_forced
            validation=evaluate_teacher_forced(model,validation_dataset,device=device,
                batch_size=args.validation_batch_size,max_batches=args.validation_max_batches,
                precision=precision,loss_kwargs={'action_objective_policy':args.action_objective_policy,
                    'audio_pointer_weight':args.audio_pointer_weight,'placement_weight':args.placement_weight,
                    'planner_weight':args.planner_weight})
            validation.update(epoch=epoch,optimizer_steps=global_step,
                manifest=str(Path(args.validation_manifest).resolve()),
                catalog_coverage=getattr(validation_dataset,'catalog_coverage',None))
            (out/'validation_latest.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding='utf-8')
            with (out/'validation_history.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(validation,ensure_ascii=False)+'\n')
            print('composer_validation '+json.dumps(validation),flush=True)
        common=dict(epoch=epoch,args=vars(args),manifest=args.manifest,
            metrics=summarize_metrics(totals,metric_batches),optimizer_steps=global_step,
            dataset_fingerprint=dataset.fingerprint,atom_catalog=dataset.codec_atoms,
            training_state={'epoch_complete':True,'batch_in_epoch':total_batches,
                'lr_schedule':lr_plan,
                'recovery_episodes':recovery_episode_state(model),
                'sample_in_epoch':min(total_batches*args.batch_size,len(dataset)),
                'sampler_batch_size':sampler_size,
                'loader_generator_state':loader_generator.get_state(),'torch_rng_state':torch.get_rng_state(),
                'cuda_rng_state':torch.cuda.get_rng_state(device) if device.type=='cuda' else None,
                'scaler':scaler.state_dict() if scaler is not None else None})
        _assert_model_finite(model)
        save(out/'composer_last.pt',model,optimizer,**common)
        if args.export_inference: save(out/'composer_inference.pt',model,None,**common)
        timing['save_sec']+=time.perf_counter()-stamp
        print('composer_epoch '+json.dumps({'epoch':epoch,'optimizer_steps':global_step,
            'elapsed_sec':time.monotonic()-started,'timing':dict(timing)}),flush=True)
        if (args.validation_manifest and args.generation_eval_songs
                and epoch%args.generation_eval_interval_epochs==0):
            from .evaluation import run_generation_panel
            panel=run_generation_panel(out/'composer_last.pt',args.validation_manifest,
                out/f'generation_eval_epoch_{epoch}',song_limit=args.generation_eval_songs,
                seeds=generation_seeds,device=str(device))
            print('composer_generation_validation '+json.dumps(panel),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
