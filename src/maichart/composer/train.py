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
from .model import ChartComposer,ComposerConfig,build_model,GROUNDED_PROFILE,WORKLOAD_PROFILES
from .events import HEAD_TIMING_DIM
from .checkpoint import save,load,migrate_music_context,migrate_audio_grounded
from .runtime_memory import GIB,host_memory,memory_gib


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
    p.add_argument('--cache-dir',default='cache/chart_composer_v1');p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--batch-size',type=int,default=4);p.add_argument('--lr',type=float,default=2e-4)
    p.add_argument('--d-model',type=int,default=192);p.add_argument('--layers',type=int,default=3)
    p.add_argument('--nhead',type=int,default=6);p.add_argument('--dim-feedforward',type=int,default=576)
    p.add_argument('--dropout',type=float,default=.1);p.add_argument('--history-events',type=int,default=128)
    p.add_argument('--candidate-count',type=int,default=96);p.add_argument('--action-slots',type=int,default=8)
    p.add_argument('--device',default='auto');p.add_argument('--precision',choices=('auto','fp32','bf16','fp16'),default='auto')
    p.add_argument('--seed',type=int,default=0);p.add_argument('--limit-charts',type=int,default=0)
    p.add_argument('--prepare-workers',type=int,default=4);p.add_argument('--audio-cache-gb',type=float,default=4.)
    p.add_argument('--checkpoint-interval-steps',type=int,default=500)
    continuation=p.add_mutually_exclusive_group()
    continuation.add_argument('--resume');continuation.add_argument('--init-checkpoint')
    p.add_argument('--rebatch-resume',action='store_true',
        help='explicitly change resume microbatch/accumulation while preserving effective batch and current epoch sample sequence')
    p.add_argument('--grad-accumulation',type=int,default=1)
    p.add_argument('--prefetch-batches',type=int,choices=range(0,17),default=2)
    p.add_argument('--sample-workers',type=int,choices=range(1,9),default=4,
                   help='parallel CPU sample preparation threads; CUDA work stays on the main thread')
    p.add_argument('--sample-locality',type=int,choices=range(1,9),default=4,
                   help='songs interleaved per batch; fewer improves locality but increases same-song gradient correlation')
    p.add_argument('--log-interval',type=int,default=25)
    p.add_argument('--max-steps',type=int,default=0,help='maximum batches per epoch; 0 uses every batch')
    p.add_argument('--optimizer-implementation',choices=('auto','standard','foreach','fused'),default='auto')
    p.add_argument('--export-inference',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--symbolic-dropout',type=float,default=.15,
                   help='probability of coherently hiding chart history/state/ledger, retaining audio and targets')
    p.add_argument('--audio-pointer-weight',type=float,default=.15,
                   help='auxiliary next-position supervision using audio/condition without chart context')
    p.add_argument('--profile-steps',type=int,default=0,
                   help='record CUDA H2D/forward/backward/update timings for first N consumed batches; 0 disables')
    p.add_argument('--architecture-profile',choices=('music_context_v2',GROUNDED_PROFILE),default=GROUNDED_PROFILE)
    p.add_argument('--lens-width',type=int,default=64)
    p.add_argument('--lens-chunk-size',type=int,default=32)
    p.add_argument('--history-residual-bound',type=float,default=1.)
    p.add_argument('--history-dropout',type=float,default=.3)
    p.add_argument('--ledger-dropout',type=float,default=.5)
    p.add_argument('--placement-weight',type=float,default=.15)
    p.add_argument('--planner-weight',type=float,default=.2)
    p.add_argument('--recovery-interval',type=int,default=100,help='optimizer updates between short recovery batches; 0 disables')
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


def _loss(output,batch,atom_vocab_size, *, audio_pointer_weight=.15,placement_weight=.15,planner_weight=.2):
    weights=batch.get('sample_weight',torch.ones_like(batch['eos_target']))
    transitions=batch.get('transition_weight',torch.ones_like(weights))
    def reduce(values, factors=weights):
        return (values * factors).sum()/factors.sum().clamp_min(1e-6)
    pointer_weights=weights*transitions*batch.get('pointer_weight',torch.ones_like(weights))
    pointer=reduce(F.cross_entropy(output['candidate_logits'],batch['candidate_target'],reduction='none'),
                   pointer_weights)
    target=batch['candidate_target']
    if output['action_logits'].ndim==3:
        action_logits,future,eos=output['action_logits'],output['future'],output['eos_logits']
    else:
        rows=torch.arange(target.size(0),device=target.device)
        action_logits=output['action_logits'][rows,target]
        future=output['future'][rows,target]
        eos=output['eos_logits'][rows,target]
    action_targets=batch['action_target']
    action_per_slot=F.cross_entropy(action_logits.reshape(-1,atom_vocab_size),
        action_targets.reshape(-1),reduction='none').reshape(action_logits.size(0),-1)
    active_slots=torch.ones_like(action_targets,dtype=torch.bool)
    if action_targets.size(1)>1:
        active_slots[:,1:]=action_targets[:,:-1].ne(0)
    slot_weights=torch.where(action_targets.ne(0),1.,.35)*active_slots.to(action_per_slot)
    action_weights=weights*transitions*batch.get('action_weight',torch.ones_like(weights))
    action=reduce((action_per_slot*slot_weights).sum(-1)/slot_weights.sum(-1).clamp_min(1e-6),
                  action_weights)
    audio_action_loss=pointer.new_zeros(())
    if 'audio_action_logits' in output:
        audio_action_logits=output['audio_action_logits']
        if audio_action_logits.ndim>3:
            rows=torch.arange(target.size(0),device=target.device)
            audio_action_logits=audio_action_logits[rows,target]
        audio_per_slot=F.cross_entropy(audio_action_logits.reshape(-1,atom_vocab_size),
            action_targets.reshape(-1),reduction='none').reshape(audio_action_logits.size(0),-1)
        audio_action_loss=reduce((audio_per_slot*slot_weights).sum(-1)/slot_weights.sum(-1).clamp_min(1e-6),
                                 action_weights)
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
        'pointer_loss':pointer.detach(),'action_loss':action.detach(),'eos_loss':eos_loss.detach(),
        'future_loss':future_loss.detach(),'workload_loss':workload_loss.detach(),
        'curve_slope_loss':curve_slope_loss.detach(),'workload_rate_mae':workload_rate_mae.detach(),
        'audio_pointer_loss':audio_pointer_loss.detach(),'audio_action_loss':audio_action_loss.detach(),
        'trailing_workload_loss':trailing_loss.detach(),
        'placement_loss':placement_loss.detach(),'planner_loss':plan_loss.detach(),
        'pointer_accuracy':(output['candidate_logits'].argmax(-1)==batch['candidate_target']).float().mean().detach()}


def _nonfinite_tensors(mapping):
    items=[(key,value) for key,value in mapping.items() if torch.is_tensor(value)]
    if not items:return []
    # One device-to-host synchronization, rather than one per tensor/layer.
    flags=torch.stack([torch.isfinite(value).all() for _,value in items]).cpu().tolist()
    return [key for (key,_),finite in zip(items,flags) if not finite]


def _assert_model_finite(model):
    bad = _nonfinite_tensors(model.state_dict())
    if bad:
        raise FloatingPointError('Composer model became non-finite: ' + ', '.join(bad[:8]))


def _forward_batch(model,tensors,*,decode_actions=True,**overrides):
    extra={key:tensors[key] for key in ('head_timing','workload_audio','ledger','interval_audio',
        'trailing_audio','audio_lens','audio_lens_mask','planner_audio','planner_mask','planner_indices','planner_mix') if key in tensors}
    extra.update(overrides)
    return model(tensors['candidate_features'],tensors['history'],tensors['condition'],
        action_state=tensors['action_state'],history_features=tensors['history_features'],
        candidate_mask=tensors['candidate_padding_mask'],history_padding_mask=tensors['history_padding_mask'],
        candidate_indices=tensors['candidate_target'],action_targets=tensors['action_target'],decode_actions=decode_actions,**extra)


def recovery_batch(dataset,model,args,device,global_step):
    from .recovery import RecoveryBuilder
    from .generate import _sample_actions,_head_metadata,_active_constraints,_truncate_probabilities
    from .events import event_metadata
    from maichart.generation.event_cli import _NoteGrammar
    from maichart.event_timing import EventPoint
    atoms=dataset.codec_atoms;grammar=dataset.action_grammar
    metadata=_head_metadata(atoms,grammar)
    rng=torch.Generator(device=device).manual_seed(args.seed+global_step+191)
    planner_cache={}
    def propose(item,index):
        candidates=item['_candidate_beats']
        constraints=[_active_constraints(item['_action_index'],beat,item['_tempo'],item['_first'],
            clock=item['_clock']) for beat in candidates]
        item['candidate_mask']=item['candidate_mask'] | torch.tensor([len(held)>=2 for held,_,_ in constraints])
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
            pointer_probs=_truncate_probabilities(logits.softmax(-1),top_p=.92,min_p=.05,max_k=16)
            chosen=int(torch.multinomial(pointer_probs,1,generator=rng))
            beat=candidates[chosen]
            cand_sec=float(item['_clock'].at(beat))
            context=result['candidate_context'][0,chosen]
            audio_ctx=result['audio_action_context'][0,chosen] if 'audio_action_context' in result else None
            occupied,blocked,_=constraints[chosen]
            cooldown_blocked=set(blocked)|{pad for pad,prior in guard.last_head_seconds.items()
                if cand_sec-prior<guard.min_same_head_sec-1e-9}
            blocked_first=set()
            for _ in range(6):
                selected=_sample_actions(context.new_zeros(model.config.action_slots,len(atoms)+1),atoms,grammar,.85,rng,
                    slot_decoder=lambda slot,prefix:model.action_slot(context,slot,[dataset.atom_to_id[x] for x in prefix],audio_context=audio_ctx),
                    occupied=occupied,blocked=cooldown_blocked,blocked_first=blocked_first,head_metadata=metadata,canonical_order=True)
                if selected:
                    point=EventPoint(candidates[chosen],'/'.join(selected))
                    if guard.accepts(point):return point
                    blocked_first.add(dataset.atom_to_id[selected[0]])
                    if guard.last_reason=='repeated_head_too_soon':
                        for head in event_metadata('/'.join(selected)).heads:
                            cooldown_blocked.add(f'A{head}' if str(head).isdigit() else str(head))
            logits[chosen]=float('-inf')
        return None
    indices=np.random.default_rng(args.seed+global_step+991).integers(0,len(dataset),size=args.recovery_samples)
    was_training=model.training
    model.eval()
    try:
        with torch.no_grad():
            return RecoveryBuilder(dataset,model,device,propose=propose).batch_from_indices(indices,steps=args.recovery_steps)
    finally:model.train(was_training)


def main(argv=None):
    import sys
    arguments=list(sys.argv[1:] if argv is None else argv)
    explicit_flags={token.split('=',1)[0] for token in arguments if token.startswith('--')}
    args=parser().parse_args(arguments);device=_device(args.device);precision=_precision(args.precision,device)
    if args.rebatch_resume and (not args.resume or args.max_steps):
        raise ValueError('--rebatch-resume requires --resume and full epochs (max_steps=0)')
    if args.epochs<1 or args.batch_size<1 or args.grad_accumulation<1 or args.log_interval<1 or args.max_steps<0 or args.profile_steps<0:
        raise ValueError('epochs, batch_size, grad_accumulation and log_interval must be positive; max_steps/profile_steps must be nonnegative')
    if not math.isfinite(args.lr) or args.lr<=0:
        raise ValueError('lr must be positive and finite')
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
    if args.resume or args.init_checkpoint:
        model,old=load(args.resume or args.init_checkpoint,device='cpu',include_optimizer=bool(args.resume))
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
            # Resume is an exact continuation: restore the saved contract
            # before dataset construction so CLI defaults do not conflict.
            for key in ('sample_locality','symbolic_dropout','audio_pointer_weight','placement_weight','planner_weight',
                        'recovery_interval','recovery_samples','recovery_steps','recovery_weight','history_dropout',
                        'ledger_dropout','history_residual_bound','lens_width','lens_chunk_size'):
                if key in (old.get('args') or {}) and '--'+key.replace('_','-') not in explicit_flags:
                    setattr(args,key,old['args'][key])
        semantics=GROUNDED_TRAINING_SEMANTICS if args.architecture_profile==GROUNDED_PROFILE else TRAINING_SEMANTICS
        if args.resume and old.get('training_semantics')!=semantics:
            raise ValueError('Composer training contract changed; use --init-checkpoint with a new output directory')
    dataset=ComposerDataset(args.manifest,cache_dir=args.cache_dir,candidate_count=args.candidate_count,
        history_events=args.history_events,action_slots=args.action_slots,preparation_workers=args.prepare_workers,
        audio_cache_gb=args.audio_cache_gb,limit_charts=args.limit_charts,seed=args.seed,architecture_profile=args.architecture_profile)
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
                history_dropout=args.history_dropout,ledger_dropout=args.ledger_dropout)
                if args.architecture_profile==GROUNDED_PROFILE else migrate_music_context(model,action_slots=args.action_slots))
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
            head_timing_dim=HEAD_TIMING_DIM,lens_width=args.lens_width,lens_chunk_size=args.lens_chunk_size,
            history_residual_bound=args.history_residual_bound,history_dropout=args.history_dropout,
            ledger_dropout=args.ledger_dropout),dataset.codec_atoms)
    model.to(device).train()
    cuda_cache_guard=CudaCacheGuard(device,0.)
    if model.config.architecture_profile==GROUNDED_PROFILE:
        for key in ('lens_width','lens_chunk_size','history_residual_bound','history_dropout','ledger_dropout'):
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
    if args.resume and old:
        for key in ('max_steps',):
            stored=old.get('args',{}).get(key,1 if key=='grad_accumulation' else 0)
            if partial_resume and stored!=getattr(args,key):
                raise ValueError(f'Mid-epoch resume requires the saved {key}={stored}; use finetune to change it')
        for key in ('sample_locality','symbolic_dropout','audio_pointer_weight','placement_weight','planner_weight',
                    'recovery_interval','recovery_samples','recovery_steps','recovery_weight','history_dropout',
                    'ledger_dropout','history_residual_bound','lens_width','lens_chunk_size'):
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
    print('composer_runtime '+json.dumps({'family':FAMILY,'training_semantics':GROUNDED_TRAINING_SEMANTICS if model.config.architecture_profile==GROUNDED_PROFILE else TRAINING_SEMANTICS,
        'device':str(device),'precision':precision,'effective_lr':effective_lr,'samples':len(dataset),'parameters':parameters,
        'model_bytes':sum(p.numel()*p.element_size() for p in model.parameters()),
        'action_head_parameters':sum(p.numel() for p in model.action_head.parameters()),
        'selected_candidate_projection':True,'optimizer_implementation':implementation,
        'audio_bank_mode':'lossless_mmap_v1','audio_feature_policy':COMPOSER_AUDIO_FEATURE_POLICY,
        'batch_size':args.batch_size,'grad_accumulation':args.grad_accumulation,
        'start_epoch':start_epoch,'prefetch_batches':args.prefetch_batches,
        'sample_workers':args.sample_workers,'sample_locality':args.sample_locality,
        'cuda_cache_guard':cuda_cache_guard.policy(),
        'cuda_allocator_config':os.environ.get('PYTORCH_CUDA_ALLOC_CONF'),
        'resume_sample_cursor':resume_cursor,'resume_sampler_batch_size':resume_sampler_size,
        'rebatch_resume':args.rebatch_resume,
        'torch_cpu_threads':torch.get_num_threads(),'python_switch_interval_sec':sys.getswitchinterval(),
        'architecture_profile':model.config.architecture_profile,'symbolic_dropout':model.config.symbolic_dropout,
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
                if (batch_index-1)%args.grad_accumulation==0: window_valid=True
                if not window_valid: continue
                events=[torch.cuda.Event(enable_timing=True) for _ in range(5)] if device.type=='cuda' and len(gpu_profiles)+gpu_timing['samples']<args.profile_steps else None
                if events: events[0].record()
                stamp=time.perf_counter()
                tensors={key:value.to(device,non_blocking=device.type=='cuda')
                         for key,value in batch.items() if torch.is_tensor(value)
                         and key not in {'song_index','event_index'}}
                timing['transfer_sec']+=time.perf_counter()-stamp
                if events: events[1].record()
                stamp=time.perf_counter()
                with torch.autocast('cuda',dtype=torch.bfloat16 if precision=='bf16' else torch.float16,
                                    enabled=device.type=='cuda' and precision in {'bf16','fp16'}):
                    output=_forward_batch(model,tensors)
                    output_bad=_nonfinite_tensors(output)
                    loss,metrics=_loss(output,tensors,model.config.atom_vocab_size,audio_pointer_weight=args.audio_pointer_weight,
                        placement_weight=args.placement_weight,planner_weight=args.planner_weight)
                if events: events[2].record()
                if output_bad or not torch.isfinite(loss):
                    details={'message':'non-finite batch skipped; checkpoint remains valid',
                        'epoch':epoch,'batch':batch_index,'output_tensors':output_bad}
                    for key in ('song_index','event_index'):
                        if key in batch:
                            details[key]=batch[key].detach().cpu().tolist()
                    print('composer_warning '+json.dumps({'message':'non-finite batch skipped; checkpoint remains valid',
                        **details}),flush=True)
                    optimizer.zero_grad(set_to_none=True)
                    window_valid=False
                    continue
                window_start=((batch_index-1)//args.grad_accumulation)*args.grad_accumulation
                window_batches=min(args.grad_accumulation,total_batches-window_start)
                backward_loss=loss/window_batches
                if not torch.isfinite(backward_loss):
                    print('composer_nonfinite_loss ' + json.dumps({'epoch':epoch,'batch':batch_index,
                        'optimizer_steps':global_step,'loss':float(loss.detach().cpu()) if torch.isfinite(loss) else None}),flush=True)
                    optimizer.zero_grad(set_to_none=True)
                    window_valid=False
                    continue
                if scaler is not None: scaler.scale(backward_loss).backward()
                else: backward_loss.backward()
                if events: events[3].record()
                update=(batch_index%args.grad_accumulation==0 or batch_index==total_batches)
                if update:
                    if (model.config.architecture_profile==GROUNDED_PROFILE and args.recovery_interval and args.recovery_weight>0
                            and (global_step+1)%args.recovery_interval==0):
                        recovery_started=time.perf_counter()
                        recovered,diagnostics=recovery_batch(dataset,model,args,device,global_step)
                        recovery_value=None
                        if recovered is not None:
                            recovered={k:v.to(device) for k,v in recovered.items() if torch.is_tensor(v)}
                            with torch.autocast('cuda',dtype=torch.bfloat16 if precision=='bf16' else torch.float16,
                                    enabled=device.type=='cuda' and precision in {'bf16','fp16'}):
                                recovery_output=_forward_batch(model,recovered,decode_actions=False)
                                per_sample=F.cross_entropy(recovery_output['candidate_logits'],recovered['candidate_target'],reduction='none')
                                importance=recovered['sample_weight']*recovered['pointer_weight']
                                recovery_loss=(per_sample*importance).sum()/importance.sum().clamp_min(1e-6)*args.recovery_weight
                            if torch.isfinite(recovery_loss):
                                recovery_value=float(recovery_loss.detach())
                                if scaler is not None:scaler.scale(recovery_loss).backward()
                                else:recovery_loss.backward()
                        print('composer_recovery '+json.dumps({'optimizer_step':global_step+1,
                            'seconds':time.perf_counter()-recovery_started,'outcomes':diagnostics,
                            'loss':recovery_value,'loss_includes_weight':True}),flush=True)
                    if scaler is not None: scaler.unscale_(optimizer)
                    # The global norm detects NaN/Inf across all gradients;
                    # no per-parameter Python scalar checks or mutation needed.
                    grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=False)
                    if not torch.isfinite(grad_norm):
                        print('composer_warning '+json.dumps({'message':'non-finite gradients skipped; checkpoint remains valid',
                            'epoch':epoch,'batch':batch_index}),flush=True)
                        optimizer.zero_grad(set_to_none=True)
                        if scaler is not None: scaler.update()
                        continue
                    if scaler is not None: scaler.step(optimizer);scaler.update()
                    else: optimizer.step()
                    # AdamW can still create non-finite parameters even when
                    # the gradient was finite (notably after BF16 overflow in
                    # an optimizer state).  Refuse to write such a checkpoint;
                    # the previous checkpoint remains the valid recovery point.
                    _assert_model_finite(model)
                    optimizer.zero_grad(set_to_none=True);global_step+=1
                    cache_started=time.perf_counter()
                    cache_event=cuda_cache_guard.check()
                    if cache_event is not None:
                        print('composer_cuda_cache_trim '+json.dumps({'optimizer_steps':global_step,
                            'seconds':time.perf_counter()-cache_started,**cache_event}),flush=True)
                if events:
                    events[4].record();gpu_profiles.append(events)
                timing['compute_sec']+=time.perf_counter()-stamp
                seen+=int(tensors['candidate_target'].numel())
                metric_batches+=1
                for key,value in metrics.items():
                    totals[key]=totals[key]+value.float() if key in totals else value.float().clone()
                if update and args.checkpoint_interval_steps and global_step%args.checkpoint_interval_steps==0:
                    _assert_model_finite(model)
                    stamp=time.perf_counter()
                    save(out/'composer_last.pt',model,optimizer,epoch=epoch,args=vars(args),manifest=args.manifest,
                         metrics={key:float(value)/max(1,metric_batches) for key,value in totals.items()},optimizer_steps=global_step,
                         dataset_fingerprint=dataset.fingerprint,atom_catalog=dataset.codec_atoms,
                         training_state={'epoch_complete':batch_index==total_batches,'batch_in_epoch':batch_index,
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
                if batch_index%args.log_interval==0 or batch_index in {1,total_batches}:
                    for recorded in gpu_profiles:
                        recorded[-1].synchronize()
                        for i,key in enumerate(('h2d_ms','forward_ms','backward_ms','update_ms')):
                            gpu_timing[key]+=recorded[i].elapsed_time(recorded[i+1])
                        gpu_timing['samples']+=1
                    gpu_profiles.clear()
                    avg={key:float(value)/max(1,metric_batches) for key,value in totals.items()}
                    elapsed=time.monotonic()-started
                    memory=memory_snapshot(device) if device.type=='cuda' else {}
                    if device.type=='cuda':
                        memory['inactive_split_gib']=torch.cuda.memory_stats(device).get('inactive_split_bytes.all.current',0)/1024**3
                    print('composer_progress '+json.dumps({'epoch':epoch,'batch':batch_index,'total':total_batches,
                        'optimizer_steps':global_step,'elapsed_sec':time.monotonic()-run_started,
                        'phase_elapsed_sec':elapsed,'batches_per_sec':(batch_index-skip)/max(elapsed,1e-6),
                        'samples_per_sec':seen/max(elapsed,1e-6),'eta_sec':(total_batches-batch_index)*elapsed/(batch_index-skip),
                        'metric_batches':metric_batches,'resume_skipped_batches':skip,
                        'timing':dict(timing),'cache':dataset.cache_stats(),'cuda_memory':memory,
                        'gpu_timing':dict(gpu_timing),
                        'cuda_cache_trim_count':cuda_cache_guard.trim_count,
                        'prefetch':batches.stats() if hasattr(batches,'stats') else {'mode':'synchronous'},
                        'host_memory':memory_gib(host_memory()),**avg}),flush=True)
        stamp=time.perf_counter()
        common=dict(epoch=epoch,args=vars(args),manifest=args.manifest,
            metrics={key:float(value)/max(1,metric_batches) for key,value in totals.items()},optimizer_steps=global_step,
            dataset_fingerprint=dataset.fingerprint,atom_catalog=dataset.codec_atoms,
            training_state={'epoch_complete':True,'batch_in_epoch':total_batches,
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
    return 0


if __name__=='__main__':raise SystemExit(main())
