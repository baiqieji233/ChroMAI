"""Small read-only Composer interventions; no training or quality gates.

python -m maichart.composer.diagnose --checkpoint ... --manifest ... --output ...
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path

import numpy as np
import torch

from .checkpoint import load
from .data import ComposerDataset, collate_composer
from .candidates import candidate_beats, candidate_features, interp_candidate_audio, _select_candidate_offsets
from .workload import prefix_ledger
from .events import head_timing_features
from .model import MUSIC_PROFILES,WORKLOAD_PROFILES,GROUNDED_PROFILE
from .acoustics import AcousticContext


def _js(left, right):
    left=left.float().softmax(-1);right=right.float().softmax(-1)
    mean=(left+right)*.5
    return float((.5*(left*(left.clamp_min(1e-12).log()-mean.clamp_min(1e-12).log())).sum(-1)
                  +.5*(right*(right.clamp_min(1e-12).log()-mean.clamp_min(1e-12).log())).sum(-1)).mean())


def phase_recovery(previous, target, *, tempo, first, end_sec, count, policy, clock):
    """Exact bounded geometry check; it does not certify learned recovery."""
    values=candidate_beats(previous,tempo=tempo,first=first,audio_end_sec=end_sec,
                          candidate_count=count,policy=policy,clock=clock)
    if target in values: return 1
    for value in values:
        if value>=target: break
        if target in candidate_beats(value,tempo=tempo,first=first,audio_end_sec=end_sec,
                                     candidate_count=count,policy=policy,clock=clock):
            return 2
    return None


def _inputs(batch, device):
    mapping={'candidate_features':'candidate_features','history':'history','condition':'condition',
        'action_state':'action_state','history_features':'history_features',
        'candidate_mask':'candidate_padding_mask','history_padding_mask':'history_padding_mask',
        'workload_audio':'workload_audio','ledger':'ledger','interval_audio':'interval_audio',
        'trailing_audio':'trailing_audio','head_timing':'head_timing',
        **{k:k for k in ('audio_lens','audio_lens_mask','planner_audio','planner_mask','planner_indices','planner_mix',
                          'selected_action_state','motif_ids','motif_relative_beats','motif_mask',
                          'structure_features','structure_transition','history_beat_ages','history_second_ages',
                          'song_unique_indices','song_inverse','song_audio','song_mask','song_clock',
                          'survival_audio','survival_edges','survival_candidate_seconds',
                          'survival_start_seconds','survival_plan_indices','survival_plan_mix',
                          'survival_prefix_counts')}}
    result={key:batch[source].to(device) for key,source in mapping.items() if source in batch}
    if 'quality_lifecycle' in batch or 'quality_future_audio' in batch:
        from .quality_inputs import quality_forward_inputs
        result['quality_inputs']={key:value.to(device) for key,value in quality_forward_inputs(batch).items()}
    return result


@torch.inference_mode()
def interventions(model, inputs, *, decode_context=None):
    """Hold audio/history/candidates fixed for workload interventions.

    Oracle is a separate optional caller diagnostic, never a production input.
    Layout JS is evaluated at the SAME base-selected candidate in every case.
    Selected-time state and enabled motif memory require a callback that builds
    causal context at that candidate; authored-target context is not equivalent.
    """
    if decode_context is None and (model.config.action_state_policy=='selected_prefix_v3'
                                   or hasattr(model,'motif_memory')):
        raise ValueError('Selected state or motif diagnostics require a causal decode_context callback')
    model.eval()
    base=model(**inputs,decode_actions=False)
    choice=base['candidate_logits'].argmax(-1)
    def decode(result, current_inputs):
        extra={} if decode_context is None else decode_context(choice,current_inputs)
        if hasattr(model,'motif_memory') and 'symbolic_keep' in current_inputs:
            extra['motif_keep']=current_inputs['symbolic_keep']
        if 'selected_state_base' in result:
            extra.setdefault('state_base',result['selected_state_base'])
            extra.setdefault('state_keep',result.get('selected_state_keep'))
        if 'previous_heads' in result:extra['previous_heads']=result['previous_heads']
        if 'echo_lane_bias' in result:extra['echo_heads']=result['echo_lane_bias']
        if 'button_load_features' in result:extra['button_load_features']=result['button_load_features']
        return model.decode_candidate(result['candidate_context'],choice,**extra)['action_logits'][...,0,:]
    baseline=decode(base,inputs)
    workload=base.get('workload')
    variants={}
    valid=(~inputs['candidate_mask'] if 'candidate_mask' in inputs else
           torch.ones(inputs['candidate_features'].shape[:2],dtype=torch.bool,device=choice.device))
    reverse=torch.arange(valid.size(1),device=valid.device)[None].expand_as(valid).clone()
    for row in range(valid.size(0)):
        indices=torch.where(valid[row])[0]
        reverse[row,indices]=indices.flip(0)
    def reversed_candidates(tensor):
        result=tensor.clone()
        shape=(*reverse.shape,*([1]*(tensor.ndim-2)))
        result[:,:valid.size(1)]=tensor[:,:valid.size(1)].gather(1,reverse.reshape(shape).expand(-1,-1,*tensor.shape[2:]))
        return result
    reversed_features=inputs['candidate_features'].clone()
    # Columns 102:120 contain progress/tempo/meter, not acoustic evidence.
    # Keep them and all 21 explicit timing columns fixed during this probe.
    acoustic_columns=min(102,model.config.audio_dim)
    reversed_features[...,:acoustic_columns]=reversed_candidates(reversed_features[...,:acoustic_columns])
    acoustic_change={'candidate_features':reversed_features}
    for key in ('workload_audio','interval_audio','trailing_audio','audio_lens','audio_lens_mask'):
        if key in inputs:acoustic_change[key]=reversed_candidates(inputs[key])
    for key in ('structure_features','structure_transition'):
        if key in inputs:
            # The first six columns are chart-time phases. Reorder only the
            # audio-derived structure evidence while holding that clock fixed.
            changed=inputs[key].clone()
            changed[...,6:]=reversed_candidates(inputs[key][...,6:])
            acoustic_change[key]=changed
    variants['candidate_acoustics_reordered_fixed_timing_history']=acoustic_change
    if workload is not None:
        # Workload rows are [candidate positions, uniform whole-song probes].
        # Keep the uniform branch fixed: this probe measures sensitivity to
        # candidate-local workload evidence, rather than changing sampling
        # coverage at the same time. Padded candidate rows stay untouched.
        candidate_count=inputs['candidate_features'].size(1)
        candidate_workload=workload[:, :candidate_count]
        valid_candidate=(~inputs['candidate_mask'] if 'candidate_mask' in inputs else
                         torch.ones(candidate_workload.shape[:2],dtype=torch.bool,
                                    device=candidate_workload.device))
        swapped=workload.clone()
        valid_count=valid_candidate.sum(1).clamp_min(1)
        for row in range(candidate_workload.size(0)):
            indices=torch.where(valid_candidate[row])[0]
            if indices.numel()>1:
                swapped[row, indices]=candidate_workload[row, indices.flip(0)]
        variants['workload_swapped']={'workload_override':swapped}
        constant=workload.clone()
        candidate_mean=(candidate_workload*valid_candidate[...,None]).sum(1)/valid_count[...,None]
        constant_candidates=torch.where(valid_candidate[...,None],candidate_mean[:,None],candidate_workload)
        constant[:, :candidate_count]=constant_candidates
        variants['workload_constant']={'workload_override':constant}
    if 'ledger' in inputs:
        variants['ledger_zero']={'ledger':torch.zeros_like(inputs['ledger'])}
    if model.config.architecture_profile in MUSIC_PROFILES:
        variants['symbolic_unread']={'symbolic_keep':torch.zeros(inputs['candidate_features'].size(0),1,
                                                                   device=inputs['candidate_features'].device)}
    result={}
    gaps=inputs['candidate_features'][...,model.config.audio_dim+1].float()
    expected=(base['candidate_logits'].float().softmax(-1)*gaps).sum(-1)
    for name,change in variants.items():
        alternate=model(**{**inputs,**change},decode_actions=False)
        action=decode(alternate,{**inputs,**change})
        result[name]={'pointer_js':_js(base['candidate_logits'],alternate['candidate_logits']),
            'first_atom_js_at_fixed_candidate':_js(baseline,action),
            'expected_gap_change_sec':float(((alternate['candidate_logits'].float().softmax(-1)*gaps).sum(-1)-expected).mean())}
    return result


def workload_coverage(dataset):
    """Per-second weights of the uniform probe branch, including masked gaps.

    This audits training coverage, not whether the curve prediction is good.
    Candidate-probe weights in old checkpoints are a separate branch.
    """
    result=[]
    for song_index in sorted({sample.song_index for sample in dataset.samples}):
        song=dataset.base.songs[song_index][0];duration=float(song.audio.duration_sec)
        table=dataset._song_table(song_index)
        count=max(1,int(np.ceil(duration)));raw=np.zeros(count);weighted=np.zeros(count)
        for index,sample in enumerate(dataset.samples):
            if sample.song_index!=song_index:continue
            probes=(np.arange(16)+((index%31)+.5)/31.)*duration/16.
            valid=table['workload_query'].weight_at(probes)
            bins=np.minimum(probes.astype(int),count-1)
            raw+=np.bincount(bins,minlength=count)
            share=1. if dataset.architecture_profile in MUSIC_PROFILES else .5
            weighted+=np.bincount(bins,weights=valid.astype(float)*sample.weight*share/16.,minlength=count)
        result.append({'song_index':song_index,'seconds_bin_width':1.,'raw_uniform_counts':raw.tolist(),
            'effective_uniform_weights':weighted.tolist(),'zero_weight_seconds_bins':np.flatnonzero(weighted==0).tolist(),
            'excluded_regions_are_unknown':True,'candidate_probe_branch_included':False})
    return result


def _omit_last_prefix(dataset, sample, item, id_map):
    """Remove one complete trailing event and recompute all causal inputs.

    There is no suffix CE here: the authored next event is not assumed to be
    the uniquely correct continuation of this altered prefix.
    """
    ev=sample.event_index-1
    if ev<0:return None
    song=dataset.base.songs[sample.song_index][0]
    table=dataset._song_table(sample.song_index);clock=table['clock']
    points=table['points']
    previous=points[ev-1].beat if ev else None
    candidates=candidate_beats(previous,tempo=song.tempo,first=song.first,
        audio_end_sec=dataset._song_audio_ends[sample.song_index],candidate_count=dataset.candidate_count,
        policy=dataset.candidate_policy,clock=clock)
    if not candidates:return None
    actual=len(candidates);candidates+= [candidates[-1]]*(dataset.candidate_count-actual)
    seconds=np.asarray([clock.at(x) for x in candidates])
    values,times=dataset._song_features(sample.song_index)
    key=dataset._song_feature_key(sample.song_index)
    with dataset._feature_lock:
        context=dataset._context_cache.get(key)
    if context is None:
        context=AcousticContext(values,times,song.audio.duration_sec,validated_bank=True)
        with dataset._feature_lock:
            dataset._context_cache[key]=context
            dataset._context_cache.move_to_end(key)
            while len(dataset._context_cache)>dataset._context_cache_limit:
                dataset._context_cache.popitem(last=False)
    start=max(0,ev-dataset.history_events)
    if dataset.action_state_policy=='candidate_prefix_v2':
        from .context_features import composer_action_state
        action_state=composer_action_state(table['state_index'],ev,candidates,dataset.action_state_policy)
    else:
        action_state=dataset._action_state(sample.song_index,ev,table)
    result=dict(item,
        candidate_features=torch.from_numpy(np.concatenate((interp_candidate_audio(values,times,seconds),
            candidate_features(candidates,previous,song.tempo,song.first,clock=clock,candidate_seconds=seconds,rich_time=True)),axis=-1)),
        candidate_mask=torch.from_numpy((np.arange(dataset.candidate_count)>=actual)
            | (seconds < 0.) | (seconds >= dataset._song_audio_ends[sample.song_index])),
        history=torch.from_numpy(id_map[table['atom_ids'][start:ev].numpy()]),
        history_features=table['history_features'][start:ev],
        action_state=action_state,
        workload_audio=torch.from_numpy(context.at(seconds)),
        ledger=torch.from_numpy(prefix_ledger(table['point_seconds'],table['head_prefix'],seconds,committed_count=ev)))
    from .history_clock import history_beat_ages
    result['history_beat_ages']=torch.from_numpy(history_beat_ages([p.beat for p in points[start:ev]]))
    origin=float(table['point_seconds'][ev-1]) if ev else 0.
    result['history_second_ages']=torch.from_numpy((origin-table['point_seconds'][start:ev]).astype(np.float32))
    result.pop('_history_atom_width', None)
    if getattr(dataset,'quality_inputs_enabled',False):
        from .quality_inputs import quality_context_inputs
        policy=getattr(dataset,'quality_inputs_policy',None)
        quality_start=start if policy=='song_paired_v3' else 0
        quality=quality_context_inputs(points=points[quality_start:ev],atom_ids=table['atom_ids'][quality_start:ev],
            include_future=getattr(dataset,'fixed_future_audio',False),
            candidate_beats=candidates,candidate_seconds=seconds,state_index=table['state_index'],
            audio_features=values,audio_times=times,acoustic_context=context,clock=clock,
            audio_duration=table['audio_duration'],visible_before_beat=points[ev].beat,
            point_seconds=table['point_seconds'][quality_start:ev],policy=policy,
            plan_grid=table.get('planner_data'))
        if 'quality_anchor_ids' in quality:
            quality['quality_anchor_ids']=_map_catalog_ids(quality['quality_anchor_ids'],id_map,'omitted-prefix anchors')
        result.update(quality)
        # This intervention is an observation, not an authored continuation
        # target. Do not carry supervision from the unmodified prefix into it.
        result['quality_phrase_weight']=torch.zeros(len(quality['quality_phrase_audio']))
        result['quality_gap_weight']=torch.tensor(0.)
    elif getattr(dataset,'fixed_future_audio',False):
        from .quality_inputs import fixed_future_audio_inputs
        result.update(fixed_future_audio_inputs(context,clock,candidates,table['audio_duration']))
    # Altering the committed prefix invalidates target-derived empty-gap
    # supervision. Keep no authored-gap audio or exposure on this intervention.
    for key in ('empty_gap_audio','empty_gap_exposure','empty_gap_weight'):
        if key in result:
            result[key]=torch.zeros_like(result[key])
    # These are selected-beat inputs. The old sample's target beat no longer
    # identifies the candidate selected from this altered prefix; rebuild them
    # through _decode_context after the new pointer choice instead.
    for key in ('selected_action_state','motif_ids','motif_relative_beats','motif_mask'):
        result.pop(key,None)
    if dataset.architecture_profile in MUSIC_PROFILES:
        result.update(interval_audio=torch.from_numpy(context.interval(float(clock.at(previous)) if previous is not None else 0.,seconds)),
                      trailing_audio=torch.from_numpy(context.trailing(seconds)))
    if ev<len(table['points']):
        result['head_timing']=torch.from_numpy(head_timing_features(table['last_head_seconds'][ev],
            table['last_head_beats'][ev],seconds,[float(x) for x in candidates]))
    if dataset.architecture_profile==GROUNDED_PROFILE:
        result.update(dataset._grounded_features(sample.song_index,seconds,seconds,table=table))
    if getattr(dataset, 'timing_distribution_policy', 'hierarchical_v1') == 'survival_integrated_v2':
        from .density_grid import survival_grid_inputs
        result.update({key: torch.from_numpy(value) for key, value in survival_grid_inputs(
            context, float(clock.at(previous)) if previous is not None else 0., seconds,
            table['audio_duration'], plan_grid=table.get('planner_data'),
            prefix_seconds=(table['point_seconds'][:ev]
                if getattr(dataset, 'density_feedback_policy', 'off') == 'causal_count_debt_v1' else None)).items()})
    # The altered prefix has no authored future/count targets. Preserve only
    # audio evidence and causal state, never labels copied from the source row.
    for key in ('future_target', 'future_weight', 'event_count_target', 'event_exposure',
                'event_rate_weight', 'workload_target', 'workload_weight',
                'trailing_target', 'trailing_weight', 'quality_phrase_target',
                'quality_phrase_weight', 'quality_gap_target', 'quality_gap_weight'):
        if key in result:
            result[key] = torch.zeros_like(result[key])
    if getattr(dataset,'structure_policy','off')!='off':
        from .structure import structure_features_and_transition
        features,transition=structure_features_and_transition(song,candidates,seconds,
            float(previous) if previous is not None else None,
            float(clock.at(previous)) if previous is not None else None,
            policy=dataset.structure_policy,cache_dir=dataset.cache_dir,backend=dataset.structure_backend)
        result['structure_features']=torch.from_numpy(features)
        result['structure_transition']=torch.from_numpy(transition)
    return result,seconds,candidates


def _decode_context(dataset, sample, candidate_beats, id_map, device, *, committed_count=None):
    """Build causal configuration inputs after the pointer has selected a beat."""
    table=dataset._song_table(sample.song_index)
    committed_count=sample.event_index if committed_count is None else committed_count
    def build(choice, _inputs=None):
        # Diagnostic reports currently collate one sample at a time.
        if choice.numel()!=1:
            raise ValueError('Causal decode context currently requires one diagnostic sample')
        beat=candidate_beats[int(choice.item())]
        extra={}
        if dataset.action_state_policy=='selected_prefix_v3':
            from .context_features import selected_action_state
            extra['selected_action_state']=selected_action_state(
                table['state_index'],committed_count,beat)[None].to(device)
        if dataset.motif_memory_policy!='off':
            from .context_features import motif_tensors
            memory=motif_tensors(table['motif_index'],beat,committed_count,
                dataset.history_events,dataset.motif_max_events)
            memory['motif_ids']=_map_catalog_ids(memory['motif_ids'],id_map,'retrieved motif')
            extra.update({key:value[None].to(device) for key,value in memory.items()})
        return extra
    return build


def _map_catalog_ids(values, id_map, label):
    """Map dataset atom IDs without silently turning unknown motifs into STOP."""
    ids=values.detach().cpu().numpy().astype(np.int64,copy=False)
    valid=(ids>=0)&(ids<len(id_map))
    mapped=np.zeros(ids.shape,dtype=np.int64)
    mapped[valid]=id_map[ids[valid]]
    unknown=(ids!=0)&(~valid | (mapped==0))
    if np.any(unknown):
        raise ValueError(f'{label} contains atom IDs absent from the checkpoint catalog')
    return torch.from_numpy(mapped)


def _sample_candidate_beats(dataset, sample):
    offsets,_,_=_select_candidate_offsets(sample.previous is None,sample.valid_candidates,
        dataset.candidate_count,sample.previous,dataset.candidate_policy)
    base=Fraction(0) if sample.previous is None else sample.previous
    candidates=[base+offset for offset in offsets]
    if len(candidates)<dataset.candidate_count:
        candidates.extend([candidates[-1]]*(dataset.candidate_count-len(candidates)))
    return candidates


def _diagnostic_dataset(args, model):
    return ComposerDataset(args.manifest,cache_dir=args.cache_dir,limit_charts=args.limit_charts,
        candidate_count=model.config.candidate_count,history_events=model.config.history_events,
        action_slots=model.config.action_slots,architecture_profile=model.config.architecture_profile,
        action_state_policy=model.config.action_state_policy,
        timing_distribution_policy=getattr(model.config,'timing_distribution_policy','hierarchical_v1'),
        density_feedback_policy=getattr(model.config,'density_feedback_policy','off'),
        head_timing_policy=model.config.head_timing_policy,
        motif_memory_policy=model.config.motif_memory_policy,
        motif_max_events=model.config.motif_max_events,
        structure_policy=getattr(model.config,'structure_policy','off'),
        structure_backend=getattr(model.config,'structure_backend','native'),
        quality_inputs_enabled=getattr(model.config,'configuration_quality_policy','off')!='off',
        quality_inputs_policy=getattr(model.config,'configuration_quality_policy','off'),
        fixed_future_audio=getattr(model.config,'action_head_policy','flat_v1')=='compact_grouped_v3',
        preparation_workers=1,audio_cache_gb=.25)


@torch.inference_mode()
def diagnose(args):
    device=torch.device('cuda' if args.device=='auto' and torch.cuda.is_available() else 'cpu' if args.device=='auto' else args.device)
    model,payload=load(args.checkpoint,device=device);model.eval()
    from .structure_backend import validate_backend_identity
    validate_backend_identity(model.config)
    if model.config.architecture_profile not in WORKLOAD_PROFILES:
        raise ValueError('These interventions require a Composer workload checkpoint')
    dataset=_diagnostic_dataset(args,model)
    if dataset.action_slots != model.config.action_slots:
        raise ValueError('Diagnostic dataset exceeds checkpoint action_slots; finetune an expanded model first')
    catalog={text:i+1 for i,text in enumerate(payload['atom_catalog'])}
    id_map=np.asarray([0]+[catalog.get(text,0) for text in dataset.codec_atoms],dtype=np.int64)
    report={'schema':'composer_interventions_v2_causal_configuration','checkpoint':str(args.checkpoint),'samples':[],
        'model_config':model.config.to_dict(),
        'scope':'fixed-state sensitivity, one-event omission and bounded geometric reachability; not a musical quality certification',
        'workload_probe_scope':'candidate valid rows only; uniform whole-song rows fixed',
        'candidate_misses':dataset.candidate_miss_categories}
    report['workload_coverage']=workload_coverage(dataset)
    selected=np.unique(np.linspace(0,len(dataset)-1,min(args.samples,len(dataset)),dtype=int))
    for index in selected:
        sample=dataset.samples[int(index)];item=dataset[int(index)]
        if (id_map[item['history'].numpy()]==0).sum() > (item['history'].numpy()==0).sum(): continue
        item['history']=torch.from_numpy(id_map[item['history'].numpy()])
        if 'motif_ids' in item:
            item['motif_ids']=_map_catalog_ids(item['motif_ids'],id_map,'sample motif')
        if 'quality_anchor_ids' in item:
            item['quality_anchor_ids']=_map_catalog_ids(item['quality_anchor_ids'],id_map,'sample quality anchors')
        batch=collate_composer([item]);inputs=_inputs(batch,device)
        if not model.config.head_timing_dim:inputs.pop('head_timing',None)
        current_beats=_sample_candidate_beats(dataset,sample)
        result=interventions(model,inputs,decode_context=_decode_context(
            dataset,sample,current_beats,id_map,device))
        table=dataset._song_table(sample.song_index);song=dataset.base.songs[sample.song_index][0]
        previous=sample.previous or Fraction(0)
        recovery=[]
        for offset in (Fraction(1,48),Fraction(1,12),Fraction(1,4)):
            shifted=previous+offset
            if shifted>=sample.target:continue
            recovery.append({'previous':str(shifted),'target':str(sample.target),
                'steps_within_two':phase_recovery(shifted,sample.target,tempo=song.tempo,first=song.first,
                    end_sec=dataset._song_audio_ends[sample.song_index],count=dataset.candidate_count,
                    policy=dataset.candidate_policy,clock=table['clock'])})
        omitted=_omit_last_prefix(dataset,sample,item,id_map)
        if omitted is not None:
            changed,seconds,omitted_beats=omitted
            changed_inputs=_inputs(collate_composer([changed]),device)
            if not model.config.head_timing_dim:changed_inputs.pop('head_timing',None)
            probe=model(**changed_inputs,decode_actions=False)
            probability=probe['candidate_logits'].float().softmax(-1)[0].cpu().numpy()
            omitted_choice=probe['candidate_logits'].argmax(-1)
            decode_extra=_decode_context(dataset,sample,omitted_beats,id_map,device,
                committed_count=sample.event_index-1)(omitted_choice)
            if 'selected_state_base' in probe:
                decode_extra.update(state_base=probe['selected_state_base'],
                    state_keep=probe.get('selected_state_keep'))
            if 'previous_heads' in probe:decode_extra['previous_heads']=probe['previous_heads']
            if 'echo_lane_bias' in probe:decode_extra['echo_heads']=probe['echo_lane_bias']
            if 'button_load_features' in probe:decode_extra['button_load_features']=probe['button_load_features']
            atom_logits=model.decode_candidate(probe['candidate_context'],omitted_choice,
                **decode_extra)['action_logits'][...,0,:].float()
            atom_probs=atom_logits.softmax(-1)
            atom_entropy=-(atom_probs*atom_probs.clamp_min(1e-12).log()).sum(-1).mean()
            result['omit_last_committed_event']={'expected_next_second':float(probability@seconds),
                'authored_next_second_reference_only':float(table['clock'].at(sample.target)),
                'supervised_recovery_loss':None,
                'first_atom_entropy_nats':float(atom_entropy)}
        report['samples'].append({'song_index':sample.song_index,'event_index':sample.event_index,
                                  'interventions':result,'phase_reachability':recovery})
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True);parser.add_argument('--manifest',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--cache-dir',default='cache/chart_composer_v1')
    parser.add_argument('--limit-charts',type=int,default=2);parser.add_argument('--samples',type=int,default=8)
    parser.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    args=parser.parse_args(argv)
    if args.samples<1 or args.limit_charts<1:parser.error('samples and limit-charts must be positive')
    result=diagnose(args);print(json.dumps({'output':args.output,'samples':len(result['samples'])}),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
