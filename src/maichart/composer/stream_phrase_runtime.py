"""Prefix-owned soft execution choices, never shared across speculative rollouts."""
import math
import hashlib
import json
from bisect import bisect_left
from collections import Counter

import torch


def _prefix_signature(context, count=None):
    """Content identity; production contexts maintain it incrementally."""
    if hasattr(context, 'prefix_signature'):
        return context.prefix_signature(count)
    prefix=context._committed
    rows=prefix if count is None else prefix[:count]
    # Lightweight fixtures and external callers may not own an indexed log.
    return hashlib.sha256(repr(tuple((s.kind,str(s.start),str(s.end),s.text,
        getattr(s,'phrase_mode',None),getattr(s,'phrase_mode_start',None)) for s in rows)).encode()).hexdigest()


def _control_signature(context):
    controls=getattr(context,'condition',None)
    version=(id(controls),getattr(controls,'_version',None))
    cached=getattr(context,'_execution_controls_signature',None)
    if cached is not None and cached[0]==version:return cached[1]
    if isinstance(controls,torch.Tensor):value=controls.detach().cpu().tolist()
    elif hasattr(controls,'to_dict'):value=controls.to_dict()
    else:value=controls
    signature=hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()
    context._execution_controls_signature=(version,signature)
    return signature


def draft_context_signature(context, phrase_start, mode=None, *, policy='beat_ordered_draft_v1'):
    """Stable entry identity for a plan; appending within a phrase is O(log N).

    A replayed/edited prefix gets a different epoch or content digest. Actual
    control edits also invalidate the plan without reading CUDA tensors on
    every event. This is an identity only, never a rejection of the whole plan.
    """
    prefix=context._committed
    count=bisect_left(prefix,float(phrase_start),key=lambda step:float(step.end))
    return (str(policy),float(phrase_start),int(getattr(context,'_prefix_epoch',0)),
            _prefix_signature(context,count),_control_signature(context),mode)


def attach_draft_runtime(context, batch, model, *, candidate=False):
    """Attach caller-owned inference cache identities without copying tensors.

    The draft module may cache static plans, never per-query alignment maps.
    Prefix replay, explicit edits and changed controls invalidate this owner.
    Ordinary NOTE/WAIT commits keep the stable phrase-entry identity.
    """
    if getattr(model.config,'draft_policy','off')!='beat_ordered_draft_v1' or model.training:
        return batch
    owner=(id(model),int(getattr(context,'_prefix_epoch',0)),_control_signature(context))
    if getattr(context,'_draft_runtime_owner',None)!=owner:
        context._draft_runtime_owner=owner
        context._draft_runtime_cache={}
    prefix='candidate_' if candidate else ''
    starts=batch.get(prefix+'phrase_start_beat',batch.get('phrase_start_beat'))
    ends=batch.get(prefix+'phrase_end_beat',batch.get('phrase_end_beat'))
    if starts is None:return batch
    # Generation construction supplies CPU phrase descriptors. This helper
    # should be called before device transfer, not once per CUDA action slot.
    start=float(starts[0,-1])
    end=None if ends is None else float(ends[0,-1])
    attached=dict(batch)
    attached['_draft_runtime_cache']=context._draft_runtime_cache
    attached['_draft_runtime_key']=(owner,draft_context_signature(context,start),end)
    if not candidate and starts.ndim==2 and starts.size(0)==1:
        # A generation/recovery call consumes only its newest decision. Keep
        # the full symbolic history for the decoder, but do not re-plan every
        # historical query or re-evaluate all of its old arrival candidates.
        last=starts.size(1)-1
        attached['_inference_last_query']=True
        attached['_timing_query_positions']=torch.tensor([[last]],dtype=torch.long)
        attached['_timing_original_flat']=torch.tensor([last],dtype=torch.long)
        attached['_timing_packed_flat']=torch.tensor([0],dtype=torch.long)
        attached['draft_query_rows']=torch.tensor([0],dtype=torch.long)
        attached['draft_query_positions']=torch.tensor([last],dtype=torch.long)
        attached['draft_query_inverse']=torch.zeros_like(starts,dtype=torch.long)
        attached['draft_query_keys']=[(0,start,end)]
    return attached


@torch.no_grad()
def freeze_recovery_song_style(model, context, batch, audio_cache, generator):
    """Bind one recovery episode's z and executed controls before its first read.

    Audio tensors may be shared by several episodes. Their completed controls,
    conditioned plan and chosen song component must remain context-owned. The
    CPU record can be persisted with an episode: after a weight update its old
    choice and controls remain the identity of the history it actually wrote.
    """
    config=model.config
    if (getattr(config,'draft_policy','off')!='beat_ordered_draft_v1' or
            getattr(config,'song_response_policy','off')!='unified_profile_response_v2'):
        return batch,audio_cache
    # The small profile network owns this evidence; inspecting its parameter
    # versions is CPU-only and also catches an update to its last-layer bias.
    revision=tuple(parameter._version for parameter in model.song_profile.parameters())
    owner=(id(model),id(audio_cache['global']),revision)
    state=getattr(context,'_recovery_style_cache',None)
    if state is None or state['owner']!=owner:
        device=audio_cache['global'].device
        record=getattr(context,'_recovery_frozen_style',None)
        source=context.condition.detach().to(device=device,dtype=torch.float32).reshape(1,-1)
        if record is None:
            known=torch.zeros_like(source,dtype=torch.bool);known[:,0]=True
        else:
            known=torch.as_tensor(record['known_mask'],device=device,dtype=torch.bool).reshape_as(source)
            # Rebuild only the current prior. Never re-complete controls for a
            # continuation whose already-generated prefix belongs to old z.
            if 'causal_condition' in record:
                source=torch.as_tensor(record['causal_condition'],device=device,dtype=source.dtype).reshape_as(source)
        evidence=model.song_profile.unified_evidence(audio_cache,source,known_mask=known,
            only_level_probability=0.)
        if record is None:
            choice=getattr(context,'song_response_choice',None)
            if choice is None:
                choice=int(torch.multinomial(evidence['prior_logits'].float().softmax(-1),1,generator=generator)[0,0])
            selected=torch.tensor([int(choice)],device=device,dtype=torch.long)
            completed=model.song_profile.execution_controls(evidence,source,selected)
            record=dict(choice=int(choice),execution_controls=completed[0].detach().cpu().clone(),
                known_mask=evidence['known_mask'][0].detach().cpu().clone(),
                causal_condition=evidence['causal_condition'][0].detach().cpu().clone())
            context._recovery_frozen_style=record
        else:
            choice=int(record['choice'])
            completed=torch.as_tensor(record['execution_controls'],device=device,dtype=source.dtype).reshape_as(source)
        context.song_response_choice=int(record['choice'])
        context.condition=torch.as_tensor(record['execution_controls'],dtype=torch.float32).detach().cpu().clone()
        frozen=dict(prior_logits=evidence['prior_logits'].detach(),choice=context.song_response_choice,
            execution_controls=completed.detach(),known_mask=evidence['known_mask'].detach(),
            causal_condition=evidence['causal_condition'].detach())
        local_cache={**audio_cache,'unified_style':frozen}
        # A song-shared cache may contain a plan conditioned on reference
        # controls. Rebuild it once using this episode's actual completion.
        local_cache.pop('conditioned_plan',None)
        if getattr(config,'plan_kind_policy','off')!='off' and hasattr(model,'prepare_plan_cache'):
            inverse=torch.as_tensor(batch['song_inverse'],device=device,dtype=torch.long)
            end=torch.as_tensor(batch['audio_end_sec'],device=device)
            local_cache=model.prepare_plan_cache(local_cache,completed,inverse,end)
        state=dict(owner=owner,cache=local_cache)
        context._recovery_style_cache=state
    conditioned={**batch,'condition':context.condition.unsqueeze(0).to(batch['condition'])}
    return conditioned,state['cache']


def ensure_song_response(model,context,output,generator):
    """Freeze an audio/control prior choice once, including recovery entry points."""
    choice=getattr(context,'song_response_choice',None)
    if choice is None and 'song_response_logits' in output:
        logits=output['song_response_logits'][0].float()
        if not bool(torch.isfinite(logits).any()):raise ValueError('No finite whole-song response mode')
        choice=int(torch.multinomial(logits.softmax(-1),1,generator=generator))
        context.song_response_choice=choice
    return choice


def _phrase_start_value(batch):
    """Runtime clocks use CPU packing metadata when it is available."""
    if '_runtime_phrase_start' in batch:return float(batch['_runtime_phrase_start'])
    for row,start,_ in reversed(batch.get('draft_query_keys',())):
        if row==0:return float(start)
    return float(batch['phrase_start_beat'][0,-1])


def _landing_phrase_batch(context,batch,beat):
    """View the selected landing phrase without altering departure timing.

    Normal generation already supplies candidate_* fields and their CPU keys;
    only external callers lacking these fields need another CPU context read.
    No future chart event is read to select the phrase.
    """
    fields=batch
    if 'candidate_phrase_start_beat' not in fields and beat is not None and hasattr(context,'candidate_batch'):
        fields={**batch,**context.candidate_batch(beat)}
    candidate={key[len('candidate_'):]:value for key,value in fields.items() if key.startswith('candidate_')}
    if 'phrase_start_beat' not in candidate:return batch
    view={**batch,**candidate}
    view['_runtime_phrase_start']=_phrase_start_value(candidate)
    return view


def ensure_phrase_mode(model, context, phrase_output, batch, generator, temperature=1.):
    if getattr(getattr(model,'config',None),'phrase_execution_policy','off') not in ('persistent_modes_v1','component_modes_v2'):return None
    temperature=float(temperature)
    if not math.isfinite(temperature) or temperature<0:raise ValueError('invalid phrase mode temperature')
    # Timing uses the departure phrase. Ordered-draft marks may explicitly
    # pass a landing view after a candidate onset has been selected.
    start=(_phrase_start_value(batch) if getattr(model.config,'draft_policy','off')=='beat_ordered_draft_v1'
           else float(batch['phrase_start_beat'][0,-1]))
    prefix=context._committed
    signature=(_prefix_signature(context),_control_signature(context),
               int(getattr(context,'_prefix_epoch',0)))
    state=getattr(context,'_execution_runtime',None)
    if state is None or state['signature']!=signature:
        state=dict(signature=signature,pending={},at_beat={})
        context._execution_runtime=state
    if start not in state['pending']:
        mode=next((s.phrase_mode for s in reversed(prefix)
                   if getattr(s,'phrase_mode_start',None)==start and getattr(s,'phrase_mode',None) is not None),None)
        if mode is None:
            song_choice=ensure_song_response(model,context,phrase_output,generator)
            scores=(model.execution_prior(phrase_output,batch,song_choice=song_choice) if song_choice is not None else
                    model.execution_prior(phrase_output,batch))[0,-1]
            if not bool(torch.isfinite(scores).any()):raise ValueError('No finite phrase execution mode')
            mode=(int(scores.argmax()) if temperature==0 else
                  int(torch.multinomial((scores.float()/temperature).softmax(-1),1,generator=generator)))
        if not 0<=int(mode)<int(model.config.phrase_modes):raise ValueError('Invalid committed phrase mode')
        state['pending'][start]=int(mode)
    state['draft_context_signature']=draft_context_signature(context,start,state['pending'][start],
        policy=getattr(model.config,'draft_policy','off'))
    return state['pending'][start]


def execution_options(model, context, phrase_output, batch, action, generator, *, temperature=1., beat=None):
    selected_batch=(_landing_phrase_batch(context,batch,beat)
        if getattr(model.config,'draft_policy','off')=='beat_ordered_draft_v1' else batch)
    mode=ensure_phrase_mode(model,context,phrase_output,selected_batch,generator,temperature)
    if mode is None:return {}
    start=(_phrase_start_value(selected_batch) if getattr(model.config,'draft_policy','off')=='beat_ordered_draft_v1'
           else float(batch['phrase_start_beat'][0,-1]))
    state=context._execution_runtime
    if beat is None:beat=float(batch.get('candidate_step_beat',batch.get('target_beat',batch['step_beat']))[0,-1])
    state['at_beat'][float(beat)]=(mode,start)
    return model.execution_proposal(action,phrase_output,selected_batch,mode)


def execution_metadata(context, beat):
    state=getattr(context,'_execution_runtime',None)
    if state is None:return {}
    value=state['at_beat'].get(float(beat))
    return {} if value is None else dict(phrase_mode=value[0],phrase_mode_start=value[1])


def execution_report(steps):
    choices={}
    notes=Counter()
    for step in steps:
        mode=getattr(step,'phrase_mode',None)
        start=getattr(step,'phrase_mode_start',None)
        if mode is not None and start is not None:
            choices.setdefault(start,int(mode))
            if getattr(step,'kind',2)==2:notes[int(mode)]+=1
    return dict(phrase_choices=len(choices),mode_phrase_counts=dict(Counter(choices.values())),
                mode_note_counts=dict(notes),choices=[dict(start_beat=start,mode=mode) for start,mode in sorted(choices.items())])
