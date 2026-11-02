"""Bounded continuation probes from actual sampled, guarded stream prefixes."""
from __future__ import annotations

import copy
from collections import Counter
from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace
import random
import math
from bisect import bisect_left

import numpy as np
import torch

from maichart.event_timing import EventPoint
from .events import event_metadata
from .guard import EventGuard
from .stream_data import (BOS, EOS, NOTE, PAD, WAIT, StreamInferenceContext, StreamStep,
                          beat_at_second, collate_stream, recovery_sample, second_at_beat)
from .stream_sampling import (kind_logits_for_audio, note_delta_logits_for_audio,
                              sample_logits, sample_guarded_event)
from .stream_proposal import sample_timed_event
from .stream_timing import current_timing_deltas
from .stream_target_correspondence import corresponding_shift, rotated_text, rotation_boundary_clear

SCHEMA = 'event_set_stream_recovery_v1'
MAX_EPISODE_STEPS = 128
LEGACY_EPISODE_STEPS = 64


def _episode_limit(policy):
    return MAX_EPISODE_STEPS if policy in ('greedy_mixed_v3','greedy_mixed_v4') else LEGACY_EPISODE_STEPS


def recovery_state(model):
    return {'schema': SCHEMA, 'episodes': copy.deepcopy(getattr(model, '_stream_recovery_episodes', [])),
            'anchor_cursor':int(getattr(model,'_stream_recovery_anchor_cursor',0)),
            'policy':getattr(model,'_stream_recovery_policy',getattr(getattr(model,'config',None),'recovery_policy','persistent_v1')),
            'target_policy':getattr(getattr(model,'config',None),'recovery_target_policy','authored_next_v1'),
            'phrase_execution_policy':getattr(getattr(model,'config',None),'phrase_execution_policy','off'),
            'mix_cursor':int(getattr(model,'_stream_recovery_mix_cursor',0)),
            'short_cursor':int(getattr(model,'_stream_recovery_short_cursor',0))}


def restore_recovery_state(model, state):
    if state is None:
        model._stream_recovery_episodes = []
        model._stream_recovery_anchor_cursor=0
        model._stream_recovery_mix_cursor=0
        model._stream_recovery_short_cursor=0
        return
    if not isinstance(state,dict) or state.get('schema') != SCHEMA or not isinstance(state.get('episodes'), list) or len(state['episodes']) > 16:
        raise ValueError('Invalid stream recovery state')
    policy=state.get('policy','persistent_v1')
    target_policy=state.get('target_policy','authored_next_v1')
    if target_policy not in ('authored_next_v1','causal_correspondence_v1'):
        raise ValueError('Invalid recovery target policy state')
    if policy not in ('persistent_v1','mixed_exit_v2','greedy_mixed_v3','greedy_mixed_v4'):raise ValueError('Invalid recovery policy state')
    for episode in state['episodes']:
        if episode is None:
            continue
        if (not isinstance(episode, dict) or not isinstance(episode.get('song'), int)
                or not isinstance(episode.get('anchor'), int) or episode['song'] < 0 or episode['anchor'] < 0
                or not isinstance(episode.get('generated'), list) or len(episode['generated']) > _episode_limit(policy)):
            raise ValueError('Invalid stream recovery episode')
        if 'decode_temperature' in episode and episode['decode_temperature'] not in (0.,.65):
            raise ValueError('Invalid recovery episode decoding mode')
        for row in episode['generated']:
            if not isinstance(row, dict) or row.get('kind') not in (NOTE,WAIT):
                raise ValueError('Invalid recovery generated step')
            Fraction(row['start']); Fraction(row['end'])
            if row['kind'] == NOTE and not isinstance(row.get('text'), str):
                raise ValueError('Invalid recovery generated event')
            mode,start=row.get('phrase_mode'),row.get('phrase_mode_start')
            if (mode is None)!=(start is None):raise ValueError('Incomplete recovery phrase execution metadata')
            if mode is not None and (not isinstance(mode,int) or isinstance(mode,bool) or
                not 0<=mode<int(getattr(getattr(model,'config',None),'phrase_modes',4)) or
                not isinstance(start,(float,int)) or not math.isfinite(start) or start>float(Fraction(row['end']))+1e-8):
                raise ValueError('Invalid recovery phrase execution metadata')
    cursor=state.get('anchor_cursor',0)
    if not isinstance(cursor,int) or cursor<0:raise ValueError('Invalid recovery anchor cursor')
    requested=getattr(getattr(model,'config',None),'recovery_policy','persistent_v1')
    requested_target=getattr(getattr(model,'config',None),'recovery_target_policy','authored_next_v1')
    if requested!=policy or requested_target!=target_policy or state.get('phrase_execution_policy','off')!=getattr(getattr(model,'config',None),'phrase_execution_policy','off'):
        restore_recovery_state(model,None)
        model._stream_recovery_policy=requested
        return
    for name in ('mix_cursor','short_cursor'):
        value=state.get(name,0)
        if not isinstance(value,int) or value<0:raise ValueError('Invalid recovery cursor')
        setattr(model,'_stream_recovery_'+name,value)
    model._stream_recovery_policy=policy
    model._stream_recovery_episodes = copy.deepcopy(state['episodes'])
    model._stream_recovery_anchor_cursor=cursor


def _anchors(dataset):
    cached = getattr(dataset, '_stream_recovery_anchors', None)
    if cached is not None:
        return cached
    wait, repeat, ordinary = [], [], []
    for song, stream in enumerate(dataset.streams):
        last = None
        for index, step in enumerate(stream):
            if step.kind == WAIT and step.weight > 0:
                wait.append((song,index))
            if step.kind == NOTE:
                heads = set(event_metadata(step.text).heads)
                if (last is not None and step.end_second-last[0] <= .2
                        and (len(heads)>1 or len(last[1])>1) and heads & last[1] and step.weight > 0):
                    repeat.append((song,index))
                last = (step.end_second,heads)
    ordinary = [(song,start) for song,start,_,_ in dataset.samples]
    cached = (wait,repeat,ordinary)
    dataset._stream_recovery_anchors = cached
    return cached


def _new_episode(dataset, rng, slot, *, rotation=None):
    if getattr(dataset,'continuation_policy','off')=='pattern_runs_v2':
        categories=_risk_exits(dataset)
        available=[(name,rows) for name,rows in categories.items() if rows]
        if available:
            name,choices=available[(rotation if rotation is not None else slot)%len(available)]
            song,index=choices[rng.randrange(len(choices))]
            return dict(song=song,anchor=max(0,index-2),exit_index=index,generated=[],risk=name+'_exit')
    wait, repeat, ordinary = _anchors(dataset)
    if getattr(dataset,'continuation_policy','off')=='typed_runs_v1':
        exits=getattr(dataset,'_stream_exit_anchors',None)
        if exits is None:
            from .stream_continuation import typed_event,ContinuationState
            touch,rhythm,button=[],[],[]
            for song,stream in enumerate(dataset.streams):
                state=ContinuationState()
                for index,target in enumerate(stream):
                    eligible,exit_label=state.exit_label(target)
                    if eligible and exit_label:
                        anchor=(song,max(0,index-2));_,buttons,is_touch=typed_event(target.text)
                        if state.touch_count>=3 and not is_touch:touch.append(anchor)
                        elif {key for key,(count,_) in state.buttons.items() if count>=2}-buttons:button.append(anchor)
                        else:rhythm.append(anchor)
                    state.append(target)
            exits=dataset._stream_exit_anchors=(touch,rhythm,button)
        categories=[('touch_exit',exits[0]),('rhythm_exit',exits[1]),('button_exit',exits[2] if len(exits)>2 else []),
                    ('overlap',repeat),('rest',wait),('ordinary',ordinary)]
        available=[item for item in categories if item[1]]
        if not available:return None
        name,choices=available[rng.randrange(len(available)) if rotation is None else rotation%len(available)]
        song,anchor=choices[rng.randrange(len(choices))]
        return {'song':song,'anchor':anchor,'generated':[],'risk':name}
    # Alternate risk categories, falling back only when a category is absent.
    preferred = (wait,repeat) if slot % 2 == 0 else (repeat,wait)
    choices = next((group for group in (*preferred,ordinary) if group), None)
    if not choices: return None
    song, anchor = choices[rng.randrange(len(choices))]
    return {'song': song, 'anchor': anchor, 'generated': []}


def _prefix(dataset, episode):
    song, anchor = episode['song'], episode['anchor']
    if song >= len(dataset.streams) or anchor >= len(dataset.streams[song]):
        raise ValueError('Recovery state does not match dataset')
    clock = dataset.clocks[song]
    prefix = list(dataset.streams[song][:anchor])
    for record in episode['generated']:
        start,end = Fraction(record['start']), Fraction(record['end'])
        prefix.append(StreamStep(record['kind'], start, end, max(0.,second_at_beat(clock,start)),
                                 max(0.,second_at_beat(clock,end)), record.get('text'),
                                 phrase_mode=record.get('phrase_mode'),phrase_mode_start=record.get('phrase_mode_start')))
    return prefix


def _guard(song, prefix, anchor, *, physical_policy='legacy_report_v1'):
    guard = EventGuard(song.tempo,song.first,float(song.audio.duration_sec),
                       touch_layout_policy='open',intercept_tap_on_slide=True,physical_policy=physical_policy)
    for index, step in enumerate(prefix):
        if step.kind == NOTE:
            point = EventPoint(step.end,step.text)
            if index >= anchor and not guard.accepts(point):
                raise ValueError('Restored generated event violates EventGuard')
            guard.commit(point)
    return guard


def _dataset_guard(dataset,song_index,prefix,anchor):
    policy=getattr(dataset,'physical_policy','legacy_report_v1')
    options={'physical_policy':policy} if policy!='legacy_report_v1' else {}
    return _guard(dataset.base.songs[song_index][0],prefix,anchor,**options)


def _next_reference(dataset, song, prefix, guard, *, target_policy='authored_next_v1'):
    """Keep only an actually reachable, known authored continuation."""
    cursor = prefix[-1].end if prefix else Fraction(0)
    current = max(0.,second_at_beat(dataset.clocks[song],cursor))
    source = dataset.streams[song]
    reference = next((s for s in source if s.kind == NOTE and
                      (s.end > cursor or (not prefix and s.end == cursor))), None)
    endpoint = reference.end if reference is not None else source[-1].end
    if endpoint < cursor: return None, 'past_end'
    if endpoint-cursor > 2:
        end = cursor+2
        target = StreamStep(WAIT,cursor,end,current,second_at_beat(dataset.clocks[song],end))
    elif reference is not None:
        target = StreamStep(NOTE,cursor,endpoint,current,reference.end_second,reference.text,reference.weight)
        if target.end == cursor and prefix: return None, 'zero_note_delta'
        if target_policy == 'authored_next_v1' and not guard.accepts(EventPoint(target.end,target.text)):
            return None, 'reference_illegal'
    else:
        target = StreamStep(EOS,cursor,endpoint,current,source[-1].end_second,weight=source[-1].weight)
    views = dataset.views[song]
    upper = min(target.end_second, np.nextafter(views.end_sec,-np.inf))
    known = float(views.window_weight(current,upper))
    if known <= 0 or target.weight <= 0:
        return None, 'unknown_reference'
    if target.kind != EOS and getattr(dataset,'timing_output_policy','relative_delta_v1')!='absolute_grid_v1':
        try: dataset.delta_vocab.encode(target.end-target.start)
        except ValueError: return None, 'delta_oov'
    return replace(target, weight=target.weight*known), 'ok'


def _qualified_continuation(dataset,song,prefix,target,guard,anchor,reference_index,policy):
    """Keep authored timing; use exact configuration only after proved history."""
    if policy == 'authored_next_v1':
        return target,True,'authored_next_v1'
    if target.kind != NOTE or reference_index is None:
        return target,False,'time_only'
    source=dataset.streams[song]
    if not (0 <= anchor <= reference_index < len(source)):
        return target,False,'time_only'
    shift=corresponding_shift(source[anchor:reference_index],prefix[anchor:])
    if shift is None:
        return target,False,'time_only'
    if shift == 0:
        return (target,True,'exact_history') if guard.accepts(EventPoint(target.end,target.text)) else (
            target,False,'time_only')
    if not rotation_boundary_clear(source[:anchor],prefix[anchor].start_second,guard):
        return target,False,'time_only'
    try:
        transformed=rotated_text(target.text,shift)
        dataset.codec.encode(transformed)
        if not guard.accepts(EventPoint(target.end,transformed)):
            return target,False,'time_only'
    except (ValueError,KeyError):
        return target,False,'time_only'
    return replace(target,text=transformed),True,'rotation_correspondence'


@torch.no_grad()
def _propose(dataset, model, song_index, prefix, guard, generator, device, *,
             context=None, audio_cache=None, temperature=.8):
    song = dataset.base.songs[song_index][0]
    cursor = prefix[-1].end if prefix else Fraction(0)
    audio_end = beat_at_second(dataset.clocks[song_index],song.audio.duration_sec)
    if cursor >= audio_end: return None, 'audio_end'
    if context is None:
        context = StreamInferenceContext(song,dataset.codec,dataset.delta_vocab,dataset.conditions[song_index],
                                         lead_in=dataset.lead_in,audio_bank=dataset._bank(song_index),
                                         audio_grid_seconds=dataset.audio_grid_seconds,
                                         history_window_policy=dataset.history_window_policy,rollout_steps=dataset.targets,
                                         continuation_policy=dataset.continuation_policy,
                                         timing_output_policy=getattr(dataset,'timing_output_policy','relative_delta_v1'),
                                         phrase_context_policy=getattr(dataset,'phrase_context_policy','off'),
        candidate_state_policy=getattr(dataset,'candidate_state_policy','cursor_v1'),
        training_window_policy=getattr(dataset,'training_window_policy','block_v1'),
        max_phrase_events=getattr(dataset,'max_phrase_events',64))
        raw = context.batch(prefix,cursor)
    else:
        raw = context.current_batch()
    cached_audio_keys = {'song_audio','song_mask','song_seconds','song_beats'} if audio_cache is not None else set()
    batch = {k:v.to(device) for k,v in raw.items() if torch.is_tensor(v) and k not in cached_audio_keys}
    cache = audio_cache if audio_cache is not None else model.encode_audio(
        batch['song_audio'],batch['song_mask'],batch['song_seconds'],batch['song_beats'])
    result = model(batch,audio_cache=cache)
    kind_scores = kind_logits_for_audio(result['kind_logits'][0,-1],cursor,audio_end)
    kind = sample_logits(kind_scores,temperature,generator)
    if kind is None: return None, 'no_legal_kind'
    if kind == EOS: return None, 'sampled_eos'
    def wait(reason):
        if audio_end-cursor <= 2: return None, reason+'_tail'
        end = cursor+2
        return StreamStep(WAIT,cursor,end,max(0.,second_at_beat(context.clock,cursor)),
                          second_at_beat(context.clock,end)), reason
    if kind == WAIT: return wait('ok')
    absolute=getattr(getattr(model,'config',None),'timing_output_policy','relative_delta_v1')=='absolute_grid_v1'
    deltas=(current_timing_deltas(model,result,cursor) if absolute
            else (Fraction(0),*dataset.delta_vocab.values))
    scores = note_delta_logits_for_audio(result['delta_logits'][0,-1],deltas,cursor,audio_end,
        context.clock,float(song.audio.duration_sec),first_step=not prefix)
    if not bool(torch.isfinite(scores).any()): return wait('no_legal_time')
    def action_at(delta_id,beat):
        from .stream_proposal import prepare_selected_candidate
        selected_batch,selected_result=prepare_selected_candidate(model,context,batch,result,beat,cache)
        ids=torch.full_like(selected_batch['history_delta'],delta_id)
        times=torch.full_like(selected_batch['step_second'],second_at_beat(context.clock,beat))
        shared=getattr(getattr(model,'config',None),'event_family_policy','legacy_v1')=='shared_continuation_v1'
        if shared:
            action,load,selected_result=model.candidate_action_state(selected_result,selected_batch,cache,times,
                return_button_load=True)
            proposal=dict(context=action[0,-1],button_load=load[0,-1],
                family_logits=model.family_logits_for_action(action)[0,-1])
        elif hasattr(model,'lane_load_projection'):
            action,load=model.action_context_for_delta(selected_result,selected_batch,ids,cache,event_seconds=times,return_button_load=True)
            proposal=dict(context=action[0,-1],button_load=load[0,-1])
        else:
            proposal=dict(context=model.action_context_for_delta(selected_result,selected_batch,ids,cache,event_seconds=times)[0,-1])
        if hasattr(model,'duration_context_at'):
            duration_context=model.duration_context_at(selected_result,selected_batch,times,cache)
            if duration_context is not None:
                from .stream_duration import select_duration_context
                proposal['duration_context']=select_duration_context(duration_context,
                    torch.tensor([selected_batch['step_second'].numel()-1],device=device))
        if getattr(getattr(model,'config',None),'event_family_policy','legacy_v1') in ('audio_semantic_v2','audio_prior_transition_v3','audio_state_phrase_v4'):
            families=model.family_context_at(cache,times,selected_batch['song_inverse'],selected_batch,**({'phrase_output':selected_result} if model.config.event_family_policy=='audio_state_phrase_v4' else {}))
            proposal['family_context']=families['full'][0,-1]
            if 'full_logits' in families:proposal['family_logits']=families['full_logits'][0,-1]
        if getattr(getattr(model,'config',None),'transition_policy','off') in ('music_exit_v1','music_exit_v2','music_exit_v3'):
            transition=model.transition_logits_at(cache,times,selected_batch['song_inverse'],selected_batch)[0,-1]
            proposal['motion_keep_logit']=transition[1]
            proposal['motion_reference_shape']=selected_result['motion_reference_shape'][0,-1]
            proposal['template_keep_logit']=transition[2]
            proposal['template_reference_shape']=selected_batch['history_shape'][0,-1].masked_fill(~selected_batch['history_kind'][0,-1].eq(NOTE),-1)
        from .stream_phrase_runtime import execution_options
        proposal.update(execution_options(model,context,selected_result,selected_batch,proposal['context'],
            generator,beat=beat))
        return proposal
    point,delta_id,retries,failed_times = sample_timed_event(model,scores,deltas,cursor,action_at,
        int(result['reference_phase'][0,-1]),
        result['reference_phase_mask'][0,-1] if 'reference_phase_mask' in result else None,
        guard,generator,pointer_temperature=temperature,action_temperature=temperature,
        repeat_delta_id=int(result['repeat_delta_id'][0,-1]) if 'repeat_delta_id' in result else -1,
        timing_grid_ids=result.get('timing_grid_ids'),
        timing_group_count=result.get('timing_group_count'),
        timing_keep_mask=result.get('timing_keep_mask'),
        motion_phase_distribution=(result['motion_phase_distribution'][0,-1]
                                   if 'motion_phase_distribution' in result else None))
    if point is None: return wait('rejected_configurations')
    end = cursor+deltas[delta_id]
    a,b = max(0.,second_at_beat(context.clock,cursor)), second_at_beat(context.clock,end)
    from .stream_phrase_runtime import execution_metadata
    return StreamStep(NOTE,cursor,end,a,b,point.note_text,**execution_metadata(context,end)), ('guard_retry' if retries else 'ok')


@torch.no_grad()
def _encode_song(model, bank, device):
    audio, seconds, beats = (value.unsqueeze(0).to(device) for value in bank)
    mask = torch.ones(1,len(bank[0]),dtype=torch.bool,device=device)
    return model.encode_audio(audio,mask,seconds,beats)


def _persistent_recovery_batch(dataset, model, device, step, *, samples=2, steps=2,_rows_out=None,_audio_caches=None):
    """Sample bounded actual continuations, then supervise legal reference nexts.

    The caller transfers the resulting CPU batch and applies the model loss
    once with the configured recovery weight. Model mode is always restored.
    """
    if not 0 <= samples <= 16 or not 0 <= steps <= 64:
        raise ValueError('Invalid recovery budget')
    rng = random.Random(int(step))
    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(int(step))
    episodes = copy.deepcopy(getattr(model,'_stream_recovery_episodes',[]))[:samples]
    anchor_cursor=int(getattr(model,'_stream_recovery_anchor_cursor',0))
    policy=getattr(getattr(model,'config',None),'recovery_policy','persistent_v1')
    limit=_episode_limit(policy)
    episodes += [None] * (samples-len(episodes))
    modes = [(module,module.training) for module in model.modules()]
    model.eval()
    diagnostics, rows = Counter(), []
    audio_caches={} if _audio_caches is None else _audio_caches
    try:
        for slot in range(samples):
            episode = episodes[slot]
            if episode is None or len(episode['generated']) >= limit:
                episode = _new_episode(dataset,rng,slot,rotation=anchor_cursor)
                if episode is not None and policy in ('greedy_mixed_v3','greedy_mixed_v4'):
                    episode['decode_temperature']=0. if anchor_cursor%2==0 else .65
                anchor_cursor+=1
                diagnostics['new_episode']+=int(episode is not None)
            if episode is None:
                diagnostics['no_anchor'] += 1
                episodes[slot] = None
                continue
            diagnostics['anchor_'+episode.get('risk','legacy')]+=1
            try:
                prefix = _prefix(dataset,episode)
                guard = _dataset_guard(dataset,episode['song'],prefix,episode['anchor'])
                song = dataset.base.songs[episode['song']][0]
                bank = dataset._bank(episode['song'])
                context = StreamInferenceContext(song,dataset.codec,dataset.delta_vocab,
                    dataset.conditions[episode['song']],lead_in=dataset.lead_in,audio_bank=bank,
                    audio_grid_seconds=dataset.audio_grid_seconds,
                    history_window_policy=dataset.history_window_policy,rollout_steps=dataset.targets,
                    continuation_policy=dataset.continuation_policy,
                    timing_output_policy=getattr(dataset,'timing_output_policy','relative_delta_v1'),
                    phrase_context_policy=getattr(dataset,'phrase_context_policy','off'),
        candidate_state_policy=getattr(dataset,'candidate_state_policy','cursor_v1'),
        training_window_policy=getattr(dataset,'training_window_policy','block_v1'),
        max_phrase_events=getattr(dataset,'max_phrase_events',64))
                context.batch(prefix,prefix[-1].end if prefix else Fraction(0))
                if episode['song'] not in audio_caches:
                    audio_caches[episode['song']] = _encode_song(model,bank,device)
                    if getattr(getattr(model,'config',None),'plan_kind_policy','off') != 'off':
                        with torch.no_grad():
                            audio_caches[episode['song']] = model.prepare_plan_cache(
                                audio_caches[episode['song']],dataset.conditions[episode['song']].to(device).unsqueeze(0),
                                torch.zeros(1,dtype=torch.long,device=device),
                                torch.tensor([float(song.audio.duration_sec)],dtype=torch.float64,device=device))
                accepted = 0
                for _ in range(min(steps,limit-len(episode['generated']))):
                    options=({'temperature':episode.get('decode_temperature',.65)} if policy in ('greedy_mixed_v3','greedy_mixed_v4') else {})
                    proposal, reason = _propose(dataset,model,episode['song'],prefix,guard,generator,device,
                        context=context,audio_cache=audio_caches[episode['song']],**options)
                    if proposal is None:
                        diagnostics[reason] += 1
                        break
                    if reason != 'ok': diagnostics[reason] += 1
                    context.commit(proposal)
                    prefix.append(proposal); accepted += 1
                    if policy in ('greedy_mixed_v3','greedy_mixed_v4'):diagnostics['greedy_generated_steps' if options['temperature']==0 else 'sampled_generated_steps']+=1
                    episode['generated'].append(dict(kind=proposal.kind,start=str(proposal.start),
                        end=str(proposal.end),text=proposal.text,
                        phrase_mode=proposal.phrase_mode,phrase_mode_start=proposal.phrase_mode_start))
                if accepted == 0:
                    episodes[slot] = None
                    continue
                target, reason = _next_reference(dataset,episode['song'],prefix,guard,
                    target_policy=getattr(getattr(model,'config',None),'recovery_target_policy','authored_next_v1'))
                if 'exit_index' in episode and prefix:
                    exit_beat=dataset.streams[episode['song']][episode['exit_index']].end
                    diagnostics['overshot']+=int(prefix[-1].end>=exit_beat)
                if target is None:
                    diagnostics[reason] += 1
                else:
                    source=dataset.streams[episode['song']]
                    reference_index=next((i for i,s in enumerate(source)
                        if s.kind==NOTE and s.end==target.end),None) if target.kind==NOTE else None
                    target,trusted,correspondence=_qualified_continuation(dataset,episode['song'],prefix,
                        target,guard,episode['anchor'],reference_index,
                        getattr(getattr(model,'config',None),'recovery_target_policy','authored_next_v1'))
                    row=recovery_sample(dataset,episode['song'],prefix,target,weight=target.weight,
                        configuration_trusted=trusted,correspondence=correspondence)
                    rows.append(row)
                    diagnostics['configuration_trusted' if trusted else 'time_only']+=1
                    diagnostics['correspondence_'+correspondence]+=1
                    diagnostics['actual_exit_supervised']+=int(row['transition_exit_mask'][-1] and row['transition_exit_labels'][-1])
                    diagnostics['supervised'] += 1
                    diagnostics['target_'+str(target.kind)] += 1
                episodes[slot] = episode if len(episode['generated']) < limit else None
            except (ValueError,KeyError) as exc:
                diagnostics['invalid_episode'] += 1
                diagnostics['last_error'] = str(exc)
                episodes[slot] = None
    finally:
        for module, mode in modes:
            module.training = mode
    model._stream_recovery_episodes = episodes
    model._stream_recovery_anchor_cursor=anchor_cursor
    diagnostics['episodes'] = sum(episode is not None for episode in episodes)
    diagnostics['generated_steps'] = sum(len(e['generated']) for e in episodes if e is not None)
    if _rows_out is not None:
        _rows_out.extend(rows)
        return None,dict(diagnostics)
    return (collate_stream(rows) if rows else None), dict(diagnostics)


def _risk_exits(dataset):
    """Distinct authored exit decisions, rather than repeated episode ticks."""
    policy=getattr(dataset,'continuation_policy','off')
    key=(policy,len(dataset.streams))
    cached=getattr(dataset,'_mixed_recovery_exit_anchors',None)
    if cached is not None and cached[0]==key:return cached[1]
    from .stream_continuation import ContinuationState
    categories={name:[] for name in ('gap','type','template','exact','period2','motion','touch','button')}
    for song,stream in enumerate(dataset.streams):
        state=ContinuationState(policy=policy)
        for index,target in enumerate(stream):
            if target.weight>0 and not dataset.unknown_steps[song][index]:
                for name in state.exit_categories(target):
                    categories.setdefault(name,[]).append((song,index))
            state.append(target)
    dataset._mixed_recovery_exit_anchors=(key,categories)
    return categories


def _context(dataset,song,bank):
    return StreamInferenceContext(dataset.base.songs[song][0],dataset.codec,dataset.delta_vocab,
        dataset.conditions[song],lead_in=dataset.lead_in,audio_bank=bank,
        audio_grid_seconds=dataset.audio_grid_seconds,history_window_policy=dataset.history_window_policy,
        rollout_steps=dataset.targets,continuation_policy=dataset.continuation_policy,
        timing_output_policy=getattr(dataset,'timing_output_policy','relative_delta_v1'),
        phrase_context_policy=getattr(dataset,'phrase_context_policy','off'),
        candidate_state_policy=getattr(dataset,'candidate_state_policy','cursor_v1'),
        training_window_policy=getattr(dataset,'training_window_policy','block_v1'),
        max_phrase_events=getattr(dataset,'max_phrase_events',64))


def _prepared_cache(dataset,model,song,bank,device,caches):
    if song not in caches:
        cache=_encode_song(model,bank,device)
        if getattr(getattr(model,'config',None),'plan_kind_policy','off')!='off':
            cache=model.prepare_plan_cache(cache,dataset.conditions[song].to(device).unsqueeze(0),
                torch.zeros(1,dtype=torch.long,device=device),
                torch.tensor([float(dataset.base.songs[song][0].audio.duration_sec)],dtype=torch.float64,device=device))
        caches[song]=cache
    return caches[song]


@torch.no_grad()
def _propose_fixed(dataset,model,context,source_step,guard,generator,device,cache,*,temperature=.8):
    """Sample configuration at an authored onset; never sample or shift time."""
    raw=context.current_batch()
    batch={k:v.to(device) for k,v in raw.items() if torch.is_tensor(v) and k not in
           ('song_audio','song_mask','song_seconds','song_beats')}
    output=model(batch,audio_cache=cache)
    from .stream_proposal import prepare_selected_candidate
    batch,output=prepare_selected_candidate(model,context,batch,output,source_step.end,cache)
    times=torch.full_like(batch['step_second'],source_step.end_second)
    shared=getattr(getattr(model,'config',None),'event_family_policy','legacy_v1')=='shared_continuation_v1'
    if shared:
        action,load,output=model.candidate_action_state(output,batch,cache,times,return_button_load=True)
        load=load[0,-1]
    elif hasattr(model,'lane_load_projection'):
        action,load=model.action_context_at(output['hidden'],times,cache,batch['song_inverse'],batch,
                                            return_button_load=True)
        load=load[0,-1]
    else:
        action=model.action_context_at(output['hidden'],times,cache,batch['song_inverse'],batch)
        load=None
    family_options={}
    if shared:family_options['family_logits']=model.family_logits_for_action(action)[0,-1]
    if hasattr(model,'duration_context_at'):
        duration_context=model.duration_context_at(output,batch,times,cache)
        if duration_context is not None:
            from .stream_duration import select_duration_context
            family_options['duration_context']=select_duration_context(duration_context,
                torch.tensor([batch['step_second'].numel()-1],device=device))
    if getattr(getattr(model,'config',None),'event_family_policy','legacy_v1') in ('audio_semantic_v2','audio_prior_transition_v3','audio_state_phrase_v4'):
        families=model.family_context_at(cache,times,batch['song_inverse'],batch,**({'phrase_output':output} if model.config.event_family_policy=='audio_state_phrase_v4' else {}))
        family_options['family_context']=families['full'][0,-1]
        if 'full_logits' in families:family_options['family_logits']=families['full_logits'][0,-1]
    if getattr(getattr(model,'config',None),'transition_policy','off') in ('music_exit_v1','music_exit_v2','music_exit_v3'):
        transition=model.transition_logits_at(cache,times,batch['song_inverse'],batch)[0,-1]
        family_options['motion_keep_logit']=transition[1]
        family_options['motion_reference_shape']=output['motion_reference_shape'][0,-1]
        family_options['template_keep_logit']=transition[2]
        family_options['template_reference_shape']=batch['history_shape'][0,-1].masked_fill(~batch['history_kind'][0,-1].eq(NOTE),-1)
    from .stream_phrase_runtime import execution_options,execution_metadata
    family_options.update(execution_options(model,context,output,batch,action[0,-1],generator,beat=source_step.end))
    point,retries=sample_guarded_event(model,action[0,-1],int(output['reference_phase'][0,-1]),
        source_step.end,guard,generator,temperature=temperature,
        reference_phase_mask=output.get('reference_phase_mask',None)[0,-1]
            if 'reference_phase_mask' in output else None,
        motion_phase_distribution=output['motion_phase_distribution'][0,-1]
            if 'motion_phase_distribution' in output else None,button_load=load,**family_options)
    return (replace(source_step,text=point.note_text,**execution_metadata(context,source_step.end)) if point is not None else None),retries


def _short_exit_row(dataset,model,device,rng,generator,cursor,steps,caches,diagnostics):
    available=[(name,anchors) for name,anchors in _risk_exits(dataset).items() if anchors]
    if not available:
        diagnostics['skip_no_exit_anchor']+=1;return None
    name,anchors=available[cursor%len(available)]
    song,index=anchors[rng.randrange(len(anchors))]
    diagnostics['new_episode']+=1;diagnostics['new_short_episode']+=1
    diagnostics['new_anchor_'+name]+=1
    source=dataset.streams[song];target=source[index]
    musical=getattr(getattr(model,'config',None),'recovery_policy','persistent_v1') in ('greedy_mixed_v3','greedy_mixed_v4')
    count=min(min(5,max(4,steps)) if musical else min(2,max(1,steps)),index)
    # Only immediately preceding NOTE actions are corrupted. A WAIT remains an
    # authored boundary, so crossing one would not be a short fixed-time run.
    while count and any(s.kind!=NOTE for s in source[index-count:index]):count-=1
    if musical and count<4:
        diagnostics['skip_short_motion_horizon']+=1;return None
    if not count:
        diagnostics['skip_no_preceding_notes']+=1;return None
    anchor=index-count;prefix=list(source[:anchor])
    guard=_dataset_guard(dataset,song,prefix,anchor)
    bank=dataset._bank(song);context=_context(dataset,song,bank)
    context.batch(prefix,prefix[-1].end if prefix else Fraction(0))
    cache=_prepared_cache(dataset,model,song,bank,device,caches)
    for reference in source[anchor:index]:
        options=({'temperature':0. if cursor%2==0 else .65} if musical else {})
        proposal,retries=_propose_fixed(dataset,model,context,reference,guard,generator,device,cache,**options)
        diagnostics['guard_retry']+=retries
        if proposal is None:
            diagnostics['skip_configuration_conflict']+=1;return None
        # Do not silently retime a sample or substitute a forced WAIT.
        if proposal.start!=reference.start or proposal.end!=reference.end:
            raise ValueError('Fixed recovery proposal changed authored timing')
        prefix.append(proposal);context.commit(proposal)
        if musical:diagnostics['greedy_generated_steps' if options['temperature']==0 else 'sampled_generated_steps']+=1
    current=prefix[-1].end if prefix else Fraction(0)
    if current!=target.start or (target.kind==NOTE and current>=target.end):
        diagnostics['overshot']+=1;diagnostics['skip_unreachable_exit']+=1;return None
    if (target.kind==NOTE and getattr(getattr(model,'config',None),'recovery_target_policy','authored_next_v1')=='authored_next_v1'
            and not guard.accepts(EventPoint(target.end,target.text))):
        diagnostics['skip_reference_conflict']+=1;return None
    target,trusted,correspondence=_qualified_continuation(dataset,song,prefix,target,guard,anchor,index,
        getattr(getattr(model,'config',None),'recovery_target_policy','authored_next_v1'))
    row=recovery_sample(dataset,song,prefix,target,weight=target.weight,
        configuration_trusted=trusted,correspondence=correspondence)
    diagnostics['configuration_trusted' if trusted else 'time_only']+=1
    diagnostics['correspondence_'+correspondence]+=1
    diagnostics['supervised']+=1;diagnostics['target_'+str(target.kind)]+=1
    diagnostics['authored_exit_supervised']+=1
    diagnostics['actual_exit_supervised']+=int(row['transition_exit_mask'][-1] and row['transition_exit_labels'][-1])
    return row


def _synthetic_loop_targets(dataset):
    """Cache trusted authored exits, including real WAIT/EOS boundaries."""
    cached=getattr(dataset,'_synthetic_loop_exit_anchors',None)
    if cached is not None:return cached
    groups={NOTE:set(),WAIT:set(),EOS:set()}
    for anchors in _risk_exits(dataset).values():
        for song,index in anchors:
            if dataset.streams[song][index].kind==NOTE:groups[NOTE].add((song,index))
    for song,source in enumerate(dataset.streams):
        for index,target in enumerate(source):
            if target.kind in (WAIT,EOS) and target.weight>0 and not dataset.unknown_steps[song][index]:
                groups[target.kind].add((song,index))
    cached={kind:tuple(sorted(rows)) for kind,rows in groups.items()}
    dataset._synthetic_loop_exit_anchors=cached
    return cached


def _synthetic_long_exit_row(dataset,model,rng,cursor,diagnostics):
    """Variable-count wrong history ending exactly at a real target's start.

    Counts are independent of the authored history density: 128 one-beat
    events can replace a longer span containing many more authored events.
    Only the causal prefix is changed; music, condition and target stay real.
    """
    groups=_synthetic_loop_targets(dataset)
    available=[kind for kind in (NOTE,WAIT,EOS) if groups[kind]]
    if not available:
        diagnostics['skip_no_synthetic_exit']+=1;return None
    kind=(NOTE,WAIT,EOS)[cursor%3]
    if not groups[kind]:kind=available[cursor%len(available)]
    anchors=groups[kind];song,index=anchors[rng.randrange(len(anchors))]
    return _synthetic_long_exit_row_at(dataset,song,index,rng,cursor,diagnostics)


def _synthetic_long_exit_row_at(dataset,song,index,rng,cursor,diagnostics,*,reference=None,repeat_gap=None):
    """Construct a legal changed-count prefix for one explicit authored target.

    This CPU-only constructor is shared by sparse recovery and the main
    batch curriculum. No target's atom is read when choosing the wrong loop.
    """
    from .stream_augmentation import _loop_texts,_validate_window,_encode_gap,event_is_sustained
    source=dataset.streams[song];target=source[index]
    if target.kind not in (NOTE,WAIT,EOS) or target.weight<=0 or dataset.unknown_steps[song][index]:
        diagnostics['skip_synthetic_unknown_target']+=1;return None
    record=dataset.base.songs[song][0];clock=dataset.clocks[song]
    boundary=target.start
    if boundary<=0 or second_at_beat(clock,boundary)>=float(record.audio.duration_sec)-1e-9:
        diagnostics['skip_synthetic_boundary']+=1;return None
    if reference is not None and (reference.kind!=NOTE or event_is_sustained(reference.text)):
        raise ValueError('Aligned loop needs a simple preceding NOTE template')
    gaps=([Fraction(repeat_gap)] if repeat_gap is not None else list(map(Fraction,('1/4','1/2','1','2'))))
    if repeat_gap is None and target.kind==NOTE and cursor%2==0:
        gaps=[gap for gap in gaps if gap!=target.end-target.start]
    choices=[]
    desired=rng.choices((32,64,128),weights=(1,2,3),k=1)[0]
    for gap in gaps:
        if not 0<gap<=2:continue
        count=min(128,desired,int(boundary//gap)+1)
        if count<4:continue
        first=boundary-(count-1)*gap
        keep=bisect_left(source,first,hi=index,key=lambda step:step.end)
        previous=source[keep-1].end if keep else Fraction(0)
        head_gap=first-previous
        if not (0<head_gap<=2 or (not keep and first==0)):continue
        try:_encode_gap(dataset,gap);_encode_gap(dataset,head_gap)
        except ValueError:continue
        choices.append((gap,count,first,keep))
    if not choices:
        diagnostics['skip_synthetic_unreachable']+=1;return None
    gap,count,first,keep=choices[cursor%len(choices)]
    prefix=list(source[:keep]);previous=prefix[-1].end if prefix else Fraction(0)
    mode=('same_key','abab','rotation','same_shape','touch','same_gap','rotation_gap')[cursor%7]
    # The motif pool is strictly before the supervised target and contains
    # no sustains that would become overlapping merely by repetition.
    motifs=[step.text for step in source[max(0,index-32):index]
            if step.kind==NOTE and not event_is_sustained(step.text)]
    if reference is not None:
        encoded=dataset.codec.encode(reference.text)
        mode='reference_template'
        if encoded['shape']:
            direction=(-1,0,1)[rng.randrange(3)]
            texts=[dataset.codec.decode(encoded['shape'],(encoded['phase']+direction*(i-count+1))%8,
                       encoded['details']) for i in range(count)]
        else:texts=[reference.text]*count
    elif mode=='same_gap':
        if not motifs:raise ValueError('No simple authored pattern for a timing loop')
        texts=[motifs[i%len(motifs)] for i in range(count)]
    else:
        pool=[SimpleNamespace(kind=NOTE,text=text) for text in motifs]
        texts=_loop_texts(dataset,pool,0,count,mode,rng)
    loop=[]
    for offset,text in enumerate(texts):
        dataset.codec.encode(text)
        beat=first+offset*gap
        loop.append(StreamStep(NOTE,previous,beat,max(0.,second_at_beat(clock,previous)),
            max(0.,second_at_beat(clock,beat)),text))
        previous=beat
    if previous!=target.start:raise ValueError('Synthetic loop changed the target cursor')
    guard=_dataset_guard(dataset,song,prefix,len(prefix))
    target_policy=getattr(dataset,'recovery_target_policy','authored_next_v1')
    _validate_window(guard,loop+[target] if target_policy=='authored_next_v1' else loop)
    if target_policy!='authored_next_v1':
        # The native window check does not mutate EventGuard. Commit exactly
        # once so the transformed target is checked against the actual loop.
        for step in loop:guard.commit(EventPoint(step.end,step.text))
    prefix.extend(loop)
    target,trusted,correspondence=_qualified_continuation(dataset,song,prefix,target,guard,keep,index,
        target_policy)
    row=recovery_sample(dataset,song,prefix,target,weight=target.weight,
        configuration_trusted=trusted,correspondence=correspondence)
    diagnostics['configuration_trusted' if trusted else 'time_only']+=1
    diagnostics['correspondence_'+correspondence]+=1
    row['_synthetic_loop_events']=count
    row['_synthetic_loop_mode']=mode
    row['_synthetic_loop_gap']=str(gap)
    row['_synthetic_loop_last_text']=loop[-1].text
    row['_synthetic_loop_reference_aligned']=reference is not None
    diagnostics['synthetic_long_rows']+=1;diagnostics['synthetic_loop_events']+=count
    diagnostics['synthetic_'+mode]+=1;diagnostics['synthetic_gap_'+str(gap)]+=1
    diagnostics['supervised']+=1;diagnostics['target_'+str(target.kind)]+=1
    diagnostics['actual_exit_supervised']+=int(row['transition_exit_mask'][-1] and row['transition_exit_labels'][-1])
    return row


def recovery_batch(dataset,model,device,step,*,samples=2,steps=2):
    """Mixed fixed-time exits plus a minority of persistent free-time drift.

    A four-slot cursor gives exactly three short risk trajectories per long
    one, including across calls with samples=2. The long pool is independent
    of short anchors, and only future-reachable authored targets supervise it.
    V4 alternates those long slots between model rollouts and immediately
    constructed variable-count loops at real NOTE/WAIT/EOS boundaries.
    """
    policy=getattr(getattr(model,'config',None),'recovery_policy','persistent_v1')
    previous=getattr(model,'_stream_recovery_policy',policy)
    if previous!=policy:restore_recovery_state(model,None)
    model._stream_recovery_policy=policy
    if policy=='persistent_v1':
        return _persistent_recovery_batch(dataset,model,device,step,samples=samples,steps=steps)
    if policy not in ('mixed_exit_v2','greedy_mixed_v3','greedy_mixed_v4'):raise ValueError('Unsupported recovery policy')
    if not 0<=samples<=16 or not 0<=steps<=64:raise ValueError('Invalid recovery budget')
    if not steps or not samples:return None,{'new_episode':0,'actual_exit_supervised':0}
    rng=random.Random(int(step));device=torch.device(device)
    generator=torch.Generator(device=device).manual_seed(int(step))
    mix=int(getattr(model,'_stream_recovery_mix_cursor',0))
    short_cursor=int(getattr(model,'_stream_recovery_short_cursor',0))
    short_count=sum((mix+i)%4!=3 for i in range(samples));long_count=samples-short_count
    diagnostics=Counter();rows=[];caches={}
    modes=[(module,module.training) for module in model.modules()];model.eval()
    try:
        for _ in range(short_count):
            try:
                row=_short_exit_row(dataset,model,device,rng,generator,short_cursor,steps,caches,diagnostics)
                if row is not None:rows.append(row)
            except (ValueError,KeyError) as exc:
                diagnostics['skip_invalid_short']+=1;diagnostics['last_error']=str(exc)
            short_cursor+=1
        if long_count:
            persistent_count=long_count
            if policy=='greedy_mixed_v4':
                long_indices=[(mix+i)//4 for i in range(samples) if (mix+i)%4==3]
                persistent_count=sum(index%2==0 for index in long_indices)
                for index in long_indices:
                    if index%2==0:continue
                    try:
                        row=_synthetic_long_exit_row(dataset,model,rng,index//2,diagnostics)
                        if row is not None:rows.append(row)
                    except (ValueError,KeyError) as exc:
                        diagnostics['skip_invalid_synthetic']+=1;diagnostics['last_error']=str(exc)
                diagnostics['synthetic_long_slots']+=long_count-persistent_count
            if persistent_count:
                _,long_diagnostics=_persistent_recovery_batch(dataset,model,device,step,
                    samples=persistent_count,steps=(8 if policy in ('greedy_mixed_v3','greedy_mixed_v4') else steps),_rows_out=rows,_audio_caches=caches)
                for key,value in long_diagnostics.items():
                    if isinstance(value,(int,float)):diagnostics[key]+=value
                    else:diagnostics[key]=value
                diagnostics['persistent_long_slots']+=persistent_count
            diagnostics['long_slots']+=long_count
        diagnostics['short_slots']+=short_count
    finally:
        for module,mode in modes:module.training=mode
    model._stream_recovery_mix_cursor=mix+samples
    model._stream_recovery_short_cursor=short_cursor
    for key in ('new_episode','actual_exit_supervised','overshot'):diagnostics.setdefault(key,0)
    return (collate_stream(rows) if rows else None),dict(diagnostics)
