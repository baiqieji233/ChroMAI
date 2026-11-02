"""Controlled configuration/time loops with complete causal input rebuilds.

This runs on CPU rows before device transfer. Supervised event onsets and the
audio plan stay authored. Phase recovery deliberately changes the first
decision's cursor and recomputes its actual gap to that same future onset.
Every edit rebuilds event/state/continuation representations together.
"""
from collections import Counter
from dataclasses import replace
import random
from fractions import Fraction
import time

import torch

from maichart.event_timing import EventPoint
from .stream_data import NOTE,WAIT,EOS,collate_stream,IncrementalState,second_at_beat
from .stream_continuation import typed_event,pattern_event,continuation_rows,ContinuationState
from .stream_recovery import _context,_guard
from maichart.generation.muri.model import RELEASE
from torch.nn.utils.rnn import pad_sequence
from .guard import _event_syntax_error
from .events import event_metadata
from .stream_target_correspondence import corresponding_shift, rotated_text, rotation_boundary_clear
from maichart.generation.post_process import detect_chart_conflicts

LEGACY_INPUT_POLICY='causal_window_config_time_first_decision_v3'
PHASE_INPUT_POLICY='phase_recovery_v4'
MUSICAL_INPUT_POLICY='musical_loop_recovery_v5'
MUSICAL_RESIDUAL_INPUT_POLICY='musical_loop_recovery_v6'
MUSICAL_BATCH_INPUT_POLICY='musical_loop_recovery_v7'
MUSICAL_AUTHORED_EXIT_INPUT_POLICY='musical_loop_recovery_v8'
_LOOP_MODES=('same_key','abab','touch','same_shape','same_gap')


def _augmentation_modes(input_policy):
    if input_policy==LEGACY_INPUT_POLICY:return _LOOP_MODES
    if input_policy==PHASE_INPUT_POLICY:return (*_LOOP_MODES,'phase_shift')
    if input_policy in (MUSICAL_INPUT_POLICY,MUSICAL_RESIDUAL_INPUT_POLICY):return (*_LOOP_MODES,'phase_shift','rotation','rotation_gap')
    if input_policy in (MUSICAL_BATCH_INPUT_POLICY,MUSICAL_AUTHORED_EXIT_INPUT_POLICY):return (*_LOOP_MODES,'phase_shift','rotation','rotation_gap','timeline_loop')
    raise ValueError('Unknown prefix augmentation input policy')


def _encode_gap(dataset,gap):
    from .stream_timing import encode_history_gap
    return encode_history_gap(dataset.delta_vocab,gap,
        policy=getattr(dataset,'timing_output_policy','relative_delta_v1'))


def _loop_texts(dataset,source,begin,start,mode,rng):
    """Codec-known literal actions; EventGuard remains the physical authority."""
    if mode in ('rotation','rotation_gap'):
        first=rng.randrange(8);direction=(-1,1)[rng.randrange(2)]
        return [str((first+direction*i)%8+1) for i in range(start-begin)]
    if mode=='same_shape':
        candidates=[s.text for s in source[begin:start] if s.kind==NOTE]
        text=candidates[rng.randrange(len(candidates))]
        return [text]*(start-begin)
    if mode=='touch':
        catalog=dataset.codec.atom_catalog
        cached=getattr(dataset,'_augmentation_simple_touch',None)
        if cached is None or cached[0] is not catalog or cached[1]!=len(catalog):
            candidates=tuple(atom for atom in catalog
                             if typed_event(atom)[2] and not event_is_sustained(atom))
            cached=dataset._augmentation_simple_touch=(catalog,len(catalog),candidates)
        candidates=cached[2]
        if not candidates:raise ValueError('No known simple Touch action')
        text=candidates[rng.randrange(len(candidates))]
        return [text]*(start-begin)
    buttons=[str(i) for i in range(1,9)]
    rng.shuffle(buttons)
    first=buttons[0]
    if mode=='same_key':return [first]*(start-begin)
    return [buttons[i%2] for i in range(start-begin)]


def event_is_sustained(text):
    from .events import event_metadata
    meta=event_metadata(text)
    return meta.hold or meta.slide


def _enforced_hard_keys(result,guard):
    """Use precisely EventGuard's enforced native hard-conflict identity."""
    if result is None:return frozenset()
    return frozenset((issue.conflict_type,round(issue.time,6),
        tuple((str(note.beat),note.raw) for note in issue.notes))
        for issue in result.conflicts_detected
        if guard.enforced(issue))


def _validate_window(guard,steps):
    """Validate a complete training edit/target window in <=2 native calls.

    The cheap per-event rules retain EventGuard syntax, action lifetime, audio
    bounds and physical head cooldown. Native judging compares the full seeded
    old context against that context plus every edited and target action. This
    intentionally screens the whole label window rather than reproducing the
    generation guard's sequence of incremental transactions. No future label
    is supplied to the model's causal feature reconstruction.
    """
    points=[EventPoint(step.end,step.text) for step in steps if step.kind==NOTE]
    if not points:return
    last_heads=dict(guard.last_head_seconds)
    first_second=None
    for point in points:
        if _event_syntax_error(point.note_text):raise ValueError('Augmentation event syntax rejected')
        second,end,_sustained,errors=guard._lifetime(point)
        if errors or second<0. or point.beat<0 or end>guard.audio_end+1e-8:
            raise ValueError('Augmentation event lifetime or audio bounds rejected')
        if first_second is None:first_second=second
        for head in event_metadata(point.note_text).heads:
            pad=f'A{head}' if guard.touch_layout_policy=='legacy_two_heads' and str(head).isdigit() else str(head)
            prior=last_heads.get(pad)
            if guard.min_same_head_sec>0 and prior is not None and second-prior<guard.min_same_head_sec-1e-9:
                raise ValueError('Augmentation repeated physical head too soon')
            last_heads[pad]=second
    cutoff=guard.interaction_start(first_second)
    context=[row[0] for row in guard.prefix if row[2]>=cutoff or row[1]>=cutoff]
    before=detect_chart_conflicts(context,guard.tempo,guard.first,guard.config) if context else None
    after=detect_chart_conflicts(context+points,guard.tempo,guard.first,guard.config)
    if after.coverage.get('parse_errors') or _enforced_hard_keys(after,guard)-_enforced_hard_keys(before,guard):
        raise ValueError('Augmented window adds a native hard conflict')


def _native_prefix(dataset,song,source,begin,*,first_step=None):
    """Locate all live old actions from immutable parsed markers, never a cap.

    Reparse only source events carrying live actions so IncrementalState uses
    its identical local-parser/key/order semantics. A long hold can originate
    arbitrarily far before the edited history and is retained here.
    """
    native=dataset.base.songs[song][4]
    if native is None or not hasattr(native,'markers'):return None
    authored=tuple(dataset.views[song].context_points)
    if tuple(native.points)!=authored:return None
    if any('source_beat' not in marker for marker in native.markers):return None
    threshold=(source[begin] if first_step is None else first_step).end_second
    cutoff=source[begin].start
    # Guard considers recent onsets and actions ending within .3 seconds.
    live={marker['source_beat'] for marker in native.markers
          if marker['source_beat']<=cutoff
          and float(marker.get('end_second',-1.))+RELEASE>=threshold-.300001}
    prior=[step for step in source[:begin] if step.kind==NOTE]
    # A live long action needs all intervening interactions, not only its
    # own source and contacts immediately before the edited window.
    guard_cutoff=min([threshold-.300001]+[step.end_second-.300001 for step in prior if step.end in live])
    guard_prefix=[step for step in prior if step.end in live or step.end_second>=guard_cutoff]
    song_record=dataset.base.songs[song][0]
    incremental=IncrementalState(song_record.tempo,song_record.first)
    points=[EventPoint(step.end,step.text) for step in prior]
    state_live={marker['source_beat'] for marker in native.markers
                if marker['source_beat']<=cutoff
                and float(marker.get('end_second',-1.))+RELEASE>=threshold-.000001}
    for position,point in enumerate(points):
        if point.beat in state_live:
            incremental.points=points[:position]
            incremental.append(point)
    incremental.points=points
    return incremental,guard_prefix


def _phrase_fallback_prefix(dataset,row,song,source,begin,local):
    if local is not None or getattr(dataset,'training_window_policy','block_v1')!='phrase_complete_v1':return local
    # A seam may cross the next block boundary. Replaying through the final
    # seam query would crop away the primary block's earlier rows. Rebuild the
    # edited suffix inside its original fixed row from the actual old prefix.
    context=_context(dataset,song,row['_song_bank'])
    prefix=source[:begin]
    context.batch(prefix,prefix[-1].end if prefix else 0)
    return context._incremental_state,prefix


def _same_gap_steps(dataset,song,source,begin,start,end,rng,*,rotation_gap=False,rotation_gaps=None):
    """Pack a constant-gap loop into the exact existing prefix boundaries."""
    count=start-begin;left=source[begin].start;cursor=source[start].start
    first_note=next((step for step in source[start:end] if step.kind==NOTE),None)
    if first_note is None:raise ValueError('No authored NOTE target for constant-gap contrast')
    target_gap=first_note.end-first_note.start
    known=set(dataset.delta_vocab.values)
    candidates=[]
    options=(rotation_gaps or ('1/2',)) if rotation_gap else ('1/8','1/4','1/2','1','2')
    for gap in map(Fraction,options):
        remainder=cursor-(count-1)*gap-left
        # The stream grammar permits a NOTE at beat zero only at BOS. Keep
        # that authored initial zero when rotating an already half-beat prefix;
        # never introduce a zero transition at an interior history boundary.
        initial_zero=(rotation_gap and remainder==0 and begin==0 and left==0
            and source[0].kind==NOTE and source[0].start==source[0].end==0)
        if gap not in known or (not rotation_gap and gap==target_gap) or not (0<remainder<=2 or initial_zero) or remainder not in known:continue
        beats=[cursor-(count-1-i)*gap for i in range(count)]
        if rotation_gap or any(beats[i]!=source[begin+i].end for i in range(count)):
            candidates.append(beats)
    if not candidates:raise ValueError('No reachable constant-gap augmentation in the exact boundaries')
    beats=candidates[rng.randrange(len(candidates))]
    modified=list(source[:end]);clock=dataset.clocks[song];previous=left
    for offset,beat in enumerate(beats):
        original=source[begin+offset]
        modified[begin+offset]=replace(original,start=previous,end=beat,
            start_second=max(0.,second_at_beat(clock,previous)),
            end_second=max(0.,second_at_beat(clock,beat)))
        previous=beat
    return modified


def _phase_shift_steps(dataset,song,source,start,end,rng):
    """Move the committed cursor, keeping every supervised onset unchanged.

    A one-NOTE tail edit is enough to put the first decision on a genuinely
    wrong phase. Keeping a NOTE as the first target avoids inserting/removing
    WAIT steps or silently changing the block's target count. Rest/EOS remain
    supervised by the ordinary and free-running recovery batches.
    """
    if start<1 or source[start-1].kind!=NOTE or source[start].kind!=NOTE:
        raise ValueError('Phase recovery needs a preceding NOTE and next NOTE target')
    previous,target=source[start-1],source[start]
    clock=dataset.clocks[song]
    offsets=[sign*gap for gap in map(Fraction,('1/48','1/24','1/16','1/12','1/8'))
             for sign in (-1,1)]
    rng.shuffle(offsets)
    for shift in offsets:
        cursor=previous.end+shift
        prior_gap=cursor-previous.start;next_gap=target.end-cursor
        # Never supervise backwards, invent a zero-gap note, or turn a NOTE
        # into a >2-beat transition that the actual stream cannot execute.
        if not 0<prior_gap<=2 or not 0<next_gap<=2:continue
        if second_at_beat(clock,cursor)<0:continue
        try:
            _encode_gap(dataset,prior_gap);_encode_gap(dataset,next_gap)
        except ValueError:continue
        modified=list(source[:end])
        second=max(0.,second_at_beat(clock,cursor))
        modified[start-1]=replace(previous,end=cursor,end_second=second)
        modified[start]=replace(target,start=cursor,start_second=second)
        return modified,shift
    raise ValueError('No reachable phase recovery cursor before the authored target')


def _rebalance_first_decision(row):
    """Redistribute a successful augmented row's existing learning budget.

    Half of each positive supervised target weight remains on its authored
    decision. The other half of their total mass is assigned to the earliest
    positive supervised target, the decision directly after the loop. This is
    a normalized budget allocation, not an exit requirement: the target may
    correctly continue. Every later target keeps positive training weight,
    total row mass is conserved, and a row is never rebalanced twice.
    """
    if row.get('_augmentation_first_decision_rebalanced',False):return row
    valid=row['loss_mask'].bool()&row['sample_weight'].gt(0)
    positions=valid.nonzero(as_tuple=False).flatten()
    if not positions.numel():return row
    result=dict(row);weights=row['sample_weight'].clone();first=int(positions[0])
    total=weights[valid].sum()
    weights[valid]*=.5
    weights[first]+=.5*total
    # Keep the same representable sum under the original tensor reduction,
    # correcting only roundoff from floating-point summation at the first row.
    original_total=row['sample_weight'].sum()
    for _ in range(3):
        residual=original_total-weights.sum()
        if not bool(residual):break
        weights[first]+=residual
    result['sample_weight']=weights
    result['_augmentation_first_decision_rebalanced']=True
    result['_augmentation_first_decision_index']=first
    result['_augmentation_first_decision_mass']=float(weights[first])
    return result


def _record_first_decision(diagnostics,row,original=None):
    if row.get('_augmentation_first_decision_rebalanced',False) and not (
            original is not None and original.get('_augmentation_first_decision_rebalanced',False)):
        diagnostics['rebalanced_rows']+=1
        diagnostics['first_decision_count']+=1
        diagnostics['first_decision_mass']+=row['_augmentation_first_decision_mass']


def _set_configuration_qualification(row,start,first,end,*,trusted,reason):
    row['configuration_mask']=row['configuration_mask'].clone()
    row['configuration_mask'][start-first:end-first]=bool(trusted)
    row['_configuration_correspondence']=reason if trusted else 'time_only'
    # The cached phrase teacher is derived from the authored chart. Once the
    # prefix changes it cannot be used as either a positive or a negative.
    if 'target_phrase_mask' in row:
        row['target_phrase_mask']=row['target_phrase_mask'].clone()
        row['target_phrase_mask'][start-first:end-first]=False
        row['target_phrase_weight']=row['target_phrase_weight'].clone()
        row['target_phrase_weight'][start-first:end-first]=0.
    return row


def _record_configuration_qualification(diagnostics,row):
    active=row['loss_mask'].bool() & row['sample_weight'].gt(0) & row['target_kind'].eq(NOTE)
    trusted=active & row['configuration_mask'].bool()
    diagnostics['configuration_trusted']+=int(trusted.sum())
    diagnostics['time_only']+=int((active & ~row['configuration_mask'].bool()).sum())
    if trusted.any():diagnostics['correspondence_'+row.get('_configuration_correspondence','authored')]+=int(trusted.sum())


def _replace_rotated_targets(dataset,row,modified,start,first,end):
    for absolute in range(start,end):
        step=modified[absolute]
        if step.kind!=NOTE:continue
        at=absolute-first
        event=dataset.codec.encode(step.text)
        row['target_shape'][at]=event['shape']
        row['target_phase_mask'][at]=torch.tensor(event['phase_mask'],dtype=torch.bool)
        width=len(event['details'])
        if width>row['target_details'].size(-1):
            row['target_details']=torch.nn.functional.pad(row['target_details'],
                (0,width-row['target_details'].size(-1)),value=-1)
        row['target_details'][at].fill_(-1)
        row['target_details'][at,:width]=torch.tensor(event['details'])
    return row


def _rebuild_local_row(dataset,row,source,modified,song,begin,start,end,incremental,*,retime=False):
    """Reuse unchanged tensors; rebuild only the edited suffix + K targets."""
    first=max(0,start-dataset.lead_in);length=end-first
    row=dict(row)
    for key in ('history_kind','history_delta','history_gap_beats','history_shape','history_phase','history_atoms',
                'history_phase_mask','continuation_features','transition_exit_mask','transition_exit_labels'):
        if key in row:row[key]=row[key].clone()
    if retime:
        for key in ('step_beat','step_second','target_delta','target_second','target_beat'):
            if key in row:row[key]=row[key].clone()
    # Count reconstruction is cheap and uses cached event descriptors; it does
    # not parse action geometry or replay the native guard over old events.
    continuation=ContinuationState(policy=dataset.continuation_policy)
    for step in source[:begin]:continuation.append(step)
    states=[row['action_state'][i] for i in range(length)]
    masks=[row['action_state_mask'][i] for i in range(length)]
    candidate_states=([row['candidate_action_state'][i] for i in range(length)]
                      if 'candidate_action_state' in row else None)
    candidate_masks=([row['candidate_action_state_mask'][i] for i in range(length)]
                     if candidate_states is not None else None)
    for absolute in range(begin,end):
        step=modified[absolute];at=absolute-first
        if at>=0:
            if candidate_states is not None:
                from .stream_candidate_state import candidate_state_at
                candidate_states[at],candidate_masks[at]=candidate_state_at(incremental,modified,absolute)
            if retime:
                row['step_beat'][at]=float(step.start);row['step_second'][at]=step.start_second
                row['target_delta'][at]=_encode_gap(dataset,step.end-step.start)
                row['target_second'][at]=step.end_second
                if 'target_beat' in row:row['target_beat'][at]=float(step.end)
            row['continuation_features'][at]=continuation.features()
            eligible,label=continuation.exit_label(step)
            row['transition_exit_mask'][at]=eligible and bool(row['loss_mask'][at]) and row['sample_weight'][at]>0
            row['transition_exit_labels'][at]=label
        continuation.append(step)
        if absolute+1>=end:break
        event=dataset.codec.encode(step.text) if step.kind==NOTE else None
        if event is not None:incremental.append(EventPoint(step.end,step.text))
        target_at=absolute+1-first
        row['history_kind'][target_at]=step.kind
        row['history_delta'][target_at]=_encode_gap(dataset,step.end-step.start)
        if 'history_gap_beats' in row:row['history_gap_beats'][target_at]=float(step.end-step.start)
        row['history_shape'][target_at]=event['shape'] if event else 0
        row['history_phase'][target_at]=event['phase'] if event else 0
        row['history_phase_mask'][target_at]=torch.tensor(event['phase_mask']) if event else False
        width=len(event['atoms']) if event else 0
        if width>row['history_atoms'].size(-1):
            row['history_atoms']=torch.nn.functional.pad(row['history_atoms'],(0,width-row['history_atoms'].size(-1)))
        row['history_atoms'][target_at].zero_()
        if event:row['history_atoms'][target_at,:width]=torch.tensor(event['atoms'])
        if not incremental.points or step.end<0:
            # Use the dataset's state width, including deliberately small test
            # fixtures; production width remains the model's native STATE_DIM.
            state=torch.zeros(1,row['action_state'].size(-1));mask=torch.ones(1,dtype=torch.bool)
        else:
            from fractions import Fraction
            cutoff=step.end+Fraction(1,10**12) if step.kind==NOTE else step.end
            state,mask,_=incremental.at(step.end,include_current=True,visible_before_beat=cutoff,
                visibility_version='causal_v2',include_metadata=False)
        states[target_at]=state;masks[target_at]=mask
    row['action_state']=pad_sequence(states,batch_first=True)
    row['action_state_mask']=pad_sequence(masks,batch_first=True,padding_value=True)
    if candidate_states is not None:
        row['candidate_action_state']=pad_sequence(candidate_states,batch_first=True)
        row['candidate_action_state_mask']=pad_sequence(candidate_masks,batch_first=True,padding_value=True)
    if getattr(dataset,'phrase_context_policy','off') != 'off':
        dataset.rebuild_phrase_row(row,song,modified,begin,first,end)
    return row


def _rebuild_fallback_candidate_states(dataset,row,modified,song,first,end):
    """Requery every edited-prefix decision with one complete native index.

    A replay context may prune old markers for its short symbolic history.
    The full modified prefix restores long-lived sustains, while each query's
    strict visible cutoff still excludes its own and later targets.
    """
    if 'candidate_action_state' not in row:return
    from maichart.generation.sustain_state import SustainStateIndex
    from .stream_candidate_state import candidate_state_at
    song_source=dataset.base.songs[song][0]
    points=[EventPoint(step.end,step.text) for step in modified[:end] if step.kind==NOTE]
    index=SustainStateIndex(points,song_source.tempo,song_source.first) if points else None
    states,masks=[],[]
    for absolute in range(first,end):
        state,mask=candidate_state_at(index,modified,absolute)
        states.append(state);masks.append(mask)
    row['candidate_action_state']=pad_sequence(states,batch_first=True)
    row['candidate_action_state_mask']=pad_sequence(masks,batch_first=True,padding_value=True)


def _augment_phase_row(dataset,index,*,rng,row=None):
    """Rebuild a phase-shifted row with exact clocks and reachable labels."""
    song,start,end,_=dataset.samples[index];source=dataset.streams[song]
    row=dataset[index] if row is None else row
    if getattr(dataset,'training_window_policy','block_v1')=='phrase_complete_v1':
        end=int(row['step_index'][-1])+1
    first=max(0,start-dataset.lead_in);at=start-first
    if not bool(row['loss_mask'][at]) or not bool(row['sample_weight'][at]>0):
        raise ValueError('Phase recovery requires a known supervised first decision')
    modified,shift=_phase_shift_steps(dataset,song,source,start,end,rng)
    begin=start-1
    local=_native_prefix(dataset,song,source,begin,first_step=modified[begin])
    local=_phrase_fallback_prefix(dataset,row,song,source,begin,local)
    guard_prefix=modified[:begin] if local is None else local[1]
    physical=getattr(dataset,'physical_policy','legacy_report_v1')
    guard=_guard(dataset.base.songs[song][0],guard_prefix,len(guard_prefix),
                 **({'physical_policy':physical} if physical!='legacy_report_v1' else {}))
    _validate_window(guard,modified[begin:end])
    started=time.perf_counter()
    if local is not None:
        result=_rebuild_local_row(dataset,row,source,modified,song,begin,start,end,local[0])
    else:
        result=dict(row)
        context=_context(dataset,song,row['_song_bank'])
        prefix=modified[:end-1]
        batch=context.batch(prefix,prefix[-1].end if prefix else 0)
        if batch['history_kind'].size(1)!=row['history_kind'].size(0):
            raise ValueError('Phase recovery changed the block-aligned history length')
        for key,value in batch.items():
            if key.startswith('history_') or key in ('action_state','action_state_mask','continuation_features'):
                result[key]=value[0]
        _,eligible,labels=continuation_rows(modified,policy=dataset.continuation_policy)
        first=max(0,start-dataset.lead_in)
        result['transition_exit_mask']=eligible[first:end]&row['loss_mask']&row['sample_weight'].gt(0)
        result['transition_exit_labels']=labels[first:end]
    first=max(0,start-dataset.lead_in)
    for key in ('step_beat','step_second','target_delta','target_second','target_beat'):
        if key in result:result[key]=result[key].clone()
    for absolute in (begin,start):
        at=absolute-first
        if at<0:continue
        step=modified[absolute]
        result['step_beat'][at]=float(step.start)
        result['step_second'][at]=step.start_second
        result['target_delta'][at]=_encode_gap(dataset,step.end-step.start)
        result['target_second'][at]=step.end_second
        if 'target_beat' in result:result['target_beat'][at]=float(step.end)
    result['_augmentation_phase_shift_beats']=float(shift)
    result['_augmentation_phase_local_rebuild']=local is not None
    result['_augmentation_timing_replay_sec']=time.perf_counter()-started
    if local is None:
        _rebuild_fallback_candidate_states(dataset,result,modified,song,first,end)
    if getattr(dataset,'phrase_context_policy','off') != 'off':
        dataset.rebuild_phrase_row(result,song,modified,begin,first,end)
    if getattr(dataset,'recovery_target_policy','authored_next_v1')=='causal_correspondence_v1':
        # Only the preceding clock changes: its event text and the entire
        # authored target structure stay intact, and _validate_window proved
        # the native target actions legal in the changed state.
        same_structure=all(modified[i].kind==source[i].kind and
            modified[i].text==source[i].text for i in range(begin,end))
        result=_set_configuration_qualification(result,start,first,end,
            trusted=same_structure,reason='phase_shift_same_configuration')
    return _rebalance_first_decision(result),'phase_shift',1


def _authored_timeline_exit(source,index):
    target=source[index]
    if target.kind in (WAIT,EOS):return True
    if index<1 or source[index-1].kind!=NOTE or target.kind!=NOTE:return False
    previous=source[index-1]
    return (target.end-target.start!=previous.end-previous.start
        or pattern_event(target.text)[0]!=pattern_event(previous.text)[0])


def _augment_timeline_row(dataset,index,*,rng,row=None,input_policy=MUSICAL_BATCH_INPUT_POLICY):
    """Use real inference layout for a long wrong prefix and one real target.

    Replace this window's supervision with its first trusted decision, while
    retaining exactly the old action-loss and planner budgets. This is not
    the sparse recovery objective and must not inherit its zero plan scale.
    """
    from .stream_recovery import _synthetic_long_exit_row_at
    row=dataset[index] if row is None else row
    valid=row['loss_mask'].bool()&row['sample_weight'].gt(0)
    positions=valid.nonzero(as_tuple=False).flatten()
    if not positions.numel():raise ValueError('Timeline loop needs a trusted target')
    song,start,end,_=dataset.samples[index]
    first=max(0,start-dataset.lead_in)
    source=dataset.streams[song]
    target_index=first+int(positions[0]);authored=False;positive_control=False
    if input_policy==MUSICAL_AUTHORED_EXIT_INPUT_POLICY:
        prefer_exit=rng.random()<.75
        positive_control=not prefer_exit
        if prefer_exit:
            candidate=next((first+int(at) for at in positions if _authored_timeline_exit(source,first+int(at))),None)
            if candidate is not None:target_index=candidate;authored=True
    cursor=rng.randrange(2**31);result=None
    if authored and target_index>0:
        reference=source[target_index-1]
        try:
            result=_synthetic_long_exit_row_at(dataset,song,target_index,rng,cursor,Counter(),
                reference=reference,repeat_gap=reference.end-reference.start)
        except (ValueError,KeyError):pass
    if result is None:
        result=_synthetic_long_exit_row_at(dataset,song,target_index,rng,cursor,Counter())
    if result is None:raise ValueError('No reachable timeline-loop history')
    # recovery_sample already supplies one active final target. Assign the
    # window's complete mass directly, without target/chart double weighting
    # and without applying the existing first-decision rebalance a second time.
    result['sample_weight'].zero_()
    result['sample_weight'][-1]=row['sample_weight'].sum()
    if 'training_phrase_group' in row:
        length=len(result['history_kind'])
        result['training_phrase_group']=torch.full((length,),-1,dtype=torch.long)
        result['training_phrase_position']=torch.full((length,),-1,dtype=torch.long)
        result['training_phrase_complete']=torch.zeros(length,dtype=torch.bool)
        result['_training_target_budget']=torch.zeros(length,dtype=torch.float32)
        result['_training_target_budget'][-1]=row['_training_target_budget'].sum()
        result['_training_reference_targets']=row['_training_reference_targets']
        if 'target_relation' in row:
            shape=(length,*row['target_relation'].shape[1:])
            result['target_relation']=torch.zeros(shape,dtype=row['target_relation'].dtype)
            result['target_relation_mask']=torch.zeros(shape,dtype=torch.bool)
    for key in ('_sample_index','_window_weight','_song_bank','_song_plan','_literal_atom_types'):
        if key in row:result[key]=row[key]
    if '_plan_scale' in row:result['_plan_scale']=row['_plan_scale']
    else:result.pop('_plan_scale',None)
    result['condition']=row['condition']
    result['_augmentation_timeline_loop']=True
    if input_policy==MUSICAL_AUTHORED_EXIT_INPUT_POLICY:
        target=source[target_index]
        result['_timeline_authored_exit']=authored
        result['_timeline_positive_control']=positive_control
        result['_timeline_actual_time_exit']=(target.kind in (WAIT,EOS)
            or target.end-target.start!=Fraction(result['_synthetic_loop_gap']))
        result['_timeline_actual_template_exit']=(target.kind in (WAIT,EOS)
            or pattern_event(target.text)[0]!=pattern_event(result['_synthetic_loop_last_text'])[0])
    return result,'timeline_loop',result['_synthetic_loop_events']


def augment_stream_row(dataset,index,*,rng,mode=None,row=None,input_policy=LEGACY_INPUT_POLICY):
    """Return a rebuilt row or raise ValueError when a legal loop cannot fit.

    All supervised targets remain authored. Legacy modes edit 4--12 NOTE
    actions; musical policies also cover sparse 16--128-event loops. Timing
    modes keep exact outer boundaries and rebuild the full causal state.
    """
    if dataset.history_window_policy!='block_aligned_v2':
        raise ValueError('Augmentation requires block-aligned rows')
    modes=_augmentation_modes(input_policy)
    # Preserve the old policy's random draw order exactly. The new recipe may
    # choose a one-note phase edit before requiring a 4--12-note loop window.
    if mode is None and input_policy in (PHASE_INPUT_POLICY,MUSICAL_INPUT_POLICY,MUSICAL_RESIDUAL_INPUT_POLICY,MUSICAL_BATCH_INPUT_POLICY,MUSICAL_AUTHORED_EXIT_INPUT_POLICY):mode=modes[rng.randrange(len(modes))]
    if mode is not None and mode not in modes:raise ValueError('Unknown augmentation mode for this input policy')
    if mode=='phase_shift':return _augment_phase_row(dataset,index,rng=rng,row=row)
    if mode=='timeline_loop':return _augment_timeline_row(dataset,index,rng=rng,row=row,input_policy=input_policy)
    song,start,end,_=dataset.samples[index]
    first=max(0,start-dataset.lead_in)
    source=dataset.streams[song]
    row=dataset[index] if row is None else row
    if getattr(dataset,'training_window_policy','block_v1')=='phrase_complete_v1':
        end=int(row['step_index'][-1])+1
    available=0
    musical=input_policy in (MUSICAL_INPUT_POLICY,MUSICAL_RESIDUAL_INPUT_POLICY,MUSICAL_BATCH_INPUT_POLICY,MUSICAL_AUTHORED_EXIT_INPUT_POLICY)
    max_history=min(128,dataset.lead_in) if musical else 12
    for step in reversed(source[max(0,start-max_history):start]):
        if step.kind!=NOTE:break
        available+=1
    if available<4:raise ValueError('Fewer than four preceding NOTE actions')
    if musical:
        lengths,weights=zip(*[(n,w) for n,w in zip((4,8,16,32,64,128),(4,4,3,2,1,1)) if n<=available])
        count=rng.choices(lengths,weights=weights,k=1)[0]
    else:count=rng.randint(4,min(12,available))
    begin=start-count
    mode=mode or modes[rng.randrange(len(modes))]
    timing_mode=mode in ('same_gap','rotation_gap')
    if timing_mode:
        modified=_same_gap_steps(dataset,song,source,begin,start,end,rng,rotation_gap=mode=='rotation_gap',
            rotation_gaps=('1/4','1/2','1','2') if input_policy in (MUSICAL_RESIDUAL_INPUT_POLICY,MUSICAL_BATCH_INPUT_POLICY,MUSICAL_AUTHORED_EXIT_INPUT_POLICY) else None)
    else:
        modified=list(source[:end])
    if mode!='same_gap':
        texts=_loop_texts(dataset,source,begin,start,mode,rng)
        for text in texts:dataset.codec.encode(text)
        for offset,text in enumerate(texts):modified[begin+offset]=replace(modified[begin+offset],text=text)
        if all(modified[i].text==source[i].text for i in range(begin,start)):
            raise ValueError('Augmentation would leave the prefix unchanged')
    target_policy=getattr(dataset,'recovery_target_policy','authored_next_v1')
    shift=None
    if target_policy=='causal_correspondence_v1':
        shift=corresponding_shift(source[begin:start],modified[begin:start])
        if shift is not None and shift!=0:
            try:
                translated={i:rotated_text(source[i].text,shift) for i in range(start,end)
                            if source[i].kind==NOTE}
                for text in translated.values():dataset.codec.encode(text)
            except (ValueError,KeyError):
                shift=None
            else:
                for i,text in translated.items():modified[i]=replace(modified[i],text=text)
    # Authored events before the edit are committed as context. Check edited
    # actions and the complete unchanged target block against their new state.
    local=_native_prefix(dataset,song,source,begin,first_step=modified[begin]) if timing_mode else _native_prefix(dataset,song,source,begin)
    local=_phrase_fallback_prefix(dataset,row,song,source,begin,local)
    guard_prefix=modified[:begin] if local is None else local[1]
    physical=getattr(dataset,'physical_policy','legacy_report_v1')
    guard=_guard(dataset.base.songs[song][0],guard_prefix,len(guard_prefix),
                 **({'physical_policy':physical} if physical!='legacy_report_v1' else {}))
    if shift is not None and shift!=0 and not rotation_boundary_clear(source[:begin],
            modified[begin].start_second,guard):
        shift=None
        for i in range(start,end):modified[i]=source[i]
    _validate_window(guard,modified[begin:end])
    row=dataset[index] if row is None else row
    if local is not None and (not timing_mode or musical):
        rebuilt=_rebuild_local_row(dataset,row,source,modified,song,begin,start,end,local[0],retime=timing_mode)
        if target_policy=='causal_correspondence_v1':
            if shift is not None and shift!=0:rebuilt=_replace_rotated_targets(dataset,rebuilt,modified,start,first,end)
            rebuilt=_set_configuration_qualification(rebuilt,start,first,end,
                trusted=shift is not None,reason='rotation_correspondence' if shift else 'exact_history')
        rebuilt['_augmentation_local_rebuild']=True
        return _rebalance_first_decision(rebuilt),mode,count
    row=dict(row)
    replay_started=time.perf_counter()
    bank=row['_song_bank'];context=_context(dataset,song,bank)
    prefix=modified[:end-1]
    batch=context.batch(prefix,prefix[-1].end if prefix else 0)
    first=max(0,start-dataset.lead_in)
    for key in ('step_index','step_beat','step_second','history_delta'):
        if timing_mode and key!='step_index':
            # The target block starts at exactly the original cursor; its
            # history_delta at the first target correctly reflects the newly
            # committed loop, while its own predicted delta remains authored.
            if key!='history_delta' and not torch.equal(row[key][start-first:],batch[key][0,start-first:]):
                raise ValueError('Timing augmentation moved the supervised target block')
            row[key]=batch[key][0]
            continue
        if not torch.equal(row[key],batch[key][0]):
            raise ValueError('Augmentation changed the training time/window contract')
    for key,value in batch.items():
        if key.startswith('history_') or key in ('action_state','action_state_mask','continuation_features'):
            row[key]=value[0]
    if timing_mode:
        row['target_delta']=row['target_delta'].clone()
        row['target_second']=row['target_second'].clone()
        if 'target_beat' in row:row['target_beat']=row['target_beat'].clone()
        for absolute in range(begin,start):
            at=absolute-first
            if at>=0:
                step=modified[absolute]
                row['target_delta'][at]=_encode_gap(dataset,step.end-step.start)
                row['target_second'][at]=step.end_second
                if 'target_beat' in row:row['target_beat'][at]=float(step.end)
    _,eligible,labels=continuation_rows(modified,policy=dataset.continuation_policy)
    row['transition_exit_mask']=eligible[first:end]&row['loss_mask']&row['sample_weight'].gt(0)
    row['transition_exit_labels']=labels[first:end]
    if timing_mode:row['_augmentation_timing_replay_sec']=time.perf_counter()-replay_started
    _rebuild_fallback_candidate_states(dataset,row,modified,song,first,end)
    if getattr(dataset,'phrase_context_policy','off') != 'off':
        dataset.rebuild_phrase_row(row,song,modified,begin,first,end)
    if target_policy=='causal_correspondence_v1':
        if shift is not None and shift!=0:row=_replace_rotated_targets(dataset,row,modified,start,first,end)
        row=_set_configuration_qualification(row,start,first,end,
            trusted=shift is not None,reason='rotation_correspondence' if shift else 'exact_history')
    return _rebalance_first_decision(row),mode,count


def augment_stream_items(dataset,items,*,rate=.15,seed=0,input_policy=LEGACY_INPUT_POLICY):
    """Producer-side augmentation: reuse already fetched rows, collate once."""
    if not 0<=rate<=1:raise ValueError('Prefix augmentation rate must be in [0,1]')
    diagnostics=Counter()
    if not rate:return items,dict(diagnostics)
    rng=random.Random(int(seed));result=list(items)
    for slot,row in enumerate(items):
        if rng.random()>=rate:continue
        diagnostics['selected']+=1;diagnostics['attempted']+=1
        index=row.get('_sample_index')
        if index is None:
            diagnostics['skipped_missing_sample_indices']+=1;continue
        try:
            replacement,mode,count=augment_stream_row(dataset,index,rng=rng,row=row,input_policy=input_policy)
            result[slot]=replacement
            diagnostics['applied']+=1;diagnostics['applied_'+mode]+=1
            diagnostics['edited_notes']+=count
            _record_configuration_qualification(diagnostics,replacement)
            _record_first_decision(diagnostics,replacement,row)
            if mode=='timeline_loop':
                diagnostics['timeline_loop_events']+=count
                diagnostics['timeline_'+replacement['_synthetic_loop_mode']]+=1
                if '_timeline_authored_exit' in replacement:
                    diagnostics['timeline_authored_exit']+=int(replacement['_timeline_authored_exit'])
                    diagnostics['timeline_positive_control']+=int(replacement['_timeline_positive_control'])
                    diagnostics['timeline_reference_aligned']+=int(replacement['_synthetic_loop_reference_aligned'])
                    diagnostics['timeline_actual_time_exit']+=int(replacement['_timeline_actual_time_exit'])
                    diagnostics['timeline_actual_template_exit']+=int(replacement['_timeline_actual_template_exit'])
            if mode in ('same_gap','rotation_gap') and not replacement.get('_augmentation_local_rebuild',False):
                diagnostics['timing_full_replay_rows']+=1
                diagnostics['timing_full_replay_sec']+=replacement.get('_augmentation_timing_replay_sec',0.)
            if mode=='phase_shift':
                diagnostics['phase_shift_rows']+=1
                diagnostics['phase_shift_abs_beats']+=abs(replacement['_augmentation_phase_shift_beats'])
                diagnostics['phase_shift_local_rows']+=int(replacement['_augmentation_phase_local_rebuild'])
                diagnostics['phase_shift_rebuild_sec']+=replacement.get('_augmentation_timing_replay_sec',0.)
                if not replacement['_augmentation_phase_local_rebuild']:
                    diagnostics['timing_full_replay_rows']+=1
                    diagnostics['timing_full_replay_sec']+=replacement.get('_augmentation_timing_replay_sec',0.)
        except (ValueError,KeyError):diagnostics['skipped_unreachable_or_conflict']+=1
    return result,dict(diagnostics)


def augment_stream_batch(dataset,batch,*,rate=.15,seed=0,input_policy=LEGACY_INPUT_POLICY):
    """CPU batch-in/batch-out; independent RNG and flat numeric diagnostics."""
    if not 0<=rate<=1:raise ValueError('Prefix augmentation rate must be in [0,1]')
    diagnostics=Counter()
    if not rate:return batch,dict(diagnostics)
    indices=batch.get('_sample_indices')
    if not indices or any(index is None for index in indices):
        return batch,{'skipped_missing_sample_indices':1}
    rng=random.Random(int(seed))
    selected=[i for i in range(len(indices)) if rng.random()<rate]
    diagnostics['selected']=len(selected)
    if not selected:return batch,dict(diagnostics)
    replacements={}
    for slot in selected:
        diagnostics['attempted']+=1
        try:
            row,mode,count=augment_stream_row(dataset,indices[slot],rng=rng,input_policy=input_policy)
            replacements[slot]=row
            diagnostics['applied']+=1;diagnostics['applied_'+mode]+=1
            diagnostics['edited_notes']+=count
            _record_configuration_qualification(diagnostics,row)
            _record_first_decision(diagnostics,row)
            if mode=='timeline_loop':
                diagnostics['timeline_loop_events']+=count
                diagnostics['timeline_'+row['_synthetic_loop_mode']]+=1
                if '_timeline_authored_exit' in row:
                    diagnostics['timeline_authored_exit']+=int(row['_timeline_authored_exit'])
                    diagnostics['timeline_positive_control']+=int(row['_timeline_positive_control'])
                    diagnostics['timeline_reference_aligned']+=int(row['_synthetic_loop_reference_aligned'])
                    diagnostics['timeline_actual_time_exit']+=int(row['_timeline_actual_time_exit'])
                    diagnostics['timeline_actual_template_exit']+=int(row['_timeline_actual_template_exit'])
            if mode=='same_gap':
                diagnostics['timing_full_replay_rows']+=1
                diagnostics['timing_full_replay_sec']+=row.get('_augmentation_timing_replay_sec',0.)
            if mode=='phase_shift':
                diagnostics['phase_shift_rows']+=1
                diagnostics['phase_shift_abs_beats']+=abs(row['_augmentation_phase_shift_beats'])
                diagnostics['phase_shift_local_rows']+=int(row['_augmentation_phase_local_rebuild'])
                diagnostics['phase_shift_rebuild_sec']+=row.get('_augmentation_timing_replay_sec',0.)
                if not row['_augmentation_phase_local_rebuild']:
                    diagnostics['timing_full_replay_rows']+=1
                    diagnostics['timing_full_replay_sec']+=row.get('_augmentation_timing_replay_sec',0.)
        except (ValueError,KeyError):
            diagnostics['skipped_unreachable_or_conflict']+=1
    if not replacements:return batch,dict(diagnostics)
    rows=[replacements[slot] if slot in replacements else dataset[index]
          for slot,index in enumerate(indices)]
    return collate_stream(rows),dict(diagnostics)
