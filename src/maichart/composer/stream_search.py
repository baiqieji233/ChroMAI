"""Bounded phrase search over isolated causal rollouts and a shared audio cache.

The authoritative state is a StreamStep log. Repairs and withdrawals truncate
that log and replay it through the normal inference context; tensor caches of
prefix state are never patched in place.
"""
from __future__ import annotations

import math
import re
import time
from fractions import Fraction

from maichart.event_timing import EventPoint
from maichart.generation.post_process import detect_chart_conflicts, _extract
from maichart.generation.note_repair import candidate_note_edits
from .stream_data import NOTE, WAIT, StreamStep, second_at_beat


def _points(steps):
    return [EventPoint(s.end,s.text) for s in steps if s.kind==NOTE]


def _cut(steps,end,clock,prefix_length=0):
    """Commit actual sampled steps; never invent a WAIT at a budget boundary."""
    result=[]
    for step in steps:
        if step.end>end:break
        result.append(step)
    if len(result)<=prefix_length and len(steps)>prefix_length:
        result=list(steps[:prefix_length+1])
    return result


def _judge(points,song,config,metrics):
    started=time.perf_counter()
    try:return detect_chart_conflicts(points,song.tempo,song.first,config)
    finally:
        metrics['native_calls']+=1
        metrics['native_sec']+=time.perf_counter()-started


def _gate(result,song,metrics,lifetimes):
    if '_physical_gate' in result:return result['_physical_gate']
    guard=result['guard']
    anchor=float(result.get('anchor_second',0.))
    relevant=[]
    for point in result['points']:
        key=(point.beat,point.note_text)
        if key not in lifetimes:
            lifetimes[key]=guard._lifetime(point)
            if len(lifetimes)>4096:lifetimes.pop(next(iter(lifetimes)))
        second,finish,_,_=lifetimes[key]
        # Exactly the guard's existing active-prefix rule, plus all new points.
        # Every omitted prefix action was already accepted and ended before
        # this anchor's 300 ms interaction horizon.
        if second>=anchor-.3 or finish>=anchor-.3:relevant.append(point)
    physical=_judge(relevant,song,guard.config,metrics)
    result['_physical_gate']=(physical,[issue for issue in physical.conflicts_detected if guard.enforced(issue)])
    return result['_physical_gate']


def _duration_signature(raw):
    return re.sub(r'\[[^\]]+\]', '[]', raw)


def _repair_prefix(result,song,clock,metrics,lifetimes,withdraw=False,codec=None,duration_catalog=None,deadline=None):
    """Yield one edit/withdrawal at a time, with its entire suffix invalidated."""
    guard=result['guard'];steps=result['steps'];points=result['points']
    proposal=getattr(guard,'last_proposal',None)
    issues=list(getattr(guard,'failure_conflicts',()))
    if getattr(guard,'failure_trials',None):
        proposal,rejected_issues=guard.failure_trials[-1]
        issues=list(rejected_issues)
    physical,hard=_gate(result,song,metrics,lifetimes)
    issues+=hard
    trial_points=list(points)
    if proposal is not None and (not points or proposal!=points[-1]):trial_points.append(proposal)
    events,texts,_=_extract(trial_points,song.tempo,song.first)
    involved={(str(n.beat),n.raw) for issue in issues for n in issue.notes}
    def keys(report):
        # Deleting an old event shifts source IDs. Musical identities do not.
        return {(x.conflict_type,round(x.time,6),tuple((str(n.beat),n.raw) for n in x.notes))
                for x in report.conflicts_detected if guard.enforced(x)}
    before=_judge(trial_points,song,guard.config,metrics)
    before_keys=keys(before)
    def improves(counter):
        if deadline is not None and time.perf_counter()>=deadline:return False
        after=_judge(counter,song,guard.config,metrics)
        after_keys=keys(after)
        return not after.coverage.get('parse_errors') and len(after_keys)<len(before_keys) and after_keys<=before_keys
    for event in events:
        if deadline is not None and time.perf_counter()>=deadline:return
        if event.note_type not in {'hold','touch_hold','slide'} or event.protected:continue
        point_index=event.source_id[0]
        if point_index>=len(points):continue
        point=points[point_index]
        if (str(event.beat),event.raw) not in involved:
            # Runtime findings can name only the blocked new head. Look for
            # still-active earlier sustains, then prove the link by rejudging.
            onset,finish,_,_=guard._lifetime(point)
            if not any(onset<=issue.time<=finish+.3 for issue in issues):continue
        step_index=next((i for i,s in enumerate(steps) if s.kind==NOTE and s.end==point.beat and s.text==point.note_text),None)
        if step_index is None:continue
        if withdraw:
            # The old onset and every dependent future step are discarded.
            if improves([p for i,p in enumerate(trial_points) if i!=point_index]):
                yield steps[:step_index],dict(operation='withdraw_sustain_and_regenerate',beat=str(point.beat),raw=event.raw)
            continue
        edits=[edit for edit in candidate_note_edits(event,None,events,song.tempo,song.first)
               if edit['operation']=='shorten_hold']
        # Preserve the original path and head: only try trained duration
        # spellings, never turn a duration repair into an endpoint move.
        original_end=guard._lifetime(EventPoint(point.beat,event.raw))[1]
        bpm=next(t.bpm for t in reversed(song.tempo) if t.beat<=point.beat)
        def slide_waits(raw):
            markers,_=guard._markers(raw,bpm)
            return tuple(m.get('slide_trace_second') for m in markers if m.get('type') in {'slide','wifi'})
        original_waits=slide_waits(event.raw) if event.note_type=='slide' else ()
        alternatives=[]
        for raw in (duration_catalog or {}).get(_duration_signature(event.raw),()):
            if deadline is not None and time.perf_counter()>=deadline:return
            if event.note_type=='slide' and slide_waits(raw)!=original_waits:continue
            finish=guard._lifetime(EventPoint(point.beat,raw))[1]
            if finish<original_end-1e-8:
                alternatives.append((original_end-finish,raw))
        edits.extend(dict(raw=raw,operation='shorten_slide' if event.note_type=='slide' else 'shorten_hold')
                     for _,raw in sorted(alternatives)[:8])
        for edit in edits:
            replacement='/'.join(edit['raw'] if key==event.source_id else raw
                                 for key,raw in texts.items() if key[0]==point_index)
            if codec is not None:
                try:codec.encode(replacement)
                except (ValueError,KeyError):continue
            changed=StreamStep(NOTE,steps[step_index].start,steps[step_index].end,
                steps[step_index].start_second,steps[step_index].end_second,replacement)
            prefix=steps[:step_index]+[changed]
            # Verify the exact edited full counterfactual before regenerating.
            counter=[EventPoint(p.beat,replacement) if i==point_index else p for i,p in enumerate(trial_points)]
            if not improves(counter):continue
            yield prefix,dict(operation=edit['operation'],beat=str(point.beat),raw=event.raw,replacement=edit['raw'])


def search_stream(rollout,song,condition,args,end,clock,curve,audio_bank=None,codec=None):
    from .stream_quality import score_phrase, PhraseQualityScorer
    scorer=PhraseQualityScorer(song,condition,audio_bank=audio_bank)
    count=int(getattr(args,'phrase_candidates',1))
    write=float(getattr(args,'phrase_write_beats',8.));commit=float(getattr(args,'phrase_commit_beats',4.))
    seconds=float(getattr(args,'phrase_time_budget_sec',30.))
    repair_budget=int(getattr(args,'phrase_repair_budget',8));rollback_budget=int(getattr(args,'phrase_rollback_budget',4))
    strength=float(getattr(args,'softcomfort_strength',1.))
    if count not in (1,2,4) or not all(math.isfinite(x) for x in (write,commit,seconds,strength)):
        raise ValueError('invalid phrase search configuration')
    if not 0<commit<=write or seconds<=0 or not 0<=strength<=2 or repair_budget<0 or rollback_budget<0 or int(getattr(args,'phrase_step_budget',256))<1:
        raise ValueError('phrase budgets must be positive; commit <= write; repair/rollback/strength nonnegative')
    committed=[];retained=[];records=[];phrase_index=0;repairs_used=rollbacks_used=0
    native_metrics=dict(native_calls=0,native_sec=0.);lifetimes={}
    duration_catalog={}
    for atom in codec.atom_catalog if codec is not None else ():
        if '[' in atom:duration_catalog.setdefault(_duration_signature(atom),[]).append(atom)
    aggregate=dict(rollouts=0,configuration_retries=0,rejected_onset_attempts=0,guard_checks=0,guard_rejections=0,forced_waits=0,guard_reasons={})
    while (committed[-1].end if committed else Fraction(0))<end:
        start=committed[-1].end if committed else Fraction(0)
        target=min(end,start+Fraction(str(write)));commit_end=min(end,start+Fraction(str(commit)))
        deadline=time.perf_counter()+seconds;valid=[];failures=[];candidate_errors=[];attempt=0
        def run(prefix):
            nonlocal attempt
            seed=(int(args.seed)+phrase_index*1000003+attempt*9176)%(2**63-1);attempt+=1
            try:
                result=rollout(prefix,seed,target,deadline)
                physical,hard=_gate(result,song,native_metrics,lifetimes)
            except ValueError as exc:
                candidate_errors.append(dict(seed=seed,error=str(exc)))
                return
            aggregate['rollouts']+=1;aggregate['configuration_retries']+=result['retries']
            aggregate['rejected_onset_attempts']+=result['timing_retries']
            aggregate['guard_checks']+=result['guard'].checks;aggregate['guard_rejections']+=result['guard'].rejections
            aggregate['forced_waits']+=result['forced_waits']
            for key,value in result['guard'].reason_counts.items():
                aggregate['guard_reasons'][key]=aggregate['guard_reasons'].get(key,0)+value
            complete=result['cursor']>=target and result['reason'] not in {'physical_dead_end','max_events','step_budget','time_budget'}
            if not complete or hard or physical.coverage.get('parse_errors'):
                failures.append(result);return
            score=score_phrase(result['points'],_points(prefix),song,condition,start,target,
                               softcomfort_strength=strength,scorer=scorer)
            valid.append((float(score['total']),result,score))
        # All requested primary candidates start at exactly the same prefix.
        for candidate_index in range(count):
            if time.perf_counter()>=deadline:break
            run(list(committed)+(list(retained) if candidate_index==0 else []))
        changes=[];current_event_retries=[]
        if not valid:
            # Retry only the current unaccepted event from a private accepted
            # prefix. Keep its real timing/type distribution and temperature.
            retry_sources=list(failures)
            for retry_index in range(2):
                if valid or not retry_sources or repairs_used>=repair_budget or time.perf_counter()>=deadline:break
                failed=retry_sources[retry_index%len(retry_sources)]
                if failed['reason']!='physical_dead_end':continue
                repairs_used+=1
                run(list(failed['steps']))
                retry_record=dict(operation='current_event_retry',beat=str(failed['cursor']),accepted=bool(valid))
                current_event_retries.append(retry_record)
                if valid:changes.append(retry_record)
        if not valid and repairs_used<repair_budget:
            for failed in list(failures):
                if valid or time.perf_counter()>=deadline:break
                for prefix,edit in _repair_prefix(failed,song,clock,native_metrics,lifetimes,
                                                codec=codec,duration_catalog=duration_catalog,deadline=deadline):
                    if repairs_used>=repair_budget or time.perf_counter()>=deadline:break
                    repairs_used+=1;run(prefix)
                    if valid:changes.append(edit);break
        if not valid and rollbacks_used<rollback_budget:
            for failed in list(failures):
                if valid or time.perf_counter()>=deadline:break
                for prefix,edit in _repair_prefix(failed,song,clock,native_metrics,lifetimes,withdraw=True,
                                                codec=codec,duration_catalog=duration_catalog,deadline=deadline):
                    if rollbacks_used>=rollback_budget or time.perf_counter()>=deadline:break
                    rollbacks_used+=1;run(prefix)
                    if valid:changes.append(edit);break
        if not valid:
            raise ValueError(f'phrase search exhausted at beat {start}: no complete physically feasible candidate; chart not exported')
        _,winner,score=max(valid,key=lambda item:item[0])
        previous_length=sum(step.end<=start for step in winner['steps'])
        committed=_cut(winner['steps'],commit_end,clock,previous_length)
        retained=winner['steps'][len(committed):]
        if changes:
            changed_from=min(Fraction(edit['beat']) for edit in changes)
            for prior in records:
                affected_end=max(Fraction(prior['write_end_beat']),Fraction(prior.get('actual_commit_end_beat',prior['commit_end_beat'])))
                if affected_end>=changed_from:
                    prior['invalidated_by_suffix_regeneration']=True
        records.append(dict(start_beat=str(start),write_end_beat=str(target),commit_end_beat=str(commit_end),
            actual_commit_end_beat=str(committed[-1].end if committed else Fraction(0)),
            requested_candidates=count,attempted_rollouts=attempt,valid_candidates=len(valid),
            score=score,changes=changes,current_event_retries=current_event_retries,candidate_errors=candidate_errors,
            elapsed_sec=seconds-max(0.,deadline-time.perf_counter())))
        phrase_index+=1
        if target==end and winner['cursor']>=end:
            # EOS/end-of-audio is terminal metadata, never a synthetic WAIT.
            committed=list(winner['steps'])
            break
    return committed,dict(policy='phrase_v1',candidate_count=count,stop_reason='audio_end',phrases=records,
        repairs_used=repairs_used,rollbacks_used=rollbacks_used,search_totals={**aggregate,**native_metrics},
        native_metric_scope='candidate_gate_and_edit_counterfactuals_excludes_proposal_guard',
        capabilities=dict(shared_readonly_audio_cache=True,independent_prefix_guard_curve_rng=True,
            complete_equal_span_selection=True,full_state_replay=True,sustain_withdrawal=True,retained_winner_tail=True,
            optional_audio_continuity_default=0,quality_verified=False))
