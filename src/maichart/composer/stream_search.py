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
from .events import event_metadata


class PhraseSearchExhausted(ValueError):
    """Fail closed while retaining reproducible private-search evidence."""
    def __init__(self,message,diagnostics):
        super().__init__(message)
        self.diagnostics=diagnostics


def _gate_evidence(result):
    gate=result.get('_physical_gate')
    return (dict(parse_errors=gate[0].coverage.get('parse_errors',[]),
        enforced_conflicts=[dict(kind=issue.conflict_type,time=issue.time,evidence=issue.evidence,
            notes=[dict(beat=str(n.beat),raw=n.raw) for n in issue.notes]) for issue in gate[1]]) if gate else None)


def _failure_evidence(result):
    guard=result['guard']
    def point(p):return dict(beat=str(p.beat),text=p.note_text)
    return dict(reason=result['reason'],cursor=str(result['cursor']),
        candidate_gate=_gate_evidence(result),
        anchor_second=result.get('anchor_second'),retries=result.get('retries'),
        timing_retries=result.get('timing_retries'),forced_waits=result.get('forced_waits'),
        seed=result.get('seed'),guard_last_reason=getattr(guard,'last_reason',None),
        guard_reasons=dict(guard.reason_counts),
        last_proposal=point(guard.last_proposal) if getattr(guard,'last_proposal',None) is not None else None,
        failure_trials=[dict(point=point(p),conflicts=[dict(kind=x.conflict_type,time=x.time,
            evidence=x.evidence,notes=[dict(beat=str(n.beat),raw=n.raw) for n in x.notes]) for x in issues])
            for p,issues in getattr(guard,'failure_trials',())],
        steps=[dict(kind=s.kind,start=str(s.start),end=str(s.end),start_second=s.start_second,
            end_second=s.end_second,text=s.text) for s in result['steps']])


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


def _step_fingerprint(steps):
    return tuple((step.kind, str(step.start), str(step.end), step.text) for step in steps)


def _advances(steps,start):
    """A rewritten history may have fewer steps yet reach a later beat."""
    return bool(steps and steps[-1].end>start)


def _committed_history_blocker(failures,committed,song):
    """Return native evidence only when every named atom is a committed sustain."""
    if not failures or not committed:return None
    events,_,_=_extract(_points(committed),song.tempo,song.first)
    sustained={(str(event.beat),event.raw):event.note_type for event in events
               if event.note_type in {'hold','touch_hold','slide'} and not event.protected}
    if not sustained:return None
    for failed in failures:
        guard=failed['guard']
        issues=[('failure_conflicts',issue) for issue in getattr(guard,'failure_conflicts',())]
        issues.extend(('failure_trials',issue) for _,trial in getattr(guard,'failure_trials',()) for issue in trial)
        for source,issue in issues:
            named={(str(note.beat),note.raw) for note in issue.notes}
            if named and named<=sustained.keys() and guard.enforced(issue):
                return dict(source=source,conflict_type=issue.conflict_type,
                    named_committed_atoms=[dict(beat=beat,raw=raw,note_type=sustained[(beat,raw)])
                                           for beat,raw in sorted(named)],
                    native_evidence=getattr(issue,'evidence',None))
    return None


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
    rows=[]
    for point in result['points']:
        key=(point.beat,point.note_text)
        if key not in lifetimes:
            lifetimes[key]=guard._lifetime(point)
            if len(lifetimes)>4096:lifetimes.pop(next(iter(lifetimes)))
        second,finish,sustained,_=lifetimes[key]
        rows.append((point,second,finish,sustained))
    # Active sustains need their intervening interaction history, not just the
    # sustain itself and taps near the newest onset. Share the guard's window.
    cutoff=guard.interaction_start(anchor,rows)
    relevant=[point for point,second,finish,_ in rows if second>=cutoff or finish>=cutoff]
    physical=_judge(relevant,song,guard.config,metrics)
    result['_physical_gate']=(physical,[issue for issue in physical.conflicts_detected if guard.enforced(issue)])
    return result['_physical_gate']


def _duration_signature(raw):
    return re.sub(r'\[[^\]]+\]', '[]', raw)


def _repair_prefix(result,song,clock,metrics,lifetimes,withdraw=False,codec=None,duration_catalog=None,deadline=None,committed_steps=None):
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
        # Only duration is editable here. Retain paths, endpoints and modifiers
        # so a different same-head slide cannot masquerade as an old finding.
        return {(x.conflict_type,round(x.time,6),tuple(sorted((str(n.beat),str(n.note_type),str(n.button),_duration_signature(n.raw)) for n in x.notes)))
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
        if committed_steps is not None:
            # Global history budgets protect precisely the authoritative log;
            # private branch notes are handled by bounded phrase-local retries.
            if step_index>=len(committed_steps) or steps[step_index]!=committed_steps[step_index]:continue
        if withdraw:
            # Withdraw exactly the implicated child. Same-onset protected and
            # unrelated children remain authoritative while the suffix is rebuilt.
            remaining='/'.join(raw for key,raw in texts.items() if key[0]==point_index and key!=event.source_id)
            counter=[EventPoint(p.beat,remaining) if i==point_index else p
                     for i,p in enumerate(trial_points) if i!=point_index or remaining]
            if improves(counter):
                prefix=list(steps[:step_index])
                if remaining:
                    old=steps[step_index]
                    prefix.append(StreamStep(NOTE,old.start,old.end,old.start_second,old.end_second,remaining))
                yield prefix,dict(operation='withdraw_sustain_and_regenerate',beat=str(point.beat),raw=event.raw)
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
    current_retries_used=0;private_tail_retries_used=0
    native_metrics=dict(native_calls=0,native_sec=0.);lifetimes={}
    duration_catalog={}
    for atom in codec.atom_catalog if codec is not None else ():
        if '[' in atom:duration_catalog.setdefault(_duration_signature(atom),[]).append(atom)
    aggregate=dict(rollouts=0,configuration_retries=0,rejected_onset_attempts=0,guard_checks=0,guard_rejections=0,forced_waits=0,guard_reasons={})
    while (committed[-1].end if committed else Fraction(0))<end:
        start=committed[-1].end if committed else Fraction(0)
        target=min(end,start+Fraction(str(write)));commit_end=min(end,start+Fraction(str(commit)))
        deadline=time.perf_counter()+seconds;valid=[];failures=[];candidate_errors=[];candidate_logs=[];attempt=0
        def run(prefix,origin='primary'):
            nonlocal attempt
            seed=(int(args.seed)+phrase_index*1000003+attempt*9176)%(2**63-1);attempt+=1
            log=dict(candidate_id=attempt-1,seed=seed,origin=origin,eligible=False,
                prefix_end_beat=str(prefix[-1].end if prefix else Fraction(0)),
                authoritative_prefix_steps=len(committed),
                retained_sampled_suffix_steps=(len(prefix)-len(committed)
                    if origin=='primary_retained_tail' else 0),
                score=None,event_count=None,head_count=None,model_diagnostics=None)
            candidate_logs.append(log)
            aggregate['rollouts']+=1
            try:
                result=rollout(prefix,seed,target,deadline)
                result['seed']=seed
                result['candidate_log_id']=log['candidate_id']
                log['model_diagnostics']=result.get('model_diagnostics')
                log['rollout_stop_reason']=result['reason']
                interval=[point for point in result['points'] if start<point.beat<=target or start==0 and point.beat==0]
                log['event_count']=len(interval)
                log['head_count']=sum(len(event_metadata(point.note_text).heads) for point in interval)
                physical,hard=_gate(result,song,native_metrics,lifetimes)
                log['physical_gate']=_gate_evidence(result)
            except ValueError as exc:
                candidate_errors.append(dict(seed=seed,error=str(exc)))
                log.update(failure_reason='rollout_value_error',error=str(exc))
                return
            aggregate['configuration_retries']+=result['retries']
            aggregate['rejected_onset_attempts']+=result['timing_retries']
            aggregate['guard_checks']+=result['guard'].checks;aggregate['guard_rejections']+=result['guard'].rejections
            aggregate['forced_waits']+=result['forced_waits']
            for key,value in result['guard'].reason_counts.items():
                aggregate['guard_reasons'][key]=aggregate['guard_reasons'].get(key,0)+value
            complete=result['cursor']>=target and result['reason'] not in {'physical_dead_end','max_events','step_budget','time_budget'}
            if not complete or hard or physical.coverage.get('parse_errors'):
                log['failure_reason']=('incomplete:'+result['reason'] if not complete else
                    'parse_errors' if physical.coverage.get('parse_errors') else 'enforced_native_hard_conflict')
                failures.append(result);return
            terminal=result['cursor']>=end and result['reason'] in {'learned_eos','tail_no_legal_onset'}
            prior_steps=sum(step.end<=start for step in result['steps'])
            proposed=_cut(result['steps'],commit_end,clock,prior_steps)
            if not _advances(proposed,start) and not terminal:
                log['failure_reason']='no_committed_progress';failures.append(result);return
            # Score the full observed write span, including its private tail.
            # If the first committed step overshoots the write target, charge
            # that actual step to this evaluation instead of omitting it.
            score_end=max(target,proposed[-1].end if proposed else target)
            score=score_phrase(result['points'],_points(committed),song,condition,start,score_end,
                               softcomfort_strength=strength,scorer=scorer,
                               observed_end_second=second_at_beat(clock,result['cursor']))
            log['score']=score
            log['score_start_beat']=str(start)
            log['score_end_beat']=str(score_end)
            log['observed_end_beat']=str(result['cursor'])
            log['proposed_commit_end_beat']=str(proposed[-1].end if proposed else start)
            interval=[point for point in result['points'] if start<point.beat<=score_end or start==0 and point.beat==0]
            log['event_count']=len(interval)
            log['head_count']=sum(len(event_metadata(point.note_text).heads) for point in interval)
            if not math.isfinite(float(score['total'])):
                log['failure_reason']='nonfinite_quality_score';failures.append(result);return
            log.update(eligible=True,failure_reason=None)
            valid.append((float(score['total']),result,score))
        # All candidates share the authoritative committed prefix. Candidate
        # zero may also preserve the prior winner's real, already sampled tail;
        # the others regenerate that tentative suffix rather than inheriting it.
        for candidate_index in range(count):
            if time.perf_counter()>=deadline:break
            run(list(committed)+(list(retained) if candidate_index==0 else []),
                origin='primary_retained_tail' if candidate_index==0 and retained else 'primary')
        changes=[];current_event_retries=[];private_tail_retries=[]
        history_blocker=_committed_history_blocker(failures,committed,song) if not valid else None
        history_priority=bool(history_blocker)
        attempted_historical_edits=set()
        def try_historical_repair():
            nonlocal repairs_used,rollbacks_used
            if not valid and repairs_used<repair_budget:
                for failed in list(failures):
                    if valid or time.perf_counter()>=deadline:break
                    for prefix,edit in _repair_prefix(failed,song,clock,native_metrics,lifetimes,
                                                    codec=codec,duration_catalog=duration_catalog,deadline=deadline,committed_steps=committed):
                        if repairs_used>=repair_budget or time.perf_counter()>=deadline:break
                        edit_key=(edit['operation'],edit['beat'],edit.get('raw'),edit.get('replacement'),_step_fingerprint(prefix))
                        if edit_key in attempted_historical_edits:continue
                        attempted_historical_edits.add(edit_key)
                        repairs_used+=1;run(prefix,origin='historical_sustain_edit')
                        if valid:changes.append(edit);break
            if not valid and rollbacks_used<rollback_budget:
                for failed in list(failures):
                    if valid or time.perf_counter()>=deadline:break
                    for prefix,edit in _repair_prefix(failed,song,clock,native_metrics,lifetimes,withdraw=True,
                                                    codec=codec,duration_catalog=duration_catalog,deadline=deadline,committed_steps=committed):
                        if rollbacks_used>=rollback_budget or time.perf_counter()>=deadline:break
                        edit_key=(edit['operation'],edit['beat'],edit.get('raw'),edit.get('replacement'),_step_fingerprint(prefix))
                        if edit_key in attempted_historical_edits:continue
                        attempted_historical_edits.add(edit_key)
                        rollbacks_used+=1;run(prefix,origin='historical_sustain_withdrawal')
                        if valid:changes.append(edit);break
        if history_priority:
            try_historical_repair()
        if valid:
            # An overshooting commit expands the common comparison horizon.
            # Never score a shorter rollout as though its unobserved tail were
            # empty; only candidates with real observations to that beat rank.
            evaluation_end=max(Fraction(log['score_end_beat']) for log in candidate_logs if log['eligible'])
            comparable=[]
            for _,result,score in valid:
                log=candidate_logs[result['candidate_log_id']]
                if result['cursor']<evaluation_end:
                    log.update(eligible=False,failure_reason='insufficient_observed_horizon',
                               required_score_end_beat=str(evaluation_end))
                    continue
                if Fraction(log['score_end_beat'])<evaluation_end:
                    score=score_phrase(result['points'],_points(committed),song,condition,start,evaluation_end,
                        softcomfort_strength=strength,scorer=scorer,
                        observed_end_second=second_at_beat(clock,result['cursor']))
                    log['score']=score
                    log['score_end_beat']=str(evaluation_end)
                    interval=[point for point in result['points'] if start<point.beat<=evaluation_end or start==0 and point.beat==0]
                    log['event_count']=len(interval)
                    log['head_count']=sum(len(event_metadata(point.note_text).heads) for point in interval)
                if not math.isfinite(float(score['total'])):
                    log.update(eligible=False,failure_reason='nonfinite_quality_score')
                    continue
                comparable.append((float(score['total']),result,score))
            valid=comparable
        if not valid:
            # Retry only the current unaccepted event from a private accepted
            # prefix. Keep its real timing/type distribution and temperature.
            for retry_index in range(2):
                retry_sources=[failed for failed in failures if failed['reason']=='physical_dead_end']
                if valid or not retry_sources or time.perf_counter()>=deadline:break
                # A retry may have advanced before finding its next rejected
                # event. Retain that private progress instead of repeatedly
                # restarting from the original failure snapshot.
                failed=max(retry_sources,key=lambda item:(item['cursor'],len(item['steps'])))
                current_retries_used+=1
                run(list(failed['steps']),origin='current_event_retry')
                retry_record=dict(operation='current_event_retry',beat=str(failed['cursor']),accepted=bool(valid))
                current_event_retries.append(retry_record)
                if valid:changes.append(retry_record)
        if not valid:
            for _ in range(2):
                private_failures=[failed for failed in failures
                    if failed['steps'][:len(committed)]==committed and len(failed['steps'])>len(committed)]
                if valid or not private_failures or time.perf_counter()>=deadline:break
                private_tail_retries_used+=1
                # The tentative suffix is a candidate, not committed history.
                # Discard it atomically and resample from the common prefix.
                run(list(committed),origin='private_tail_retry')
                retry_record=dict(operation='private_tail_retry',beat=str(start),accepted=bool(valid))
                private_tail_retries.append(retry_record)
                if valid:changes.append(retry_record)
        if not history_priority:
            try_historical_repair()
        if not valid:
            evidence=dict(start_beat=str(start),target_beat=str(target),commit_budget_end_beat=str(commit_end),
                candidate_count=count,attempted_rollouts=attempt,repairs_used=repairs_used,repair_budget=repair_budget,
                rollbacks_used=rollbacks_used,rollback_budget=rollback_budget,
                current_event_retries_used=current_retries_used,current_event_retry_limit_per_phrase=2,
                private_tail_retries_used=private_tail_retries_used,private_tail_retry_limit_per_phrase=2,
                history_priority=history_priority,
                retry_schedule='historical_blocker_first' if history_priority else 'current_event_first',
                history_blocker_evidence=history_blocker,
                time_budget_sec=seconds,deadline_exhausted=time.perf_counter()>=deadline,
                candidate_errors=candidate_errors,candidate_logs=candidate_logs,current_event_retries=current_event_retries,private_tail_retries=private_tail_retries,
                recent_failures=[_failure_evidence(failed) for failed in failures[-8:]],
                committed_steps=[dict(kind=s.kind,start=str(s.start),end=str(s.end),start_second=s.start_second,
                    end_second=s.end_second,text=s.text) for s in committed],
                retained_tail=[dict(kind=s.kind,start=str(s.start),end=str(s.end),start_second=s.start_second,
                    end_second=s.end_second,text=s.text) for s in retained],
                prior_phrases=records,search_totals={**aggregate,**native_metrics})
            raise PhraseSearchExhausted(f'phrase search exhausted at beat {start}: no complete physically feasible candidate; chart not exported',evidence)
        _,winner,score=max(valid,key=lambda item:item[0])
        previous_retained=_step_fingerprint(retained)
        previous_length=sum(step.end<=start for step in winner['steps'])
        committed=_cut(winner['steps'],commit_end,clock,previous_length)
        retained=winner['steps'][len(committed):]
        if previous_retained and _step_fingerprint(winner['steps'][previous_length:])[:len(previous_retained)]!=previous_retained:
            for prior in records:
                if prior.get('tentative_tail_fingerprint')==previous_retained:
                    prior['invalidated_by_suffix_regeneration']=True
        if changes:
            changed_from=min(Fraction(edit['beat']) for edit in changes)
            for prior in records:
                affected_end=max(Fraction(prior['write_end_beat']),Fraction(prior.get('actual_commit_end_beat',prior['commit_end_beat'])))
                if affected_end>=changed_from:
                    prior['invalidated_by_suffix_regeneration']=True
        records.append(dict(start_beat=str(start),write_end_beat=str(target),commit_end_beat=str(commit_end),
            actual_commit_end_beat=str(committed[-1].end if committed else Fraction(0)),
            score_start_beat=str(start),
            score_end_beat=next(log['score_end_beat'] for log in candidate_logs if log['candidate_id']==winner['candidate_log_id']),
            tentative_tail_fingerprint=_step_fingerprint(retained),
            requested_candidates=count,attempted_rollouts=attempt,valid_candidates=len(valid),
            score=score,changes=changes,current_event_retries=current_event_retries,private_tail_retries=private_tail_retries,candidate_errors=candidate_errors,
            history_priority=history_priority,
            retry_schedule='historical_blocker_first' if history_priority else 'current_event_first',
            history_blocker_evidence=history_blocker,
            candidate_logs=candidate_logs,winner_candidate_id=winner['candidate_log_id'],winner_seed=winner['seed'],
            model_diagnostics=winner.get('model_diagnostics'),
            winner_reason='only_physically_feasible_candidate' if len(valid)==1 else 'highest_music_and_trajectory_quality_score',
            score_effect='diagnostic_only_single_candidate' if count==1 else 'rank_physically_feasible_candidates',
            elapsed_sec=seconds-max(0.,deadline-time.perf_counter())))
        phrase_index+=1
        if winner['cursor']>=end and (target==end or winner['reason'] in {'learned_eos','tail_no_legal_onset'}):
            # EOS/end-of-audio is terminal metadata, never a synthetic WAIT.
            committed=list(winner['steps'])
            break
        if not committed or committed[-1].end<=start:
            raise PhraseSearchExhausted(f'phrase search made no progress at beat {start}',
                                        dict(start_beat=str(start),winner_reason=winner['reason'],prior_phrases=records))
    final_score=score_phrase(_points(committed),[],song,condition,Fraction(0),end,
        softcomfort_strength=strength,scorer=scorer,observed_end_second=second_at_beat(clock,end))
    return committed,dict(policy='phrase_v1',candidate_count=count,stop_reason='audio_end',phrases=records,
        final_export_score=final_score,
        repairs_used=repairs_used,rollbacks_used=rollbacks_used,search_totals={**aggregate,**native_metrics},
        current_event_retries_used=current_retries_used,current_event_retry_limit_per_phrase=2,
        private_tail_retries_used=private_tail_retries_used,private_tail_retry_limit_per_phrase=2,
        repair_counter_scope='historical_sustain_edit_attempts_excludes_current_event_retry',
        native_metric_scope='candidate_gate_and_edit_counterfactuals_excludes_proposal_guard',
        capabilities=dict(shared_readonly_audio_cache=True,independent_prefix_guard_curve_rng=True,
            complete_equal_span_selection=True,full_state_replay=True,sustain_withdrawal=True,retained_winner_tail=True,
            optional_audio_continuity_default=0,quality_verified=False))
