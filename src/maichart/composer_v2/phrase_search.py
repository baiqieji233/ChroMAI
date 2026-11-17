"""Receding-horizon trials over complete events, never musical output blocks.

Ranking is a conservative heuristic. It does not reward note count, penalize
rests/repetition, or force a rhythmic family. Native checks remain authoritative.
"""
from copy import deepcopy
from collections import OrderedDict
from fractions import Fraction
from pathlib import Path
import math
import time
import hashlib
import json
import uuid
import numpy as np


RANK_POLICIES = ('comfort_v1', 'comfort_v2', 'likelihood_veto_v1')
PHRASE_MODES = ('event', 'layout', 'joint', 'hybrid')
COMFORT_VETO_FLOOR = 0.22


class _FixedBeat:
    """Fix an event's exact beat while leaving note/chord cardinality free."""
    __slots__ = ('beat', 'note_text', 'beat_only')
    def __init__(self, beat, note_text='1'):
        self.beat = beat
        self.note_text = note_text
        self.beat_only = True


class CheckedEvent(tuple):
    """A normal event pair bound to its exact already-checked guard transition."""
    def __new__(cls, point, ids, *, expected_guard, checked_guard):
        value = super().__new__(cls, (point, ids))
        value.expected_guard = expected_guard
        value.checked_guard = checked_guard
        value._point_key = (point.beat, point.note_text)
        value._token_ids = tuple(ids)
        return value


def checked_event_guard(candidate, guard):
    """Reuse a transition only while its input guard and bound event are intact."""
    if not isinstance(candidate, CheckedEvent) or candidate.expected_guard is not guard:
        return None
    point, ids = candidate
    if candidate._point_key != (point.beat, point.note_text) or candidate._token_ids != tuple(ids):
        return None
    return candidate.checked_guard


def rank_policy(args):
    policy = getattr(args, 'phrase_rank_policy', 'comfort_v1') or 'comfort_v1'
    if policy not in RANK_POLICIES:
        raise ValueError('Invalid phrase rank policy')
    return policy


def settings(args):
    mode = getattr(args, 'phrase_search', 'event')
    look = float(getattr(args, 'phrase_beats', 4.))
    commit = float(getattr(args, 'phrase_commit_beats', 2.))
    rank_policy(args)
    if mode not in PHRASE_MODES or not 1 <= look <= 16 or not 0 < commit <= look:
        raise ValueError('Invalid phrase search mode or beat budget')
    return mode, Fraction(str(look)), Fraction(str(commit))


def timing_evidence(points, song, start, stop):
    """Penalize unsupported selected attacks; silence and weak evidence neutral.

    Compare matched local neighborhoods, not the number of captured waveform
    peaks. This cannot decide whether an audible ornament should be charted.
    """
    points = [p for p in points if start <= p.beat < stop]
    if not points:
        return .5
    seconds = song.clock.at_floats(np.asarray([float(p.beat) for p in points]))
    offsets = np.array([-.10, -.06, -.03, -.01, 0., .01, .03, .06, .10])
    cached = getattr(song, '_phrase_flux', None)
    if cached is None:
        cached = (song.audio.mel_times_sec, np.maximum(np.asarray(song.audio.mel[:, 96], dtype=np.float32), 0.).copy())
        song._phrase_flux = cached
    flux = np.interp((seconds[:, None] + offsets).ravel(), *cached).reshape(-1, len(offsets))
    local = flux.max(1)
    center = flux[:, 3:6].max(1)
    scale = float(np.median(local[local > 0])) if np.any(local > 0) else 0.
    if scale <= 1e-8:
        return .5
    confidence = np.minimum(local / max(scale, 1e-8), 1.)
    penalty = ((local-center) / np.maximum(local, 1e-8)) * confidence
    return float(.5 - .5 * penalty.mean())


def _grammar_confidence(rows, probabilities, start, stop, *, score_scope='configuration'):
    if not rows or not probabilities:
        return .5, None, 0, 0
    logs = [log for (point, _), log in zip(rows, probabilities)
            if start <= point.beat < stop and log is not None]
    if not logs:
        return .5, None, 0, 0
    wanted_role = 'configuration' if score_scope == 'configuration' else None
    selected_rows = []
    deterministic_tokens = 0
    for event in logs:
        choice_counts = getattr(event, 'choice_counts', None)
        if choice_counts is None:
            choice_counts = [2] * len(event.rows)
        for row, role, choices in zip(event.rows, event.roles, choice_counts):
            if wanted_role is not None and role != wanted_role:
                continue
            if int(choices) <= 1:
                deterministic_tokens += 1
                continue
            selected_rows.append(row)
    if not selected_rows:
        return .5, None, 0, deterministic_tokens
    import torch
    transferred = torch.stack(selected_rows)[:, 1].detach().double().cpu().tolist()
    values = [float(val) for val in transferred if math.isfinite(val)]
    if not values:
        return .5, None, 0, deterministic_tokens
    mean_logprob = sum(values) / len(values)
    return float(math.exp(max(mean_logprob, -20.))), float(mean_logprob), len(values), deterministic_tokens


def rank(prefix, points, song, tempo, start, stop, *, motion_cache=None,
         policy='comfort_v1', score_scope='configuration', rows=None, probabilities=None):
    if policy not in RANK_POLICIES:
        raise ValueError('Invalid phrase rank policy')
    if policy == 'comfort_v2':
        from .hand_flow import chirality_comfort
        comfort, coverage = chirality_comfort([*prefix, *points], tempo, song.clock,
            song.clock.at(start), song.clock.at(stop), song.audio.duration_sec,
            motion_cache=motion_cache)
    else:
        from maichart.composer.stream_quality import _trajectory_comfort
        comfort, coverage = _trajectory_comfort([*prefix, *points], tempo, song.clock,
            song.clock.at(start), song.clock.at(stop), song.audio.duration_sec,
            motion_cache=motion_cache)
    timing = timing_evidence(points, song, start, stop)
    components = dict(comfort=float(comfort), timing_support=timing,
                      comfort_coverage=coverage)
    if policy == 'comfort_v1':
        return .75 * float(comfort) + .25 * timing, components
    model_conf, mean_lp, scored_tokens, deterministic_tokens = _grammar_confidence(
        rows if rows is not None else [(p, ()) for p in points],
        probabilities, start, stop, score_scope=score_scope)
    known_fraction = float((coverage or {}).get('known_fraction', 1.))
    comfort_veto = bool(known_fraction > 0. and float(comfort) < COMFORT_VETO_FLOOR)
    components.update(model_confidence=model_conf, mean_grammar_logprob=mean_lp,
                      scored_grammar_tokens=scored_tokens,
                      deterministic_grammar_tokens=deterministic_tokens,
                      score_scope=score_scope,
                      comfort_veto=comfort_veto,
                      comfort_veto_floor=COMFORT_VETO_FLOOR, ranking_policy=policy)
    if comfort_veto:
        return -1. + float(comfort), components
    if policy == 'comfort_v2':
        if scored_tokens > 0:
            return .55 * float(comfort) + .25 * model_conf + .20 * timing, components
        return .75 * float(comfort) + .25 * timing, components
    return .80 * model_conf + .20 * timing, components


def trial_search(sampler, target, prefix, guard, primary, *, args, song, tempo,
                 end, count, make_sampler, check_window, after_floor=None):
    """Run speculative branches while preserving the caller's retry ownership."""
    names=('_retry_scope','_retry_base','_retry_tree','_retry_condition','_retry_cache_dirty',
           '_fixed_retry_scope','_fixed_retry_tree','_retry_offset',
           'proposal_scope','_free_choices','_active_guard')
    saved={name:deepcopy(getattr(sampler,name)) for name in names if hasattr(sampler,name)}
    probability=getattr(sampler,'last_probability_event',None)
    try:
        return _trial_search(sampler,target,prefix,guard,primary,args=args,song=song,
            tempo=tempo,end=end,count=count,make_sampler=make_sampler,
            check_window=check_window,retry_snapshot=saved,after_floor=after_floor)
    finally:
        sampler._rewind(target)
        for name in names:
            if name in saved:setattr(sampler,name,saved[name])
            elif hasattr(sampler,name):delattr(sampler,name)
        # Rewind materializes the original prefix; no divergent KV suffix survives.
        sampler._retry_cache_dirty=False
        if hasattr(sampler,'last_probability_event'):sampler.last_probability_event=probability


def _trial_search(sampler, target, prefix, guard, primary, *, args, song, tempo,
                  end, count, make_sampler, check_window,retry_snapshot,after_floor=None):
    """Share immutable audio; rewind symbolic cache before every divergent trial.

    Layout trials use the baseline's complete timing/cardinality skeleton. Joint
    trials sample their own timing. A failure in the speculative suffix never
    blacklists the common first event. Trials must reach the comparison boundary
    to be ranked. A predicted chart end is resolved on the committed prefix.
    """
    from .stream_generate import GridDurationRejected

    mode, look, commit = settings(args)
    policy = rank_policy(args)
    # Every branch in the pool uses the same grammar token roles. When
    # timing is shared, variation comes from configuration conditioned on it.
    score_scope = 'configuration' if mode == 'layout' or (mode == 'hybrid' and count <= 2) else 'joint'
    after = prefix[-1].beat if prefix else Fraction(-1)
    if after_floor is not None:
        after = max(after, after_floor)
    start = max(Fraction(0), after)
    stop = min(end, start + look)
    attempts = min(4, max(1, int(getattr(args, 'window_retries', 16))))
    skeleton = []
    trials = []
    began = time.perf_counter()
    motion_cache = OrderedDict()
    primary_probability=getattr(sampler,'last_probability_event',None)
    record=bool(getattr(args,'record_candidate_probabilities',False))
    work = dict(proposal_calls=0, native_window_checks=0, rank_calls=0,
                proposal_sec=0., native_window_sec=0., rank_sec=0.)
    if record:
        from .probability_audit import summarize,span_diagnostics
    for branch in range(count):
        use_skeleton = (mode == 'layout' and branch > 0) or (
            mode == 'hybrid' and branch > 0 and (count <= 2 or branch < count - 1)
        )
        if use_skeleton and not trials[0]['complete']:
            # An incomplete skeleton cannot produce an eligible layout/hybrid trial.
            # Skip its sampling work explicitly; skipped branches consume no RNG.
            skipped = dict(rows=[], guard_chain=[], complete=False, ended=False,
                reason='layout_baseline_incomplete', baseline_reason=trials[0]['reason'],
                score=.5, components={}, skipped=True, scored=False,
                unscored_reason='skipped_layout_baseline_incomplete')
            if record:
                skipped.update(probability=summarize([]), boundary_probability=summarize([]),
                    termination_probability=None,
                    diagnostics=span_diagnostics([],sampler.codec,start,stop))
            trials.append(skipped)
            continue
        sampler._rewind(target)
        for name,value in retry_snapshot.items():setattr(sampler,name,deepcopy(value))
        active = sampler
        current = list(target)
        rows = []
        guard_chain = []
        probabilities=[];termination_probability=None
        trial_guard = deepcopy(guard)
        cursor = after
        complete = False
        reason = None
        ended = False
        # A compute bound, not a note-density rule. Unfinished trials are reported.
        for step in range(128):
            point = None
            ids = None
            for retry in range(attempts):
                try:
                    if use_skeleton and step < len(skeleton):
                        fixed = _FixedBeat(skeleton[step].beat, skeleton[step].note_text) if mode == 'hybrid' else skeleton[step]
                    else:
                        fixed = None
                    if use_skeleton and step >= len(skeleton):
                        complete = bool(trials[0]['complete'])
                        break
                    if branch == 0 and step == 0 and retry == 0:
                        point, ids = primary
                        probability=primary_probability
                    else:
                        work['proposal_calls'] += 1
                        call_began = time.perf_counter()
                        if hasattr(trial_guard, 'tempo') and hasattr(trial_guard, 'prefix'):
                            active._active_guard = trial_guard
                        try:
                            ids = active.propose(current, cursor, end, fixed=fixed)
                        finally:
                            work['proposal_sec'] += time.perf_counter() - call_began
                        probability=getattr(active,'last_probability_event',None)
                        if ids == [active.codec.eos_id]:
                            # EOS predicts the end of the whole chart on a
                            # speculative prefix. It does not cover this span.
                            termination_probability=probability
                            ended = True
                            reason = 'speculative_eos_before_boundary'
                            break
                        point = active.codec.decode_points(ids)[0]
                    details = []
                    expected_guard = trial_guard if rows else guard
                    work['native_window_checks'] += 1
                    call_began = time.perf_counter()
                    try:
                        checked, reason = check_window(trial_guard, [point], details)
                    finally:
                        work['native_window_sec'] += time.perf_counter() - call_began
                    if checked is not None:
                        # _check_window returns a separate transaction each time.
                        # Keep every step, not just the speculative horizon state.
                        guard_chain.append((expected_guard, checked))
                        trial_guard = checked
                        reason = None
                        break
                    # Only reject this branch's actual event prefix.
                    if hasattr(active, 'reject') and not (branch == 0 and step == 0):
                        active.reject(ids)
                    point = None
                except GridDurationRejected as error:
                    reason = str(error)
                    point = None
                    continue
                except ValueError as error:
                    # Refresh is a computational event; preserve actual trial history.
                    if error.__class__.__name__ == 'RefreshContext' and rows:
                        active = make_sampler([*prefix, *[r[0] for r in rows]])
                        current = []
                        continue
                    reason = str(error)
                    point = None
                    break
            if ended or complete:
                break
            if point is None:
                break
            rows.append((point, ids))
            probabilities.append(probability)
            if branch == 0:
                skeleton.append(point)
            current.extend(ids)
            cursor = point.beat
            if point.beat >= stop:
                complete = True
                break
            if cursor >= getattr(active, 'reference_refresh_beat', end) or len(current) >= active.max_tokens // 2 - 96:
                active = make_sampler([*prefix, *[r[0] for r in rows]])
                current = []
        comparison_events = sum(start <= point.beat < stop for point, _ in rows)
        scored = bool(complete and comparison_events)
        if not complete:
            unscored_reason = 'incomplete_candidate'
        elif not rows:
            unscored_reason = 'empty_candidate'
        elif not comparison_events:
            unscored_reason = 'no_onsets_in_comparison_span'
        else:
            unscored_reason = None
        score, components = .5, {}
        if scored:
            work['rank_calls'] += 1
            call_began = time.perf_counter()
            score, components = rank(prefix, [r[0] for r in rows], song, tempo, start, stop,
                                     motion_cache=motion_cache, policy=policy, score_scope=score_scope,
                                     rows=rows, probabilities=probabilities)
            work['rank_sec'] += time.perf_counter() - call_began
        trials.append(dict(rows=rows, guard_chain=guard_chain, complete=complete, ended=ended, reason=reason,
                           score=score, components=components, skipped=False,
                           scored=scored, unscored_reason=unscored_reason,
                           comparison_events=comparison_events))
        if record:
            inside=[log for (point,_),log in zip(rows,probabilities) if start<=point.beat<stop]
            outside=[log for (point,_),log in zip(rows,probabilities) if not start<=point.beat<stop]
            excluded=sum(len(ids) for point,ids in rows if not start<=point.beat<stop)
            trials[-1].update(probability=summarize(inside,boundary_tokens=excluded),
                boundary_probability=summarize(outside),
                termination_probability=summarize([termination_probability]) if ended else None,
                diagnostics=span_diagnostics([p for p,_ in rows],sampler.codec,start,stop))
    # A crossing event only proves coverage of the elapsed interval. Its
    # configuration and likelihood are outside the scored range. Retain a
    # valid primary rest without using missing motion as a quality advantage
    # or replacing that rest with an alternative solely because it has notes.
    preserve_primary_rest = bool(trials and trials[0]['complete'] and trials[0]['rows']
                                and trials[0].get('comparison_events') == 0)
    eligible = [i for i, trial in enumerate(trials)
                if trial['scored'] or (i == 0 and preserve_primary_rest)]
    if preserve_primary_rest:
        selected = 0
    elif eligible:
        selected = max(eligible, key=lambda i: trials[i]['score'])
    else:
        selected = None
    margin = .01 if policy == 'likelihood_veto_v1' else .02
    baseline_vetoed = bool(trials[0]['components'].get('comfort_veto', False)) if trials else False
    if selected is not None and 0 in eligible and not baseline_vetoed and trials[selected]['score'] < trials[0]['score'] + margin:
        selected = 0  # Small heuristic differences are not reliable improvements.
    queued = []
    if selected is not None:
        rows = trials[selected]['rows']
        for row, (expected_guard, checked_guard) in zip(rows, trials[selected]['guard_chain']):
            if queued and row[0].beat > start + commit:
                break
            queued.append(CheckedEvent(*row, expected_guard=expected_guard, checked_guard=checked_guard))
    all_vetoed = bool(
        policy == 'likelihood_veto_v1'
        and eligible
        and all(trials[i]['components'].get('comfort_veto', False) for i in eligible)
    )
    evidence = dict(start_beat=str(start), end_beat=str(stop),
                    start_sec=float(song.clock.at(start)), end_sec=float(song.clock.at(stop)),
                    selected=selected,mode=mode,ranking_policy=policy,
                    selection_basis=('preserve_primary_rest' if preserve_primary_rest
                                     else 'ranked_trials' if selected is not None else 'event_fallback'),
                    all_comfort_vetoed=all_vetoed if policy == 'likelihood_veto_v1' else False,
                    partial_preview=True,preview_files_exported=False,
                    elapsed_sec=time.perf_counter()-began,
                    requested_candidates=count,
                    evaluated_candidates=sum(not t['skipped'] for t in trials),
                    skipped_candidates=sum(t['skipped'] for t in trials),
                    computation_policy='skip_ineligible_layout_branches_v1',
                    skipped_branch_rng_policy='no_sampling_draws',
                    compute=dict(**work, speculative_events=sum(len(t['rows']) for t in trials)),
                    candidates=[dict(score=t['score'], components=t['components'],
                        events=len(t['rows']), complete=t['complete'], ended=t['ended'],
                        eligible=i in eligible,comparison_events=t.get('comparison_events',0),
                        reason=t['reason'],skipped=t['skipped'],scored=t['scored'],
                        unscored_reason=t['unscored_reason'],
                        **{k:t[k] for k in ('baseline_reason','probability','boundary_probability','termination_probability','diagnostics') if k in t}) for i,t in enumerate(trials)])
    return queued, evidence, trials


def attach_candidate_content(evidence, prefix, trials, *, tempo=None, first=None):
    """Keep audit candidates immutable without compiling whole-prefix previews.

    Called only when recording is enabled. The exact committed-prefix identity
    distinguishes candidates produced after rollback or duration repair; trial
    deltas include their crossing-boundary event for faithful inspection.
    """
    def events(points):
        return [dict(beat=str(point.beat), note_text=point.note_text) for point in points]
    prefix_events=events(prefix)
    encoded=json.dumps(prefix_events,ensure_ascii=False,separators=(',',':')).encode('utf-8')
    evidence['candidate_content_schema']='composer-v2-candidate-events-v1'
    evidence['common_prefix']=dict(sha256=hashlib.sha256(encoded).hexdigest(),
        events=len(prefix_events),last_beat=prefix_events[-1]['beat'] if prefix_events else None,
        identity='ordered_exact_beats_and_note_text',source='committed_generation_prefix')
    evidence['comparison_range']=dict(start_beat=evidence['start_beat'],end_beat=evidence['end_beat'],
        bounds='start_inclusive_end_exclusive',boundary_event='retained_in_event_delta_excluded_from_probability')
    if tempo is not None:
        evidence['time_axis']=dict(first=float(first),
            tempo_map=[dict(beat=str(change.beat),bpm=float(change.bpm)) for change in tempo])
    for candidate,trial in zip(evidence['candidates'],trials):
        candidate.pop('path',None)
        candidate['event_delta']=events(point for point,_ in trial['rows'])
    return evidence


def write_previews(output, number, prefix, trials, evidence, song, tempo, level):
    """Explicit diagnostic export; ordinary generation never calls this helper.

    Unique names preserve old references if a caller chooses to export clips.
    """
    from maichart.generation.event_compile import compile_event_points
    directory = Path(output).with_suffix('') / 'phrase_candidates'
    directory.mkdir(parents=True, exist_ok=True)
    snapshot=uuid.uuid4().hex[:12]
    for index, trial in enumerate(trials):
        if not trial['rows'] or not trial['complete']:
            continue
        points = [*prefix, *[r[0] for r in trial['rows']]]
        finish = max(Fraction(evidence['end_beat']), points[-1].beat + Fraction(1, 48))
        text, _ = compile_event_points(points, tempo_changes=[t for t in tempo if t.beat <= finish],
            bpm=tempo[0].bpm, first=song.clock.seconds[0], end_beat=finish,
            title=f'{song.song_id} 候选片段 {number+1}-{index+1}', level=str(level),
            designer='Composer v2 partial trial', strict_notes=False)
        path = directory / f'trial_{number+1:06d}_{index}_{snapshot}.txt'
        with path.open('x',encoding='utf-8') as handle:handle.write(text)
        evidence['candidates'][index]['path'] = str(path.resolve())
        evidence['preview_files_exported'] = True
