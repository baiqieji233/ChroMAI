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


def settings(args):
    mode = getattr(args, 'phrase_search', 'event')
    look = float(getattr(args, 'phrase_beats', 4.))
    commit = float(getattr(args, 'phrase_commit_beats', 2.))
    if mode not in ('event', 'layout', 'joint') or not 1 <= look <= 16 or not 0 < commit <= look:
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


def rank(prefix, points, song, tempo, start, stop, *, motion_cache=None):
    from maichart.composer.stream_quality import _trajectory_comfort
    comfort, coverage = _trajectory_comfort([*prefix, *points], tempo, song.clock,
        song.clock.at(start), song.clock.at(stop), song.audio.duration_sec,
        motion_cache=motion_cache)
    timing = timing_evidence(points, song, start, stop)
    components = dict(comfort=float(comfort), timing_support=timing,
                      comfort_coverage=coverage)
    return .75 * float(comfort) + .25 * timing, components


def trial_search(sampler, target, prefix, guard, primary, *, args, song, tempo,
                 end, count, make_sampler, check_window):
    """Run speculative branches while preserving the caller's retry ownership."""
    names=('_retry_scope','_retry_base','_retry_tree','_retry_condition','_retry_cache_dirty',
           'proposal_scope','_free_choices')
    saved={name:deepcopy(getattr(sampler,name)) for name in names if hasattr(sampler,name)}
    probability=getattr(sampler,'last_probability_event',None)
    try:
        return _trial_search(sampler,target,prefix,guard,primary,args=args,song=song,
            tempo=tempo,end=end,count=count,make_sampler=make_sampler,
            check_window=check_window,retry_snapshot=saved)
    finally:
        sampler._rewind(target)
        for name in names:
            if name in saved:setattr(sampler,name,saved[name])
            elif hasattr(sampler,name):delattr(sampler,name)
        # Rewind materializes the original prefix; no divergent KV suffix survives.
        sampler._retry_cache_dirty=False
        if hasattr(sampler,'last_probability_event'):sampler.last_probability_event=probability


def _trial_search(sampler, target, prefix, guard, primary, *, args, song, tempo,
                  end, count, make_sampler, check_window,retry_snapshot):
    """Share immutable audio; rewind symbolic cache before every divergent trial.

    Layout trials use the baseline's complete timing/cardinality skeleton. Joint
    trials sample their own timing. A failure in the speculative suffix never
    blacklists the common first event. Incomplete trials cannot win by stopping.
    """
    mode, look, commit = settings(args)
    after = prefix[-1].beat if prefix else Fraction(-1)
    start = max(Fraction(0), after)
    stop = min(end, start + look)
    attempts = min(4, max(1, int(getattr(args, 'window_retries', 16))))
    skeleton = []
    trials = []
    began = time.perf_counter()
    motion_cache = OrderedDict()
    primary_probability=getattr(sampler,'last_probability_event',None)
    record=bool(getattr(args,'record_candidate_probabilities',False))
    for branch in range(count):
        sampler._rewind(target)
        for name,value in retry_snapshot.items():setattr(sampler,name,deepcopy(value))
        active = sampler
        current = list(target)
        rows = []
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
                    fixed = skeleton[step] if mode == 'layout' and branch and step < len(skeleton) else None
                    if mode == 'layout' and branch and step >= len(skeleton):
                        complete = bool(trials[0]['complete'])
                        ended = bool(trials[0]['ended'])
                        break
                    if branch == 0 and step == 0 and retry == 0:
                        point, ids = primary
                        probability=primary_probability
                    else:
                        ids = active.propose(current, cursor, end, fixed=fixed)
                        probability=getattr(active,'last_probability_event',None)
                        if ids == [active.codec.eos_id]:
                            termination_probability=probability
                            complete = True
                            ended = True
                            reason = None
                            break
                        point = active.codec.decode_points(ids)[0]
                    details = []
                    checked, reason = check_window(trial_guard, [point], details)
                    if checked is not None:
                        trial_guard = checked
                        reason = None
                        break
                    # Only reject this branch's actual event prefix.
                    if hasattr(active, 'reject') and not (branch == 0 and step == 0):
                        active.reject(ids)
                    point = None
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
        # A layout baseline that failed cannot define a truncated winning skeleton.
        if mode == 'layout' and branch and not trials[0]['complete']:
            complete = False
        score, components = rank(prefix, [r[0] for r in rows], song, tempo, start, stop,
                                 motion_cache=motion_cache) if rows else (.5, {})
        trials.append(dict(rows=rows, complete=complete, ended=ended, reason=reason,
                           score=score, components=components))
        if record:
            from .probability_audit import summarize,span_diagnostics
            inside=[log for (point,_),log in zip(rows,probabilities) if start<=point.beat<stop]
            outside=[log for (point,_),log in zip(rows,probabilities) if not start<=point.beat<stop]
            excluded=sum(len(ids) for point,ids in rows if not start<=point.beat<stop)
            trials[-1].update(probability=summarize(inside,boundary_tokens=excluded),
                boundary_probability=summarize(outside),
                termination_probability=summarize([termination_probability]) if ended else None,
                diagnostics=span_diagnostics([p for p,_ in rows],sampler.codec,start,stop))
    eligible = [i for i, trial in enumerate(trials) if trial['complete'] and trial['rows']]
    selected = max(eligible, key=lambda i: trials[i]['score']) if eligible else None
    if selected is not None and 0 in eligible and trials[selected]['score'] < trials[0]['score'] + .02:
        selected = 0  # Small heuristic differences are not reliable improvements.
    queued = []
    if selected is not None:
        rows = trials[selected]['rows']
        for row in rows:
            if queued and row[0].beat > start + commit:
                break
            queued.append(row)
    evidence = dict(start_beat=str(start), end_beat=str(stop),
                    start_sec=float(song.clock.at(start)), end_sec=float(song.clock.at(stop)),
                    selected=selected,mode=mode,partial_preview=True,preview_files_exported=False,
                    elapsed_sec=time.perf_counter()-began,
                    candidates=[dict(score=t['score'], components=t['components'],
                        events=len(t['rows']), complete=t['complete'], ended=t['ended'],
                        eligible=bool(t['complete'] and t['rows']),
                        reason=t['reason'],**{k:t[k] for k in ('probability','boundary_probability','termination_probability','diagnostics') if k in t}) for t in trials])
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
