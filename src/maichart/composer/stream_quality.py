"""Conservative phrase ranking heuristics, not calibrated human preferences.

Physical feasibility belongs to EventGuard. These scores never reject a
technique, require a note at every acoustic peak, or reward retry counts.
"""
from collections import OrderedDict
from functools import lru_cache
import math

import numpy as np

from maichart.generation.event_quality import parse_event_notes, MIACODE_DURATION_SEMANTICS
from maichart.tempo_clock import TempoClock
from .stream_continuation import typed_event
from .stream_phrase import acoustic_descriptors


def _unit(values):
    values = np.maximum(np.nan_to_num(values,nan=0.,posinf=0.,neginf=0.),0.)
    scale = float(np.percentile(values,95)) if len(values) else 0.
    return np.clip(values/max(scale,1e-8),0.,1.)


def _condition(condition):
    if hasattr(condition,'radar'):
        return float(condition.level),float(condition.radar.get('note_density',.3)),float(condition.radar.get('chord',.25))
    if isinstance(condition,dict):
        radar = condition.get('radar',condition)
        return float(condition.get('level',13.)),float(radar.get('note_density',.3)),float(radar.get('chord',.25))
    values = np.asarray(condition.detach().cpu() if hasattr(condition,'detach') else condition,dtype=np.float64).reshape(-1)
    return (float(values[0])*15 if len(values) else 13.,float(values[1]) if len(values)>1 else .3,
            float(values[7]) if len(values)>7 else .25)


def _lane(note):
    from maichart.generation.muri.model import CENTERS
    value = f'A{note.position}' if isinstance(note.position,int) else str(note.position)
    return tuple(float(x)/205. for x in CENTERS[value]) if value in CENTERS else None


def _distance(first,second):
    if first is None or second is None:return 0.
    return math.dist(first,second)


@lru_cache(maxsize=8192)
def _native_markers(text,bpm):
    from maichart.generation.muri.parser import parse_markers
    markers,errors,_=parse_markers(f'({bpm}){{4}}{text},E')
    # Cached parser values are read only; all absolute clocks below are separate.
    return markers,errors


def _motion_actions(points,tempo,clock,cache=None):
    """Read real native segments; unknown paths remain explicitly unknown."""
    if cache is not None:
        jobs=[];pulses=[];unknown=[];count=0
        for point in points:
            key=(point.beat,point.note_text)
            if key not in cache:
                cache[key]=_motion_actions([point],tempo,clock)
                while len(cache)>8192:cache.popitem(last=False)
            cached_jobs,cached_pulses,coverage=cache[key]
            cache.move_to_end(key)
            # Action dictionaries/path references are immutable after parsing;
            # assignments and branch state live only inside the local beam.
            jobs.extend(cached_jobs);pulses.extend(cached_pulses)
            unknown.extend(coverage['unknown']);count+=coverage['native_markers']
        return jobs,pulses,dict(native_markers=count,unknown=unknown,
            trajectory_source='MiaCode_reference_real_path_samples')
    from maichart.generation.muri.data import reference_data
    from maichart.generation.muri.model import CENTERS,RADII
    references=reference_data();jobs=[];pulses=[];unknown=[];seen_heads=set();markers_count=0
    def pad_position(pad):return tuple(float(x)/205. for x in CENTERS[pad])
    for point in points:
        onset=float(clock.at_floats(np.asarray([float(point.beat)]))[0])
        bpm=next(float(t.bpm) for t in reversed(tempo) if t.beat<=point.beat)
        try:markers,errors=_native_markers(point.note_text,bpm)
        except (ValueError,KeyError,TypeError) as exc:
            unknown.append(dict(beat=str(point.beat),raw=point.note_text,reason=str(exc)));continue
        unknown.extend(dict(beat=str(point.beat),raw=point.note_text,reason=str(error)) for error in errors)
        for marker in markers:
            kind=marker['type'];markers_count+=1
            if marker.get('is_mine') or marker.get('track_mine'):continue
            second=onset+float(marker.get('second',0.))
            pad=marker.get('touch_pad') if kind in ('touch','touch_hold') else f"A{marker.get('lane')}"
            if pad not in CENTERS:
                unknown.append(dict(beat=str(point.beat),raw=marker['raw'],reason='unknown_pad'));continue
            position=pad_position(pad)
            if kind in ('hold','touch_hold'):
                finish=onset+float(marker.get('end_second',0.))
                jobs.append(dict(start=second,trace=second,end=max(second,finish),position=position,segments=[],kind=kind))
            elif kind in ('slide','wifi'):
                head=(second,pad)
                if marker.get('has_head_star',True) and head not in seen_heads:
                    pulses.append((second,position,'button',RADII[pad[0]]/205.));seen_heads.add(head)
                try:
                    trace=onset+float(marker['slide_trace_second']);finish=onset+float(marker['end_second'])
                    if trace<second-1e-7 or finish<=trace:raise ValueError('invalid_native_trace_clock')
                    if kind=='wifi':
                        from maichart.generation.sustain_state import _wifi_paths
                        paths=_wifi_paths(references.get('wifi',{}).get(marker['slide_track_key'],{}))
                        groups=[[(trace,finish,path)] for path in paths]
                    else:
                        keys=marker['slide_segment_keys'];shoots=marker['slide_segment_shoot_seconds'];durations=marker['slide_segment_durations']
                        if not keys or len(keys)!=len(shoots) or len(keys)!=len(durations):raise ValueError('incomplete_native_segments')
                        segments=[]
                        for key,shoot,length in zip(keys,shoots,durations):
                            entry=references.get('slides',{}).get(key,{})
                            path=entry.get('real_path_samples') or entry.get('samples') or []
                            if len(path)<2:raise ValueError('native_path_missing:'+str(key))
                            segments.append((onset+float(shoot),onset+float(shoot)+float(length),path))
                        groups=[segments]
                    for segments in groups:
                        # Keep the preparation bound for motion stream consumers.
                        # Hand assignment uses trace for the start of contact.
                        jobs.append(dict(start=max(second,trace-.12),onset=second,trace=trace,end=finish+.016666667,
                            position=position,segments=segments,kind=kind, track_key=marker.get('slide_track_key',''),
                            head_contact=bool(marker.get('has_head_star',True))))
                except (ValueError,KeyError,TypeError) as exc:
                    unknown.append(dict(beat=str(point.beat),raw=marker['raw'],reason=str(exc)))
            else:pulses.append((second,position,'touch' if kind=='touch' else 'button',RADII[pad[0]]/205.))
    return jobs,pulses,dict(native_markers=markers_count,unknown=unknown,trajectory_source='MiaCode_reference_real_path_samples')


def _job_position(job,second):
    from maichart.generation.sustain_state import _sample_path
    if not job['segments']:return job['position']
    segments=job['segments']
    for begin,end,path in segments:
        if second<=end+1e-9:
            progress=min(1.,max(0.,(second-begin)/max(1e-8,end-begin)))
            return tuple(float(x)/205. for x in _sample_path(path,progress))
    return tuple(float(x)/205. for x in _sample_path(segments[-1][2],1.))


def _trajectory_comfort(points,tempo,clock,start,end,observed_end,motion_cache=None):
    """Bounded, unlabeled two-trajectory assignment; never a legality gate."""
    jobs,pulses,coverage=_motion_actions(points,tempo,clock,cache=motion_cache)
    warm=start-.6;horizon=max(end,min(end+.35,observed_end))
    jobs=[job for job in jobs if job['end']>=warm and job['trace']<=horizon]
    positions={}
    def position_at(index,second):
        key=(index,second)
        if key not in positions:positions[key]=_job_position(jobs[index],second)
        return positions[key]
    events=[]
    for index,job in enumerate(jobs):
        begin=max(warm,job['trace']);finish=min(horizon,job['end'])
        # The path interpolation is native, sampled at roughly 50 ms.
        count=min(96,max(1,int(math.ceil((finish-begin)/.05))))
        clocks=sorted(set(np.linspace(begin,finish,count+1).tolist()+
            [x for segment in job['segments'] for x in segment[:2] if begin<=x<=finish]))
        previous=begin
        for second in clocks:
            events.append((second,index,position_at(index,second),max(.025,second-previous),False))
            previous=second
    # Nearby simultaneous Touch sensors can share one soft contact group. Use
    # complete-link geometry, avoiding transitive chains across the whole panel.
    groups=[]
    for second,position,kind,radius in sorted(pulses,key=lambda x:x[0]):
        if not warm<=second<=horizon:continue
        group=next((g for g in groups if kind=='touch' and g['kind']=='touch' and abs(g['second']-second)<1e-7
            and all(math.dist(position,p)<=min(.5,radius+r+.15) for p,r in g['members'])),None)
        if group is None:groups.append(dict(second=second,kind=kind,members=[(position,radius)]))
        else:group['members'].append((position,radius))
    for group in groups:
        position=tuple(np.asarray([p for p,_ in group['members']]).mean(0))
        events.append((group['second'],-1,position,1.,True))
    events.sort(key=lambda event:(event[0],event[1]))
    beam=[(0.,((None,warm,-1),(None,warm,-1)),tuple([-1]*len(jobs)))]
    scored_events=0
    for second,job_id,target,weight,pulse in events:
        updated=[]
        is_slide=job_id>=0 and jobs[job_id]['kind'] in ('slide','wifi')
        for cost,hands,owners in beam:
            assigned=owners[job_id] if job_id>=0 else -1
            choices=((assigned,1-assigned) if assigned>=0 and is_slide
                     else (assigned,) if assigned>=0 else (0,1))
            for hand in choices:
                old,previous,previous_job=hands[hand]
                occupied=[index for index,owner in enumerate(owners) if owner==hand and index!=job_id
                    and jobs[index]['trace']<=second<=jobs[index]['end']]
                origins=[position_at(index,second) for index in occupied]
                origin=origins[0] if origins else old
                # Following the same path needs no jump between sampled points.
                # An intervening contact, hand transfer, or concurrent job still does.
                follows_path=is_slide and assigned==hand and previous_job==job_id and not origins
                distance=0. if origin is None or follows_path else math.dist(origin,target)
                elapsed=max(.04,second-previous)
                spread=.45 if pulse and abs(second-previous)<1e-7 else 0.
                moving=min(3.,max(0.,max(0.,distance-spread)/elapsed-3.)/6.)
                busy=sum(max(0.,math.dist(position,target)-.28)/.7 for position in origins)
                transfer=.08 if assigned>=0 and hand!=assigned else 0.
                penalty=(moving+busy)*(1. if pulse else weight/.1)+transfer
                state=list(hands);state[hand]=(target,second,job_id)
                allocation=list(owners)
                if job_id>=0:allocation[job_id]=hand
                updated.append((cost+(penalty if second>=start else 0.),tuple(state),tuple(allocation)))
        updated.sort(key=lambda entry:entry[0]);beam=updated[:32]
        if second>=start:scored_events+=1
    cost=beam[0][0] if beam else 0.
    unknown_count=len(coverage['unknown'])
    relevant_actions=len(jobs)+len(groups)
    known=(relevant_actions/max(1,relevant_actions+unknown_count)) if unknown_count else 1.
    score=math.exp(-cost/max(1.,horizon-start))
    coverage.update(known_fraction=float(known),unknown_count=unknown_count,
        complete=not coverage['unknown'],sample_step_sec=.05,max_samples_per_action=96,
        active_jobs=len(jobs),touch_contact_groups=sum(g['kind']=='touch' for g in groups),
        hand_assignment='beam32_unlabeled_2d_soft',lookahead_sec=max(0.,horizon-end),events=scored_events,
        warmup_sec=.6,boundary_state='all_active_actions_plus_recent_contacts',
        known_fraction_scope='relevant_native_actions_and_unknown_reports')
    coverage['unknown']=coverage['unknown'][:32]
    return float(known*score+(1.-known)*.5),coverage


def _comfort(notes, start_second, end_second):
    """Compatibility onset/hold estimator; the main scorer uses native paths.

    Holds retain their hand location until release. Extra simultaneous heads
    and movement while occupied incur only soft costs, allowing alternatives.
    """
    records = []
    for note in notes:
        second = float(note.time_sec)
        duration = max(0.,float(note.duration_sec or 0.)) if note.note_type in ('hold','touch_hold') else 0.
        release = second+duration
        if second < end_second and (second>=start_second-.35 or release>start_second):
            if note.note_type=='slide' and note.modifiers.get('no_head',False):continue
            records.append((second,_lane(note),release))
    records.sort(key=lambda item:item[0])
    # state = (cost, ((last lane, last time, lock until), ...)). No handedness.
    beam = [(0.,((None,-1e9,-1e9),(None,-1e9,-1e9)))]
    for second,lane,release in records:
        candidates = []
        for cost,hands in beam:
            for hand in range(2):
                old,previous,locked = hands[hand]
                distance = _distance(old,lane)
                elapsed = max(.035,second-previous)
                moving = max(0.,distance/elapsed-8.)/16.
                occupied = max(0.,locked-second)
                penalty = min(3.,moving)+(min(1.,occupied/.3)*distance/2. if occupied>0 else 0.)
                updated = list(hands)
                if locked>second:
                    updated[hand] = (old,second,max(locked,release))
                else:updated[hand] = (lane,second,release)
                candidates.append((cost+(penalty if second>=start_second else 0.),tuple(updated)))
        candidates.sort(key=lambda entry:entry[0])
        # Beam pruning is a bounded approximation, not a claim of exact optimum.
        beam = candidates[:32]
    cost = beam[0][0] if beam else 0.
    return float(math.exp(-cost/max(1.,end_second-start_second)))


class PhraseQualityScorer:
    def __init__(self,song,condition,*,audio_bank=None):
        self.song,self.condition = song,condition
        self.clock = TempoClock(song.tempo,song.first)
        if audio_bank is None:
            from .stream_data import build_audio_bank
            audio_bank = build_audio_bank(song,self.clock)
        self.bank = audio_bank
        self.acoustic = acoustic_descriptors(audio_bank)
        self.seconds = audio_bank[1].detach().cpu().numpy()
        values = np.nan_to_num(audio_bank[0].detach().cpu().numpy(),nan=0.,posinf=0.,neginf=0.)
        padded = np.zeros((len(values),238),dtype=np.float64)
        padded[:,:min(238,values.shape[-1])] = values[:,:238]
        local = padded[:,104:224]
        self.energy = _unit(local[:,93])
        onset,flux = _unit(local[:,92]),_unit(local[:,94])
        energy_change = _unit(np.maximum(0.,np.diff(local[:,93],prepend=local[0,93])))
        self.salience = np.clip(.42*onset+.28*flux+.18*energy_change+.12*self.energy,0.,1.)
        self._prefix_cache = OrderedDict()
        self._motion_cache = OrderedDict()

    def _parse(self,points):
        return parse_event_notes(points,self.song.tempo,self.song.first,
            duration_semantics=MIACODE_DURATION_SEMANTICS).notes

    def _prefix_notes(self,points):
        key = tuple((str(point.beat),point.note_text) for point in points)
        if key not in self._prefix_cache:
            self._prefix_cache[key] = self._parse(points)
            while len(self._prefix_cache)>4:self._prefix_cache.popitem(last=False)
        self._prefix_cache.move_to_end(key)
        return self._prefix_cache[key]

    def score_phrase(self,points,prefix_points,start_beat,end_beat,*,softcomfort_strength=1.,reference=None,observed_end_second=None):
        start,end = map(float,self.clock.at_floats(np.asarray([float(start_beat),float(end_beat)])))
        duration = max(end-start,1e-6)
        # Callers may supply an entire rollout including its committed prefix.
        # Parse only the candidate span, so held prefix actions are counted once.
        # A StreamStep NOTE is committed at step.end. Attribute an onset at
        # the right boundary to this span, never to the following span.
        current_points = [point for point in points if
                          (float(start_beat)<float(point.beat)<=float(end_beat)
                           or float(start_beat)==0. and float(point.beat)==0.)]
        notes = self._parse(current_points)
        active = [note for note in notes if start-1e-8<=float(note.time_sec)<=end+1e-8
                  and not (note.note_type=='slide' and note.modifiers.get('no_head',False))]
        # Compare identical elapsed spans; dense chords cannot multiply coverage.
        cell_count = max(4,int(math.ceil(duration/.125)))
        edges = np.linspace(start,end,cell_count+1)
        centers = (edges[:-1]+edges[1:])/2
        salience = np.zeros(cell_count);energy = np.zeros(cell_count)
        for i,(left,right) in enumerate(zip(edges[:-1],edges[1:])):
            lo,hi = np.searchsorted(self.seconds,(left,right))
            if hi>lo:
                salience[i] = float(self.salience[lo:hi].max())
                energy[i] = float(self.energy[lo:hi].mean())
        onsets = np.asarray(sorted(set(float(note.time_sec) for note in active)))
        occupancy = np.zeros(cell_count)
        for onset in onsets:
            occupancy = np.maximum(occupancy,np.exp(-.5*((centers-onset)/.09)**2))
        activity = float(np.maximum(salience,energy*.35).mean())
        salience_mass = float(salience.sum())
        coverage = float((salience*np.minimum(1.,occupancy/.45)).sum()/salience_mass) if salience_mass>1e-8 else (1. if not active else .5)
        # Missing a few peaks is allowed. Large musical silence stays valuable.
        coverage_score = coverage if activity>.025 else 1.
        precision = float(1.-np.maximum(0.,occupancy-(salience+.18)).mean())
        if not active and activity>.08:coverage_score = 0.
        level,density,chord = _condition(self.condition)
        target_nps = max(.5,16.*max(0.,density))
        actual_nps = len(active)/duration
        # A broad, symmetric adequacy preference, never a minimum-note rule.
        # Sparse and dense deviations both cost; local acoustic activity scales
        # the level-derived target rather than treating a model plan as truth.
        local_target=target_nps*(.35+.65*min(1.25,activity/.3))
        density_score=(math.exp(-(math.log((actual_nps+.5)/(local_target+.5))/.85)**2)
                       if activity>.08 else math.exp(-actual_nps/3.))
        head_counts = {}
        for note in active:
            second = float(note.time_sec);head_counts[second] = head_counts.get(second,0)+1
        high_heads = max(2.,2.+chord*3.+max(0.,level-12.)*.2)
        head_score = math.exp(-sum(max(0.,count-high_heads) for count in head_counts.values())/max(1.,duration))
        # Phrase organization: match broad subphrase contrasts, not mandatory variety.
        groups = min(4,cell_count)
        audio_shape = np.asarray([salience[ix].mean() for ix in np.array_split(np.arange(cell_count),groups)])
        chart_shape = np.asarray([occupancy[ix].mean() for ix in np.array_split(np.arange(cell_count),groups)])
        centered_audio,centered_chart = audio_shape-audio_shape.mean(),chart_shape-chart_shape.mean()
        norm = np.linalg.norm(centered_audio)*np.linalg.norm(centered_chart)
        organization = .5+.5*float(centered_audio@centered_chart/norm) if norm>1e-8 else .75
        boundary = float(1.-np.maximum(0.,occupancy[[0,-1]]-(salience[[0,-1]]+.3)).mean())
        # The completed rollout is authoritative at each onset. Only actual
        # provided future points/WAIT horizon participate in the handoff.
        merged={point.beat:point for point in prefix_points}
        merged.update({point.beat:point for point in points})
        actual_points=[merged[beat] for beat in sorted(merged)]
        observed=(max([end]+[float(self.clock.at_floats(np.asarray([float(p.beat)]))[0]) for p in points])
                  if observed_end_second is None else float(observed_end_second))
        comfort,comfort_coverage=_trajectory_comfort(actual_points,self.song.tempo,self.clock,start,end,observed,self._motion_cache)
        # A previous committed span is optional reference only when its actual
        # music resembles this interval. Unrelated music is entirely neutral.
        prior_start = max(0.,float(start_beat)-(float(end_beat)-float(start_beat)))
        reference_points = (list(reference) if isinstance(reference,(tuple,list)) else
                            [p for p in prefix_points if prior_start<=float(p.beat)<float(start_beat)] if reference is None else [])
        def types(points_):
            result = np.zeros(5)
            for point in points_:
                present,_,_ = typed_event(point.note_text)
                result += [float(kind in present) for kind in ('tap','hold','slide','touch','touch_hold')]
            return result/max(1.,result.sum())
        reference_audio_similarity = 0.
        reference_used = False
        reference_score = .5
        if reference_points:
            # Explicit points need a known preceding interval; unsupported or
            # out-of-interval references remain neutral instead of guessing.
            valid_reference = all(prior_start<=float(point.beat)<float(start_beat) for point in reference_points)
            if valid_reference:
                reference_start = float(self.clock.at_floats(np.asarray([prior_start]))[0])
                lo,mid,hi = np.searchsorted(self.seconds,(reference_start,start,end))
                if mid>lo and hi>mid:
                    old = self.acoustic[lo:mid].mean(0)
                    current = self.acoustic[mid:hi].mean(0)
                    norm = float(np.linalg.norm(old)*np.linalg.norm(current))
                    cosine = float(old@current/norm) if norm>1e-8 else 0.
                    # Magnitude agreement also matters; centered descriptors
                    # with no evidence do not qualify as a matched phrase.
                    difference = float(np.sqrt(np.mean((old-current)**2)))
                    reference_audio_similarity = max(0.,cosine)*math.exp(-difference)
                    if reference_audio_similarity>.65:
                        reference_used = True
                        similarity = float(1.-np.abs(types(current_points)-types(reference_points)).sum()/2.)
                        reference_score = .5+.5*reference_audio_similarity*similarity
        strength = max(0.,float(softcomfort_strength))
        total = (.34*coverage_score+.16*precision+.12*organization+.06*boundary
                 +.12*density_score+.025*head_score+.025*reference_score+.15*strength*comfort)/(.85+.15*strength)
        components = dict(heuristic_uncalibrated=True,audio_coverage=coverage_score,audio_precision=precision,
            audio_activity=activity,subphrase_organization=organization,boundary=boundary,
            density_range=density_score,head_range=head_score,softcomfort=comfort,softcomfort_coverage=comfort_coverage,
            weak_committed_reference=reference_score,actual_nps=float(actual_nps),target_nps=float(target_nps),
            reference_audio_similarity=float(reference_audio_similarity),reference_used=bool(reference_used),
            local_density_target=float(local_target),density_target_scope='level_and_observed_acoustic_activity_heuristic',
            boundary_scope='acoustic_edges_plus_softcomfort_actual_trial_handoff',
            comfort_assignment='beam32_unlabeled_2d_native_trajectories')
        return dict(total=float(total),components=components)


def score_phrase(points,prefix_points,song,condition,start_beat,end_beat,*,softcomfort_strength=1.,reference=None,scorer=None,observed_end_second=None):
    scorer = scorer or PhraseQualityScorer(song,condition)
    return scorer.score_phrase(points,prefix_points,start_beat,end_beat,
        softcomfort_strength=softcomfort_strength,reference=reference,observed_end_second=observed_end_second)
