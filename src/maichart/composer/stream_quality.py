"""Conservative phrase ranking heuristics, not calibrated human preferences.

Physical feasibility belongs to EventGuard. These scores never reject a
technique, require a note at every acoustic peak, or reward retry counts.
"""
from collections import OrderedDict
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
    value = note.position
    if isinstance(value,int):return float((value-1)%8)
    value = str(value)
    if value == 'C':return None
    return float((int(value[1:])-1)%8) if value[1:].isdigit() else None


def _distance(first,second):
    if first is None or second is None:return 0.
    delta = abs(first-second)%8
    return min(delta,8-delta)


def _comfort(notes, start_second, end_second):
    """Beam minimum over unlabeled hand assignments with exact hold lifetime.

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

    def score_phrase(self,points,prefix_points,start_beat,end_beat,*,softcomfort_strength=1.,reference=None):
        start,end = map(float,self.clock.at_floats(np.asarray([float(start_beat),float(end_beat)])))
        duration = max(end-start,1e-6)
        # Callers may supply an entire rollout including its committed prefix.
        # Parse only the candidate span, so held prefix actions are counted once.
        current_points = [point for point in points if float(start_beat)<=float(point.beat)<float(end_beat)]
        notes = self._parse(current_points)
        active = [note for note in notes if start<=float(note.time_sec)<end
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
        coverage_score = min(1.,coverage/.55) if activity>.025 else 1.
        precision = float(1.-np.maximum(0.,occupancy-(salience+.18)).mean())
        if not active and activity>.08:coverage_score = 0.
        level,density,chord = _condition(self.condition)
        target_nps = max(.5,16.*max(0.,density))
        lower,upper = target_nps*.25,target_nps*2.5+2.
        actual_nps = len(active)/duration
        outside = max(0.,lower-actual_nps,actual_nps-upper)
        density_score = math.exp(-outside/max(2.,target_nps)) if activity>.08 else math.exp(-max(0.,actual_nps-2.)/4.)
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
        prefix_notes = [note for note in self._prefix_notes(prefix_points) if float(note.time_sec)<start]
        comfort = _comfort(prefix_notes+notes,start,end)
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
        total = (.39*coverage_score+.21*precision+.12*organization+.06*boundary
                 +.055*density_score+.025*head_score+.02*reference_score+.12*strength*comfort)/(.88+.12*strength)
        components = dict(heuristic_uncalibrated=True,audio_coverage=coverage_score,audio_precision=precision,
            audio_activity=activity,subphrase_organization=organization,boundary=boundary,
            density_range=density_score,head_range=head_score,softcomfort=comfort,
            weak_committed_reference=reference_score,actual_nps=float(actual_nps),target_nps=float(target_nps),
            reference_audio_similarity=float(reference_audio_similarity),reference_used=bool(reference_used),
            boundary_scope='current_interval_edges_only_no_future_chart_observed',
            comfort_assignment='beam32_unlabeled_hands_exact_hold_seconds')
        return dict(total=float(total),components=components)


def score_phrase(points,prefix_points,song,condition,start_beat,end_beat,*,softcomfort_strength=1.,reference=None,scorer=None):
    scorer = scorer or PhraseQualityScorer(song,condition)
    return scorer.score_phrase(points,prefix_points,start_beat,end_beat,
        softcomfort_strength=softcomfort_strength,reference=reference)
