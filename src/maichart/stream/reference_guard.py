"""Bounded reference checks on a candidate and still-relevant authored actions."""
from collections import Counter

from maichart.event_timing import seconds_at_beat
from maichart.generation.event_compile import compile_event_points
from maichart.generation.miacode_python import analyze_maidata
from maichart.generation.muri.parser import parse_markers
from maichart.generation.muri.model import has_head,head_key
from maichart.generation.muri.runtime import simple_notes,build_actions,_merged_points


HARD_KINDS = {'MultiTouch','Overlap','TapOnSlide','SlideHeadTap','SlideTooFast'}


def _involves_current(item, second, keys):
    sources = [item, item.get('affected') or {}, item.get('cause') or {}, *(item.get('sources') or [])]
    return any(source.get('marker_key',source.get('key')) in keys or
               abs(float(source.get('second', -1e9))-second) <= 1e-5 for source in sources)


def summarize_reference(report):
    hard = [item for item in report['diagnostics'] if item.get('alert_level') == 'Muri']
    return {'implementation': 'miacode_python', 'playable_verified': False,
        'parse_ok':report['parse']['ok'], 'runtime_hard_count':len(hard),
        'runtime_hard_counts_by_kind':dict(Counter(item['kind'] for item in hard)),
        'static_reference_count':len(report['static_references']),
        'max_note_head_sec':max((float(m['second']) for m in report['markers']),default=None),
        'max_action_end_sec':max((max(float(m['second']),float(m.get('end_second',-1))) for m in report['markers']),default=None),
        'diagnostics':report['visible_panel_entries'][:64]}


class ReferenceCandidateGuard:
    def __init__(self, tempo, first):
        self.tempo,self.first = tempo,first
        self.records = []
        self.checks,self.rejections,self.fast_accepts = 0,0,0
        self.last_errors = []
        self.actions = []
        self._baseline_cache = {}
        self._live_slide_heads = []

    def _relevant(self,second):
        starts=[start for start,until in self._live_slide_heads if until>=second-1e-8]
        earliest=min(starts) if starts else None
        return [(p,until) for p,until in self.records if until>=second-1e-8 or
                (earliest is not None and float(seconds_at_beat(p.beat,self.tempo,self.first))>=earliest-.2)]

    @staticmethod
    def _signatures(report):
        markers={m['key']:m for m in report['markers']}
        for marker in report['markers']:
            if has_head(marker): markers[head_key(marker)]=marker
        signatures=set()
        for item in report['diagnostics']:
            if item.get('alert_level')!='Muri': continue
            marker=markers.get(item.get('marker_key'),item.get('affected') or item)
            signatures.add((item['kind'],round(float(marker.get('second',0.)),8),
                marker.get('type',marker.get('marker_type')),marker.get('lane'),marker.get('raw')))
        return signatures

    def _baseline(self, points):
        if not points: return set()
        key=tuple((p.beat,p.note_text) for p in points)
        if key not in self._baseline_cache:
            text,_=compile_event_points(points,tempo_changes=self.tempo,first=self.first)
            self._baseline_cache[key]=self._signatures(analyze_maidata(text))
            if len(self._baseline_cache)>32:
                del self._baseline_cache[next(iter(self._baseline_cache))]
        return self._baseline_cache[key]

    def button_capacity(self, beat):
        second = float(seconds_at_beat(beat,self.tempo,self.first))
        active = [i for i,action in enumerate(self.actions)
                  if action.marker['type'] not in {'touch','touch_hold'} and action.sample(second) is not None]
        points,_ = _merged_points(self.actions,active,second)
        # Native runtime coalesces identical static positions as one action.
        static, moving = set(),0
        for i,position,_ in points:
            if self.actions[i].merge: moving += 1
            else: static.add((position,self.actions[i].radius))
        return max(0,2-len(static)-moving)

    def _markers(self, point):
        second = float(seconds_at_beat(point.beat,self.tempo,self.first))
        bpm = next(t.bpm for t in reversed(self.tempo) if t.beat <= point.beat)
        markers,errors,_ = parse_markers(f'({bpm}){{4}}{point.note_text},',first=second)
        if errors: raise ValueError(f'Cannot check candidate {point.note_text!r}: {errors[:1]}')
        return second,markers

    def check(self, point):
        second,markers = self._markers(point)
        relevant = [p for p,until in self._relevant(second)]
        if not relevant and not any(m['type'] in {'slide','wifi'} for m in markers):
            self.fast_accepts += 1
            return True,[]
        text,_ = compile_event_points([*relevant,point],tempo_changes=self.tempo,first=self.first)
        report = analyze_maidata(text)
        baseline=self._baseline(relevant)
        current_signatures=self._signatures(report)
        introduced=current_signatures-baseline
        keys = set()
        for marker in report['markers']:
            if abs(marker['second']-second) <= 1e-5:
                keys.add(marker['key'])
                if has_head(marker): keys.add(head_key(marker))
        self.checks += 1
        errors = [item for item in report['diagnostics']
            if item.get('kind') in HARD_KINDS and item.get('alert_level') == 'Muri'
            and (_involves_current(item,second,keys) or introduced)]
        if errors:
            self.rejections += 1
            self.last_errors = [{'kind':e['kind'],'second':e['second'],'detail':e.get('detail','')}
                                for e in errors[:8]]
        return not errors, errors

    def commit(self, point):
        second,markers = self._markers(point)
        self._live_slide_heads=[(start,end) for start,end in self._live_slide_heads if end>=second-1e-8]
        self.records = self._relevant(second)
        # Retain a past head through its judge window; sustained/Slide actions
        # remain until their actual end. This is only the detector's context,
        # not a fixed music-generation segment or a note-position grid.
        until = max([second+.2, *(float(m.get('end_second',second))+.2 for m in markers)])
        self.records.append((point,until))
        if any(m['type'] in {'slide','wifi'} for m in markers):
            self._live_slide_heads.append((second,until))
        notes,_ = simple_notes(markers)
        self.actions = [action for action in self.actions if action.end >= second-1e-8]
        self.actions.extend(build_actions(markers,notes))

    def report(self):
        return {'checks':self.checks,'rejections':self.rejections,'fast_accepts':self.fast_accepts,
                'last_rejected':self.last_errors,'playable_verified':False,
                'scope':'candidate_and_live_causal_actions',
                'attribution':'counterfactual_reference_delta_or_current_source',
                'limitations':['Future authored actions are unknown. Full-chart audit remains necessary.']}
