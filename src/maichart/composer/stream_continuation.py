"""Exact causal run statistics, independent of codec shape/rotation labels."""
from functools import lru_cache
from fractions import Fraction
import math

import torch

from maichart.notes import parse_timing_point_notes
from maichart.timing import RawTimingPoint
from .event_set_codec import _representation

POLICIES = ('off','typed_runs_v1','pattern_runs_v2')
FEATURE_DIM = 16


def continuation_dim(policy):
    if policy not in POLICIES:raise ValueError('invalid continuation_policy')
    return FEATURE_DIM if policy == 'pattern_runs_v2' else 8


@lru_cache(maxsize=65536)
def typed_event(text):
    parsed = parse_timing_point_notes(RawTimingPoint(index=0,raw=text,note_text=text,
        beat=Fraction(0),tick=0,time_sec=0.,division=4,bpm=120.))
    if parsed.unknown_tokens or not parsed.notes:
        return frozenset(),frozenset(),False
    types = frozenset(n.note_type for n in parsed.notes)
    buttons = frozenset(int(n.position) for n in parsed.notes
        if isinstance(n.position,int) and n.note_type in ('tap','hold','slide')
        and not (n.note_type=='slide' and n.modifiers.get('no_head',False)))
    touch = all(n.note_type in ('touch','touch_hold') for n in parsed.notes)
    return types,buttons,touch


@lru_cache(maxsize=65536)
def pattern_event(text):
    """Duration-free shape, complete exact signature, and typed head geometry."""
    parsed=parse_timing_point_notes(RawTimingPoint(index=0,raw=text,note_text=text,
        beat=Fraction(0),tick=0,time_sec=0.,division=4,bpm=120.))
    if parsed.unknown_tokens or not parsed.notes:return None,None,()
    shape,_,_,atoms,fallback=_representation(text)
    geometry=[]
    for note in parsed.notes:
        position=note.position
        if isinstance(position,int):group,lane='button',position
        else:
            position=str(position);group=position[0]
            lane=0 if group=='C' else int(position[1:])
        geometry.append((note.note_type,group,lane))
    geometry=tuple(sorted(geometry))
    template=('fallback',geometry) if fallback else ('regular',shape)
    return template,tuple(sorted(atoms)),geometry


@lru_cache(maxsize=65536)
def rotation_relation(previous,current):
    """Set of common ring rotations matching every typed physical head."""
    if not previous or not current or len(previous)!=len(current):return None
    rotations=[]
    for shift in range(8):
        moved=tuple(sorted((kind,group,((lane-1+shift)%8+1) if lane else 0)
                           for kind,group,lane in previous))
        if moved==current:rotations.append(shift)
    # Center-only configurations carry no directional information.
    return tuple(rotations) if rotations and len(rotations)<8 else None


class ContinuationState:
    """Counters cover the full committed prefix; only output rows are sliced.

    Gap elapsed includes every same-delta transition in the run. Other elapsed
    values measure last minus first onset, so singleton typed/button runs are 0.
    WAIT and unsupported events reset every run. All outputs use log1p. V2
    appends template/exact/ABAB/motion count and elapsed-second pairs to the
    unchanged first eight values; the legacy policy retains its exit semantics.
    """
    def __init__(self,policy='typed_runs_v1'):
        continuation_dim(policy)
        self.policy=policy
        self.reset()

    def reset(self):
        self.gap=None; self.gap_count=0; self.gap_elapsed=Fraction(0)
        self.types=frozenset();self.type_count=0;self.type_start=0.
        self.touch_count=0;self.touch_start=0.
        self.buttons={}; self.second=0.
        self.template=None;self.template_count=0;self.template_start=0.
        self.exact=None;self.exact_count=0;self.exact_start=0.
        self.period_history=[];self.period2_count=0;self.period2_start=0.
        self.geometry=();self.motion_relation=None;self.motion_count=0;self.motion_start=0.

    def append(self,step):
        if self.policy=='off':return
        if step.kind != 2:
            self.reset();return
        types,buttons,touch=typed_event(step.text)
        if not types:
            self.reset();return
        if self.policy=='pattern_runs_v2':self._append_pattern(step)
        gap=step.end-step.start
        if gap == self.gap:
            self.gap_count+=1;self.gap_elapsed+=gap
        else:
            self.gap=gap;self.gap_count=1;self.gap_elapsed=gap
        second=float(step.end_second)
        if types == self.types:
            self.type_count+=1
        else:
            self.types=types;self.type_count=1;self.type_start=second
        if touch:
            if not self.touch_count:self.touch_start=second
            self.touch_count+=1
        else:self.touch_count=0
        self.buttons={key:(self.buttons[key][0]+1,self.buttons[key][1]) if key in self.buttons else (1,second)
                      for key in buttons}
        self.second=second

    def _append_pattern(self,step):
        template,exact,geometry=pattern_event(step.text)
        second=float(step.end_second)
        if template==self.template:self.template_count+=1
        else:self.template=template;self.template_count=1;self.template_start=second
        if exact==self.exact:self.exact_count+=1
        else:self.exact=exact;self.exact_count=1;self.exact_start=second
        history=self.period_history
        if len(history)>=2 and exact==history[-2][0] and exact!=history[-1][0]:
            self.period2_count+=1
        elif history and exact!=history[-1][0]:
            self.period2_count=2;self.period2_start=history[-1][1]
        else:self.period2_count=0;self.period2_start=second
        history.append((exact,second))
        if len(history)>2:del history[0]
        relation=rotation_relation(self.geometry,geometry)
        if relation is not None:
            if relation==self.motion_relation:self.motion_count+=1
            else:self.motion_count=1;self.motion_start=self.second
        else:self.motion_count=0;self.motion_start=second
        self.motion_relation=relation;self.geometry=geometry

    def features(self):
        count,start=max(self.buttons.values(),default=(0,self.second),key=lambda x:(x[0],-x[1]))
        values=(self.gap_count,float(self.gap_elapsed),self.type_count,
                self.second-self.type_start if self.type_count else 0.,self.touch_count,
                self.second-self.touch_start if self.touch_count else 0.,count,self.second-start)
        if self.policy=='pattern_runs_v2':
            values+=tuple(v for count,start in ((self.template_count,self.template_start),
                (self.exact_count,self.exact_start),(self.period2_count,self.period2_start),
                (self.motion_count,self.motion_start)) for v in
                (count,self.second-start if count else 0.))
        return torch.tensor([math.log1p(max(0.,v)) for v in values],dtype=torch.float32)

    def eligible_categories(self):
        if self.policy=='off':return ()
        result=[]
        if self.gap_count>=4:result.append('gap')
        if self.touch_count>=3:result.append('touch')
        if any(count>=2 for count,_ in self.buttons.values()):result.append('button')
        if self.policy=='pattern_runs_v2':
            for name,eligible in (('type',self.type_count>=3),('template',self.template_count>=3),
                ('exact',self.exact_count>=2),('period2',self.period2_count>=4),('motion',self.motion_count>=3)):
                if eligible:result.append(name)
        return tuple(result)

    def exit_categories(self,target):
        if target.weight<=0:return ()
        eligible=self.eligible_categories()
        if self.policy=='pattern_runs_v2' and target.kind in (3,4):return eligible
        if target.kind!=2:return ()
        types,buttons,touch=typed_event(target.text)
        if not types:return ()
        exits=dict(gap=target.end-target.start!=self.gap,touch=not touch,
                   button=any(count>=2 and key not in buttons for key,(count,_) in self.buttons.items()))
        if self.policy=='pattern_runs_v2':
            template,exact,geometry=pattern_event(target.text)
            exits.update(type=types!=self.types,template=template!=self.template,exact=exact!=self.exact,
                period2=len(self.period_history)<2 or exact!=self.period_history[-2][0],
                motion=rotation_relation(self.geometry,geometry)!=self.motion_relation)
        return tuple(name for name in eligible if exits.get(name,False))

    def exit_label(self,target,*,exit_categories=None):
        """Return eligibility/exit labels; these are never model inputs."""
        if self.policy=='pattern_runs_v2':
            eligible=self.eligible_categories()
            if target.weight<=0 or target.kind not in (2,3,4):return False,False
            if target.kind==2 and not typed_event(target.text)[0]:return False,False
            return bool(eligible),bool(self.exit_categories(target) if exit_categories is None else exit_categories)
        if target.kind != 2 or target.weight <= 0:
            return False,False
        types,buttons,touch=typed_event(target.text)
        if not types:return False,False
        gap_eligible=self.gap_count>=4
        touch_eligible=self.touch_count>=3
        repeated_buttons={key for key,(count,_) in self.buttons.items() if count>=2}
        button_eligible=bool(repeated_buttons)
        button_exit=bool(repeated_buttons-buttons)
        return (gap_eligible or touch_eligible or button_eligible,
                (gap_eligible and target.end-target.start != self.gap) or
                (touch_eligible and not touch) or button_exit)


def continuation_rows(stream,policy='typed_runs_v1',*,exit_anchors=None,unknown_steps=None,song_index=0):
    """Optionally collect recovery exits during the existing causal traversal.

    Unknown/zero-weight steps still update prefix state, but cannot become
    recovery anchors. The three returned training tensors are unchanged.
    """
    state=ContinuationState(policy);features=[];masks=[];labels=[]
    for index,step in enumerate(stream):
        features.append(state.features())
        categories=state.exit_categories(step) if exit_anchors is not None else None
        eligible,label=state.exit_label(step,exit_categories=categories)
        masks.append(eligible);labels.append(label)
        if exit_anchors is not None and step.weight>0 and (unknown_steps is None or not unknown_steps[index]):
            for name in categories:exit_anchors.setdefault(name,[]).append((song_index,index))
        state.append(step)
    return (torch.stack(features) if features else torch.zeros(0,continuation_dim(policy)),
            torch.tensor(masks,dtype=torch.bool),torch.tensor(labels,dtype=torch.bool))
