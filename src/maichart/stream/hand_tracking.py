"""Causal, bounded hypotheses for which hand struck each button event.

Assignments are latent possibilities, not player labels or hard constraints.
A sweep may use one hand repeatedly; a trill may alternate; neither follows
from its subdivision or from a fixed t -> t-2 rule. Slide ownership is unknown
from a star tap alone, so this branch does not pin that tap's hand to its Slide.
The existing action/trajectory branch supplies actual active Slide geometry.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from collections import OrderedDict
import math
import time

import numpy as np
import torch

HAND_DIM=24
_EMPTY=(0,0.,0.,0.,0,0.)  # lane, last second, dx, dy, same-hand run, Hold end
_XY={lane:(math.sin((lane-.5)*math.pi/4),math.cos((lane-.5)*math.pi/4)) for lane in range(1,9)}


def prepare_training_backend(requested='auto',*,enabled=True):
    """Resolve before launching input workers; missing Numba is supported."""
    if requested not in {'auto','python','native'}:raise ValueError('Unknown hand compute backend')
    if not enabled or requested=='python':return {'requested':requested,'effective':'python'}
    started=time.monotonic()
    try:
        from .native_hand_tracking import warmup
        warmup()
    except Exception as exc:
        if requested=='native':raise ValueError(f'Native hand backend unavailable: {exc}') from exc
        reason=(str(exc).splitlines() or [type(exc).__name__])[0]
        return {'requested':requested,'effective':'python','fallback':reason[:240]}
    return {'requested':requested,'effective':'native','prepare_seconds':time.monotonic()-started}


@dataclass(frozen=True,slots=True)
class Assignment:
    hands: tuple=(_EMPTY,_EMPTY)
    cost: float=0.
    actor: int=-1
    repeated: bool=False
    switched: bool=False


@lru_cache(maxsize=512)
def _splits(lanes):
    return tuple((tuple(lane for j,lane in enumerate(lanes) if bits>>j&1),
                  tuple(lane for j,lane in enumerate(lanes) if not bits>>j&1))
                 for bits in range(1<<len(lanes)))


@lru_cache(maxsize=512)
def _split_geometry(lanes):
    """Cabinet geometry is independent of the latent hand assignment."""
    specs=[]
    for groups in _splits(lanes):
        active=[i for i,g in enumerate(groups) if g]
        actor=active[0] if len(active)==1 else 2
        targets=tuple(min(group,key=lambda x:sum(math.dist(_XY[x],_XY[y]) for y in group))
                      if group else 0 for group in groups)
        spans=tuple(max((math.dist(_XY[a],_XY[b]) for a in group for b in group),default=0.) for group in groups)
        specs.append((groups,actor,targets,spans))
    return tuple(specs)


def _canonical(hands,actor):
    # Remove only whole-hand permutations; clocks, movements, Hold binding
    # and last actor travel with their track, never reassigned independently.
    if hands[1]<hands[0]:return (hands[1],hands[0]),1-actor if actor in (0,1) else actor
    return hands,actor


def advance(states,lanes,second,trajectories=(),*,hidden=False,beam=8):
    if hidden:return (Assignment(),)
    lanes=tuple(sorted(set(lanes)))
    if not lanes:return states  # known buttonless event does not invent a hand
    if len(lanes)>4:return (Assignment(),)  # unresolved, not a rejection of the chart
    holds={head:end for kind,head,_dest,_trace,end in trajectories if kind==0}
    specs=[(groups,actor,targets,spans,tuple(max((holds.get(g,0.) for g in group),default=0.)
                                          for group in groups))
           for groups,actor,targets,spans in _split_geometry(lanes)]
    choices={}
    for state in states:
        for groups,actor,targets,spans,ends in specs:
            hands=[];extra=0.
            repeated=actor in (0,1) and actor==state.actor
            switched=actor in (0,1) and state.actor in (0,1) and actor!=state.actor
            for index,(old,group) in enumerate(zip(state.hands,groups)):
                if not group:hands.append(old);continue
                lane,when,dx,dy,run,hold_end=old
                # A palm group is represented by its observed medoid. Span
                # adds soft ambiguity cost, never a grammar prohibition.
                target=targets[index]
                span=spans[index]
                extra+=max(0.,span-.9)
                if lane:
                    nx,ny=_XY[target][0]-_XY[lane][0],_XY[target][1]-_XY[lane][1]
                    elapsed=max(.04,second-when)
                    extra+=.02*math.hypot(nx,ny)/elapsed
                    if dx or dy:extra+=.03*math.hypot(nx-dx,ny-dy)
                    if hold_end>second+1e-9 and target!=lane:extra+=2.
                else:nx=ny=0.
                hands.append((target,second,nx,ny,min(8,run+1) if repeated else 1,
                              ends[index]))
            hands,actor=_canonical(tuple(hands),actor)
            candidate=Assignment(hands,state.cost+extra,actor,repeated,switched)
            key=(hands,actor,repeated,switched)
            if key not in choices or candidate.cost<choices[key].cost:choices[key]=candidate
    ordered=sorted(choices.values(),key=lambda x:x.cost)
    if not ordered:return (Assignment(),)
    # Reserve room for both interpretations when possible. A cheap local
    # assignment must not erase every single-hand sweep or every alternation.
    chosen=[];selected=set()
    for predicate in (lambda s:s.repeated,lambda s:s.switched):
        candidate=next((s for s in ordered if predicate(s)),None)
        if candidate is not None and id(candidate) not in selected:
            chosen.append(candidate);selected.add(id(candidate))
    continuous=max(ordered,key=lambda s:(max(s.hands[0][4],s.hands[1][4]),-s.cost))
    if id(continuous) not in selected:chosen.append(continuous);selected.add(id(continuous))
    # choices already deduplicated complete state keys. Values in ordered are
    # distinct, so identity membership is equivalent to costly dataclass ==.
    chosen.extend(s for s in ordered if id(s) not in selected)
    minimum=ordered[0].cost
    return tuple(Assignment(s.hands,s.cost-minimum,s.actor,s.repeated,s.switched) for s in chosen[:beam])


def features(states,now,active_slides=0):
    costs=np.asarray([s.cost for s in states],dtype=np.float64)
    weights=np.exp(-(costs-costs.min()));weights/=weights.sum()
    values=np.zeros((len(states),HAND_DIM),dtype=np.float64)
    for row,state in enumerate(states):
        # Slots are spatially ordered at readout; each slot's motion and clock
        # still belong to the same latent hand history. No anatomical side rule.
        tracks=sorted(state.hands,key=lambda h:(_XY[h[0]][0] if h[0] else 0.,h[0]))
        for hand,(lane,second,dx,dy,run,end) in enumerate(tracks):
            if lane:
                x,y=_XY[lane]
                values[row,hand*8:hand*8+8]=(x,y,dx/2,dy/2,
                    min(1.,math.log1p(max(0.,now-second))/math.log(9.)),1.,
                    min(1.,max(0.,end-now)/8.),run/8.)
        a,b=tracks
        values[row,16]=math.dist(_XY[a[0]],_XY[b[0]])/2 if a[0] and b[0] else 0.
        values[row,19]=float(state.repeated)
        values[row,20]=float(state.switched)
        values[row,21]=min(1.,active_slides/2.)
        values[row,22]=sum(bool(h[0]) for h in tracks)/2.
    values[:,17]=-float((weights*np.log(weights.clip(1e-12))).sum())/math.log(max(2,len(states)))
    values[:,18]=float(weights.max())
    # A live Slide's tracing hand can differ from its star-tapping hand. Expose
    # unresolved ownership rather than claiming these strike tracks are exact.
    values[:,23]=float(active_slides>0)
    return (weights[:,None]*values).sum(0).astype(np.float32)


class HandHistory:
    """Sparse prefix checkpoints: bounded CPU memory and cheap masked replay."""
    def __init__(self,lanes,seconds,trajectories=None,*,stride=32,keep_recent=True):
        if stride<1:raise ValueError('Hand checkpoint stride must be positive')
        self.lanes=lanes;self.seconds=seconds;self.trajectories=trajectories
        self.stride=stride;self.checkpoints={0:(Assignment(),)}
        self.keep_recent=keep_recent
        self.count=0;self.states=self.checkpoints[0]
        self.max_slide_duration=[0.]
        self.recent=OrderedDict({0:self.states})

    def _trajectory(self,index):
        return self.trajectories[index] if self.trajectories is not None and index<len(self.trajectories) else ()

    def _extend(self,count):
        while self.count<count:
            self.states=advance(self.states,self.lanes[self.count],float(self.seconds[self.count]),self._trajectory(self.count))
            self.count+=1
            if self.keep_recent:
                self.recent[self.count]=self.states
                while len(self.recent)>2*self.stride:self.recent.popitem(last=False)
            if self.count%self.stride==0:self.checkpoints[self.count]=self.states

    def _ensure_timing(self,count):
        # Timing needs only a cheap prefix maximum. It does not require
        # solving hand assignments for an entire song hidden by a later MASK.
        for index in range(len(self.max_slide_duration)-1,count):
            duration=max((max(0.,end-float(self.seconds[index])) for kind,_h,_e,_t,end
                          in self._trajectory(index) if kind==1),default=0.)
            self.max_slide_duration.append(max(self.max_slide_duration[-1],duration))

    def rows(self,committed,seconds,*,masked_events=()):
        counts=[int(c) for c in committed];now=[float(s) for s in seconds]
        if len(counts)!=len(now) or any(c<0 or c>len(self.lanes) for c in counts):
            raise ValueError('Invalid causal hand query')
        if any(not math.isfinite(t) for t in now):raise ValueError('Nonfinite hand query time')
        result=np.zeros((len(counts),HAND_DIM),dtype=np.float32)
        if not counts:return torch.from_numpy(result)
        hidden=frozenset(int(i) for i in masked_events if 0<=int(i)<max(counts))
        self._ensure_timing(max(counts))
        # A masked event resets both latent hands to Assignment(). Everything
        # before the most recent reset preceding the first query is provably
        # irrelevant, including raw song-prefix checkpoints. Skip that work.
        prior_reset=max((event for event in hidden if event<min(counts)),default=-1)
        if prior_reset>=0:
            at=prior_reset+1;states=(Assignment(),)
        else:
            self._extend(min(counts) if hidden else max(counts))
            at=min(counts)//self.stride*self.stride
            states=self.checkpoints[at]
        readouts={}
        for row in sorted(range(len(counts)),key=lambda i:counts[i]):
            count=counts[row]
            if not hidden and count in self.recent:
                states=self.recent[count];at=count
            else:
                while at<count:
                    states=advance(states,self.lanes[at],float(self.seconds[at]),self._trajectory(at),hidden=at in hidden)
                    at+=1
            if not count:continue
            key=(count,now[row])
            if key in readouts:
                result[row]=readouts[key];continue
            # Only committed actions can be active; inspect their native timing.
            start=int(np.searchsorted(self.seconds[:count],now[row]-self.max_slide_duration[count]-1e-9))
            slides=sum(kind==1 and trace<=now[row]<end for i in range(start,count)
                       if i not in hidden for kind,_h,_e,trace,end in self._trajectory(i))
            result[row]=features(states,now[row],slides)
            readouts[key]=result[row]
        return torch.from_numpy(result)
