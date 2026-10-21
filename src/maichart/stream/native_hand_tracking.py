"""Optional GIL-free implementation of the existing bounded hand search.

No fastmath, parallel reduction, new search objective or changed beam policy.
Cabinet distances/turn norms are precomputed with the original Python math;
query readout reuses the exact existing features() implementation.
"""
from __future__ import annotations

import math
import numpy as np
import torch
from numba import njit

from .hand_tracking import HandHistory,Assignment,HAND_DIM,features,_XY,_split_geometry


def _geometry():
    groups=np.zeros((256,16,2),dtype=np.int64)
    targets=np.zeros_like(groups);spans=np.zeros((256,16,2),dtype=np.float64)
    sizes=np.zeros(256,dtype=np.int64);xy=np.zeros((9,2),dtype=np.float64)
    for lane,point in _XY.items():xy[lane]=point
    for mask in range(1,256):
        lanes=tuple(i+1 for i in range(8) if mask>>i&1)
        if len(lanes)>4:continue
        specs=_split_geometry(lanes);sizes[mask]=len(specs)
        for j,(split,_actor,head,width) in enumerate(specs):
            for hand in range(2):
                groups[mask,j,hand]=sum(1<<(lane-1) for lane in split[hand])
                targets[mask,j,hand]=head[hand];spans[mask,j,hand]=width[hand]
    delta=np.zeros((9,9,2),dtype=np.float64)
    distance=np.zeros((9,9),dtype=np.float64)
    turns=np.zeros((9,9,9),dtype=np.float64)
    for old in range(1,9):
        for new in range(1,9):
            dx,dy=xy[new]-xy[old];delta[old,new]=(dx,dy)
            distance[old,new]=math.hypot(dx,dy)
    for previous in range(1,9):
        for old in range(1,9):
            for new in range(1,9):
                dx,dy=delta[old,new]-delta[previous,old]
                turns[previous,old,new]=math.hypot(dx,dy)
    return groups,targets,spans,sizes,delta,distance,turns


_TABLES=None


def _tables():
    global _TABLES
    if _TABLES is None:_TABLES=_geometry()
    return _TABLES


@njit(cache=True,nogil=True,fastmath=False)
def _empty():
    values=np.zeros((8,18),dtype=np.float64)
    values[0,13]=-1.
    return values


@njit(cache=True,nogil=True,fastmath=False)
def _advance(states,count,mask,second,holds,groups,targets,spans,sizes,delta,distance,turns,hidden):
    if hidden:return _empty(),1
    if mask==0:return states,count
    if sizes[mask]==0:return _empty(),1
    candidates=np.zeros((128,18),dtype=np.float64);number=0
    for row in range(count):
        for split in range(sizes[mask]):
            candidate=states[row].copy();extra=0.
            a,b=groups[mask,split,0],groups[mask,split,1]
            actor=0 if a and not b else 1 if b and not a else 2
            repeated=actor<2 and actor==states[row,13]
            switched=actor<2 and 0<=states[row,13]<2 and actor!=states[row,13]
            for hand in range(2):
                group=groups[mask,split,hand]
                if not group:continue
                offset=hand*6;lane=int(states[row,offset])
                target=targets[mask,split,hand]
                extra+=max(0.,spans[mask,split,hand]-.9)
                if lane:
                    nx,ny=delta[lane,target,0],delta[lane,target,1]
                    elapsed=max(.04,second-states[row,offset+1])
                    extra+=.02*distance[lane,target]/elapsed
                    if states[row,offset+2] or states[row,offset+3]:
                        previous=int(states[row,16+hand])
                        extra+=.03*turns[previous,lane,target]
                    if states[row,offset+5]>second+1e-9 and target!=lane:extra+=2.
                    previous=lane
                else:nx=ny=0.;previous=target
                end=0.
                for button in range(8):
                    if group>>button&1:end=max(end,holds[button+1])
                candidate[offset]=target;candidate[offset+1]=second
                candidate[offset+2]=nx;candidate[offset+3]=ny
                candidate[offset+4]=min(8,states[row,offset+4]+1) if repeated else 1
                candidate[offset+5]=end;candidate[16+hand]=previous
            swap=False
            for field in range(6):
                if candidate[6+field]<candidate[field]:swap=True;break
                if candidate[6+field]>candidate[field]:break
            if swap:
                for field in range(6):candidate[field],candidate[6+field]=candidate[6+field],candidate[field]
                candidate[16],candidate[17]=candidate[17],candidate[16]
                if actor<2:actor=1-actor
            candidate[12]=states[row,12]+extra
            candidate[13]=actor;candidate[14]=repeated;candidate[15]=switched
            duplicate=-1
            for existing in range(number):
                equal=True
                for field in range(16):
                    if field!=12 and candidates[existing,field]!=candidate[field]:equal=False;break
                if equal:duplicate=existing;break
            if duplicate<0:candidates[number]=candidate;number+=1
            elif candidate[12]<candidates[duplicate,12]:candidates[duplicate]=candidate
    if not number:return _empty(),1
    # Stable insertion order matches Python's dict followed by stable sorted.
    order=np.arange(number)
    for i in range(1,number):
        item=order[i];j=i-1
        while j>=0 and candidates[order[j],12]>candidates[item,12]:
            order[j+1]=order[j];j-=1
        order[j+1]=item
    selected=np.zeros(number,dtype=np.bool_);chosen=np.empty(8,dtype=np.int64);chosen_count=0
    for flag in (14,15):
        for index in order:
            if candidates[index,flag]:
                if not selected[index]:chosen[chosen_count]=index;chosen_count+=1;selected[index]=True
                break
    continuous=order[0]
    for index in order[1:]:
        run=max(candidates[index,4],candidates[index,10])
        best_run=max(candidates[continuous,4],candidates[continuous,10])
        if run>best_run:continuous=index
    if not selected[continuous]:
        chosen[chosen_count]=continuous;chosen_count+=1;selected[continuous]=True
    for index in order:
        if chosen_count==8:break
        if not selected[index]:chosen[chosen_count]=index;chosen_count+=1
    result=np.zeros((8,18),dtype=np.float64)
    minimum=candidates[order[0],12]
    for row in range(chosen_count):
        result[row]=candidates[chosen[row]];result[row,12]-=minimum
    return result,chosen_count


@njit(cache=True,nogil=True,fastmath=False)
def _solve(states,count,start,end,masks,seconds,holds,hidden,groups,targets,spans,sizes,delta,distance,turns,stride):
    slots=end//stride-start//stride
    checkpoints=np.zeros((slots,8,18),dtype=np.float64);numbers=np.zeros(slots,dtype=np.int64);slot=0
    for event in range(start,end):
        states,count=_advance(states,count,masks[event],seconds[event],holds[event],
            groups,targets,spans,sizes,delta,distance,turns,hidden[event])
        if (event+1)%stride==0:
            checkpoints[slot]=states;numbers[slot]=count;slot+=1
    return states,count,checkpoints,numbers


@njit(cache=True,nogil=True,fastmath=False)
def _query(states,count,start,counts,masks,seconds,holds,hidden,groups,targets,spans,sizes,delta,distance,turns):
    rows=np.zeros((len(counts),8,18),dtype=np.float64);numbers=np.zeros(len(counts),dtype=np.int64)
    at=start
    for row in range(len(counts)):
        while at<counts[row]:
            states,count=_advance(states,count,masks[at],seconds[at],holds[at],
                groups,targets,spans,sizes,delta,distance,turns,hidden[at]);at+=1
        rows[row]=states;numbers[row]=count
    return rows,numbers


def warmup():
    """Compile once before CPU prefetch competes with CUDA dispatch."""
    masks=np.array([1,128],dtype=np.int64);seconds=np.array([0.,.125]);holds=np.zeros((2,9))
    hidden=np.zeros(2,dtype=np.bool_);state=_empty()
    _solve(state,1,0,2,masks,seconds,holds,hidden,*_tables(),1)
    _query(state,1,0,np.array([1,2],dtype=np.int64),masks,seconds,holds,hidden,*_tables())


class NativeHandHistory(HandHistory):
    """Immutable song inputs with numeric sparse checkpoints and native replay."""
    def __init__(self,lanes,seconds,trajectories=None,*,stride=128,keep_recent=False):
        if keep_recent:raise ValueError('Native history is for immutable training songs only')
        super().__init__(lanes,seconds,trajectories,stride=stride,keep_recent=False)
        self.native_seconds=np.ascontiguousarray(seconds,dtype=np.float64)
        self.masks=np.fromiter((sum(1<<(lane-1) for lane in set(row)) for row in lanes),dtype=np.int64,count=len(lanes))
        self.holds=np.zeros((len(lanes),9),dtype=np.float64)
        for index in range(len(lanes)):
            for kind,head,_end,_trace,end in self._trajectory(index):
                if kind==0:self.holds[index,head]=end
        self.raw_hidden=np.zeros(len(lanes),dtype=np.bool_)
        self.checkpoints={0:(_empty(),1)}
        self.native_state,self.native_count=self.checkpoints[0]

    def _extend(self,count):
        if count<=self.count:return
        start=self.count
        state,number,snapshots,numbers=_solve(self.native_state,self.native_count,start,count,
            self.masks,self.native_seconds,self.holds,self.raw_hidden,*_tables(),self.stride)
        for index,next_count in enumerate(range((start//self.stride+1)*self.stride,count+1,self.stride)):
            self.checkpoints[next_count]=(snapshots[index].copy(),int(numbers[index]))
        self.native_state,self.native_count=state,int(number);self.count=count

    def rows(self,committed,seconds,*,masked_events=()):
        counts=np.asarray(committed,dtype=np.int64);now=np.asarray(seconds,dtype=np.float64)
        if counts.ndim!=1 or now.shape!=counts.shape or np.any(counts<0) or np.any(counts>len(self.lanes)):
            raise ValueError('Invalid causal hand query')
        if not np.isfinite(now).all():raise ValueError('Nonfinite hand query time')
        result=np.zeros((len(counts),HAND_DIM),dtype=np.float32)
        if not len(counts):return torch.from_numpy(result)
        hidden=frozenset(int(i) for i in masked_events if 0<=int(i)<int(counts.max()))
        self._ensure_timing(int(counts.max()))
        reset=max((event for event in hidden if event<int(counts.min())),default=-1)
        if reset>=0:at=reset+1;state=_empty();number=1
        else:
            self._extend(int(counts.min()) if hidden else int(counts.max()))
            at=int(counts.min())//self.stride*self.stride;state,number=self.checkpoints[at]
        mask=self.raw_hidden.copy() if hidden else self.raw_hidden
        if hidden:mask[list(hidden)]=True
        unique,inverse=np.unique(counts,return_inverse=True)
        rows,numbers=_query(state,number,at,unique,self.masks,self.native_seconds,
            self.holds,mask,*_tables())
        readouts={}
        for row,position in enumerate(inverse):
            count=int(counts[row])
            if not count:continue
            key=(count,float(now[row]))
            if key in readouts:
                result[row]=readouts[key];continue
            values=rows[position]
            states=tuple(Assignment((tuple(v[:6]),tuple(v[6:12])),float(v[12]),int(v[13]),bool(v[14]),bool(v[15]))
                         for v in values[:numbers[position]])
            start=int(np.searchsorted(self.native_seconds[:count],now[row]-self.max_slide_duration[count]-1e-9))
            slides=sum(kind==1 and trace<=now[row]<end for i in range(start,count) if i not in hidden
                       for kind,_h,_e,trace,end in self._trajectory(i))
            result[row]=features(states,float(now[row]),slides)
            readouts[key]=result[row]
        return torch.from_numpy(result)
