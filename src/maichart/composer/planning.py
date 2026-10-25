"""Audio-only coarse planning on fixed beat spans, with interpolated readout."""
from fractions import Fraction
import numpy as np
import torch
from torch import nn
from .acoustics import CONTEXT_DIM
from .events import event_metadata

PLAN_POLICY='four_beat_audio_plan_v2_merged_boundaries'
PLAN_CHANNELS=('event_rate','head_rate','hold','slide','touch','rest')


class PlanGrid:
    def __init__(self, clock, end_sec, acoustic_context, views=None):
        self.end_sec=float(end_sec)
        # A four-beat computational span is not asserted to be a musical bar.
        seconds=[0.]
        first=float(clock.at(0))
        if 0. < first < self.end_sec:
            seconds.append(first)
        cursor=Fraction(4)
        while float(clock.at(cursor))<end_sec:
            second=max(0.,float(clock.at(cursor)))
            if second>seconds[-1]+1e-9:seconds.append(second)
            cursor+=4
        seconds=np.unique(np.clip(np.r_[seconds,end_sec],0.,end_sec))
        if len(seconds)<2:seconds=np.array([0.,max(end_sec,1e-6)])
        # Merge short boundary fragments with their adjacent four-beat span.
        # Full musical rests remain visible; a tiny [0, first] or final sliver
        # must not dominate interpolation or turn one event into 1000 events/s.
        if len(seconds)>2 and seconds[1]-seconds[0] < .5*(seconds[2]-seconds[1]):
            seconds=np.delete(seconds,1)
        if len(seconds)>2 and seconds[-1]-seconds[-2] < .5*(seconds[-2]-seconds[-3]):
            seconds=np.delete(seconds,-2)
        self.edges=seconds;self.centers=(seconds[:-1]+seconds[1:])*.5
        self.audio=acoustic_context.at(self.centers)
        self.targets=np.zeros((len(self.centers),6),np.float32)
        self.weights=np.ones(len(self.centers),np.float32)
        if views is not None:
            self.weights=views.window_weight(self.edges[:-1],np.nextafter(self.edges[1:],-np.inf))
            indices=np.clip(np.searchsorted(self.edges,views.seconds,side='right')-1,0,len(self.centers)-1)
            counts=np.zeros((len(self.centers),5),np.float64)
            for index,point,known in zip(indices,views.context_points,views.trusted):
                if not known:continue
                meta=event_metadata(point.note_text)
                counts[index]+=np.array([1.,len(meta.heads),meta.hold,meta.slide,meta.touch])
            duration=np.maximum(np.diff(self.edges),1e-6)
            self.targets[:,:2]=np.log1p(counts[:,:2]/duration[:,None])
            self.targets[:,2:5]=counts[:,2:]/np.maximum(counts[:,:1],1.)
            self.targets[:,5]=counts[:,0]==0
        # Clipped spans contribute in proportion to real temporal support.
        self.weights *= (np.diff(self.edges)/np.diff(self.edges).mean()).astype(np.float32)

    def locate(self, seconds):
        seconds=np.asarray(seconds,dtype=np.float64)
        right=np.clip(np.searchsorted(self.centers,seconds),0,len(self.centers)-1)
        left=np.maximum(0,right-1)
        mix=np.clip((seconds-self.centers[left])/np.maximum(self.centers[right]-self.centers[left],1e-9),0,1)
        return np.column_stack((left,right)).astype(np.int64),mix.astype(np.float32)


class AudioPlanner(nn.Module):
    def __init__(self,width,condition_dim):
        super().__init__()
        self.input=nn.Linear(CONTEXT_DIM+condition_dim,width)
        self.layers=nn.ModuleList(nn.Conv1d(width,width,3,padding=d,dilation=d) for d in (1,2,4))
        self.output=nn.Linear(width,6)

    def forward(self,audio,condition,padding):
        controls=condition[:,None].expand(-1,audio.size(1),-1)
        x=self.input(torch.cat((audio,controls),-1)).masked_fill(padding[...,None],0.).transpose(1,2)
        for layer in self.layers:
            x=(x+torch.nn.functional.gelu(layer(x))).masked_fill(padding[:,None],0.)
        raw=self.output(x.transpose(1,2)).float()
        return torch.cat((torch.nn.functional.softplus(raw[...,:2]),raw[...,2:].sigmoid()),-1)


def read_plan(plan,indices,mix):
    width=plan.size(-1)
    left=plan.gather(1,indices[...,0,None].expand(-1,-1,width))
    right=plan.gather(1,indices[...,1,None].expand(-1,-1,width))
    return left+(right-left)*mix[...,None]
