"""Section preferences integrated over seconds on the nonuniform audio clock."""
from __future__ import annotations

import math

import numpy as np
from maichart.form.curve import decode_density

# HEAD_TYPES = tap, slide, hold, touch, touch_hold.
_EPS = .02


class SectionTypeProfile:
    def __init__(self, curve, times, *, window_seconds=4.):
        curve = np.asarray(curve, dtype=np.float64)
        times = np.asarray(times, dtype=np.float64)
        if curve.ndim != 2 or curve.shape[1] != 4 or len(curve) != len(times) or not len(times):
            raise ValueError('Section profile requires a [T,4] curve on its time axis')
        if not math.isfinite(window_seconds) or window_seconds <= 0:
            raise ValueError('window_seconds must be finite and positive')
        if not np.isfinite(curve).all() or not np.isfinite(times).all():
            raise ValueError('Section profile must be finite')
        order=np.argsort(times,kind='stable');times=times[order];values=np.clip(curve[order],0.,1.)
        unique,inverse,counts=np.unique(times,return_inverse=True,return_counts=True)
        merged=np.zeros((len(unique),4));np.add.at(merged,inverse,values)
        self.times=unique;self.values=merged/counts[:,None]
        self.rates=decode_density(self.values[:,0],'musical_log_v1')
        widths=np.diff(self.times)
        self.cumulative=np.vstack((np.zeros((1,4)),np.cumsum(
            widths[:,None]*(self.values[:-1]+self.values[1:])*.5,axis=0)))
        self.rate_cumulative=np.r_[0.,np.cumsum(widths*(self.rates[:-1]+self.rates[1:])*.5)]
        if len(unique)==1:
            self.mean_density=float(self.values[0,0]);self.reference=self.values[0,1:].copy()
        else:
            duration=unique[-1]-unique[0];density_area=self.cumulative[-1,0]
            self.mean_density=float(density_area/duration)
            d0,d1=self.values[:-1,0,None],self.values[1:,0,None]
            s0,s1=self.values[:-1,1:],self.values[1:,1:]
            # Exact product integral of the two linear interpolants.
            weighted=(widths[:,None]*(2*d0*s0+d0*s1+d1*s0+2*d1*s1)/6.).sum(0)
            self.reference=weighted/density_area if density_area>1e-9 else self.cumulative[-1,1:]/duration
        self.window = float(window_seconds)

    def _integral(self,second,values,cumulative):
        second=float(np.clip(second,self.times[0],self.times[-1]))
        if len(self.times)==1:return np.zeros_like(values[0])
        i=int(np.clip(np.searchsorted(self.times,second,side='right')-1,0,len(self.times)-2))
        delta=second-self.times[i]
        slope=(values[i+1]-values[i])/(self.times[i+1]-self.times[i])
        return cumulative[i]+delta*values[i]+.5*delta*delta*slope

    def _mean(self,second,window,values,cumulative):
        if len(self.times)==1:return values[0]
        left=max(self.times[0],float(second)-window);right=min(self.times[-1],float(second)+window)
        if right<=left:return values[0] if second<self.times[0] else values[-1]
        return (self._integral(right,values,cumulative)-self._integral(left,values,cumulative))/(right-left)

    def density_factor(self, second):
        local=float(self._mean(second,self.window,self.values,self.cumulative)[0])
        return float(np.clip((local+_EPS)/(self.mean_density+_EPS), .25, 2.5))

    def head_rate(self,second,window_seconds=1.):
        """Predicted note heads/second, including simultaneous heads."""
        return float(self._mean(second,window_seconds,self.rates,self.rate_cumulative))

    def __call__(self, second):
        local=self._mean(second,self.window,self.values,self.cumulative)[1:]
        tap, sustain, slide_touch = (local+_EPS)/(self.reference+_EPS)
        mixed = math.sqrt(sustain*slide_touch)
        return np.asarray([tap, mixed, sustain, slide_touch, mixed])
