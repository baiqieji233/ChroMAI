"""Optional ending evidence from learned chart workload, never audio loudness."""
from __future__ import annotations

import math
import numpy as np
from maichart.form.curve import decode_density


class LearnedEnding:
    def __init__(self,times,density,end,strength=1.):
        self.strength=float(strength)
        self.end=float(end)
        times=np.asarray(times,dtype=np.float64)
        density=np.asarray(density,dtype=np.float64)
        if times.ndim!=1 or times.shape!=density.shape or not len(times):
            raise ValueError('Ending curve requires matching nonempty time/density vectors')
        if not np.isfinite(times).all() or np.any(np.diff(times)<=0):
            raise ValueError('Ending curve clock must be finite and increasing')
        rates=decode_density(density,'musical_log_v1')
        # Duplicate endpoint values are extensions of learned estimates, not
        # zero labels inserted at EOF. Seconds are absolute audio coordinates.
        self.times=np.unique(np.r_[0.,times[(times>0)&(times<self.end)],self.end])
        self.rates=np.interp(self.times,times,rates)
        areas=np.diff(self.times)*(self.rates[:-1]+self.rates[1:])*.5
        self.suffix=np.r_[np.cumsum(areas[::-1])[::-1],0.]
        self.last=None

    def remaining_heads(self,second):
        second=float(np.clip(second,0.,self.end))
        index=min(len(self.times)-2,max(0,int(np.searchsorted(self.times,second,side='right')-1)))
        rate=float(np.interp(second,self.times,self.rates))
        return float(self.suffix[index+1]+(self.times[index+1]-second)*(rate+self.rates[index+1])*.5)

    def eos_logit(self,raw_margin,second):
        remaining=self.remaining_heads(second)
        # exp(-expected count) is a Poisson zero-count approximation, not a
        # calibrated confidence. The existing EOS remains possible everywhere;
        # this extra soft evidence matters only when little chart load remains.
        curve_zero=math.exp(-min(remaining,745.))
        log_survival=-float(np.logaddexp(0.,float(raw_margin)))
        if self.strength:
            log_survival+=self.strength*math.log1p(-min(curve_zero,1.-1e-12))
        probability=-math.expm1(log_survival)
        adjusted=math.log(max(probability,1e-30))-log_survival
        self.last={'audio_second':float(second),'remaining_predicted_heads':remaining,
                   'raw_eos_minus_at':float(raw_margin),'guided_eos_minus_at':adjusted,
                   'zero_count_approximation':curve_zero}
        return adjusted

    def report(self):
        return {'enabled':bool(self.strength),'strength':self.strength,'last':self.last,
            'source':'learned_chart_head_density','policy':'poisson_remaining_count_v1',
            'amplitude_threshold':None,
            'limitations':['The predicted count and Poisson approximation are not calibrated end probabilities.',
                           'A wrong workload prediction can still end too early or too late; strength 0 disables guidance.']}
