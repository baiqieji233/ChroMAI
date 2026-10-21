"""Bounded evidence from the trained authored-onset pointer, never a grid."""
from __future__ import annotations

import math
import numpy as np


class PositionEvidence:
    def __init__(self,times,scores,selection,*,strength=.25):
        times,scores,selection=(np.asarray(x,dtype=np.float64) for x in (times,scores,selection))
        if times.ndim!=1 or times.shape!=scores.shape or times.shape!=selection.shape:
            raise ValueError('Position evidence requires matching vectors')
        valid=np.isfinite(times)&np.isfinite(scores)&np.isfinite(selection)&(scores>-1e3)
        times,scores,selection=times[valid],scores[valid],selection[valid]
        order=np.argsort(times,kind='stable');times,scores,selection=times[order],scores[order],selection[order]
        keep=np.r_[True,np.diff(times)>0] if len(times) else np.zeros(0,dtype=bool)
        self.times,self.scores=times[keep],scores[keep]
        selected=selection[keep]
        self.strength=float(strength);self.confidence=0.
        if len(self.scores)>1:
            p=np.exp(self.scores-self.scores.max());p/=p.sum()
            concentration=max(0.,1.+float((p*np.log(p.clip(1e-30))).sum())/math.log(len(p)))
            # Class-weighted selection training does not calibrate sigmoid.
            # Use relative contrast only, not a mandatory confidence threshold.
            contrast=min(1.,4.*float(np.ptp(selected)))
            self.confidence=concentration*contrast
        self.queries=self.unsupported=0;self.max_cost=0.

    def cost(self,second):
        self.queries+=1;second=float(second)
        if (not len(self.times) or not self.times[0]<=second<=self.times[-1]
                or not self.confidence or not self.strength):
            self.unsupported+=1;return 0.
        score=float(np.interp(second,self.times,self.scores))
        value=self.strength*self.confidence*min(2.,max(0.,float(self.scores.max())-score))
        self.max_cost=max(self.max_cost,value)
        return value

    def report(self):
        return {'source':'authored_next_event_pointer','strength':self.strength,
            'confidence_heuristic':self.confidence,'queries':self.queries,'unsupported':self.unsupported,
            'support_seconds':[float(self.times[0]),float(self.times[-1])] if len(self.times) else [],
            'max_cost':self.max_cost,'cost_bound':2.*self.strength,
            'not_a_calibrated_probability':True,'output_positions_quantized':False}
