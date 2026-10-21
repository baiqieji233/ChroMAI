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
        self.selection=selected
        self.strength=float(strength);self.confidence=0.;self.alignment_confidence=0.;self.selection_confidence=0.
        if len(self.scores)>1:
            p=np.exp(self.scores-self.scores.max());p/=p.sum()
            concentration=max(0.,1.+float((p*np.log(p.clip(1e-30))).sum())/math.log(len(p)))
            # Class-weighted selection training does not calibrate sigmoid.
            # Use relative contrast only, not a mandatory confidence threshold.
            contrast=min(1.,4.*float(np.ptp(selected)))
            self.alignment_confidence=concentration
            self.selection_confidence=contrast
            self.confidence=concentration*contrast
        self.queries=self.unsupported=0;self.max_cost=0.

    def cost(self,second):
        self.queries+=1;second=float(second)
        if (not len(self.times) or not self.times[0]<=second<=self.times[-1]
                or not self.strength or not (self.alignment_confidence or self.selection_confidence)):
            self.unsupported+=1;return 0.
        score=float(np.interp(second,self.times,self.scores))
        alignment_cost=min(2.,max(0.,float(self.scores.max())-score))
        # Selection is an authored-event likelihood proxy, not a calibrated
        # probability.  Price a candidate by its local relative weakness so a
        # whole quiet window is not shifted merely because its absolute score
        # is low.  The bounded term changes candidate ranking without turning
        # every acoustic peak into a mandatory note.
        selected=float(np.interp(second,self.times,self.selection))
        radius=max(.04,min(.35,(self.times[-1]-self.times[0])/32.))
        nearby=self.selection[np.abs(self.times-second)<=radius]
        if nearby.size < 2:
            nearby=self.selection
        low=float(np.quantile(nearby,.15)); high=float(np.quantile(nearby,.85))
        support=float(np.clip((selected-low)/(high-low+1e-6),0.,1.))
        selection_cost=(1.-support)*.65
        # Keep the original conservative rule that alignment-only evidence is
        # inactive when the authored selection head is flat, while allowing a
        # clear selection signal to act even if the alignment pointer itself
        # is diffuse.
        value=self.strength*(self.alignment_confidence*self.selection_confidence*alignment_cost
                             +self.selection_confidence*selection_cost)
        if value < 1e-12:
            value=0.
        self.max_cost=max(self.max_cost,value)
        return value

    def support(self, second):
        """Return bounded local authored-selection support for diagnostics/gating."""
        if not len(self.times):
            return 0.
        second=float(second)
        if second < float(self.times[0]) or second > float(self.times[-1]):
            return 0.
        selected=float(np.interp(second,self.times,self.selection))
        radius=max(.04,min(.35,(self.times[-1]-self.times[0])/32.))
        nearby=self.selection[np.abs(self.times-second)<=radius]
        if nearby.size < 2: nearby=self.selection
        low=float(np.quantile(nearby,.15)); high=float(np.quantile(nearby,.85))
        return float(np.clip((selected-low)/(high-low+1e-6),0.,1.))

    def report(self):
        return {'source':'authored_next_event_pointer','strength':self.strength,
            'confidence_heuristic':self.confidence,'queries':self.queries,'unsupported':self.unsupported,
            'support_seconds':[float(self.times[0]),float(self.times[-1])] if len(self.times) else [],
            'max_cost':self.max_cost,'cost_bound':2.65*self.strength,
            'alignment_confidence':self.alignment_confidence,
            'selection_confidence':self.selection_confidence,
            'selection_local_relative':True,'selection_cost_bound':.65*self.strength,
            'not_a_calibrated_probability':True,'output_positions_quantized':False}
