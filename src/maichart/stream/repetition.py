"""Separate bounded feedback for long configuration cycles and head overload.

Repeated authored figures remain legal. Duration, exact beat span, rhythm and
predicted chart workload all participate; neither note-count thresholds nor
audio amplitude turn a normal trill into a forbidden pattern.
"""
from __future__ import annotations

from collections import deque
from functools import lru_cache
from fractions import Fraction
import numpy as np


@lru_cache(maxsize=4096)
def describe(text):
    from maichart.form.type_guidance import _counts
    counts=_counts(text)
    if counts is None:return None
    return tuple(int(x) for x in counts),int(counts.sum())


class RepetitionGuard:
    def __init__(self,profile,*,strength=.6,structure_strength=0.,min_seconds=4.,min_beats=8.):
        self.profile=profile;self.strength=float(strength)
        self.structure_strength=float(structure_strength)
        self.min_seconds=float(min_seconds);self.min_beats=float(min_beats)
        self.rows=deque(maxlen=1024);self.considered=self.active=0
        self.peak_penalty=0.;self.diagnostics=deque(maxlen=32)
        self.structural_events=self.overload_events=0

    def commit(self,text,beat,second):
        item=describe(text)
        if item is None:return
        self.rows.append((str(text),item[0],item[1],Fraction(beat),float(second)))

    def event(self,base,beat,second):
        self.considered+=1;rows=list(self.rows);cases=[]
        if not (self.strength or self.structure_strength) or len(rows)<3:return RepeatBias(base,())
        beat=Fraction(beat);second=float(second)
        predicted=self.profile.head_rate(second) if self.profile is not None else None
        structural_peak=overload_peak=0.;cycle_peak=0
        for period in range(1,min(8,len(rows)//2)+1):
            for structural in (False,True):
                field=1 if structural else 0
                pattern=[row[field] for row in rows[-period:]]
                # A generic one-Tap stream is not a configuration loop. Other
                # constant types (double Holds, chords) need bounded feedback.
                if structural and len(set(pattern))==1 and pattern[0]==(1,0,0,0,0):continue
                if [row[field] for row in rows[-2*period:-period]]!=pattern:continue
                if any(rows[i][3]-rows[i-1][3]!=rows[i+period][3]-rows[i+period-1][3]
                       for i in range(len(rows)-2*period+1,len(rows)-period)):
                    continue
                # Continue the exact rhythmic cycle only; changing the timing
                # already leaves it, even if the configuration text repeats.
                expected=rows[-period]
                old_index=len(rows)-period
                if old_index<=0 or beat-rows[-1][3]!=rows[old_index][3]-rows[old_index-1][3]:continue
                initial_start=len(rows)-2*period
                start=initial_start
                while start>0:
                    i=start-1
                    if rows[i][field]!=rows[i+period][field]:break
                    if i and rows[i][3]-rows[i-1][3]!=rows[i+period][3]-rows[i+period-1][3]:break
                    start-=1
                cycle_count=2+max(0,(initial_start-start)//period)
                cycle_peak=max(cycle_peak,cycle_count)
                span=float(beat-rows[start][3]);duration=second-rows[start][4]
                if duration<=self.min_seconds or span<=self.min_beats:continue
                elapsed=rows[-1][4]-rows[start][4]
                if elapsed<=0:continue
                observed=sum(row[2] for row in rows[start+1:])/elapsed
                excess=max(0.,observed/max(.25,predicted)-1.35) if predicted is not None else 0.
                persistence=min(1.,(duration-self.min_seconds)/self.min_seconds,
                                  (span-self.min_beats)/self.min_beats)
                overload=2.*self.strength*min(1.,excess)*persistence
                # Monotony is independent of heads/second. A sparse loop can
                # be just as mechanically repetitive as an overloaded one.
                support=1.
                owner=getattr(base,'owner',None)
                intent=getattr(owner,'_pending_intent',None)
                if structural and intent is not None and len(intent)>=4:
                    counts=expected[field]
                    relevant=[float(intent[i]) for i,present in ((0,sum(counts)>1),(1,counts[1]),
                              (2,counts[2]+counts[4]),(3,counts[3]+counts[4])) if present]
                    if relevant:support=1.-.5*float(np.clip(max(relevant),0.,1.))
                # Pressure must keep growing after the current plateau.  The
                # former 1.2-logit ceiling could be ignored by a confident
                # decoder indefinitely, even after many bars of the same loop.
                cycle_pressure=min(8.,2.+.75*max(0,cycle_count-2))
                monotony=self.structure_strength*cycle_pressure*persistence*support
                condition=getattr(owner,'_condition',None)
                reuse=(condition.style.get('pattern_reuse',0.) if condition is not None else 0.)
                monotony*=1.-.75*float(np.clip(reuse,0.,1.))
                penalty=overload+monotony
                structural_peak=max(structural_peak,monotony);overload_peak=max(overload_peak,overload)
                if penalty>0:
                    cases.append((expected[field],structural,penalty))
        if cases:
            self.active+=1;peak=max(c[2] for c in cases);self.peak_penalty=max(self.peak_penalty,peak)
            self.structural_events+=int(structural_peak>0);self.overload_events+=int(overload_peak>0)
            self.diagnostics.append({'second':second,'beat':str(beat),'predicted_heads_per_sec':predicted,
                'max_penalty':peak,'patterns':len(cases),'monotony_penalty':structural_peak,
                'overload_penalty':overload_peak,'cycle_repetitions':cycle_peak})
        return RepeatBias(base,cases)

    def report(self):
        return {'enabled':bool(self.strength or self.structure_strength),'strength':self.strength,
            'structure_strength':self.structure_strength,
            'policy':'separate_configuration_cycles_and_head_overload_v2','minimum_seconds':self.min_seconds,
            'minimum_beats':self.min_beats,'considered_events':self.considered,'active_events':self.active,
            'penalty_bound':max(2.*self.strength,8.*self.structure_strength),'peak_penalty':self.peak_penalty,
            'monotony_events':self.structural_events,'overload_events':self.overload_events,
            'diagnostics':list(self.diagnostics),'hard_rejections':0,
            'limitations':['Predicted workload is uncertain; matching it is not a musical quality certificate.']}


class RepeatBias:
    def __init__(self,base,cases):
        self.base=base;self.cases=tuple(cases);self._prefix_cache={}
        self.exploration_requested=bool(cases)
    def __getattr__(self,name):return getattr(self.base,name)

    def _penalty(self,text):
        description=None;value=0.
        for expected,structural,penalty in self.cases:
            if structural:
                if description is None:description=describe(text)
                matches=description is not None and description[0]==expected
            else:matches=text==expected
            if matches:value=max(value,penalty)
        return value

    def _prefix(self,text):
        if text in self._prefix_cache:return self._prefix_cache[text]
        owner=getattr(self.base,'owner',None)
        if owner is None:
            return -max((penalty for expected,structural,penalty in self.cases
                         if not structural and text and expected.startswith(text)),default=0.)
        parts=owner.grammar._split_slashes(text)
        fixed=self.base._completed(parts[:-1])
        used=frozenset().union(*(owner.grammar.heads.get(part,()) for part in parts[:-1]))
        indices=owner.compatible(parts[-1],used)
        if fixed is None or not len(indices):return 0.
        counts=owner.matrix[indices]+fixed
        costs=np.zeros(len(indices))
        novelty=np.zeros(len(indices))
        prefix='/'.join(parts[:-1]);prefix=prefix+'/' if prefix else ''
        for expected,structural,penalty in self.cases:
            if structural:
                matches=np.all(counts==np.asarray(expected),axis=1)
                # A repeated structure must create an escape route during
                # proposal generation.  Penalising only the matching rows
                # gives every same-structure candidate an equal constant,
                # which softmax cancels.  A bounded novelty potential makes
                # structurally different atoms more likely to be explored
                # while leaving the terminal model law unchanged.
                novelty=np.maximum(novelty, (~matches)*min(2.0,.7*penalty))
            else:matches=np.asarray([prefix+owner.atoms[i]==expected for i in indices])
            costs=np.maximum(costs,matches*penalty)
        scores=-costs+novelty;top=float(scores.max())
        value=top+float(np.log(np.exp(scores-top).mean()))
        self._prefix_cache[text]=value
        return value

    def terminal_score(self,text):return self.base.terminal_score(text)-self._penalty(text)

    def terminal_exploration_score(self,text):
        """Optional candidate-only escape score for structural repetition."""
        return self.terminal_score(text)+self._novelty(text)

    def repeat_penalty(self,text):return self._penalty(text)

    def biases(self,text,continuations):
        parent=self._prefix(text)
        values=[]
        for value,char in zip(self.base.biases(text,continuations),continuations):
            child=(-self._penalty(text) if char is None else self._prefix(text+char))
            # Keep the novelty potential on the terminal transition as well;
            # otherwise a guided proposal can discover a new structure and
            # then lose the entire escape reward when it emits END.
            if char is None and self.cases:
                child += self._novelty(text)
            values.append(value+child-parent)
        return values

    def _novelty(self,text):
        if not self.cases:return 0.
        value=0.
        description=describe(text)
        for expected,structural,penalty in self.cases:
            if structural and description is not None and description[0]!=expected:
                value=max(value,min(2.,.7*penalty))
        return value

    def token_bias(self,text,continuations):return self.base.token_bias(text,continuations)
