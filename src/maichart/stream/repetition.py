"""Bounded configuration feedback when a long cycle exceeds learned head load.

Repeated authored figures remain legal. Duration, exact beat span, rhythm and
predicted chart workload all participate; neither note-count thresholds nor
audio amplitude turn a normal trill into a forbidden pattern.
"""
from __future__ import annotations

from collections import deque
from functools import lru_cache
from fractions import Fraction


@lru_cache(maxsize=4096)
def describe(text):
    from maichart.form.type_guidance import _counts
    counts=_counts(text)
    if counts is None:return None
    return tuple(int(x) for x in counts),int(counts.sum())


class RepetitionGuard:
    def __init__(self,profile,*,strength=.6,min_seconds=4.,min_beats=8.):
        self.profile=profile;self.strength=float(strength)
        self.min_seconds=float(min_seconds);self.min_beats=float(min_beats)
        self.rows=deque(maxlen=1024);self.considered=self.active=0
        self.peak_penalty=0.;self.diagnostics=deque(maxlen=32)

    def commit(self,text,beat,second):
        item=describe(text)
        if item is None:return
        self.rows.append((str(text),item[0],item[1],Fraction(beat),float(second)))

    def event(self,base,beat,second):
        self.considered+=1;rows=list(self.rows);cases=[]
        if not self.strength or self.profile is None or len(rows)<3:return RepeatBias(base,())
        beat=Fraction(beat);second=float(second)
        predicted=self.profile.head_rate(second)
        for period in range(1,min(8,len(rows)//2)+1):
            for structural in (False,True):
                field=1 if structural else 0
                pattern=[row[field] for row in rows[-period:]]
                # Constant Tap shape would penalize every ordinary stream.
                if structural and (period==1 or len(set(pattern))==1):continue
                if [row[field] for row in rows[-2*period:-period]]!=pattern:continue
                if any(rows[i][3]-rows[i-1][3]!=rows[i+period][3]-rows[i+period-1][3]
                       for i in range(len(rows)-2*period+1,len(rows)-period)):
                    continue
                # Continue the exact rhythmic cycle only; changing the timing
                # already leaves it, even if the configuration text repeats.
                expected=rows[-period]
                old_index=len(rows)-period
                if old_index<=0 or beat-rows[-1][3]!=rows[old_index][3]-rows[old_index-1][3]:continue
                start=len(rows)-2*period
                while start>0:
                    i=start-1
                    if rows[i][field]!=rows[i+period][field]:break
                    if i and rows[i][3]-rows[i-1][3]!=rows[i+period][3]-rows[i+period-1][3]:break
                    start-=1
                span=float(beat-rows[start][3]);duration=second-rows[start][4]
                if duration<=self.min_seconds or span<=self.min_beats:continue
                elapsed=rows[-1][4]-rows[start][4]
                if elapsed<=0:continue
                observed=sum(row[2] for row in rows[start+1:])/elapsed
                excess=max(0.,observed/max(.25,predicted)-1.35)
                persistence=min(1.,(duration-self.min_seconds)/self.min_seconds,
                                  (span-self.min_beats)/self.min_beats)
                penalty=2.*self.strength*min(1.,excess)*persistence
                if penalty>0:
                    cases.append((expected[field],structural,penalty))
        if cases:
            self.active+=1;peak=max(c[2] for c in cases);self.peak_penalty=max(self.peak_penalty,peak)
            self.diagnostics.append({'second':second,'beat':str(beat),'predicted_heads_per_sec':predicted,
                'max_penalty':peak,'patterns':len(cases)})
        return RepeatBias(base,cases)

    def report(self):
        return {'enabled':bool(self.strength and self.profile is not None),'strength':self.strength,
            'policy':'long_cycle_excess_authored_head_load_v1','minimum_seconds':self.min_seconds,
            'minimum_beats':self.min_beats,'considered_events':self.considered,'active_events':self.active,
            'penalty_bound':2.*self.strength,'peak_penalty':self.peak_penalty,
            'diagnostics':list(self.diagnostics),'hard_rejections':0,
            'limitations':['Predicted workload is uncertain; matching it is not a musical quality certificate.']}


class RepeatBias:
    def __init__(self,base,cases):self.base=base;self.cases=tuple(cases)
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
        # Steer exact-cycle proposals before END; terminal importance weights
        # below account for this steering, so the penalty is not counted twice.
        return -max((penalty for expected,structural,penalty in self.cases
                     if not structural and text and expected.startswith(text)),default=0.)

    def terminal_score(self,text):return self.base.terminal_score(text)-self._penalty(text)

    def biases(self,text,continuations):
        parent=self._prefix(text)
        return [value+(-self._penalty(text) if char is None else self._prefix(text+char))-parent
                for value,char in zip(self.base.biases(text,continuations),continuations)]

    def token_bias(self,text,continuations):return self.base.token_bias(text,continuations)
