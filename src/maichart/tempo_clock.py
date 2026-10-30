"""Validated tempo integration reused at many exact positions."""
from bisect import bisect_left,bisect_right
from fractions import Fraction
import numpy as np
from maichart.event_timing import EventTempo,_exact_beat,_finite_number


class TempoClock:
    def __init__(self,tempo,first=0.):
        changes=sorted((EventTempo(_exact_beat(t.beat),_finite_number(t.bpm,'BPM',positive=True)) for t in tempo),key=lambda t:t.beat)
        if not changes or changes[0].beat!=0 or any(t.beat<0 for t in changes):
            raise ValueError('Tempo map requires nonnegative beats and BPM at beat 0')
        self.beats=[t.beat for t in changes];self.bpms=[t.bpm for t in changes]
        elapsed=_finite_number(first,'first');self.seconds=[elapsed]
        for previous,current in zip(changes,changes[1:]):
            elapsed+=float(current.beat-previous.beat)*60./previous.bpm
            self.seconds.append(elapsed)
        self._beat_floats=np.asarray([float(b) for b in self.beats],dtype=np.float64)
        self._seconds_arr=np.asarray(self.seconds,dtype=np.float64)
        self._scale_arr=60./np.asarray(self.bpms,dtype=np.float64)
    def at(self,beat):
        target=beat if type(beat) is Fraction else _exact_beat(beat)
        if target<0:raise ValueError('beat must be nonnegative')
        index=max(0,bisect_left(self.beats,target)-1)
        return self.seconds[index]+float(target-self.beats[index])*60./self.bpms[index]
    def at_floats(self,beat_floats):
        arr=np.asarray(beat_floats,dtype=np.float64)
        if len(self.beats)==1:
            return self.seconds[0]+arr*self._scale_arr[0]
        idx=np.clip(np.searchsorted(self._beat_floats,arr,side='left')-1,0,len(self._beat_floats)-1)
        return self._seconds_arr[idx]+(arr-self._beat_floats[idx])*self._scale_arr[idx]
    def bpm(self,beat):return self.bpms[max(0,bisect_right(self.beats,beat)-1)]
