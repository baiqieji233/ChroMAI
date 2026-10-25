"""Explicit authored context, trusted labels and decision supervision views."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math

import numpy as np

from maichart.tempo_clock import TempoClock
from .events import EVENT_SEMANTICS, event_metadata

EVENT_VIEW_POLICY = 'in_range_context_trusted_labels_v1'


@dataclass(frozen=True, slots=True)
class EventViews:
    context_points: tuple
    source_indices: tuple[int, ...]
    seconds: np.ndarray
    trusted: np.ndarray
    local_weights: np.ndarray
    unknown_intervals: tuple[tuple[float,float], ...]
    regions: tuple[tuple[float,float,float], ...]
    end_sec: float
    final_known: bool
    source_digest: str

    @classmethod
    def build(cls, points, tempo, first, end_sec, *, excluded=(), annotation=None, action_slots=8):
        """Keep excluded events as context; never turn their deletion into rest.

        All indices used by Composer samples refer to context_points. Outside
        audio heads cannot become generated-prefix input or supervised labels.
        A known EOS needs the original final event and a known trailing region.
        """
        end_sec=float(end_sec)
        if not math.isfinite(end_sec) or end_sec<0:
            raise ValueError('Composer event-view audio end must be finite and nonnegative')
        regions=[]
        unknown=[]
        for lo,hi in excluded:
            if not math.isfinite(float(lo)) or not math.isfinite(float(hi)) or float(hi)<float(lo):
                raise ValueError('Invalid excluded interval')
            if hi>lo:unknown.append((max(0.,float(lo)),min(end_sec,float(hi))))
        for row in (annotation or {}).get('regions',[]):
            lo,hi,weight=float(row['start_sec']),float(row['end_sec']),float(row.get('weight',1.))
            if not all(map(math.isfinite,(lo,hi,weight))) or hi<lo:
                raise ValueError('Invalid Composer annotation region')
            regions.append((lo,hi,max(0.,weight)))
            if weight<=0 and hi>lo:unknown.append((max(0.,lo),min(end_sec,hi)))
        clock=TempoClock(tempo,first)
        # Cropped/negative-first charts may carry sustained actions into the
        # audio. Generation starts without them: treat their live span as
        # unknown rather than teaching an unoccupied beginning as trustworthy.
        for point in points:
            if Fraction(point.beat)<0:continue
            second=float(clock.at(point.beat))
            if second>=0.:break
            from maichart.generation.muri.parser import parse_markers
            markers,errors,_=parse_markers(f'({clock.bpm(point.beat)}){{4}}{point.note_text},',first=second)
            if errors:raise ValueError('Cannot parse carry-in Composer action')
            finish=max((float(m.get('end_second',second)) for m in markers),default=second)
            if finish>0.:unknown.append((0.,min(end_sec,finish)))
        merged=[]
        for lo,hi in sorted((lo,hi) for lo,hi in unknown if hi>lo):
            if merged and lo<=merged[-1][1]:merged[-1]=(merged[-1][0],max(hi,merged[-1][1]))
            else:merged.append((lo,hi))
        context=[];indices=[];seconds=[];trusted=[];weights=[];prior=None
        for index,point in enumerate(points):
            beat=Fraction(point.beat)
            if prior is not None and beat<=prior:
                raise ValueError('Composer authored events must have finite, strictly increasing times')
            prior=beat
            if beat<0:continue
            second=float(clock.at(beat))
            if not math.isfinite(second):
                raise ValueError('Composer authored events must have finite, strictly increasing times')
            if second<0 or second>=end_sec:continue
            meta=event_metadata(str(point.note_text))
            if len(meta.atoms)>action_slots:
                raise ValueError(f'Composer action_slots={action_slots} truncates {len(meta.atoms)} atoms '
                                 f'at source event {index}, beat {beat}; increase action_slots')
            confidence=1.
            for lo,hi,weight in regions:
                if lo<=second<hi:confidence*=weight
            known=confidence>0 and not any(lo<=second<hi for lo,hi in merged)
            context.append(point);indices.append(index);seconds.append(second)
            trusted.append(known);weights.append(confidence if known else 0.)
        final_known=bool(context and indices[-1]==len(points)-1 and trusted[-1]
                         and not any(hi>seconds[-1] for lo,hi in merged))
        identity={'policy':EVENT_VIEW_POLICY,'semantics':EVENT_SEMANTICS,
            'points':[(str(p.beat),str(p.note_text)) for p in points],
            'tempo':[(str(t.beat),float(t.bpm)) for t in tempo],'first':float(first),
            'end':end_sec,'regions':regions,'unknown':merged,'context_source_indices':indices}
        digest=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        arrays=[np.asarray(seconds,dtype=np.float64),np.asarray(trusted,dtype=bool),np.asarray(weights,dtype=np.float32)]
        for value in arrays:value.flags.writeable=False
        return cls(tuple(context),tuple(indices),*arrays,tuple(merged),tuple(regions),end_sec,final_known,digest)

    def window_weight(self, lower, upper):
        """Confidence in (lower, upper], with exact unknown-span overlap masks."""
        lower,upper=np.broadcast_arrays(np.asarray(lower,dtype=np.float64),np.asarray(upper,dtype=np.float64))
        weight=np.ones(lower.shape,dtype=np.float32)
        for lo,hi in self.unknown_intervals:
            weight[(upper>=lo)&(lower<hi)]=0.
        for lo,hi,value in self.regions:
            if value>0:
                weight[(upper>=lo)&(lower<hi)]*=min(1.,value)
        weight[(upper<0.)|(upper>=self.end_sec)|(lower>=self.end_sec)]=0.
        return weight

    def eos(self, index):
        # A later trusted event proves continuation. An unknown tail proves
        # neither continuation nor ending, so it contributes no EOS loss.
        if np.any(self.trusted[index+1:]):return 0.,1.
        if index==len(self.context_points)-1 and self.final_known:return 1.,1.
        return 0.,0.

    def future_mask(self, index, stop):
        mask=np.zeros(8,dtype=np.float32)
        if not self.trusted[index]:return mask
        before=self.seconds[index-1] if index else 0.
        mask[0]=float(self.window_weight(before,self.seconds[index])>0)
        enough=(stop-index==4 or (stop==len(self.context_points) and self.final_known))
        known=enough and bool(np.all(self.trusted[index:stop])) and bool(self.window_weight(before,self.seconds[stop-1])>0)
        mask[1:7]=float(known)
        mask[7]=float(index>0 and self.trusted[index-1] and self.window_weight(before,self.seconds[index])>0)
        return mask
