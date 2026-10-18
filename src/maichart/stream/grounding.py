"""Exact causal query clocks and target-independent acoustic evidence."""
from __future__ import annotations

from fractions import Fraction
from bisect import bisect_right
from collections import OrderedDict
import hashlib
import json
import math
from pathlib import Path
import tempfile

import numpy as np
import torch

from maichart.event_timing import seconds_at_beat
from maichart.form.rhythm_geometry import CausalGeometry
from maichart.form.rational_audio import acoustic_at_times, audio_queries
from maichart.form.timing import metrical_features
from maichart.tempo_clock import TempoClock

PROFILE = 'causal_audio_v1'
PROFILES = ('none', PROFILE)
CLOCK_DIM = 20


class CausalClock:
    """A Fraction becomes queryable only after its NOTE token is consumed."""
    def __init__(self, codec, tempo, first, duration, *, meters=None,
                 previous=Fraction(0), previous_previous=None, committed=0):
        self.geometry=CausalGeometry(codec,profile='exact_fraction_v2',meters=meters)
        self.geometry.current=Fraction(previous)
        self.tempo,self.first,self.duration=tempo,float(first),float(duration)
        self.timeline=TempoClock(tempo,first)
        self.committed=int(committed)
        before=Fraction(previous if previous_previous is None else previous_previous)
        self.delta=float(Fraction(previous)-before)
        self.delta_sec=self.timeline.at(previous)-self.timeline.at(before)
        self._key=None;self._features=None

    @property
    def beat(self):return self.geometry.current

    @property
    def pending(self):return self.geometry.state=='note'

    @property
    def second(self):return self.timeline.at(self.beat)

    def consume(self,token):
        old=self.beat;state=self.geometry.state
        row=self.geometry.consume(int(token))
        if token==self.geometry.codec.note_id and self.geometry.state=='note':
            self.delta=float(self.beat-old)
            self.delta_sec=self.timeline.at(self.beat)-self.timeline.at(old)
        if token==self.geometry.codec.end_id and state=='note':self.committed+=1
        key=(self.beat,self.delta,self.committed,self.pending)
        if key!=self._key:
            second=self.second
            bpm=self.timeline.bpm(self.beat)
            self._features=np.asarray((*row,second/120.,second/max(self.duration,1e-6),
                math.log1p(max(0.,self.delta))/math.log(65),
                math.log1p(max(0.,self.delta_sec))/math.log(21),
                self.committed/1024.,float(self.pending),float(bpm)/300.),dtype=np.float32)
            self._key=key
        return self._features


def clock_inputs(ids,codec,song,*,previous=Fraction(0),previous_previous=None,committed=0):
    clock=CausalClock(codec,song.tempo,song.first,song.audio.duration_sec,meters=song.meters,
                      previous=previous,previous_previous=previous_previous,committed=committed)
    rows=[];queries=[];indices=[];times=[];beats=[];counts=[];pending=[];lookup={}
    for token in ids:
        row=clock.consume(token)
        rows.append(row)
        key=(clock.beat,clock.committed,clock.pending)
        if key not in lookup:
            lookup[key]=len(queries);queries.append(row);times.append(clock.second)
            beats.append(clock.beat);counts.append(clock.committed);pending.append(clock.pending)
        indices.append(lookup[key])
    return {'decoder_clock':torch.from_numpy(np.asarray(rows,dtype=np.float32).reshape(-1,CLOCK_DIM)),
        'query_clock':torch.from_numpy(np.asarray(queries,dtype=np.float32).reshape(-1,CLOCK_DIM)),
        'decoder_query_index':torch.tensor(indices,dtype=torch.long),
        'query_seconds':torch.tensor(times,dtype=torch.float64),
        'query_committed':torch.tensor(counts,dtype=torch.long),'query_pending':pending,'query_beats':beats}


def history_clock(ids,start,end,positions,points,codec,song):
    if start==end:return torch.zeros((0,CLOCK_DIM))
    event=max(0,bisect_right(positions,start)-1)
    begin=positions[event] if positions and start>=positions[0] else 0
    previous=points[event-1].beat if event else Fraction(0)
    before=points[event-2].beat if event>1 else Fraction(0)
    clock=CausalClock(codec,song.tempo,song.first,song.audio.duration_sec,meters=song.meters,
                      previous=previous,previous_previous=before,committed=event)
    rows=[]
    for index in range(begin,end):
        row=clock.consume(ids[index])
        if index>=start:rows.append(row)
    return torch.from_numpy(np.asarray(rows,dtype=np.float32).reshape(-1,CLOCK_DIM))


def features_at_beats(song,beats):
    """Native waveform probes at arbitrary known Fractions; no output grid."""
    beats=tuple(map(Fraction,beats))
    audio=song.audio
    times=np.asarray([seconds_at_beat(b,song.tempo,song.first) for b in beats],dtype=np.float64)
    acoustic=acoustic_at_times(audio,times,derived_policy='native_measured_v2')
    duration=float(audio.duration_sec);extent=song.regions[-1].end if song.regions else song.spans[-1].end
    values=np.asarray([float(b) for b in beats])
    bpm=np.asarray([next(t.bpm for t in reversed(song.tempo) if t.beat<=b) for b in beats])
    context=np.column_stack((np.clip(times/max(duration,1e-6),0,1),np.full(len(beats),min(duration/300,1)),
        bpm/300,np.sin(2*np.pi*values),np.cos(2*np.pi*values),values/max(float(extent),1),np.zeros((len(beats),4))))
    rhythm=np.column_stack((acoustic[:,92],acoustic[:,92],acoustic[:,94],acoustic[:,93]))
    result=np.concatenate((acoustic,rhythm,context,metrical_features(beats,song.meters)),1).astype(np.float32)
    result[(times<0)|(times>=duration)]=0
    return torch.from_numpy(result),times


def exact_probe_cache(song,beats,cache_dir):
    """Cache measurements, never augmented history or future chart features.

    The cache may contain all requested chart centers for efficient preparation;
    callers expose only the centers selected by already-consumed prefix clocks.
    """
    beats=tuple(map(Fraction,beats));audio=song.audio
    identity={'profile':PROFILE,'source':audio.metadata['audio_sha256'],'first':float(song.first),
        'tempo':[(str(t.beat),t.bpm) for t in song.tempo],'meters':[m.to_dict() for m in song.meters],
        'beats':list(map(str,beats)),'duration':float(audio.duration_sec),
        'extent':str(song.regions[-1].end if song.regions else song.spans[-1].end)}
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    root=Path(cache_dir)/'stream_exact_probes_v1';root.mkdir(parents=True,exist_ok=True)
    path=root/(key+'.npz')
    try:
        with np.load(path,allow_pickle=False) as data:
            values=data['values'].copy()
        if values.shape==(len(beats),120) and np.isfinite(values).all():return torch.from_numpy(values),'hit'
    except (OSError,ValueError,KeyError,EOFError):pass
    result,_=features_at_beats(song,beats)
    with tempfile.NamedTemporaryFile(dir=root,suffix='.tmp',delete=False) as stream:
        temporary=Path(stream.name);np.savez_compressed(stream,values=result.numpy())
    try:
        from maichart.form.rational_audio import _publish_query_cache
        _publish_query_cache(temporary,path)
    finally:temporary.unlink(missing_ok=True)
    return result,'rebuilt'


def detail_queries(song,max_nodes):
    return audio_queries(song,song.spans[0].start,song.spans[-1].end,max_nodes=max_nodes)


def time_cell_weights(times,start,end):
    """Voronoi quadrature on a nonuniform source clock, zero outside audio."""
    times=np.asarray(times,dtype=np.float64)
    result=np.zeros(len(times),dtype=np.float32)
    selected=np.flatnonzero((times>=start)&(times<end))
    if len(selected):
        values=times[selected]
        widths=np.diff(np.r_[start,(values[:-1]+values[1:])*.5,end])
        result[selected]=widths/max(end-start,1e-9)
    return torch.from_numpy(result)


def action_summaries(index,beats,pending,committed):
    cache=getattr(index,'_stream_query_summaries',None)
    if cache is None:cache=index._stream_query_summaries=OrderedDict()
    result=[]
    for beat,open_note,count in zip(beats,pending,committed):
        if not int(count):
            result.append(torch.zeros(51));continue
        key=(Fraction(beat),bool(open_note))
        if key not in cache:
            values,padding,_=index.at(beat,include_current=not open_note)
            active=values[~padding]
            row=torch.cat((active.mean(0) if len(active) else values.new_zeros(50),values.new_tensor([len(active)/8.])))
            cache[key]=row
            if len(cache)>512:cache.popitem(last=False)
        cache.move_to_end(key);result.append(cache[key])
    return torch.stack(result) if result else torch.zeros(0,51)


def masked_note_inputs(ids,codec,rng,probability,*,targets=None,quality=None):
    """Hide complete authored cells; never teach false continuations in them."""
    result=list(ids);quality=None if quality is None else list(quality)
    # A decoder input is shifted by one. Targets still identify full event
    # boundaries, including a final END absent from the input slice.
    source=list(targets) if targets is not None else list(ids)
    starts=[i for i,t in enumerate(source) if t==codec.at_id]
    for left,right in zip(starts,starts[1:]+[len(source)]):
        if rng.random()>=probability:continue
        for index in range(left,right):
            if source[index]==codec.eos_id:continue
            if quality is not None:quality[index]=0.
            input_index=index+1 if targets is not None else index
            if input_index<len(result) and codec.tokens[result[input_index]].startswith('N:'):
                result[input_index]=codec.mask_id
    return result,quality
