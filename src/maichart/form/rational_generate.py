"""Joint exact-position and configuration decoding for song_form_v2."""
from __future__ import annotations
from dataclasses import replace
from fractions import Fraction
import json
import math

import numpy as np
import torch

from maichart.event_timing import EventPoint, seconds_at_beat
from maichart.form.data import collate_form, input_example
from maichart.form.generate import MAX_NOTE_CHARACTERS, _curve_condition, to_device
from maichart.generation.event_cli import _NoteGrammar


class PositionGrammar:
    """A finite text budget, with no list of legal denominators or note grids."""
    def __init__(self,lower,upper,*,characters=24,valid_position=None,allowed_ranges=None,
                 max_denominator=4_096,min_spacing=Fraction(0),max_delta_denominator=16_384):
        self.lower,self.upper = Fraction(lower),Fraction(upper)
        if self.upper <= self.lower or not 4 <= characters <= 128:
            raise ValueError('Invalid rational position bounds/budget')
        self.characters = characters
        # A rational clock has no musical denominator whitelist, but an
        # unbounded decimal grammar lets an under-trained decoder spend all of
        # its budget inventing giant numerators (for example ``114125526``)
        # which are only theoretically recoverable by a giant denominator.
        # Keep the representation exact while bounding that useless tail.  A
        # 4096 denominator still covers ordinary and unusual chart values such
        # as 37/997, while remaining independent of 16/24/32 grids.
        self.max_denominator = int(max_denominator)
        if self.max_denominator < 1:
            raise ValueError('max_denominator must be positive')
        self.max_numerator = max(1, int(math.ceil(float(self.upper) * self.max_denominator)))
        self.min_spacing = max(Fraction(0),Fraction(min_spacing))
        self.max_delta_denominator = int(max_delta_denominator)
        self.valid_position = valid_position or (lambda _:True)
        self.ranges=tuple((Fraction(a),Fraction(b)) for a,b in allowed_ranges) if allowed_ranges is not None else ((Fraction(0),self.upper),)
        self._prefix_cache = {}
        self._allowed_cache = {}

    def complete(self,text):
        try:
            numerator,slash,denominator=text.partition('/')
            if (not numerator.isascii() or not numerator.isdecimal() or
                (len(numerator)>1 and numerator[0]=='0') or
                (slash and (numerator=='0' or not denominator.isascii() or not denominator.isdecimal() or denominator[0]=='0'))):
                return False
            beat = Fraction(text)
            return (self.lower < beat < self.upper and any(a<=beat<b for a,b in self.ranges)
                    and beat.denominator <= self.max_denominator
                    and beat.numerator <= self.max_numerator
                    and beat-self.lower >= self.min_spacing
                    and (beat-self.lower).denominator <= self.max_delta_denominator
                    and self.valid_position(beat))
        except (ValueError,ZeroDivisionError): return False

    def denominator_possible(self,n,lo,hi):
        for start,end in self.ranges:
            end=min(end,self.upper)
            start=max(start,self.lower+self.min_spacing)
            if end<=0 or end<=start: continue
            minimum=max(lo,1,(Fraction(n)/end).__floor__()+1)
            maximum=min(hi,self.max_denominator)
            if start>0: maximum=min(maximum,(Fraction(n)/start).__ceil__()-1)
            if minimum<=maximum:return True
        return False

    def numerator_possible(self,n,remaining):
        def interval_has_integer(low, high, start, end, den):
            first = max(low, (start * den).__floor__() + 1)
            last = min(high, (end * den).__ceil__() - 1)
            return first <= last

        for growth in range(remaining+1):
            low=n*10**growth;high=(n+1)*10**growth-1
            high=min(high,self.max_numerator)
            if low>high: continue
            for start,end in self.ranges:
                end=min(end,self.upper)
                start=max(start,self.lower+self.min_spacing)
                if end<=start:continue
                # The numerator may finish as an integer, or consume the
                # remaining characters as ``/denominator``.  Both checks are
                # interval tests.  The final integer check is exact; for a
                # wide denominator interval a two-beat-width guarantee avoids
                # scanning every value, while narrow musical windows inspect
                # candidates exactly so a prefix can never lead to a dead end.
                if interval_has_integer(low,high,start,end,1): return True
                room=remaining-growth-1
                if room<1:continue
                den_low=max(1,(Fraction(low)/end).__floor__()+1)
                den_high=min(self.max_denominator,10**room-1)
                if start>0:den_high=min(den_high,(Fraction(high)/start).__ceil__()-1)
                if den_low>den_high:continue
                if den_high-den_low>4096 and (end-start)*den_low >= 2:
                    return True
                for den in range(den_low,den_high+1):
                    if interval_has_integer(low,high,start,end,den): return True
        return False

    def prefix(self,text):
        cached = self._prefix_cache.get(text)
        if cached is not None:
            return cached
        if not text or len(text)>self.characters or any(c not in '0123456789/' for c in text) or text.count('/')>1:
            self._prefix_cache[text] = False
            return False
        numerator,_,denominator = text.partition('/')
        if not numerator or (len(numerator)>1 and numerator[0]=='0'):
            self._prefix_cache[text] = False
            return False
        n = int(numerator)
        remaining = self.characters-len(text)
        if '/' not in text:
            if self.complete(text):
                self._prefix_cache[text] = True
                return True
            # An integer can still grow or turn into an arbitrary fraction.
            result = n != 0 and self.numerator_possible(n,remaining)
        elif n==0 or denominator.startswith('0'):
            result = False
        elif not denominator:
            result = remaining>0 and self.denominator_possible(n,1,10**remaining-1)
        elif self.complete(text):
            result = True
        else:
            d = int(denominator)
            result = any(self.denominator_possible(n,d*10**digits,(d+1)*10**digits-1)
                         for digits in range(1,remaining+1))
        self._prefix_cache[text] = result
        return result

    def allowed(self,text,codec):
        cached = self._allowed_cache.get(text)
        if cached is not None:
            return list(cached)
        ids = [codec.ids['P:'+c] for c in '0123456789/' if self.prefix(text+c)]
        if self.complete(text): ids.append(codec.note_id)
        self._allowed_cache[text] = tuple(ids)
        return ids


def compose_span(model,codec,example,atoms,*,origin,end,condition=None,temperature=.7,
                 position_temperature=.2,seed=0,device='cpu',history_last=None,
                 valid_position=None,position_characters=24,type_guidance=None,
                 max_events=None,min_spacing_beats=Fraction(0)):
    """Choose timing and notes jointly. At a resource cap retain completed events."""
    grammar = _NoteGrammar.from_atoms(atoms)
    if grammar is None: raise ValueError('Checkpoint note grammar is empty')
    if not all(math.isfinite(float(t)) and t>=0 for t in (temperature,position_temperature)):
        raise ValueError('Decoder temperatures must be finite and nonnegative')
    batch = to_device(collate_form([example]),device)
    ids = [codec.bos_id]
    points = []
    lower = Fraction(history_last)-origin if history_last is not None else Fraction(-1)
    limit = model.config.max_tokens
    state,text,position = 'event','',None
    header_start=0;normalizations=0
    chars = [(i,t[2:]) for i,t in enumerate(codec.tokens) if t.startswith('N:')]
    rng = torch.Generator(device=device).manual_seed(seed)
    model.eval()
    with torch.inference_mode():
        memory,mask = model.encode(batch)
        cache=None;consumed=0
        def logits():
            nonlocal cache,consumed
            while consumed<len(ids):
                output,cache=model.decode_step(torch.tensor([[ids[consumed]]],device=device),consumed,memory,mask,cache)
                consumed+=1
            return output[0,-1]
        position_grammar = PositionGrammar(lower,end-origin,characters=position_characters,
            valid_position=lambda b: b>=0 and (valid_position is None or valid_position(origin+b)),
            allowed_ranges=example.get('_position_ranges'),min_spacing=min_spacing_beats)
        bias=None
        while len(ids)<limit-1:
            if state=='event':
                if max_events is not None and len(points) >= max_events:
                    return points,{'closed':True,'reason':'event_budget','tokens':len(ids),
                                   'max_events':int(max_events)}
                allowed=[codec.eos_id]
                # Checking only the interval end is insufficient when the
                # previous event is very close to the span boundary: there
                # may be no representable Fraction left even though ``b`` is
                # technically greater than the lower bound.  Ask the grammar
                # whether at least one first numerator digit has a reachable
                # continuation before exposing <at> to the model.
                if position_grammar.allowed('',codec):
                    allowed.append(codec.at_id)
                if len(ids)+position_characters+MAX_NOTE_CHARACTERS+4>limit and points:
                    return points,{'closed':False,'reason':'token_budget','tokens':len(ids)}
            elif state=='position': allowed=position_grammar.allowed(text,codec)
            else:
                allowed=[i for i,c in chars if grammar.can_continue(text+c)]
                if grammar.is_complete(text): allowed.append(codec.end_id)
                if len(text)>=MAX_NOTE_CHARACTERS:
                    if not grammar.is_complete(text):
                        raise ValueError('Rational decoder exceeded note text budget without a complete configuration')
                    allowed=[codec.end_id]
            if not allowed:
                if state == 'position' and points:
                    # A weak checkpoint can walk into an exact Fraction whose
                    # reduced delta would require an impractical simai
                    # subdivision.  Drop only that unfinished event and close
                    # the span; never approximate it to a fixed grid or abort
                    # the whole song.
                    return points,{'closed':True,'reason':'invalid_position_prefix',
                                   'tokens':len(ids),'discarded_partial_event':True}
                raise ValueError(f'No exact rational continuation for {state} prefix {text!r}; no timing approximation was applied')
            values=logits()[allowed]
            if state=='position':
                # Keep exact arbitrary Fractions available, but gently prefer
                # short canonical spellings.  This prevents an untrained
                # checkpoint from wandering through giant decimal prefixes;
                # strong learned logits can still select an unusual value.
                complexity=[]
                for token_id in allowed:
                    token_text=codec.tokens[token_id][2:] if codec.tokens[token_id].startswith('P:') else ''
                    complexity.append(-0.04 * len(text) if token_text.isdigit() else
                                      (0.08 if token_text in {'/'} or token_id==codec.note_id else 0.0))
                values=values+torch.as_tensor(complexity,device=values.device,dtype=values.dtype)
            if state=='note' and bias is not None:
                hints=bias.biases(text,[None if t==codec.end_id else codec.tokens[t][2:] for t in allowed])
                values=values+torch.as_tensor(hints,device=values.device,dtype=values.dtype)
            temp=temperature if state=='note' else position_temperature
            chosen=int(values.argmax()) if temp==0 else int(torch.multinomial(torch.softmax(values/temp,dim=-1),1,generator=rng))
            token=allowed[chosen];ids.append(token)
            if state=='event':
                if token==codec.eos_id:
                    return points,{'closed':True,'reason':'eos','tokens':len(ids),'equivalent_fraction_normalizations':normalizations}
                header_start=len(ids)-1
                state,text='position',''
            elif state=='position':
                if token==codec.note_id:
                    position=origin+Fraction(text)
                    canonical=codec.position(position-origin)
                    if ids[header_start:]!=canonical:
                        ids=ids[:header_start]+canonical
                        if hasattr(model,'rewind_decode_cache'):
                            cache=model.rewind_decode_cache(cache,header_start);consumed=header_start
                        else:cache=None;consumed=0
                        normalizations+=1
                    state,text='note',''
                    if type_guidance is not None and type_guidance.local_chords is not None:
                        second=float(seconds_at_beat(position,example['_tempo'],example['_first']))
                        profile=example.get('_curve_profile')
                        factor=1.
                        if profile is not None:
                            from maichart.form.generate import _curve_factor
                            clock=second-example['_first'] if profile.get('time_axis','chart_relative_v1')=='chart_relative_v1' else second
                            factor=_curve_factor(profile,clock,**example['_curve_options'])
                        type_guidance.local_chords.cues[position]=(second,factor)
                    bias=type_guidance.event(position) if type_guidance is not None else None
                else: text+=codec.tokens[token][2:]
            elif token==codec.end_id:
                points.append(EventPoint(position,text))
                if type_guidance is not None: type_guidance.commit(text,position)
                state,text='event',''
                position_grammar.lower=position-origin
                position_grammar._prefix_cache.clear()
                position_grammar._allowed_cache.clear()
            else: text+=codec.tokens[token][2:]
        return points,{'closed':False,'reason':'token_budget','tokens':len(ids), 'discarded_partial_event':state!='event'}


def generate_rational_chart(model,planner,codec,song,condition,atoms,*,temperature=.7,
        position_temperature=.2,seed=0,device='cpu',memory_strength=1.,intensity_curve=None,
        intensity_curve_strength=1.,intensity_curve_span=.5,intensity_curve_mode='follow',
        style_balance_strength=1.,position_characters=24,max_events_per_span=0,
        min_spacing_ms=2.0,**unused_legacy_options):
    from maichart.form.type_guidance import TypeGuidance
    from maichart.generation.event_quality import assess_event_chart
    from maichart.form.local_load import local_load_statistics
    from maichart.form.curve import resample_load_curve, load_curve_bin_centers
    from maichart.form.rational_audio import audio_queries, beat_at_second
    from maichart.form.data import _form_descriptor
    # Keep the nonuniform/native clock, but use the same bounded evidence
    # budget as training.  Passing every transient in a long song (often many
    # thousands of MP3 decoder peaks) makes each tiny autoregressive decoder
    # step attend to an enormous memory and leaves CUDA under-occupied.  This
    # selects the strongest native attacks; it does not quantize or whitelist
    # the Fraction positions the decoder may emit.
    audio,times,_ = audio_queries(song,song.spans[0].start,song.spans[-1].end,max_nodes=1024)
    with torch.inference_mode():
        prediction=planner(audio[None].to(device),torch.from_numpy(_form_descriptor(song,0))[None].to(device),condition.to_vector(device=device)[None])
    if song.load_curve is None:
        # Match the workload-bin clock used by layout training. Acoustic query
        # density must not change the number or time meaning of curve tokens.
        bins=load_curve_bin_centers(max(0.,song.audio.duration_sec-song.first))
        song.load_curve_times_sec=bins.astype(np.float32)
        song.load_curve=resample_load_curve(times-song.first,
            prediction['load_curve'][0].sigmoid().cpu().numpy(),bins)
    local_strength=unused_legacy_options.get('local_chord_strength',1.)
    guidance=TypeGuidance(atoms,condition,[],planner_strength=0,style_strength=style_balance_strength,
        local_chord_strength=local_strength) if style_balance_strength or local_strength else None
    accepted=[];records=[]
    offset=song.first if intensity_curve is not None and intensity_curve.get('time_axis','chart_relative_v1')=='chart_relative_v1' else 0
    rests=[(r['start_sec']+offset,r['end_sec']+offset) for r in (intensity_curve or {}).get('rest_regions',[])]
    audio_start=max(Fraction(0),beat_at_second(0,song.tempo,song.first))
    audio_end=beat_at_second(song.audio.duration_sec,song.tempo,song.first)
    permitted=[];cursor=audio_start
    for lo,hi in sorted(rests):
        low=beat_at_second(lo,song.tempo,song.first);high=beat_at_second(hi,song.tempo,song.first)
        if high<=cursor:continue
        if low>cursor:permitted.append((cursor,min(low,audio_end)))
        cursor=max(cursor,high)
        if cursor>=audio_end:break
    if cursor<audio_end:permitted.append((cursor,audio_end))
    def valid(beat):
        second=seconds_at_beat(beat,song.tempo,song.first)
        return 0<=second<song.audio.duration_sec and not any(lo<=second<hi for lo,hi in rests)
    for span in song.spans:
        # Planner channel zero is a normalized event density (events / second
        # divided by eight).  Use it as a soft resource guard only: the model
        # still chooses every exact Fraction, but an under-trained checkpoint
        # cannot emit hundreds of self-invented events and monopolize the GPU.
        span_start_sec = float(seconds_at_beat(span.start,song.tempo,song.first)-song.first)
        span_end_sec = float(seconds_at_beat(span.end,song.tempo,song.first)-song.first)
        span_seconds = max(0.0, span_end_sec-span_start_sec)
        if max_events_per_span:
            span_event_budget = int(max_events_per_span)
        elif song.load_curve is not None and span_seconds > 0:
            curve_clock = np.asarray(song.load_curve_times_sec,dtype=np.float64)
            density = np.interp(
                np.linspace(span_start_sec,span_end_sec,max(2,int(math.ceil(span_seconds*4))+1)),
                curve_clock,np.asarray(song.load_curve)[:,0])
            clock_samples = np.linspace(span_start_sec,span_end_sec,len(density))
            integrate = getattr(np,'trapezoid',np.trapz)
            expected = float(integrate(np.maximum(density,0),clock_samples)*8.0)
            span_event_budget = max(8,int(math.ceil(expected*3.0+4.0)))
            span_event_budget = min(span_event_budget,192)
        else:
            span_event_budget = None
        max_bpm = max(float(t.bpm) for t in song.tempo)
        min_spacing_beats = Fraction(str(max(0.0,float(min_spacing_ms)))) * Fraction(str(max_bpm)) / 60000
        passes=0;contexts=[]
        current=span
        while True:
            source_spans=song.spans
            song.spans=[current if s.index==span.index else s for s in source_spans]
            try:
                example=input_example(song,span.index,accepted,condition=condition,codec=codec,
                    max_memory_tokens=model.config.max_memory_tokens,memory_strength=memory_strength,
                    history_before=(accepted[-1].beat+Fraction(1,10**128) if passes else None))
                example['skeleton']=torch.tensor([codec.pad_id])
                example['_tempo'],example['_first']=song.tempo,song.first
                example['_curve_profile']=intensity_curve
                example['_curve_options']=dict(strength=intensity_curve_strength,span=intensity_curve_span,mode=intensity_curve_mode)
                example['_position_ranges']=[(max(start,current.start)-current.start,min(stop,current.end)-current.start)
                    for start,stop in permitted if max(start,current.start)<min(stop,current.end)]
                contexts.append(example['meta'])
                curve,mask,clock=_curve_condition(song,current,intensity_curve,strength=intensity_curve_strength,
                    span_limit=intensity_curve_span,mode=intensity_curve_mode)
                if curve is not None:
                    example.update(load_curve=curve,load_curve_condition_mask=mask,load_curve_times_sec=clock)
                points,status=compose_span(model,codec,example,atoms,origin=current.start,end=current.end,
                    temperature=temperature,position_temperature=position_temperature,seed=seed+span.index*1009+passes,
                    device=device,history_last=accepted[-1].beat if accepted else None,valid_position=valid,
                    position_characters=position_characters,type_guidance=guidance,
                    max_events=span_event_budget,min_spacing_beats=min_spacing_beats)
            finally: song.spans=source_spans
            accepted.extend(points);passes+=1
            if status['closed']: break
            if not points: raise ValueError('Exact event decoder exhausted its resource budget without progress')
            # Keep the original rational origin on resource continuation.
            # Only the causal history cutoff moves; note times are untouched.
        row=dict(span=span.index,start=str(span.start),end=str(span.end),resource_passes=passes,
                 onsets=sum(span.start<=p.beat<span.end for p in accepted),contexts=contexts,status=status)
        records.append(row);print('form_arrange '+json.dumps(row),flush=True)
    if not accepted and permitted: raise ValueError('Exact event decoder emitted an empty chart; inspect event training rather than changing a frame threshold')
    quality=assess_event_chart(accepted,song.tempo,first=song.first,duration_sec=song.audio.duration_sec,target_condition=condition)
    quality['local_load']=local_load_statistics(accepted,song.tempo,song.first,target_chord_ratio=condition.style['chord_bias'])
    if not quality.get('syntax_valid'): raise ValueError('Exact event output failed syntax verification')
    blueprint={'schema':'maichart-rational-event-plan-v1','onsets':[dict(beat=str(p.beat),second=seconds_at_beat(p.beat,song.tempo,song.first)) for p in accepted],
               'selection':{'algorithm':'joint_exact_fraction_events_v1','frame_seconds':None,'candidate_position_whitelist':False,
                            'onset_threshold_used':False,'rhythm_snap_used':False,'explicit_all_rest':not bool(permitted)},'musical_units':'exact_quarter_fraction'}
    print('form_onsets '+json.dumps(blueprint['selection']),flush=True)
    return accepted,dict(blueprint=blueprint,arrangement=records,quality=quality,decoder_fallbacks=0,
                         memory_strength=memory_strength,type_guidance=guidance.report() if guidance else None,
                         intensity_curve={'applied':intensity_curve is not None,'guidance':'event-decoder local curve condition and explicit rests'})
