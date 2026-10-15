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
                 max_denominator=4_096,min_spacing=Fraction(0),max_delta_denominator=None,
                 origin=Fraction(0),max_beat_denominator=None,delta_reference=None):
        self.lower,self.upper = Fraction(lower),Fraction(upper)
        self.origin = Fraction(origin)
        self.delta_reference = (Fraction(delta_reference) if delta_reference is not None
                                else self.lower if self.lower >= 0 else -self.origin)
        self.max_beat_denominator = max_beat_denominator
        if max_beat_denominator is not None and (isinstance(max_beat_denominator,bool)
                or not isinstance(max_beat_denominator,int) or max_beat_denominator < 1):
            raise ValueError('max_beat_denominator must be a positive integer or None')
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
        # An interval may have a much larger denominator than either absolute
        # position. No default interval-denominator cap: it creates holes
        # between otherwise legal positions. Explicit callers can still test
        # a restricted domain with the same completion/reachability predicate.
        self.max_delta_denominator = (None if max_delta_denominator is None
                                      else int(max_delta_denominator))
        if self.max_delta_denominator is not None and self.max_delta_denominator < 1:
            raise ValueError('max_delta_denominator must be positive or None')
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
                    and (self.max_beat_denominator is None
                         or (self.origin+beat).denominator <= self.max_beat_denominator)
                    and beat.numerator <= self.max_numerator
                    and beat-self.lower >= self.min_spacing
                    and (self.max_delta_denominator is None
                         or (beat-self.delta_reference).denominator <= self.max_delta_denominator)
                    and self.valid_position(beat))
        except (ValueError,ZeroDivisionError): return False

    def denominator_possible(self,n,lo,hi):
        for start,end in self.ranges:
            end=min(end,self.upper)
            start=max(start,self.lower+self.min_spacing)
            if end<=0 or end<=start: continue
            minimum=max(lo,1,(Fraction(n)/end).__floor__()+1)
            maximum=min(hi,self.max_denominator)
            if start>0: maximum=min(maximum,(Fraction(n)/start).__floor__())
            # Range reachability alone is insufficient: after an unusual
            # previous Fraction, every candidate here may violate the delta
            # denominator guard. Use the same final predicate as completion.
            for denominator in range(minimum, maximum+1):
                if self.complete(f'{n}/{denominator}'):
                    return True
        return False

    def numerator_possible(self,n,remaining):
        def interval_has_integer(low, high, start, end, den):
            first = max(low, (start * den).__ceil__(), (self.lower * den).__floor__()+1)
            last = min(high, (end * den).__ceil__() - 1)
            if first > last:
                return False
            # Delta denominator feasibility repeats every ``den`` numerators.
            # Test at most one period; no musical grid is introduced.
            for numerator in range(first, min(last, first+den-1)+1):
                text = str(numerator) if den == 1 else f'{numerator}/{den}'
                if self.complete(text):
                    return True
            return False

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
                if start>0:den_high=min(den_high,(Fraction(high)/start).__floor__())
                if den_low>den_high:continue
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


def _spacing_beat_gap(previous, tempo, seconds):
    """Integrate a time-distance guard across BPM boundaries using Fractions."""
    previous = Fraction(previous)
    remaining = Fraction(str(seconds))
    cursor = previous
    active = tempo[0]
    for entry in tempo:
        if entry.beat <= cursor:
            active = entry
            continue
        duration = (Fraction(entry.beat) - cursor) * 60 / Fraction(str(active.bpm))
        if remaining <= duration:
            return cursor - previous + remaining * Fraction(str(active.bpm)) / 60
        remaining -= duration
        cursor, active = Fraction(entry.beat), entry
    return cursor - previous + remaining * Fraction(str(active.bpm)) / 60


def compose_span(model,codec,example,atoms,*,origin,end,condition=None,temperature=.7,
                 position_temperature=.2,seed=0,device='cpu',history_last=None,
                 valid_position=None,position_characters=24,type_guidance=None,
                 max_events=None,min_spacing_beats=Fraction(0),min_spacing_seconds=0.,
                 position_complexity=0.2,early_eos_penalty=0.,position_beam_size=1,
                 max_beat_denominator=0,max_interval_denominator=0):
    """Choose timing and notes jointly. At a resource cap retain completed events."""
    grammar = _NoteGrammar.from_atoms(atoms)
    if grammar is None: raise ValueError('Checkpoint note grammar is empty')
    if not all(math.isfinite(float(t)) and t>=0 for t in (temperature,position_temperature)):
        raise ValueError('Decoder temperatures must be finite and nonnegative')
    if not math.isfinite(position_complexity) or not 0<=position_complexity<=2:
        raise ValueError('position_complexity must be between 0 and 2')
    if not math.isfinite(early_eos_penalty) or not 0<=early_eos_penalty<=8:
        raise ValueError('early_eos_penalty must be between 0 and 8')
    if not math.isfinite(float(min_spacing_seconds)) or min_spacing_seconds < 0:
        raise ValueError('min_spacing_seconds must be finite and nonnegative')
    if isinstance(position_beam_size,bool) or not isinstance(position_beam_size,int) or not 1<=position_beam_size<=8:
        raise ValueError('position_beam_size must be an integer between 1 and 8')
    for name,value in (('max_beat_denominator',max_beat_denominator),
                       ('max_interval_denominator',max_interval_denominator)):
        if isinstance(value,bool) or not isinstance(value,int) or not 0<=value<=4096:
            raise ValueError(f'{name} must be an integer between 0 and 4096')
    batch = to_device(collate_form([example]),device)
    ids = [codec.bos_id]
    points = []
    lower = Fraction(history_last)-origin if history_last is not None else Fraction(-1)
    limit = model.config.max_tokens
    state,text,position = 'event','',None
    header_start=0;normalizations=0;recoveries=0;position_tried={};initial_eos_margin=None
    # Distinguish an intentional/forced EOS from a span that merely has no
    # permitted audio interval.  ``has_allowed_positions`` in the outer
    # report is not sufficient for this decision.
    event_start_allowed=False
    chars = [(i,t[2:]) for i,t in enumerate(codec.tokens) if t.startswith('N:')]
    rng = torch.Generator(device=device).manual_seed(seed)
    model.eval()
    with torch.inference_mode():
        memory,mask = model.encode(batch)
        cache=None;consumed=0
        from maichart.form.rhythm_decode import DecodeGeometry
        geometry=DecodeGeometry(model,codec,origin)
        def logits():
            nonlocal cache,consumed
            while consumed<len(ids):
                output,cache=model.decode_step(torch.tensor([[ids[consumed]]],device=device),consumed,memory,mask,cache,
                    **geometry.kwargs(ids,consumed,device))
                consumed+=1
            return output[0,-1]
        def rewind(length):
            nonlocal ids,cache,consumed
            ids=ids[:length]
            # Replay the retained last token to recover its next-token logits;
            # an old output tensor belongs to the discarded prefix.
            retained=min(consumed,max(0,length-1))
            if hasattr(model,'rewind_decode_cache'):
                cache=model.rewind_decode_cache(cache,retained)
                consumed=retained
            else:
                cache=None;consumed=0
        def position_cost(prefix):
            # A soft cost on *representations*, with no selected denominator
            # list. Strong learned evidence can still choose 37/997, etc.
            try:
                value=Fraction(prefix)
                delta=value-position_grammar.lower
                return min(12.,math.log2(value.denominator)+.5*math.log2(delta.denominator))
            except (ValueError,ZeroDivisionError):
                return 0.
        def spacing_gap(previous):
            temporal = (_spacing_beat_gap(previous, example['_tempo'], min_spacing_seconds)
                        if min_spacing_seconds and previous is not None else Fraction(0))
            return max(Fraction(min_spacing_beats), temporal)
        position_grammar = PositionGrammar(lower,end-origin,characters=position_characters,
            valid_position=lambda b: b>=0 and (valid_position is None or valid_position(origin+b)),
            allowed_ranges=example.get('_position_ranges'),min_spacing=spacing_gap(history_last),
            origin=origin,max_beat_denominator=max_beat_denominator or None,
            max_delta_denominator=max_interval_denominator or None,
            delta_reference=Fraction(history_last)-origin if history_last is not None else -origin)
        def event_bias(beat):
            if type_guidance is None:
                return None
            if type_guidance.local_chords is not None:
                second=float(seconds_at_beat(beat,example['_tempo'],example['_first']))
                profile=example.get('_curve_profile');factor=1.
                if profile is not None:
                    from maichart.form.generate import _curve_factor
                    clock=second-example['_first'] if profile.get('time_axis','chart_relative_v1')=='chart_relative_v1' else second
                    factor=_curve_factor(profile,clock,**example['_curve_options'])
                type_guidance.local_chords.cues[beat]=(second,factor)
            return type_guidance.event(beat)
        bias=None
        while len(ids)<limit-1:
            if state=='position' and not text and position_beam_size>1:
                from maichart.form.rational_position_search import search_position
                root_logits=logits()
                selected,search=search_position(model,codec,position_grammar,root_logits,cache,memory,mask,
                    consumed=consumed,width=position_beam_size,complexity=position_complexity,
                    timing_features=geometry.search_features(device))
                if selected is None:
                    return points,{'closed':False,'reason':'position_search_exhausted',
                        'tokens':len(ids),'discarded_partial_event':True,
                        'position_recoveries':recoveries,
                        'event_start_allowed':event_start_allowed}
                position=origin+Fraction(selected)
                # Search has its own causal cache. Commit only the selected
                # canonical spelling, replaying through the normal decoder.
                ids=ids[:header_start]+codec.position(position-origin)
                if hasattr(model,'rewind_decode_cache'):
                    cache=model.rewind_decode_cache(cache,header_start);consumed=header_start
                else:cache=None;consumed=0
                state,text='note',''
                bias=event_bias(position)
                continue
            if state=='event':
                if max_events is not None and len(points) >= max_events:
                    return points,{'closed':False,'reason':'event_budget','tokens':len(ids),
                                   'max_events':int(max_events),
                                   'event_start_allowed':event_start_allowed}
                allowed=[codec.eos_id]
                # Checking only the interval end is insufficient when the
                # previous event is very close to the span boundary: there
                # may be no representable Fraction left even though ``b`` is
                # technically greater than the lower bound.  Ask the grammar
                # whether at least one first numerator digit has a reachable
                # continuation before exposing <at> to the model.
                event_start_allowed=bool(position_grammar.allowed('',codec))
                if event_start_allowed:
                    allowed.append(codec.at_id)
                if len(ids)+position_characters+MAX_NOTE_CHARACTERS+4>limit and points:
                    return points,{'closed':False,'reason':'token_budget','tokens':len(ids),
                                   'event_start_allowed':event_start_allowed}
            elif state=='position':
                allowed=[token for token in position_grammar.allowed(text,codec)
                         if token not in position_tried.get(text,set())]
            else:
                allowed=[i for i,c in chars if grammar.can_continue(text+c)]
                if grammar.is_complete(text): allowed.append(codec.end_id)
                if len(text)>=MAX_NOTE_CHARACTERS:
                    if not grammar.is_complete(text):
                        raise ValueError('Rational decoder exceeded note text budget without a complete configuration')
                    allowed=[codec.end_id]
            if not allowed:
                if state == 'position' and text and recoveries < 64:
                    # Reconsider only the unfinished position, retaining all
                    # committed notes. Never turn a numeric dead end into a
                    # successful EOS for the remaining musical region.
                    previous=text[:-1]
                    position_tried.setdefault(previous,set()).add(codec.ids['P:'+text[-1]])
                    text=previous;rewind(header_start+1+len(text));recoveries+=1
                    continue
                if state == 'position':
                    return points,{'closed':False,'reason':'position_search_exhausted',
                                   'tokens':len(ids),'discarded_partial_event':True,
                                   'position_recoveries':recoveries,
                                   'event_start_allowed':event_start_allowed}
                raise ValueError(f'No exact rational continuation for {state} prefix {text!r}; no timing approximation was applied')
            values=logits()[allowed]
            if state=='event' and not points and initial_eos_margin is None and codec.at_id in allowed:
                initial_eos_margin=float((values[allowed.index(codec.eos_id)]-
                                          values[allowed.index(codec.at_id)]).detach().cpu())
            if state=='event' and codec.at_id in allowed and early_eos_penalty:
                # Optional bounded EOS logit penalty. Explicit rest holes and
                # exhausted ranges still allow EOS only, and remain untouched.
                values=values.clone()
                values[allowed.index(codec.eos_id)]-=early_eos_penalty
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
                if position_complexity:
                    # Potential differences avoid rewarding a long spelling
                    # once per character, and price relative IOIs as well as
                    # absolute positions. No emitted position is moved.
                    parent=position_cost(text)
                    costs=[(parent if t==codec.note_id else position_cost(text+codec.tokens[t][2:]))
                           for t in allowed]
                    values=values-position_complexity*torch.as_tensor(
                        [cost-parent for cost in costs],device=values.device,dtype=values.dtype)
            if state=='note' and bias is not None:
                hints=bias.biases(text,[None if t==codec.end_id else codec.tokens[t][2:] for t in allowed])
                values=values+torch.as_tensor(hints,device=values.device,dtype=values.dtype)
            temp=temperature if state=='note' else position_temperature
            chosen=int(values.argmax()) if temp==0 else int(torch.multinomial(torch.softmax(values/temp,dim=-1),1,generator=rng))
            token=allowed[chosen];ids.append(token)
            if state=='event':
                if token==codec.eos_id:
                    return points,{'closed':True,'reason':'eos','tokens':len(ids),
                                   'eos_forced':not event_start_allowed,
                                   'event_start_allowed':event_start_allowed,
                                   'equivalent_fraction_normalizations':normalizations,
                                   'position_recoveries':recoveries,'initial_eos_minus_at':initial_eos_margin}
                header_start=len(ids)-1
                position_tried={}
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
                    bias=event_bias(position)
                else: text+=codec.tokens[token][2:]
            elif token==codec.end_id:
                points.append(EventPoint(position,text))
                if type_guidance is not None: type_guidance.commit(text,position)
                state,text='event',''
                position_grammar.lower=position-origin
                position_grammar.delta_reference=position-origin
                position_grammar.min_spacing=spacing_gap(position)
                position_grammar._prefix_cache.clear()
                position_grammar._allowed_cache.clear()
            else: text+=codec.tokens[token][2:]
        return points,{'closed':False,'reason':'token_budget','tokens':len(ids), 'discarded_partial_event':state!='event'}


def generate_rational_chart(model,planner,codec,song,condition,atoms,*,temperature=.7,
        position_temperature=.2,seed=0,device='cpu',memory_strength=1.,intensity_curve=None,
        intensity_curve_strength=1.,intensity_curve_span=.5,intensity_curve_mode='follow',
        style_balance_strength=1.,position_characters=24,max_events_per_span=0,
        min_spacing_ms=20.0,position_complexity=.2,early_eos_penalty=0.,position_beam_size=1,
        max_beat_denominator=96,max_interval_denominator=96,
        rational_grid_snap_ms=8.0,rational_grid_divisions=(4,6,8,12,16,24,32,48,64,96),
        **unused_legacy_options):
    from maichart.form.type_guidance import TypeGuidance
    from maichart.generation.event_quality import assess_event_chart
    from maichart.form.local_load import local_load_statistics
    from maichart.form.curve import resample_load_curve, load_curve_bin_centers
    from maichart.form.rational_audio import planner_audio_queries, beat_at_second
    from maichart.form.data import _form_descriptor
    from maichart.form.rational_grid import (normalize_rational_grid_divisions,
                                              simplify_rational_events)
    rational_grid_divisions = normalize_rational_grid_divisions(rational_grid_divisions)
    if not math.isfinite(float(rational_grid_snap_ms)) or not 0 <= float(rational_grid_snap_ms) <= 50:
        raise ValueError('rational_grid_snap_ms must be in [0, 50]')
    # Keep the nonuniform/native clock, but use the same bounded evidence
    # budget as training.  Passing every transient in a long song (often many
    # thousands of MP3 decoder peaks) makes each tiny autoregressive decoder
    # step attend to an enormous memory and leaves CUDA under-occupied.  This
    # selects the strongest native attacks; it does not quantize or whitelist
    # the Fraction positions the decoder may emit.
    audio,times,_ = planner_audio_queries(song)
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
    accepted=[];records=[];grid_simplification=[]
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
        # Planner channel zero is a normalized head density (heads / second
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
        min_spacing_seconds = max(0.0, float(min_spacing_ms)) / 1000.0
        passes=0;contexts=[];search_failures=0;budget_expansions=0
        current=span
        while True:
            source_spans=song.spans
            song.spans=[current if s.index==span.index else s for s in source_spans]
            try:
                example=input_example(song,span.index,accepted,condition=condition,codec=codec,
                    max_memory_tokens=model.config.max_memory_tokens,memory_strength=memory_strength,
                    history_before=(accepted[-1].beat+Fraction(1,10**128)
                        if passes and accepted and accepted[-1].beat >= span.start else None))
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
                    if song.rhythm_geometry_profile != 'none':
                        from maichart.form.data import curve_geometry
                        example['load_curve_geometry']=curve_geometry(song,current.start,clock)
                points,status=compose_span(model,codec,example,atoms,origin=current.start,end=current.end,
                    temperature=temperature,position_temperature=position_temperature,seed=seed+span.index*1009+passes,
                    device=device,history_last=accepted[-1].beat if accepted else None,valid_position=valid,
                    position_characters=position_characters,type_guidance=guidance,
                    max_events=(None if span_event_budget is None else max(0,span_event_budget-
                        sum(span.start<=p.beat<span.end for p in accepted))),
                    min_spacing_seconds=min_spacing_seconds,
                    position_complexity=position_complexity,early_eos_penalty=early_eos_penalty,
                    position_beam_size=position_beam_size,
                    max_beat_denominator=max_beat_denominator,max_interval_denominator=max_interval_denominator)
            finally: song.spans=source_spans
            accepted.extend(points);passes+=1
            # Apply the optional near-grid cleanup before the next span uses
            # this history.  It does not restrict arbitrary Fractions: only
            # locally-supported positions within the configured time
            # tolerance are moved, so genuine unusual rhythms remain intact.
            if rational_grid_snap_ms:
                accepted, grid_stats = simplify_rational_events(
                    accepted, song.tempo, song.first,
                    tolerance_ms=rational_grid_snap_ms,
                    divisions=rational_grid_divisions,
                    min_spacing_ms=min_spacing_ms,
                    end_beat=song.spans[-1].end,
                    valid_position=valid)
                grid_simplification.append(dict(span=span.index, **grid_stats))
            if status['closed']: break
            if status['reason']=='event_budget':
                if max_events_per_span or budget_expansions >= 3 or span_event_budget >= 768:
                    status.update(incomplete=True,
                        budget_policy='explicit_user_cap' if max_events_per_span else 'automatic_cap_reached',
                        budget_expansions=budget_expansions)
                    break
                budget_expansions += 1
                old_budget = span_event_budget
                span_event_budget = min(768, max(old_budget + 16, old_budget * 2))
                print('form_event_budget '+json.dumps(dict(span=span.index,
                    previous_budget=old_budget,new_budget=span_event_budget,
                    expansion=budget_expansions)),flush=True)
                continue
            if status['reason']=='position_search_exhausted':
                search_failures+=1
                print('form_position_recovery '+json.dumps(dict(span=span.index,attempt=search_failures,
                    retained_events=len(points),reason=status['reason'],position_recoveries=status.get('position_recoveries',0))),flush=True)
                if search_failures >= 3:
                    # Report incomplete coverage instead of silently claiming
                    # the model chose an intentional musical rest.
                    status.update(retries=search_failures,incomplete=True)
                    break
                continue
            if not points: raise ValueError('Exact event decoder exhausted its resource budget without progress')
            # Keep the original rational origin on resource continuation.
            # Only the causal history cutoff moves; note times are untouched.
        row=dict(span=span.index,start=str(span.start),end=str(span.end),resource_passes=passes,
                 onsets=sum(span.start<=p.beat<span.end for p in accepted),contexts=contexts,status=status,
                 has_allowed_positions=any(max(a,span.start)<min(b,span.end) for a,b in permitted))
        records.append(row);print('form_arrange '+json.dumps(row),flush=True)
        if status.get('incomplete'):
            print('form_warning '+json.dumps(dict(
                message='生成区域因资源限制未覆盖完成；保留已有事件，不将剩余空白视为模型主动休止。',
                span=span.index,reason=status['reason'],budget_policy=status.get('budget_policy')),
                ensure_ascii=False),flush=True)
    if not accepted and permitted: raise ValueError('Exact event decoder emitted an empty chart; inspect event training rather than changing a frame threshold')
    quality=assess_event_chart(accepted,song.tempo,first=song.first,duration_sec=song.audio.duration_sec,target_condition=condition)
    quality['local_load']=local_load_statistics(accepted,song.tempo,song.first,target_chord_ratio=condition.style['chord_bias'])
    if not quality.get('syntax_valid'): raise ValueError('Exact event output failed syntax verification')
    incomplete=[record['span'] for record in records if record['status'].get('incomplete')]
    from maichart.form.rational_diagnostics import rational_diagnostics
    diagnostics=rational_diagnostics(accepted,records,song.tempo,song.first,song.audio.duration_sec)
    diagnostics['near_grid_simplification'] = {
        'enabled': bool(rational_grid_snap_ms),
        'tolerance_ms': float(rational_grid_snap_ms),
        'divisions': list(rational_grid_divisions),
        'changed': sum(item.get('changed', 0) for item in grid_simplification),
        'maximum_shift_ms': max((item.get('maximum_shift_ms', 0.) for item in grid_simplification), default=0.),
        'span_records': grid_simplification[-32:],
    }
    print('form_rational_diagnostics '+json.dumps(diagnostics),flush=True)
    blueprint={'schema':'maichart-rational-event-plan-v1','onsets':[dict(beat=str(p.beat),second=seconds_at_beat(p.beat,song.tempo,song.first)) for p in accepted],
               'selection':{'algorithm':'joint_exact_fraction_events_v1','frame_seconds':None,'candidate_position_whitelist':False,
                         'onset_threshold_used':False,'rhythm_snap_used':False,
                         'near_grid_simplification_used':bool(rational_grid_snap_ms),
                         'explicit_all_rest':not bool(permitted),
                            'incomplete_spans':incomplete},'musical_units':'exact_quarter_fraction'}
    print('form_onsets '+json.dumps(blueprint['selection']),flush=True)
    return accepted,dict(blueprint=blueprint,arrangement=records,quality=quality,decoder_fallbacks=0,
                         rational_diagnostics=diagnostics,
                         memory_strength=memory_strength,type_guidance=guidance.report() if guidance else None,
                         intensity_curve={'applied':intensity_curve is not None,'guidance':'event-decoder local curve condition and explicit rests'})
