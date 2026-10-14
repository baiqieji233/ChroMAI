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
    def __init__(self,lower,upper,*,characters=24,valid_position=None):
        self.lower,self.upper = Fraction(lower),Fraction(upper)
        if self.upper <= self.lower or not 4 <= characters <= 128:
            raise ValueError('Invalid rational position bounds/budget')
        self.characters = characters
        self.valid_position = valid_position or (lambda _:True)

    def complete(self,text):
        try:
            beat = Fraction(text)
            return str(beat)==text and self.lower < beat < self.upper and self.valid_position(beat)
        except (ValueError,ZeroDivisionError): return False

    def prefix(self,text):
        if not text or len(text)>self.characters or any(c not in '0123456789/' for c in text) or text.count('/')>1:
            return False
        numerator,_,denominator = text.partition('/')
        if not numerator or (len(numerator)>1 and numerator[0]=='0'):
            return False
        n = int(numerator)
        remaining = self.characters-len(text)
        if '/' not in text:
            if self.complete(text): return True
            # An integer can still grow or turn into an arbitrary fraction.
            if n == 0: return False
            return remaining>=2 and Fraction(n,10**(remaining-1)-1)<self.upper
        if n==0 or denominator.startswith('0'):
            return False
        if not denominator: return remaining>0
        if self.complete(text): return True
        d = int(denominator)
        largest = d*10**remaining+10**remaining-1
        return Fraction(n,largest)<self.upper and Fraction(n,d)>self.lower and remaining>0

    def allowed(self,text,codec):
        ids = [codec.ids['P:'+c] for c in '0123456789/' if self.prefix(text+c)]
        if self.complete(text): ids.append(codec.note_id)
        return ids


def compose_span(model,codec,example,atoms,*,origin,end,condition=None,temperature=.7,
                 position_temperature=.2,seed=0,device='cpu',history_last=None,
                 valid_position=None,position_characters=24,type_guidance=None):
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
            valid_position=lambda b: b>=0 and (valid_position is None or valid_position(origin+b)))
        bias=None
        while len(ids)<limit-1:
            if state=='event':
                allowed=[codec.eos_id,codec.at_id]
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
                raise ValueError(f'No exact rational continuation for {state} prefix {text!r}; no timing approximation was applied')
            values=logits()[allowed]
            if state=='note' and bias is not None:
                hints=bias.biases(text,[None if t==codec.end_id else codec.tokens[t][2:] for t in allowed])
                values=values+torch.as_tensor(hints,device=values.device,dtype=values.dtype)
            temp=temperature if state=='note' else position_temperature
            chosen=int(values.argmax()) if temp==0 else int(torch.multinomial(torch.softmax(values/temp,dim=-1),1,generator=rng))
            token=allowed[chosen];ids.append(token)
            if state=='event':
                if token==codec.eos_id:
                    return points,{'closed':True,'reason':'eos','tokens':len(ids)}
                state,text='position',''
            elif state=='position':
                if token==codec.note_id:
                    position=origin+Fraction(text)
                    state,text='note',''
                    if type_guidance is not None and type_guidance.local_chords is not None:
                        type_guidance.local_chords.cues[position]=(float(seconds_at_beat(position,example['_tempo'],example['_first'])),1.)
                    bias=type_guidance.event(position) if type_guidance is not None else None
                else: text+=codec.tokens[token][2:]
            elif token==codec.end_id:
                points.append(EventPoint(position,text))
                if type_guidance is not None: type_guidance.commit(text,position)
                state,text='event',''
                position_grammar.lower=position-origin
            else: text+=codec.tokens[token][2:]
        return points,{'closed':False,'reason':'token_budget','tokens':len(ids), 'discarded_partial_event':state!='event'}


def generate_rational_chart(model,planner,codec,song,condition,atoms,*,temperature=.7,
        position_temperature=.2,seed=0,device='cpu',memory_strength=1.,intensity_curve=None,
        intensity_curve_strength=1.,intensity_curve_span=.5,intensity_curve_mode='follow',
        style_balance_strength=1.,position_characters=24,**unused_legacy_options):
    from maichart.form.type_guidance import TypeGuidance
    from maichart.generation.event_quality import assess_event_chart
    from maichart.form.local_load import local_load_statistics
    from maichart.form.curve import resample_load_curve
    from maichart.form.rational_audio import audio_queries
    from maichart.form.data import _form_descriptor
    audio,times,_ = audio_queries(song,song.spans[0].start,song.spans[-1].end)
    with torch.inference_mode():
        prediction=planner(audio[None].to(device),torch.from_numpy(_form_descriptor(song,0))[None].to(device),condition.to_vector(device=device)[None])
    if song.load_curve is None:
        song.load_curve_times_sec = (times-song.first).astype(np.float32)
        song.load_curve = prediction['load_curve'][0].sigmoid().cpu().numpy()
    local_strength=unused_legacy_options.get('local_chord_strength',1.)
    guidance=TypeGuidance(atoms,condition,[],planner_strength=0,style_strength=style_balance_strength,
        local_chord_strength=local_strength) if style_balance_strength or local_strength else None
    accepted=[];records=[]
    offset=song.first if intensity_curve is not None and intensity_curve.get('time_axis','chart_relative_v1')=='chart_relative_v1' else 0
    rests=[(r['start_sec']+offset,r['end_sec']+offset) for r in (intensity_curve or {}).get('rest_regions',[])]
    def valid(beat):
        second=seconds_at_beat(beat,song.tempo,song.first)
        return 0<=second<song.audio.duration_sec and not any(lo<=second<hi for lo,hi in rests)
    for span in song.spans:
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
                contexts.append(example['meta'])
                curve,mask,clock=_curve_condition(song,current,intensity_curve,strength=intensity_curve_strength,
                    span_limit=intensity_curve_span,mode=intensity_curve_mode)
                if curve is not None:
                    example.update(load_curve=curve,load_curve_condition_mask=mask,load_curve_times_sec=clock)
                points,status=compose_span(model,codec,example,atoms,origin=current.start,end=current.end,
                    temperature=temperature,position_temperature=position_temperature,seed=seed+span.index*1009+passes,
                    device=device,history_last=accepted[-1].beat if accepted else None,valid_position=valid,
                    position_characters=position_characters,type_guidance=guidance)
            finally: song.spans=source_spans
            accepted.extend(points);passes+=1
            if status['closed']: break
            if not points: raise ValueError('Exact event decoder exhausted its resource budget without progress')
            # Keep the original rational origin on resource continuation.
            # Only the causal history cutoff moves; note times are untouched.
        row=dict(span=span.index,start=str(span.start),end=str(span.end),resource_passes=passes,
                 onsets=sum(span.start<=p.beat<span.end for p in accepted),contexts=contexts,status=status)
        records.append(row);print('form_arrange '+json.dumps(row),flush=True)
    if not accepted: raise ValueError('Exact event decoder emitted an empty chart; inspect event training rather than changing a frame threshold')
    quality=assess_event_chart(accepted,song.tempo,first=song.first,duration_sec=song.audio.duration_sec,target_condition=condition)
    quality['local_load']=local_load_statistics(accepted,song.tempo,song.first,target_chord_ratio=condition.style['chord_bias'])
    if not quality.get('syntax_valid'): raise ValueError('Exact event output failed syntax verification')
    blueprint={'schema':'maichart-rational-event-plan-v1','onsets':[dict(beat=str(p.beat),second=seconds_at_beat(p.beat,song.tempo,song.first)) for p in accepted],
               'selection':{'algorithm':'joint_exact_fraction_events_v1','frame_seconds':None,'candidate_position_whitelist':False,
                            'onset_threshold_used':False,'rhythm_snap_used':False},'musical_units':'exact_quarter_fraction'}
    print('form_onsets '+json.dumps(blueprint['selection']),flush=True)
    return accepted,dict(blueprint=blueprint,arrangement=records,quality=quality,decoder_fallbacks=0,
                         memory_strength=memory_strength,type_guidance=guidance.report() if guidance else None,
                         intensity_curve={'applied':intensity_curve is not None,'guidance':'event-decoder local curve condition and explicit rests'})
