"""Shared read/write components; time is absolute within a stable beat window.

Grammar templates constrain syntax, not musical density or repetition. Exact
out-of-grid times are retained in training; rare/old context clocks use a
numeric escape instead of silently moving or deleting an authored event.
"""
from fractions import Fraction
import re
from collections import defaultdict
from functools import lru_cache

from maichart.event_timing import EventPoint
from maichart.composer.event_set_codec import _atoms
from maichart.notes import parse_timing_point_notes
from maichart.timing import RawTimingPoint
from maichart.duration import parse_duration_expr

SPECIAL=('[PAD]','[BOS]','[CTX]','[REF]','[CARRY]','[TGT]','[EOS]','[END_NOTE]','[TIME_EXT]','[TIME_END]','[END_DURATION]')
KINDS=('tap','hold','slide','touch','touch_hold')
LEX=re.compile(r'\[[^\]]*\]|[ABDE][1-8]|C[1-8]?|pp|qq|[1-8]|.')


@lru_cache(maxsize=65536)
def atom_components(atom):
    atom=re.sub(r'C[1-8]', 'C',atom)
    parsed=parse_timing_point_notes(RawTimingPoint(index=0,raw=atom,note_text=atom,
        beat=Fraction(0),tick=0,time_sec=0.,division=4,bpm=120.))
    if parsed.unknown_tokens or not parsed.notes:
        raise ValueError(f'v2 cannot silently encode unknown syntax: {atom}')
    kind=parsed.notes[0].note_type
    if kind not in KINDS:raise ValueError(f'Unsupported v2 component kind: {kind}')
    result=['N:'+kind]
    for piece in LEX.findall(atom):
        if piece.startswith('['):
            value=parse_duration_expr(piece)
            grid=value.kind=='grid_fraction' and value.beats is not None
            body=str(value.beats) if grid else piece[1:-1]
            result.extend(['D:B' if grid else 'D:R',*('V:'+c for c in body),'[END_DURATION]'])
        elif re.fullmatch(r'[1-8]|[ABDE][1-8]|C',piece):result.append('P:'+piece)
        else:result.append('S:'+piece)
    return result+['[END_NOTE]']


def render_atom(tokens):
    pieces=[];duration=None;body=''
    for token in tokens:
        if token in ('D:B','D:R'):duration=token;body='';continue
        if token.startswith('V:'):body+=token[2:];continue
        if token=='[END_DURATION]':
            if duration=='D:B':
                value=Fraction(body);pieces.append(f'[{4*value.denominator}:{value.numerator}]')
            elif duration=='D:R':pieces.append('['+body+']')
            else:raise ValueError('Duration terminator without duration')
            duration=None;continue
        if token.startswith(('N:', '[')):continue
        if token.startswith(('P:','S:')):pieces.append(token[2:])
        else:raise ValueError('Unknown component '+token)
    if duration is not None:raise ValueError('Truncated duration')
    return ''.join(pieces)


def template(tokens):
    compressed=[];duration=False
    for token in tokens:
        if token in ('D:B','D:R'):duration=True;compressed.append('<duration>')
        elif token=='[END_DURATION]':duration=False
        elif not duration:compressed.append(token)
    return tuple('<button>' if t.startswith('P:') and t[2:].isdigit() else
        '<touch>' if t.startswith('P:') else t for t in compressed)


def _duration_prefix(kind,body):
    """Lexical validity only; native physics remains the final duration check."""
    if kind=='D:B':return re.fullmatch(r'[+-]?\d*(?:/(?:[1-9]\d*)?)?',body) is not None
    number=r'[+-]?\d*(?:\.\d*)?'
    # Seconds, explicit wait##travel, and BPM-qualified native durations.
    return any(re.fullmatch(pattern,body) is not None for pattern in (
        '#'+number,number+'(?:#(?:#'+number+'|'+number+'(?::[+-]?\d*)?)?)?'))


def _duration_complete(kind,body):
    try:
        if kind=='D:B':Fraction(body);return True
        return (parse_duration_expr('['+body+']').kind!='unknown'
            or re.fullmatch(r'[+]?[0-9]+(?:\.[0-9]+)?#[1-9][0-9]*:[+-]?[0-9]+',body) is not None)
    except (ValueError,ZeroDivisionError):return False


class ComponentCodec:
    schema='composer_v2_components_v2'
    def __init__(self,tokens,templates,ticks_per_beat=48,window_beats=16):
        self.tokens=list(tokens);self.ids={t:i for i,t in enumerate(tokens)}
        self.templates=tuple(tuple(x) for x in templates)
        self.ticks_per_beat=int(ticks_per_beat);self.window_beats=int(window_beats)
        for name in ('pad','bos','ctx','ref','carry','tgt','eos'):
            setattr(self,name+'_id',self.ids['['+name.upper()+']'])
        self.end_note_id=self.ids['[END_NOTE]'];self.vocab_size=len(self.tokens)
        self.time_values={i:Fraction(t[2:]) for i,t in enumerate(self.tokens) if t.startswith('T:')}
        self.domains={
            '<button>':{i for i,t in enumerate(tokens) if t.startswith('P:') and t[2:].isdigit()},
            '<touch>':{i for i,t in enumerate(tokens) if t.startswith('P:') and not t[2:].isdigit()},
            '<duration>':{i for i,t in enumerate(tokens) if t.startswith('D:')}}
        self.kind_ids={self.ids['N:'+k] for k in KINDS if 'N:'+k in self.ids}
        self._allowed_cache={}

    @classmethod
    def fit(cls,texts,ticks_per_beat=48,window_beats=16):
        values=set();patterns=set()
        for text in texts:
            for atom in _atoms(text):
                sequence=atom_components(atom);values.update(sequence);patterns.add(template(sequence))
        values.update('P:'+str(i) for i in range(1,9))
        values.update('P:'+area+str(i) for area in 'ABDE' for i in range(1,9));values.add('P:C')
        values.update('NUM:'+c for c in '-/0123456789')
        values.update('V:'+c for c in '-+./:#0123456789')
        values.update(('D:B','D:R'))
        values.update('T:'+str(Fraction(i,ticks_per_beat)) for i in range(-window_beats*ticks_per_beat,window_beats*ticks_per_beat))
        return cls([*SPECIAL,*sorted(values-set(SPECIAL))],sorted(patterns),ticks_per_beat,window_beats)

    def configure_times(self,relative_times):
        additions=sorted({'T:'+str(Fraction(x)) for x in relative_times if -self.window_beats<=Fraction(x)<self.window_beats}-self.ids.keys())
        if additions:self.__init__(self.tokens+additions,self.templates,self.ticks_per_beat,self.window_beats)

    def encode_atom(self,text):
        parts=atom_components(text)
        try:return [self.ids[t] for t in parts]
        except KeyError as exc:raise ValueError(f'Component outside checkpoint vocabulary: {exc}') from exc

    def _time(self,value):
        token='T:'+str(value)
        if token in self.ids:return [self.ids[token]]
        return [self.ids['[TIME_EXT]'],*[self.ids['NUM:'+c] for c in str(value)],self.ids['[TIME_END]']]

    def encode_points(self,points,origin,region='target'):
        origin=Fraction(origin);result=[]
        for point in sorted(points,key=lambda x:x.beat):
            result.extend(self._time(point.beat-origin))
            for atom in _atoms(point.note_text):result.extend(self.encode_atom(atom))
        return result

    def token_clocks(self,ids):
        clock=Fraction(0);numeric=None;result=[]
        for value in ids:
            token=self.tokens[int(value)]
            if token.startswith('T:'):clock=Fraction(token[2:])
            elif token=='[TIME_EXT]':numeric=''
            elif numeric is not None and token.startswith('NUM:'):numeric+=token[4:]
            elif token=='[TIME_END]':clock=Fraction(numeric);numeric=None
            elif token in ('[BOS]','[CTX]','[REF]','[CARRY]','[TGT]'):clock=Fraction(0)
            result.append(float(clock))
        return result

    def decode_points(self,ids,origin=0):
        clock=None;numeric=None;atom=[];events=defaultdict(list)
        for value in ids:
            token=self.tokens[int(value)]
            if token.startswith('T:'):clock=Fraction(token[2:])
            elif token=='[TIME_EXT]':numeric=''
            elif numeric is not None and token.startswith('NUM:'):numeric+=token[4:]
            elif token=='[TIME_END]':clock=Fraction(numeric);numeric=None
            elif token=='[END_NOTE]':
                if clock is None or not atom:raise ValueError('Incomplete v2 note')
                events[Fraction(origin)+clock].append(render_atom(atom));atom=[]
            elif token.startswith(('N:','P:','S:','D:','V:')) or token=='[END_DURATION]':atom.append(token)
        if atom or numeric is not None:raise ValueError('Truncated v2 component sequence')
        return [EventPoint(beat,'/'.join(atoms)) for beat,atoms in sorted(events.items())]

    def allowed_atom_next(self,prefix):
        key=tuple(prefix)
        if key in self._allowed_cache:return self._allowed_cache[key]
        # A variable-length numeric value occupies one duration grammar field.
        collapsed=[];kind=None;body=''
        for value in key:
            token=self.tokens[value]
            if token in ('D:B','D:R'):kind=token;body=''
            elif kind is not None and token.startswith('V:'):body+=token[2:]
            elif token=='[END_DURATION]' and kind is not None:
                if not _duration_complete(kind,body):return ()
                collapsed.append('<duration>');kind=None
            else:collapsed.append(value)
        if kind is not None:
            answer=[self.ids['V:'+c] for c in '-+./:#0123456789' if _duration_prefix(kind,body+c)]
            if _duration_complete(kind,body):answer.append(self.ids['[END_DURATION]'])
            return tuple(sorted(answer))
        allowed=set()
        for pattern in self.templates:
            if len(collapsed)>=len(pattern):continue
            if all(value in self.domains[field] if field in self.domains else value==self.ids[field]
                   for value,field in zip(collapsed,pattern) if value!='<duration>') and all(
                       field=='<duration>' for value,field in zip(collapsed,pattern) if value=='<duration>'):
                field=pattern[len(collapsed)];allowed.update(self.domains[field] if field in self.domains else (self.ids[field],))
        answer=tuple(sorted(allowed))
        if len(self._allowed_cache)>8192:self._allowed_cache.clear()
        self._allowed_cache[key]=answer
        return answer

    def allowed_next(self,target_ids,end_beat=None):
        end=self.window_beats if end_beat is None else Fraction(end_beat)
        last=None;prefix=[];need_note=False
        for value in target_ids:
            value=int(value)
            if value in self.time_values:last=self.time_values[value];prefix=[];need_note=True
            elif value==self.end_note_id:prefix=[];need_note=False
            elif value in (self.tgt_id,self.bos_id):continue
            else:prefix.append(value)
        if prefix:return self.allowed_atom_next(prefix)
        if need_note:return tuple(sorted(self.kind_ids))
        result={self.eos_id}
        if last is not None:result.update(self.kind_ids)
        result.update(i for i,beat in self.time_values.items() if 0<=beat<end and (last is None or beat>last))
        return tuple(sorted(result))

    def to_dict(self):return dict(schema=self.schema,tokens=self.tokens,templates=self.templates,ticks_per_beat=self.ticks_per_beat,window_beats=self.window_beats)
    @classmethod
    def from_dict(cls,value):
        from .stream_codec import StreamCodec
        if value.get('schema')==StreamCodec.schema:
            return StreamCodec(value['tokens'],value['templates'],value['ticks_per_beat'],value.get('window_beats',16))
        if value.get('schema')!=cls.schema:raise ValueError('Incompatible v2 component vocabulary')
        return cls(value['tokens'],value['templates'],value['ticks_per_beat'],value['window_beats'])
