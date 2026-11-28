"""Continuous event language: absolute beat, exact phase, complete onset.

There is no musical block boundary in this language. EOS means end of chart.
Fractional phases come from explicit rational families and training data, never
from float-to-Fraction conversion of an estimated audio clock.
"""
from fractions import Fraction
from bisect import bisect_left,bisect_right
from collections import OrderedDict
from .codec import ComponentCodec, SPECIAL, VocabularyCoverageError, component_inventory, render_atom
from maichart.event_timing import EventPoint

POLICY='continuous_events_v1'
LEGACY='window_v1'


class StreamCodec(ComponentCodec):
    schema='composer_v2_absolute_events_v1'
    def __init__(self,tokens,templates,ticks_per_beat=48,window_beats=16):
        super().__init__(tokens,templates,ticks_per_beat,window_beats)
        self.at_id=self.ids['[AT]'];self.phase_id=self.ids['[PHASE]']
        self.event_end_id=self.ids['[END_EVENT]']
        self.phases={i:Fraction(t[2:]) for i,t in enumerate(tokens) if t.startswith('F:')}
        self._phases_by_time=sorted(self.phases,key=self.phases.__getitem__)
        self._phase_order=[self.phases[i] for i in self._phases_by_time]
        self._phase_ids=tuple(self.phases)
        self._phase_bounds=(self._phase_order[0],self._phase_order[-1]) if self.phases else None
        self.digits={self.ids['NUM:'+str(i)]:str(i) for i in range(10)}
        self._note_tokens = OrderedDict()

    def _encode_note_text(self, text):
        cached = self._note_tokens.get(text)
        if cached is not None:
            self._note_tokens.move_to_end(text)
            return cached
        tokens = tuple(token for atom in self.note_atoms(text) for token in self.encode_atom(atom))
        self._note_tokens[text] = tokens
        if len(self._note_tokens) > 8192:
            self._note_tokens.popitem(last=False)
        return tokens
    @classmethod
    def fit(cls,texts,ticks_per_beat=48,window_beats=16,phases=()):
        values, patterns = component_inventory(texts)
        values.update('P:'+str(i) for i in range(1,9))
        values.update('P:'+a+str(i) for a in 'ABDE' for i in range(1,9));values.add('P:C')
        values.update('NUM:'+c for c in '-/0123456789')
        values.update('V:'+c for c in '-+./:#0123456789');values.update(('D:B','D:R'))
        # These are available rhythmic positions, not compulsory grids or meter.
        denominators=set(range(1,33))|{48,64,96,128,192,int(ticks_per_beat)}
        fractions={Fraction(n,d) for d in denominators for n in range(d)}
        fractions.update(Fraction(p)%1 for p in phases)
        values.update('F:'+str(p) for p in fractions)
        special=(*SPECIAL,'[AT]','[PHASE]','[END_EVENT]')
        return cls([*special,*sorted(values-set(special))],sorted(patterns),ticks_per_beat,window_beats)
    def _time(self,value):
        value=Fraction(value)
        if value<0:raise ValueError('Continuous clock uses nonnegative absolute beats')
        integer=value.numerator//value.denominator;phase=value-integer
        token='F:'+str(phase)
        if token not in self.ids:raise VocabularyCoverageError(f'Unseen exact phase {phase}')
        return [self.at_id,*[self.ids['NUM:'+c] for c in str(integer)],self.phase_id,self.ids[token]]

    def encode_points(self,points,origin=0,region='target'):
        result=[]
        for point in sorted(points,key=lambda p:p.beat):
            result.extend(self._time(point.beat))
            result.extend(self._encode_note_text(point.note_text))
            result.append(self.event_end_id)
        return result
    def clock_states(self,ids,cursor=0,*,initial=(0.,'',False)):
        """Yield the clock and parser state after each token, including partial onsets."""
        current,integer,reading=initial
        for value in ids:
            value=int(value)
            if value==self.at_id:integer='';reading=True
            elif reading and value in self.digits:integer+=self.digits[value]
            elif value==self.phase_id:
                current=float(int(integer));reading=False
            elif value in self.phases:current=float(Fraction(int(integer))+self.phases[value])
            elif value==self.tgt_id:current=float(Fraction(cursor))
            elif value in (self.bos_id,self.ctx_id,self.carry_id,self.ref_id):current=0.
            yield current,integer,reading

    def token_clocks(self,ids,cursor=0):
        return [state[0] for state in self.clock_states(ids,cursor)]
    def decode_points(self,ids,origin=0):
        points=[];integer='';clock=None;atom=[];atoms=[];reading=False
        for value in ids:
            value=int(value);token=self.tokens[value]
            if value==self.at_id:
                if atoms or atom:raise ValueError('Missing complete onset terminator')
                integer='';reading=True
            elif reading and value in self.digits:integer+=self.digits[value]
            elif value==self.phase_id:reading=False
            elif value in self.phases:clock=Fraction(int(integer))+self.phases[value]
            elif value==self.end_note_id:atoms.append(render_atom(atom));atom=[]
            elif value==self.event_end_id:
                if clock is None or not atoms or atom:raise ValueError('Incomplete onset')
                points.append(EventPoint(clock,'/'.join(atoms)));atoms=[]
            elif token.startswith(('N:','P:','S:','D:','V:')) or token=='[END_DURATION]':atom.append(token)
        if atom or atoms or reading:raise ValueError('Truncated continuous event')
        return points
    def integer_bounds(self,after,end):
        """Integers that have at least one strictly future phase before audio end.

        Only the two boundary integers can lose phases to the time limits.
        Interior integers always contain the same nonempty phase vocabulary.
        """
        after=Fraction(after);end=Fraction(end)
        lo=max(0,after.numerator//after.denominator)
        hi=(end.numerator-1)//end.denominator
        if not self._phase_order or end<=after:return 1,0
        if bisect_right(self._phase_order,after-lo)==len(self._phase_order):lo+=1
        if bisect_left(self._phase_order,end-hi)==0:hi-=1
        if lo==hi and bisect_right(self._phase_order,after-lo)>=bisect_left(self._phase_order,end-lo):lo+=1
        return lo,hi

    def time_choices(self,prefix,after,end):
        """Finite decimal integer prefix plus a canonical exact phase."""
        after=Fraction(after);end=Fraction(end)
        lo,hi=self.integer_bounds(after,end)
        if not prefix:return (self.at_id,self.eos_id) if lo<=hi else (self.eos_id,)
        if prefix[0]==self.eos_id:return ()
        if self.phase_id in prefix:
            split=prefix.index(self.phase_id)
            if len(prefix)!=split+1:return ()
            whole=int(''.join(self.digits[i] for i in prefix[1:split]))
            if self._phase_bounds is None:return ()
            lower,upper=after-whole,end-whole
            if lower<self._phase_bounds[0] and self._phase_bounds[1]<upper:
                return self._phase_ids
            # At a boundary only a slice of the exact rational vocabulary is
            # legal. Preserve token-ID order, including equivalent spellings.
            start=bisect_right(self._phase_order,lower)
            stop=bisect_left(self._phase_order,upper)
            return tuple(sorted(self._phases_by_time[start:stop]))
        digits=''.join(self.digits[i] for i in prefix[1:])
        def possible(s):
            if len(s)>1 and s[0]=='0':return False
            n=int(s)
            for extra in range(len(str(hi))-len(s)+1):
                if s=='0' and extra:break
                if max(lo,n*10**extra)<=min(hi,(n+1)*10**extra-1):return True
            return False
        allowed=[i for i,c in self.digits.items() if possible(digits+c)]
        if digits and lo<=int(digits)<=hi:allowed.append(self.phase_id)
        return tuple(allowed)
