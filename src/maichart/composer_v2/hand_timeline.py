"""Causal facts for the hand readout, shared by training and streaming decode."""
from functools import lru_cache
from typing import NamedTuple

import numpy as np

from maichart.generation.muri.data import reference_data
from maichart.generation.muri.parser import _parse_one
from maichart.legacy_hands import LINK_SECONDS
from .codec import atom_components, explicit_note_atoms, render_atom

PADS = tuple(sorted([*'12345678', 'C', *(f'{z}{i}' for z in 'ABDE' for i in range(1,9))]))
PAD_ID = {pad:i for i,pad in enumerate(PADS,1)}
# Pointers are one-based token offsets into this exact candidate's prefix.
FIRST, SECOND, PREVIOUS, RETURN, LATEST, HOLD, HOLD_SLOT, HOLD_PAD, N_HOLD, WAIT, TRACE = range(11)
FAMILY = 11  # per-target validation strata; 1 + bits for Hold and Touch
WIDTH = FAMILY+5
EMPTY = (0,) * WIDTH


@lru_cache(maxsize=16384)
def describe(text,bpm):
    heads, spans = [], []
    for atom in explicit_note_atoms(text):
        parts = atom_components(atom)
        kind = parts[0][2:]
        pads = [t[2:] for t in parts if t.startswith('P:')]
        supported = kind in ('tap','hold','touch','touch_hold') and len(pads)==1 and pads[0] in PAD_ID
        heads.append(PAD_ID[pads[0]] if supported else 0)
        if kind in ('slide','hold','touch_hold'):
            try:
                markers = _parse_one(atom,1,1,0.,0,bpm,reference_data())
            except (ValueError, KeyError, IndexError, OverflowError):
                heads[-1] = 0
                continue
            for m in markers:
                if m['end_second'] <= 0:
                    continue
                if m['type'] in ('slide','wifi'):
                    spans.append((0,m['slide_trace_second'],m['end_second']))
                elif m['type'] in ('hold','touch_hold'):
                    spans.append((heads[-1],0.,m['end_second']))
    heads = tuple(sorted(heads)) if 0 not in heads and len(heads)<=2 and len(set(heads))==len(heads) else ()
    return heads, tuple(spans)


class Group(NamedTuple):
    pos: int
    second: float
    heads: tuple
    kinds: tuple


class Snapshot(NamedTuple):
    region: int = 0
    opened: int = -1
    valid: bool = False
    second: float = 0.
    bpm: float = 120.
    last: Group | None = None
    before: Group | None = None
    active: tuple = ()
    row: tuple = EMPTY


def _facts(t, last, active):
    row = list(EMPTY)
    if last and 0 <= t-last.second <= LINK_SECONDS:
        row[LATEST] = last.pos
    holds = []
    for pad,trace,end,pos,slot in active:
        if t >= end-1e-8:
            continue
        if pad:
            holds.append((pad,pos,slot))
        else:
            row[WAIT if t < trace-1e-8 else TRACE] += 1
    row[N_HOLD] = len(holds)
    if len(holds)==1:
        row[HOLD_PAD],row[HOLD],row[HOLD_SLOT] = holds[0]
    return row


class Timeline:
    def __init__(self, codec, clock):
        self.codec, self.clock = codec,clock
        self.states = []

    def extend(self, ids, retained=0, masked=frozenset()):
        codec = self.codec
        del self.states[retained:]
        s = self.states[-1] if retained else Snapshot()
        rows = []
        for i in range(retained,len(ids)):
            token = ids[i]
            if token in (codec.carry_id,codec.ctx_id,codec.ref_id,codec.tgt_id):
                # Reference exemplars run on a separate clock. Never install their hands.
                s = s._replace(region=token,opened=-1,valid=False,row=EMPTY)
            elif s.region != codec.ref_id:
                if token == codec.at_id:
                    s = s._replace(opened=i,valid=True,row=EMPTY)
                elif i in masked:
                    s = s._replace(valid=False,row=EMPTY)
                elif token in codec.phases and s.opened>=0:
                    integer=int(''.join(codec.digits[ids[j]] for j in range(s.opened+1,i-1)))
                    beat=integer+codec.phases[token]
                    t = self.clock.at(beat)
                    active = tuple(a for a in s.active if a[2]>t+1e-8)
                    s = s._replace(active=active,second=t,bpm=self.clock.bpm(beat),row=tuple(_facts(t,s.last,active)))
                elif token == codec.event_end_id and s.opened>=0:
                    if s.valid:
                        # Handle compact chord tokens and independent closures in canonical pad order.
                        atoms, parts = [], []
                        for j in range(s.opened+1,i):
                            name = codec.tokens[ids[j]]
                            if ids[j]==codec.end_note_id:
                                atoms.append(render_atom(parts))
                                parts = []
                            elif name.startswith(('N:','P:','S:','D:','V:')) or name=='[END_DURATION]':
                                parts.append(name)
                        heads, spans = describe('/'.join(atoms),s.bpm) if atoms and not parts else ((),())
                        t = s.second
                        carry = s.region == codec.carry_id
                        active = s.active + tuple((pad,t+trace,t+end,
                            i+1 if pad in heads and not carry else 0,heads.index(pad) if pad in heads else 0)
                            for pad,trace,end in spans)
                        held_pads={pad for pad,_,_ in spans if pad}
                        kinds=tuple(int(pad in held_pads) | (0 if PADS[pad-1].isdigit() else 2) for pad in heads)
                        group = Group(i+1,t,heads,kinds)
                        row = _facts(t,group if not carry else None,active)
                        row[:len(heads)] = heads
                        row[FAMILY:FAMILY+len(kinds)]=[1+k for k in kinds]
                        if len(kinds)==2:
                            row[FAMILY+3]=1+(kinds[0]|kinds[1])
                        if len(heads)==1 and s.last and len(s.last.heads)==1 and 0<=t-s.last.second<=LINK_SECONDS:
                            row[PREVIOUS] = s.last.pos
                            row[FAMILY+2]=1+(kinds[0]|s.last.kinds[0])
                            if (s.before and s.before.heads==heads and s.last.heads!=heads
                                    and 0<=s.last.second-s.before.second<=LINK_SECONDS):
                                row[RETURN] = s.before.pos
                                row[FAMILY+4]=1+(kinds[0]|s.last.kinds[0]|s.before.kinds[0])
                        s = s._replace(last=None if carry else group,before=s.last,active=active,row=tuple(row))
                    else:
                        s = s._replace(last=None,before=None,row=EMPTY)
                    s = s._replace(opened=-1)
            self.states.append(s)
            rows.append(s.row)
        return np.asarray(rows,dtype=np.int32).reshape(-1,WIDTH)
