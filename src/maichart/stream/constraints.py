"""Explicit basic configuration constraints, independent of soft style bias.

This checks button-head count and definite Hold occupancy only. It makes no
claim about Slide geometry, travel speed, Touch reachability, or playability.
"""
from __future__ import annotations

from functools import lru_cache

from maichart.event_timing import seconds_at_beat
from maichart.generation.muri.model import has_head
from maichart.generation.muri.parser import parse_markers


def _head(marker):
    if marker.get('is_mine') or marker.get('head_mine'):
        return None
    kind = marker['type']
    if kind in {'touch', 'touch_hold'}:
        return str(marker.get('touch_pad') or '') or None
    if kind in {'tap', 'hold'} or has_head(marker):
        lane = marker.get('lane')
        return str(lane) if lane in range(1, 9) else None
    return None


def _pad(head):
    return 'A'+head if len(head)==1 and head in '12345678' else head


@lru_cache(maxsize=16384)
def _struck_heads(text):
    markers, errors, _ = parse_markers(f'(120){{4}}{text},')
    if errors:
        return None
    return frozenset(head for marker in markers if (head := _head(marker)))


@lru_cache(maxsize=16384)
def _action_duration(text,bpm):
    markers,errors,_ = parse_markers(f'({bpm}){{4}}{text},')
    return None if errors else max([0.,*(float(marker.get('end_second',0.)) for marker in markers)])


class RestrictedCell:
    def __init__(self, owner, forbidden, button_capacity=None, held_buttons=frozenset(), *, beat=0):
        self.owner, self.base, self.forbidden = owner, owner.grammar, forbidden
        self.button_capacity = button_capacity
        self.held_buttons = held_buttons
        self.second = float(seconds_at_beat(beat,owner.tempo,owner.first))
        self.bpm = next(change.bpm for change in reversed(owner.tempo) if change.beat <= beat)

    def _duration_fits(self,text):
        if self.owner.max_action_end is None: return True
        duration = _action_duration(text,self.bpm)
        return duration is not None and self.second+duration <= self.owner.max_action_end+1e-8

    def _accept(self, heads):
        buttons = frozenset(head for head in (heads or ()) if len(head)==1 and head in '12345678')
        held = self.held_buttons
        return (heads is not None and not heads & self.forbidden
                and len({_pad(head) for head in heads}) == len(heads)
                and (self.button_capacity is None or len(buttons) <= self.button_capacity)
                and (not self.owner.max_button_heads or
                     len(buttons | held) <= self.owner.max_button_heads))

    def can_continue(self, text):
        if not self.base.can_continue(text):
            return False
        parts = self.base._split_slashes(text)
        used, occupied = set(), set()
        for part in parts[:-1]:
            heads = _struck_heads(part)
            if heads is None:
                return False
            if not self._duration_fits(part): return False
            used.update(heads)
            occupied.update(self.base.heads[part])
        if not self._accept(frozenset(used)):
            return False
        for structure, heads,atom in self.owner.completions(parts[-1]):
            if not structure & occupied and self._accept(frozenset(used) | heads) and self._duration_fits(atom):
                return True
        return False

    def is_complete(self, text):
        return self.base.is_complete(text) and self._accept(_struck_heads(text)) and self._duration_fits(text)


class BasicConfigurationGuard:
    def __init__(self, grammar, tempo, first, *, max_button_heads=2, protect_holds=True,max_action_end=None):
        self.grammar, self.tempo, self.first = grammar, tempo, first
        self.max_button_heads, self.protect_holds = max_button_heads, protect_holds
        self.max_action_end = max_action_end
        self.held_until = {}
        self._completions = {}
        self.blocked_token_choices = 0
        self.events = 0

    def completions(self, prefix):
        if prefix not in self._completions:
            values = set()
            for atom in self.grammar.atoms:
                if atom.startswith(prefix):
                    heads = _struck_heads(atom)
                    if heads is not None:
                        values.add((self.grammar.heads[atom],heads,atom))
            self._completions[prefix] = tuple(values)
        return self._completions[prefix]

    def event(self, beat, *, button_capacity=None):
        second = float(seconds_at_beat(beat, self.tempo, self.first))
        # Queries are speculative: a rejected later position may be followed
        # by an earlier alternative. Only commit may discard expired actions.
        active = {head for head,end in self.held_until.items() if end > second+1e-8}
        forbidden = frozenset(active) if self.protect_holds else frozenset()
        held_buttons = frozenset(head for head in forbidden if len(head)==1 and head in '12345678')
        # Button 4 and Touch A4 refer to the same physical pad.
        aliases = set(forbidden)
        for head in forbidden:
            if len(head)==1 and head in '12345678': aliases.add('A'+head)
            elif len(head)==2 and head[0]=='A' and head[1] in '12345678': aliases.add(head[1])
        return RestrictedCell(self,frozenset(aliases),button_capacity,held_buttons,beat=beat)

    def commit(self, point):
        when = float(seconds_at_beat(point.beat, self.tempo, self.first))
        self.held_until = {head:end for head,end in self.held_until.items() if end>when+1e-8}
        bpm = next(change.bpm for change in reversed(self.tempo) if change.beat <= point.beat)
        markers, errors, _ = parse_markers(f'({bpm}){{4}}{point.note_text},', first=when)
        if errors:
            raise ValueError(f'Cannot update Hold occupancy for {point.note_text!r}')
        for marker in markers:
            if marker['type'] in {'hold', 'touch_hold'} and (head := _head(marker)):
                self.held_until[head] = max(self.held_until.get(head, when), float(marker['end_second']))
        self.events += 1

    def report(self):
        return {'enabled': True, 'max_button_heads': self.max_button_heads,
                'protect_holds': self.protect_holds, 'committed_events': self.events,
                'blocked_token_choices': self.blocked_token_choices,
                'max_action_end_sec':self.max_action_end,
                'playable_verified': False,
                'scope': ['simultaneous_button_head_count', 'head_on_active_hold'],
                'limitations': ['No guarantee for Slide conflicts, travel or full playability.']}
