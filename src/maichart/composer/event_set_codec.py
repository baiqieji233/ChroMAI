"""Lossless unordered event codec with conservative C8 button rotations.

Shape and regular detail IDs are zero based. Atom IDs start at one; zero is STOP. Detail IDs
are conditional on (shape, slot); they never enumerate complete event products.
Unsupported syntax stays in an exact literal event class. Decoding canonicalizes
slash order only, retaining duplicate atoms and all atom spellings.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
from functools import lru_cache

from maichart.generation.event_cli import _NoteGrammar
from maichart.notes import (BASIC_HOLD_RE, BASIC_SLIDE_RE, BASIC_TAP_RE,
                            parse_timing_point_notes)
from maichart.timing import RawTimingPoint

SCHEMA = 'event_set_codec_v1'
FALLBACK_SHAPE = (('literal', 0),)


@lru_cache(maxsize=512)
def _slide_equivariant(start, pattern, arguments):
    """Prove a complete C8 orbit against pinned native path and judge assets."""
    import numpy as np
    from maichart.generation.muri.data import reference_data
    group = reference_data()['wifi' if pattern == 'w' else 'slides']
    key = str(start) + pattern + arguments
    base = group.get(key)
    if base is None:
        return False
    paths = base.get('lane_samples') if pattern == 'w' else [base.get('samples', [])]
    def pads(value, shift):
        if isinstance(value, str):
            return value[:-1] + str((int(value[-1])-1+shift)%8+1) if len(value)==2 and value[-1] in '12345678' else value
        if isinstance(value, list):
            result = [pads(item, shift) for item in value]
            return sorted(result) if result and all(isinstance(item, str) for item in result) else result
        return value
    judge = 'tri_judge_sequence' if pattern == 'w' else 'judge_sequence'
    for shift in range(8):
        moved = group.get(str((start-1+shift)%8+1) + pattern +
                          ''.join(str((int(arg)-1+shift)%8+1) for arg in arguments))
        if moved is None or pads(base.get(judge, []), shift) != pads(moved.get(judge, []), 0):
            return False
        other = moved.get('lane_samples') if pattern == 'w' else [moved.get('samples', [])]
        if not paths or len(paths) != len(other):
            return False
        angle = shift*np.pi/4
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        for a, b in zip(paths, other):
            xy = np.asarray([[point['x'], point['y']] for point in a])
            target = np.asarray([[point['x'], point['y']] for point in b])
            if xy.shape != target.shape or not len(xy) or not np.allclose(xy@rotation.T, target, atol=.002, rtol=0):
                return False
    return True


def _atoms(text):
    atoms = tuple(sorted(_NoteGrammar._split_slashes(str(text))))
    if not atoms or any(not atom for atom in atoms):
        raise ValueError('An event must contain at least one atom')
    return atoms


def _parsed(atom):
    try:
        result = parse_timing_point_notes(RawTimingPoint(index=0, raw=atom,
            note_text=atom, beat=Fraction(0), tick=0, time_sec=0., division=4, bpm=120.))
    except (ValueError, TypeError):
        return None
    if result.unknown_tokens or len(result.notes) != 1:
        return None
    return result.notes[0]


@lru_cache(maxsize=65536)
def _rotation_spec(atom):
    note = _parsed(atom)
    if note is None or not isinstance(note.position, int):
        return None
    regex = {'tap': BASIC_TAP_RE, 'hold': BASIC_HOLD_RE,
             'slide': BASIC_SLIDE_RE}.get(note.note_type)
    match = regex.fullmatch(atom) if regex is not None else None
    if match is None:
        return None
    # Rotation is admitted only when the native parser AND every rotated
    # numerical path/judge sequence agree. Arcs/compound syntax stays literal.
    if note.note_type == 'slide':
        if match['pattern'] not in {'-', 'v', 'p', 'q', 's', 'z', 'w', 'V'}:
            return None
        if not _slide_equivariant(int(match['position']), match['pattern'], match['end_position']):
            return None
    return note.note_type, match


@lru_cache(maxsize=262144)
def rotate_atom(atom, steps):
    """Rotate only parser-identified position fields, preserving all other bytes."""
    spec = _rotation_spec(atom)
    if spec is None:
        raise ValueError(f'Atom has no proved rotation: {atom!r}')
    kind, match = spec
    fields = ['position'] + (['end_position'] if kind == 'slide' else [])
    result = atom
    for field in sorted(fields, key=lambda key: match.start(key), reverse=True):
        lo, hi = match.span(field)
        value = ''.join(str((int(digit) - 1 + int(steps)) % 8 + 1) for digit in match[field])
        result = result[:lo] + value + result[hi:]
    if _rotation_spec(result) is None:
        raise ValueError('Rotated atom did not pass the native parser')
    return result


@lru_cache(maxsize=65536)
def _representation(text):
    atoms = _atoms(text)
    if any(_rotation_spec(atom) is None for atom in atoms):
        return (('literal', 0),), ('/'.join(atoms),), 0, atoms, True
    candidates = []
    for phase in range(8):
        relative = tuple(sorted((rotate_atom(atom, -phase) for atom in atoms),
                                key=lambda atom: (_rotation_spec(atom)[0], int(atom[0]), atom)))
        shape = tuple((_rotation_spec(atom)[0], int(atom[0]) - 1) for atom in relative)
        candidates.append((shape, relative, phase))
    shape, relative, phase = min(candidates)
    return shape, tuple(atom[1:] for atom in relative), phase, atoms, False


class EventSetCodec:
    schema = SCHEMA

    @classmethod
    def fit(cls, events):
        catalogs = {}
        atom_set = set()
        reasons = Counter()
        for text in events:
            shape, details, _, atoms, fallback = _representation(str(text))
            reasons['literal_fallback' if fallback else 'rotatable'] += 1
            atom_set.update(atoms)
            if not fallback:
                slots = catalogs.setdefault(shape, [set() for _ in shape])
                for slot, detail in zip(slots, details):
                    slot.add(detail)
        total = sum(reasons.values())
        if not total:
            raise ValueError('Cannot fit an empty event catalog')
        codec = cls()
        codec.shape_catalog = [FALLBACK_SHAPE] + sorted(catalogs)
        original_atoms = tuple(atom_set)
        atom_set.update(rotate_atom(atom, phase) for atom in original_atoms
                        if _rotation_spec(atom) is not None for phase in range(8))
        codec.atom_catalog = sorted(atom_set)
        codec.detail_catalog = []
        for shape in codec.shape_catalog:
            if shape == FALLBACK_SHAPE:
                codec.detail_catalog.append([])
                continue
            codec.detail_catalog.append([sorted(slot) for slot in catalogs[shape]])
        codec.coverage = {'events': total, 'rotatable_events': reasons['rotatable'],
                          'fallback_events': reasons['literal_fallback'],
                          'rotation_fraction': reasons['rotatable'] / total}
        codec._index()
        return codec

    def _index(self):
        self._shapes = {shape: i for i, shape in enumerate(self.shape_catalog)}
        self._atoms = {atom: i + 1 for i, atom in enumerate(self.atom_catalog)}
        self._details = [[{value: i for i, value in enumerate(slot)} for slot in shape]
                         for shape in self.detail_catalog]

    def encode(self, note_text):
        shape, details, phase, atoms, fallback = _representation(note_text)
        # A new combination of known atoms remains expressible without fitting
        # a validation catalog or adding a new normal shape/detail class.
        if not fallback and (shape not in self._shapes or any(
                detail not in self._details[self._shapes[shape]][slot]
                for slot, detail in enumerate(details))):
            shape, phase, fallback = FALLBACK_SHAPE, 0, True
        sid = self._shapes[shape]
        detail_ids = ([self._atoms[atom] for atom in atoms] + [0] if fallback else
                      [self._details[sid][slot][detail] for slot, detail in enumerate(details)])
        canonical = '/'.join(atoms)
        mask = self.phase_equivalents(sid, phase, detail_ids)
        return dict(shape=sid, phase=phase, details=detail_ids,
                    atoms=[self._atoms[atom] for atom in atoms], phase_mask=mask,
                    fallback=fallback)

    def decode(self, shape, phase, details):
        shape, phase = int(shape), int(phase)
        if not 0 <= shape < len(self.shape_catalog) or not 0 <= phase < 8:
            raise ValueError('Shape or phase is outside the codec catalog')
        slots = self.shape_catalog[shape]
        if slots == FALLBACK_SHAPE:
            values = []
            stopped = False
            for index in details:
                index = int(index)
                if index == 0:
                    stopped = True
                    continue
                if stopped or not 1 <= index <= len(self.atom_catalog):
                    raise ValueError('Invalid fallback atom ID or atoms after STOP')
                values.append(self.atom_catalog[index - 1])
            if not values:
                raise ValueError('Fallback event cannot be empty')
            return '/'.join(sorted(values))
        if len(details) != len(slots):
            raise ValueError('Detail count must equal the shape slot count')
        values = []
        for slot, index in enumerate(details):
            if not 0 <= int(index) < len(self.detail_catalog[shape][slot]):
                raise ValueError('Detail ID is outside its conditional catalog')
            values.append(self.detail_catalog[shape][slot][int(index)])
        return '/'.join(sorted(rotate_atom(str(head + 1) + detail, phase)
                              for (_, head), detail in zip(slots, values)))

    @property
    def num_shapes(self):
        return len(self.shape_catalog)

    @property
    def num_atoms(self):
        return len(self.atom_catalog) + 1

    @property
    def fallback_shape_id(self):
        return 0

    @property
    def shape_sizes(self):
        """Regular slot counts; fallback is variable length and has size zero."""
        return [0 if shape == FALLBACK_SHAPE else len(shape) for shape in self.shape_catalog]

    @property
    def detail_sizes(self):
        return [[len(slot) for slot in shape] for shape in self.detail_catalog]

    def phase_equivalents(self, shape, phase, details):
        target = self.decode(shape, phase, details)
        return [self.decode(shape, candidate, details) == target for candidate in range(8)]

    def to_dict(self):
        return dict(schema=SCHEMA, shape_catalog=self.shape_catalog,
                    detail_catalog=self.detail_catalog, atom_catalog=self.atom_catalog,
                    coverage=self.coverage)

    @classmethod
    def from_dict(cls, state):
        if state.get('schema') != SCHEMA:
            raise ValueError('Unsupported event set codec schema')
        codec = cls()
        codec.shape_catalog = [tuple((str(kind), int(head)) for kind, head in shape)
                               for shape in state['shape_catalog']]
        codec.detail_catalog = [[list(slot) for slot in shape] for shape in state['detail_catalog']]
        codec.atom_catalog = list(state['atom_catalog'])
        codec.coverage = dict(state['coverage'])
        codec._index()
        return codec
