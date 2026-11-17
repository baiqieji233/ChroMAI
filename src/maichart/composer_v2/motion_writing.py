"""Observable contact sequences on the DX panel, without assigning a hand."""
from functools import lru_cache

import numpy as np

from maichart.composer.event_set_codec import _atoms
from .codec import atom_components


MOVEMENT = ('same_key', 'return_in_two', 'ring_continuation', 'turn_back')
PATHS = ('straight', 'arc', 'p', 'q', 'pp', 'qq', 's', 'z', 'vee', 'fan', 'compound')
SHAPES = {'-': 'straight', '<': 'arc', '>': 'arc', '^': 'arc', 'v': 'vee', 'V': 'vee',
          'w': 'fan', **{shape: shape for shape in ('p', 'q', 'pp', 'qq', 's', 'z')}}
CONTACT = ('tap', 'hold', 'slide', 'touch', 'touch_hold', 'chord', 'break_head')
EVENT_FIELDS = (*CONTACT, *MOVEMENT, *(f'slide_{x}' for x in PATHS))
ORDER_FIELDS = tuple(f'{part}_{field}' for part in ('early', 'late') for field in EVENT_FIELDS)


@lru_cache(maxsize=8192)
def contact(text):
    parts = [atom_components(atom) for atom in _atoms(text)]
    fields = set()
    heads = set()
    for atom in parts:
        kind = atom[0][2:]
        fields.add(kind)
        positions = [t[2:] for t in atom if t.startswith('P:')]
        if positions:
            heads.update(positions if kind == 'tap' else positions[:1])
        if kind == 'slide':
            shapes = [SHAPES[t[2:]] for t in atom if t.startswith('S:') and t[2:] in SHAPES]
            fields.update('slide_' + shape for shape in shapes)
            if len(shapes) > 1:
                fields.add('slide_compound')
        head_end = next((i for i,t in enumerate(atom) if t.startswith('D:') or
                         (t.startswith('S:') and t[2:] in SHAPES)), len(atom))
        if 'S:b' in atom[:head_end]:
            fields.add('break_head')
    if len(heads) > 1:
        fields.add('chord')
    key = next(iter(heads)) if len(heads) == 1 and 'tap' in fields and len(parts) == 1 else ''
    return tuple(EVENT_FIELDS.index(field) for field in fields), int(key) if key.isdigit() else 0


def event_rows(points, approved=None):
    """A turn or a return needs three observed consecutive single rim taps."""
    result = np.zeros((len(points), len(EVENT_FIELDS)), dtype=np.float32)
    valid = np.ones_like(result)
    first = len(CONTACT)
    before = last = 0
    for i, point in enumerate(points):
        indices, key = contact(point.note_text)
        if approved is None or approved[i]:
            result[i, list(indices)] = 1
        if approved is not None and not approved[i]:
            valid[i] = 0
            key = 0
        else:
            valid[i, first] = bool(last and key)
            valid[i, first+1:first+4] = bool(before and last and key)
            if last and key:
                result[i, first] = last == key
            if before and last and key:
                a, b = (last - before) % 8, (key - last) % 8
                result[i, first+1] = key == before and key != last
                if a not in (0, 4) and b not in (0, 4):
                    result[i, first+2] = (a < 4) == (b < 4)
                    result[i, first+3] = (a < 4) != (b < 4)
                else:
                    valid[i, first+2:first+4] = 0
        before, last = last, key
    return result, valid


def sums(points, approved=None):
    values, valid = event_rows(points, approved)
    zeros = np.zeros((1, len(EVENT_FIELDS)), dtype=np.float32)
    return tuple(np.concatenate((zeros, np.cumsum(table, axis=0))) for table in (values, valid))


def ordered_counts(tables, start, stop):
    # The second half of a one-event interval has no observation.
    mid = (start + stop + 1) // 2
    return tuple(np.concatenate((table[mid]-table[start], table[stop]-table[mid])) for table in tables)
