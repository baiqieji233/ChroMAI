"""Learn authored timing after locally over-dense or sparse committed history."""
from fractions import Fraction
import re
import numpy as np
from maichart.event_timing import EventPoint

POLICY = 'causal_rhythm_v1'
CHOICES = ('auto', 'off', POLICY)
FRESH_DEFAULT = POLICY
_TAP = re.compile(r'[1-8][bx]*')
SPAN = Fraction(4)
MAX_EDITS = 8


def is_tap(point):
    return _TAP.fullmatch(point.note_text) is not None


def perturb_prefix(points, phases, seed, dense, span=SPAN):
    """Keep the last committed event, all sustains, and the original clock."""
    if len(points) < 2:
        return None
    cursor = points[-1].beat
    candidates = [i for i, point in enumerate(points[:-1])
                  if point.beat >= cursor - span and is_tap(point)]
    if not candidates:
        return None
    rng = np.random.default_rng(seed)
    if dense:
        pairs = [i for i in candidates if is_tap(points[i + 1])
                 and ((points[i].beat + points[i + 1].beat) / 2) % 1 in phases]
        if pairs:
            chosen = rng.permutation(pairs)[:MAX_EDITS]
            added = [EventPoint((points[i].beat + points[i + 1].beat) / 2, points[i].note_text)
                     for i in chosen]
            return sorted([*points, *added], key=lambda p: p.beat), 'dense'
    removed = set(rng.permutation(candidates)[:min(MAX_EDITS, max(1, len(candidates) // 2))])
    return [p for i, p in enumerate(points) if i not in removed], 'sparse'


def timing_ids(codec):
    return frozenset(i for i, token in enumerate(codec.tokens)
        if token.startswith(('T:', 'F:', 'NUM:'))
        or token in ('[AT]', '[PHASE]', '[TIME_EXT]', '[TIME_END]', '[EOS]'))
