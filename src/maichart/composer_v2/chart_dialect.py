"""Normalize supported Simai dialects at the Composer v2 training boundary."""

from dataclasses import replace
from fractions import Fraction
from functools import lru_cache
import re

from maichart.duration import parse_duration_expr
from maichart.event_timing import EventPoint
from maichart.notes import BASIC_HOLD_RE, BASIC_TOUCH_HOLD_RE, _split_top_level_slashes
from maichart.training.chart_sanitization import LEGACY_STRICT_POLICY
from .codec import atom_components


POLICY = 'simai_training_dialect_v1'
STAR_TAP = re.compile(r'([1-8])([bx$]+)')
SLIDE_HEAD = re.compile(r'([1-8])([bx@?!]*)(?=[-^v<>Vpqszw])')


@lru_cache(maxsize=65536)
def _normalize_atom(atom):
    head = SLIDE_HEAD.match(atom)
    if head:
        modifiers = head[2]
        if '?' in modifiers or '!' in modifiers:
            return None, 'headless_slide'
        if '@' in modifiers:
            atom = head[1] + modifiers.replace('@', '') + atom[head.end():]

    star = STAR_TAP.fullmatch(atom)
    if star and star[2].count('$') in (1, 2):
        atom = star[1] + star[2].replace('$', '')

    hold = BASIC_HOLD_RE.fullmatch(atom) or BASIC_TOUCH_HOLD_RE.fullmatch(atom)
    if hold:
        raw_duration = hold['duration']
        duration = parse_duration_expr(raw_duration) if raw_duration else None
        if (duration is None or duration.beats in (0, Fraction(1, 320))
                or (duration.kind == 'seconds' and duration.seconds == 0)):
            atom = atom.replace('h', '')
            if raw_duration:
                atom = atom.replace(raw_duration, '')

    # Match the actual encoder, including compound and same-head Slide forms.
    try:
        atom_components(atom)
    except (ValueError, OverflowError, ZeroDivisionError):
        return None, 'unsupported'
    return atom, None


def normalize_training_chart(chart):
    """Keep the original clock, even when a cell loses all its atoms."""
    points = []
    edits = dict(normalized_atoms=0, headless_slide=0, unsupported=0)
    for point in chart.points:
        # The shared timing parser already maps pseudo EACH (`) to slash.
        kept = []
        for original in _split_top_level_slashes(point.note_text):
            atom, reason = _normalize_atom(original)
            if reason is not None:
                edits[reason] += 1
            else:
                edits['normalized_atoms'] += atom != original
                kept.append(atom)
        if kept:
            points.append(EventPoint(point.beat, '/'.join(kept)))
    if any(edits.values()):
        chart = replace(chart, points=points)
    return chart, edits
