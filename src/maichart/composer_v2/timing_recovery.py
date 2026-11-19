"""Learn authored timing after locally over-dense or sparse committed history."""
from fractions import Fraction
import re
import numpy as np
from maichart.event_timing import EventPoint

POLICY = 'causal_rhythm_v1'
LAYOUT_POLICY = 'causal_layout_v2'
GESTURE_POLICY = 'gesture_pairs_v1'
POLICIES = ('off', POLICY, LAYOUT_POLICY, GESTURE_POLICY)
CHOICES = ('auto', *POLICIES)
FRESH_DEFAULT = POLICY
_TAP = re.compile(r'[1-8][bx]*')
SPAN = Fraction(4)
MAX_EDITS = 8


def resolve(requested, *, sequence_policy, saved, resume):
    if requested == 'auto':
        policy = saved if resume else FRESH_DEFAULT if sequence_policy == 'continuous_events_v1' else 'off'
    else:
        policy = requested
    if policy not in POLICIES or policy != 'off' and sequence_policy != 'continuous_events_v1':
        raise ValueError('前文纠错只适用于连续事件模型')
    if resume and policy != saved:
        raise ValueError('更换前文纠错策略请使用微调和新输出目录')
    return policy


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


def perturb_phase_prefix(points, phases, seed, *, next_beat, span=SPAN):
    """Equal-volume quarter/sixth-phase history changes; true future stays ahead.

    Only Tap times change. Sustains, key choices, rests and future targets are
    retained. This is a timing-only recovery view, not a label that all other
    legal chart interpretations are musically wrong.
    """
    if not points:return None
    cursor=points[-1].beat
    candidates=[i for i,p in enumerate(points) if p.beat>=cursor-span and is_tap(p)
                and ((4*(p.beat%1)).denominator==1 or (6*(p.beat%1)).denominator==1)]
    rng=np.random.default_rng(seed)
    changed=list(points);edits=0
    for i in rng.permutation(candidates):
        point=changed[i];phase=point.beat%1
        denominator=6 if (4*phase).denominator==1 else 4
        options=[Fraction(j,denominator) for j in range(denominator)
                 if Fraction(j,denominator) in phases and Fraction(j,denominator)!=phase]
        lower=changed[i-1].beat if i else Fraction(-1)
        upper=changed[i+1].beat if i+1<len(changed) else Fraction(next_beat)
        options=[point.beat-phase+p for p in options if lower<point.beat-phase+p<upper]
        if not options:continue
        closest=min(abs(p-point.beat) for p in options)
        options=[p for p in options if abs(p-point.beat)==closest]
        changed[i]=EventPoint(options[int(rng.integers(len(options)))],point.note_text)
        edits+=1
        if edits==2:break
    return changed if edits else None
