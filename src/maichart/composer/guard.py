"""Transactional checks of new events using the existing MiaCode judge."""
from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from copy import deepcopy
import math
from collections import Counter

from maichart.event_timing import seconds_at_beat,EventPoint
from maichart.generation.muri.parser import parse_markers
from maichart.generation.post_process import PostProcessConfig, detect_chart_conflicts
from .events import event_metadata
from maichart.notes import parse_timing_point_notes
from maichart.timing import RawTimingPoint

_EPS = 1e-9


def enforced_native_hard_conflict(issue, *, intercept_tap_on_slide=False):
    """Recognize enforced runtime findings and exact static pad occupancy.

    Static Overlap checks an identical physical pad during another note's
    lifetime. Other static geometric references keep their existing report
    semantics; Touch circles intersecting is not a MultiTouch prohibition.
    """
    if issue.severity!='hard' or issue.action!='enforce':return False
    evidence=issue.evidence
    return (evidence.startswith('miacode_runtime:')
        or (evidence.startswith('miacode_static:') and issue.conflict_type=='miacode_Overlap'
            and _static_held_head_overlap(issue.notes))
        or (intercept_tap_on_slide and issue.conflict_type=='miacode_TapOnSlide'))


def _static_held_head_overlap(notes):
    """A real sustained head shares its own physical pad with a new onset.

    Native static references also include near Tap/Tap pairs and merge a
    button with its A-ring sensor. Those are broader than this guard's open
    layout/head-cooldown contract. Keep their previous runtime-only handling.
    """
    def pad(note):
        value=str(getattr(note,'button',''))
        return 'C' if value.startswith('C') and value[1:].isdigit() else value
    for held in notes:
        if getattr(held,'note_type',None) not in ('hold','touch_hold'):continue
        duration=float(getattr(held,'duration',0.) or 0.)
        if duration<=_EPS:continue
        start=float(held.time);finish=start+duration;head=pad(held)
        if not head:continue
        for other in notes:
            if other is held or pad(other)!=head:continue
            second=float(other.time)
            if start-_EPS<=second<finish-_EPS:return True
    return False


@lru_cache(maxsize=16384)
def _event_syntax_error(text):
    """Match the strict exporter's onset contract before native judging.

    The native runtime permits some repeated Touch markers. The exported
    chart parser rejects them, including Tap/TouchHold mixtures on one pad.
    Parse complete notes so multi-branch Slides still have just one head.
    """
    parsed=parse_timing_point_notes(RawTimingPoint(index=0,raw=text,note_text=text,
        beat=Fraction(0),tick=0,time_sec=0.,division=4,bpm=120.))
    if parsed.unknown_tokens or not parsed.notes:return 'unsupported_or_empty_event'
    heads=Counter(str(note.position) for note in parsed.notes)
    return 'duplicate_event_head' if any(count>1 for count in heads.values()) else None


def first_rejected_prefix(guard,beat,atoms,*,protect_reported_hard=False):
    """Localize a rejected combination before excluding future proposals.

    Called only after the full event was rejected. Check at most S-1 shorter
    prefixes; the first bad prefix also catches an illegal first atom. These
    diagnostic checks do not inflate primary proposal/rejection statistics.
    """
    original_reason=guard.last_reason
    counts={name:getattr(guard,name) for name in ('checks','rejections') if hasattr(guard,name)}
    reasons=guard.reason_counts.copy() if hasattr(guard,'reason_counts') else None
    probes=0
    try:
        for length in range(1,len(atoms)):
            probes+=1
            if not guard.accepts(EventPoint(beat,'/'.join(atoms[:length])),protect_reported_hard=protect_reported_hard):
                return tuple(atoms[:length])
        guard.last_reason=original_reason
        return tuple(atoms)
    finally:
        for name,value in counts.items():setattr(guard,name,value)
        if reasons is not None:
            guard.reason_counts.clear();guard.reason_counts.update(reasons)
        guard.retry_prefix_checks=getattr(guard,'retry_prefix_checks',0)+probes


@lru_cache(maxsize=2048)
def _cached_markers(text, bpm):
    # Pure shared cache: keys never retain a guard or its chart prefix.
    markers, errors, _ = parse_markers(f'({bpm}){{4}}{text},', first=0.)
    return markers, errors


class EventGuard:
    def __init__(self, tempo, first, audio_end, *, min_same_head_sec=.05, intercept_tap_on_slide=False, touch_layout_policy='legacy_two_heads'):
        min_same_head_sec = float(min_same_head_sec)
        if not math.isfinite(min_same_head_sec) or not 0. <= min_same_head_sec <= .2:
            raise ValueError('min_same_head_sec must be finite and between 0 and 0.2 seconds')
        if touch_layout_policy not in {'open','legacy_two_heads'}:
            raise ValueError('unknown touch_layout_policy')
        self.touch_layout_policy=touch_layout_policy
        self.tempo, self.first, self.audio_end = tempo, first, float(audio_end)
        self.min_same_head_sec = min_same_head_sec
        self.intercept_tap_on_slide = bool(intercept_tap_on_slide)
        self.prefix = []
        self.checks = self.rejections = 0
        self.reason_counts = Counter()
        self.last_head_seconds = {}
        self.config = PostProcessConfig(enable_conflict_detection=True, detect_soft_conflicts=True,
            conflict_engine='miacode', conflict_kind_modes={'MultiTouch':'enforce',
                'TapOnSlide':'enforce' if self.intercept_tap_on_slide else 'report',
                'Overlap':'enforce','SlideHeadTap':'report','SlideTooFast':'report'})
        self.last_reason = None
        self._before_cache = None

    @staticmethod
    def _markers(text, bpm):
        # Parser output contains mutable nested dictionaries/lists. Never expose
        # the cached object to callers, including other guard instances.
        return deepcopy(_cached_markers(text, bpm))

    def _lifetime(self, point):
        second = float(seconds_at_beat(point.beat, self.tempo, self.first))
        bpm = next(t.bpm for t in reversed(self.tempo) if t.beat <= point.beat)
        markers, errors = self._markers(point.note_text, bpm)
        # Explicit timing brackets are measured by the existing parser.
        end = max((float(m.get('end_second', 0.)) for m in markers), default=0.) + second
        # Zero-duration Hold/TouchHold is an accepted source dialect. It has
        # an onset like Tap/Touch, but no sustained occupancy. Keep its spelling
        # for model history and let the native judge check ordinary conflicts.
        sustained = any(m['type'] in {'hold','touch_hold','slide','wifi'}
                        and float(m.get('end_second', 0.)) > _EPS for m in markers)
        return second, max(second, end), sustained, errors

    def accepts(self, point, *, protect_reported_hard=False):
        syntax_error=_event_syntax_error(point.note_text)
        if syntax_error:
            self._reject(syntax_error)
            return False
        second, end, sustained, errors = self._lifetime(point)
        self.last_reason = None
        if errors or second<0. or point.beat<0 or end > self.audio_end + 1e-8:
            self._reject('parse_or_action_past_audio_end')
            return False
        if self.min_same_head_sec > 0.:
            # Check only actual onset heads. Parsed EventMetadata deduplicates
            # shared Slide branches, and deliberately excludes Slide endpoints.
            for head in event_metadata(point.note_text).heads:
                pad = f'A{head}' if self.touch_layout_policy=='legacy_two_heads' and str(head).isdigit() else str(head)
                prior = self.last_head_seconds.get(pad)
                if prior is not None and second - prior < self.min_same_head_sec - _EPS:
                    self._reject('repeated_head_too_soon')
                    return False
        active = [(p, t, finish, sustain) for p, t, finish, sustain in self.prefix
                  if finish >= second - .3 or t >= second - .3]
        # A first event can itself contain duplicate heads or impossible
        # simultaneous actions. Taps in the recent prefix also require the
        # same native transaction as events containing sustained actions.
        context = [row[0] for row in active]
        self.checks += 1
        def hard_keys(result, *, enforced_only=True):
            if result is None: return frozenset()
            return frozenset((issue.conflict_type, round(issue.time, 6),
                     tuple((str(n.beat), n.raw) for n in issue.notes))
                    for issue in result.conflicts_detected
                    if issue.severity == 'hard' and (not enforced_only or
                        enforced_native_hard_conflict(issue,intercept_tap_on_slide=self.intercept_tap_on_slide)))
        # Candidate time can change the active window without a commit. Keep
        # only an immutable summary of the exact latest context/configuration.
        key = (tuple((p.beat, p.note_text) for p in context),
               tuple((t.beat, t.bpm) for t in self.tempo), self.first,
               repr(self.config), self.intercept_tap_on_slide)
        if self._before_cache is None or self._before_cache[0] != key:
            before = detect_chart_conflicts(context, self.tempo, self.first, self.config) if context else None
            self._before_cache = (key, hard_keys(before), hard_keys(before, enforced_only=False),
                                  bool(before and before.coverage.get('parse_errors')))
        _, before_enforced, before_all, _ = self._before_cache
        after = detect_chart_conflicts(context + [point], self.tempo, self.first, self.config)
        # Only an explicitly guided proposal requests this stronger check.
        # Keep configured report/enforce behavior unchanged for all other
        # generation proposals and for training recovery (the default call).
        if protect_reported_hard and hard_keys(after,enforced_only=False)-before_all:
            self._reject('guided_proposal_new_reported_hard_conflict')
            return False
        if after.coverage.get('parse_errors') or hard_keys(after) - before_enforced:
            self._reject('new_miacode_runtime_hard_conflict')
            return False
        return True

    def _reject(self, reason):
        self.last_reason = reason
        self.rejections += 1
        self.reason_counts[reason] += 1

    def commit(self, point):
        self._before_cache = None
        second, end, sustained, _ = self._lifetime(point)
        self.prefix = [row for row in self.prefix if row[2] >= second - .3 or row[1] >= second - .3]
        self.prefix.append((point, second, end, sustained))
        for head in event_metadata(point.note_text).heads:
            pad = f'A{head}' if self.touch_layout_policy=='legacy_two_heads' and str(head).isdigit() else str(head)
            self.last_head_seconds[pad] = second
