"""Transactional checks of new events using the existing MiaCode judge."""
from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
import math
from collections import Counter

from maichart.event_timing import seconds_at_beat
from maichart.generation.muri.parser import parse_markers
from maichart.generation.post_process import PostProcessConfig, detect_chart_conflicts
from .events import event_metadata

_EPS = 1e-9


class EventGuard:
    def __init__(self, tempo, first, audio_end, *, min_same_head_sec=.05, intercept_tap_on_slide=False):
        min_same_head_sec = float(min_same_head_sec)
        if not math.isfinite(min_same_head_sec) or not 0. <= min_same_head_sec <= .2:
            raise ValueError('min_same_head_sec must be finite and between 0 and 0.2 seconds')
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

    @lru_cache(maxsize=2048)
    def _markers(self, text, bpm):
        markers, errors, _ = parse_markers(f'({bpm}){{4}}{text},', first=0.)
        return markers, errors

    def _lifetime(self, point):
        second = float(seconds_at_beat(point.beat, self.tempo, self.first))
        bpm = next(t.bpm for t in reversed(self.tempo) if t.beat <= point.beat)
        markers, errors = self._markers(point.note_text, bpm)
        # Explicit timing brackets are measured by the existing parser.
        end = max((float(m.get('end_second', 0.)) for m in markers), default=0.) + second
        sustained = any(m['type'] in {'hold','touch_hold','slide','wifi'} for m in markers)
        return second, max(second, end), sustained, errors

    def accepts(self, point, *, protect_reported_hard=False):
        second, end, sustained, errors = self._lifetime(point)
        self.last_reason = None
        if errors or second<0. or point.beat<0 or end > self.audio_end + 1e-8:
            self._reject('parse_or_action_past_audio_end')
            return False
        if self.min_same_head_sec > 0.:
            # Check only actual onset heads. Parsed EventMetadata deduplicates
            # shared Slide branches, and deliberately excludes Slide endpoints.
            for head in event_metadata(point.note_text).heads:
                pad = f'A{head}' if str(head).isdigit() else str(head)
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
        before = detect_chart_conflicts(context, self.tempo, self.first, self.config) if context else None
        after = detect_chart_conflicts(context + [point], self.tempo, self.first, self.config)
        def hard_keys(result, *, enforced_only=True):
            if result is None: return set()
            return {(issue.conflict_type, round(issue.time, 6),
                     tuple((str(n.beat), n.raw) for n in issue.notes))
                    for issue in result.conflicts_detected
                    if issue.severity == 'hard' and (not enforced_only or
                        (issue.action == 'enforce' and (issue.evidence.startswith('miacode_runtime:')
                         or (self.intercept_tap_on_slide and issue.conflict_type == 'miacode_TapOnSlide'))))}
        # Only an explicitly guided proposal requests this stronger check.
        # Keep configured report/enforce behavior unchanged for all other
        # generation proposals and for training recovery (the default call).
        if protect_reported_hard and hard_keys(after,enforced_only=False)-hard_keys(before,enforced_only=False):
            self._reject('guided_proposal_new_reported_hard_conflict')
            return False
        if after.coverage.get('parse_errors') or hard_keys(after) - hard_keys(before):
            self._reject('new_miacode_runtime_hard_conflict')
            return False
        return True

    def _reject(self, reason):
        self.last_reason = reason
        self.rejections += 1
        self.reason_counts[reason] += 1

    def commit(self, point):
        second, end, sustained, _ = self._lifetime(point)
        self.prefix = [row for row in self.prefix if row[2] >= second - .3 or row[1] >= second - .3]
        self.prefix.append((point, second, end, sustained))
        for head in event_metadata(point.note_text).heads:
            pad = f'A{head}' if str(head).isdigit() else str(head)
            self.last_head_seconds[pad] = second
