"""Curve-aware sampling feedback without changing the trained model layout.

The ledger counts completed note heads, not text tokens. Hard limits apply to
short absolute-time regions and never end the whole song merely because one
region is full. Explicit rests forbid new heads while sustaining actions may
cross their boundary.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import Counter
from fractions import Fraction
import math

from maichart.event_timing import seconds_at_beat
from maichart.generation.music_intensity import validate_intensity_profile


class CurveDensityFeedback:
    def __init__(self, profile, *, base_density, start_beat, end_beat, tempo_changes,
                 first=0.0, position_offset=Fraction(0), mode="follow", strength=1.0,
                 prior_cells=(), region_sec=2.0, owned_start_sec=None):
        self.profile = validate_intensity_profile(profile)
        self.mode = mode
        self.steering = mode in {"follow", "hard"} and float(strength) > 0
        self.hard = mode == "hard" and self.steering
        self.strength = float(strength)
        self.start_beat, self.end_beat = Fraction(start_beat), Fraction(end_beat)
        self.position_offset = Fraction(position_offset)
        self.tempo, self.first = tempo_changes, float(first)
        self.low = max(0.0, float(seconds_at_beat(self.start_beat, self.tempo, self.first)))
        if owned_start_sec is not None:
            self.low = max(self.low, float(owned_start_sec))
        self.high = min(float(self.profile["duration_sec"]),
                        float(seconds_at_beat(self.end_beat, self.tempo, self.first)))
        self.region_sec = float(region_sec)
        self.nps = max(0.0, float(base_density)) * 16
        self.times = [point["time_sec"] for point in self.profile["points"]]
        self.values = [point["intensity"] * 2 for point in self.profile["points"]]
        self.rests = [(region["start_sec"], region["end_sec"])
                      for region in self.profile.get("rest_regions", [])]
        self.counts = Counter()
        for cell in prior_cells:
            self.counts[self.region(float(cell["time_sec"]))] += int(cell["heads"])
        self.blocked = set()
        self.events = 0
        self.last_time = None
        self.last_rejected_region = None
        # Set by shape_logits for the current position decision. A feedback
        # window may accept an empty ending when every remaining legal position
        # is unavailable or requests at most half the baseline workload. This
        # keeps follow/hard from turning a transient EOS preference into a
        # blank high-demand window.
        self._empty_end_allowed = False
        self._positions = None
        self._budgets = {}
        self._penalty_cache = {}
        self.stats = {"mode": mode, "region_sec": self.region_sec, "accepted_heads": 0,
                      "rejected_cells": 0, "rejected_heads": 0, "blocked_position_choices": 0,
                      "soft_position_penalties": 0, "deferred_eos": 0}

    def region(self, when):
        return int(math.floor((float(when) + 1e-8) / self.region_sec))

    def rest_at(self, when):
        return any(low <= when < high for low, high in self.rests)

    def _value(self, time):
        index = max(0, min(len(self.times) - 2, bisect_right(self.times, time) - 1))
        low, high = self.times[index:index + 2]
        mix = max(0.0, min(1.0, (time - low) / max(high - low, 1e-12)))
        return self.values[index] * (1 - mix) + self.values[index + 1] * mix

    def _plain_integral(self, low, high):
        if high <= low:
            return 0.0
        grid = [low] + [time for time in self.times if low < time < high] + [high]
        return sum((self._value(a) + self._value(b)) * (b - a) * .5 for a, b in zip(grid, grid[1:]))

    def quota(self, low, high):
        # first is the chart beat-zero time, so a positive offset's pre-chart
        # silence must not grant extra heads to the first partial region.
        low, high = max(0.0, self.first, low), min(float(self.profile["duration_sec"]), high)
        total = self._plain_integral(low, high)
        for start, end in self.rests:
            a, b = max(low, start), min(high, end)
            if b > a:
                total -= self._plain_integral(a, b)
        return max(0.0, total * self.nps)

    def budget(self, region):
        if region not in self._budgets:
            self._budgets[region] = self.quota(region * self.region_sec, (region + 1) * self.region_sec)
        return self._budgets[region]

    def time_for(self, position):
        absolute = self.start_beat + Fraction(position) - self.position_offset
        return float(seconds_at_beat(absolute, self.tempo, self.first))

    def _can_start(self, when):
        if not self.low - 1e-8 <= when < self.high - 1e-8 or self.rest_at(when):
            return False
        if self.hard:
            region = self.region(when)
            return region not in self.blocked and self.counts[region] + 1 <= math.floor(self.budget(region) + 1e-8)
        return True

    def _later_required(self, available_times):
        if not available_times:
            return False
        now = self.last_time if self.last_time is not None else self.low
        after_rest = any(end > now and any(time >= end for time in available_times) for _, end in self.rests)
        active = self.last_rejected_region if self.last_rejected_region is not None else self.region(now)
        full = active in self.blocked or self.counts[active] + 1 > math.floor(self.budget(active) + 1e-8)
        after_full = self.hard and full and any(self.region(time) > active for time in available_times)
        current_value = self._value(now)
        higher_later = self.steering and any(
            time > now + .25 and self._value(time) > max(.6, current_value * 1.5)
            for time in available_times)
        if not (after_rest or after_full or higher_later):
            return False
        # Only defer ending when there is an actual later playable demand. This
        # does not fill all under-target regions or force a note in a quiet window.
        return self.quota(max(now, min(available_times)), self.high) >= 1.0

    def shape_logits(self, logits, vocabulary, *, state, last_beat):
        if state != "position":
            return logits
        from maichart.generation.event_model import EVENT_EOS
        result = logits.clone()
        if self._positions is None:
            self._positions = {index: (self.time_for(Fraction(token[3:])), Fraction(token[3:]))
                               for index, token in enumerate(vocabulary.tokens) if token.startswith("AT:")}
        available = []
        blocked, penalized, penalties = [], [], []
        for index, (when, beat) in self._positions.items():
            if last_beat is not None and beat <= last_beat:
                continue
            if not self._can_start(when):
                blocked.append(index)
                self.stats["blocked_position_choices"] += 1
                continue
            available.append(when)
            if self.steering and not self.hard:
                region = self.region(when)
                budget = self.budget(region)
                # Allow a small chord/burst margin, then penalize early crowding
                # as well as total overshoot; later positions remain available.
                if index not in self._penalty_cache:
                    margin = min(2.0, max(.5, budget * .15))
                    progress = self.quota(region * self.region_sec, min((region + 1) * self.region_sec, when + .25))
                    self._penalty_cache[index] = (progress, margin)
                progress, margin = self._penalty_cache[index]
                excess = max(0.0, self.counts[region] + 1 - progress - margin)
                penalty = min(12.0, 4.0 * self.strength * excess / max(1.0, budget * .2))
                if penalty > 0:
                    penalized.append(index)
                    penalties.append(penalty)
                    self.stats["soft_position_penalties"] += 1
        if blocked:
            result[blocked] = float("-inf")
        if penalized:
            import torch
            result[penalized] -= torch.tensor(penalties, dtype=result.dtype, device=result.device)
        eos = vocabulary.token_to_id[EVENT_EOS]
        if self._later_required(available):
            result[eos] = float("-inf")
            self.stats["deferred_eos"] += 1
        if not available:
            # Explicit rest or exhausted region(s): ending here is intentional,
            # not an empty-window failure requiring forced re-rolls.
            result[eos] = logits[eos]
        self._empty_end_allowed = not available or (self.steering and all(
            self._value(time) <= .5 for time in available))
        return result

    def accept_cell(self, position, text):
        from maichart.notes import parse_timing_point_notes
        from maichart.timing import RawTimingPoint
        when = self.time_for(position)
        parsed = parse_timing_point_notes(RawTimingPoint(index=0, raw=text, note_text=text,
                                                        beat=Fraction(0), tick=0, time_sec=0.0,
                                                        division=4, bpm=120.0))
        heads = len(parsed.notes)
        region = self.region(when)
        if self.rest_at(when) or (self.hard and self.counts[region] + heads > math.floor(self.budget(region) + 1e-8)):
            self.blocked.add(region)
            self.last_rejected_region = region
            self.stats["rejected_cells"] += 1
            self.stats["rejected_heads"] += heads
            return False
        self.counts[region] += heads
        self.events += 1
        self.last_time = when
        self.last_rejected_region = None
        self.stats["accepted_heads"] += heads
        if self.hard and self.counts[region] >= math.floor(self.budget(region) + 1e-8):
            self.blocked.add(region)
        return True

    def allow_empty_end(self):
        return self._empty_end_allowed

