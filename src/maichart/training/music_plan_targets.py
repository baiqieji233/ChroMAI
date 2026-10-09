"""Chart-authored local workload targets, independent of audio loudness."""
from __future__ import annotations

import math
from bisect import bisect_left
from collections import Counter, defaultdict
from fractions import Fraction

import torch

from maichart.event_timing import seconds_at_beat
from maichart.generation.event_quality import parse_event_notes


def _statistics(notes, duration: float, span: float, start: Fraction) -> tuple[list[float], float]:
    count = len(notes)
    groups = Counter(note.beat for note in notes)
    chord = sum(value > 1 for value in groups.values()) / max(len(groups), 1)
    slide = sum(note.note_type == "slide" for note in notes) / max(count, 1)
    touch = sum(note.note_type in {"touch", "touch_hold"} for note in notes) / max(count, 1)
    hold = sum(note.note_type in {"hold", "touch_hold"} for note in notes) / max(count, 1)
    singles = [note for note in notes if isinstance(note.position, int)]
    jacks = sum(a.position == b.position and float(b.beat - a.beat) <= 1.0
                for a, b in zip(singles, singles[1:])) / max(len(singles) - 1, 1)
    jumps = sum(min(abs(a.position - b.position), 8 - abs(a.position - b.position)) >= 3
                for a, b in zip(singles, singles[1:])) / max(len(singles) - 1, 1)
    density = count / max(duration, 1e-6)
    # This is an explainable chart workload proxy, not a difficulty rating or
    # an emotion label. It includes simultaneous actions and spatial/rhythm
    # demands so a dense easy pattern and a dense awkward one differ.
    workload = density * (1.0 + 0.35 * chord + 0.2 * slide + 0.1 * hold
                           + 0.15 * jacks + 0.15 * jumps)
    last = max((float(note.beat - start) / max(span, 1e-6) for note in notes), default=0.0)
    values = [min(1.0, math.log1p(count) / math.log1p(64.0)),
              min(1.0, density / 16.0), chord, slide, min(1.0, last), touch, hold]
    return values, workload


class MusicPlanTargets:
    """Parse a chart once and derive exact-time local targets on demand."""

    def __init__(self, chart):
        self.chart = chart
        parsed = parse_event_notes(chart.points, chart.tempo_changes, first=chart.first)
        if not parsed.syntax_valid:
            raise ValueError("音乐结构监督要求语法完整的谱面")
        self.notes = sorted(parsed.notes, key=lambda note: note.beat)
        self.beats = [note.beat for note in self.notes]
        duration = float(seconds_at_beat(chart.end_beat, chart.tempo_changes, chart.audio_first)
                         - seconds_at_beat(Fraction(0), chart.tempo_changes, chart.audio_first))
        _, self.reference = _statistics(self.notes, duration, float(chart.end_beat), Fraction(0))

    def window(self, start: Fraction, end: Fraction) -> torch.Tensor:
        start, end = Fraction(start), Fraction(end)
        notes = self.notes[bisect_left(self.beats, start):bisect_left(self.beats, end)]
        duration = float(seconds_at_beat(end, self.chart.tempo_changes, self.chart.audio_first)
                         - seconds_at_beat(start, self.chart.tempo_changes, self.chart.audio_first))
        values, workload = _statistics(notes, duration, float(end - start), start)
        intensity = min(1.0, max(0.0, 0.5 * workload / max(self.reference, 1e-6)))
        by_beat = defaultdict(list)
        for note in notes:
            by_beat[note.beat].append(note)
        classes = Counter()
        for cell in by_beat.values():
            types = {note.note_type for note in cell}
            category = (3 if types & {"touch", "touch_hold"} else 2 if "slide" in types
                        else 1 if len(cell) > 1 else 0)
            classes[category] += 1
        # One isolated Touch must not label an otherwise plain window Touch.
        # Continuous ratios retain the minority actions; this category is the
        # dominant local configuration, with deterministic ties.
        config = max(classes, key=lambda key: (classes[key], key)) if classes else 0
        return torch.tensor(values + [intensity, config / 3.0], dtype=torch.float32)
