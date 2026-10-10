"""Chart-authored local workload targets, independent of audio loudness."""
from __future__ import annotations

import math
from bisect import bisect_left
from collections import Counter, defaultdict
from fractions import Fraction

import torch

from maichart.event_timing import seconds_at_beat
from maichart.generation.event_quality import parse_event_notes
from maichart.generation.music_plan import (
    MUSIC_STRUCTURE_PROFILE, MUSIC_STRUCTURE_V3_PROFILE, is_music_structure,
    validate_profile,
)


# Versioned v3 training target: gaps at most 33/128 beat. Keep this definition
# stable for existing v3 checkpoints. The 1/128 tolerance is a heuristic, not
# evidence of a minimum chart grid or a physical timing limit. This scalar
# measures short gaps; it does not classify all 12th-note/syncopated rhythms or
# identify independent musical voices. The report tool also exposes raw grids.
FINE_RHYTHM_QUARTER_BEAT = Fraction(1, 4)
FINE_RHYTHM_TOLERANCE = Fraction(1, 128)
FINE_RHYTHM_GAP_CEILING = FINE_RHYTHM_QUARTER_BEAT + FINE_RHYTHM_TOLERANCE


def fine_note_ratio(beats) -> float:
    """Share of adjacent onsets separated by a quarter beat or less.

    Beat-normalized on purpose: the mask is about *grid subdivision*, not about
    notes per second, so a 24th-note run counts the same at 90 BPM and at 200
    BPM, and a 16th-note run now counts as fine whereas an 8th-note run does not.
    ``event_ctx_dataset._fine_note_ratio`` delegates here, because one
    implementation must both pick the oversampled windows and define what the
    planner is asked to predict; two copies would drift and the disagreement
    would look like a modelling failure.

    Fewer than two distinct onsets have no interval to measure and report 0.
    """

    unique = sorted({Fraction(beat) for beat in beats})
    if len(unique) < 2:
        return 0.0
    gaps = [b - a for a, b in zip(unique, unique[1:]) if b > a]
    if not gaps:
        return 0.0
    return sum(1 for gap in gaps if gap <= FINE_RHYTHM_GAP_CEILING) / len(gaps)


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

    def __init__(self, chart, profile: str = MUSIC_STRUCTURE_PROFILE):
        validate_profile(profile)
        if not is_music_structure(profile):
            raise ValueError("MusicPlanTargets requires a structure planner profile.")
        self.profile = profile
        self.fine_rhythm_enabled = profile == MUSIC_STRUCTURE_V3_PROFILE
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
        continuous = values + [intensity]
        if self.fine_rhythm_enabled:
            continuous.append(fine_note_ratio(note.beat for note in notes))
        return torch.tensor(continuous + [config / 3.0], dtype=torch.float32)
