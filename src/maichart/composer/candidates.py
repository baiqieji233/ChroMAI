"""Shared candidate-time generation for training and free generation."""
from __future__ import annotations

from fractions import Fraction
import math

from maichart.form.rational_audio import beat_at_second
from maichart.event_timing import seconds_at_beat


def _tempo_step_seconds(tempo, first, beat, fraction: Fraction) -> float:
    before = float(seconds_at_beat(beat, tempo, first))
    after = float(seconds_at_beat(beat + fraction, tempo, first))
    return max(0., after - before)


def candidate_beats(previous: Fraction, *, tempo, first: float, audio_end_sec: float,
                   candidate_count: int = 64, max_rest_sec: float = 8.,
                   target: Fraction | None = None) -> list[Fraction]:
    """Return deterministic, non-grid-locked event candidates.

    The set combines common musical fractions, exact second horizons and
    logarithmically spaced rests.  The same function is used by dataset
    preparation and generation; a true event is never silently injected when
    it is absent from this proposal set.
    """
    previous = Fraction(previous)
    steps = [Fraction(1, 16), Fraction(1, 12), Fraction(1, 8), Fraction(1, 6),
             Fraction(1, 4), Fraction(1, 3), Fraction(3, 8), Fraction(1, 2),
             Fraction(2, 3), Fraction(3, 4), Fraction(1), Fraction(4, 3),
             Fraction(3, 2), Fraction(2), Fraction(3), Fraction(4)]
    values = {previous + step for step in steps}
    if target is not None and target > previous:
        values.add(Fraction(target))
    for seconds in (0.08, 0.12, 0.16, 0.24, 0.32, 0.5, 0.75, 1., 1.5, 2., 3., 4., 6., max_rest_sec):
        beat = beat_at_second(float(seconds_at_beat(previous, tempo, first)) + seconds, tempo, first)
        values.add(beat)
    values = sorted(value for value in values if value > previous and
                    float(seconds_at_beat(value, tempo, first)) < float(audio_end_sec))
    if len(values) > int(candidate_count):
        # Preserve near, medium and far choices so long rests remain possible.
        indices = sorted(set(round(i * (len(values)-1) / max(1, candidate_count-1))
                             for i in range(candidate_count)))
        values = [values[index] for index in indices]
        if target is not None and target not in values:
            values[-1] = Fraction(target)
            values = sorted(set(values))
    return values


def candidate_index(target: Fraction, candidates: list[Fraction]) -> int:
    if target in candidates:
        return candidates.index(target)
    return -1


def candidate_features(candidates, previous, tempo, first, audio_times=None):
    """Numeric time features; audio probes are attached by the dataset/model."""
    import numpy as np
    rows=[]
    previous_second=float(seconds_at_beat(previous, tempo, first))
    for beat in candidates:
        second=float(seconds_at_beat(beat, tempo, first))
        delta=max(0., second-previous_second)
        rows.append((float(beat-previous), delta, math.log1p(delta),
                     math.sin(2*math.pi*float(beat)), math.cos(2*math.pi*float(beat))))
    return np.asarray(rows,dtype=np.float32)
