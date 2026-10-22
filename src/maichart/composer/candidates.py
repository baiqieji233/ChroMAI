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


def _canonical_offsets(max_rest_beats: Fraction) -> list[Fraction]:
    """Return a bounded musical proposal grid shared by train and inference.

    The fine grid is 1/48 beat, which covers the common 8th/12th/16th/24th
    and 32nd-note subdivisions while keeping every denominator bounded.  A
    small set of explicitly common triplet and rest offsets preserves phrase
    movement without converting arbitrary wall-clock seconds into fractions.
    """
    fine = [Fraction(index, 48) for index in range(1, 73)]  # 1/48 .. 3/2
    common = [Fraction(5, 3), Fraction(7, 4), Fraction(11, 6), Fraction(2),
              Fraction(7, 3), Fraction(5, 2), Fraction(3), Fraction(4),
              Fraction(6), Fraction(8), Fraction(1, 5), Fraction(2, 5),
              Fraction(3, 5), Fraction(4, 5), Fraction(6, 5), Fraction(7, 5)]
    return sorted({value for value in fine + common if value > 0 and value <= max_rest_beats})


def candidate_beats(previous: Fraction, *, tempo, first: float, audio_end_sec: float,
                   candidate_count: int = 96, max_rest_sec: float = 8.,
                   target: Fraction | None = None) -> list[Fraction]:
    """Return deterministic bounded-fraction candidates.

    ``target`` is accepted for API compatibility but is deliberately never
    injected into the proposal set.  Training and inference must see the same
    candidates; callers should record or reject authored events that are not
    recalled by this grid instead of silently teaching a special candidate.
    ``max_rest_sec`` is converted only into an upper beat horizon.  Candidate
    positions themselves always come from the canonical musical grid.
    """
    del target
    previous = Fraction(previous)
    if int(candidate_count) < 1:
        raise ValueError("candidate_count must be positive")
    max_rest_sec = max(0.0, float(max_rest_sec))
    clock_previous=max(Fraction(0),previous)
    previous_second=float(seconds_at_beat(clock_previous,tempo,first))
    max_rest_beats=beat_at_second(previous_second+max_rest_sec,tempo,first)-clock_previous
    max_rest_beats=max(Fraction(1,48),max_rest_beats)
    offsets = _canonical_offsets(max_rest_beats)
    values = [previous + offset for offset in offsets
              if float(seconds_at_beat(previous + offset, tempo, first)) < float(audio_end_sec)]
    count = int(candidate_count)
    if len(values) > count:
        priority_offsets=[Fraction(1,48),Fraction(1,32),Fraction(1,24),Fraction(1,16),
            Fraction(1,12),Fraction(1,8),Fraction(1,6),Fraction(1,4),Fraction(1,3),
            Fraction(3,8),Fraction(1,2),Fraction(2,3),Fraction(3,4),Fraction(1),
            Fraction(4,3),Fraction(3,2),Fraction(2),
            Fraction(7,3),Fraction(3),Fraction(4),Fraction(6),Fraction(8)]
        by_offset={value-previous:value for value in values}
        selected=[by_offset[offset] for offset in priority_offsets if offset in by_offset][:count]
        remaining=[value for value in values if value not in selected]
        slots=count-len(selected)
        if slots and remaining:
            indices=sorted(set(round(i*(len(remaining)-1)/max(1,slots-1)) for i in range(slots)))
            selected.extend(remaining[index] for index in indices)
        values=sorted(set(selected))
    return values


def candidate_index(target: Fraction, candidates: list[Fraction]) -> int:
    if target in candidates:
        return candidates.index(target)
    return -1


def candidate_features(candidates, previous, tempo, first, audio_times=None):
    """Numeric time features; audio probes are attached by the dataset/model."""
    import numpy as np
    rows=[]
    previous_second=float(seconds_at_beat(max(Fraction(0),previous), tempo, first))
    for beat in candidates:
        second=float(seconds_at_beat(beat, tempo, first))
        delta=max(0., second-previous_second)
        rows.append((float(beat-previous), delta, math.log1p(delta),
                     math.sin(2*math.pi*float(beat)), math.cos(2*math.pi*float(beat))))
    return np.asarray(rows,dtype=np.float32)
