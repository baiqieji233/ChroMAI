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

    The fine grid uses denominators supported by common maimai subdivisions.
    It deliberately excludes arbitrary 1/48-beat steps, which would create
    192nd-note proposals.  Coarser quarter/half-beat anchors preserve phrase
    movement and long rests without converting wall-clock seconds into
    fractions.
    """
    fine = [Fraction(index, denominator) for denominator in (1,2,3,4,6,8,12,16)
            for index in range(1, denominator*2+1)]
    coarse = [Fraction(index, 4) for index in range(6, 65)]
    triplet = [Fraction(index, 3) for index in range(7, 49)]
    common = [Fraction(5, 3), Fraction(7, 4), Fraction(11, 6), Fraction(7, 3),
              Fraction(17, 8), Fraction(19, 8), Fraction(23, 8), Fraction(25, 8),
              Fraction(5, 2), Fraction(11, 4), Fraction(13, 4), Fraction(7, 2),
              Fraction(15, 4), Fraction(4), Fraction(6), Fraction(8), Fraction(12), Fraction(16)]
    return sorted({value for value in fine + coarse + triplet + common if value > 0 and value <= max_rest_beats})


def candidate_beats(previous: Fraction | None, *, tempo, first: float, audio_end_sec: float,
                   candidate_count: int = 96, max_rest_sec: float = 16.,
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
    first_event=previous is None
    previous = Fraction(0) if previous is None else Fraction(previous)
    if int(candidate_count) < 1:
        raise ValueError("candidate_count must be positive")
    max_rest_sec = max(0.0, float(max_rest_sec))
    clock_previous=max(Fraction(0),previous)
    previous_second=float(seconds_at_beat(clock_previous,tempo,first))
    max_rest_beats=beat_at_second(previous_second+max_rest_sec,tempo,first)-clock_previous
    max_rest_beats=max(Fraction(1,48),max_rest_beats)
    offsets = _canonical_offsets(max_rest_beats)
    values = ([previous] if first_event else []) + [previous + offset for offset in offsets
              if float(seconds_at_beat(previous + offset, tempo, first)) < float(audio_end_sec)]
    count = int(candidate_count)
    if len(values) > count:
        fine_offsets=sorted({Fraction(index, denominator)
            for denominator in (1,2,3,4,6,8,12,16)
            for index in range(1, denominator*2+1)})
        near_required=[Fraction(1,16),Fraction(1,12),Fraction(1,8),Fraction(1,6),Fraction(1,4),
            Fraction(1,3),Fraction(3,8),Fraction(1,2),Fraction(2,3),Fraction(3,4),Fraction(1)]
        priority_offsets=((([Fraction(0)] if first_event else [])+near_required+
            [offset for offset in fine_offsets if offset not in near_required]+[
            Fraction(7,3),Fraction(9,4),Fraction(17,8),Fraction(5,2),Fraction(11,4),Fraction(3),
            Fraction(13,4),Fraction(7,2),Fraction(15,4),Fraction(4),Fraction(6),Fraction(8),Fraction(12),Fraction(16)]))
        by_offset={value-previous:value for value in values}
        far_offsets=[Fraction(2),Fraction(3),Fraction(4),Fraction(6),Fraction(8),Fraction(12),Fraction(16)]
        far=[by_offset[offset] for offset in far_offsets if offset in by_offset]
        far_slots=min(len(far),max(1,count//4))
        near_slots=max(1,count-far_slots)
        selected=[by_offset[offset] for offset in priority_offsets if offset in by_offset][:near_slots]
        remaining_far=[value for value in far if value not in selected]
        if far_slots and remaining_far:
            indices=sorted(set(round(i*(len(remaining_far)-1)/max(1,far_slots-1)) for i in range(far_slots)))
            selected.extend(remaining_far[index] for index in indices)
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
    previous_beat=Fraction(0) if previous is None else max(Fraction(0),Fraction(previous))
    previous_second=float(seconds_at_beat(previous_beat, tempo, first))
    for beat in candidates:
        second=float(seconds_at_beat(beat, tempo, first))
        delta=max(0., second-previous_second)
        rows.append((float(beat-previous_beat), delta, math.log1p(delta),
                     math.sin(2*math.pi*float(beat)), math.cos(2*math.pi*float(beat))))
    return np.asarray(rows,dtype=np.float32)
