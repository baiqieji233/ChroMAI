"""Exact metrical spans and features for event-based charts.

All positions use quarter-note beats as :class:`fractions.Fraction`. Simai grid
directives describe event spacing; they do not define a time signature.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Iterable, Mapping

import numpy as np


def _beat(value: Fraction | int | str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (Fraction, int, str)):
        raise TypeError("Beat positions must be Fraction, integer, or rational string.")
    return Fraction(value)


@dataclass(frozen=True, slots=True)
class MeterChange:
    """A meter declaration whose beat also anchors the new bar phase."""

    beat: Fraction
    numerator: int
    denominator: int
    grouping: tuple[int, ...] = ()
    confidence: float = 1.0
    source: str = "manual"

    def __post_init__(self) -> None:
        object.__setattr__(self, "beat", _beat(self.beat))
        if self.beat < 0:
            raise ValueError("Meter change beats must be nonnegative.")
        if isinstance(self.numerator, bool) or not isinstance(self.numerator, int) or self.numerator <= 0:
            raise ValueError("Meter numerator must be a positive integer.")
        if isinstance(self.denominator, bool) or not isinstance(self.denominator, int) or self.denominator <= 0:
            raise ValueError("Meter denominator must be a positive integer.")
        grouping = tuple(self.grouping)
        if any(isinstance(group, bool) or not isinstance(group, int) or group <= 0 for group in grouping):
            raise ValueError("Grouping entries must be positive integers.")
        if grouping and sum(grouping) != self.numerator:
            raise ValueError("Grouping entries must sum to the meter numerator.")
        confidence = float(self.confidence)
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Meter confidence must be finite and between 0 and 1.")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("Meter source must be a nonempty string.")
        object.__setattr__(self, "grouping", grouping)
        object.__setattr__(self, "confidence", confidence)

    @property
    def beats_per_bar(self) -> Fraction:
        return Fraction(self.numerator * 4, self.denominator)

    def to_dict(self) -> dict[str, Any]:
        return {
            "beat": str(self.beat),
            "numerator": self.numerator,
            "denominator": self.denominator,
            "grouping": list(self.grouping),
            "confidence": self.confidence,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MeterChange:
        return cls(
            beat=_beat(data["beat"]),
            numerator=data["numerator"],
            denominator=data["denominator"],
            grouping=tuple(data.get("grouping", ())),
            confidence=data.get("confidence", 1.0),
            source=data.get("source", "manual"),
        )


@dataclass(frozen=True, slots=True)
class MetricSpan:
    """A contiguous chart interval, split at phrases or meter changes."""

    index: int
    start: Fraction
    end: Fraction
    numerator: int | None
    denominator: int | None
    grouping: tuple[int, ...]
    beats_per_bar: Fraction | None
    bar_boundaries: tuple[Fraction, ...]
    partial: bool
    confidence: float
    source: str

    @property
    def known(self) -> bool:
        return self.numerator is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start": str(self.start),
            "end": str(self.end),
            "numerator": self.numerator,
            "denominator": self.denominator,
            "grouping": list(self.grouping),
            "beats_per_bar": None if self.beats_per_bar is None else str(self.beats_per_bar),
            "bar_boundaries": [str(beat) for beat in self.bar_boundaries],
            "partial": self.partial,
            "confidence": self.confidence,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MetricSpan:
        return cls(
            index=data["index"],
            start=_beat(data["start"]),
            end=_beat(data["end"]),
            numerator=data.get("numerator"),
            denominator=data.get("denominator"),
            grouping=tuple(data.get("grouping", ())),
            beats_per_bar=(
                None if data.get("beats_per_bar") is None else _beat(data["beats_per_bar"])
            ),
            bar_boundaries=tuple(_beat(beat) for beat in data.get("bar_boundaries", ())),
            partial=bool(data["partial"]),
            confidence=float(data["confidence"]),
            source=str(data["source"]),
        )


def normalize_meter_map(
    meters: Iterable[MeterChange | Mapping[str, Any]] | None,
) -> list[MeterChange]:
    """Validate a meter map; an empty map means unknown meter throughout."""

    if meters is None:
        return []
    normalized = [
        item if isinstance(item, MeterChange) else MeterChange.from_dict(item)
        for item in meters
    ]
    if not normalized:
        return []
    if normalized[0].beat != 0:
        raise ValueError("An explicit meter map must begin at beat 0.")
    if any(right.beat <= left.beat for left, right in zip(normalized, normalized[1:])):
        raise ValueError("Meter change beats must be strictly increasing.")
    return normalized


def _boundaries_for_span(start: Fraction, end: Fraction, origin: Fraction, bar: Fraction) -> tuple[Fraction, ...]:
    first_index = max(0, math.ceil((start - origin) / bar))
    last_index = math.floor((end - origin) / bar)
    return tuple(origin + index * bar for index in range(first_index, last_index + 1))


def build_spans(
    end_beat: Fraction | int | str,
    meters: Iterable[MeterChange | Mapping[str, Any]] | None = None,
    *,
    bars_per_span: int = 4,
    max_beats: Fraction | int | str = 16,
    phrase_boundaries: Iterable[Fraction | int | str] = (),
) -> list[MetricSpan]:
    """Partition ``[0, end_beat)`` with exact, gap-free metrical spans.

    Meter changes restart the bar phase at their declared beat. Phrase
    boundaries split spans without resetting that phase. Unknown meter uses
    16-quarter-beat windows and carries no invented bar boundaries.
    """

    end = _beat(end_beat)
    limit = _beat(max_beats)
    if end < 0:
        raise ValueError("end_beat must be nonnegative.")
    if limit <= 0:
        raise ValueError("max_beats must be positive.")
    if isinstance(bars_per_span, bool) or not isinstance(bars_per_span, int) or bars_per_span <= 0:
        raise ValueError("bars_per_span must be a positive integer.")
    meter_map = normalize_meter_map(meters)
    phrases = sorted({_beat(beat) for beat in phrase_boundaries})
    if any(beat <= 0 or beat >= end for beat in phrases):
        raise ValueError("Phrase boundaries must be strictly inside (0, end_beat).")
    if end == 0:
        return []

    result: list[MetricSpan] = []
    cursor = Fraction(0)
    meter_index = 0
    while cursor < end:
        while meter_index + 1 < len(meter_map) and meter_map[meter_index + 1].beat <= cursor:
            meter_index += 1
        meter = meter_map[meter_index] if meter_map else None
        segment_origin = meter.beat if meter is not None else Fraction(0)
        next_meter = meter_map[meter_index + 1].beat if meter is not None and meter_index + 1 < len(meter_map) else end
        bar = meter.beats_per_bar if meter is not None else None
        target = min(limit, bar * bars_per_span) if bar is not None else limit
        elapsed = cursor - segment_origin
        ordinal = elapsed // target
        regular_end = segment_origin + (ordinal + 1) * target
        next_phrase = next((beat for beat in phrases if beat > cursor), end)
        span_end = min(end, next_meter, next_phrase, regular_end)
        if span_end <= cursor:
            raise RuntimeError("Span builder failed to advance.")

        if meter is None:
            boundaries: tuple[Fraction, ...] = ()
            partial = span_end - cursor != target
            span = MetricSpan(
                len(result), cursor, span_end, None, None, (), None, boundaries,
                partial, 0.0, "unknown",
            )
        else:
            boundaries = _boundaries_for_span(cursor, span_end, segment_origin, bar)
            start_on_bar = (cursor - segment_origin) % bar == 0
            end_on_bar = (span_end - segment_origin) % bar == 0
            partial = not start_on_bar or not end_on_bar
            span = MetricSpan(
                len(result), cursor, span_end, meter.numerator, meter.denominator,
                meter.grouping, bar, boundaries, partial, meter.confidence, meter.source,
            )
        result.append(span)
        cursor = span_end
    return result


def metrical_features(
    beats: Iterable[Fraction | int | str],
    meters: Iterable[MeterChange | Mapping[str, Any]] | None = None,
) -> np.ndarray:
    """Return ``[N, 8]`` float32 metric features for exact quarter-beat inputs.

    Columns are sine/cosine pairs for quarter-beat phase, bar phase, and
    within-group phase, followed by meter-known and grouping-known flags.
    Unknown meter retains quarter-beat phase and zeros the unknown features.
    """

    positions = [_beat(value) for value in beats]
    if any(value < 0 for value in positions):
        raise ValueError("Feature beats must be nonnegative.")
    meter_map = normalize_meter_map(meters)
    if not meter_map:
        output = np.zeros((len(positions), 8), dtype=np.float32)
        for row, beat in enumerate(positions):
            angle = 2 * math.pi * float(beat % 1)
            output[row, 0] = math.sin(angle)
            output[row, 1] = math.cos(angle)
        return output
    meter_beats = [change.beat for change in meter_map]
    output = np.zeros((len(positions), 8), dtype=np.float32)
    for row, beat in enumerate(positions):
        beat_phase = beat % 1
        angle = 2 * math.pi * float(beat_phase)
        output[row, 0:2] = (math.sin(angle), math.cos(angle))
        active = meter_map[bisect_right(meter_beats, beat) - 1] if beat >= meter_beats[0] else None
        if active is None:
            continue
        bar = active.beats_per_bar
        bar_phase = ((beat - active.beat) % bar) / bar
        angle = 2 * math.pi * float(bar_phase)
        output[row, 2:4] = (math.sin(angle), math.cos(angle))
        output[row, 6] = 1.0
        if active.grouping:
            within_bar = (beat - active.beat) % bar
            group_start = Fraction(0)
            for group_size in active.grouping:
                group_length = Fraction(group_size * 4, active.denominator)
                if within_bar < group_start + group_length:
                    group_phase = (within_bar - group_start) / group_length
                    break
                group_start += group_length
            else:
                group_phase = Fraction(0)
            angle = 2 * math.pi * float(group_phase)
            output[row, 4:6] = (math.sin(angle), math.cos(angle))
            output[row, 7] = 1.0
    return output
