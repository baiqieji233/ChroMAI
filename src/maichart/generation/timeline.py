"""Canonical explicit tempo maps shared by generation, curve analysis and UI."""
from __future__ import annotations

import hashlib
import json
import math
from fractions import Fraction
from collections.abc import Mapping

from maichart.event_timing import EventTempo


def normalize_tempo_map(value) -> list[EventTempo]:
    """Validate authored timing without sorting, snapping or inventing changes."""
    if isinstance(value, Mapping):
        value = value.get("tempo_changes")
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("tempo_map must be a nonempty list or {tempo_changes: [...]} object")
    result = []
    previous = None
    for raw in value:
        try:
            raw_beat = raw["beat"] if isinstance(raw, Mapping) else raw.beat
            raw_bpm = raw["bpm"] if isinstance(raw, Mapping) else raw.bpm
            if isinstance(raw_beat, bool) or isinstance(raw_bpm, bool):
                raise ValueError("boolean timing is invalid")
            beat = Fraction(str(raw_beat))
            bpm = float(raw_bpm)
        except (KeyError, AttributeError, ValueError, TypeError, ZeroDivisionError, OverflowError) as exc:
            raise ValueError("Each tempo-map entry requires an exact beat and positive finite bpm") from exc
        if beat < 0 or not math.isfinite(bpm) or bpm <= 0:
            raise ValueError("tempo_map requires nonnegative beats and positive finite BPM")
        if previous is None and beat != 0:
            raise ValueError("Explicit tempo_map must start at beat 0")
        if previous is not None and beat <= previous:
            raise ValueError("tempo_map beats must be strictly increasing; duplicate or unsorted entries are invalid")
        result.append(EventTempo(beat, bpm))
        previous = beat
    return result


def serialize_tempo_map(value) -> list[dict]:
    return [{"beat": str(t.beat), "bpm": float(t.bpm)} for t in normalize_tempo_map(value)]


def tempo_map_fingerprint(value) -> str | None:
    if value is None:
        return None
    canonical = json.dumps(serialize_tempo_map(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def beat_at_audio_end(duration_sec, tempo_changes, first=0.0) -> Fraction:
    """Invert the complete piecewise tempo clock; preserve fractional end beats."""
    if not math.isfinite(float(duration_sec)) or float(duration_sec) < 0 or not math.isfinite(float(first)):
        raise ValueError("duration must be finite/nonnegative and first must be finite")
    entries = normalize_tempo_map(tempo_changes)
    remaining = Fraction(str(duration_sec)) - Fraction(str(first))
    if remaining <= 0:
        return Fraction(0)
    for i, entry in enumerate(entries):
        bpm = Fraction(str(entry.bpm))
        if i + 1 == len(entries):
            return entry.beat + remaining * bpm / 60
        next_beat = entries[i + 1].beat
        elapsed = (next_beat - entry.beat) * 60 / bpm
        if remaining <= elapsed:
            return entry.beat + remaining * bpm / 60
        remaining -= elapsed
    raise AssertionError("nonempty tempo map required")
