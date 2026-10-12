"""Audio-driven musical regions and bounded computational spans."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Iterable

import numpy as np

from maichart.event_timing import EventTempo, seconds_at_beat
from maichart.form.timing import (
    MeterChange,
    MetricSpan,
    normalize_meter_map,
)
from maichart.generation.event_audio import EventAudio
from maichart.generation.timeline import beat_at_audio_end, normalize_tempo_map


@dataclass(frozen=True, slots=True)
class MusicRegion:
    """Content boundary interval; a computational cap may split it into spans."""

    index: int
    start: Fraction
    end: Fraction
    confidence: float
    start_reason: str
    end_reason: str


class Segmentation(list[MetricSpan]):
    """List-compatible computational spans with their underlying music regions."""

    def __init__(
        self,
        spans: Iterable[MetricSpan] = (),
        *,
        regions: list[MusicRegion] | None = None,
        region_spans: list[MetricSpan] | None = None,
        span_region_indices: list[int] | None = None,
        audit: dict[str, Any] | None = None,
        boundary_reasons: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(spans)
        self.regions = regions or []
        self.region_spans = region_spans or []
        self.span_region_indices = span_region_indices or []
        self.audit = audit or {}
        self.boundary_reasons = boundary_reasons or []

    @property
    def spans(self) -> list[MetricSpan]:
        """Computational spans, also exposed by the list value itself."""

        return list(self)


def _normal_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, np.maximum(norms, 1e-9))


def _audio_novelty(audio: EventAudio) -> np.ndarray:
    features = np.asarray(audio.features, dtype=np.float64)
    count = len(features)
    if count < 2:
        return np.zeros(count, dtype=np.float64)

    rate = 1.0 / float(np.median(np.diff(audio.times))) if count > 1 else 50.0
    chroma = _normal_rows(np.clip(features[:, 80:92], 0.0, None))
    mel = features[:, :80]
    edges = np.linspace(0, 80, 9, dtype=int)
    timbre = np.stack([mel[:, edges[i]:edges[i + 1]].mean(axis=1) for i in range(8)], axis=1)
    timbre -= timbre.mean(axis=1, keepdims=True)
    timbre = _normal_rows(timbre)
    rhythm = features[:, 92:95]

    def context_means(values: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
        prefix = np.vstack((np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)))
        positions = np.arange(count)
        left_start = np.maximum(0, positions - width)
        right_end = np.minimum(count, positions + width)
        left_sum = prefix[positions] - prefix[left_start]
        right_sum = prefix[right_end] - prefix[positions]
        left_count = np.maximum(1, positions - left_start)[:, None]
        right_count = np.maximum(1, right_end - positions)[:, None]
        return left_sum / left_count, right_sum / right_count

    def cosine_change(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        left_norm = np.linalg.norm(left, axis=1)
        right_norm = np.linalg.norm(right, axis=1)
        active = (left_norm > 1e-5) & (right_norm > 1e-5)
        similarity = np.sum(left * right, axis=1) / np.maximum(left_norm * right_norm, 1e-8)
        return np.where(active, np.clip(1.0 - similarity, 0.0, 1.0), 0.0)

    # Compare stable left/right contexts at two scales. Frame derivatives turn
    # repeated beats and production texture into false phrase boundaries; these
    # contrasts respond when the surrounding musical material changes.
    scales: list[np.ndarray] = []
    for seconds in (1.0, 2.0):
        width = max(2, int(round(rate * seconds)))
        chroma_left, chroma_right = context_means(chroma, width)
        timbre_left, timbre_right = context_means(timbre, width)
        rhythm_left, rhythm_right = context_means(rhythm, width)
        chroma_change = cosine_change(chroma_left, chroma_right)
        timbre_change = cosine_change(timbre_left, timbre_right)
        rhythm_scale = np.maximum(np.std(rhythm, axis=0), 0.05)
        rhythm_change = np.linalg.norm((rhythm_left - rhythm_right) / rhythm_scale, axis=1)
        rhythm_change = np.clip(rhythm_change / math.sqrt(rhythm.shape[1]), 0.0, 1.0)
        scales.append(0.45 * chroma_change + 0.35 * timbre_change + 0.20 * rhythm_change)

    novelty = 0.65 * np.maximum(scales[0], scales[1]) + 0.35 * np.minimum(scales[0], scales[1])
    # A short final smooth removes frame-level jumps without erasing multi-second
    # region transitions.
    width = max(1, int(round(rate * 0.25)))
    width = min(width, count)
    if width > 1:
        novelty = np.convolve(novelty, np.ones(width) / width, mode="same")
    baseline = float(np.percentile(novelty, 85))
    scale = max(baseline, 0.20)
    return np.clip(novelty / scale, 0.0, 1.0)


def _beat_for_time(timestamp: float, tempo: list[EventTempo], first: float) -> Fraction:
    return beat_at_audio_end(float(timestamp), tempo, first)


def _active_meter(beat: Fraction, meters: list[MeterChange]) -> MeterChange | None:
    active = None
    for meter in meters:
        if meter.beat > beat:
            break
        active = meter
    return active


def _nearest_grid(
    beat: Fraction,
    meters: list[MeterChange],
    *,
    prefer_bar: bool,
) -> tuple[Fraction, str]:
    active = _active_meter(beat, meters)
    if prefer_bar and active is not None:
        bar = active.beats_per_bar
        ordinal = round(float((beat - active.beat) / bar))
        candidate = active.beat + ordinal * bar
        if abs(float(candidate - beat)) <= min(2.0, float(bar) / 2.0):
            return candidate, "meter_bar"
    # Symmetric context contrast peaks can trail an abrupt change by part of
    # a beat. Floor to a quarter-beat boundary so the target region does not
    # start after the detected change. Explicit meter bars above remain nearest.
    return Fraction(math.floor(float(beat))), "beat"


def _bar_boundaries(start: Fraction, end: Fraction, meters: list[MeterChange]) -> tuple[Fraction, ...]:
    boundaries: set[Fraction] = set()
    for index, meter in enumerate(meters):
        stop = meters[index + 1].beat if index + 1 < len(meters) else end
        if stop <= start or meter.beat >= end:
            continue
        bar = meter.beats_per_bar
        ordinal = max(0, math.ceil(float((start - meter.beat) / bar)))
        beat = meter.beat + ordinal * bar
        while beat <= end and beat < stop:
            if start <= beat <= end:
                boundaries.add(beat)
            beat += bar
    return tuple(sorted(boundaries))


def _metric_span(
    index: int,
    start: Fraction,
    end: Fraction,
    meters: list[MeterChange],
    confidence: float,
    source: str,
) -> MetricSpan:
    meter = _active_meter(start, meters)
    if meter is None:
        return MetricSpan(index, start, end, None, None, (), None, (), True, confidence, source)
    bar = meter.beats_per_bar
    partial = (start - meter.beat) % bar != 0 or (end - meter.beat) % bar != 0
    return MetricSpan(
        index, start, end, meter.numerator, meter.denominator, meter.grouping,
        bar, _bar_boundaries(start, end, meters), partial, confidence, source,
    )


def _computational_chunks(
    region: MusicRegion,
    tempo: list[EventTempo],
    first: float,
    meters: list[MeterChange],
    max_seconds: float,
) -> list[tuple[Fraction, Fraction, str]]:
    result: list[tuple[Fraction, Fraction, str]] = []
    cursor = region.start
    end_time = seconds_at_beat(region.end, tempo, first)
    while cursor < region.end:
        cursor_time = seconds_at_beat(cursor, tempo, first)
        if end_time - cursor_time <= max_seconds + 1e-8:
            result.append((cursor, region.end, region.end_reason))
            break
        cap_time = cursor_time + max_seconds
        approximate = _beat_for_time(cap_time, tempo, first)
        candidate, _ = _nearest_grid(approximate, meters, prefer_bar=True)
        # The cap is a resource bound: choose an earlier exact beat when a bar
        # snap would exceed it, and retain a fractional beat only for very slow
        # tempos where even one beat is longer than the cap.
        if candidate <= cursor or seconds_at_beat(candidate, tempo, first) > cap_time + 1e-8:
            candidate = Fraction(math.floor(float(approximate)))
        if candidate <= cursor:
            candidate = approximate.limit_denominator(960)
        if candidate <= cursor or candidate >= region.end:
            result.append((cursor, region.end, region.end_reason))
            break
        result.append((cursor, candidate, "resource_cap"))
        cursor = candidate
    return result


def segment_music(
    audio: EventAudio,
    tempo_changes: Iterable[EventTempo | tuple[Fraction, float]],
    first: float,
    meters: Iterable[MeterChange | dict[str, Any]] | None = None,
    min_seconds: float = 2.0,
    max_seconds: float = 14.0,
    novelty_threshold: float = 0.65,
) -> Segmentation:
    """Find audio-change regions, then split only as needed for compute limits.

    Unknown meter uses observed audio peaks snapped to quarter-beat boundaries;
    it does not invent bar or phrase cycles. ``max_seconds`` limits individual
    model spans and does not label a musical boundary.
    """

    for name, value in (("min_seconds", min_seconds), ("max_seconds", max_seconds)):
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if min_seconds > max_seconds:
        raise ValueError("min_seconds must not exceed max_seconds")
    if not math.isfinite(float(novelty_threshold)) or not 0.0 <= novelty_threshold <= 1.0:
        raise ValueError("novelty_threshold must be finite and between 0 and 1")
    tempo_items = [
        item if isinstance(item, EventTempo) else EventTempo(Fraction(item[0]), float(item[1]))
        for item in tempo_changes
    ]
    tempo = normalize_tempo_map(tempo_items)
    meter_map = normalize_meter_map(meters)
    if audio.duration_sec <= max(0.0, float(first)):
        return Segmentation(audit={"status": "empty_audio", "novelty_candidates": 0})
    final_beat = beat_at_audio_end(audio.duration_sec, tempo, first)
    if final_beat <= 0:
        return Segmentation(audit={"status": "empty_audio", "novelty_candidates": 0})

    novelty = _audio_novelty(audio)
    frame_times = np.asarray(audio.times, dtype=np.float64)
    if len(frame_times) != len(novelty):
        raise ValueError("audio times and features must have matching frame counts")
    local_peaks = [
        i for i in range(1, len(novelty) - 1)
        if novelty[i] >= novelty_threshold and novelty[i] >= novelty[i - 1] and novelty[i] > novelty[i + 1]
    ]
    # Keep the strongest nearby peak when two changes are closer than the
    # requested soft minimum region length.
    kept: list[int] = []
    edge_guard = max(float(min_seconds), 2.0)
    for candidate in sorted(local_peaks, key=lambda i: (-novelty[i], i)):
        if frame_times[candidate] - frame_times[0] < edge_guard:
            continue
        if audio.duration_sec - frame_times[candidate] < edge_guard:
            continue
        nearby = [index for index in kept if abs(frame_times[candidate] - frame_times[index]) < min_seconds]
        if nearby:
            weaker = min(nearby, key=lambda index: novelty[index])
            if novelty[candidate] > novelty[weaker]:
                kept.remove(weaker)
            else:
                continue
        kept.append(candidate)
    kept.sort()

    boundaries: dict[Fraction, dict[str, Any]] = {
        Fraction(0): {"reason": "song_start", "confidence": 1.0},
        final_beat: {"reason": "song_end", "confidence": 1.0},
    }
    for index, meter in enumerate(meter_map):
        if 0 < meter.beat < final_beat:
            boundaries[meter.beat] = {"reason": "meter_change", "confidence": meter.confidence}
    for frame in kept:
        raw_beat = _beat_for_time(float(frame_times[frame]), tempo, first)
        snapped, reason = _nearest_grid(raw_beat, meter_map, prefer_bar=True)
        if not 0 < snapped < final_beat:
            continue
        evidence = float(novelty[frame])
        current = boundaries.get(snapped)
        if current is None or evidence > float(current.get("confidence", 0.0)):
            boundaries[snapped] = {
                "reason": "audio_novelty_bar" if reason == "meter_bar" else "audio_novelty_beat",
                "confidence": evidence,
                "novelty": evidence,
                "observed_beat": str(raw_beat),
            }

    # Enforce the soft minimum against already established endpoints and meter
    # changes. Content peaks remain the deciding evidence when two compete.
    selected: list[Fraction] = [Fraction(0)]
    for beat in sorted(boundaries):
        if beat in (Fraction(0), final_beat):
            continue
        previous = selected[-1]
        seconds_gap = seconds_at_beat(beat, tempo, first) - seconds_at_beat(previous, tempo, first)
        if seconds_gap < min_seconds:
            previous_info = boundaries[previous]
            current_info = boundaries[beat]
            if current_info.get("reason") == "meter_change":
                selected.append(beat)
                continue
            if previous != 0 and previous_info.get("reason", "").startswith("audio_novelty") \
                    and float(current_info.get("confidence", 0)) > float(previous_info.get("confidence", 0)):
                selected[-1] = beat
            continue
        selected.append(beat)
    selected.append(final_beat)

    regions: list[MusicRegion] = []
    for index, (start, end) in enumerate(zip(selected, selected[1:])):
        start_info = boundaries[start]
        end_info = boundaries[end]
        regions.append(MusicRegion(
            index=index,
            start=start,
            end=end,
            confidence=min(float(start_info["confidence"]), float(end_info["confidence"])),
            start_reason=str(start_info["reason"]),
            end_reason=str(end_info["reason"]),
        ))

    spans: list[MetricSpan] = []
    span_region_indices: list[int] = []
    for region in regions:
        chunks = _computational_chunks(region, tempo, first, meter_map, max_seconds)
        for chunk_index, (start, end, _end_reason) in enumerate(chunks):
            source = region.start_reason if chunk_index == 0 else "resource_cap"
            spans.append(_metric_span(len(spans), start, end, meter_map, region.confidence, source))
            span_region_indices.append(region.index)

    accepted_boundaries = set(selected)
    reason_rows = [{"beat": str(beat), **boundaries[beat]} for beat in sorted(accepted_boundaries)]
    region_spans = [
        _metric_span(region.index, region.start, region.end, meter_map, region.confidence, region.start_reason)
        for region in regions
    ]
    return Segmentation(
        spans,
        regions=regions,
        region_spans=region_spans,
        span_region_indices=span_region_indices,
        audit={
            "status": "ok",
            "novelty_candidates": len(local_peaks),
            "accepted_novelty_boundaries": sum(
                1 for beat in accepted_boundaries for info in [boundaries[beat]]
                if str(info.get("reason", "")).startswith("audio_novelty")
            ),
            "region_count": len(regions),
            "computational_span_count": len(spans),
            "region_spans": [span.to_dict() for span in region_spans],
            "span_region_indices": list(span_region_indices),
            "resource_cap_seconds": float(max_seconds),
            "resource_cap_is_musical_boundary": False,
            "meter_known": bool(meter_map),
        },
        boundary_reasons=reason_rows,
    )
