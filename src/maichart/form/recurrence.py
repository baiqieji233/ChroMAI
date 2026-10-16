"""Audio-only recurrence discovery and causal chart motif memory."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Iterable, Sequence

import numpy as np

from maichart.event_timing import EventPoint, EventTempo, seconds_at_beat
from maichart.generation.event_audio import EventAudio
from maichart.form.timing import MetricSpan

MEL_BANDS = 8
CHROMA_DIM = 12
ONSET_DIM = 1
DESCRIPTOR_CHANNELS = CHROMA_DIM + ONSET_DIM + MEL_BANDS
DESCRIPTOR_DIM = 24 * DESCRIPTOR_CHANNELS


@dataclass(frozen=True, slots=True)
class Relation:
    """Provisional audio evidence that an earlier span recurs in a later one."""

    source: int
    target: int
    similarity: float
    confidence: float
    duration_ratio: float
    phase_shift_bins: int = 0
    phase_bins: int = 24


@dataclass(slots=True)
class SongRelations:
    spans: tuple[MetricSpan, ...]
    descriptors: np.ndarray
    links: list[Relation] = field(default_factory=list)
    audit: dict[str, object] = field(default_factory=dict)
    layout_descriptors: np.ndarray | None = None

    def prior_links(self, target: int) -> list[Relation]:
        """Return only relations whose source is wholly before ``target``."""

        target_span = self._span(target)
        return sorted(
            (link for link in self.links
             if link.target == target and self._span(link.source).end <= target_span.start),
            key=lambda link: (-link.similarity, link.source),
        )

    def _span(self, index: int) -> MetricSpan:
        for span in self.spans:
            if span.index == index:
                return span
        raise IndexError(f"Unknown span index: {index}")


@dataclass(frozen=True, slots=True)
class MotifReference:
    target: int
    source: int | None
    points: tuple[EventPoint, ...]
    relation: Relation | None
    provenance: str


def _tempo_map(changes: Iterable[EventTempo | tuple[Fraction, float]]) -> list[EventTempo]:
    result = [
        item if isinstance(item, EventTempo) else EventTempo(Fraction(item[0]), float(item[1]))
        for item in changes
    ]
    result.sort(key=lambda item: item.beat)
    if not result or result[0].beat != 0:
        raise ValueError("tempo_changes must include a BPM at beat 0")
    return result


def _sample(audio: EventAudio, times: np.ndarray) -> np.ndarray:
    if not len(audio.times):
        return np.zeros((len(times), audio.features.shape[1]), dtype=np.float32)
    if (getattr(audio, "metadata", {}) or {}).get("profile") == "rational_audio_queries_v1" \
            and getattr(audio, "feature_policy", "legacy_sparse_v1") == "native_measured_v2":
        from maichart.form.rational_audio import acoustic_at_times
        return acoustic_at_times(audio, times, derived_policy="native_measured_v2")
    stamps = np.asarray(audio.times, dtype=np.float64)
    features = np.asarray(audio.features, dtype=np.float64)
    return np.stack(
        [np.interp(times, stamps, features[:, column], left=0.0, right=0.0)
         for column in range(features.shape[1])],
        axis=1,
    )


def _unit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(values))
    return values / norm if norm > 1e-10 else np.zeros_like(values)


def _aligned_similarity(left: np.ndarray, right: np.ndarray, bins: int) -> tuple[float, int]:
    """Best cosine under a small circular phase offset in normalized time.

    Descriptors retain their public channel-major layout: all chroma bins,
    then onset bins, then timbre bins. Unpack those blocks before shifting so
    only normalized-time positions move and feature channels remain aligned.
    """

    expected_size = bins * DESCRIPTOR_CHANNELS
    if bins <= 0 or left.size != expected_size or right.size != expected_size:
        raise ValueError("Descriptors must contain the configured number of bins")
    widths = (CHROMA_DIM, ONSET_DIM, MEL_BANDS)
    offsets = np.cumsum((0, *(bins * width for width in widths)))
    source_blocks = [
        left[offsets[i]:offsets[i + 1]].reshape(bins, width)
        for i, width in enumerate(widths)
    ]
    target_blocks = [
        right[offsets[i]:offsets[i + 1]].reshape(bins, width)
        for i, width in enumerate(widths)
    ]
    candidates = [
        (
            float(sum(
                np.sum(source * np.roll(target, shift, axis=0))
                for source, target in zip(source_blocks, target_blocks)
            )),
            shift,
        )
        for shift in range(-2, 3)
    ]
    similarity, phase_shift = max(candidates, key=lambda item: (item[0], -abs(item[1]), -item[1]))
    return max(-1.0, min(1.0, similarity)), phase_shift


def _describe(samples: np.ndarray, bins: int) -> tuple[np.ndarray, bool]:
    if not len(samples):
        return np.zeros(bins * DESCRIPTOR_CHANNELS, dtype=np.float32), False

    mel = samples[:, :80]
    chroma = np.clip(samples[:, 80:92], 0.0, None)
    onset = np.clip(samples[:, 92], 0.0, None)
    chroma_bins = np.zeros((bins, CHROMA_DIM), dtype=np.float64)
    onset_bins = np.zeros((bins, 1), dtype=np.float64)
    timbre_bins = np.zeros((bins, MEL_BANDS), dtype=np.float64)
    frame_bins = np.minimum((np.arange(len(samples)) * bins / len(samples)).astype(int), bins - 1)
    for index in range(bins):
        mask = frame_bins == index
        block_chroma = chroma[mask]
        block_mel = mel[mask]
        chroma_bins[index] = block_chroma.mean(axis=0) if len(block_chroma) else 0.0
        onset_bins[index, 0] = onset[mask].mean() if mask.any() else 0.0
        if len(block_mel):
            edges = np.linspace(0, 80, MEL_BANDS + 1, dtype=int)
            timbre_bins[index] = [block_mel[:, edges[j]:edges[j + 1]].mean() for j in range(MEL_BANDS)]

    chroma_bins /= np.maximum(chroma_bins.sum(axis=1, keepdims=True), 1e-8)
    onset_bins = onset_bins / max(float(onset_bins.max()), 1e-8)
    # Retain changes over normalized time, not a static pitch or timbre snapshot.
    chroma_shape = chroma_bins - chroma_bins.mean(axis=0, keepdims=True)
    onset_shape = onset_bins - onset_bins.mean(axis=0, keepdims=True)
    # Remove both frame level and a static song-wide spectral envelope.
    timbre_bins -= timbre_bins.mean(axis=1, keepdims=True)
    timbre_shape = timbre_bins - timbre_bins.mean(axis=0, keepdims=True)
    raw_pieces = (chroma_shape, onset_shape, timbre_shape)
    pieces = [_unit(piece) for piece in raw_pieces]
    descriptor = _unit(np.concatenate(pieces))
    temporal_evidence = any(float(np.linalg.norm(piece)) > 1e-6 for piece in raw_pieces)
    return descriptor.astype(np.float32), temporal_evidence and bool(np.linalg.norm(descriptor) > 1e-8)


LAYOUT_DYNAMICS_CHANNELS = (
    'rms_mean', 'rms_p90', 'attack_mean', 'attack_p90',
    'spectral_detail_mean', 'spectral_change_mean', 'chroma_change_mean', 'rms_change_mean',
)


def _layout_dynamics(samples):
    """Eight non-centered summaries; normalize across the song, not per span."""
    samples = np.asarray(samples, dtype=np.float64)
    if not len(samples):
        return np.zeros(len(LAYOUT_DYNAMICS_CHANNELS), dtype=np.float64)
    rms = np.maximum(samples[:, 93], 0.)
    attack = np.maximum(samples[:, 92], 0.)
    mel = samples[:, :80]
    chroma = samples[:, 80:92]
    mel = mel/np.maximum(np.linalg.norm(mel, axis=1, keepdims=True), 1e-8)
    chroma = chroma/np.maximum(np.linalg.norm(chroma, axis=1, keepdims=True), 1e-8)
    def change(values):
        if len(values) < 2:
            return 0.
        differences = np.diff(values, axis=0)
        return float(np.linalg.norm(differences, axis=1).mean() if differences.ndim == 2
                     else np.abs(differences).mean())
    return np.asarray((rms.mean(), np.percentile(rms, 90),
        attack.mean(), np.percentile(attack, 90),
        np.maximum(samples[:, 94], 0.).mean(), change(mel), change(chroma), change(rms)))


def analyze_recurrence(
    audio: EventAudio,
    spans: Sequence[MetricSpan],
    tempo_changes: Iterable[EventTempo | tuple[Fraction, float]],
    first: float,
    bins: int = 24,
    top_k: int = 3,
    min_similarity: float = 0.75,
    min_separation: int = 1,
    quality_profile: str = 'legacy',
) -> SongRelations:
    """Compare beat-normalized audio spans without accessing chart events."""

    from maichart.form.quality_profile import enhanced
    use_dynamics = enhanced(quality_profile)
    if bins <= 0 or top_k < 0 or min_separation < 0:
        raise ValueError("bins must be positive; top_k and min_separation must be nonnegative")
    if not math.isfinite(min_similarity) or not -1 <= min_similarity <= 1:
        raise ValueError("min_similarity must be finite and between -1 and 1")
    tempo = _tempo_map(tempo_changes)
    ordered = tuple(sorted(spans, key=lambda span: (span.start, span.end, span.index)))
    descriptors = np.zeros((len(ordered), bins * DESCRIPTOR_CHANNELS), dtype=np.float32)
    valid = np.zeros(len(ordered), dtype=bool)
    dynamics = np.zeros((len(ordered), len(LAYOUT_DYNAMICS_CHANNELS)), dtype=np.float64)
    durations: list[float] = []
    for row, span in enumerate(ordered):
        if span.end <= span.start:
            raise ValueError("Metric spans must have positive duration")
        duration = float(span.end - span.start)
        durations.append(duration)
        beats = [span.start + (span.end - span.start) * Fraction(2 * i + 1, 2 * bins) for i in range(bins)]
        stamps = np.asarray([seconds_at_beat(beat, tempo, first) for beat in beats], dtype=np.float64)
        samples = _sample(audio, stamps)
        descriptors[row], valid[row] = _describe(samples, bins)
        if use_dynamics:
            dynamics[row] = _layout_dynamics(samples)

    links: list[Relation] = []
    candidate_count = 0
    candidate_max: float | None = None
    for target_row, target in enumerate(ordered):
        if not valid[target_row] or top_k == 0:
            continue
        candidates: list[Relation] = []
        for source_row, source in enumerate(ordered[:target_row]):
            if target.index - source.index <= min_separation or source.end > target.start:
                continue
            if not valid[source_row]:
                continue
            candidate_count += 1
            similarity, phase_shift = _aligned_similarity(
                descriptors[source_row], descriptors[target_row], bins
            )
            if candidate_max is None or similarity > candidate_max:
                candidate_max = similarity
            if similarity < min_similarity:
                continue
            confidence = max(0.0, min(1.0, (similarity - min_similarity) / max(1e-8, 1.0 - min_similarity)))
            candidates.append(Relation(
                source=source.index,
                target=target.index,
                similarity=similarity,
                confidence=confidence,
                duration_ratio=durations[target_row] / durations[source_row],
                phase_shift_bins=phase_shift,
                phase_bins=bins,
            ))
        links.extend(sorted(candidates, key=lambda item: (-item.similarity, item.source))[:top_k])

    layout_descriptors = None
    if use_dynamics:
        # A single scale per channel across all regions retains the quiet/loud
        # contrast deliberately removed from recurrence matching descriptors.
        scales = np.maximum(np.percentile(np.abs(dynamics), 95, axis=0), 1e-8) if len(dynamics) else np.ones(8)
        normalized = np.clip(dynamics/scales, 0., 4.).astype(np.float32)
        layout_descriptors = np.concatenate((descriptors, normalized), axis=1)
    result = SongRelations(
        spans=ordered,
        descriptors=descriptors,
        links=links,
        audit={
            "status": "provisional_audio_recurrence",
            "confirmed_chorus_labels": False,
            "valid_span_count": int(valid.sum()),
            "rejected_flat_span_indices": [span.index for i, span in enumerate(ordered) if not valid[i]],
            "feature_source": "audio_only",
            "candidate_count": candidate_count,
            "candidate_max": candidate_max,
            "empty_reason": (
                None if links else
                "top_k_zero" if top_k == 0 else
                "no_eligible_prior_pairs" if candidate_count == 0 else
                "below_similarity_threshold"
            ),
        },
        layout_descriptors=layout_descriptors,
    )
    if use_dynamics:
        result.audit.update(layout_descriptor_policy='song_relative_dynamics_v1',
            layout_dynamics_channels=list(LAYOUT_DYNAMICS_CHANNELS),
            layout_descriptor_dim=int(layout_descriptors.shape[1]))
    return result


class CausalMotifMemory:
    """Committed chart points with strictly earlier, nonbinding motif references."""

    def __init__(self, spans: Sequence[MetricSpan]):
        self.spans = {span.index: span for span in spans}
        self._committed: dict[int, tuple[EventPoint, ...]] = {}
        self._pending_sources: dict[int, tuple[int, Relation | None]] = {}

    def commit(self, index: int, points: Iterable[EventPoint]) -> None:
        span = self._span(index)
        normalized = tuple(EventPoint(point.beat, point.note_text) for point in points)
        if any(point.beat < span.start or point.beat >= span.end for point in normalized):
            raise ValueError(f"Committed points must lie within span {index}")
        self._committed[index] = normalized
        self._pending_sources.pop(index, None)

    def reference(self, index: int, relations: SongRelations, max_events: int = 256, *,
                  target_start: Fraction | None = None,
                  target_end: Fraction | None = None) -> MotifReference:
        """Project a prior motif, filtering the requested target range first.

        A musical region can span many decoder compute windows. Applying the
        event budget before this filter would permanently hide the later part
        of a long motif from all later windows. Omitted bounds preserve the
        original whole-region API.
        """
        if max_events < 0:
            raise ValueError("max_events must be nonnegative")
        target = self._span(index)
        lower = target.start if target_start is None else Fraction(target_start)
        upper = target.end if target_end is None else Fraction(target_end)
        if not target.start <= lower <= upper <= target.end:
            raise ValueError("Motif target range must lie within its musical region")
        for relation in relations.prior_links(index):
            source = self._span(relation.source)
            source_points = self._committed.get(relation.source)
            if source_points is None:
                continue
            scale = (target.end - target.start) / (source.end - source.start)
            offset = Fraction(relation.phase_shift_bins, relation.phase_bins) * (target.end - target.start)
            projected_points = []
            if max_events:
                for point in source_points:
                    target_beat = target.start + (point.beat - source.start) * scale - offset
                    if lower <= target_beat < upper:
                        projected_points.append(EventPoint(target_beat, point.note_text))
                        if len(projected_points) >= max_events:
                            break
            projected = tuple(projected_points)
            self._pending_sources[index] = (relation.source, relation)
            return MotifReference(index, relation.source, projected, relation, "audio_recurrence_prior_chart")
        self._pending_sources[index] = (-1, None)
        return MotifReference(index, None, (), None, "no_committed_prior_motif")

    def invalidate_from(self, index: int) -> None:
        """Drop this span and all later outputs, whose references may depend on it."""

        for key in [key for key in self._committed if key >= index]:
            del self._committed[key]
        for key in [key for key in self._pending_sources if key >= index]:
            del self._pending_sources[key]

    def _span(self, index: int) -> MetricSpan:
        try:
            return self.spans[index]
        except KeyError as exc:
            raise IndexError(f"Unknown span index: {index}") from exc
