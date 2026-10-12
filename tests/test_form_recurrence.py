from fractions import Fraction

import numpy as np
import pytest

from maichart.event_timing import EventPoint, EventTempo
from maichart.form.recurrence import (
    DESCRIPTOR_DIM,
    CausalMotifMemory,
    Relation,
    SongRelations,
    analyze_recurrence,
)
from maichart.form.timing import MetricSpan
from maichart.form.segmentation import segment_music
from maichart.generation.event_audio import ACOUSTIC_DIM, EventAudio


def _span(index, start, end):
    return MetricSpan(
        index, Fraction(start), Fraction(end), 4, 4, (), Fraction(4), (),
        False, 1.0, "test",
    )


def _audio(patterns, *, bpm=120, pattern_beats=2):
    """Expand beat-normalized patterns to frames using a supplied beat clock."""
    tempo_changes = [EventTempo(Fraction(0), bpm)]
    beat_duration = 60.0 / bpm
    duration = len(patterns) * pattern_beats * beat_duration
    times = np.arange(int(duration * 50) + 1, dtype=np.float32) / 50
    features = np.zeros((len(times), ACOUSTIC_DIM), dtype=np.float32)
    palette = [1, 4, 7]
    for frame, seconds in enumerate(times):
        beat = seconds / beat_duration
        span = min(int(beat // pattern_beats), len(patterns) - 1)
        phase = (beat % pattern_beats) / pattern_beats
        kind = patterns[span]
        chroma_index = palette[kind]
        features[frame, 80 + chroma_index] = 1.0
        # A non-flat onset contour distinguishes temporal shapes in the descriptor.
        features[frame, 92] = float((phase < (0.16 if kind == 1 else 0.38)) and kind != 2)
        features[frame, 93] = 0.2 if kind != 2 else 0.0
        features[frame, 0:8] = np.linspace(-8, -1, 8, dtype=np.float32) * (kind + 1)
    return EventAudio(features, times, float(duration), float(bpm))


def test_repeated_audio_pattern_links_nonadjacent_span_and_excludes_neighbor():
    spans = [_span(0, 0, 2), _span(1, 2, 4), _span(2, 4, 6)]
    relations = analyze_recurrence(
        _audio([0, 1, 0]), spans, [EventTempo(Fraction(0), 120)], 0,
        min_similarity=0.75,
    )

    assert relations.descriptors.shape == (3, DESCRIPTOR_DIM)
    assert [(link.source, link.target) for link in relations.links] == [(0, 2)]
    assert relations.audit["confirmed_chorus_labels"] is False


def test_silence_does_not_form_false_recurrence_group():
    spans = [_span(0, 0, 2), _span(1, 2, 4), _span(2, 4, 6)]
    audio = EventAudio(
        np.zeros((151, ACOUSTIC_DIM), dtype=np.float32),
        np.arange(151, dtype=np.float32) / 50,
        3.0,
        120.0,
    )
    relations = analyze_recurrence(audio, spans, [EventTempo(Fraction(0), 120)], 0)

    assert relations.links == []
    assert relations.audit["valid_span_count"] == 0


def test_static_nonzero_texture_is_rejected_as_recurrence_evidence():
    spans = [_span(0, 0, 2), _span(1, 2, 4)]
    features = np.zeros((101, ACOUSTIC_DIM), dtype=np.float32)
    features[:, 81] = 1.0
    features[:, 0:8] = np.linspace(-8, -1, 8, dtype=np.float32)
    audio = EventAudio(features, np.arange(101, dtype=np.float32) / 50, 2.0, 120.0)
    relations = analyze_recurrence(
        audio, spans, [EventTempo(Fraction(0), 120)], 0,
        min_separation=0,
    )

    assert relations.links == []
    assert relations.audit["valid_span_count"] == 0


def test_partial_chroma_only_descriptors_normalize_to_full_similarity():
    spans = [_span(0, 0, 2), _span(1, 2, 4)]
    times = np.arange(201, dtype=np.float32) / 50
    features = np.zeros((len(times), ACOUSTIC_DIM), dtype=np.float32)
    for index, stamp in enumerate(times):
        beat = stamp * 2
        pattern_index = int((beat % 2) * 4) % 4
        features[index, 80 + (pattern_index * 2)] = 1.0
    audio = EventAudio(features, times, 4.0, 120.0)
    relations = analyze_recurrence(
        audio, spans, [EventTempo(Fraction(0), 120)], 0,
        min_similarity=0.99,
        min_separation=0,
    )

    assert len(relations.links) == 1
    assert relations.links[0].similarity == pytest.approx(1.0)


def test_repeated_realistic_clip_starting_half_second_later_matches():
    rng = np.random.default_rng(23)
    coarse = rng.normal(size=(10, ACOUSTIC_DIM)).astype(np.float32)
    coarse[:, 80:92] = np.abs(coarse[:, 80:92])
    clip = np.repeat(coarse, 40, axis=0)
    clip = np.stack([
        np.convolve(clip[:, column], np.ones(9, dtype=np.float32) / 9, mode="same")
        for column in range(ACOUSTIC_DIM)
    ], axis=1)
    features = np.zeros((851, ACOUSTIC_DIM), dtype=np.float32)
    features[:400] = clip
    features[425:825] = clip
    times = np.arange(len(features), dtype=np.float32) / 50
    audio = EventAudio(features, times, float(times[-1]), 120.0)
    spans = [_span(0, 0, 16), _span(1, 17, 33)]

    relations = analyze_recurrence(
        audio, spans, [EventTempo(Fraction(0), 120)], 0,
        min_similarity=0.7,
        min_separation=0,
    )

    assert len(relations.links) == 1
    assert relations.links[0].similarity >= 0.7


def test_recurrence_graph_uses_music_regions_independent_of_compute_cap():
    audio = _audio([0, 1, 0], pattern_beats=6)
    tempo = [EventTempo(Fraction(0), 120)]
    regions_wide = segment_music(audio, tempo, 0, min_seconds=0.2, max_seconds=14.0)
    regions_capped = segment_music(audio, tempo, 0, min_seconds=0.2, max_seconds=0.5)
    assert len(regions_wide) != len(regions_capped)
    assert regions_wide.region_spans == regions_capped.region_spans
    assert regions_capped.span_region_indices == [index // 6 for index in range(18)]

    wide = analyze_recurrence(audio, regions_wide.region_spans, tempo, 0, min_separation=1)
    capped = analyze_recurrence(audio, regions_capped.region_spans, tempo, 0, min_separation=1)
    assert [(link.source, link.target) for link in wide.links] == [(0, 2)]
    assert [(link.source, link.target) for link in capped.links] == [(0, 2)]


def test_bpm_change_uses_chart_beat_clock_and_reports_duration_ratio():
    spans = [_span(0, 0, 2), _span(1, 2, 4), _span(2, 4, 6)]
    # The target section is twice as long in seconds after the beat-4 slowdown.
    times = np.arange(251, dtype=np.float32) / 50
    features = np.zeros((len(times), ACOUSTIC_DIM), dtype=np.float32)
    for frame, stamp in enumerate(times):
        if stamp < 1.0:
            phase = stamp
        elif stamp < 2.0:
            phase = 10 + stamp
        else:
            phase = (stamp - 2.0) / 2.0
        chroma = 2 if 0 <= phase < 1 else 6
        features[frame, 80 + chroma] = 1.0
        features[frame, 92] = 1.0 if int(phase * 8) % 4 == 0 else 0.0
        features[frame, 0:8] = np.linspace(-8, -1, 8)
    audio = EventAudio(features, times, 5.0, 120.0)
    relations = analyze_recurrence(
        audio, spans,
        [EventTempo(Fraction(0), 120), EventTempo(Fraction(4), 60)],
        0, min_similarity=0.7,
    )

    link = next(item for item in relations.links if (item.source, item.target) == (0, 2))
    # Matching proves the later audio was sampled at 60 BPM (beats 4..6 map to
    # seconds 2..4), rather than at the initial tempo's seconds 2..3.
    assert link.similarity >= 0.7
    assert link.duration_ratio == pytest.approx(1.0)


def test_causal_reference_projects_only_committed_prior_points_and_invalidation_clears_later():
    spans = [_span(0, 0, 2), _span(1, 2, 4), _span(2, 4, 8)]
    relations = SongRelations(
        spans=tuple(spans),
        descriptors=np.zeros((3, DESCRIPTOR_DIM), dtype=np.float32),
        links=[Relation(0, 2, 0.9, 0.6, 2.0)],
    )
    memory = CausalMotifMemory(spans)
    memory.commit(0, [EventPoint(Fraction(1, 2), "1"), EventPoint(Fraction(3, 2), "2")])
    with pytest.raises(ValueError, match="within span"):
        memory.commit(0, [EventPoint(Fraction(5, 2), "3")])

    reference = memory.reference(2, relations)
    assert reference.source == 0
    assert [point.beat for point in reference.points] == [Fraction(5), Fraction(7)]
    assert reference.provenance == "audio_recurrence_prior_chart"
    # A future target cannot read this output as its source unless it is committed.
    assert memory.reference(1, relations).source is None
    memory.commit(2, [EventPoint(Fraction(5), "9")])
    memory.invalidate_from(2)
    assert memory.reference(2, relations).points == reference.points
    memory.invalidate_from(0)
    assert memory.reference(2, relations).source is None


def test_phase_aligned_relation_shifts_chart_points_and_clips_without_wrapping():
    spans = [_span(0, 0, 8), _span(1, 10, 18)]
    relation = Relation(0, 1, 0.95, 0.8, 1.0, phase_shift_bins=3, phase_bins=24)
    relations = SongRelations(
        spans=tuple(spans),
        descriptors=np.zeros((2, DESCRIPTOR_DIM), dtype=np.float32),
        links=[relation],
    )
    memory = CausalMotifMemory(spans)
    memory.commit(0, [
        EventPoint(Fraction(1, 2), "edge"),
        EventPoint(Fraction(2), "inside"),
        EventPoint(Fraction(15, 2), "late"),
    ])

    reference = memory.reference(1, relations)

    # Three of 24 phase bins advances the target motif by one target beat.
    # The first point crosses the left edge and is dropped instead of wrapped.
    assert [point.beat for point in reference.points] == [Fraction(11), Fraction(33, 2)]
    assert [point.note_text for point in reference.points] == ["inside", "late"]
