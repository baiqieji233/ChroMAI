from fractions import Fraction as F

import numpy as np

from maichart.event_timing import EventPoint, EventTempo
from maichart.form.codec import RationalCodec
from maichart.form.data import FormDataset, FormSong, input_example, organize_song
from maichart.generation.event_audio import ACOUSTIC_DIM, EventAudio
from maichart.generation.sustain_state import SustainStateIndex


def _repeating_audio():
    times = np.arange(451, dtype=np.float32) / 50
    features = np.zeros((len(times), ACOUSTIC_DIM), dtype=np.float32)
    for index, stamp in enumerate(times):
        beat = stamp * 2
        region = min(int(beat // 6), 2)
        phase = (beat % 6) / 6
        kind = (0, 1, 0)[region]
        features[index, 80 + (2 + 4 * kind)] = 1.0
        features[index, 92] = float(phase < (0.2 if kind == 0 else 0.45))
        features[index, 0:8] = np.linspace(-8, -1, 8, dtype=np.float32) * (kind + 1)
    return EventAudio(features, times, 9.0, 120.0)


def _organized():
    return organize_song(
        _repeating_audio(), [EventTempo(F(0), 120)], 0,
        min_seconds=0.2, max_seconds=0.5,
    )


def test_compute_span_reference_is_region_scoped_then_clipped_to_current_chunk():
    song = _organized()
    assert len(song.regions) == 3
    assert len(song.spans) == 18
    assert song.span_region_indices == [index // 6 for index in range(18)]
    assert [(link.source, link.target) for link in song.relations.links] == [(0, 2)]

    points = [
        EventPoint(F(1, 2), "1"), EventPoint(F(3, 2), "2"),
        EventPoint(F(13, 2), "3"), EventPoint(F(15, 2), "4"),
    ]
    example = input_example(song, 12, points)
    codec = RationalCodec()
    projected = codec.decode(example["motif"].tolist(), origin=song.spans[12].start)

    assert example["meta"]["region_index"] == 2
    assert example["meta"]["reference_source"] == 0
    assert [point.beat for point in projected] == [F(25, 2)]


def test_dataset_skeleton_contains_only_masked_future_positions():
    song = _organized()
    points = [
        EventPoint(F(1, 2), "1"), EventPoint(F(3, 2), "2"),
        EventPoint(F(13, 2), "3"), EventPoint(F(15, 2), "4"),
        EventPoint(F(25, 2), "5"), EventPoint(F(27, 2), "6"),
    ]
    codec = RationalCodec()
    dataset = FormDataset.__new__(FormDataset)
    dataset.codec = codec
    dataset.items = [(0, 12, codec.encode([points[-2]], origin=F(12)), 0)]
    dataset.songs = [(song, points, 13.0, 1.0, SustainStateIndex(points, song.tempo, song.first))]
    dataset.seed = 0
    dataset.max_memory_tokens = 128

    item = dataset[0]
    skeleton = item["skeleton"].tolist()
    assert skeleton.count(codec.mask_id) == 2
    assert item["relation_valid"].item() is True
    assert item["meta"]["reference_source"] == 0


def test_legacy_form_song_constructor_defaults_to_identity_region_mapping():
    song = _organized()
    legacy = FormSong(song.audio, song.tempo, song.first, song.meters,
                      song.spans, song.relations, song.audit)

    assert legacy.regions == song.spans
    assert legacy.span_region_indices == list(range(len(song.spans)))


def test_context_report_counts_actual_retained_history_and_keeps_older_sustain_state():
    song = _organized()
    history = [EventPoint(F(0),'1h[1:4]')]+[EventPoint(F(i,2),'2') for i in range(1,24)]
    codec = RationalCodec()
    example = input_example(song,12,history,codec=codec,max_memory_tokens=20)
    retained = codec.decode(example['history'].tolist(),origin=history[0].beat)
    assert retained[0].beat > 0
    assert example['meta']['history_events_available']==len(history)
    assert example['meta']['history_events_kept']==len(retained)<len(history)
    assert example['meta']['history_tokens']==example['history'].numel()<=20
    assert example['meta']['history_coverage_seconds'] == float((F(12)-retained[0].beat)/2)
    assert example['meta']['state_count']==1


def test_late_compute_span_retains_overlapping_long_region_motif():
    from maichart.form.recurrence import DESCRIPTOR_DIM, Relation, SongRelations
    from maichart.form.timing import MetricSpan

    def span(index, start, end):
        return MetricSpan(index, F(start), F(end), 4, 4, (), F(4), (), False, 1., 'test')

    regions = [span(0, 0, 40), span(1, 80, 120)]
    spans = [span(0, 0, 40), span(1, 80, 110), span(2, 110, 120)]
    relations = SongRelations(tuple(regions), np.zeros((2, DESCRIPTOR_DIM), dtype=np.float32),
        [Relation(0, 1, .95, .8, 1.)], layout_descriptors=np.zeros((2, 512), dtype=np.float32))
    audio = EventAudio(np.zeros((61, ACOUSTIC_DIM), dtype=np.float32),
        np.arange(61, dtype=np.float32), 60., 120.)
    song = FormSong(audio, [EventTempo(F(0), 120)], 0., [], spans, relations, {},
        regions=regions, span_region_indices=[0, 1, 1], quality_profile='musical_evidence_v1')
    history = [EventPoint(F(index, 10), '1') for index in range(400)]
    codec = RationalCodec()
    example = input_example(song, 2, history, codec=codec, max_memory_tokens=2048)
    motif = codec.decode(example['motif'].tolist(), origin=F(110))

    assert example['meta']['reference_source'] == 0
    assert len(motif) == 100
    assert motif[0].beat == 110
    assert motif[-1].beat == F(1199, 10)
