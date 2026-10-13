from fractions import Fraction

import numpy as np

from maichart.event_timing import parse_event_chart_text
from maichart.builder import build_chart_ir_by_difficulty_index
from maichart.generation.event_audio import (
    ACOUSTIC_DIM,
    CONTEXT_OFFSET,
    EventAudio,
    audio_segment_tensor,
    compress_event_audio,
)
from maichart.maidata import parse_maidata_metadata, resolve_chart_first
from maichart.timing import tokenize_chart_difficulty_timing
from maichart.training.quality_annotations import chart_sha256
from maichart.validation import validate_raw_maidata_chart


def _chart(first='1.25', slot_first=''):
    slot_line = f'&first_4={slot_first}\n' if slot_first != '' else ''
    return ('&title=First\n&wholebpm=120\n&first=' + first + '\n' + slot_line
            + '&lv_4=12\n&inote_4=(120){4}1,2,E\n'
            + '&lv_5=13\n&inote_5=(120){4}3,4,E\n')


def test_difficulty_first_overrides_global_and_zero_is_not_treated_as_missing():
    parsed = parse_maidata_metadata(_chart(slot_first='0'))
    assert resolve_chart_first(parsed, 4) == '0'
    assert resolve_chart_first(parsed, 5) == '1.25'

    slot4 = parse_event_chart_text(_chart(slot_first='0'), 4)
    slot5 = parse_event_chart_text(_chart(slot_first='0'), 5)
    assert slot4.first == 0.0
    assert slot5.first == 1.25
    assert slot4.points[0].beat == slot5.points[0].beat == Fraction(0)


def test_legacy_timing_and_quality_identity_use_slot_first(tmp_path):
    source = tmp_path / 'maidata.txt'
    source.write_text(_chart(slot_first='-2.5'), encoding='utf-8')
    parsed = parse_maidata_metadata(source.read_text(encoding='utf-8'))
    points4 = tokenize_chart_difficulty_timing(parsed, 4, initial_bpm=120)
    points5 = tokenize_chart_difficulty_timing(parsed, 5, initial_bpm=120)
    assert [point.time_sec for point in points4] == [-2.5, -2.0, -1.5]
    assert [point.time_sec for point in points5] == [1.25, 1.75, 2.25]
    ir = build_chart_ir_by_difficulty_index(parsed, 4)
    assert ir.metadata.offset == -2.5
    assert ir.notes[0].time_sec == -2.5

    hash4_before = chart_sha256(source, 4)
    hash5_before = chart_sha256(source, 5)
    source.write_text(_chart(first='9', slot_first='-2.5'), encoding='utf-8')
    assert chart_sha256(source, 4) == hash4_before
    assert chart_sha256(source, 5) != hash5_before


def test_legacy_validation_accepts_a_negative_first_beyond_one_second():
    report = validate_raw_maidata_chart(parse_maidata_metadata(_chart(slot_first='-2.5')))
    assert not any(issue.code == 'non-monotonic-time' for issue in report.issues)


def test_short_audio_allows_far_negative_first_and_keeps_context_clock():
    times = np.arange(50, dtype=np.float32) / 50
    features = np.ones((len(times), ACOUSTIC_DIM), dtype=np.float32)
    audio = EventAudio(features, times, 1.0, 120.0)
    tensor = audio_segment_tensor(audio, -2.0, -1.0, song_duration=1.0,
                                  bpm=120.0, start_beat=0, first=-2.0,
                                  tempo_changes=[(Fraction(0), 120.0)]).numpy()
    assert np.allclose(tensor[:, :ACOUSTIC_DIM], 0.0)
    assert np.allclose(tensor[:, ACOUSTIC_DIM:CONTEXT_OFFSET], 0.0)
    assert np.isfinite(tensor).all()
    assert not np.allclose(tensor[:, CONTEXT_OFFSET:], 0.0)


def test_global_memory_silences_negative_audio_seconds_in_acoustic_and_rhythm():
    times = np.arange(50, dtype=np.float32) / 50
    features = np.ones((len(times), ACOUSTIC_DIM), dtype=np.float32)
    audio = EventAudio(features, times, 1.0, 120.0)
    tensor = compress_event_audio(audio, song_duration=1.0, bpm=120.0,
                                 first=-2.0,
                                 tempo_changes=[(Fraction(0), 120.0)], frames=64).numpy()
    negative = np.linspace(-2.0, 1.0, 64, endpoint=False) < 0.0
    assert negative.any()
    assert np.allclose(tensor[negative, :CONTEXT_OFFSET], 0.0)
    assert np.isfinite(tensor).all()
    assert not np.allclose(tensor[negative, CONTEXT_OFFSET:], 0.0)
