"""A negative chart offset must become silence in front of the audio.

A maimai chart needs blank space before its body.  When the audio has no
lead-in, the charter writes a negative delay, which moves the whole chart
earlier: the part that lands before audio time 0 has no music behind it and
behaves exactly like silent audio prepended to the file.

Two defects this pins down:

* the validator rejected such charts outright ("delay must be nonnegative"),
  dropping legal, high-quality charts from training;
* the audio sampler clamped the start to 0, so those beats were rendered from
  the *first* audio frame -- the window's music was shifted relative to its
  notes instead of being silent.
"""

from fractions import Fraction

import numpy as np
import pytest

from maichart.event_timing import EventTempo, parse_event_chart_text
from maichart.generation.event_audio import (
    ACOUSTIC_DIM,
    CONTEXT_KEYS,
    CONTEXT_OFFSET,
    EventAudio,
    INPUT_DIM,
    audio_segment_tensor,
    compress_event_audio,
)

TEMPO = [EventTempo(Fraction(0), 120.0)]
FRAME_RATE = 50.0


def audio(features_value: float = 1.0, seconds: float = 4.0) -> EventAudio:
    frames = int(seconds * FRAME_RATE)
    return EventAudio(
        features=np.full((frames, ACOUSTIC_DIM), features_value, dtype=np.float32),
        times=np.arange(frames, dtype=np.float32) / FRAME_RATE,
        duration_sec=seconds,
        tempo_bpm=120.0,
        tempo_changes=[],
        metadata={},
    )


def chart_text(first: float, notes: str = "1,2,3,4,") -> str:
    return (f"&title=T\n&wholebpm=120\n&first={first}\n&lv_5=13\n"
            f"&inote_5=(120){{4}}{notes}\n")


def test_negative_first_renders_silence_before_the_audio():
    """The pre-audio region must be zeros, not a repeat of the first frame."""

    source = audio(1.0, 4.0)
    tensor = audio_segment_tensor(
        source, -0.5, 1.5, song_duration=4.0, bpm=120.0, start_beat=0.0,
        first=-0.5, tempo_changes=TEMPO,
    )
    values = tensor.numpy()
    times = -0.5 + np.arange(values.shape[0]) / FRAME_RATE
    before = times < 0.0
    after = times >= 0.0
    assert before.any() and after.any()
    # Silence in front, real audio behind.
    assert np.allclose(values[before][:, :ACOUSTIC_DIM], 0.0), "the missing audio must be silent"
    assert not np.allclose(values[after][:, :ACOUSTIC_DIM], 0.0)


def test_no_negative_first_still_samples_normally():
    source = audio(1.0, 4.0)
    # At 120 BPM, audio second 0.5 is chart beat 1.
    tensor = audio_segment_tensor(
        source, 0.5, 1.5, song_duration=4.0, bpm=120.0, start_beat=1.0,
        first=0.0, tempo_changes=TEMPO,
    )
    values = tensor.numpy()
    assert values.shape[0] > 0
    assert np.all(np.isfinite(values))
    # No silence injected when the chart does not start early.
    assert not np.allclose(values[:, :ACOUSTIC_DIM], 0.0)


def test_global_memory_covers_the_chart_start_when_it_starts_early():
    """The full-song memory must begin at the chart, not 0.5 s into the music."""

    source = audio(1.0, 4.0)
    memory = compress_event_audio(
        source, song_duration=4.0, bpm=120.0, first=-0.5, tempo_changes=TEMPO, frames=64,
    ).numpy()
    assert memory.shape == (64, INPUT_DIM)
    # Chart beat 0 is 0.5 s before the audio, so the first frames are the silent
    # pre-roll and later ones carry music.
    assert np.allclose(memory[0, :ACOUSTIC_DIM], 0.0)
    assert not np.allclose(memory[-1, :ACOUSTIC_DIM], 0.0)
    # beat_progress (context column 5) starts at 0, never negative.
    assert memory[0, CONTEXT_OFFSET + 5] == pytest.approx(0.0, abs=1e-6)


def test_leading_negative_delay_remains_note_local():
    """A negative slide wait is local and clamped like the MiaCode parser."""

    chart = parse_event_chart_text(chart_text(0.0, "1>-2[-0.4##0.5],3,4,"), 5)
    assert chart.time_offset == 0.0
    assert chart.audio_first == 0.0
    assert chart.points[0].beat == 0


def test_positive_or_absent_delay_leaves_the_offset_alone():
    assert parse_event_chart_text(chart_text(0.0, "1>-2[0.4##0.5],3,4,"), 5).time_offset == 0.0
    assert parse_event_chart_text(chart_text(0.0, "1>-2[-0.4##0.5],3,4,"), 5).time_offset == 0.0
    assert parse_event_chart_text(chart_text(0.0), 5).time_offset == 0.0
    # `&first` still works on its own, negative or positive.
    assert parse_event_chart_text(chart_text(-0.3), 5).audio_first == pytest.approx(-0.3)
    assert parse_event_chart_text(chart_text(0.3), 5).audio_first == pytest.approx(0.3)


def test_chart_offset_combines_with_first():
    chart = parse_event_chart_text(chart_text(-0.1, "1>-2[-0.4##0.5],3,4,"), 5)
    assert chart.audio_first == pytest.approx(-0.1)


def test_difficulty_specific_first_remains_the_global_clock():
    chart = parse_event_chart_text(
        "&wholebpm=120\n&first=-0.1\n&first_5=-0.75\n&lv_5=13\n"
        "&inote_5=1-4[-0.4##0.5],2,3,E\n",
        5,
    )

    assert chart.first == pytest.approx(-0.75)
    assert chart.audio_first == pytest.approx(-0.75)
    assert chart.time_offset == 0.0


def test_negative_first_and_negative_note_wait_keep_distinct_clocks():
    from maichart.generation.muri.parser import parse_markers
    from maichart.generation.event_quality import parse_event_notes
    from maichart.generation.event_compile import compile_event_points

    chart = parse_event_chart_text(
        chart_text(-0.5, "1-4[-0.4##0.5],(180),2,3,"), 5
    )
    inote = "(120){4}1-4[-0.4##0.5],(180),2,3,E"
    marker, errors, _warnings = parse_markers(inote, first=chart.first)
    notes = parse_event_notes(chart.points, chart.tempo_changes, chart.audio_first)
    compiled_text, _compiled_ir = compile_event_points(
        chart.points, tempo_changes=chart.tempo_changes,
        first=chart.first, end_beat=chart.end_beat,
    )
    compiled = parse_event_chart_text(compiled_text, 5)

    assert not errors
    assert chart.audio_first == pytest.approx(-0.5)
    assert compiled.first == pytest.approx(-0.5)
    assert compiled.time_offset == 0.0
    assert compiled.tempo_changes == chart.tempo_changes
    assert notes.points[0].notes[0].time_sec == pytest.approx(-0.5)
    assert marker[0]["second"] == pytest.approx(-0.5)
    assert marker[0]["slide_trace_second"] == pytest.approx(-0.5)
    assert notes.points[1].notes[0].time_sec == pytest.approx(1 / 3)
    assert marker[1]["second"] == pytest.approx(1 / 3)


def test_static_slide_trace_clamps_negative_wait_like_miacode():
    from maichart.generation.post_process import _extract

    points = [type("Point", (), {"beat": Fraction(0), "note_text": "1-4[-0.4##0.5]"})()]
    events, _serializable, coverage = _extract(points, TEMPO, -0.5)

    assert coverage["status"] == "static_only"
    assert events[0].trajectories[0].start_time == pytest.approx(-0.5)


def test_negative_delay_is_accepted_and_reported():
    from maichart.generation.event_quality import parse_event_notes

    class Point:
        def __init__(self, beat, text):
            self.beat = beat
            self.note_text = text

    accepted = parse_event_notes([Point(Fraction(0), "1<2[-0.5##1.0]")], TEMPO)
    assert accepted.syntax_valid, "a negative delay is legal maimai"
    assert [warning["code"] for warning in accepted.warnings] == ["negative_delay"]
    assert accepted.warnings[0]["delay"] == pytest.approx(-0.5)


def test_hold_placeholder_duration_is_accepted():
    """`[1:0]` on a hold is a placeholder, not a broken chart.

    Measured on this corpus: 158 occurrences across 657 difficulties, 150 of them
    on holds and 8 on touch holds, at every position in the song (52 of them in
    the last 10%).  Rejecting the chart over one of them was dropping real data.
    """

    from maichart.generation.event_quality import parse_event_notes

    class Point:
        def __init__(self, beat, text):
            self.beat = beat
            self.note_text = text

    parsed = parse_event_notes([Point(Fraction(312), "E5h[1:0]/D5h[1:0]")], TEMPO)
    assert parsed.syntax_valid
    codes = {warning["code"] for warning in parsed.warnings}
    assert "zero_or_negative_grid_duration" in codes
    assert not parsed.fatal_errors


def test_negative_grid_count_does_not_change_the_global_chart_clock():
    """Unrecognized/local negative grid syntax cannot override ``&first``."""

    chart = parse_event_chart_text(chart_text(0.0, "5-1[-16:1],2,3,4,"), 5)
    assert chart.time_offset == 0.0
    assert chart.audio_first == 0.0
