"""Regression tests for whole-song windowed event generation.

These pin the three defects that made generated charts unusable: a tracker lag
bin written into the chart clock, output limited to the first training window,
and a compiler that padded the chart out to the audio duration.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from maichart.event_timing import EventPoint, EventTempo, parse_event_chart_text
from maichart.generation import event_cli
from maichart.generation.event_compile import _format_number, compile_event_points
from maichart.generation.event_model import EventVocabulary


# --- the chart clock -------------------------------------------------------


@pytest.mark.parametrize(
    ("detected", "expected"),
    [
        # librosa reports tempogram lag bins: 3000/22 and 1000/11.
        (1500 / 11, 136.5),
        (1000 / 11, 91.0),
        (136.36363636363637, 136.5),
        (90.9090909090909, 91.0),
        (82.20721, 82.0),
        (96.01815, 96.0),
        # Values already on the authored grid are unchanged.
        (150.0, 150.0),
        (155.0, 155.0),
        (128.5, 128.5),
        (199.9, 200.0),
        (173.4, 173.5),
    ],
)
def test_detected_bpm_lands_on_the_authored_half_bpm_grid(detected, expected):
    assert event_cli._musical_bpm(detected) == pytest.approx(expected)


def test_explicit_bpm_is_used_verbatim_and_never_snapped(tmp_path, monkeypatch):
    """A caller who knows the real tempo must be able to pin it exactly."""

    audio_path = tmp_path / "song.wav"
    audio_path.write_bytes(b"")
    output = tmp_path / "out" / "maidata.txt"

    def fake_extract(path, sample_rate=22050):
        from maichart.generation.event_audio import ACOUSTIC_DIM, EventAudio
        return EventAudio(
            np.zeros((4, ACOUSTIC_DIM), dtype=np.float32),
            np.arange(4, dtype=np.float32) * 0.02,
            0.08,
            174.63,
        )

    monkeypatch.setattr(
        event_cli, "load_event_model",
        lambda path, device: (object(), EventVocabulary.from_event_charts([], segment_beats=4),
                             {"epoch": 1, "schema": "maichart-event-generator-checkpoint-v1",
                              "data_contract": {"target_schema": "event_v1", "segment_beats": 4}}),
    )
    monkeypatch.setattr(event_cli, "extract_event_audio", fake_extract)
    monkeypatch.setattr(
        event_cli, "_decode",
        lambda *args, **kwargs: event_cli.EventSequence([Fraction(0)], ["1"]),
    )
    code = event_cli.main([
        "--checkpoint", str(tmp_path / "model.pt"),
        "--audio", str(audio_path),
        "--output", str(output),
        "--device", "cpu",
        "--bpm", "174.63",
    ])
    assert code == 0
    text = output.read_text(encoding="utf-8")
    # The detected 174.63 would quantize to 174.5; an explicit override must win.
    assert "&wholebpm=174.63" in text, text
    assert "174.5" not in text


@pytest.mark.parametrize("value", [0.0, -12.0, float("nan"), float("inf")])
def test_nonpositive_or_nonfinite_bpm_is_rejected(value):
    with pytest.raises(ValueError):
        event_cli._musical_bpm(value)


def test_written_bpm_preserves_the_clock_chosen_by_the_caller():
    # Automatic tempo quantization belongs to _musical_bpm. The lossless
    # compiler must also preserve callers' unquantized or explicit BPMs.
    assert _format_number(136.36363636363637) == "136.36363636363637"
    assert _format_number(91.0) == "91"
    assert _format_number(128.5) == "128.5"
    assert _format_number(174.63) == "174.63"


# --- the 16-beat ceiling ---------------------------------------------------


def test_windows_tile_the_whole_song_not_only_the_first_segment():
    windows = event_cli._segment_windows(Fraction(292), 16)
    assert windows[0] == (Fraction(0), Fraction(16))
    assert windows[-1] == (Fraction(288), Fraction(292))
    assert len(windows) == 19
    # Contiguous and strictly advancing, so no beat is skipped or double-used.
    assert all(end == windows[index + 1][0] for index, (_, end) in enumerate(windows[:-1]))


def test_short_song_still_yields_one_window():
    assert event_cli._segment_windows(Fraction(3, 2), 16) == [(Fraction(0), Fraction(3, 2))]
    assert event_cli._segment_windows(Fraction(0), 16) == []


def test_window_limit_bounds_generation():
    assert len(event_cli._segment_windows(Fraction(292), 16, limit=3)) == 3


def test_relative_positions_are_reanchored_onto_the_song_clock():
    """The checkpoint emits segment-relative beats; they must become absolute.

    Without this step every window repeats beats 0..16 and the chart collapses
    back to a single 16-beat fragment.
    """

    windows = [
        event_cli._DecodedWindow(event_cli.EventSequence([Fraction(0), Fraction(4)], ["1", "2"]),
                                 Fraction(0), Fraction(16), True),
        event_cli._DecodedWindow(event_cli.EventSequence([Fraction(1, 2), Fraction(8)], ["3", "4"]),
                                 Fraction(16), Fraction(32), True),
    ]
    points = event_cli._merge_windows(windows)
    assert [(point.beat, point.note_text) for point in points] == [
        (Fraction(0), "1"),
        (Fraction(4), "2"),
        (Fraction(33, 2), "3"),
        (Fraction(24), "4"),
    ]


def test_merge_drops_duplicate_beats_the_compiler_would_reject():
    windows = [
        event_cli._DecodedWindow(event_cli.EventSequence([Fraction(0)], ["1"]), Fraction(0), Fraction(16), True),
        event_cli._DecodedWindow(event_cli.EventSequence([Fraction(16)], ["2"]), Fraction(0), Fraction(16), True),
    ]
    # The second event would land on the same absolute beat as the next
    # window's start; the compiler allows one EventPoint per beat.
    points = event_cli._merge_windows(windows)
    assert [point.beat for point in points] == [Fraction(0)]


def test_merge_rejects_positions_outside_their_own_window():
    windows = [
        event_cli._DecodedWindow(event_cli.EventSequence([Fraction(40)], ["1"]), Fraction(0), Fraction(16), True),
    ]
    assert event_cli._merge_windows(windows) == []


# --- the manufactured blank tail ------------------------------------------


def test_omitting_end_beat_leaves_no_padded_blank_cells():
    points = [EventPoint(Fraction(0), "1"), EventPoint(Fraction(31, 2), "7")]
    padded, _ = compile_event_points(
        points, bpm=136.5, tempo_changes=[EventTempo(Fraction(0), 136.5)],
        end_beat=Fraction(384121, 882),
    )
    compact, _ = compile_event_points(
        points, bpm=136.5, tempo_changes=[EventTempo(Fraction(0), 136.5)],
    )
    # The fractional audio-duration anchor forced a pathological {3528} grid of
    # empty cells; closing one tail_step after the last note removes both.
    assert "{3528}" in padded
    assert padded.count(",,") > compact.count(",,")
    assert "{3528}" not in compact
    assert len(compact) < len(padded)
    reparsed = parse_event_chart_text(compact, 5, strict_notes=True)
    assert [(point.beat, point.note_text) for point in reparsed.points] == [
        (Fraction(0), "1"), (Fraction(31, 2), "7")
    ]
    # One tail_step past the last note, with no audio-duration anchor.
    assert reparsed.end_beat == Fraction(31, 2) + Fraction(1, 4)
