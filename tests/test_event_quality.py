from dataclasses import dataclass
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from maichart.generation.conditions import GenerationCondition
from maichart.generation.event_quality import (
    EventNoteError,
    assess_event_chart,
    condition_from_events,
    parse_event_notes,
)


@dataclass(frozen=True)
class Point:
    beat: Fraction
    note_text: str


@dataclass(frozen=True)
class Tempo:
    beat: Fraction
    bpm: float


TEMPO = [Tempo(Fraction(0), 120)]
ROOT = Path(__file__).resolve().parents[1]


def points(*texts):
    return [Point(Fraction(index), text) for index, text in enumerate(texts)]


def condition(events, *, duration=4, tempo=TEMPO, audio=None):
    return condition_from_events(events, tempo, level=13, duration_sec=duration, audio=audio)


def test_complete_tokens_preserve_slash_notes_and_parallel_star_branches():
    parsed = parse_event_notes(points("1-4[8:1]*-6[4:1]/2/Ch[4:1]/34b"), TEMPO, first=1.25)
    assert parsed.syntax_valid
    assert len(parsed.notes) == 5
    slide = parsed.notes[0]
    assert len(slide.slide_segments) == 2
    assert [segment["start_position"] for segment in slide.slide_segments] == [1, 1]
    assert [segment["duration_sec"] for segment in slide.slide_segments] == [0.25, 0.5]
    assert slide.duration_sec == 0.5  # Parallel durations are not added.
    assert slide.time_sec == 1.25
    assert slide.raw == "1-4[8:1]*-6[4:1]"


def test_arbitrary_fractions_are_not_fixed_frame_or_tick_quantized():
    events = [Point(Fraction(1, 7), "1/2"), Point(Fraction(2, 7), "3")]
    parsed = parse_event_notes(events, TEMPO)
    assert parsed.syntax_valid
    assert [note.beat for note in parsed.notes] == [Fraction(1, 7), Fraction(1, 7), Fraction(2, 7)]
    assert all(note.tick == 0 for note in parsed.notes)
    report = assess_event_chart(events, TEMPO)
    assert report["syntax_valid"]
    assert report["offgrid_fractions"]["note_count"] == 3
    assert report["offgrid_fractions"]["beats"] == ["1/7", "2/7"]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("text", ["1/UNKNOWN", "1/", "1-4[8:1]*BAD", "9", "1h[BAD]"])
def test_unsupported_tokens_are_fatal_and_never_partial_conditions(text):
    parsed = parse_event_notes(points(text), TEMPO)
    assert parsed.unsupported
    assert not parsed.syntax_valid
    with pytest.raises(EventNoteError) as error:
        condition(points(text))
    assert error.value.unsupported
    report = assess_event_chart(points(text), TEMPO)
    assert not report["syntax_valid"]
    assert report["statistics_partial"]
    assert report["control"]["actual"] is None
    assert not report["playable_verified"]


@pytest.mark.parametrize("events", [points("1/1"), points("1/1h[4:1]"),
                                     points("1-4[8:1]/1"), points("11"),
                                     [Point(Fraction(0), "1"), Point(Fraction(0), "1")]])
def test_duplicate_button_heads_are_explicit_fatal_errors(events):
    report = assess_event_chart(events, TEMPO)
    assert not report["syntax_valid"]
    assert any(issue["code"] == "duplicate_button_head" for issue in report["fatal_errors"])
    with pytest.raises(EventNoteError):
        condition(events)


def test_split_simultaneous_events_form_one_chord():
    events = [Point(Fraction(0), "1"), Point(Fraction(0), "2")]
    report = assess_event_chart(events, TEMPO)
    assert report["counts"]["chords"] == 1
    assert report["counts"]["occupied_events"] == 1
    assert report["chord_ratio"] == 1


def test_tempo_seconds_and_grid_duration_cross_bpm_boundary():
    tempo = [Tempo(Fraction(0), 120), Tempo(Fraction(1), 60)]
    parsed = parse_event_notes(points("1h[4:2]/2-6[4:2]*-8[#0.75]", "", "3"), tempo, first=2)
    assert parsed.syntax_valid
    assert parsed.notes[-1].time_sec == 3.5
    assert parsed.notes[0].duration_sec == 1.5
    assert [part["duration_sec"] for part in parsed.notes[1].slide_segments] == [1.5, 0.75]
    report = assess_event_chart(points("1", "2", "3"), tempo, first=2, duration_sec=3)
    assert report["note_density_nps"] == 1
    assert report["peak_nps"] == 2


def test_density_uses_song_seconds_and_peak_is_window_nps_not_single_chord():
    spaced = points("1", "2", "3", "4")
    burst = [Point(Fraction(index, 8), str(index + 1)) for index in range(4)]
    slow = condition(spaced, duration=8)
    fast = condition(spaced, duration=4)
    dense = condition(burst, duration=4)
    assert fast.radar["note_density"] == 2 * slow.radar["note_density"]
    assert dense.radar["peak"] > fast.radar["peak"]
    assert dense.metadata["statistics"]["peak_nps"] == 4
    assert dense.metadata["statistics"]["counts"]["chords"] == 0
    chord = condition([Point(Fraction(0), "12345678")], duration=4)
    assert chord.metadata["statistics"]["peak_nps"] == 1
    assert isinstance(fast, GenerationCondition)
    assert fast.to_vector().shape == (16,)
    assert fast.metadata["computed_in_memory"]
    assert not fast.metadata["persisted_to_training_zip"]


def test_type_ratios_counts_entropy_chords_and_slide_paths():
    events = points("1b/2h[4:1]", "3-6[8:1]*-7[4:1]/A1", "Ch[4:1]")
    report = assess_event_chart(events, TEMPO, duration_sec=4)
    assert report["counts"]["notes"] == 5
    assert report["types"] == {"tap": 1, "hold": 1, "slide": 1, "touch": 1, "touch_hold": 1}
    assert report["counts"]["break_notes"] == 1
    assert report["counts"]["slide_branches"] == 2
    assert report["slides"]["paths"] == {"3-6": 1, "3-7": 1}
    assert report["entropy"]["types"] == pytest.approx(1)
    actual = condition(events)
    assert actual.style["tap_bias"] == 0.2
    assert actual.style["hold_bias"] == 0.4
    assert actual.style["touch_bias"] == 0.4
    assert actual.radar["slide_tricky"] > 0


def test_jacks_are_time_local_and_direction_is_not_a_constant():
    assert condition(points("1", "1", "1")).radar["jack"] == 1
    sparse = [Point(Fraction(0), "1"), Point(Fraction(8), "1")]
    assert condition(sparse).radar["jack"] == 0
    forward = condition(points("1", "2", "3", "4"))
    alternating = condition(points("1", "2", "1", "2", "1"))
    assert forward.style["directionality"] == 1
    assert alternating.style["directionality"] == 0


def test_four_beat_motif_reuse_requires_nonconstant_content():
    repeated = points("1", "2", "3", "4", "1", "2", "3", "4")
    unique = points("1", "2", "3", "4", "5", "6", "7", "8")
    constant = points(*(["1"] * 8))
    assert condition(repeated).style["pattern_reuse"] == 0.5
    assert condition(unique).style["pattern_reuse"] == 0
    assert condition(constant).style["pattern_reuse"] == 0
    assert assess_event_chart(constant, TEMPO)["max_repeated_run"] == 8


def test_audio_absence_is_zero_and_onsets_are_only_a_labeled_proxy():
    no_audio = condition(points("1", "2"))
    assert no_audio.style["melody_following"] == 0
    assert not no_audio.metadata["melody_following"]["available"]
    proxy = condition(points("1", "2"), audio={"first": 1, "onsets": [1, 1.5]})
    assert proxy.style["melody_following"] == 1
    assert proxy.metadata["melody_following"]["kind"] == "onset_alignment_proxy"
    assert not proxy.metadata["melody_following"]["true_melody_measurement"]
    assert condition(points("1"), audio={"onsets": []}).style["melody_following"] == 0


def test_event_audio_feature_matrix_can_supply_labeled_onset_proxy():
    features = [[0.0] * 103 for _ in range(3)]
    features[1][92] = 4.0
    audio = {"times": [0.0, 0.5, 1.0], "features": features}
    proxy = condition(points("1", "2"), audio=audio)
    assert proxy.style["melody_following"] == 0.5
    assert proxy.metadata["melody_following"]["feature_source"] == "EventAudio.features[onset]"


@pytest.mark.parametrize("events, expected", [([], "degenerate"),
                                              (points("1", "1", "1"), "degenerate"),
                                              (points("1", "2", "3", "4"), "needs_review"),
                                              (points("1/2", "3/4"), "needs_review"),
                                              (points("1"), "needs_review")])
def test_degenerate_does_not_mean_all_tap_charts_are_rejected(events, expected):
    report = assess_event_chart(events, TEMPO)
    assert report["syntax_valid"]
    assert report["quality_status"] == expected
    assert report["playable_verified"] is False
    assert report["entropy"]["buttons"] >= 0


def test_actual_requested_comparison_exposes_unmeasured_coordinates():
    target = GenerationCondition(radar={"peak": 0.8}, style={"melody_following": 1})
    report = assess_event_chart(points("1", "2"), TEMPO, duration_sec=4, target_condition=target)
    control = report["control"]
    assert control["requested"]["radar"]["peak"] == 0.8
    assert control["actual_minus_requested"]["radar"]["peak"] == pytest.approx(2 / 32 - 0.8)
    assert control["actual_minus_requested"]["style"]["melody_following"] is None
    assert not control["level_estimated"]
    assert control["unmeasured"] == ["level", "melody_following"]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf")])
def test_invalid_duration_interval_is_not_repaired(duration):
    with pytest.raises(ValueError, match="duration_sec"):
        condition(points("1"), duration=duration)
    with pytest.raises(ValueError, match="duration_sec"):
        assess_event_chart(points("1"), TEMPO, duration_sec=duration)


@pytest.mark.parametrize("tempo", [[], [(0, 0)], [(0, float("nan"))], [(1, 120)],
                                    [(0, 120), (0, 150)]])
def test_invalid_tempo_is_explicit(tempo):
    report = assess_event_chart(points("1"), tempo)
    assert not report["syntax_valid"]
    assert report["fatal_errors"][0]["code"] == "invalid_tempo_map"


@pytest.mark.parametrize("text", ["1h[4:0]", "1h[4:-1]", "1-4[#0]", "1-4[0.5##-1]"])
def test_zero_and_negative_durations_are_accepted_with_a_warning(text):
    """Real charts use these as placeholders or as a shift earlier.

    `[1:0]` on a hold appears 158 times in this corpus (essentially always on
    holds); a negative value is how a charter moves a note -- or, when it leads,
    the whole chart -- earlier in time.  Neither changes where the note's head
    sits, which is all the training token stream encodes, so rejecting the entire
    chart over one of them costs real data for no benefit.
    """

    parsed = parse_event_notes(points(text), TEMPO)
    assert parsed.syntax_valid, f"{text} must not reject the chart"
    assert not parsed.fatal_errors
    assert parsed.warnings, f"{text} must be reported, not silently ignored"


def test_negative_delay_is_accepted_because_it_shifts_the_chart():
    """A negative delay is legal maimai, not a broken duration.

    It encodes the blank space a machine needs before the chart body: the chart
    moves earlier as a whole, and whatever lands before audio time 0 behaves like
    silent audio prepended to the file.  Only the duration part must be positive.
    """

    parsed = parse_event_notes(points("1-4[-1##0.5]"), TEMPO)
    assert parsed.syntax_valid, "a leading negative delay must not reject the chart"
    assert not parsed.fatal_errors
    assert [warning["code"] for warning in parsed.warnings] == ["negative_delay"]
    assert parsed.warnings[0]["delay"] == -1.0


def run_audit(path, *args):
    return subprocess.run([sys.executable, str(ROOT / "tools" / "audit_generated_chart.py"),
                           str(path), *args], capture_output=True, text=True, check=False)


def test_audit_cli_fractional_grid_checkpoint_and_recent_statistics(tmp_path):
    path = tmp_path / "maidata.txt"
    path.write_text("&title=Audit\n&first=1.25\n&wholebpm=120\n&lv_5=13+\n"
                    "&inote_5=(120){7}1,2,3,4,E\n", encoding="utf-8")
    checkpoint = tmp_path / "model.pt"
    torch.save({"schema": "test-event-checkpoint", "paradigm": "exact_events",
                "epoch": 7, "paradigm_history": [{"epoch": index, "nps": index}
                                                 for index in range(6)]}, checkpoint)
    (tmp_path / "metrics.jsonl").write_text(
        "\n".join(json.dumps({"epoch": index, "loss": index / 10}) for index in range(4)),
        encoding="utf-8",
    )
    result = run_audit(path, "--checkpoint", str(checkpoint), "--recent", "2")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    chart = payload["charts"][0]
    assert chart["syntax_valid"]
    assert chart["offgrid_fractions"]["note_count"] == 3
    assert chart["source_level"] == 13.5
    assert chart["duration_sec"] == pytest.approx(8 / 7)
    assert payload["checkpoint"]["paradigm"] == "exact_events"
    assert len(payload["checkpoint"]["recent_statistics"]["paradigm_history"]) == 2
    assert len(payload["checkpoint"]["recent_statistics"]["metrics.jsonl"]) == 2


def test_audit_cli_fatal_errors_have_json_and_nonzero_exit(tmp_path):
    path = tmp_path / "maidata.txt"
    path.write_text("&wholebpm=120\n&inote_5=1/1/UNKNOWN,E\n", encoding="utf-8")
    result = run_audit(path)
    assert result.returncode == 1
    report = json.loads(result.stdout)["charts"][0]
    assert not report["syntax_valid"]
    assert {issue["code"] for issue in report["fatal_errors"]} == {
        "unsupported_token", "duplicate_button_head"}


def test_audit_cli_reports_missing_paradigm_without_inference(tmp_path):
    path = tmp_path / "maidata.txt"
    path.write_text("&inote_5=1,2,E\n", encoding="utf-8")
    checkpoint = tmp_path / "model.pt"
    torch.save({"model_state_dict": {"weight": torch.zeros(1)}, "epoch": 2}, checkpoint)
    result = run_audit(path, "--checkpoint", str(checkpoint))
    assert result.returncode == 0
    summary = json.loads(result.stdout)["checkpoint"]
    assert summary["paradigm"] is None
    assert not summary["paradigm_available"]
    assert not summary["statistics_available"]
