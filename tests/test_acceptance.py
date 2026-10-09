"""Acceptance metrics must be correct and must actually discriminate.

The blank-structure metrics are the ones the project needs: a window that runs
out of token budget leaves its remaining beats empty, so the metric has to
report a long blank run even when the chart is otherwise dense.  The audio-follow
metric similarly has to separate a chart that tracks the music from one that
ignores it.
"""

from fractions import Fraction

import numpy as np
import pytest

from maichart.event_timing import EventPoint, EventTempo
from maichart.generation.acceptance import (
    GOLD_METRICS,
    audio_follow_metrics,
    configuration_metrics,
    evaluate,
    evaluate_maidata,
    gold_reference,
    metric_bundle,
    structural_metrics,
    timeline_series,
)
from maichart.generation.event_audio import ACOUSTIC_DIM, RELATIVE_RMS_COLUMN, EventAudio

TEMPO = [EventTempo(Fraction(0), 120.0)]  # 1 beat = 0.5 s


def chart(beats):
    return [EventPoint(Fraction(beat), "1") for beat in beats]


def fake_audio(intensity, frame_seconds=0.02):
    """EventAudio whose relative-RMS column follows the given per-frame values."""

    features = np.zeros((len(intensity), ACOUSTIC_DIM), dtype=np.float32)
    features[:, RELATIVE_RMS_COLUMN] = intensity
    times = (np.arange(len(intensity)) * frame_seconds).astype(np.float32)
    return EventAudio(
        features=features, times=times,
        duration_sec=float(times[-1]) if len(times) else 0.0,
        tempo_bpm=120.0, tempo_changes=[], metadata={},
    )


def test_blank_structure_counts_internal_holes():
    # Measures 0 and 2 have notes, measure 1 is empty.
    report = structural_metrics(chart([0, 1, 2, 3, 8, 9]), TEMPO)
    assert report["measures"] == 3
    assert report["blank_measures"] == 1
    assert report["blank_fraction"] == pytest.approx(1 / 3)
    assert report["max_blank_run_measures"] == 1
    # One measure at 120 BPM is 2 seconds.
    assert report["max_blank_run_sec"] == pytest.approx(2.0)
    assert report["notes"] == 6


def test_blank_structure_reports_a_long_hole():
    # The failure this project hit: content stops, then resumes much later.
    report = structural_metrics(chart([0] + [40]), TEMPO)
    assert report["measures"] == 11
    assert report["blank_measures"] == 9
    assert report["max_blank_run_measures"] == 9
    assert report["max_blank_run_sec"] == pytest.approx(18.0)  # 9 measures x 2 s
    assert report["blank_fraction"] == pytest.approx(9 / 11)


def test_structural_metrics_handle_an_empty_chart():
    report = structural_metrics([], TEMPO)
    assert report["notes"] == 0
    assert report["blank_fraction"] is None
    assert report["max_blank_run_sec"] == 0.0


def test_audio_follow_separates_musical_from_ignoring_charts():
    # 60 s of audio: loud during 10-20 s and 30-40 s, quiet elsewhere.
    # At 120 BPM one beat is 0.5 s, so those windows are beats 20-40 and 60-80.
    frames, frame_seconds = 3000, 0.02
    envelope = np.full(frames, 0.4, dtype=np.float32)
    for start, end in ((10.0, 20.0), (30.0, 40.0)):
        envelope[int(start / frame_seconds):int(end / frame_seconds)] = 2.0
    audio = fake_audio(envelope, frame_seconds)
    audio.duration_sec = frames * frame_seconds

    follows = [beat for beat in range(0, 120) if 20 <= beat < 40 or 60 <= beat < 80]
    ignores = [beat for beat in range(0, 120) if 0 <= beat < 20 or 40 <= beat < 60]

    good = audio_follow_metrics(chart(follows), audio, TEMPO, duration_sec=audio.duration_sec)
    bad = audio_follow_metrics(chart(ignores), audio, TEMPO, duration_sec=audio.duration_sec)
    assert good["rms_density_correlation"] > 0.5
    # The 2 s window smears section boundaries, so the anti-aligned chart scores
    # clearly negative rather than exactly -1.
    assert bad["rms_density_correlation"] < -0.3
    assert good["windows"] > 20


def test_timeline_series_marks_blank_spans():
    audio = fake_audio(np.ones(1000, dtype=np.float32))
    audio.duration_sec = 20.0
    series = timeline_series(chart([0, 1, 20, 21]), audio, TEMPO, duration_sec=20.0, resolution=40)
    assert len(series["points"]) == 40
    assert series["blank_spans"], "the hole between beat 1 and beat 20 must be reported"
    # Blanks are detected per measure, so the hole spans measure 1 to measure 4:
    # beats 4..20, i.e. 2.0 s to 10.0 s at 120 BPM.
    start, end = series["blank_spans"][0]
    assert 1.5 < start < 2.5 and 9.5 < end < 10.5


def test_gold_reference_and_evaluate_verdicts():
    metric_sets = []
    for index in range(21):
        metric_sets.append({
            "note_density_nps": 4.0 + index * 0.1,
            "peak_nps": 12.0 + index,
            "chord_ratio": 0.25,
            "slide_ratio": 0.12,
            "touch_ratio": 0.04,
            "jack_ratio": 0.12,
            "entropy_buttons": 0.98,
            "blank_fraction": 0.05,
            "max_blank_run_sec": 2.5,
            "coverage_ratio": 0.97,
            "rms_density_correlation": 0.5,
        })
    reference = gold_reference(metric_sets)
    assert reference["charts"] == 21
    assert reference["metrics"]["note_density_nps"]["p10"] <= 5.0

    inside = dict(metric_sets[10])
    verdicts = {item["key"]: item["verdict"] for item in evaluate(inside, reference)["criteria"]}
    assert verdicts["note_density_nps"] == "pass"
    assert verdicts["blank_fraction"] == "pass"

    broken = dict(metric_sets[10])
    broken.update({"blank_fraction": 0.34, "max_blank_run_sec": 32.0, "syntax_valid": False,
                   "duplicate_head_count": 2})
    report = evaluate(broken, reference)
    failing = {item["key"] for item in report["criteria"] if item["verdict"] == "fail"}
    assert {"blank_fraction", "max_blank_run_sec", "syntax_valid"} <= failing
    assert report["verdict"] == "fail"
    assert report["failures"] >= 3


def test_evaluate_without_reference_reports_unknowns_not_failures():
    report = evaluate({"syntax_valid": True, "duplicate_head_count": 0, "blank_fraction": 0.03})
    verdicts = {item["key"]: item["verdict"] for item in report["criteria"]}
    assert verdicts["syntax_valid"] == "pass"
    assert verdicts["note_density_nps"] == "unknown"
    assert report["failures"] == 0


def test_metric_bundle_reports_every_gold_metric():
    # A varying envelope is required: a constant one makes the correlation
    # undefined, which is correct behaviour but would hide a missing key.
    envelope = np.concatenate([np.full(1000, 0.5, dtype=np.float32),
                               np.full(1000, 2.0, dtype=np.float32)])
    audio = fake_audio(envelope)
    audio.duration_sec = 40.0
    metrics = metric_bundle(chart([0, 1, 2, 3, 8, 9, 16] + list(range(40, 60))),
                            audio, TEMPO, duration_sec=40.0)
    for key, _label, _help in GOLD_METRICS:
        assert key in metrics, key
        assert metrics[key] is not None, key
    # The last note sits at beat 59, so measure indices run 0..14.
    assert metrics["measures"] == 15
    assert metrics["syntax_valid"] is True


def test_configuration_metrics_separate_homogeneous_from_varied_charts():
    """The "one long stretch of one style" failure needs a number, not an eye.

    A chart that repeats the same bar for 64 beats must score a long run and a
    low switch rate; a chart that alternates configurations must score the
    opposite.  Real gold charts sit around 4-11 bars and 0.32-0.65 respectively.
    """

    homogeneous = [EventPoint(Fraction(beat), "1") for beat in range(0, 65)]
    # One configuration per bar, cycling: singles, chords, slides, touches.
    varied = []
    for base in range(0, 64, 16):
        for offset in range(4):
            varied.append(EventPoint(Fraction(base + offset), "1"))
        for offset in range(4):
            varied.append(EventPoint(Fraction(base + 4 + offset), "2/3"))
        for offset in range(4):
            varied.append(EventPoint(Fraction(base + 8 + offset), "4>7[4:1]"))
        for offset in range(4):
            varied.append(EventPoint(Fraction(base + 12 + offset), "E"))
    flat = configuration_metrics(homogeneous, TEMPO)
    mixed = configuration_metrics(varied, TEMPO)
    assert flat["longest_config_run_bars"] > 12
    assert flat["configuration_switch_rate"] < 0.1
    assert mixed["configuration_switch_rate"] > flat["configuration_switch_rate"]
    assert mixed["longest_config_run_bars"] <= 2
    assert mixed["section_variety"] > flat["section_variety"]
    assert set(flat["configuration_labels"]) and set(mixed["configuration_labels"])


def test_configuration_metrics_handle_an_empty_chart():
    report = configuration_metrics([], TEMPO)
    assert report["longest_config_run_bars"] is None
    assert report["configuration_switch_rate"] is None


def test_metric_bundle_includes_the_homogeneity_diagnostics():
    audio = fake_audio(np.concatenate([np.full(500, 0.5, dtype=np.float32),
                                       np.full(500, 2.0, dtype=np.float32)]))
    audio.duration_sec = 20.0
    metrics = metric_bundle([EventPoint(Fraction(beat), "1") for beat in range(0, 20)],
                            audio, TEMPO, duration_sec=20.0)
    assert metrics["longest_config_run_bars"] is not None
    assert metrics["configuration_switch_rate"] is not None
    assert "configuration_labels" in metrics
    # The new keys must be part of the gold comparison too, or the WebUI would
    # never show them.
    keys = {key for key, _label, _help in GOLD_METRICS}
    assert {"longest_config_run_bars", "configuration_switch_rate", "section_variety"} <= keys


def test_evaluate_maidata_end_to_end(tmp_path):
    maidata = tmp_path / "maidata.txt"
    # Four notes, then four empty cells (five commas) = one full empty measure,
    # then four more notes.
    maidata.write_text(
        "&title=Acceptance\n&artist=none\n&first=0\n&wholebpm=120\n"
        "&lv_5=13\n&des_5=Tester\n&inote_5=(120){4}1,2,3,4,,,,,5,6,7,8\n",
        encoding="utf-8",
    )
    report = evaluate_maidata(maidata, difficulty=5, reference=None, resolution=32)
    assert report["schema"] == "maichart-acceptance-v1"
    assert report["metrics"]["notes"] == 8
    # Measure 0 holds beats 0-3, measures 1 holds the rest, measures 2 holds the
    # final four notes: exactly one empty measure out of three.
    assert report["metrics"]["measures"] == 3
    assert report["metrics"]["blank_measures"] == 1
    assert report["metrics"]["blank_fraction"] == pytest.approx(1 / 3)
    assert len(report["timeline"]["points"]) == 32
    assert report["audio"] is None
    assert any(item["key"] == "syntax_valid" for item in report["criteria"])
