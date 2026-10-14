"""Regression coverage for source-time rhythm anchors."""

from fractions import Fraction as F
import math

import pytest

from maichart.event_timing import EventTempo, seconds_at_beat
from maichart.form.generate import exact_candidate
from maichart.form.rhythm_snap import approximate_onsets, validate_source_ms


def test_point_ranking_uses_source_second_for_the_130_bpm_tie_case():
    tempo = [EventTempo(F(0), 130)]
    record = {"beat": str(F(1627, 24)), "second": 31.2884615385,
              "source_second": 31.2800006866, "score": 0.9}
    fixed, report = approximate_onsets([record], tempo, 0, tolerance_ms=50,
        divisions=[4, 8, 12, 16, 24, 32], mode="point")
    assert fixed[0]["beat"] == "271/4"
    assert fixed[0]["rhythm_snap_division"] == 16
    assert report["changed"] == 1


def test_nonfinite_source_second_is_rejected():
    tempo = [EventTempo(F(0), 130)]
    record = {"beat": "1/4", "second": 0.1153846, "source_second": math.nan}
    with pytest.raises(ValueError, match="source times must be finite"):
        approximate_onsets([record], tempo, 0, tolerance_ms=50)


def test_zero_tolerance_and_source_cap_preserve_original_beat():
    tempo = [EventTempo(F(0), 130)]
    record = {"beat": "1627/24", "second": seconds_at_beat(F(1627,24), tempo, 0),
              "source_second": 31.2800006866,
              "score": 0.9}
    unchanged, zero = approximate_onsets([record], tempo, 0, tolerance_ms=0,
        source_tolerance_ms=30, divisions=[4, 8, 16], mode="point")
    assert unchanged == [record] and zero["changed"] == 0
    unchanged, capped = approximate_onsets([record], tempo, 0, tolerance_ms=50,
        source_tolerance_ms=5, mode="point")
    assert unchanged == [record] and capped["changed"] == 0
    assert capped["blocked"] == 1
    moved, _ = approximate_onsets([record], tempo, 0, tolerance_ms=50,
        source_tolerance_ms=20, mode="point")
    assert moved[0]['beat'] == '271/4'
    assert validate_source_ms(20) == 20
    for bad in (-1, 51, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="rhythm_snap_source_ms"):
            validate_source_ms(bad)


def test_source_cap_20_vs_30_controls_strong_five_onset_sequence_repair():
    tempo = [EventTempo(F(0), 130)]
    raw = [141.240005493, 141.36000061, 141.46000671, 141.6000061035, 141.699996948]
    records = [{"beat": str(exact_candidate(second, tempo, 0)),
                "second": seconds_at_beat(exact_candidate(second, tempo, 0), tempo, 0),
                "source_second": second, "score": 0.9} for second in raw]
    at_20, report20 = approximate_onsets(records, tempo, 0, tolerance_ms=50,
        source_tolerance_ms=20, divisions=[4, 8, 12, 16, 24, 32], mode="sequence")
    at_30, report30 = approximate_onsets(records, tempo, 0, tolerance_ms=50,
        source_tolerance_ms=30, divisions=[4, 8, 12, 16, 24, 32], mode="sequence")
    assert [F(item["beat"]) for item in at_20] == [F("306"), F("1225/4"), F("613/2"), F("1841/6"), F("307")]
    assert [F(item["beat"]) for item in at_30] == [F("306"), F("1225/4"), F("613/2"), F("1227/4"), F("307")]
    assert report20["source_tolerance_ms"] == 20 and report30["source_tolerance_ms"] == 30
    assert report30["anchored_runs_count"] >= 1
    for before, after in zip(records, at_30):
        if after['beat'] != before['beat']:
            assert abs(after['second']-before['source_second']) <= .030000001


@pytest.mark.parametrize('offset', [0,1000,10000])
def test_exact_midpoint_ties_do_not_flip_to_triplets_late_in_song(offset):
    tempo = [EventTempo(F(0),130)]
    beat = F(1627,24)+offset
    stamp = seconds_at_beat(beat,tempo,0)
    record = {'beat':str(beat),'second':stamp,'source_second':stamp}
    fixed, _ = approximate_onsets([record],tempo,0,tolerance_ms=50)
    assert F(fixed[0]['beat']) == F(271,4)+offset


def test_zero_source_tolerance_disables_moves_even_when_raw_peak_is_on_grid():
    tempo = [EventTempo(F(0),130)]
    record = {'beat':'1627/24','second':seconds_at_beat(F(1627,24),tempo,0),
              'source_second':seconds_at_beat(F(271,4),tempo,0)}
    unchanged, stats = approximate_onsets([record],tempo,0,tolerance_ms=50,source_tolerance_ms=0)
    assert unchanged == [record]
    assert stats['changed'] == 0


def test_true_12_and_24_subdivision_onsets_are_retained():
    tempo = [EventTempo(F(0), 130)]
    beats = [F(1, 3), F(1, 2), F(2, 3), F(5, 6)]
    records = [{"beat": str(beat), "second": seconds_at_beat(beat, tempo, 0),
                "source_second": seconds_at_beat(beat, tempo, 0), "score": 0.9}
               for beat in beats]
    fixed, report = approximate_onsets(records, tempo, 0, tolerance_ms=50,
        source_tolerance_ms=30, divisions=[4, 8, 12, 16, 24, 32], mode="sequence")
    assert [F(item["beat"]) for item in fixed] == beats
    assert report["changed"] == 0
