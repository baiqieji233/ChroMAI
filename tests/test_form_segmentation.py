from fractions import Fraction

import numpy as np
import pytest

from maichart.event_timing import EventTempo
from maichart.form.segmentation import Segmentation, segment_music
from maichart.form.timing import MeterChange, MetricSpan
from maichart.generation.event_audio import ACOUSTIC_DIM, EventAudio


def _audio(duration, changes=()):
    times = np.arange(int(duration * 50) + 1, dtype=np.float32) / 50
    features = np.zeros((len(times), ACOUSTIC_DIM), dtype=np.float32)
    for index, stamp in enumerate(times):
        state = sum(stamp >= boundary for boundary in changes)
        chroma = (2 + state * 3) % 12
        features[index, 80 + chroma] = 1.0
        features[index, 0:8] = np.linspace(-8.0, -1.0, 8, dtype=np.float32)
        features[index, 8 + (state % 8)] = -2.0
        features[index, 92] = 0.2 + 0.05 * (state % 2)
    return EventAudio(features, times, float(duration), 120.0)


def test_audio_novelty_creates_regions_and_meter_snaps_nearby_change_to_bar():
    audio = _audio(32.0, changes=(7.75, 20.0))
    result = segment_music(
        audio, [EventTempo(Fraction(0), 120)], 0,
        meters=[MeterChange(Fraction(0), 4, 4)],
    )

    assert isinstance(result, list)
    assert all(isinstance(span, MetricSpan) for span in result)
    assert result.spans == list(result)
    assert len(result.region_spans) == len(result.regions)
    assert result.span_region_indices == [0, 1, 2]
    assert [region.start for region in result.regions] == [Fraction(0), Fraction(16), Fraction(40)]
    assert [region.end for region in result.regions] == [Fraction(16), Fraction(40), Fraction(64)]
    assert any(row["beat"] == "16" and row["reason"] == "audio_novelty_bar" for row in result.boundary_reasons)
    assert result.audit["resource_cap_is_musical_boundary"] is False


def test_boundary_reasons_and_audit_only_include_accepted_boundaries():
    result = segment_music(
        _audio(20.0, changes=(0.5, 10.0)),
        [EventTempo(Fraction(0), 120)], 0,
        min_seconds=2.0,
    )

    assert result.audit["novelty_candidates"] >= 2
    assert result.audit["accepted_novelty_boundaries"] == 1
    assert [row["reason"] for row in result.boundary_reasons].count("audio_novelty_beat") == 1
    assert "1" not in [row["beat"] for row in result.boundary_reasons]


def test_unknown_meter_has_no_fixed_sixteen_beat_regions_and_cap_is_computational():
    result = segment_music(
        _audio(24.0), [EventTempo(Fraction(0), 120)], 0,
        min_seconds=2.0,
        max_seconds=14.0,
    )

    assert len(result.regions) == 1
    assert result.regions[0].end == Fraction(48)
    assert [(span.start, span.end) for span in result] == [
        (Fraction(0), Fraction(28)), (Fraction(28), Fraction(48)),
    ]
    assert [span.source for span in result] == ["song_start", "resource_cap"]


def test_short_audio_keeps_frame_count_and_returns_single_partial_region():
    result = segment_music(_audio(0.1), [EventTempo(Fraction(0), 120)], 0)

    assert len(result.regions) == 1
    assert result.regions[0].end == Fraction(1, 5)


def test_bpm_change_maps_novelty_to_exact_chart_beat():
    result = segment_music(
        _audio(14.0, changes=(5.0,)),
        [EventTempo(Fraction(0), 120), EventTempo(Fraction(8), 60)],
        0,
        novelty_threshold=0.4,
    )

    # Beat 8 arrives at 4 s, then the 60 BPM section reaches beat 9 at 5 s.
    assert any(region.start == Fraction(9) for region in result.regions)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan")])
def test_rejects_invalid_novelty_threshold(threshold):
    with pytest.raises(ValueError, match="novelty_threshold"):
        segment_music(_audio(4.0), [EventTempo(Fraction(0), 120)], 0, novelty_threshold=threshold)
