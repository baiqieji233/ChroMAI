from fractions import Fraction as F
import math
import random
from unittest.mock import patch

import torch

from maichart.event_timing import EventPoint, EventTempo
from maichart.generation.sustain_state import SustainStateIndex


def test_natural_parser_roundoff_across_zero_second():
    index = SustainStateIndex([EventPoint(F(1, 3), "1h[4:4]")],
        [EventTempo(F(0), 92.55)], -0.21609940572663427)
    assert index.markers[0]["second"] == 0.0
    _, _, metadata = index.at(F(1, 3), include_current=True,
                              visibility_version="causal_v2")
    assert metadata["boundary_second"] < 0
    assert metadata["state_count"] == 1
    assert index.at(F(1, 3), include_current=True,
                    visibility_version="legacy_v1")[2]["state_count"] == 0


def test_positive_integer_bin_keeps_causal_tolerance_and_exact_legacy_filter():
    index = SustainStateIndex([EventPoint(F(2), "1h[4:4]")],
                              [EventTempo(F(0), 120)], 0)
    boundary = math.nextafter(1.0, -math.inf)
    with patch("maichart.generation.sustain_state.seconds_at_beat", return_value=boundary):
        assert index.at(F(2), include_current=True,
                        visibility_version="causal_v2")[2]["state_count"] == 1
        assert index.at(F(2), include_current=True,
                        visibility_version="legacy_v1")[2]["state_count"] == 0
        assert index.at(F(2), include_current=False,
                        visibility_version="causal_v2")[2]["state_count"] == 0
    # A head beyond the tolerance remains future even if its coarse bin exists.
    with patch("maichart.generation.sustain_state.seconds_at_beat", return_value=1.0-2e-9):
        assert index.at(F(2), include_current=True,
                        visibility_version="causal_v2")[2]["state_count"] == 0


def test_bucket_queries_match_full_scan_at_dense_random_boundaries():
    rng = random.Random(173)
    points = [EventPoint(F(i, 3), f"{i % 8 + 1}h[4:{rng.randint(1, 16)}]")
              for i in range(48)]
    index = SustainStateIndex(points, [EventTempo(F(0), 137)], -2.0)
    queries = [F(i, 12) for i in range(220)]
    queries += [F(rng.randrange(0, 16000), 1000) for _ in range(40)]
    for policy in ("legacy_v1", "causal_v2"):
        for include_current in (False, True):
            for beat in queries:
                kwargs = dict(include_current=include_current, visibility_version=policy,
                              visible_before_beat=F(10))
                actual = index.at(beat, **kwargs)
                with patch.object(index, "_candidate_markers", return_value=index.markers):
                    expected = index.at(beat, **kwargs)
                torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
                assert torch.equal(actual[1], expected[1])
                assert actual[2] == expected[2]


def test_append_invalidates_cached_intervals_before_multiple_commits():
    from maichart.stream.grounded_inference import IncrementalState

    index = IncrementalState([EventTempo(F(0), 120)], 0)
    index.append(EventPoint(F(0), "1h[4:1]"))
    assert index.at(F(0), include_current=True)[2]["state_count"] == 1
    assert index._active_marker_index is not None
    assert index._release_index is not None
    for beat in (F(2), F(4), F(6)):
        index.append(EventPoint(beat, "2h[4:1]"))
        assert index._active_marker_index is None
        assert index._release_index is None
    actual = index.at(F(6), include_current=True)
    with patch.object(index, "_candidate_markers", return_value=index.markers):
        expected = index.at(F(6), include_current=True)
    assert actual[2]["state_count"] == 1
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    assert torch.equal(actual[1], expected[1])
    assert actual[2] == expected[2]


def test_replacing_same_length_markers_rebuilds_both_indexes():
    index = SustainStateIndex([EventPoint(F(0), "1h[4:4]")],
                              [EventTempo(F(0), 120)], 0)
    index.at(F(1))
    old_markers = index.markers
    assert index._active_marker_index[0] is old_markers
    assert index._release_index[0] is old_markers
    replacement = SustainStateIndex([EventPoint(F(0), "2h[4:4]")],
                                    [EventTempo(F(0), 120)], 0)
    index.markers = replacement.markers
    actual = index.at(F(1))
    expected = replacement.at(F(1))
    assert index._active_marker_index[0] is index.markers
    assert index._release_index[0] is index.markers
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    assert actual[2] == expected[2]
    # The unbounded-interval fallback must also release the previous cache.
    index.markers = [dict(index.markers[0], end_second=5000.0)]
    assert index._candidate_markers(0.5) is index.markers
    assert index._active_marker_index is None
