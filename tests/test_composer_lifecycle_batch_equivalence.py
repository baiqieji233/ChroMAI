from fractions import Fraction

import numpy as np

from maichart.composer.quality_inputs import lifecycle_features
from maichart.event_timing import EventPoint, EventTempo
from maichart.generation.muri.model import PADS
from maichart.generation.sustain_state import SustainStateIndex, _SLIDE_PAD_CACHE_LIMIT


class ScalarIndex:
    def __init__(self, index):
        self.index = index
        self.markers = index.markers

    def _candidate_markers(self, clock):
        return self.index._candidate_markers(clock)

    def _terminal_lanes(self, marker):
        return self.index._terminal_lanes(marker)

    def _slide_pads(self, marker, clock):
        return self.index._slide_pads(marker, clock)


def test_batch_geometry_and_lifecycle_match_scalar_with_visibility_cutoff():
    tempo = [EventTempo(Fraction(0), 120.)]
    points = [EventPoint(Fraction(0), '1-3-5[4:4]/2w6[4:4]'),
              EventPoint(Fraction(2), '4-8[4:4]')]
    index = SustainStateIndex(points, tempo, 0.)
    rng = np.random.default_rng(321)
    for marker in index.markers:
        if marker['type'] not in {'slide', 'wifi'}:
            continue
        trace = float(marker['slide_trace_second'])
        end = float(marker['end_second'])
        clocks = list(rng.uniform(trace, end, 300)) + [trace, end]
        for shoot in marker.get('slide_segment_shoot_seconds', []):
            clocks.extend([float(shoot) - 1e-9, float(shoot), float(shoot) + 1e-9])
        # Only clocks inside the actual tracing interval are queried by lifecycle.
        clocks = np.asarray([t for t in clocks if trace <= t <= end])
        actual = index._slide_pad_matrix(marker, clocks)
        expected = np.asarray([[pad in index._slide_pads(marker, float(t)) for pad in PADS]
                               for t in clocks])
        np.testing.assert_array_equal(actual, expected)

    clocks = np.sort(np.r_[rng.uniform(0., 4., 500), [0., .5, 1., 2., 4.]])
    for cutoff in (None, Fraction(1), Fraction(2), Fraction(3)):
        actual = lifecycle_features(index, clocks, visible_before_beat=cutoff)
        expected = lifecycle_features(ScalarIndex(index), clocks, visible_before_beat=cutoff)
        np.testing.assert_array_equal(actual, expected)


def test_geometry_cache_reuses_exact_clocks_and_invalidates_marker_list(monkeypatch):
    index = SustainStateIndex([EventPoint(Fraction(0), '1-5[0##1]')],
                              [EventTempo(Fraction(0), 120.)], 0.)
    marker = index.markers[0]
    uncached = index._slide_pad_matrix_uncached
    calls = []

    def counted(item, clocks):
        calls.append(len(clocks))
        return uncached(item, clocks)

    monkeypatch.setattr(index, '_slide_pad_matrix_uncached', counted)
    first = index._slide_pad_matrix(marker, [.2, .4])
    second = index._slide_pad_matrix(marker, [.4, .2, .6])
    np.testing.assert_array_equal(first, second[[1, 0]])
    assert calls == [2, 1]
    index.markers = list(index.markers)
    np.testing.assert_array_equal(index._slide_pad_matrix(marker, [.2, .4]), first)
    assert calls == [2, 1, 2]
    marker['slide_segment_durations'] = list(marker['slide_segment_durations'])
    marker['slide_segment_durations'][0] *= 0.9
    index._slide_pad_matrix(marker, [.2])
    assert calls[-1] == 1
    index.markers.append(dict(marker))
    index._slide_pad_matrix(marker, [.2])
    assert calls[-2:] == [1, 1]
    index._slide_pad_matrix(marker, np.linspace(.01, .5, _SLIDE_PAD_CACHE_LIMIT + 100))
    assert len(index._slide_pad_cache[1][2]) == _SLIDE_PAD_CACHE_LIMIT
