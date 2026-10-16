"""Independent pre-optimization reference for causal release snapshots."""
from fractions import Fraction
import copy
import math

import pytest
import torch

from maichart.event_timing import EventPoint, EventTempo, seconds_at_beat
from maichart.generation.muri.model import RELEASE
from maichart.generation.sustain_state import MAX_STATES, STATE_DIM, SustainStateIndex


def _reference_release(index, marker, boundary):
    lane = marker.get('lane') if marker['type'] in {'hold', 'touch_hold'} else marker.get('end_lane')
    for following in index.markers:
        if (following['key'] != marker['key'] and following['type'] in {'slide', 'wifi'}
                and following['second'] < boundary-1e-9 and following['lane'] == lane
                and abs(following['slide_trace_second']-marker['end_second']) < 1/180):
            return 0.
    return RELEASE


def _reference_snapshot(index, beat):
    boundary = float(seconds_at_beat(beat, index.tempo_changes, index.first))
    selected = []
    for marker in index.markers:
        kind = marker.get('type')
        if (kind not in {'hold', 'touch_hold', 'slide', 'wifi'}
                or marker.get('is_mine') or marker.get('track_mine')):
            continue
        onset = float(marker.get('second', 0.))
        if onset >= boundary-1e-9:
            continue
        release = _reference_release(index, marker, boundary)
        end = float(marker.get('end_second', -1.))
        if not math.isfinite(end):
            raise ValueError('Invalid reference end')
        deadline = end+release
        trace = float(marker.get('slide_trace_second', -1.))
        waiting = kind in {'slide', 'wifi'} and trace > boundary+1e-9
        if deadline <= boundary+1e-9:
            continue
        selected.append((deadline, int(marker.get('parse_order', -1)), marker, waiting))
    selected.sort(key=lambda item: (item[0], item[1], int(item[2].get('line', 0)),
                                    int(item[2].get('col', 0))))
    retained = selected[:MAX_STATES]
    states = torch.zeros((max(1, len(retained)), STATE_DIM), dtype=torch.float32)
    for row, (deadline, _order, marker, waiting) in enumerate(retained):
        index._encode_state(states[row], marker, boundary, deadline, waiting)
    metadata = [index._state_metadata(marker, boundary, deadline, waiting)
                for deadline, _, marker, waiting in retained]
    return states, not retained, len(selected), metadata


@pytest.mark.parametrize('first', [-1.75, 0., .42])
def test_release_index_and_expiry_match_original_full_scan(first):
    points = [EventPoint(Fraction(i), (
        f'{i%8+1}h[4:3]' if i%3 else
        f'{i%8+1}-{(i+4)%8+1}[4:2]')) for i in range(35)]
    points += [EventPoint(Fraction(1, 2), '2w6[4:4]'),
               EventPoint(Fraction(3, 2), 'Ch[4:10]')]
    points.sort(key=lambda point: point.beat)
    tempos = [EventTempo(Fraction(0), 120.), EventTempo(Fraction(18), 173.)]
    index = SustainStateIndex(points, tempos, first)
    for beat in [Fraction(i, 8) for i in range(0, 321, 3)]:
        expected, empty, count, metadata = _reference_snapshot(index, beat)
        actual, mask, result = index.at(beat)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert mask.tolist() == ([True] if empty else [False]*len(actual))
        assert result['selected_state_count'] == count
        assert result['states'] == metadata


def test_expired_markers_do_not_search_following_heads(monkeypatch):
    index = SustainStateIndex([EventPoint(Fraction(i), f'{i%8+1}h[4:1]')
                               for i in range(150)], [EventTempo(Fraction(0), 120.)], 0.)
    def unexpected(*args):
        raise AssertionError('Expired sustain unnecessarily searched all following heads')
    monkeypatch.setattr(index, '_release_for', unexpected)
    assert index.at(200)[2]['state_count'] == 0


def test_restored_preparation_cache_builds_release_index_lazily():
    original = SustainStateIndex([EventPoint(Fraction(0), '1h[4:2]'),
                                  EventPoint(Fraction(1), '1-5[4:1]')],
                                 [EventTempo(Fraction(0), 120.)], 0.)
    restored = SustainStateIndex.__new__(SustainStateIndex)
    restored.__dict__.update({key: copy.deepcopy(value) for key, value in original.__dict__.items()
                              if key != '_release_index'})
    for beat in [Fraction(1), Fraction(2), Fraction(17, 8), Fraction(3)]:
        a, am, meta = original.at(beat)
        b, bm, other = restored.at(beat)
        assert torch.equal(a, b) and torch.equal(am, bm) and meta == other
    assert restored._release_index is not None


def test_causal_head_filter_and_strict_trace_tolerance_are_unchanged():
    index = SustainStateIndex([], [EventTempo(Fraction(0), 120.)], 0.)
    hold = dict(type='hold', key='held', lane=1, end_second=1.)
    following = dict(type='slide', key='next', lane=1, slide_trace_second=1., second=.5)
    index.markers = [hold, following]
    for boundary in [.5-1e-8, .5, .5+1e-10, .5+2e-9, .75]:
        assert index._release_for(hold, boundary) == _reference_release(index, hold, boundary)
    for delta in [-1/180-1e-12, -1/180, -1/180+1e-12, 1/180-1e-12, 1/180, 1/180+1e-12]:
        index.markers = [hold, dict(following, slide_trace_second=1.+delta)]
        assert index._release_for(hold, .75) == _reference_release(index, hold, .75)
