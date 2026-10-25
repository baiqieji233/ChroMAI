from fractions import Fraction as F
import math

import torch

from maichart.event_timing import EventPoint, EventTempo
from maichart.generation.sustain_state import SustainStateIndex
from maichart.composer.prefix_state import candidate_prefix_states, prefix_state_summary


def index(points, first=-2.0):
    return SustainStateIndex([EventPoint(F(beat), text) for beat, text in points],
        [EventTempo(F(0), 137.0), EventTempo(F(2), 193.0)], first)


def test_future_suffix_and_incremental_prefix_equivalent():
    prefix = [(0, "1h[4:4]/2-6[4:3]")]
    full = index(prefix + [(1, "1-5[0##1]"), (3, "7h[4:8]")])
    incremental = index(prefix)
    a = candidate_prefix_states(full, [F(1, 2), F(2), F(4)], visible_before_beat=F(1))
    b = candidate_prefix_states(incremental, [F(1, 2), F(2), F(4)], visible_before_beat=F(1))
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)
    assert a.shape == (3, 50)
    assert not torch.equal(a[0], a[-1])
    assert a[0,49].item() == 0.0


def test_following_head_cannot_shorten_uncommitted_release():
    prefix = [(0, "1h[4:1]")]
    full = index(prefix + [(1, "1-5[0##1]")], first=0)
    partial = index(prefix, first=0)
    # Hold end = beat 1; following slide trace coincides with that end.
    query = F(101,100)
    torch.testing.assert_close(prefix_state_summary(full,query,visible_before_beat=F(1)),
                               prefix_state_summary(partial,query,visible_before_beat=F(1)))


def test_empty_visibility_and_exact_source_binding():
    value = index([(0,"1h[4:8]"),(F(1,3),"2h[4:8]")])
    assert {m["source_beat"] for m in value.markers} == {F(0),F(1,3)}
    assert not prefix_state_summary(value,F(2),visible_before_beat=F(0)).any()


def test_include_current_roundoff_has_explicit_legacy_path():
    value=index([(F(1,3),"1h[4:4]")])
    marker=value.markers[0]
    from maichart.event_timing import seconds_at_beat
    boundary=seconds_at_beat(F(1,3),value.tempo_changes,value.first)
    marker["second"]=math.nextafter(boundary,math.inf)
    assert value.at(F(1,3),include_current=True,visibility_version="causal_v2")[2]["state_count"]==1
    assert value.at(F(1,3),include_current=True,visibility_version="legacy_v1")[2]["state_count"]==0


def test_committed_count_full_chart_matches_incremental():
    from maichart.composer.prefix_state import committed_prefix_states
    full=index([(0,"1hx[4:8]"),(1,"2h[4:8]")])
    partial=index([(0,"1hx[4:8]")])
    torch.testing.assert_close(
        committed_prefix_states(full,[F(3)],committed_events=1),
        committed_prefix_states(partial,[F(3)],committed_events=1))
    assert committed_prefix_states(full,[F(3)],committed_events=1)[0,49] == 1
