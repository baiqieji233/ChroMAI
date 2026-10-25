from fractions import Fraction as F

from maichart.composer.generate import density_observation
from maichart.composer.guard import EventGuard
from maichart.event_timing import EventPoint,EventTempo
from maichart.tempo_clock import TempoClock


def test_open_density_and_guard_agree_on_button_and_touch_identity():
    tempo=[EventTempo(F(0),120.)];clock=TempoClock(tempo,0.)
    points=[EventPoint(F(0),'1'),EventPoint(F(1,16),'A1')]
    guard=EventGuard(tempo,0.,10.,touch_layout_policy='open')
    for point in points:
        assert guard.accepts(point)
        guard.commit(point)
    report=density_observation(points,clock,touch_layout_policy='open')
    assert report['min_event_interval_ms']==31.25
    assert report['min_same_head_interval_ms'] is None
    assert report['same_head_intervals_below_50ms']==0
    assert density_observation(points,clock)['same_head_intervals_below_50ms']==1
    points.append(EventPoint(F(1,8),'A1'))
    assert not guard.accepts(points[-1])
    report=density_observation(points,clock,touch_layout_policy='open')
    assert report['same_head_intervals_below_50ms']==1
