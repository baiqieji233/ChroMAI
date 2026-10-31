"""A real native-judge interaction lost by the former 300ms prefix pruning."""
from fractions import Fraction as F
from maichart.composer.guard import EventGuard
from maichart.event_timing import EventPoint,EventTempo
from maichart.generation.post_process import detect_chart_conflicts


class ShortWindowGuard(EventGuard):
    def interaction_start(self,second,rows=None):
        return float(second)-.3


def test_pending_slide_keeps_intervening_judgment_inputs_until_it_finishes():
    tempo=[EventTempo(F(0),205.)]
    points=[EventPoint(F(63623,192),'1bx-5[16.2406##0.2256]'),
            EventPoint(F(346),'3/8-4[8:1]'),EventPoint(F(699,2),'5')]
    options=dict(min_same_head_sec=.085,touch_layout_policy='open',physical_policy='runtime_hard_v2')
    old=ShortWindowGuard(tempo,0.,137.14866666666666,**options)
    guard=EventGuard(tempo,0.,137.14866666666666,**options)
    for point in points[:-1]:
        assert guard.accepts(point) and old.accepts(point)
        guard.commit(point);old.commit(point)
    full=detect_chart_conflicts(points,tempo,0.,guard.config)
    assert any(guard.enforced(x) and x.conflict_type=='miacode_SlideTooFast' for x in full.conflicts_detected)
    assert old.accepts(points[-1])  # Demonstrate the old concrete false negative.
    assert not guard.accepts(points[-1])
    assert guard.last_reason=='new_miacode_runtime_hard_conflict'
    assert any(p.beat==346 for p,_,_,_ in guard.prefix)
    # Once pending actions have completed, the old interaction history is freed.
    guard.commit(EventPoint(F(450),'2'))
    assert len(guard.prefix)==1


def test_legacy_context_policy_keeps_its_original_window():
    guard=EventGuard([EventTempo(F(0),120.)],0.,30.,physical_policy='legacy')
    guard.commit(EventPoint(F(0),'1h[4:20]'))
    guard.commit(EventPoint(F(2),'5'))
    assert guard.interaction_start(2.)==1.7
