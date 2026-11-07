from fractions import Fraction as F
from types import SimpleNamespace as NS
import pytest
from maichart.event_timing import EventPoint as P,EventTempo
from maichart.composer_v2.generate import _new_guard,parser
from maichart.composer_v2.tail_guard import cleanup,options
from maichart.generation.post_process import detect_chart_conflicts


TEMPO=[EventTempo(F(0),180)]
# Extracted from actual 2026-11-07 generation, 78.5 seconds, A6 early 22.2ms.
ACTUAL=[P(F(234),'2-6[8:1]'),P(F(471,2),'6/1')]


def test_real_tail_interception_and_exact_chord_child_cleanup():
    old=_new_guard(TEMPO,0,200);old.commit(ACTUAL[0])
    assert old.accepts(P(ACTUAL[1].beat,'6'))
    guard=_new_guard(TEMPO,0,200,guard_options={'enabled':True,'threshold_ms':200})
    guard.commit(ACTUAL[0]);assert not guard.accepts(ACTUAL[1])
    repaired,report=cleanup(ACTUAL,TEMPO,0,guard)
    assert repaired==[ACTUAL[0],P(F(471,2),'1')]
    assert report['removed'][0]['raw']=='6'
    remaining=detect_chart_conflicts(repaired,TEMPO,0,guard.config)
    assert not any(x.conflict_type=='miacode_TapOnSlide' and x.severity=='hard' for x in remaining.conflicts_detected)
    untouched,report=cleanup(ACTUAL,TEMPO,0,old)
    assert untouched is ACTUAL and not report['removed']


def test_threshold_is_not_a_hard_conflict_disable_switch():
    g=_new_guard(TEMPO,0,200,guard_options={'enabled':True,'threshold_ms':0})
    g.commit(ACTUAL[0]);assert not g.accepts(ACTUAL[1])
    for bad in (-1,251,float('nan'),float('inf')):
        with pytest.raises(ValueError):options(NS(tap_on_slide_threshold_ms=bad))
    args=parser().parse_args(['--checkpoint','a','--audio','b','--output','c','--intercept-tap-on-slide','--tap-on-slide-threshold-ms','80'])
    assert options(args)=={'enabled':True,'threshold_ms':80}


def test_static_warning_is_retained():
    tempo=[EventTempo(F(0),200)]
    points=[P(F(164),'1-5[8:1]'),P(F(166),'5')]
    g=_new_guard(tempo,.0540396012138,100,guard_options={'enabled':True,'threshold_ms':200})
    result=detect_chart_conflicts(points,tempo,.0540396012138,g.config)
    assert any(i.conflict_type=='miacode_TapOnSlide' and i.severity=='soft' for i in result.conflicts_detected)
    assert cleanup(points,tempo,.0540396012138,g)[0]==points
