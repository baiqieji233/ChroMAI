from fractions import Fraction
from pathlib import Path

from maichart.composer.stream_data import NOTE, WAIT, StreamStep
from maichart.composer.stream_search import _cut


def test_cut_preserves_real_steps_without_boundary_wait():
    steps=[StreamStep(NOTE,Fraction(0),Fraction(1),0.,.5,'1'),
           StreamStep(NOTE,Fraction(1),Fraction(5),.5,2.5,'2')]
    assert _cut(steps,Fraction(4),None)==steps[:1]
    assert _cut(steps,Fraction(4),None,prefix_length=1)==steps
    assert all(step.kind!=WAIT for step in _cut(steps,Fraction(4),None))


def test_write_budget_does_not_modify_sampled_timing_distribution():
    source=(Path(__file__).parents[1]/'src/maichart/composer/stream_generate.py').read_text(encoding='utf-8')
    assert 'cursor+d>=stop_beat' not in source
    assert 'min(cursor+2,stop_beat)' not in source
