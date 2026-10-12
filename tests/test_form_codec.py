from fractions import Fraction as F

import pytest

from maichart.event_timing import EventPoint
from maichart.form.codec import RationalCodec


def test_positions_are_compositional_and_not_limited_to_16_beats_or_training_grid():
    codec = RationalCodec()
    points = [EventPoint(F(0), '1'), EventPoint(F(1,3),'2/6'),
              EventPoint(F(7,5),'3h[8:3]'), EventPoint(F(193,7),'1-5[4:2]')]
    recovered = codec.decode(codec.encode(points))
    assert [(p.beat,p.note_text) for p in recovered] == [(p.beat,p.note_text) for p in points]
    assert RationalCodec.from_dict(codec.to_dict()).to_dict() == codec.to_dict()


def test_memory_budget_never_splits_an_event_or_changes_its_relative_time():
    codec = RationalCodec()
    points = [EventPoint(F(i,3),'1') for i in range(8)]
    encoded = codec.bounded_memory(points,max_tokens=30)
    assert len(encoded) <= 30
    restored = codec.decode(encoded)
    assert restored[-1].beat == points[-1].beat
    assert restored[0].beat > 0
    with pytest.raises(ValueError,match='Incomplete'):
        codec.decode(codec.encode(points)[:-1])
