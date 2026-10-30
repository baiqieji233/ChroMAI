"""Zero holds keep their vocabulary identity and ordinary onset safety."""
from fractions import Fraction

import pytest

from maichart.composer.guard import EventGuard
from maichart.event_timing import EventPoint, EventTempo


def guard():
    return EventGuard([EventTempo(Fraction(0), 120.)], 0., 10., touch_layout_policy='open')


@pytest.mark.parametrize('text,tap', [
    ('1h', '1'), ('1h[16:0]', '1'), ('1bh[4:0]', '1b'),
    ('Ch[16:0]', 'C'), ('E4h', 'E4'), ('Cfh[4:0]', 'Cf'),
])
def test_zero_hold_has_tap_lifetime_and_does_not_leave_active_sustain(text, tap):
    hold_guard, tap_guard = guard(), guard()
    held, tapped = EventPoint(Fraction(1), text), EventPoint(Fraction(1), tap)
    assert hold_guard.accepts(held)
    assert tap_guard.accepts(tapped)
    assert hold_guard._lifetime(held)[:3] == tap_guard._lifetime(tapped)[:3]
    hold_guard.commit(held)
    tap_guard.commit(tapped)
    assert hold_guard.prefix[0][0].note_text == text  # no vocabulary rewriting
    assert not hold_guard.prefix[0][3]
    for later in (Fraction(17, 16), Fraction(5, 4), Fraction(2)):
        assert hold_guard.accepts(EventPoint(later, tap)) == tap_guard.accepts(EventPoint(later, tap))


def test_zero_hold_is_not_an_escape_from_occupancy_or_duplicate_head_checks():
    current = guard()
    assert not current.accepts(EventPoint(Fraction(0), '1/1h[4:0]'))
    current.commit(EventPoint(Fraction(0), '1h[4:4]'))
    assert not current.accepts(EventPoint(Fraction(1), '1h[16:0]'))
    assert current.accepts(EventPoint(Fraction(1), '2h[16:0]'))


def test_positive_hold_keeps_sustained_lifetime_and_invalid_syntax_is_rejected():
    current = guard()
    assert current._lifetime(EventPoint(Fraction(0), '1h[4:1]'))[:3] == (0., .5, True)
    assert not current.accepts(EventPoint(Fraction(0), '1h[broken]'))
