from fractions import Fraction
from types import SimpleNamespace

import pytest

from maichart.composer.candidates import (
    MUSIC_CANDIDATE_POLICY,
    MUSIC_CANDIDATE_POLICY_V6,
    candidate_beats,
)
from maichart.composer.guard import EventGuard
from maichart.event_timing import EventPoint, EventTempo


@pytest.mark.parametrize('previous', [Fraction(5, 16), Fraction(1, 3), Fraction(11, 48)])
def test_phase_recovery_does_not_invent_sub_64th_bridge_events(previous):
    tempo = [EventTempo(Fraction(0), 120.)]
    candidates = candidate_beats(previous, tempo=tempo, first=0., audio_end_sec=20.,
                                 policy=MUSIC_CANDIDATE_POLICY_V6)
    assert candidates == sorted(set(candidates))
    assert all(beat - previous >= Fraction(1, 16) for beat in candidates)
    assert any(beat.denominator <= 2 for beat in candidates)


def test_v6_preserves_direct_subdivisions_and_v5_checkpoint_proposals():
    tempo = [EventTempo(Fraction(0), 120.)]
    common = dict(tempo=tempo, first=0., audio_end_sec=20.)
    direct = candidate_beats(Fraction(0), policy=MUSIC_CANDIDATE_POLICY_V6, **common)
    assert {Fraction(1, 16), Fraction(1, 12), Fraction(1, 3)} <= set(direct)
    triplet = candidate_beats(Fraction(2, 3), policy=MUSIC_CANDIDATE_POLICY_V6, **common)
    assert Fraction(5, 6) in triplet
    legacy = candidate_beats(Fraction(5, 16), policy=MUSIC_CANDIDATE_POLICY, **common)
    assert Fraction(1, 3) in legacy


def test_guard_checks_first_event_without_committing_on_rejection(monkeypatch):
    import maichart.composer.guard as module

    seen = []

    def detect(points, tempo, first, config):
        seen.append(list(points))
        return SimpleNamespace(coverage={'parse_errors': ['invalid native event']},
                               conflicts_detected=[])

    monkeypatch.setattr(module, 'detect_chart_conflicts', detect)
    guard = EventGuard([EventTempo(Fraction(0), 120.)], 0., 8.)
    point = EventPoint(Fraction(0), '1')
    assert not guard.accepts(point)
    assert seen == [[point]]
    assert guard.checks == guard.rejections == 1
    assert guard.prefix == []


def test_guard_native_transaction_rejects_pure_tap_without_independent_press():
    guard = EventGuard([EventTempo(Fraction(0), 120.)], 0., 8., min_same_head_sec=0.)
    first = EventPoint(Fraction(0), '1')
    assert guard.accepts(first)
    guard.commit(first)
    before = list(guard.prefix)
    # At 120 BPM this is 10.4 ms, shorter than MiaCode's 3/180 s
    # electrical release. A second tap cannot get its own rising edge.
    assert not guard.accepts(EventPoint(Fraction(1, 48), '1'))
    assert guard.prefix == before
    assert guard.accepts(EventPoint(Fraction(1, 48), '2'))
    assert guard.accepts(EventPoint(Fraction(1, 4), '1'))


def test_guard_native_transaction_checks_internal_first_event_overlap():
    guard = EventGuard([EventTempo(Fraction(0), 120.)], 0., 8.)
    assert not guard.accepts(EventPoint(Fraction(0), '1/1'))
    assert guard.prefix == []


def test_guard_same_head_safety_is_explicit_bounded_and_reason_counted():
    tempo = [EventTempo(Fraction(0), 120.)]
    guard = EventGuard(tempo, 0., 8., min_same_head_sec=.05)
    first = EventPoint(Fraction(0), '1')
    guard.commit(first)
    too_soon = EventPoint(Fraction(1, 16), '1')  # 31.25 ms at 120 BPM
    assert not guard.accepts(too_soon)
    assert guard.last_reason == 'repeated_head_too_soon'
    assert guard.reason_counts['repeated_head_too_soon'] == 1
    # The same rhythm on a different physical button is not a jack repeat.
    assert guard.accepts(EventPoint(Fraction(1, 16), '2'))
    # 62.5 ms clears the conservative 50 ms product policy.
    assert guard.accepts(EventPoint(Fraction(1, 8), '1'))


def test_guard_zero_same_head_threshold_disables_only_product_policy():
    guard = EventGuard([EventTempo(Fraction(0), 120.)], 0., 8., min_same_head_sec=0.)
    first = EventPoint(Fraction(0), '1')
    guard.commit(first)
    # Product policy is off but the native independent-press check remains.
    assert not guard.accepts(EventPoint(Fraction(1, 48), '1'))
    assert guard.accepts(EventPoint(Fraction(1, 16), '1'))


@pytest.mark.parametrize('value', [-.001, .200001, float('nan'), float('inf')])
def test_guard_rejects_invalid_same_head_threshold(value):
    with pytest.raises(ValueError, match='min_same_head_sec'):
        EventGuard([EventTempo(Fraction(0), 120.)], 0., 8., min_same_head_sec=value)


def test_guard_same_head_policy_uses_distinct_actual_heads_not_slide_endpoints():
    guard = EventGuard([EventTempo(Fraction(0), 120.)], 0., 8., min_same_head_sec=.05)
    # The atom's onset is button 1; destination 5 does not become a new head.
    slide = EventPoint(Fraction(0), '1-5[4:1]')
    guard.commit(slide)
    assert guard.last_head_seconds == {'A1': 0.}
    assert 'A5' not in guard.last_head_seconds


def test_guard_tap_on_slide_interception_is_optional():
    default_guard = EventGuard([EventTempo(Fraction(0), 190.)], 0., 200.)
    slide = EventPoint(Fraction(447), '4/5<4[2:1]')
    assert default_guard.accepts(slide)
    default_guard.commit(slide)
    assert default_guard.accepts(EventPoint(Fraction(897, 2), '7'))

    strict_guard = EventGuard([EventTempo(Fraction(0), 190.)], 0., 200., intercept_tap_on_slide=True)
    assert strict_guard.accepts(slide)
    strict_guard.commit(slide)
    assert not strict_guard.accepts(EventPoint(Fraction(897, 2), '7'))
    assert strict_guard.last_reason == 'new_miacode_runtime_hard_conflict'


def test_guided_replacement_can_reject_reported_hard_without_changing_default_policy():
    guard=EventGuard([EventTempo(Fraction(0),190.)],0.,200.)
    slide=EventPoint(Fraction(447),'4/5<4[2:1]')
    assert guard.accepts(slide)
    guard.commit(slide)
    proposal=EventPoint(Fraction(897,2),'7')
    assert guard.accepts(proposal)
    before=list(guard.prefix)
    assert not guard.accepts(proposal,protect_reported_hard=True)
    assert guard.last_reason=='guided_proposal_new_reported_hard_conflict'
    assert guard.prefix==before
    assert guard.config.conflict_kind_modes['TapOnSlide']=='report'
    assert guard.accepts(proposal)  # recovery/default calls retain their policy
