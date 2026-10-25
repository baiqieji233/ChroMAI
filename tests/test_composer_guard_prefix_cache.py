from fractions import Fraction
from types import SimpleNamespace

import pytest

from maichart.composer.guard import EventGuard
from maichart.event_timing import EventPoint, EventTempo


def setup_guard(monkeypatch, issues=None, parse_errors=False):
    import maichart.composer.guard as module
    calls = []

    def detect(points, tempo, first, config):
        calls.append(tuple((p.beat, p.note_text) for p in points))
        return SimpleNamespace(conflicts_detected=issues(points) if issues else [],
                               coverage={'parse_errors': parse_errors})

    monkeypatch.setattr(module, 'detect_chart_conflicts', detect)
    guard = EventGuard([EventTempo(Fraction(0), 120.)], 0., 20., min_same_head_sec=0.)
    guard.commit(EventPoint(Fraction(0), '1'))
    return guard, calls


def test_retries_reuse_only_before_and_after_is_always_checked(monkeypatch):
    guard, calls = setup_guard(monkeypatch)
    assert guard.accepts(EventPoint(Fraction(1, 4), '2'))
    assert guard.accepts(EventPoint(Fraction(1, 3), '3'))
    assert len(calls) == 3
    assert [len(c) for c in calls] == [1, 2, 2]


@pytest.mark.parametrize('mutation', ['window', 'commit', 'config', 'tempo', 'first', 'point'])
def test_context_and_configuration_changes_invalidate(monkeypatch, mutation):
    guard, calls = setup_guard(monkeypatch)
    candidate = EventPoint(Fraction(1, 4), '2')
    assert guard.accepts(candidate)
    if mutation == 'window':
        candidate = EventPoint(Fraction(1), '2')  # old tap leaves 0.3-second window
    elif mutation == 'commit':
        guard.commit(EventPoint(Fraction(1, 8), '3'))
    elif mutation == 'config':
        guard.config.conflict_kind_modes['SlideHeadTap'] = 'enforce'
    elif mutation == 'tempo':
        guard.tempo = [EventTempo(Fraction(0), 121.)]
    elif mutation == 'first':
        guard.first = .01
    else:
        guard.prefix[0][0].note_text = '4'
    assert guard.accepts(candidate)
    assert len(calls) == (3 if mutation == 'window' else 4)
    if mutation == 'window':
        assert len(calls[-1]) == 1


def issue(action='report', raw='1'):
    return SimpleNamespace(conflict_type='miacode_Overlap', time=0., severity='hard',
                           action=action, evidence='miacode_runtime:test',
                           notes=[SimpleNamespace(beat=Fraction(0), raw=raw)])


def test_existing_reported_conflict_is_not_new_and_rejection_does_not_commit(monkeypatch):
    def findings(points):
        return [issue()] + ([issue(raw='new')] if points[-1].note_text == '4' else [])

    guard, calls = setup_guard(monkeypatch, findings)
    original = list(guard.prefix)
    assert guard.accepts(EventPoint(Fraction(1, 4), '2'), protect_reported_hard=True)
    assert not guard.accepts(EventPoint(Fraction(1, 4), '4'), protect_reported_hard=True)
    assert guard.last_reason == 'guided_proposal_new_reported_hard_conflict'
    assert guard.prefix == original
    assert guard.accepts(EventPoint(Fraction(1, 4), '4'))
    assert len(calls) == 4


def test_cached_before_parse_error_does_not_hide_after_parse_error(monkeypatch):
    guard, calls = setup_guard(monkeypatch, parse_errors=True)
    for _ in range(2):
        assert not guard.accepts(EventPoint(Fraction(1, 4), '2'))
    assert len(calls) == 3
    assert len(guard.prefix) == 1
