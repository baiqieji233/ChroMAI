import copy
import io
from types import SimpleNamespace

import pytest
import torch

from maichart.composer.recovery import (
    RECOVERY_EPISODE_SCHEMA, recovery_episode_state, restore_recovery_episodes,
)
from test_composer_recovery import builder, point


def test_episode_continues_generated_history_and_resets_at_horizon():
    recovery = builder()
    recovery.maximum_generated_events = 6
    seen = []

    def propose(item, index):
        prefix = item['_prefix_points']
        seen.append([p.note_text for p in prefix])
        return point(len(prefix), '4')

    recovery.propose = propose
    episodes = []
    for expected in (2, 4, 6):
        before = len(seen)
        batch, outcomes, episodes, diagnostics = recovery.batch_from_episodes([0], episodes, steps=2)
        assert len(seen) - before <= 2
        assert outcomes == {'accepted': 1}
        assert len(episodes[0]['events']) == expected
        assert diagnostics['generated_history_events'] == [expected]
        assert batch['history'][0, :, 0].tolist() == [1] + [4] * expected
    assert seen[2] == ['1', '4', '4']
    assert seen[4] == ['1', '4', '4', '4', '4']
    _, _, episodes, _ = recovery.batch_from_episodes([0], episodes, steps=2)
    assert len(episodes[0]['events']) == 2
    assert seen[6] == ['1']


def test_episode_ends_at_reference_tail_and_next_call_restarts():
    recovery = builder(source=[point(i) for i in range(4)])
    recovery.maximum_generated_events = 8
    seen = []
    def propose(item, index):
        prefix = item['_prefix_points']
        seen.append(len(prefix))
        return point(len(prefix), '4')
    recovery.propose = propose
    _, _, episodes, _ = recovery.batch_from_episodes([0], [], steps=2)
    assert episodes[0] is not None
    _, outcomes, episodes, _ = recovery.batch_from_episodes([0], episodes, steps=2)
    assert outcomes == {'no_reference_continuation': 1}
    assert episodes == [None]
    recovery.batch_from_episodes([0], episodes, steps=2)
    assert seen[-2:] == [1, 2]


def test_rejected_reference_retains_only_committed_proposals_and_counts_types():
    recovery = builder()
    recovery.maximum_generated_events = 8
    seen = []
    def propose(item, index):
        prefix = item['_prefix_points']
        seen.append([p.note_text for p in prefix])
        return point(len(prefix), '4')
    recovery.propose = propose
    original_build = recovery.build
    kinds = iter(('button', 'touch', 'mixed'))
    def rejecting_build(prefix, index, generated_count):
        item, reason = original_build(prefix, index, generated_count=generated_count)
        recovery.last_reference_kind = next(kinds)
        return None, 'illegal_reference_continuation'
    recovery.build = rejecting_build
    _, outcomes, episodes, stats = recovery.batch_from_episodes([0, 0, 0], [], steps=2)
    assert len(seen) <= 2 * 3
    assert outcomes == {'illegal_reference_continuation': 3}
    assert all([text for _, text in e['events']] == ['4', '4'] for e in episodes)
    assert set(stats['reference_types']) == {'button', 'touch', 'mixed'}
    for kind in stats['reference_types'].values():
        assert kind['accepted'] == 0 and kind['rejected'] == 1
        assert kind['rejected_weight'] == pytest.approx(.7)
    recovery.build = original_build
    recovery.batch_from_episodes([0], episodes[:1], steps=2)
    assert seen[-2] == ['1', '4', '4']


def test_episode_state_roundtrip_and_copy_isolation():
    source = SimpleNamespace(_recovery_episodes=[{'sample_index': 0, 'events': [['1/2', '4']]}])
    state = recovery_episode_state(source)
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    restored_state = torch.load(buffer, weights_only=True)
    target = SimpleNamespace()
    restore_recovery_episodes(target, restored_state)
    assert target._recovery_episodes == source._recovery_episodes
    target._recovery_episodes[0]['events'][0][1] = '1'
    assert source._recovery_episodes[0]['events'][0][1] == '4'
    assert state['episodes'][0]['events'][0][1] == '4'


@pytest.mark.parametrize('state', [
    {'schema': 'unknown', 'episodes': []},
    {'schema': RECOVERY_EPISODE_SCHEMA, 'episodes': [None] * 9},
    {'schema': RECOVERY_EPISODE_SCHEMA, 'episodes': [{'sample_index': -1, 'events': []}]},
    {'schema': RECOVERY_EPISODE_SCHEMA, 'episodes': [{'sample_index': 0, 'events': [['1', '1']] * 133}]},
    {'schema': RECOVERY_EPISODE_SCHEMA, 'episodes': [{'sample_index': 0, 'events': [['1']]}]},
])
def test_restore_rejects_invalid_state(state):
    with pytest.raises(ValueError):
        restore_recovery_episodes(SimpleNamespace(), copy.deepcopy(state))


def test_legacy_batch_still_starts_fresh_and_uses_short_rollout():
    recovery = builder()
    recovery.maximum_generated_events = 8
    seen = []
    def propose(item, index):
        seen.append(len(item['_prefix_points']))
        return point(len(item['_prefix_points']), '4')
    for _ in range(2):
        batch, outcomes = recovery.batch_from_indices([0], steps=2, propose=propose)
        assert outcomes == {'accepted': 1}
        assert batch['history'][0, :, 0].tolist() == [1, 4, 4]
    assert seen == [1, 2, 1, 2]


def test_episode_fills_128_event_history_with_generated_events(monkeypatch):
    # A regular grid and RecordingGuard isolate persistent state/history from
    # real candidate-policy reachability and physical playability.
    from fractions import Fraction
    from maichart.composer import recovery as module

    source = [point(Fraction(i, 32), '1') for i in range(180)]
    recovery = builder(source=source)
    recovery.maximum_generated_events = 128
    recovery.dataset.history_events = 128
    def grid(previous, **kwargs):
        start = Fraction(previous or 0)
        return [start + Fraction(i, 32) for i in range(1, kwargs['candidate_count'] + 1)]
    monkeypatch.setattr(module, 'candidate_beats', grid)
    calls = []
    def propose(item, index):
        calls.append(len(item['_prefix_points']))
        return point(item['_candidate_beats'][0], '4')
    recovery.propose = propose
    episodes = []
    for expected in range(2, 129, 2):
        before = len(calls)
        batch, outcomes, episodes, diagnostics = recovery.batch_from_episodes([0], episodes, steps=2)
        assert len(calls) - before == 2
        assert outcomes == {'accepted': 1}
        assert len(episodes[0]['events']) == expected
        assert diagnostics['generated_history_events'] == [expected]
    assert len(calls) == 128
    assert batch['history'][0, :, 0].tolist() == [4] * 128
    assert all(text == '4' for _, text in episodes[0]['events'])


@pytest.mark.parametrize(('target', 'kind'), [
    ('2', 'button'), ('A1', 'touch'), ('2/A1', 'mixed'),
])
def test_real_build_classifies_reference_target(target, kind):
    source = [point(i, target if i == 3 else '1') for i in range(12)]
    recovery = builder(source=source)
    item, reason = recovery.build([point(0), point(1, '4'), point(2, '4')], 0, generated_count=2)
    assert reason == 'accepted'
    assert item is not None
    assert recovery.last_reference_kind == kind


def test_stopped_rollout_discards_episode_and_restarts():
    recovery = builder()
    recovery.maximum_generated_events = 8
    seen = []
    def propose(item, index):
        prefix = item['_prefix_points']
        seen.append(len(prefix))
        return point(len(prefix), '4')
    recovery.propose = propose
    _, _, episodes, _ = recovery.batch_from_episodes([0], [], steps=2)
    assert len(episodes[0]['events']) == 2
    recovery.propose = lambda item, index: None
    _, outcomes, episodes, _ = recovery.batch_from_episodes([0], episodes, steps=2)
    assert outcomes == {'rollout_stopped': 1}
    assert episodes == [None]
    recovery.propose = propose
    recovery.batch_from_episodes([0], episodes, steps=2)
    assert seen == [1, 2, 1, 2]
