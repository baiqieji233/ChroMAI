from fractions import Fraction
from types import SimpleNamespace

import torch

from maichart.composer import ACTION_SAMPLING_POLICY
from maichart.composer import generate, recovery, train
from test_composer_recovery import builder, point


def test_recovery_sampling_with_short_tail_keeps_padding_mask(monkeypatch):
    """Exercise train's proposal callback with two real and 94 padded beats."""
    fixture = builder()
    class Dataset(SimpleNamespace):
        def __len__(self):
            return len(self.samples)
    dataset = Dataset(**vars(fixture.dataset))
    dataset.action_grammar = object()
    dataset.history_events = 8
    monkeypatch.setattr(recovery, 'candidate_beats', lambda previous, **kw: [Fraction(1), Fraction(2)])
    item, actual = fixture._inputs([point(0)], 0)
    assert len(actual) == 2 and len(item['_candidate_beats']) == 96
    assert item['candidate_mask'].tolist() == [False, False] + [True] * 94
    item['_proposal_guard'] = SimpleNamespace(
        last_head_seconds={}, min_same_head_sec=0, accepts=lambda *args, **kwargs: True)
    class Model:
        training = True
        config = SimpleNamespace(architecture_profile='fixture', action_state_policy='fixture', action_slots=3)
        def eval(self): self.training = False
        def train(self, value=True): self.training = value
    model = Model()
    args = SimpleNamespace(seed=9, recovery_samples=1, recovery_steps=2,
        recovery_sampling_policy=ACTION_SAMPLING_POLICY)
    monkeypatch.setattr(generate, '_head_metadata', lambda *args: {})
    monkeypatch.setattr(generate, '_touch_head_sets', lambda *args: set())
    monkeypatch.setattr(generate, '_active_constraints', lambda *args, **kwargs: (set(), set(), None))
    monkeypatch.setattr(generate, '_candidate_has_capacity', lambda *args: True)
    monkeypatch.setattr(generate, '_candidate_playability', lambda candidates, playable, *args: playable)
    monkeypatch.setattr(generate, '_sample_actions', lambda *args, **kwargs: ['4'])
    def forward(model, tensors, **kwargs):
        mask = tensors['candidate_padding_mask'][0]
        assert mask.tolist() == [False, False] + [True] * 94
        return {'candidate_logits': torch.zeros(1, 96),
                'candidate_bins': torch.zeros(1, 96, dtype=torch.long),
                'candidate_context': torch.zeros(1, 96, 4)}
    monkeypatch.setattr(train, '_forward_batch', forward)
    def advance(self, indices, episodes, *, steps):
        result = self.propose(item, 0)
        assert result.beat in actual
        assert result.note_text == '4'
        return None, {'accepted': 1}, [], {}
    monkeypatch.setattr(recovery.RecoveryBuilder, 'batch_from_episodes', advance)
    train.recovery_batch(dataset, model, args, torch.device('cpu'), 0)
    assert model.training


def test_recovery_contract_versions_preserve_v3_and_default_to_v4():
    from maichart.composer import (ACTION_SAMPLING_POLICY, PREVIOUS_ACTION_SAMPLING_POLICY,
                                    LEGACY_ACTION_SAMPLING_POLICY)
    from maichart.composer.sampling_contract import resolve_recovery_sampling, recovery_sampling_settings
    assert resolve_recovery_sampling('auto') == ACTION_SAMPLING_POLICY
    assert resolve_recovery_sampling('auto', {'action_sampling_policy': PREVIOUS_ACTION_SAMPLING_POLICY},
                                     resume=True) == PREVIOUS_ACTION_SAMPLING_POLICY
    old = recovery_sampling_settings(PREVIOUS_ACTION_SAMPLING_POLICY, .5, .75)
    new = recovery_sampling_settings(ACTION_SAMPLING_POLICY, .5, .75)
    legacy = recovery_sampling_settings(LEGACY_ACTION_SAMPLING_POLICY)
    assert old['pointer_sampler'] == new['pointer_sampler'] == 'grouped'
    assert old['retry_policy'] == new['retry_policy'] == 'failed_prefix_v2'
    assert old['touch_history_guard'] and new['touch_history_guard']
    assert old['touch_legal_mask'] is False and new['touch_legal_mask'] is True
    assert legacy['pointer_sampler'] == 'global' and legacy['touch_legal_mask'] is False


def test_resume_rejects_explicit_recovery_policy_change():
    import pytest
    from maichart.composer import ACTION_SAMPLING_POLICY, PREVIOUS_ACTION_SAMPLING_POLICY
    from maichart.composer.sampling_contract import resolve_recovery_sampling
    with pytest.raises(ValueError, match='Resume inherits'):
        resolve_recovery_sampling(ACTION_SAMPLING_POLICY,
            {'action_sampling_policy': PREVIOUS_ACTION_SAMPLING_POLICY}, resume=True)
