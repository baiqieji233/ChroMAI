"""Exercise phase echo together with the formal song-paired training path."""
import pytest
import torch

from test_composer_quality_integration import _fixture


def _echo_fixture(policy):
    model, data = _fixture(configuration_quality_policy=policy,
        action_head_policy='conserved_grouped_v2',
        cross_event_relation_policy='relative_heads_v1', echo_prior_policy='phase_echo_v1')
    data['history_beat_ages'] = torch.tensor([[2., 0.], [2., 0.]])
    if policy == 'song_paired_v3':
        quality = data['quality_inputs']
        quality.update(
            quality_plan_clock=torch.arange(4.).view(1, 4, 1).expand(2, -1, 2),
            quality_history_audio=torch.randn(2, 2, 104),
            quality_history_clock=torch.tensor([[[0., 0.], [2., 1.]]] * 2),
            quality_history_plan_indices=torch.tensor([[[0, 1], [1, 2]]] * 2),
            quality_history_plan_mix=torch.full((2, 2), .5),
            quality_phrase_plan_indices=torch.tensor([[[0, 0], [1, 1], [2, 2], [3, 3]]] * 2),
            quality_phrase_plan_mix=torch.zeros(2, 4))
    with torch.no_grad():
        model.echo_prior.table_E.fill_(.2)
        model.echo_prior.rtable.fill_(.3)
    return model, data


@pytest.mark.parametrize('policy', ['cooperative_v2', 'song_paired_v3'])
def test_echo_training_and_deferred_decoding_agree(policy):
    model, data = _echo_fixture(policy)
    full = model(**data)
    packed = model(**data, packed_actions=True)
    rows, slots = packed['action_slot_indices'].unbind(-1)
    torch.testing.assert_close(packed['action_logits'], full['action_logits'][rows, slots])
    deferred = model(**dict(data, decode_actions=False))
    decoded = model.decode_candidate(deferred['candidate_context'], data['candidate_indices'],
        action_targets=data['action_targets'], audio_context=deferred['audio_action_context'],
        previous_heads=deferred['previous_heads'], echo_heads=deferred['echo_lane_bias'])
    torch.testing.assert_close(decoded['action_logits'], full['action_logits'])
    torch.nn.functional.cross_entropy(full['action_logits'][:, 0], torch.tensor([1, 2])).backward()
    assert model.echo_prior.table_E.grad.abs().sum() > 0


@pytest.mark.parametrize('policy', ['cooperative_v2', 'song_paired_v3'])
def test_echo_ignores_targets_and_history_dropout_and_preserves_audio_reference(policy):
    model, data = _echo_fixture(policy)
    baseline = model(**data)
    changed = dict(data, action_targets=torch.tensor([[4, 3, 0], [2, 1, 0]]))
    changed['quality_inputs'] = dict(data['quality_inputs'],
        quality_phrase_target=torch.full_like(data['quality_inputs']['quality_phrase_target'], 999.),
        quality_gap_target=torch.tensor([6, 6]))
    after = model(**changed)
    for key in ('candidate_logits', 'echo_lane_bias'):
        torch.testing.assert_close(baseline[key], after[key], rtol=0, atol=0)
    altered_history = model(**dict(data, history=data['history'].roll(1, dims=0)))
    for key in ('audio_candidate_logits', 'audio_action_logits'):
        torch.testing.assert_close(baseline[key], altered_history[key], rtol=0, atol=0)
    hidden = model(**data, symbolic_keep=torch.zeros(2, 1))
    assert not hidden['echo_lane_bias'].any()
    torch.testing.assert_close(hidden['candidate_logits'], hidden['audio_candidate_logits'], atol=2e-6, rtol=1e-5)
    model.train()
    model.config.history_dropout = .9
    first = model(**data, symbolic_keep=torch.ones(2, 1))['echo_lane_bias']
    second = model(**data, symbolic_keep=torch.ones(2, 1))['echo_lane_bias']
    torch.testing.assert_close(first, second, rtol=0, atol=0)
