"""Full Composer integration for the opt-in cooperative configuration policy."""
import torch
from torch.nn import functional as F

from maichart.composer.model import build_model
from maichart.composer.quality_context import quality_context_loss
from test_composer_audio_grounded_v3 import ATOMS, config, inputs


def _quality_inputs():
    return {
        'quality_lifecycle': torch.rand(2, 6, 88),
        'quality_candidate_offsets': torch.tensor([[.25, .5, 1., 2., 6., 12.]] * 2),
        'quality_anchor_audio': torch.randn(2, 8, 120),
        'quality_anchor_ids': torch.randint(1, 5, (2, 8, 3)),
        'quality_anchor_ages': torch.arange(8).float().expand(2, -1),
        'quality_anchor_mask': torch.ones(2, 8, dtype=torch.bool),
        'quality_phrase_audio': torch.randn(2, 4, 104),
        'quality_phrase_indices': torch.tensor([[0, 0, 1, 2, 2, 3]] * 2),
        'quality_phrase_target': torch.rand(2, 4, 8),
        'quality_phrase_weight': torch.ones(2, 4),
        'quality_gap_target': torch.tensor([2, 4]),
        'quality_gap_weight': torch.ones(2),
        'candidate_target': torch.tensor([1, 2]),
        'sample_weight': torch.ones(2),
    }


def _fixture(**overrides):
    torch.manual_seed(122)
    options = dict(configuration_quality_policy='cooperative_v1',
                   history_dropout=0., ledger_dropout=0., symbolic_dropout=0.)
    options.update(overrides)
    model = build_model(config(**options), ATOMS).eval()
    data = inputs()
    data['quality_inputs'] = _quality_inputs()
    data['candidate_indices'] = torch.tensor([1, 2])
    data['action_targets'] = torch.tensor([[1, 2, 0], [3, 0, 0]])
    return model, data


def test_quality_policy_off_keeps_old_forward_and_state_shape():
    model, data = _fixture(configuration_quality_policy='off')
    data.pop('quality_inputs')
    result = model(**data)
    assert 'quality_gap_logits' not in result
    assert not any(key.startswith('quality_context.') for key in model.state_dict())
    assert result['action_logits'].shape == (2, 3, 5)


def test_quality_policy_on_is_initially_an_exact_action_residual():
    model, data = _fixture()
    old = build_model(config(configuration_quality_policy='off',
        history_dropout=0., ledger_dropout=0., symbolic_dropout=0.), ATOMS).eval()
    old.load_state_dict({key: value for key, value in model.state_dict().items()
                        if key in old.state_dict()})
    result = model(**data)
    old_result = old(**{key: value for key, value in data.items() if key != 'quality_inputs'})
    assert result['quality_phrase_prediction'].shape == (2, 4, 8)
    assert result['quality_gap_logits'].shape == (2, 6, 7)
    for key in ('candidate_logits', 'action_logits', 'audio_action_logits'):
        torch.testing.assert_close(result[key], old_result[key], atol=1e-6, rtol=1e-5)


def test_quality_packed_and_dense_action_training_agree():
    model, data = _fixture(action_head_policy='conserved_grouped_v2')
    with torch.no_grad():
        model.quality_context.output.weight.normal_(std=.05)
    dense = model(**data)
    packed = model(**data, packed_actions=True)
    rows, slots = packed['action_slot_indices'].unbind(-1)
    for key in ('action_logits', 'audio_action_logits'):
        torch.testing.assert_close(packed[key], dense[key][rows, slots], atol=2e-5, rtol=1e-5)
    torch.testing.assert_close(packed['quality_gap_logits'], dense['quality_gap_logits'])


def test_quality_supervision_never_leaks_into_full_model_forward():
    model, data = _fixture()
    with torch.no_grad():
        model.quality_context.output.weight.normal_(std=.05)
    before = model(**data)
    changed = dict(data)
    changed_quality = dict(data['quality_inputs'])
    changed_quality['quality_phrase_target'] = torch.full((2, 4, 8), 999.)
    changed_quality['quality_gap_target'] = torch.full((2,), 6)
    changed['quality_inputs'] = changed_quality
    after = model(**changed)
    for key in ('candidate_logits', 'action_logits', 'quality_phrase_prediction', 'quality_gap_logits'):
        torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
    # Predicted quality is action-side information, not a new timing override.
    changed_quality['quality_phrase_audio'] = torch.randn(2, 4, 104) * 3
    changed_audio = model(**changed)
    torch.testing.assert_close(before['candidate_logits'], changed_audio['candidate_logits'], rtol=0, atol=0)
    assert not torch.allclose(before['action_logits'], changed_audio['action_logits'])


def test_quality_predictions_and_residual_receive_training_gradient():
    model, data = _fixture()
    result = model(**data)
    loss, metrics = quality_context_loss(result, data['quality_inputs'])
    loss = loss + F.cross_entropy(result['action_logits'].flatten(0, 1), data['action_targets'].flatten())
    loss.backward()
    for weight in (model.quality_context.output.weight, model.quality_context.phrase[-1].weight,
                   model.quality_context.gap_head[-1].weight,
                   model.quality_context.anchor_audio[1].weight):
        assert weight.grad is not None and weight.grad.abs().sum() > 0
    assert metrics['quality_gap_full_loss'] > 0
    assert metrics['quality_gap_audio_loss'] > 0


def test_selected_prefix_replacement_keeps_the_quality_residual():
    model, data = _fixture(action_state_policy='selected_prefix_v3')
    data['selected_action_state'] = torch.zeros(2, 50)
    with torch.no_grad():
        model.quality_context.output.weight.normal_(std=.05)
    before = model(**data)
    altered = dict(data, quality_inputs=dict(data['quality_inputs']))
    altered['quality_inputs']['quality_phrase_audio'] = torch.randn(2, 4, 104) * 2
    after = model(**altered)
    assert not torch.allclose(before['action_logits'], after['action_logits'])
    torch.testing.assert_close(before['candidate_logits'], after['candidate_logits'], rtol=0, atol=0)
