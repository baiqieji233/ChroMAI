"""Behavioral contracts for the learned configuration context (no checkpoints)."""
import torch

from maichart.composer.quality_context import (
    QUALITY_GAP_CLASSES,
    QualityContext,
    quality_context_loss,
)


def _fixture():
    torch.manual_seed(43)
    batch, candidates, anchors, slots, width, phrases = 2, 5, 8, 3, 24, 4
    audio = torch.randn(batch, candidates, width)
    condition = torch.randn(batch, 16)
    inputs = {
        'quality_lifecycle': torch.rand(batch, candidates, 88),
        'quality_candidate_offsets': torch.arange(1, candidates + 1).float()[None].expand(batch, -1) * .25,
        'quality_anchor_audio': torch.randn(batch, anchors, 120),
        'quality_anchor_ids': torch.randint(1, 20, (batch, anchors, slots)),
        'quality_anchor_ages': torch.arange(anchors).float().expand(batch, -1),
        'quality_anchor_mask': torch.ones(batch, anchors, dtype=torch.bool),
        'quality_phrase_audio': torch.randn(batch, phrases, 104),
        'quality_phrase_indices': torch.tensor([[0, 0, 1, 2, 3], [0, 1, 1, 2, 3]]),
        'quality_phrase_target': torch.rand(batch, phrases, 8),
        'quality_phrase_weight': torch.ones(batch, phrases),
        'quality_gap_target': torch.tensor([2, 4]),
        'quality_gap_weight': torch.ones(batch),
        'candidate_target': torch.tensor([1, 3]),
        'sample_weight': torch.ones(batch),
    }
    embedding = torch.randn(batch, anchors, slots, width)
    model = QualityContext(width).eval()
    return model, audio, condition, inputs, embedding


def _activate(model):
    with torch.no_grad():
        model.output.weight.normal_(std=.05)


def test_zero_residual_preserves_baseline_but_output_and_predictors_can_learn():
    model, audio, condition, inputs, embeddings = _fixture()
    result = model(audio, condition, inputs, anchor_embeddings=embeddings)
    assert torch.count_nonzero(result['quality_residual']) == 0
    assert torch.count_nonzero(result['quality_audio_residual']) == 0
    loss, _ = quality_context_loss(result, inputs)
    (loss + (result['quality_residual'] * torch.randn_like(audio)).sum()).backward()
    assert model.output.weight.grad.abs().sum() > 0
    assert model.phrase[-1].weight.grad.abs().sum() > 0
    assert model.gap_head[-1].weight.grad.abs().sum() > 0


def test_future_chart_labels_never_condition_predictions_or_decoding():
    model, audio, condition, inputs, embeddings = _fixture()
    _activate(model)
    before = model(audio, condition, inputs, anchor_embeddings=embeddings)
    changed = dict(inputs)
    changed['quality_phrase_target'] = torch.full_like(inputs['quality_phrase_target'], 999.)
    changed['quality_gap_target'] = torch.zeros_like(inputs['quality_gap_target'])
    changed['quality_phrase_weight'] = torch.zeros_like(inputs['quality_phrase_weight'])
    changed['quality_gap_weight'] = torch.zeros_like(inputs['quality_gap_weight'])
    after = model(audio, condition, changed, anchor_embeddings=embeddings)
    for key in before:
        torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)


def test_symbolic_dropout_hides_pair_and_lifecycle_together():
    model, audio, condition, inputs, embeddings = _fixture()
    _activate(model)
    result = model(audio, condition, inputs, anchor_embeddings=embeddings,
                   symbolic_keep=torch.zeros(audio.size(0), 1))
    torch.testing.assert_close(result['quality_residual'], result['quality_audio_residual'], rtol=0, atol=0)
    torch.testing.assert_close(result['quality_gap_logits'], result['quality_gap_audio_logits'], rtol=0, atol=0)


def test_paired_anchors_preserve_set_order_and_masked_values_are_irrelevant():
    model, audio, condition, inputs, embeddings = _fixture()
    _activate(model)
    order = torch.tensor([7, 3, 4, 0, 6, 2, 1, 5])
    shuffled = dict(inputs)
    for name in ('quality_anchor_audio', 'quality_anchor_ids', 'quality_anchor_ages', 'quality_anchor_mask'):
        shuffled[name] = inputs[name][:, order]
    before = model(audio, condition, inputs, anchor_embeddings=embeddings)
    after = model(audio, condition, shuffled, anchor_embeddings=embeddings[:, order])
    torch.testing.assert_close(before['quality_residual'], after['quality_residual'], atol=2e-6, rtol=1e-5)
    absent = dict(inputs, quality_anchor_mask=torch.zeros_like(inputs['quality_anchor_mask']))
    masked_before = model(audio, condition, absent, anchor_embeddings=embeddings)
    absent['quality_anchor_audio'] = inputs['quality_anchor_audio'] * 1000
    masked_after = model(audio, condition, absent, anchor_embeddings=embeddings * 1000)
    torch.testing.assert_close(masked_before['quality_residual'], masked_after['quality_residual'], rtol=0, atol=0)
    # The audio auxiliary is independent of symbol values even when available.
    changed = model(audio, condition, inputs, anchor_embeddings=embeddings.flip(-1))
    torch.testing.assert_close(before['quality_audio_residual'], changed['quality_audio_residual'], rtol=0, atol=0)
    assert not torch.allclose(before['quality_residual'], changed['quality_residual'])


def test_phrase_prediction_is_stable_under_candidate_window_repacking():
    model, audio, condition, inputs, embeddings = _fixture()
    _activate(model)
    before = model(audio, condition, inputs, anchor_embeddings=embeddings)
    order = torch.tensor([4, 2, 0])
    packed = dict(inputs)
    packed['quality_phrase_indices'] = inputs['quality_phrase_indices'][:, order]
    packed['quality_lifecycle'] = inputs['quality_lifecycle'][:, order]
    packed['quality_candidate_offsets'] = inputs['quality_candidate_offsets'][:, order]
    after = model(audio[:, order], condition, packed, anchor_embeddings=embeddings)
    torch.testing.assert_close(before['quality_phrase_prediction'], after['quality_phrase_prediction'], rtol=0, atol=0)
    torch.testing.assert_close(before['quality_residual'][:, order], after['quality_residual'], atol=2e-6, rtol=1e-5)


def test_unknown_tail_and_recovery_label_masks_produce_zero_auxiliary_gradient():
    model, audio, condition, inputs, embeddings = _fixture()
    result = model(audio, condition, inputs, anchor_embeddings=embeddings)
    masked = dict(inputs)
    masked['candidate_target'] = torch.full_like(inputs['candidate_target'], audio.size(1))
    masked['quality_gap_target'] = torch.full_like(inputs['quality_gap_target'], -1)
    masked['quality_gap_weight'] = torch.zeros_like(inputs['quality_gap_weight'])
    masked['quality_phrase_weight'] = torch.zeros_like(inputs['quality_phrase_weight'])
    loss, metrics = quality_context_loss(result, masked)
    assert loss == 0
    assert metrics['quality_gap_ce_weight'] == 0
    assert metrics['quality_phrase_weight'] == 0
    loss.backward()
    assert model.phrase[-1].weight.grad.count_nonzero() == 0
    assert model.gap_head[-1].weight.grad.count_nonzero() == 0


def test_only_selected_trusted_event_supervises_next_gap():
    model, audio, condition, inputs, embeddings = _fixture()
    result = model(audio, condition, inputs, anchor_embeddings=embeddings)
    result['quality_gap_logits'].retain_grad()
    inputs['quality_phrase_weight'].zero_()
    inputs['quality_gap_weight'][1] = 0
    loss, _ = quality_context_loss(result, inputs)
    loss.backward()
    grad = result['quality_gap_logits'].grad
    assert grad[0, 1].count_nonzero() == QUALITY_GAP_CLASSES
    assert grad[0, [0, 2, 3, 4]].count_nonzero() == 0
    assert grad[1].count_nonzero() == 0


def test_padded_candidates_have_no_action_residual():
    model, audio, condition, inputs, embeddings = _fixture()
    _activate(model)
    valid = torch.tensor([[True, True, False, False, False], [True] * 5])
    result = model(audio, condition, inputs, anchor_embeddings=embeddings, valid_mask=valid)
    assert result['quality_residual'][~valid].count_nonzero() == 0
    assert result['quality_audio_residual'][~valid].count_nonzero() == 0


def test_context_strength_zero_disables_paired_history_and_lifecycle_effect():
    model, audio, condition, inputs, embeddings = _fixture()
    _activate(model)
    before = model(audio, condition, inputs, anchor_embeddings=embeddings, context_strength=0.)
    changed = dict(inputs, quality_lifecycle=inputs['quality_lifecycle'] * 10,
                   quality_anchor_audio=inputs['quality_anchor_audio'].flip(-1))
    after = model(audio, condition, changed, anchor_embeddings=embeddings.flip(-1), context_strength=0.)
    torch.testing.assert_close(before['quality_residual'], after['quality_residual'], rtol=0, atol=0)
    torch.testing.assert_close(before['quality_gap_logits'], after['quality_gap_logits'], rtol=0, atol=0)
    torch.testing.assert_close(before['quality_residual'], before['quality_audio_residual'], rtol=0, atol=0)


def test_plan_strength_zero_disables_plan_conditioning_but_keeps_predictions():
    model, audio, condition, inputs, embeddings = _fixture()
    _activate(model)
    before = model(audio, condition, inputs, anchor_embeddings=embeddings, plan_strength=0.)
    changed = dict(inputs, quality_phrase_audio=inputs['quality_phrase_audio'] * 10)
    after = model(audio, condition, changed, anchor_embeddings=embeddings, plan_strength=0.)
    assert not torch.allclose(before['quality_phrase_prediction'], after['quality_phrase_prediction'])
    for key in ('quality_residual', 'quality_audio_residual', 'quality_gap_logits'):
        torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)


def test_action_gradients_to_predictors_are_scaled_once_but_auxiliary_is_unchanged(monkeypatch):
    import maichart.composer.quality_context as module
    model, audio, condition, inputs, embeddings = _fixture()
    _activate(model)
    probe = torch.randn_like(audio)
    parameters = (model.phrase[-1].weight, model.gap_head[-1].weight)
    action_grads, auxiliary_grads, residuals = [], [], []
    for scale in (1., .1):
        monkeypatch.setattr(module, 'QUALITY_ACTION_PREDICTION_GRAD_SCALE', scale)
        result = model(audio, condition, inputs, anchor_embeddings=embeddings)
        residuals.append(result['quality_residual'].detach())
        action_loss = ((result['quality_residual'] + result['quality_audio_residual']) * probe).sum()
        action_grads.append(torch.autograd.grad(action_loss, parameters, retain_graph=True))
        auxiliary, _ = quality_context_loss(result, inputs)
        auxiliary_grads.append(torch.autograd.grad(auxiliary, parameters))
    torch.testing.assert_close(residuals[0], residuals[1], rtol=0, atol=0)
    for reference, scaled in zip(action_grads[0], action_grads[1]):
        assert reference.abs().sum() > 0
        torch.testing.assert_close(scaled, reference * .1, atol=1e-6, rtol=1e-5)
    for reference, scaled in zip(auxiliary_grads[0], auxiliary_grads[1]):
        torch.testing.assert_close(scaled, reference, rtol=0, atol=0)


def test_exact_candidate_offset_changes_pairing_even_with_one_anchor():
    model, audio, _condition, inputs, embeddings = _fixture()
    inputs['quality_anchor_mask'].zero_()
    inputs['quality_anchor_mask'][:, 0] = True
    query = model.query(audio[:, :1]).expand(-1, audio.size(1), -1)
    inputs['quality_candidate_offsets'] = torch.tensor([[.125, .25, 1., 8., 32.]] * 2)
    pair = model._paired_context(query, inputs, embeddings)
    assert not torch.allclose(pair[:, 0], pair[:, 1])
    assert not torch.allclose(pair[:, 2], pair[:, 4])
    # Every candidate still reads the same event, but at its own elapsed time.
    inputs['quality_candidate_offsets'].zero_()
    same_time = model._paired_context(query, inputs, embeddings)
    torch.testing.assert_close(same_time[:, 0], same_time[:, 4], rtol=0, atol=0)
