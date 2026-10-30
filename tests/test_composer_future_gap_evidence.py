import pytest
import torch

from maichart.composer.future_gap_evidence import future_gap_evidence
from maichart.composer.quality_context import QualityContext


def test_gap_bins_observed_support_padding_and_fp32_clock():
    offsets = torch.tensor([[32., 32.125, 32.25, 32.5, 33., 34., 36., 40., float('nan')]])
    valid = torch.tensor([[True] * 8 + [False]])
    placement = torch.full((1, 9, 5), .3)
    attack = torch.full((1, 9), .7)
    placement[:, -1] = float('nan')
    attack[:, -1] = float('nan')
    result = future_gap_evidence(offsets, valid, placement, attack).reshape(1, 9, 7, 3)
    torch.testing.assert_close(result[0, 0], torch.tensor([[.3, .7, 1.]] * 7))
    assert torch.count_nonzero(result[0, -1]) == 0
    assert torch.count_nonzero(result[0, -2]) == 0
    assert torch.isfinite(result).all()


def test_summary_is_detached_and_missing_evidence_is_explicit():
    offsets = torch.tensor([[0., .25]])
    summary = future_gap_evidence(offsets, torch.ones_like(offsets, dtype=torch.bool),
                                  torch.rand(1, 2, 5, requires_grad=True),
                                  torch.rand(1, 2, requires_grad=True))
    assert not summary.requires_grad
    model = QualityContext(24, future_evidence=True)
    assert torch.count_nonzero(model.future_evidence_projection.weight) == 0


def test_modifier_confidence_cannot_become_false_presence_evidence():
    offsets=torch.tensor([[0.,.25]])
    placement=torch.full((1,2,5),.99);placement[...,0]=.1
    result=future_gap_evidence(offsets,torch.ones_like(offsets,dtype=torch.bool),placement,
        torch.zeros_like(offsets)).reshape(1,2,7,3)
    assert result[0,0,1,0]==pytest.approx(.1)
    assert result[0,0,1,2]==1.


def test_candidate_competition_keeps_nearby_offsets_at_large_bf16_clock():
    from maichart.composer.grounded_model import _CandidateCompetition

    width = 8
    model = _CandidateCompetition(width).bfloat16().eval()
    captured = []
    handle = model.in_proj.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
    try:
        model(torch.randn(1, 2, width, dtype=torch.bfloat16),
              torch.ones(1, 2, dtype=torch.bool),
              candidate_time=torch.tensor([[[32., 16.], [32.125, 16.0625]]]))
    finally:
        handle.remove()
    # The first candidate's right beat gap survives; BF16 subtraction at 32
    # would round both candidate beats to the same number and yield zero.
    right_log_gap = captured[0][0, 0, width * 3 + 21].float()
    torch.testing.assert_close(right_log_gap, torch.log1p(torch.tensor(.125)), rtol=.01, atol=.001)


def test_zero_projection_preserves_gap_predictions_then_can_learn_without_placement_gradient():
    from test_composer_configuration_quality_context import _fixture

    baseline, audio, condition, inputs, anchors = _fixture()
    model = QualityContext(24, future_evidence=True).eval()
    model.load_state_dict(baseline.state_dict(), strict=False)
    valid = torch.ones(audio.shape[:2], dtype=torch.bool)
    placement = torch.rand(*audio.shape[:2], 5, requires_grad=True)
    attack = torch.rand(*audio.shape[:2], requires_grad=True)
    expected = baseline(audio, condition, inputs, anchor_embeddings=anchors, valid_mask=valid)
    with pytest.raises(ValueError, match='explicit valid_mask'):
        model(audio, condition, inputs, anchor_embeddings=anchors)
    actual = model(audio, condition, inputs, anchor_embeddings=anchors, valid_mask=valid,
                   placement_probs=placement, attack=attack)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    actual['quality_gap_logits'].square().sum().backward()
    assert model.future_evidence_projection.weight.grad.abs().sum() > 0
    assert placement.grad is None
    assert attack.grad is None
