import pytest
import torch

from maichart.generation.event_model import EventConditionedTransformer, EventTransformerConfig


def test_relative_preference_changes_decoder_input_per_window_independently_of_curve():
    model = EventConditionedTransformer(EventTransformerConfig(
        input_dim=4, vocab_size=8, d_model=16, nhead=2, num_layers=1,
        dim_feedforward=32, max_tokens=32, planner_profile="music_structure_v3"))
    model.eval()
    model.planner_enabled = True
    memory = torch.zeros(1, 4, 16)
    mask = torch.zeros(1, 4, dtype=torch.bool)
    captured = []
    handle = model.plan_condition.register_forward_pre_hook(
        lambda _module, args: captured.append(args[0].detach().clone()))
    try:
        # Different musical-window predictions, not one fixed song-wide quota.
        for raw in (-3.0, 1.0):
            with torch.no_grad():
                model.plan_head[-1].weight.zero_()
                model.plan_head[-1].bias.zero_()
                model.plan_head[-1].bias[8] = raw
            values = []
            for preference in (0.0, .5, 1.0):
                _, _, prediction = model.append_plan_memory(memory, mask, plan_controls={
                    "__fine_rhythm_bias": preference, "density": .9, "__strength": 0.0})
                actual = captured[-1]
                values.append(float(actual[0, 8]))
                assert float(actual[0, 8]) == pytest.approx(
                    torch.sigmoid(torch.tensor(raw + 4 * (preference - .5))).item())
                assert float(prediction[0, 8]) == pytest.approx(raw)
                assert float(actual[0, 1]) == pytest.approx(.5)  # curve strength 0
                assert actual[0, 9:].sum().item() == pytest.approx(1.0)
            assert values[0] < values[1] < values[2]
    finally:
        handle.remove()


def test_old_planner_cannot_accidentally_read_fine_bias():
    model = EventConditionedTransformer(EventTransformerConfig(
        input_dim=4, vocab_size=8, d_model=16, nhead=2, num_layers=1,
        dim_feedforward=32, max_tokens=32, planner_profile="music_structure_v2"))
    model.eval()
    memory = torch.zeros(1, 4, 16)
    plain, _, _ = model.append_plan_memory(memory, None)
    biased, _, _ = model.append_plan_memory(memory, None, plan_controls={"__fine_rhythm_bias": 1.0})
    assert torch.equal(plain, biased)
