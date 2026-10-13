import copy

import pytest
import torch

from maichart.form.model import FormConfig, FormTransformer


def make_model():
    torch.manual_seed(11)
    config = FormConfig(
        vocab_size=19,
        d_model=32,
        nhead=4,
        layers=1,
        dim_feedforward=64,
        dropout=0,
        audio_dim=6,
        descriptor_dim=5,
        condition_dim=3,
        max_tokens=16,
        max_audio_tokens=8,
        max_memory_tokens=16,
    )
    return FormTransformer(config).eval()


def make_batch(batch_size=2):
    return {
        "audio": torch.randn(batch_size, 4, 6),
        "audio_padding_mask": torch.zeros(batch_size, 4, dtype=torch.bool),
        "form": torch.randn(batch_size, 2, 8),
        "form_padding_mask": torch.zeros(batch_size, 2, dtype=torch.bool),
        "history": torch.tensor([[1, 2, 0], [3, 0, 0]])[:batch_size],
        "motif": torch.tensor([[4, 5], [6, 0]])[:batch_size],
        "draft": torch.tensor([[0], [0]])[:batch_size],
        "right": torch.tensor([[0], [0]])[:batch_size],
        "condition": torch.randn(batch_size, 3),
        "task": torch.zeros(batch_size, dtype=torch.long),
        "relation": torch.randn(batch_size, 4),
        "memory_strength": torch.ones(batch_size),
        "action_state": torch.randn(batch_size, 3, 50),
        "action_state_padding_mask": torch.zeros(batch_size, 3, dtype=torch.bool),
        "decoder_input": torch.tensor([[1, 7, 8], [1, 9, 0]])[:batch_size],
    }


def test_incremental_decoder_matches_full_prefix_without_recomputing_it():
    model=make_model();batch=make_batch(1)
    with torch.inference_mode():
        memory,mask=model.encode(batch)
        expected=model.decode(batch['decoder_input'],memory,mask)
        cache=None;steps=[]
        for index in range(batch['decoder_input'].size(1)):
            logits,cache=model.decode_step(batch['decoder_input'][:,index:index+1],index,memory,mask,cache)
            steps.append(logits)
        torch.testing.assert_close(torch.cat(steps,1),expected,atol=1e-5,rtol=1e-5)


def test_model_outputs_logits_and_relation_auxiliary():
    model = make_model()
    output = model(make_batch())

    assert output["logits"].shape == (2, 3, 19)
    assert output["relation_logits"].shape == (2, 2)
    assert torch.isfinite(output["logits"]).all()
    assert torch.isfinite(output["relation_logits"]).all()


def test_motif_can_change_logits_and_zero_strength_removes_its_effect():
    model = make_model()
    original = make_batch(batch_size=1)
    alternate = copy.deepcopy(original)
    alternate["motif"] = torch.tensor([[10, 11]])
    with torch.no_grad():
        regular_a = model(original)["logits"]
        regular_b = model(alternate)["logits"]
    assert not torch.allclose(regular_a, regular_b)

    original["memory_strength"] = torch.zeros(1)
    alternate["memory_strength"] = torch.zeros(1)
    with torch.no_grad():
        muted_a = model(original)["logits"]
        muted_b = model(alternate)["logits"]
    assert torch.equal(muted_a, muted_b)


def test_fully_empty_or_masked_memories_remain_finite():
    model = make_model()
    batch = make_batch(batch_size=1)
    batch["audio"] = torch.empty(1, 0, 6)
    batch["form"] = torch.empty(1, 0, 8)
    batch["history"] = torch.zeros(1, 1, dtype=torch.long)
    batch["motif"] = torch.zeros(1, 1, dtype=torch.long)
    batch["action_state_padding_mask"] = torch.ones(1, 3, dtype=torch.bool)
    batch["audio_padding_mask"] = torch.empty(1, 0, dtype=torch.bool)
    batch["form_padding_mask"] = torch.empty(1, 0, dtype=torch.bool)
    with torch.no_grad():
        memory, mask = model.encode(batch)
        output = model(batch)

    assert memory.shape[0] == 1 and mask.shape == memory.shape[:2]
    assert torch.isfinite(memory).all()
    assert torch.isfinite(output["logits"]).all()


def test_draft_task_rejects_draft_or_right_context_leakage():
    model = make_model()
    batch = make_batch(batch_size=1)
    batch["draft"] = torch.tensor([[3]])
    with pytest.raises(ValueError, match="draft tokens cannot be supplied"):
        model.encode(batch)

    batch["draft"] = torch.zeros(1, 1, dtype=torch.long)
    batch["right"] = torch.tensor([[4]])
    with pytest.raises(ValueError, match="right tokens cannot be supplied"):
        model.encode(batch)


def test_gradients_reach_audio_form_motif_and_action_branches():
    model = make_model()
    batch = make_batch(batch_size=1)
    for name in ("audio", "form", "action_state"):
        batch[name] = batch[name].detach().requires_grad_(True)
    output = model(batch)
    (output["logits"].square().mean() + output["relation_logits"].square().mean()).backward()

    assert batch["audio"].grad is not None and batch["audio"].grad.abs().sum() > 0
    assert batch["form"].grad is not None and batch["form"].grad.abs().sum() > 0
    assert batch["action_state"].grad is not None and batch["action_state"].grad.abs().sum() > 0
    assert model.token_embedding.weight.grad is not None
    assert model.token_embedding.weight.grad[4:7].abs().sum() > 0


def test_forward_matches_reused_encode_decode_in_eval_mode():
    model = make_model()
    batch = make_batch()
    with torch.no_grad():
        memory, memory_mask = model.encode(batch)
        output = model(batch)
        logits = model.decode(batch["decoder_input"], memory, memory_mask)

    assert torch.allclose(output["logits"], logits)
    assert torch.allclose(output["relation_logits"], model.relation_head(memory[:, -1]))


def test_audio_uses_its_own_cap_and_chart_memory_cap_is_per_branch():
    config = FormConfig(
        vocab_size=19, d_model=32, nhead=4, layers=1, dim_feedforward=64,
        dropout=0, audio_dim=6, descriptor_dim=5, condition_dim=3,
        max_tokens=16, max_audio_tokens=8, max_memory_tokens=2,
    )
    model = FormTransformer(config).eval()
    batch = make_batch(batch_size=1)
    batch["form"] = torch.empty(1, 0, 8)
    batch["history"] = torch.tensor([[1, 2]])
    batch["motif"] = torch.tensor([[4, 5]])
    batch["draft"] = torch.zeros(1, 0, dtype=torch.long)
    batch["right"] = torch.zeros(1, 0, dtype=torch.long)
    batch["action_state"] = torch.empty(1, 0, 50)
    batch["action_state_padding_mask"] = torch.empty(1, 0, dtype=torch.bool)
    batch["audio"] = torch.randn(1, 4, 6)
    batch["audio_padding_mask"] = torch.zeros(1, 4, dtype=torch.bool)
    batch["form_padding_mask"] = torch.empty(1, 0, dtype=torch.bool)

    memory, _ = model.encode(batch)
    assert memory.size(1) == 4 + 2 + 2 + 3


def test_combined_branch_positions_do_not_impose_a_false_total_sequence_cap():
    config = FormConfig(
        vocab_size=19, d_model=32, nhead=4, layers=1, dim_feedforward=64,
        dropout=0, audio_dim=6, descriptor_dim=5, condition_dim=3,
        max_tokens=16, max_audio_tokens=8, max_memory_tokens=40,
    )
    model = FormTransformer(config).eval()
    batch = make_batch(batch_size=1)
    batch["audio"] = torch.randn(1, 4, 6)
    batch["form"] = torch.empty(1, 0, 8)
    batch["form_padding_mask"] = torch.empty(1, 0, dtype=torch.bool)
    batch["history"] = torch.ones(1, 40, dtype=torch.long)
    batch["motif"] = torch.full((1, 40), 2, dtype=torch.long)
    batch["draft"] = torch.full((1, 40), 3, dtype=torch.long)
    batch["right"] = torch.full((1, 40), 4, dtype=torch.long)
    batch["task"] = torch.ones(1, dtype=torch.long)
    batch["action_state"] = torch.empty(1, 0, 50)
    batch["action_state_padding_mask"] = torch.empty(1, 0, dtype=torch.bool)

    memory, _ = model.encode(batch)
    assert memory.size(1) == 4 * 40 + 4 + 3
    assert memory.size(1) > model.position_embedding.num_embeddings


def test_batch_padding_in_other_examples_does_not_shift_audio_or_memory_positions():
    model = make_model()
    single = make_batch(batch_size=1)
    other = make_batch(batch_size=1)
    other["audio"] = torch.randn(1, 8, 6)
    other["form"] = torch.randn(1, 4, 8)
    other["history"] = torch.tensor([[2, 3, 4, 5, 6, 7, 8]])
    other["motif"] = torch.tensor([[3, 4, 5, 6, 7]])
    other["action_state"] = torch.randn(1, 5, 50)

    batched = {}
    for key in ("audio", "form", "history", "motif", "draft", "right", "action_state"):
        a, b = single[key], other[key]
        width = max(a.size(1), b.size(1))
        padded = a.new_zeros((2, width, *a.shape[2:]))
        padded[0, :a.size(1)] = a[0]
        padded[1, :b.size(1)] = b[0]
        batched[key] = padded
    for key, source_key in (("condition", "condition"), ("task", "task"), ("relation", "relation"),
                            ("memory_strength", "memory_strength"), ("decoder_input", "decoder_input")):
        batched[key] = torch.cat((single[source_key], other[source_key]), dim=0)
    for key in ("audio", "form"):
        lengths = [single[key].size(1), other[key].size(1)]
        batched[key + "_padding_mask"] = torch.arange(batched[key].size(1))[None, :] >= torch.tensor(lengths)[:, None]
    batched["action_state_padding_mask"] = torch.arange(batched["action_state"].size(1))[None, :] >= torch.tensor(
        [single["action_state"].size(1), other["action_state"].size(1)]
    )[:, None]

    with torch.no_grad():
        alone = model(single)
        together = model(batched)
    assert torch.allclose(alone["logits"][0], together["logits"][0], atol=1e-5)
    assert torch.allclose(alone["relation_logits"][0], together["relation_logits"][0], atol=1e-5)


def test_curve_visibility_hides_unknown_channels_and_padded_rows():
    model = make_model()
    batch = make_batch(batch_size=1)
    batch["load_curve"] = torch.tensor([[[0.2, 0.3, 0.4, 0.5],
                                         [0.6, 0.7, 0.8, 0.9]]])
    batch["load_curve_condition_mask"] = torch.tensor([[[True, False, True, False],
                                                        [True, True, True, True]]])
    batch["load_curve_padding_mask"] = torch.tensor([[False, True]])
    changed = copy.deepcopy(batch)
    changed["load_curve"][0, 0, 1] = 0.99
    changed["load_curve"][0, 0, 3] = 0.01
    changed["load_curve"][0, 1] = torch.tensor([0.01, 0.02, 0.03, 0.04])
    with torch.no_grad():
        original = model(batch)["logits"]
        altered = model(changed)["logits"]
    torch.testing.assert_close(original, altered)


def test_two_dimensional_curve_mask_retains_legacy_hidden_row_semantics():
    model = make_model()
    batch = make_batch(batch_size=1)
    batch["load_curve"] = torch.tensor([[[0.2, 0.3, 0.4, 0.5],
                                         [0.6, 0.7, 0.8, 0.9]]])
    batch["load_curve_condition_mask"] = torch.tensor([[True, False]])
    changed = copy.deepcopy(batch)
    changed["load_curve"][0, 0] = torch.tensor([0.01, 0.02, 0.03, 0.04])
    with torch.no_grad():
        original = model(batch)["logits"]
        altered = model(changed)["logits"]
    torch.testing.assert_close(original, altered)


def test_decoder_is_causal_and_checks_token_limits():
    model = make_model()
    batch = make_batch(batch_size=1)
    memory, memory_mask = model.encode(batch)
    first = torch.tensor([[1, 7, 8]])
    changed_future = torch.tensor([[1, 7, 12]])
    with torch.no_grad():
        logits_a = model.decode(first, memory, memory_mask)
        logits_b = model.decode(changed_future, memory, memory_mask)
    assert torch.allclose(logits_a[:, :2], logits_b[:, :2], atol=1e-6)
    with pytest.raises(ValueError, match="max_tokens"):
        model.decode(torch.ones(1, 17, dtype=torch.long), memory, memory_mask)


@pytest.mark.parametrize("kwargs", [{"d_model": 30, "nhead": 8}, {"vocab_size": 0}, {"dropout": 1.0}])
def test_form_config_rejects_invalid_architecture(kwargs):
    values = dict(vocab_size=19)
    values.update(kwargs)
    with pytest.raises(ValueError):
        FormConfig(**values)
