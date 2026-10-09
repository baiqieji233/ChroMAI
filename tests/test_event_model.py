from fractions import Fraction

import torch

from maichart.generation.event_model import (
    EventConditionedTransformer,
    EventSequence,
    EventTransformerConfig,
    EventVocabulary,
)


def test_event_vocabulary_is_compositional_and_exact():
    vocabulary = EventVocabulary.from_event_charts([], segment_beats=4)
    sequence = EventSequence([Fraction(1, 3), Fraction(3, 4)], ["1-4[8:1]*-6[4:1]", "34"])
    ids = vocabulary.encode(sequence)
    decoded = vocabulary.decode(ids)
    assert decoded == sequence


def test_event_transformer_shape_with_global_audio_context():
    config = EventTransformerConfig(input_dim=103, vocab_size=64, d_model=32, nhead=4,
                                    num_layers=1, dim_feedforward=64, max_audio_tokens=8,
                                    global_audio_tokens=4, max_tokens=32)
    model = EventConditionedTransformer(config)
    logits = model(torch.randn(2, 10, 103), torch.ones(2, 7, dtype=torch.long),
                   torch.zeros(2, 16), global_audio=torch.randn(2, 4, 103))
    assert logits.shape == (2, 7, 64)


def _tiny_config(**overrides):
    options = dict(input_dim=8, condition_dim=16, vocab_size=16, d_model=16, nhead=2,
                   num_layers=1, dim_feedforward=32, max_audio_tokens=8,
                   global_audio_tokens=4, max_tokens=16)
    options.update(overrides)
    return EventTransformerConfig(**options)


def _legacy_state_dict(model):
    """The state dict a checkpoint trained before the condition token would have."""

    return {key: value.clone() for key, value in model.state_dict().items()
            if key != "condition_type_embedding"}


def test_window_planner_returns_continuous_targets_and_config_logits():
    model = EventConditionedTransformer(_tiny_config())
    logits, plan = model(
        torch.randn(2, 5, 8), torch.ones(2, 3, dtype=torch.long),
        torch.zeros(2, 16), global_audio=torch.randn(2, 4, 8), return_plan=True,
    )
    assert logits.shape == (2, 3, 16)
    assert plan.shape == (2, 9)


    model = EventConditionedTransformer(_tiny_config())
    memory, _ = model.encode_audio(torch.randn(1, 5, 8), torch.zeros(1, 16),
                                   global_audio=torch.randn(1, 4, 8))
    # segment + global + the condition token
    assert memory.size(1) == 5 + 4 + 1
    assert model.config.to_dict()["condition_fusion"] == "memory_token"


def test_legacy_checkpoint_loads_with_its_own_fusion_instead_of_failing():
    """The failure the user hit: "Missing key(s) ... condition_type_embedding"."""

    reference = EventConditionedTransformer(_tiny_config())
    state = _legacy_state_dict(reference)

    restored = EventConditionedTransformer(_tiny_config())
    compatibility = restored.load_weights_compatible(state)

    assert compatibility["legacy"] is True
    assert compatibility["condition_fusion"] == "per_frame_bias"
    assert restored.config.condition_fusion == "per_frame_bias"
    # No condition token is appended in the legacy layout, which is what makes
    # the pre-token weights behave exactly as they were trained.
    memory, _ = restored.encode_audio(torch.randn(1, 5, 8), torch.zeros(1, 16),
                                      global_audio=torch.randn(1, 4, 8))
    assert memory.size(1) == 5 + 4
    logits = restored(torch.randn(1, 5, 8), torch.ones(1, 3, dtype=torch.long),
                      torch.zeros(1, 16), global_audio=torch.randn(1, 4, 8))
    assert logits.shape == (1, 3, 16)


def test_legacy_mode_survives_save_and_reload():
    """A finetuned legacy run keeps its fusion even though the parameter is saved.

    The unused ``condition_type_embedding`` is still serialized, so the recorded
    config -- not the key's presence -- has to decide the fusion on reload.
    """

    source = EventConditionedTransformer(_tiny_config())
    source.load_weights_compatible(_legacy_state_dict(source))
    reloaded_state = source.state_dict()
    assert "condition_type_embedding" in reloaded_state

    restored = EventConditionedTransformer(
        EventTransformerConfig(**source.config.to_dict()))
    compatibility = restored.load_weights_compatible(reloaded_state)
    assert compatibility["condition_fusion"] == "per_frame_bias"
    assert restored.config.condition_fusion == "per_frame_bias"


def test_new_checkpoint_keeps_the_condition_token_path():
    source = EventConditionedTransformer(_tiny_config())
    restored = EventConditionedTransformer(_tiny_config())
    compatibility = restored.load_weights_compatible(source.state_dict())
    assert compatibility["legacy"] is False
    assert compatibility["condition_fusion"] == "memory_token"


def test_checkpoint_without_planner_loads_with_planner_disabled():
    source = EventConditionedTransformer(_tiny_config())
    state = {
        key: value.clone()
        for key, value in source.state_dict().items()
        if not (key.startswith("plan_head.") or key.startswith("plan_condition."))
    }
    restored = EventConditionedTransformer(_tiny_config())
    compatibility = restored.load_weights_compatible(state)
    assert compatibility["planner"] is False
    memory, mask, plan = restored.append_plan_memory(
        torch.zeros(1, 5, 8), torch.zeros(1, 5, dtype=torch.bool)
    )
    assert memory.shape[1] == 5
    assert mask.shape[1] == 5
    assert plan.shape == (1, 9)


def test_unrelated_structure_mismatch_is_still_a_hard_error():
    source = EventConditionedTransformer(_tiny_config())
    state = {key: value.clone() for key, value in source.state_dict().items()
             if key != "global_type_embedding"}
    restored = EventConditionedTransformer(_tiny_config())
    try:
        restored.load_weights_compatible(state)
    except ValueError as error:
        assert "不兼容" in str(error)
    else:  # pragma: no cover - the mismatch must never load silently
        raise AssertionError("a real structure mismatch must not load silently")


def test_unknown_condition_fusion_is_rejected():
    try:
        EventConditionedTransformer(_tiny_config(condition_fusion="typo"))
    except ValueError as error:
        assert "condition_fusion" in str(error)
    else:  # pragma: no cover
        raise AssertionError("an unknown fusion mode must be rejected")
