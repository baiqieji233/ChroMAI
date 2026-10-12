import torch

from maichart.generation.event_model import EventConditionedTransformer, EventTransformerConfig


def _config(**overrides):
    values = dict(
        input_dim=8,
        condition_dim=4,
        vocab_size=12,
        d_model=16,
        nhead=2,
        num_layers=1,
        dim_feedforward=32,
        max_audio_tokens=8,
        global_audio_tokens=4,
        max_tokens=16,
    )
    values.update(overrides)
    return EventTransformerConfig(**values)


def test_none_profile_preserves_state_dict_and_rejects_action_inputs():
    torch.manual_seed(23)
    default_model = EventConditionedTransformer(_config())
    torch.manual_seed(23)
    explicit_none_model = EventConditionedTransformer(_config(action_state_profile="none"))

    assert default_model.state_dict().keys() == explicit_none_model.state_dict().keys()
    assert not any("action_state" in key for key in default_model.state_dict())
    for key, value in default_model.state_dict().items():
        torch.testing.assert_close(value, explicit_none_model.state_dict()[key])
    default_model.eval()
    explicit_none_model.eval()
    audio = torch.randn(1, 3, 8)
    decoder = torch.ones(1, 2, dtype=torch.long)
    condition = torch.zeros(1, 4)
    with torch.no_grad():
        torch.testing.assert_close(default_model(audio, decoder, condition),
                                   explicit_none_model(audio, decoder, condition))
    try:
        default_model.encode_audio(torch.randn(1, 3, 8), torch.zeros(1, 4),
                                   action_state=torch.zeros(1, 1, 50))
    except ValueError as error:
        assert "action_state_profile" in str(error)
    else:  # pragma: no cover
        raise AssertionError("disabled action-state inputs must not be silently ignored")


def test_enabled_profile_requires_state_and_encodes_fixed_width_memory_tokens():
    model = EventConditionedTransformer(_config(action_state_profile="sustain_state_v1"))
    audio = torch.randn(2, 5, 8)
    condition = torch.zeros(2, 4)
    states = torch.randn(2, 3, 50)
    state_padding = torch.tensor([[False, False, True], [False, True, True]])

    try:
        model.encode_audio(audio, condition)
    except ValueError as error:
        assert "action_state is required" in str(error)
    else:  # pragma: no cover
        raise AssertionError("enabled action-state profile must require state tensors")

    memory, mask = model.encode_audio(audio, condition, action_state=states,
                                      action_state_padding_mask=state_padding)
    assert model.action_state_projection.weight.shape == (16, 50)
    assert model.action_state_type_embedding.shape == (16,)
    assert memory.shape == (2, 5 + 1 + 3, 16)
    assert mask.shape == (2, 5 + 1 + 3)
    assert mask[:, -3:].tolist() == state_padding.tolist()
    assert memory._planner_layout == (5, 0, 4)


def test_masked_state_token_values_cannot_change_unmasked_memory():
    torch.manual_seed(9)
    model = EventConditionedTransformer(_config(action_state_profile="sustain_state_v1"))
    model.eval()
    audio = torch.randn(1, 4, 8)
    condition = torch.zeros(1, 4)
    states = torch.randn(1, 2, 50)
    padding = torch.tensor([[False, True]])
    changed = states.clone()
    changed[:, 1] = 10_000.0

    with torch.no_grad():
        first, _ = model.encode_audio(audio, condition, action_state=states,
                                      action_state_padding_mask=padding)
        second, _ = model.encode_audio(audio, condition, action_state=changed,
                                       action_state_padding_mask=padding)

    # Audio, condition, and the unmasked sustain token cannot attend to the
    # masked sustain slot as a key, regardless of its input value.
    torch.testing.assert_close(first[:, :-1], second[:, :-1], atol=1e-6, rtol=1e-6)


def test_action_state_weights_need_explicit_migration_and_must_be_complete():
    old_model = EventConditionedTransformer(_config())
    old_state = old_model.state_dict()
    enabled = EventConditionedTransformer(_config(action_state_profile="sustain_state_v1"))
    try:
        enabled.load_weights_compatible(old_state)
    except ValueError as error:
        assert "no trained action-state projection" in str(error)
    else:  # pragma: no cover
        raise AssertionError("adding an untrained state input must require explicit migration")

    migrated = enabled.load_weights_compatible(old_state, allow_missing_action_state=True)
    assert migrated["action_state"] is False
    new_state = enabled.state_dict()
    restored = EventConditionedTransformer(_config(action_state_profile="sustain_state_v1"))
    assert restored.load_weights_compatible(new_state)["action_state"] is True

    partial = {key: value for key, value in new_state.items()
               if key != "action_state_projection.bias"}
    try:
        restored.load_weights_compatible(partial, allow_missing_action_state=True)
    except ValueError as error:
        assert "incomplete action-state" in str(error)
    else:  # pragma: no cover
        raise AssertionError("partial action-state weights must never be treated as a fresh head")


def test_disabled_profile_rejects_checkpoint_that_contains_action_state_weights():
    enabled = EventConditionedTransformer(_config(action_state_profile="sustain_state_v1"))
    disabled = EventConditionedTransformer(_config())
    try:
        disabled.load_weights_compatible(enabled.state_dict())
    except ValueError as error:
        assert "action_state_profile is disabled" in str(error)
    else:  # pragma: no cover
        raise AssertionError("trained action-state weights must not be silently discarded")


def test_forward_passes_action_state_to_encoder():
    model = EventConditionedTransformer(_config(action_state_profile="sustain_state_v1"))
    logits = model(torch.randn(1, 4, 8), torch.ones(1, 3, dtype=torch.long),
                   torch.zeros(1, 4), action_state=torch.zeros(1, 2, 50),
                   action_state_padding_mask=torch.tensor([[False, True]]))
    assert logits.shape == (1, 3, 12)


def test_active_state_changes_planner_and_receives_training_gradients():
    torch.manual_seed(19)
    model = EventConditionedTransformer(_config(action_state_profile="sustain_state_v1",
        planner_profile="music_structure_v3", plan_dim=13))
    model.eval()
    audio, condition = torch.randn(1, 4, 8), torch.zeros(1, 4)
    short, long = torch.zeros(1, 1, 50), torch.zeros(1, 1, 50)
    long[0, 0, 0] = 1.0  # ongoing Hold
    long[0, 0, 4] = 1.0  # A1
    long[0, 0, 46] = .7  # remaining duration
    plans = []
    for states, padding in ((short, torch.ones(1, 1, dtype=torch.bool)),
                            (long, torch.zeros(1, 1, dtype=torch.bool))):
        memory, mask = model.encode_audio(audio, condition, action_state=states,
                                          action_state_padding_mask=padding)
        plans.append(model.append_plan_memory(memory, mask)[2])
    assert not torch.allclose(plans[0], plans[1])
    plans[1].square().sum().backward()
    grad = model.action_state_projection.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
