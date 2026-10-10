from fractions import Fraction

import torch

from maichart.generation.event_model import (
    EventConditionedTransformer,
    EventSequence,
    EventTransformerConfig,
    EventVocabulary,
    pool_local_tracks,
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


def test_per_sample_pooling_keeps_short_sample_grid_stable_in_mixed_batch():
    short = torch.arange(3, dtype=torch.float32).reshape(1, 3, 1)
    long = torch.arange(7, dtype=torch.float32).reshape(1, 7, 1)
    short_batched = torch.cat([short, torch.zeros(1, 4, 1)], dim=1)
    tracks = torch.cat([short_batched, long], dim=0)
    padding = torch.tensor([[False, False, False, True, True, True, True],
                            [False, False, False, False, False, False, False]])

    mixed, mixed_mask = pool_local_tracks(tracks, padding, 4, "per_sample_v2")
    solo, solo_mask = pool_local_tracks(short, None, 4, "per_sample_v2")

    assert mixed.shape == (2, 4, 1)
    assert mixed_mask[0].tolist() == [False, False, False, True]
    assert mixed_mask[1].tolist() == [False, False, False, False]
    torch.testing.assert_close(mixed[0, :3], solo[0])
    assert solo_mask.shape == (1, 3)
    torch.testing.assert_close(mixed[1], torch.nn.functional.adaptive_avg_pool1d(long.transpose(1, 2), 4).transpose(1, 2)[0])


def test_legacy_uniform_pooling_keeps_historical_short_sample_expansion():
    tracks = torch.tensor([[[1.0], [3.0], [0.0], [0.0], [0.0], [0.0], [0.0]],
                           [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]])
    padding = torch.tensor([[False, False, True, True, True, True, True],
                            [False, False, False, False, False, False, False]])
    pooled, pooled_mask = pool_local_tracks(tracks, padding, 4, "legacy_uniform")
    expected = torch.nn.functional.adaptive_avg_pool1d(tracks[0, :2].transpose(0, 1).unsqueeze(0), 4)[0].transpose(0, 1)
    assert pooled.shape == (2, 4, 1)
    assert not pooled_mask.any()
    torch.testing.assert_close(pooled[0], expected)


def test_per_sample_encoder_memory_matches_alone_and_in_a_mixed_length_batch():
    model = EventConditionedTransformer(_tiny_config(
        planner_profile="music_structure_v3", onset_profile="chart_onsets_v1",
        local_pooling_policy="per_sample_v2", max_audio_tokens=4))
    model.eval()
    short = torch.randn(1, 3, 8)
    long = torch.randn(1, 7, 8)
    mixed = torch.cat([torch.cat([short, torch.zeros(1, 4, 8)], dim=1), long], dim=0)
    padding = torch.tensor([[False, False, False, True, True, True, True],
                            [False, False, False, False, False, False, False]])
    core = ~padding
    global_audio = torch.randn(1, 2, 8).expand(2, -1, -1).clone()
    with torch.no_grad():
        solo_memory, _ = model.encode_audio(short, torch.zeros(1, 16), global_audio=global_audio[:1])
        batch_memory, batch_mask = model.encode_audio(
            mixed, torch.zeros(2, 16), audio_padding_mask=padding,
            global_audio=global_audio, plan_audio_mask=padding)
    torch.testing.assert_close(batch_memory[0, :3], solo_memory[0, :3], atol=1e-5, rtol=1e-5)
    assert batch_mask[0, :4].tolist() == [False, False, False, True]
    assert model.local_onset_logits(batch_memory).shape == (2, 4)


def test_optional_onset_head_forward_keeps_old_return_contract_and_adds_local_logits():
    base = EventConditionedTransformer(_tiny_config(planner_profile="music_structure_v3"))
    enabled = EventConditionedTransformer(_tiny_config(
        planner_profile="music_structure_v3", onset_profile="chart_onsets_v1",
        local_pooling_policy="per_sample_v2"))
    enabled.load_weights_compatible(base.state_dict(), allow_missing_onset_head=True)
    enabled.eval()
    args = (torch.randn(2, 5, 8), torch.ones(2, 3, dtype=torch.long), torch.zeros(2, 16))
    with torch.no_grad():
        old_shape = base(*args, return_plan=True)
        output = enabled(*args, return_plan=True, return_onset=True)
    assert len(old_shape) == 2
    assert len(output) == 3
    assert output[0].shape == (2, 3, 16)
    assert output[1].shape == (2, 13)
    assert output[2].shape == (2, 5)


def test_onset_head_requires_explicit_init_migration_and_complete_weights():
    legacy = EventConditionedTransformer(_tiny_config(planner_profile="music_structure_v3"))
    enabled = EventConditionedTransformer(_tiny_config(
        planner_profile="music_structure_v3", onset_profile="chart_onsets_v1",
        local_pooling_policy="per_sample_v2"))
    try:
        enabled.load_weights_compatible(legacy.state_dict())
    except ValueError as error:
        assert "onset head" in str(error)
    else:  # pragma: no cover
        raise AssertionError("a missing onset head must require explicit init migration")

    compatibility = enabled.load_weights_compatible(
        legacy.state_dict(), allow_missing_onset_head=True)
    assert compatibility["onset_head"] is False
    complete = EventConditionedTransformer(_tiny_config(
        planner_profile="music_structure_v3", onset_profile="chart_onsets_v1",
        local_pooling_policy="per_sample_v2"))
    assert complete.load_weights_compatible(enabled.state_dict())["onset_head"] is True
    partial = {key: value for key, value in enabled.state_dict().items() if key != "onset_head.2.bias"}
    try:
        complete.load_weights_compatible(partial, allow_missing_onset_head=True)
    except ValueError as error:
        assert "incomplete" in str(error)
    else:  # pragma: no cover
        raise AssertionError("partial onset weights must never be treated as a fresh head")
