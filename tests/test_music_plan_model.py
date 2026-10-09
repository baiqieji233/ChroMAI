import torch

from maichart.generation.event_model import EventConditionedTransformer, EventTransformerConfig
from maichart.generation.music_plan import MUSIC_STRUCTURE_PROFILE, plan_values, target_to_logits


def _config(**extra):
    values = dict(input_dim=8, condition_dim=4, vocab_size=12, d_model=16, nhead=2,
                  num_layers=1, dim_feedforward=32, max_audio_tokens=8,
                  global_audio_tokens=4, max_tokens=16, dropout=0.0)
    values.update(extra)
    return EventTransformerConfig(**values)


def test_v2_profile_forces_twelve_channels_and_uses_three_pools():
    model = EventConditionedTransformer(_config(planner_profile=MUSIC_STRUCTURE_PROFILE, plan_dim=9))
    audio = torch.randn(2, 5, 8)
    global_audio = torch.randn(2, 3, 8)
    _, plan = model(audio, torch.ones(2, 2, dtype=torch.long), torch.zeros(2, 4),
                    global_audio=global_audio, return_plan=True)
    assert model.config.plan_dim == 12
    assert plan.shape == (2, 12)
    assert model.plan_pool_projection[0].in_features == 48


def test_v2_pooling_keeps_local_and_global_regions_distinct():
    model = EventConditionedTransformer(_config(planner_profile=MUSIC_STRUCTURE_PROFILE))
    model.eval()
    local = torch.ones(1, 2, 8)
    global_audio = torch.zeros(1, 2, 8)
    first, _ = model.encode_audio(local, torch.zeros(1, 4), global_audio=global_audio)
    second, _ = model.encode_audio(global_audio, torch.zeros(1, 4), global_audio=local)
    _, _, plan_first = model.append_plan_memory(first, None)
    _, _, plan_second = model.append_plan_memory(second, None)
    assert not torch.allclose(plan_first, plan_second)


def test_v2_core_mask_excludes_margin_from_local_pooling():
    model = EventConditionedTransformer(_config(planner_profile=MUSIC_STRUCTURE_PROFILE))
    model.eval()
    audio = torch.zeros(1, 4, 8)
    audio[:, 1:3] = 3.0
    core_mask = torch.tensor([[True, False, False, True]])
    memory, _ = model.encode_audio(audio, torch.zeros(1, 4), plan_audio_mask=core_mask)
    _, _, masked_plan = model.append_plan_memory(memory, None)
    memory_unmasked, _ = model.encode_audio(audio, torch.zeros(1, 4))
    _, _, full_plan = model.append_plan_memory(memory_unmasked, None)
    assert not torch.allclose(masked_plan, full_plan)
    assert getattr(memory, "_planner_core_mask").shape == (1, 4)


def test_v2_core_mask_tracks_audio_downsampling():
    model = EventConditionedTransformer(_config(planner_profile=MUSIC_STRUCTURE_PROFILE, max_audio_tokens=2))
    audio = torch.randn(1, 5, 8)
    core_mask = torch.tensor([[True, False, False, True, True]])
    memory, _ = model.encode_audio(audio, torch.zeros(1, 4), plan_audio_mask=core_mask)
    assert memory._planner_core_mask.shape == (1, 2)


def test_teacher_mix_changes_decoder_memory_but_not_raw_prediction():
    torch.manual_seed(3)
    model = EventConditionedTransformer(_config(planner_profile=MUSIC_STRUCTURE_PROFILE),)
    model.eval()
    audio = torch.randn(2, 5, 8)
    tokens = torch.ones(2, 2, dtype=torch.long)
    condition = torch.zeros(2, 4)
    target = torch.tensor([[.4, .2, .1, .2, .8, .1, .2, .7, 1 / 3],
                           [.2, .1, .3, .1, .4, .3, .1, .2, 1.0]])
    logits0, plan0 = model(audio, tokens, condition, plan_target=target,
                            plan_teacher_mix=0.0, return_plan=True)
    logits1, plan1 = model(audio, tokens, condition, plan_target=target,
                            plan_teacher_mix=1.0, return_plan=True)
    assert torch.allclose(plan0, plan1)
    assert not torch.allclose(logits0, logits1)
    assert model.plan_head[-1].weight.grad is None
    logits1.sum().backward()
    assert model.plan_head[-1].weight.grad is not None


def test_v2_control_is_soft_and_legacy_six_channel_still_constructs():
    model = EventConditionedTransformer(_config(planner_profile=MUSIC_STRUCTURE_PROFILE))
    memory = torch.randn(2, 4, 16)
    _, _, raw = model.append_plan_memory(memory, None,
                                         plan_controls={"count": .9, "__strength": .5})
    assert raw.shape == (2, 12)
    old = EventConditionedTransformer(_config(plan_dim=6))
    assert old.config.plan_dim == 6
    assert plan_values(torch.zeros(1, 6), profile="legacy_v1").shape == (1, 6)


def test_target_contract_is_nine_values_to_v2_logits():
    target = torch.tensor([[.5, .2, .1, .1, .9, .2, .3, .4, 2 / 3]])
    assert target_to_logits(target, profile=MUSIC_STRUCTURE_PROFILE).shape == (1, 12)
