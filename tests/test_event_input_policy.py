"""Input migrations preserve old continuation and require explicit contracts."""
import copy

import pytest
import torch

from maichart.training.event_input_policy import (
    NEW_INPUT_POLICIES, resolve_input_policies, stored_input_policies,
)
from maichart.training.event_checkpoints import compare_event_contract
from maichart.generation.event_model import pool_local_tracks


def old_checkpoint():
    return {"model_config": {"local_pooling_policy": "legacy_uniform", "onset_profile": "none"},
            "data_contract": {"family_params": {"context_tokens": 512}},
            "optimizer_state_dict": {"state": {}}}


def test_fresh_and_finetune_correct_inputs_without_enabling_onset():
    for mode in ("fresh", "finetune"):
        result = resolve_input_policies({}, checkpoint=old_checkpoint(), mode=mode)
        assert result == NEW_INPUT_POLICIES
        assert result["local_pooling_policy"] == "per_sample_v2"


def test_legacy_resume_inherits_old_input_semantics():
    old = old_checkpoint()
    result = resolve_input_policies({}, checkpoint=old, mode="resume")
    assert result == stored_input_policies(old)
    assert result["context_budget_policy"] == "legacy_dynamic_v1"
    compare_event_contract(old, {**old["data_contract"], "input_policies": result}, mode="resume")
    for key, value in NEW_INPUT_POLICIES.items():
        with pytest.raises(ValueError, match="finetune"):
            resolve_input_policies({key: value}, checkpoint=old, mode="resume")


def test_input_migration_and_larger_context_are_finetune_only():
    old = old_checkpoint()
    current = copy.deepcopy(old["data_contract"])
    current["family_params"]["context_tokens"] = 768
    current["input_policies"] = dict(NEW_INPUT_POLICIES)
    compare_event_contract(old, current, mode="init")
    with pytest.raises(ValueError, match="input_policies"):
        compare_event_contract(old, current, mode="resume")
    current["input_policies"] = stored_input_policies(old)
    with pytest.raises(ValueError, match="family_params"):
        compare_event_contract(old, current, mode="resume")


def test_pooling_is_independent_but_onset_requires_corrected_pooling():
    assert resolve_input_policies({"local_pooling_policy": "per_sample_v2"}, onset_profile="none")
    with pytest.raises(ValueError, match="per_sample_v2"):
        resolve_input_policies({"local_pooling_policy": "legacy_uniform"}, onset_profile="chart_onsets_v1")


def test_short_clip_keeps_its_frames_beside_a_long_clip():
    short = torch.arange(5.).view(1, 5, 1)
    batch = torch.cat((torch.nn.functional.pad(short, (0,0,0,7)), torch.arange(12.).view(1,12,1)))
    mask = torch.zeros(2,12,dtype=torch.bool)
    mask[0,5:] = True
    actual, actual_mask = pool_local_tracks(batch, mask, 10, "per_sample_v2")
    torch.testing.assert_close(actual[0, ~actual_mask[0]], short[0])
    assert int((~actual_mask[0]).sum()) == 5
