from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from maichart.generation.model import ConditionedChartTransformer, ConditionedChartTransformerConfig
from maichart.training.checkpoints import compare_checkpoint_contract, load_checkpoint


def test_legacy_state_dict_is_available_for_finetuning_but_not_resume(tmp_path: Path) -> None:
    model = ConditionedChartTransformer(
        ConditionedChartTransformerConfig(d_model=16, nhead=4, num_layers=1, dim_feedforward=32)
    )
    path = tmp_path / "legacy.pt"
    torch.save(model.state_dict(), path)

    checkpoint = load_checkpoint(path)
    init_errors, init_warnings = compare_checkpoint_contract(checkpoint, mode="finetune")
    resume_errors, _ = compare_checkpoint_contract(checkpoint, mode="resume")

    assert checkpoint["checkpoint_kind"] == "legacy_weights"
    assert not init_errors
    assert init_warnings
    assert any("不能用于断点继续" in error for error in resume_errors)


def test_checkpoint_contract_reports_feature_mismatch() -> None:
    checkpoint = {
        "schema": "maichart-conditioned-generator-checkpoint-v2",
        "model_config": {"input_dim": 7, "d_model": 32},
        "data_contract": {
            "feature_profile": "basic_v1",
            "audio_feature_keys": ["a"],
        },
        "optimizer_state_dict": {},
        "scheduler_state_dict": {},
        "epoch": 2,
    }
    errors, _ = compare_checkpoint_contract(
        checkpoint,
        model_config={"input_dim": 77, "d_model": 32},
        data_contract={"feature_profile": "rhythm_v2", "audio_feature_keys": ["b"]},
        mode="resume",
    )

    assert any("input_dim" in error for error in errors)
    assert any("feature_profile" in error for error in errors)
