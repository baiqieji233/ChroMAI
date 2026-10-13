import torch
import pytest

from maichart.form import CHECKPOINT_SCHEMA, FAMILY
from maichart.form.checkpoint import INPUT_CONTRACT, LEGACY_INPUT_CONTRACT, load
from maichart.form.codec import RationalCodec
from maichart.form.model import FormConfig, FormTransformer
from maichart.form.blueprint import BlueprintConfig, SongBlueprint


def test_legacy_song_form_contract_still_loads(tmp_path):
    codec = RationalCodec()
    model = FormTransformer(FormConfig(vocab_size=len(codec), d_model=16, nhead=1,
                                       layers=1, dim_feedforward=32, max_tokens=32,
                                       max_memory_tokens=16))
    planner = SongBlueprint(BlueprintConfig(width=16, form_heads=4, form_layers=1,
                                            temporal_layers=1))
    path = tmp_path / "legacy.pt"
    model_state = {k: v for k, v in model.state_dict().items()
                   if k not in {"condition_null_embedding", "load_curve_projection.weight",
                                "load_curve_projection.bias"}}
    planner_state = {k: v for k, v in planner.state_dict().items()
                     if not k.startswith("load_curve_head.")}
    torch.save({"schema": CHECKPOINT_SCHEMA, "family": FAMILY,
                "input_contract": LEGACY_INPUT_CONTRACT,
                "model_config": model.config.__dict__ if hasattr(model.config, "__dict__") else {
                    "vocab_size": len(codec), "d_model": 16, "nhead": 1, "layers": 1,
                    "dim_feedforward": 32, "dropout": 0.1, "audio_dim": 120,
                    "descriptor_dim": 504, "condition_dim": 16, "max_tokens": 32,
                    "max_audio_tokens": 1024, "max_memory_tokens": 16},
                "model_state_dict": model_state,
                "blueprint_config": {"audio_dim": 120, "descriptor_dim": 504,
                                      "condition_dim": 16, "width": 16, "form_heads": 4,
                                      "form_layers": 1, "temporal_layers": 1, "dropout": 0.1,
                                      "max_form_steps": 128},
                "blueprint_state_dict": planner_state, "codec": codec.to_dict(),
                "optimizer_steps": 1, "blueprint_steps": 1,
                "task_training": {"draft": 1, "revision": 1}}, path)
    loaded, _, _, payload = load(path, inference=True)
    assert loaded.config.vocab_size == len(codec)
    assert payload["input_contract"] == LEGACY_INPUT_CONTRACT


def test_current_contract_rejects_missing_curve_parameters(tmp_path):
    codec = RationalCodec()
    model = FormTransformer(FormConfig(vocab_size=len(codec), d_model=16, nhead=1,
                                       layers=1, dim_feedforward=32, max_tokens=32,
                                       max_memory_tokens=16))
    planner = SongBlueprint(BlueprintConfig(width=16, form_heads=4, form_layers=1,
                                            temporal_layers=1))
    for name, model_state, planner_state in (
        ("model", {k: v for k, v in model.state_dict().items()
                    if k != "load_curve_projection.weight"}, planner.state_dict()),
        ("planner", model.state_dict(), {k: v for k, v in planner.state_dict().items()
                                          if k != "load_curve_head.bias"}),
    ):
        path = tmp_path / f"current_missing_{name}.pt"
        torch.save({"schema": CHECKPOINT_SCHEMA, "family": FAMILY,
                    "input_contract": INPUT_CONTRACT,
                    "model_config": {"vocab_size": len(codec), "d_model": 16, "nhead": 1,
                                     "layers": 1, "dim_feedforward": 32, "dropout": 0.1,
                                     "audio_dim": 120, "descriptor_dim": 504,
                                     "condition_dim": 16, "max_tokens": 32,
                                     "max_audio_tokens": 1024, "max_memory_tokens": 16},
                    "model_state_dict": model_state,
                    "blueprint_config": {"audio_dim": 120, "descriptor_dim": 504,
                                         "condition_dim": 16, "width": 16, "form_heads": 4,
                                         "form_layers": 1, "temporal_layers": 1, "dropout": 0.1,
                                         "max_form_steps": 128},
                    "blueprint_state_dict": planner_state,
                    "codec": codec.to_dict()}, path)
        with pytest.raises(ValueError, match="parameter contract mismatch"):
            load(path)
