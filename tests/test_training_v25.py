from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from maichart.models.transformer_v25 import MaichartTransformerV25
from maichart.training.collate import collate_v25
from maichart.training.dataset_v25 import MaichartV25Dataset
from maichart.training.evaluate_v25 import compute_v25_metrics
from maichart.training.losses_v25 import compute_v25_losses
from tests.v25_fixture import write_v25_fixture


def test_v25_minimal_dataset_collate_forward_loss_and_metrics(tmp_path) -> None:
    manifest = write_v25_fixture(tmp_path)
    dataset = MaichartV25Dataset(manifest, cache_dir=tmp_path / "cache")

    sample = dataset[0]
    assert sample["meta"]["song_id"] == "song001"
    assert sample["meta"]["difficulty_index"] == 4
    assert sample["x"].shape == (4, dataset.input_dim)
    assert sample["y"]["note_presence"].shape == (4, 1)
    assert sample["y"]["buttons"].shape == (4, 8)

    batch = collate_v25([sample])
    assert batch["x"].shape == (1, 4, dataset.input_dim)
    assert batch["padding_mask"].shape == (1, 4)
    assert batch["loss_mask"].all()

    model = MaichartTransformerV25(
        input_dim=dataset.input_dim,
        num_note_types=dataset.num_note_types,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_feedforward=64,
        dropout=0.0,
    )
    outputs = model(batch["x"], padding_mask=batch["padding_mask"])
    assert outputs["note_presence_logits"].shape == (1, 4, 1)
    assert outputs["button_logits"].shape == (1, 4, 8)

    losses = compute_v25_losses(
        outputs,
        batch["y"],
        batch["loss_mask"],
        note_pos_weight=1.0,
        button_pos_weight="auto",
    )
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])
    assert any(parameter.grad is not None for parameter in model.parameters())

    metrics = compute_v25_metrics(outputs, batch["y"], batch["loss_mask"])
    assert set(metrics) >= {
        "note_presence_precision",
        "note_presence_recall",
        "note_presence_f1",
        "button_micro_precision",
        "button_micro_recall",
        "button_micro_f1",
        "button_micro_f1@0.05",
        "button_micro_f1@0.10",
        "button_micro_f1@0.20",
        "button_macro_f1",
        "button_best_f1",
        "button_best_threshold",
        "note_type_accuracy",
        "note_best_f1",
        "note_best_threshold",
        "density_mae",
    }
    assert all(isinstance(value, float) for value in metrics.values())
