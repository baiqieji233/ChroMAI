import json
from pathlib import Path

import numpy as np
import pytest
import torch

from maichart.generation.event_audio import ACOUSTIC_DIM, EventAudio
from maichart.training import event_dataset as event_dataset_module
from maichart.training import prepare_events
from maichart.training.event_dataset import EventDataset, collate_events
from maichart.training.event_ctx_dataset import EventCtxDataset
from maichart.training.event_checkpoints import compare_event_contract
from maichart.training.quality_annotations import iter_catalog, save_annotations
from maichart.training.train_events import _quality_weighted_mean


@pytest.fixture
def quality_manifest(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    note_text = "(120){4}1,2,3,4," * 100
    for name in ("excellent", "excluded", "normal"):
        song = raw / name
        song.mkdir()
        (song / "maidata.txt").write_text(
            f"&title={name}\n&wholebpm=120\n&first=0\n&lv_5=13\n&des_5=Tester\n&inote_5={note_text}\n",
            encoding="utf-8",
        )
        (song / "track.mp3").write_bytes(name.encode())
    manifest = tmp_path / "manifest.json"
    assert prepare_events.main([str(raw), "--cache-dir", str(tmp_path / "cache"), "-o", str(manifest)]) == 0
    audio = EventAudio(
        features=np.zeros((500, ACOUSTIC_DIM), dtype=np.float32),
        times=np.arange(500, dtype=np.float32) * 0.02,
        duration_sec=10.0, tempo_bpm=120.0, tempo_changes=[], metadata={},
    )
    monkeypatch.setattr(event_dataset_module, "extract_event_audio", lambda *a, **kw: audio)
    rows = iter_catalog(manifest)
    by_title = {Path(row.source_path).parent.name: row for row in rows}
    save_annotations(manifest, [
        {"chart_id": by_title["excellent"].chart_id, "song_id": by_title["excellent"].song_id, "difficulty_index": 5,
         "source_path": by_title["excellent"].source_path, "chart_sha256": by_title["excellent"].chart_sha256,
         "label": "excellent", "weight": 2.0,
         "regions": [{"start_sec": 0.0, "end_sec": 2.1, "label": "locally less convincing", "weight": 0.25}]},
        {"chart_id": by_title["excluded"].chart_id, "song_id": by_title["excluded"].song_id, "difficulty_index": 5,
         "source_path": by_title["excluded"].source_path, "chart_sha256": by_title["excluded"].chart_sha256,
         "label": "excluded", "weight": 0.0},
    ])
    return manifest, tmp_path / "cache"


def _dataset(manifest, cache, cls=EventDataset, **kwargs):
    return cls(manifest, cache_dir=cache, segment_beats=400, max_tokens=2048,
               quality_annotations=manifest.with_name("quality_annotations.json"), **kwargs)


def test_quality_weights_are_visible_at_event_token_granularity(quality_manifest):
    manifest, cache = quality_manifest
    dataset = _dataset(manifest, cache)
    assert len(dataset.samples) == 3
    # The excluded chart's two windows do not enter positive SFT.
    assert len(dataset) == 4
    assert len(dataset.preference_segments) == 6
    excellent_index = next(i for i, ref in enumerate(dataset.samples) if Path(ref.source_path).parent.name == "excellent")
    first_window = next(i for i, segment in enumerate(dataset.segments)
                        if segment[0] == excellent_index and segment[1] == 0)
    item = dataset[first_window]
    weights = item["quality_token_weights"]
    assert weights.min().item() == pytest.approx(0.5)  # 2.0 chart weight × 0.25 local weight
    assert weights.max().item() == pytest.approx(2.0)
    assert 1.0 < item["quality_sample_weight"].item() < 2.0
    normalizers = dataset.quality_normalizers()
    assert 1.0 < normalizers["token"] < 2.0
    batch = collate_events([item, dataset[0]])
    assert batch["quality_token_weights"].shape == batch["target"].shape
    assert batch["quality_sample_weights"].shape == (2,)


def test_local_quality_does_not_turn_rejected_notes_into_sparse_planner_targets(quality_manifest):
    manifest, cache = quality_manifest
    dataset = _dataset(manifest, cache)
    index = next(i for i, ref in enumerate(dataset.samples) if Path(ref.source_path).parent.name == "excellent")
    ref = dataset.samples[index]
    raw = dataset._window_plan(ref, 0, 16)
    filtered = dataset._quality_window_plan(index, ref, 0, 16)
    assert torch.equal(filtered, raw)
    assert dataset._quality_plan_weight(index, 0, 16, 2.0) == 0.0
    assert dataset._quality_plan_weight(index, 16, 32, 2.0) == 2.0


def test_context_dataset_removes_excluded_samples_and_keeps_compact_quality_spans(quality_manifest):
    manifest, cache = quality_manifest
    dataset = _dataset(manifest, cache, EventCtxDataset, context_beats=8, context_tokens=128,
                       context_dropout=0)
    assert len(dataset.preference_segments) > len(dataset)
    assert all(spec is None or len(spec) < len(entry[3]) - 1
               for entry, spec in zip(dataset.segments, dataset.segment_quality_token_weights))
    item = dataset[0]
    assert item["quality_token_weights"].numel() == item["target"].numel()
    batch = collate_events([item])
    assert batch["quality_token_weights"].shape == batch["target"].shape


def test_quality_token_loss_uses_fixed_normalizer_and_event_weights():
    losses = torch.tensor([[2.0, 4.0]], requires_grad=True)
    weights = torch.tensor([[1.0, 3.0]])
    valid = torch.tensor([[True, True]])
    loss = _quality_weighted_mean(losses, weights, normalizer=2.0, valid=valid)
    assert loss.item() == pytest.approx(3.5)
    loss.backward()
    assert losses.grad[0, 0].item() == pytest.approx(0.25)
    assert losses.grad[0, 1].item() == pytest.approx(0.75)


def test_quality_annotation_changes_require_finetune_not_resume():
    fixed = {
        "target_schema": "event_v1", "feature_profile": "event_audio_v2_rhythm",
        "input_dim": 112, "sample_rate": 22050, "hop_length": 441,
        "segment_beats": 16, "max_tokens": 2048, "vocabulary_signature": "vocab",
        "condition_keys": ["level"], "dataset_signature": "dataset-v1",
        "quality_params": {"annotations_fingerprint": "old-labels"},
    }
    stored = {"data_contract": fixed, "optimizer_state_dict": {"state": {}}}
    current = {**fixed, "dataset_signature": "dataset-v2",
               "quality_params": {"annotations_fingerprint": "new-labels"}}
    with pytest.raises(ValueError, match="quality_params"):
        compare_event_contract(stored, current, mode="resume")
    compare_event_contract(stored, current, mode="finetune")
