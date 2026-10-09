"""Isolated CPU smoke for the new planner, quality weighting and resume."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from maichart.generation.event_audio import EventAudio, ACOUSTIC_DIM
from maichart.training import event_dataset, train_events, train_event_ctx
from maichart.training.event_ctx_dataset import EventCtxDataset
from maichart.training.quality_annotations import iter_catalog, save_annotations


def test_structure_training_quality_preference_resume_and_finetune(tmp_path, monkeypatch):
    for name, notes in (("normal", "1,2,3,4,1,2,3,4,"),
                        ("negative", "1,2,1,2,1,2,1,2,")):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "maidata.txt").write_text(
            f"&title={name}\n&wholebpm=120\n&first=0\n&lv_5=13\n&inote_5=(120){{4}}{notes}E\n", encoding="utf-8")
        (folder / "track.mp3").write_bytes(name.encode())
    manifest = tmp_path / "training_manifest.json"
    manifest.write_text(json.dumps({"songs": [
        {"song_id": name, "maidata_path": f"{name}/maidata.txt", "audio_path": f"{name}/track.mp3",
         "difficulties": [{"difficulty_index": 5, "level_raw": "13"}]}
        for name in ("normal", "negative")]}), encoding="utf-8")
    corrected = tmp_path / "corrected.txt"
    corrected.write_text("&wholebpm=120\n&first=0\n&lv_5=13\n&inote_5=(120){4}1,3,2,4,1,3,2,4,E\n", encoding="utf-8")
    rows = iter_catalog(manifest)
    normal, negative = (next(row for row in rows if row.song_id == name)
                        for name in ("normal", "negative"))
    records = [{**normal.__dict__, "label": "excellent", "weight": 2.0},
               {**negative.__dict__, "label": "negative", "weight": 0.0,
                "corrected_path": str(corrected), "corrected_difficulty": 5}]
    annotation_path = save_annotations(manifest, records)
    features = np.zeros((200, ACOUSTIC_DIM), dtype=np.float32)
    features[:, 95] = 1.0
    audio = EventAudio(features=features, times=np.arange(200, dtype=np.float32) * 0.02,
                       duration_sec=4.0, tempo_bpm=120.0, tempo_changes=[], metadata={})
    monkeypatch.setattr(event_dataset, "extract_event_audio", lambda *a, **kw: audio)
    monkeypatch.setattr(train_events, "DATASET_CLASS", EventCtxDataset)
    monkeypatch.setattr(train_events, "TARGET_SCHEMA", "event_ctx_v1")
    monkeypatch.setattr(train_events, "dataset_kwargs", train_event_ctx.family_dataset_kwargs)
    common = ["--manifest", str(manifest), "--cache-dir", str(tmp_path / "cache"),
              "--device", "cpu", "--epochs", "1", "--max-steps", "1", "--batch-size", "1",
              "--segment-beats", "4", "--max-tokens", "128", "--max-audio-tokens", "32",
              "--global-frames", "4", "--context-beats", "0", "--audio-margin-beats", "0",
              "--context-tokens", "16", "--context-dropout", "0", "--d-model", "16",
              "--nhead", "2", "--num-layers", "1", "--dim-feedforward", "32", "--dropout", "0",
              "--grad-accumulation", "1", "--prewarm-workers", "off", "--prefetch-batches", "0",
              "--quality-annotations", str(annotation_path), "--preference-enabled",
              "--preference-max-pairs", "2", "--slide-oversample", "1", "--touch-oversample", "1"]
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        output = tmp_path / "fresh"
        positive = [arg for arg in common if arg != "--preference-enabled"]
        assert train_events.main(positive + ["--checkpoint-dir", str(output)]) == 0
        fresh = torch.load(output / "event_last.pt", map_location="cpu", weights_only=False)
        assert fresh["model_config"]["plan_dim"] == 12
        assert fresh["model_config"]["planner_profile"] == "music_structure_v2"
        assert fresh["data_contract"]["planner_params"]["teacher_mix"] == 0.5
        preference_dir = tmp_path / "preference"
        assert train_events.main(common + ["--init-checkpoint", str(output / "event_last.pt"),
                                          "--checkpoint-dir", str(preference_dir)]) == 0
        checkpoint = preference_dir / "event_last.pt"
        preferred = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert preferred["metrics"]["preference"]["pairs"] == 2
        assert preferred["preference_reference_scores"]
        resumed_dir = tmp_path / "resumed"
        assert train_events.main(common + ["--resume", str(checkpoint), "--checkpoint-dir", str(resumed_dir)]) == 0
        resumed = torch.load(resumed_dir / "event_last.pt", map_location="cpu", weights_only=False)
        assert resumed["epoch"] == 2
        assert resumed["preference_reference_scores"] == preferred["preference_reference_scores"]
        records[0]["weight"] = 3.0
        save_annotations(manifest, records)
        with pytest.raises(ValueError, match="quality_params"):
            train_events.main(common + ["--resume", str(checkpoint), "--checkpoint-dir", str(tmp_path / "invalid")])
        finetune = tmp_path / "finetune"
        assert train_events.main(common + ["--init-checkpoint", str(checkpoint), "--checkpoint-dir", str(finetune)]) == 0
        tuned = torch.load(finetune / "event_last.pt", map_location="cpu", weights_only=False)
        assert tuned["training_origin"]["optimizer_reset"] is True
        assert tuned["epoch"] == 1
        assert tuned["data_contract"]["quality_params"] != preferred["data_contract"]["quality_params"]
    finally:
        torch.set_num_threads(threads)
