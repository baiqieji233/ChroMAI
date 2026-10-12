"""CPU-only old-weight migration and causal generation/curve integration."""
import hashlib
import json
from fractions import Fraction as F

import numpy as np
import pytest
import torch

from maichart.event_timing import EventTempo
from maichart.generation.conditions import GenerationCondition
from maichart.generation.event_audio import ACOUSTIC_DIM, EventAudio
from maichart.generation.event_model import EventSequence
from maichart.generation.sustain_state import neutral_action_state
from maichart.training import event_dataset, train_events, train_event_ctx
from maichart.training.event_checkpoints import load_event_checkpoint, load_event_model
from maichart.training.event_ctx_dataset import EventCtxDataset


def test_existing_weights_finetune_resume_and_generate_with_sustain_state(tmp_path, monkeypatch):
    folder = tmp_path / "song"
    folder.mkdir()
    (folder / "maidata.txt").write_text(
        "&title=test\n&wholebpm=120\n&lv_5=13\n&inote_5=(120){4}1h[4:20],2,3,4,5,6,7,8,1,2,3,4,E\n",
        encoding="utf-8")
    (folder / "track.mp3").write_bytes(b"test audio only")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"songs": [{"song_id": "test", "maidata_path": "song/maidata.txt",
        "audio_path": "song/track.mp3", "difficulties": [{"difficulty_index": 5, "level_raw": "13"}]}]}))
    audio = EventAudio(np.zeros((351, ACOUSTIC_DIM), dtype=np.float32),
                       np.arange(351,dtype=np.float32)*.02, 7., 120.)
    monkeypatch.setattr(event_dataset, "extract_event_audio", lambda *a, **kw: audio)
    monkeypatch.setattr(train_events, "DATASET_CLASS", EventCtxDataset)
    monkeypatch.setattr(train_events, "TARGET_SCHEMA", "event_ctx_v1")
    monkeypatch.setattr(train_events, "dataset_kwargs", train_event_ctx.family_dataset_kwargs)
    common = ["--manifest", str(manifest), "--cache-dir", str(tmp_path/'cache'), "--device", "cpu",
              "--epochs", "1", "--batch-size", "1", "--segment-beats", "4", "--max-tokens", "128",
              "--context-beats", "4", "--context-tokens", "32", "--context-dropout", "0",
              "--audio-margin-beats", "0", "--max-audio-tokens", "16", "--global-frames", "4",
              "--d-model", "16", "--nhead", "2", "--num-layers", "1", "--dim-feedforward", "32",
              "--grad-accumulation", "1", "--prewarm-workers", "off", "--prefetch-batches", "0"]
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        legacy_dir, tuned_dir, resume_dir = (tmp_path/x for x in ("legacy", "tuned", "resumed"))
        assert train_events.main(common + ["--checkpoint-dir", str(legacy_dir),
            "--local-pooling-policy", "legacy_uniform", "--context-budget-policy", "legacy_dynamic_v1",
            "--chart-filter-policy", "legacy_strict_v1"]) == 0
        legacy = legacy_dir/'event_last.pt'
        original_hash = hashlib.sha256(legacy.read_bytes()).hexdigest()
        assert train_events.main(common + ["--checkpoint-dir", str(tuned_dir), "--init-checkpoint", str(legacy),
                                          "--action-state-profile", "sustain_state_v1"]) == 0
        path = tuned_dir/'event_last.pt'
        payload = torch.load(path, map_location='cpu',weights_only=False)
        assert payload["action_state_training"]["trained"]
        assert payload["action_state_training"]["optimizer_steps"] > 0
        assert payload["metrics"]["validation"] is None
        assert payload["model_config"]["local_pooling_policy"] == "per_sample_v2"
        assert payload["model_config"]["onset_profile"] == "none"
        assert payload["model_config"]["action_state_profile"] == "sustain_state_v1"
        assert train_events.main(common + ["--checkpoint-dir", str(resume_dir), "--resume", str(path)]) == 0
        assert hashlib.sha256(legacy.read_bytes()).hexdigest() == original_hash
        model, vocab, saved = load_event_model(path, torch.device('cpu'))
        # An enabled but never-updated projection can resume training, while
        # inference must not silently use its random weights.
        untrained = dict(payload, action_state_training={
            "profile": "sustain_state_v1", "trained": False, "optimizer_steps": 0})
        untrained_path = tmp_path / "untrained.pt"
        torch.save(untrained, untrained_path)
        assert load_event_checkpoint(untrained_path)["action_state_training"]["trained"] is False
        with pytest.raises(ValueError, match="有效训练更新"):
            load_event_model(untrained_path, torch.device('cpu'))
        from maichart.webui.server import _checkpoint_inspection
        inspection = _checkpoint_inspection(untrained_path, mode="resume", manifest=None,
                                            model_family="event_ctx_v1")
        assert inspection["can_resume"]
        assert not inspection["generation_capabilities"]["sustained_action_memory"]

        # Exercise the real learned-curve entry point with the migrated model;
        # only audio loading/visual reference extraction are replaced.
        from maichart.generation import learned_curve
        monkeypatch.setattr(learned_curve, "extract_event_audio", lambda *a, **kw: audio)
        monkeypatch.setattr(learned_curve, "generate_intensity_profile", lambda *a, **kw: {
            "schema": "maichart-intensity-curve-v1", "duration_sec": 7., "audio_sha256": "test",
            "points": [{"time_sec": 0., "intensity": .5}, {"time_sec": 7., "intensity": .5}],
        })
        curve = learned_curve.generate_learned_profile(folder / "track.mp3", path, bpm=120)
        assert curve["action_state_context"] == "none_before_chart_generation"
        assert curve["method"] == "learned_music_structure_v3"
        assert curve["planner_windows"]
        assert all(np.isfinite(p["intensity"]) for p in curve["points"])

        from maichart.generation import event_cli
        seen = []
        def decode(*args, **kwargs):
            seen.append((kwargs["action_state"].clone(), kwargs["action_state_padding_mask"].clone()))
            return EventSequence([F(0)], ["1h[4:20]" if len(seen)==1 else "2"])
        monkeypatch.setattr(event_cli, "_decode", decode)
        monkeypatch.setattr(event_cli, "audio_segment_tensor", lambda *a, **k: torch.zeros(10,112))
        result = event_cli._decode_windows(model, vocab, audio, torch.zeros(4,112), GenerationCondition(),
            torch.device('cpu'), windows=[(F(0),F(4)),(F(4),F(8)),(F(8),F(12))],
            tempo_changes=[EventTempo(F(0),120)], first=0, song_duration=7, bpm=120,
            max_tokens=128, temperature=.7, sample=False, seed=1, context_beats=4, context_tokens=32)
        assert seen[0][1].all()
        assert not seen[2][1].all()  # source older than the 4-beat textual prefix
        assert result[2].action_state["count"] == 1
        # Quality-excluded long notes cannot leak back through the new memory.
        dataset = EventCtxDataset(manifest, cache_dir=tmp_path/'cache', segment_beats=4,
            max_tokens=128, context_beats=4, context_tokens=32, context_dropout=0,
            action_state_profile="sustain_state_v1")
        dataset.quality_records_by_sample[0] = {"weight": 1., "regions": [
            {"start_sec": 0., "end_sec": .5, "weight": 0.}]}
        later = next(i for i, entry in enumerate(dataset.segments) if entry[1] == 8)
        assert dataset[later]["meta"]["action_state_count"] == 0
        # Curve prediction, which has no prior chart, can explicitly encode an empty state.
        memory, _ = model.encode_audio(torch.zeros(1,10,112), torch.zeros(1,16),
                                     global_audio=torch.zeros(1,4,112), **neutral_action_state(model))
        assert torch.isfinite(memory).all()
    finally:
        torch.set_num_threads(old_threads)
