import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from maichart.generation import learned_curve
from maichart.generation.event_audio import EventAudio, ACOUSTIC_DIM


def test_curve_preserves_model_predictions_and_records_analysis_inputs(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"model identity")
    features = np.zeros((100, ACOUSTIC_DIM), dtype=np.float32)
    features[:, 95] = 1
    audio = EventAudio(features=features, times=np.arange(100, dtype=np.float32) * .02,
                       duration_sec=2, tempo_bpm=120, tempo_changes=[], metadata={})

    class Model:
        config = SimpleNamespace(planner_profile="music_structure_v2", global_audio_tokens=4)

        def encode_audio(self, x, condition, **kwargs):
            assert condition[0, 0] == pytest.approx(14 / 15)
            return torch.zeros(1, 5, 8), None

        def append_plan_memory(self, memory, padding):
            logits = torch.zeros(1, 12)
            logits[0, 7] = torch.logit(torch.tensor(.7))
            return memory, padding, logits

    contract = {"segment_beats": 4, "family_params": {"audio_margin_beats": 0}}
    monkeypatch.setattr(learned_curve, "load_event_model", lambda *args: (Model(), None, {"data_contract": contract}))
    monkeypatch.setattr(learned_curve, "extract_event_audio", lambda *a, **kw: audio)
    monkeypatch.setattr(learned_curve, "generate_intensity_profile", lambda *a, **kw: {
        "schema": "maichart-intensity-curve-v1", "duration_sec": 2,
        "audio_sha256": "audio", "points": [{"time_sec": 0, "intensity": .4},
                                                 {"time_sec": 2, "intensity": .4}],
        "spectrogram": {"values": [[0, 255]]},
    })
    profile = learned_curve.generate_learned_profile(tmp_path / "audio.wav", checkpoint,
                                                     bpm=120, condition={"level": 14})
    assert profile["method"] == "learned_music_structure_v2"
    assert profile["points"][0]["intensity"] == pytest.approx(.7)
    assert profile["condition"]["level"] == 14
    assert profile["bpm_input"] == 120
    assert profile["checkpoint_identity"]["size"] == len(b"model identity")
    assert profile["planner_windows"][0]["touch"] == .5
    json.dumps(profile)


def test_old_checkpoint_cannot_claim_to_predict_authored_workload(monkeypatch):
    model = SimpleNamespace(config=SimpleNamespace(planner_profile="legacy_v1"))
    monkeypatch.setattr(learned_curve, "load_event_model", lambda *a: (model, None, {}))
    with pytest.raises(ValueError, match="music_structure_v2"):
        learned_curve.generate_learned_profile("audio.wav", "old.pt")


def test_v3_curve_reads_all_nine_continuous_channels_and_four_config_logits(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model-v3.pt"
    checkpoint.write_bytes(b"v3 model")
    features = np.zeros((100, ACOUSTIC_DIM), dtype=np.float32)
    features[:, 95] = 1
    audio = EventAudio(features=features, times=np.arange(100, dtype=np.float32) * .02,
                       duration_sec=2, tempo_bpm=120, tempo_changes=[], metadata={})

    class Model:
        config = SimpleNamespace(planner_profile="music_structure_v3", global_audio_tokens=4)

        def encode_audio(self, x, condition, **kwargs):
            return torch.zeros(1, 5, 8), None

        def append_plan_memory(self, memory, padding):
            logits = torch.zeros(1, 13)
            logits[0, 7] = torch.logit(torch.tensor(.7))
            logits[0, 8] = torch.logit(torch.tensor(.9))
            logits[0, 9:] = torch.tensor([-2.0, -1.0, 0.0, 2.0])
            return memory, padding, logits

    contract = {"segment_beats": 4, "family_params": {"audio_margin_beats": 0}}
    monkeypatch.setattr(learned_curve, "load_event_model", lambda *args: (Model(), None, {"data_contract": contract}))
    monkeypatch.setattr(learned_curve, "extract_event_audio", lambda *a, **kw: audio)
    monkeypatch.setattr(learned_curve, "generate_intensity_profile", lambda *a, **kw: {
        "schema": "maichart-intensity-curve-v1", "duration_sec": 2,
        "audio_sha256": "audio", "points": [{"time_sec": 0, "intensity": .4},
                                                 {"time_sec": 2, "intensity": .4}],
    })
    profile = learned_curve.generate_learned_profile(tmp_path / "audio.wav", checkpoint, bpm=120)
    window = profile["planner_windows"][0]
    assert profile["method"] == "learned_music_structure_v3"
    assert window["intensity"] == pytest.approx(.7)
    assert window["fine_rhythm"] == pytest.approx(.9)
    assert len(window["configuration_probabilities"]) == 4


def test_learned_curve_integrates_explicit_tempo_map_and_anchors_clipped_audio(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model-tempo.pt"
    checkpoint.write_bytes(b"tempo model")
    features = np.zeros((100, ACOUSTIC_DIM), dtype=np.float32)
    features[:, 95] = 1
    audio = EventAudio(features=features, times=np.arange(100, dtype=np.float32) * .02,
                       duration_sec=2, tempo_bpm=180, tempo_changes=[], metadata={})

    class Model:
        config = SimpleNamespace(planner_profile="music_structure_v2", global_audio_tokens=4)

        def encode_audio(self, x, condition, **kwargs):
            return torch.zeros(1, 5, 8), None

        def append_plan_memory(self, memory, padding):
            logits = torch.zeros(1, 12)
            logits[0, 7] = torch.logit(torch.tensor(.6))
            return memory, padding, logits

    contract = {"segment_beats": 4, "family_params": {"audio_margin_beats": 0}}
    anchors = []
    monkeypatch.setattr(learned_curve, "load_event_model", lambda *args: (Model(), None, {"data_contract": contract}))
    monkeypatch.setattr(learned_curve, "extract_event_audio", lambda *a, **kw: audio)
    monkeypatch.setattr(learned_curve, "audio_segment_tensor", lambda audio, start, end, **kwargs: (
        anchors.append((start, end, kwargs["start_beat"])) or torch.zeros(100, ACOUSTIC_DIM)))
    monkeypatch.setattr(learned_curve, "generate_intensity_profile", lambda *a, **kw: {
        "schema": "maichart-intensity-curve-v1", "duration_sec": 2,
        "audio_sha256": "audio", "points": [{"time_sec": 0, "intensity": .4},
                                                 {"time_sec": 2, "intensity": .4}],
    })
    tempo_map = [{"beat": "0", "bpm": 120}, {"beat": "1", "bpm": 240}]
    result = learned_curve.generate_learned_profile(
        tmp_path / "audio.wav", checkpoint, bpm=180, first=-1, tempo_map=tempo_map)
    assert result["bpm"] == pytest.approx(120)
    assert result["bpm_input"] == 180
    assert result["tempo_map_sha256"]
    assert result["explicit_tempo_map"] == [{"beat": "0", "bpm": 120.0},
                                            {"beat": "1", "bpm": 240.0}]
    assert anchors[0][0] == pytest.approx(0)
    assert anchors[0][2] == pytest.approx(3)
    assert anchors[-1][1] == pytest.approx(2)
