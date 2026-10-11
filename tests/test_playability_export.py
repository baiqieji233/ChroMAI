import json
from fractions import Fraction as F

import numpy as np
import pytest

from maichart.generation import event_cli
from maichart.generation.event_audio import EventAudio, ACOUSTIC_DIM
from maichart.generation.event_model import EventVocabulary, EventSequence


@pytest.mark.parametrize("mode", ["none", "local", "reject"])
def test_cli_policy_reaches_exported_artifacts(tmp_path, monkeypatch, mode):
    vocabulary = EventVocabulary.from_event_charts([], segment_beats=4)
    checkpoint = {"schema": "maichart-event-generator-checkpoint-v1", "epoch": 20,
                  "data_contract": {"target_schema": "event_v1", "segment_beats": 4},
                  "model_config": {"planner_profile": "legacy_v1", "plan_dim": 9}}
    monkeypatch.setattr(event_cli, "load_event_model", lambda *a: (object(), vocabulary, checkpoint))
    audio = EventAudio(np.zeros((101, ACOUSTIC_DIM), dtype=np.float32),
                       np.linspace(0, 2, 101, dtype=np.float32), 2.0, 120.0)
    monkeypatch.setattr(event_cli, "extract_event_audio", lambda *a, **k: audio)
    monkeypatch.setattr(event_cli, "_decode", lambda *a, **k:
                        EventSequence([F(0), F(1)], ["1h[4:3]", "1/2b/C"]))
    output = tmp_path / "chart.txt"
    if mode == "reject":
        output.write_text("previous chart", encoding="utf-8")
    code = event_cli.main(["--checkpoint", "test.pt", "--audio", "test.wav",
                           "--output", str(output), "--device", "cpu", "--bpm", "120",
                           "--no-audio-bounds-trim", "--no-intensity-follow",
                           "--enable-conflict-detection", "--conflict-engine", "static",
                           "--conflict-repair-mode", mode])
    report = json.loads(output.with_suffix(".quality.json").read_text(encoding="utf-8"))
    ir = json.loads(output.with_suffix(".chart.json").read_text(encoding="utf-8"))
    assert ir["tempo_changes"] == [{"beat": "0", "bpm": 120.0}]
    if mode == "reject":
        assert code != 0 and not output.exists()
        assert report["generation_status"] == "rejected_by_playability"
        assert list((tmp_path / ".history").glob("*/chart.txt"))[0].read_text(encoding="utf-8") == "previous chart"
    elif mode == "local":
        assert code == 0
        assert ir["points"][-1]["note_text"] == "2b/C"
        assert output.with_suffix(".repaired.txt").read_bytes() == output.read_bytes()
        original = json.loads(output.with_suffix(".original.chart.json").read_text(encoding="utf-8"))
        assert original["points"][-1]["note_text"] == "1/2b/C"
        assert output.with_suffix(".original.txt").exists()
        assert report["playability"]["repairs"]
    else:
        assert code == 0 and ir["points"][-1]["note_text"] == "1/2b/C"
        assert report["playability"]["hard"] == 1
        assert not output.with_suffix(".repaired.txt").exists()
