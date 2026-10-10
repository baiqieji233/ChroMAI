import json
from fractions import Fraction as F

import numpy as np
import pytest

from maichart.generation import event_cli
from maichart.generation.event_audio import ACOUSTIC_DIM, EventAudio
from maichart.generation.event_model import EventSequence, EventVocabulary
from maichart.generation.timeline import normalize_tempo_map, serialize_tempo_map, tempo_map_fingerprint, beat_at_audio_end
from maichart.event_timing import seconds_at_beat


MAP = [{"beat": "0", "bpm": 120}, {"beat": "4", "bpm": 60}]


@pytest.mark.parametrize("bad", [[], [{"beat": 1, "bpm": 120}],
    [{"beat": 0, "bpm": 0}], [{"beat": 0, "bpm": float("nan")}],
    [{"beat": 0, "bpm": 120}, {"beat": 0, "bpm": 60}],
    [{"beat": 0, "bpm": True}], [{"beat": True, "bpm": 120}]])
def test_explicit_map_rejects_ambiguous_timing(bad):
    with pytest.raises(ValueError):
        normalize_tempo_map(bad)


def test_piecewise_inverse_includes_first_and_round_trips_fractional_end():
    entries = normalize_tempo_map(MAP)
    assert beat_at_audio_end(6, entries) == 8  # not 12 at initial BPM
    assert beat_at_audio_end(6, entries, first=-.25) == F(33, 4)
    assert float(seconds_at_beat(F(33, 4), entries, -.25)) == pytest.approx(6)
    assert beat_at_audio_end(1, entries, first=2) == 0
    assert tempo_map_fingerprint(MAP) == tempo_map_fingerprint({"tempo_changes": MAP})
    assert serialize_tempo_map(entries) == [{"beat": "0", "bpm": 120.}, {"beat": "4", "bpm": 60.}]


def test_event_cli_explicit_clock_controls_features_windows_and_export(tmp_path, monkeypatch):
    audio = EventAudio(np.zeros((301, ACOUSTIC_DIM), dtype=np.float32),
                       np.linspace(0, 6, 301, dtype=np.float32), 6., 180.)
    vocab = EventVocabulary.from_event_charts([], segment_beats=4)
    checkpoint = {"schema": "maichart-event-generator-checkpoint-v1", "epoch": 1,
                  "data_contract": {"target_schema": "event_v1", "segment_beats": 4}}
    monkeypatch.setattr(event_cli, "load_event_model", lambda *a: (object(), vocab, checkpoint))
    monkeypatch.setattr(event_cli, "extract_event_audio", lambda *a, **k: audio)
    calls = []
    def decode(*a, **kw):
        calls.append(kw["position_end"])
        return EventSequence([F(0), F(3)], ["1", "2"])
    monkeypatch.setattr(event_cli, "_decode", decode)
    path = tmp_path / "tempo.json"
    path.write_text(json.dumps(MAP), encoding="utf-8")
    output = tmp_path / "out.txt"
    assert event_cli.main(["--checkpoint", "unused.pt", "--audio", "unused.wav", "--output", str(output),
        "--device", "cpu", "--tempo-map", str(path), "--bpm", "240", "--first", "-0.25",
        "--no-intensity-follow", "--no-audio-bounds-trim"]) == 0
    ir = json.loads(output.with_suffix(".chart.json").read_text(encoding="utf-8"))
    assert ir["tempo_changes"] == serialize_tempo_map(MAP)
    assert F(ir["end_beat"]) == F(33, 4)
    assert calls == [F(4), F(4), F(1, 4)]
    assert ir["first"] == -.25
    assert ir["quality"]["timing_inputs"]["tempo_map_sha256"] == tempo_map_fingerprint(MAP)


def test_cli_rejects_curve_made_with_different_explicit_timing(tmp_path, monkeypatch):
    from maichart.generation import music_intensity
    audio = EventAudio(np.zeros((101, ACOUSTIC_DIM), dtype=np.float32),
                       np.linspace(0, 2, 101, dtype=np.float32), 2., 120.)
    vocab = EventVocabulary.from_event_charts([], segment_beats=4)
    checkpoint = {"schema": "maichart-event-generator-checkpoint-v1",
                  "data_contract": {"target_schema": "event_v1"}}
    monkeypatch.setattr(event_cli, "load_event_model", lambda *a: (object(), vocab, checkpoint))
    monkeypatch.setattr(event_cli, "extract_event_audio", lambda *a, **kw: audio)
    monkeypatch.setattr(music_intensity, "audio_sha256", lambda _: "hash")
    curve = tmp_path / "curve.json"
    curve.write_text(json.dumps({"schema": "maichart-intensity-curve-v1", "audio_sha256": "hash",
        "duration_sec":2.,"points":[{"time_sec":0,"intensity":.5},{"time_sec":2,"intensity":.5}],
        "method":"learned_music_structure_v3", "tempo_map_sha256":tempo_map_fingerprint(MAP)}), encoding="utf-8")
    assert event_cli.main(["--checkpoint","model.pt","--audio","song.wav","--output",str(tmp_path/'out.txt'),
                           "--intensity-curve",str(curve),"--device","cpu"]) == 1
    error=json.loads((tmp_path/'out.error.json').read_text(encoding='utf-8'))
    assert 'tempo map' in str(error)
