import copy

import pytest

from maichart.generation.conditions import GENERATION_DEFAULTS, GenerationCondition
from maichart.generation.event_cli import resolve_generation_condition, validate_learned_curve_contract
from maichart.event_timing import EventTempo


def _condition():
    return GenerationCondition(level=13.3, radar=GENERATION_DEFAULTS, style=GENERATION_DEFAULTS)


def _curve(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    condition = _condition()
    return checkpoint, condition, {
        "method": "learned_music_structure_v3",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_identity": {"size": checkpoint.stat().st_size, "mtime_ns": checkpoint.stat().st_mtime_ns},
        "condition": condition.to_dict(),
        "first": 0.25,
        "tempo_map_sha256": None,
        "tempo_mode": "constant",
        "bpm_input": None,
    }


def test_standalone_curve_contract_accepts_matching_context(tmp_path):
    checkpoint, condition, curve = _curve(tmp_path)

    validate_learned_curve_contract(
        curve,
        checkpoint_path=checkpoint,
        checkpoint_identity={"path": str(checkpoint), "size": checkpoint.stat().st_size,
                             "mtime_ns": checkpoint.stat().st_mtime_ns},
        condition=condition,
        first=0.25,
        tempo_map=None,
        tempo_mode="constant",
        bpm_input=None,
    )


def test_standalone_curve_contract_accepts_old_browser_rounded_mtime(tmp_path):
    checkpoint, condition, curve = _curve(tmp_path)
    curve["checkpoint_identity"]["mtime_ns"] = float(curve["checkpoint_identity"]["mtime_ns"])
    validate_learned_curve_contract(
        curve,
        checkpoint_path=checkpoint,
        checkpoint_identity={"path": str(checkpoint), "size": checkpoint.stat().st_size,
                             "mtime_ns": checkpoint.stat().st_mtime_ns,
                             "sha256": "new-curves-also-record-a-hash"},
        condition=condition,
        first=0.25,
        tempo_map=None,
        tempo_mode="constant",
        bpm_input=None,
    )


def test_standalone_curve_contract_rejects_same_size_replacement(tmp_path):
    checkpoint, condition, curve = _curve(tmp_path)
    curve["checkpoint_identity"]["sha256"] = "original-content"
    checkpoint_identity = {"size": checkpoint.stat().st_size,
                           "mtime_ns": checkpoint.stat().st_mtime_ns,
                           "sha256": "replacement-content"}
    with pytest.raises(ValueError, match="identity changed"):
        validate_learned_curve_contract(
            curve, checkpoint_path=checkpoint, checkpoint_identity=checkpoint_identity,
            condition=condition, first=0.25, tempo_map=None, tempo_mode="constant", bpm_input=None)


@pytest.mark.parametrize("field,value", [
    ("first", 0.0),
    ("checkpoint", "other.pt"),
    ("checkpoint_identity", {"size": 1, "mtime_ns": 2}),
    ("tempo_mode", "auto"),
    ("bpm_input", 150.0),
])
def test_standalone_curve_contract_rejects_context_mismatch(tmp_path, field, value):
    checkpoint, condition, curve = _curve(tmp_path)
    altered = copy.deepcopy(curve)
    altered[field] = value

    with pytest.raises(ValueError, match="regenerate it"):
        validate_learned_curve_contract(
            altered,
            checkpoint_path=checkpoint,
            checkpoint_identity={"size": checkpoint.stat().st_size,
                                 "mtime_ns": checkpoint.stat().st_mtime_ns},
            condition=condition,
            first=0.25,
            tempo_map=None,
            tempo_mode="constant",
            bpm_input=None,
        )


def test_standalone_curve_contract_rejects_explicit_tempo_map_change(tmp_path):
    checkpoint, condition, curve = _curve(tmp_path)
    tempo = [EventTempo(0, 120.0)]
    curve["tempo_map_sha256"] = "map-for-120"

    with pytest.raises(ValueError, match="explicit tempo map"):
        validate_learned_curve_contract(
            curve,
            checkpoint_path=checkpoint,
            checkpoint_identity={"size": checkpoint.stat().st_size,
                                 "mtime_ns": checkpoint.stat().st_mtime_ns},
            condition=condition,
            first=0.25,
            tempo_map=tempo,
            tempo_mode="constant",
            bpm_input=None,
        )


@pytest.mark.parametrize("section,key,value", [(None, "level", 14.5), (None, "difficulty", 6),
                                               ("radar", "note_density", .8), ("style", "slide_bias", .6)])
def test_changed_curve_controls_warn_and_preserve_original_snapshot(tmp_path, section, key, value):
    checkpoint, condition, curve = _curve(tmp_path)
    if section:
        curve["condition"][section][key] = value
    else:
        curve["condition"][key] = value
    original = copy.deepcopy(curve)
    messages = validate_learned_curve_contract(
        curve, checkpoint_path=checkpoint,
        checkpoint_identity={"size": checkpoint.stat().st_size, "mtime_ns": checkpoint.stat().st_mtime_ns},
        condition=condition, first=.25, tempo_map=None, tempo_mode="constant", bpm_input=None)
    assert len(messages) == 1 and "沿用原曲线" in messages[0]
    assert curve == original


def test_condition_resolver_accepts_memory_partial_data_and_explicit_zero():
    condition, _ = resolve_generation_condition(
        level=14, condition_data={"level": 14, "style": {"melody_following": 0.0}},
        contract_params={"mask_prob": 0.0},
    )
    assert condition.level == 14
    assert condition.style["melody_following"] == 0.0
    assert condition.radar["note_density"] == GENERATION_DEFAULTS["note_density"]

    with pytest.raises(ValueError, match="not both"):
        resolve_generation_condition(level=14, condition_file="condition.json",
                                     condition_data={"level": 14})


def test_cli_exports_using_current_condition_and_original_curve(tmp_path, monkeypatch, capsys):
    import json
    from fractions import Fraction as F
    import numpy as np
    from maichart.generation import event_cli
    from maichart.generation.event_audio import ACOUSTIC_DIM, EventAudio
    from maichart.generation.event_model import EventVocabulary, EventSequence
    from maichart.generation.music_intensity import audio_sha256

    checkpoint, condition, curve = _curve(tmp_path)
    audio_path = tmp_path / "song.wav"
    audio_path.write_bytes(b"audio identity")
    audio = EventAudio(np.zeros((101, ACOUSTIC_DIM), dtype=np.float32), np.linspace(0, 2, 101), 2.0, 120.0)
    vocab = EventVocabulary.from_event_charts([], segment_beats=4)
    cp = {"schema": "maichart-event-generator-checkpoint-v1", "epoch": 1,
          "data_contract": {"target_schema": "event_v1", "segment_beats": 4},
          "model_config": {"planner_profile": "music_structure_v3", "plan_dim": 13}}
    monkeypatch.setattr(event_cli, "load_event_model", lambda *a: (object(), vocab, cp))
    monkeypatch.setattr(event_cli, "extract_event_audio", lambda *a, **kw: audio)
    curve.update(schema="maichart-intensity-curve-v1", audio_sha256=audio_sha256(audio_path),
                 duration_sec=2, first=0, bpm_input=120,
                 points=[{"time_sec": 0, "intensity": .5}, {"time_sec": 2, "intensity": .3}],
                 rest_regions=[{"start_sec": 1, "end_sec": 1.5}])
    curve["condition"]["level"] = 12.5
    curve_path = tmp_path / "curve.json"
    curve_path.write_text(json.dumps(curve), encoding="utf-8")
    before_bytes = curve_path.read_bytes()
    def decode(*args, **kw):
        assert args[4].level == 14.5  # current generation target, never old 12.5
        assert kw["intensity_curve"]["points"] == curve["points"]
        assert kw["intensity_curve"]["condition"]["level"] == 12.5
        return [event_cli._DecodedWindow(EventSequence([F(0), F(1)], ["1", "2"]), F(0), F(4), True)]
    monkeypatch.setattr(event_cli, "_decode_windows", decode)
    output = tmp_path / "maidata.txt"
    code = event_cli.main(["--checkpoint", str(checkpoint), "--audio", str(audio_path),
        "--output", str(output), "--device", "cpu", "--bpm", "120", "--level", "14.5",
        "--no-audio-bounds-trim", "--intensity-curve", str(curve_path)])
    assert code == 0 and output.is_file()
    report = json.loads(output.with_suffix(".quality.json").read_text(encoding="utf-8"))["intensity_curve"]
    assert report["reused_with_changed_condition"] is True
    assert report["source_condition"]["level"] == 12.5
    assert report["generation_condition"]["level"] == 14.5
    assert report["rest_regions"] == curve["rest_regions"]
    assert curve_path.read_bytes() == before_bytes
    assert "intensity_curve_condition_warning" in capsys.readouterr().out
