import copy

import pytest

from maichart.generation.conditions import GENERATION_DEFAULTS, GenerationCondition
from maichart.generation.event_cli import validate_learned_curve_contract
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
    ("condition", GenerationCondition(level=14.0, radar=GENERATION_DEFAULTS,
                                       style=GENERATION_DEFAULTS).to_dict()),
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
