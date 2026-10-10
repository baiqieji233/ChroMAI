"""The condition profile is the bridge between training controls and generation.

Training conditions on 16 controls, 15 of which are computed *from the chart
being predicted*.  Generation has no chart yet, so it must supply values the
model actually saw.  These tests pin the measurement, the default vector, and the
mismatch that made the old 0.5 default wrong.
"""

import json

import pytest

from maichart.generation.conditions import CONDITION_KEYS, RADAR_KEYS, STYLE_KEYS, GenerationCondition
from maichart.training.condition_profile import (
    PROFILE_NAME,
    PROFILE_SCHEMA,
    condition_from_profile,
    default_profile_path,
    load_profile,
    measure_condition_profile,
    profile_default_vector,
)


def _write_chart(root, name, *, notes, designer="Tester"):
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "maidata.txt").write_text(
        f"&title={name}\n&artist=Tester\n&wholebpm=134\n&first=0\n&des_4={designer}\n"
        f"&lv_4=13\n&inote_4=(134){{4}}{notes}\n",
        encoding="utf-8",
    )
    (directory / "track.mp3").write_bytes(b"\x00" * 32)
    return directory


def _manifest(tmp_path, charts):
    from maichart.training import prepare_events

    raw = tmp_path / "raw"
    for name, notes in charts.items():
        _write_chart(raw, name, notes=notes)
    output = tmp_path / "training_manifest.json"
    assert prepare_events.main([str(raw), "--cache-dir", str(tmp_path / "cache"), "-o", str(output)]) == 0
    return output


def test_profile_measures_the_corpus_and_differs_from_the_old_default(tmp_path):
    # Two charts: one plain single notes, one with chords and a slide mixed in.
    # Both need >= 50 notes to count as usable training difficulties.  The slide
    # uses the ``start>end[duration]`` form the parser accepts.
    plain = "1,2,3,4," * 16
    rich = "1/2,3/4,5,6,1>4[8:1],2,3,4," * 8
    manifest = _manifest(tmp_path, {"plain": plain, "rich": rich})

    profile = measure_condition_profile(manifest)
    assert profile["schema"] == PROFILE_SCHEMA
    assert profile["charts"] == 2
    assert profile["skipped"] == 0
    assert set(profile["controls"]) == set(CONDITION_KEYS) - {"level"}
    assert profile["control_measurements"]["melody_following"]["measured"] is False
    assert profile["control_measurements"]["melody_following"]["source"] == "audio_onset_alignment_proxy"
    assert "no audio" in profile["control_measurements"]["melody_following"]["reason"].lower()
    for key in RADAR_KEYS + STYLE_KEYS:
        entry = profile["controls"][key]
        assert entry["p10"] <= entry["median"] <= entry["p90"]

    # The whole point: the old UI default of 0.5 is not what the corpus contains.
    chord = profile["controls"]["chord_bias"]["median"]
    assert chord < 0.5, "the mixed corpus must sit below the 0.5 default"

    vector = profile_default_vector(profile)
    assert len(vector) == len(CONDITION_KEYS)
    condition = condition_from_profile(profile, level=13.5)
    assert condition.level == 13.5
    assert condition.radar["chord"] == pytest.approx(profile["controls"]["chord"]["median"])
    assert condition.style["melody_following"] == pytest.approx(0.9)
    assert set(condition.style) == set(STYLE_KEYS)
    # The old behaviour, for contrast.
    legacy = GenerationCondition(level=13.5)
    assert legacy.radar["chord"] == 0.5 and condition.radar["chord"] != 0.5


def test_profile_uses_the_median_level_when_none_is_given(tmp_path):
    manifest = _manifest(tmp_path, {"a": "1,2,3,4," * 16})
    profile = measure_condition_profile(manifest)
    condition = condition_from_profile(profile)
    assert condition.level == pytest.approx(profile["level"]["median"])


def test_profile_file_round_trip_and_validation(tmp_path):
    manifest = _manifest(tmp_path, {"a": "1,2,3,4," * 16})
    profile = measure_condition_profile(manifest)
    target = default_profile_path(manifest)
    assert target.name == PROFILE_NAME
    target.write_text(json.dumps(profile, ensure_ascii=False), encoding="utf-8")
    loaded = load_profile(target)
    assert loaded["charts"] == profile["charts"]
    assert loaded["control_measurements"] == profile["control_measurements"]

    broken = tmp_path / "broken.json"
    broken.write_text('{"schema": "something-else"}', encoding="utf-8")
    with pytest.raises(ValueError, match="条件基线档案"):
        load_profile(broken)


def test_profile_reports_when_nothing_is_measurable(tmp_path):
    empty = tmp_path / "training_manifest.json"
    empty.write_text(json.dumps({"schema": "maichart-training-manifest-v1", "songs": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="没有可测量的谱面"):
        measure_condition_profile(empty)


def test_legacy_zero_melody_placeholder_falls_back_but_explicit_measured_zero_survives():
    legacy = {"controls": {
        "melody_following": {"median": 0.0},
        "note_density": {"median": 0.41},
    }}
    restored = condition_from_profile(legacy, level=13.0)
    assert restored.style["melody_following"] == pytest.approx(0.9)
    assert restored.radar["note_density"] == pytest.approx(0.41)

    measured_zero = {"controls": {"melody_following": {"median": 0.0}},
                     "control_measurements": {"melody_following": {"measured": True}}}
    measured = condition_from_profile(measured_zero, level=13.0)
    assert measured.style["melody_following"] == 0.0


def test_absent_profile_controls_use_generation_defaults_not_half():
    condition = condition_from_profile({"controls": {}}, level=13.0)
    assert condition.radar["note_density"] == pytest.approx(0.305)
    assert condition.style["melody_following"] == pytest.approx(0.9)
