import json
from pathlib import Path

import pytest

from maichart.training.dataset_catalog import compose_datasets


def _manifest(root: Path, *, profile: str, audio_metadata: dict | None = None,
              target_schema: str = "event_v1") -> Path:
    song = root / "song"
    song.mkdir(parents=True)
    maidata = song / "maidata.txt"
    audio = song / "track.wav"
    maidata.write_text("&title=Source\n&wholebpm=120\n", encoding="utf-8")
    audio.write_bytes(b"source audio")
    preprocessing = {"target_schema": target_schema, "feature_profile": profile}
    preprocessing.update(audio_metadata or {})
    manifest = root / "training_manifest.json"
    manifest.write_text(json.dumps({
        "schema": "maichart-training-manifest-v1",
        "source_root": str(root),
        "cache_dir": str(root / "cache"),
        "preprocessing": preprocessing,
        "songs": [{
            "song_id": root.name,
            "maidata_path": str(maidata),
            "audio_path": str(audio),
            "difficulties": [{"difficulty_index": 5, "event_usable_for_training": True}],
        }],
    }), encoding="utf-8")
    return manifest


def _read(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_song_form_v2_merges_legacy_and_rational_event_sources(tmp_path):
    old = _manifest(
        tmp_path / "old", profile="event_audio_v2_rhythm",
        audio_metadata={"sample_rate": 22050, "hop_length": 441, "feature_rate_hz": 50,
                        "input_dim": 112, "feature_config": {"kind": "fixed_frame"}},
    )
    rational = _manifest(tmp_path / "rational", profile="rational_audio_queries_v1")
    old_before = old.read_bytes()
    rational_before = rational.read_bytes()

    result = compose_datasets([old, rational], tmp_path / "composed", "song_form_v2")

    merged = _read(result["manifest"])
    preprocessing = merged["preprocessing"]
    assert preprocessing["target_schema"] == "event_v1"
    assert preprocessing["feature_profile"] == "rational_audio_queries_v1"
    for key in ("sample_rate", "hop_length", "feature_rate_hz", "input_dim", "feature_config"):
        assert key not in preprocessing
    assert len(merged["songs"]) == 2
    assert old.read_bytes() == old_before
    assert rational.read_bytes() == rational_before


def test_song_form_v2_still_requires_matching_event_target_schema(tmp_path):
    event = _manifest(tmp_path / "event", profile="event_audio_v2_rhythm")
    frame = _manifest(tmp_path / "frame", profile="rational_audio_queries_v1",
                      target_schema="frame_v2")

    with pytest.raises(ValueError, match="target_schema"):
        compose_datasets([event, frame], tmp_path / "composed", "song_form_v2")


def test_event_model_still_rejects_mixed_audio_profiles(tmp_path):
    old = _manifest(tmp_path / "old", profile="event_audio_v2_rhythm")
    rational = _manifest(tmp_path / "rational", profile="rational_audio_queries_v1")

    with pytest.raises(ValueError, match="feature_profile"):
        compose_datasets([old, rational], tmp_path / "composed", "event_v1")


def test_single_old_manifest_remains_usable_for_song_form_v2(tmp_path):
    old = _manifest(
        tmp_path / "old", profile="event_audio_v2_rhythm",
        audio_metadata={"sample_rate": 22050, "hop_length": 441, "feature_rate_hz": 50},
    )
    before = old.read_bytes()

    result = compose_datasets([old], tmp_path / "composed", "song_form_v2")

    assert result["manifest"] == str(old.resolve())
    assert old.read_bytes() == before
