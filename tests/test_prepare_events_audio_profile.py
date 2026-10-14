from pathlib import Path

import pytest

from maichart.training.prepare_events import (
    LEGACY_AUDIO_PROFILE,
    RATIONAL_AUDIO_PROFILE,
    build_event_manifest,
)


def _raw_song(root: Path) -> None:
    song = root / "song"
    song.mkdir(parents=True)
    (song / "maidata.txt").write_text(
        "&title=Test\n&wholebpm=120\n&lv_4=13\n&inote_4=(120){4}1,2,3,4,\n",
        encoding="utf-8",
    )


def test_legacy_audio_profile_keeps_fixed_frame_contract(tmp_path):
    raw = tmp_path / "raw"
    _raw_song(raw)

    preprocessing = build_event_manifest(raw, cache_dir=tmp_path / "cache")["preprocessing"]

    assert preprocessing["feature_profile"] == LEGACY_AUDIO_PROFILE
    assert preprocessing["sample_rate"] == 22050
    assert preprocessing["hop_length"] == 441
    assert preprocessing["feature_rate_hz"] == 50


def test_rational_profile_does_not_claim_fixed_frame_audio_contract(tmp_path):
    raw = tmp_path / "raw"
    _raw_song(raw)

    manifest = build_event_manifest(
        raw, cache_dir=tmp_path / "cache", audio_profile=RATIONAL_AUDIO_PROFILE,
    )
    preprocessing = manifest["preprocessing"]

    assert preprocessing["feature_profile"] == RATIONAL_AUDIO_PROFILE
    assert "hop_length" not in preprocessing
    assert "feature_rate_hz" not in preprocessing
    assert "input_dim" not in preprocessing
    assert "feature_config" not in preprocessing
    assert manifest["cache_dir"] == (tmp_path / "cache").as_posix()


def test_unknown_audio_profile_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unsupported audio profile"):
        build_event_manifest(
            tmp_path / "missing", cache_dir=tmp_path / "cache", audio_profile="unknown",
        )
