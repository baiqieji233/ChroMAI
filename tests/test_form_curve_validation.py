import pytest

from maichart.form.curve_validation import validate_saved_curve
from maichart.generation.event_cli import _file_identity
from maichart.generation.music_intensity import audio_sha256
from maichart.generation.timeline import tempo_map_fingerprint


def _curve(audio, checkpoint):
    identity = _file_identity(checkpoint)
    return {
        "time_axis": "audio_absolute_v2",
        "audio_sha256": audio_sha256(audio),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_identity": {
            "size": identity["size"], "sha256": identity["sha256"],
        },
        "first": -0.5,
        "tempo_map_sha256": None,
        "explicit_tempo_map": None,
        "tempo_changes": [{"beat": "0", "bpm": 120.0}],
        "bpm_input": 120.0,
        "condition": {"level": 13.3, "radar": {"note_density": 0.5}, "style": {}},
    }


def _validate(profile, audio, checkpoint, condition=None, *, first=-0.5,
              tempo_map=None, bpm_input=120.0, resolved_tempo_map=None):
    return validate_saved_curve(
        profile, audio_path=audio, checkpoint_path=checkpoint,
        checkpoint_identity=_file_identity(checkpoint), first=first,
        tempo_map=tempo_map, bpm_input=bpm_input,
        resolved_tempo_map=(resolved_tempo_map if resolved_tempo_map is not None
                            else [{"beat": "0/1", "bpm": 120}]),
        condition=condition or profile["condition"],
    )


def test_saved_curve_allows_condition_changes_with_warning(tmp_path):
    audio = tmp_path / "song.wav"
    checkpoint = tmp_path / "form.pt"
    audio.write_bytes(b"audio")
    checkpoint.write_bytes(b"weights")
    profile = _curve(audio, checkpoint)

    warnings = _validate(profile, audio, checkpoint,
                         {"level": 14.0, "radar": {"note_density": 0.7}, "style": {}})

    assert warnings and "沿用原曲线" in warnings[0]


@pytest.mark.parametrize("field", ["audio_sha256", "checkpoint_identity", "first", "tempo_changes", "time_axis"])
def test_saved_curve_rejects_stale_or_missing_identity_metadata(tmp_path, field):
    audio = tmp_path / "song.wav"
    checkpoint = tmp_path / "form.pt"
    audio.write_bytes(b"audio")
    checkpoint.write_bytes(b"weights")
    profile = _curve(audio, checkpoint)
    profile.pop(field)

    with pytest.raises(ValueError, match="lacks identity metadata"):
        _validate(profile, audio, checkpoint)


def test_saved_curve_rejects_checkpoint_overwritten_at_same_path(tmp_path):
    audio = tmp_path / "song.wav"
    checkpoint = tmp_path / "form.pt"
    audio.write_bytes(b"audio")
    checkpoint.write_bytes(b"original weights")
    profile = _curve(audio, checkpoint)
    checkpoint.write_bytes(b"replacement weights")

    with pytest.raises(ValueError, match="checkpoint weights changed"):
        _validate(profile, audio, checkpoint)


def test_omitted_null_map_fingerprint_does_not_reject_the_standard_auto_bpm_flow(tmp_path):
    audio=tmp_path/'song.wav';audio.write_bytes(b'audio')
    checkpoint=tmp_path/'form.pt';checkpoint.write_bytes(b'weights')
    profile=_curve(audio,checkpoint)
    profile.update(bpm_input=None,explicit_tempo_map=None,tempo_changes=[{'beat':'0','bpm':184}])
    profile.pop('tempo_map_sha256')
    arguments=dict(audio_path=audio,checkpoint_path=checkpoint,
        checkpoint_identity=_file_identity(checkpoint),first=-.5,tempo_map=None,
        bpm_input=None,condition=profile['condition'],resolved_tempo_map=[{'beat':'0','bpm':184}])
    assert validate_saved_curve(profile,**arguments) == []
    arguments['resolved_tempo_map']=[{'beat':'0','bpm':185}]
    with pytest.raises(ValueError,match='another tempo map'):
        validate_saved_curve(profile,**arguments)


def test_saved_curve_checks_audio_first_and_timing_inputs(tmp_path):
    audio = tmp_path / "song.wav"
    checkpoint = tmp_path / "form.pt"
    audio.write_bytes(b"audio")
    checkpoint.write_bytes(b"weights")
    profile = _curve(audio, checkpoint)
    profile["audio_sha256"] = "different"
    with pytest.raises(ValueError, match="another audio file"):
        _validate(profile, audio, checkpoint)

    profile = _curve(audio, checkpoint)
    profile["first"] = 0
    with pytest.raises(ValueError, match="another first/audio offset"):
        _validate(profile, audio, checkpoint)

    profile = _curve(audio, checkpoint)
    profile["bpm_input"] = 128.0
    with pytest.raises(ValueError, match="another BPM input"):
        _validate(profile, audio, checkpoint)

    profile = _curve(audio, checkpoint)
    profile["tempo_map_sha256"] = tempo_map_fingerprint([{"beat": "0", "bpm": 120}])
    with pytest.raises(ValueError, match="another tempo map"):
        _validate(profile, audio, checkpoint)


def test_saved_curve_checks_resolved_tempo_when_bpm_was_automatic(tmp_path):
    audio = tmp_path / "song.wav"
    checkpoint = tmp_path / "form.pt"
    audio.write_bytes(b"audio")
    checkpoint.write_bytes(b"weights")
    profile = _curve(audio, checkpoint)
    profile["bpm_input"] = None

    with pytest.raises(ValueError, match="resolves to another tempo map"):
        validate_saved_curve(
            profile, audio_path=audio, checkpoint_path=checkpoint,
            checkpoint_identity=_file_identity(checkpoint), first=-0.5,
            tempo_map=None, bpm_input=None, resolved_tempo_map=[{"beat": "0", "bpm": 124}],
            condition=profile["condition"],
        )

    with pytest.raises(ValueError, match="another BPM input"):
        validate_saved_curve(
            profile, audio_path=audio, checkpoint_path=checkpoint,
            checkpoint_identity=_file_identity(checkpoint), first=-0.5,
            tempo_map=None, bpm_input=120.0, resolved_tempo_map=[{"beat": "0", "bpm": 120}],
            condition=profile["condition"],
        )


def test_saved_curve_uses_canonical_resolved_tempo_fingerprint(tmp_path):
    audio = tmp_path / "song.wav"
    checkpoint = tmp_path / "form.pt"
    audio.write_bytes(b"audio")
    checkpoint.write_bytes(b"weights")
    profile = _curve(audio, checkpoint)
    profile["tempo_changes"] = [{"beat": "0", "bpm": 92.54999999999788}]

    assert _validate(profile, audio, checkpoint,
                     resolved_tempo_map=[{"beat": "0/1", "bpm": 92.55}]) == []


def test_first_uses_absolute_tolerance(tmp_path):
    audio = tmp_path / "song.wav"
    checkpoint = tmp_path / "form.pt"
    audio.write_bytes(b"audio")
    checkpoint.write_bytes(b"weights")
    profile = _curve(audio, checkpoint)

    assert _validate(profile, audio, checkpoint, first=-0.5000000005) == []
    with pytest.raises(ValueError, match="another first/audio offset"):
        _validate(profile, audio, checkpoint, first=-0.500000002)


def test_explicit_tempo_hash_noise_uses_full_map_metadata(tmp_path):
    audio = tmp_path / "song.wav"
    checkpoint = tmp_path / "form.pt"
    audio.write_bytes(b"audio")
    checkpoint.write_bytes(b"weights")
    profile = _curve(audio, checkpoint)
    profile["bpm_input"] = None
    profile["explicit_tempo_map"] = [{"beat": "0", "bpm": 92.54999999999788}]
    profile["tempo_changes"] = profile["explicit_tempo_map"]
    profile["tempo_map_sha256"] = tempo_map_fingerprint(profile["explicit_tempo_map"])
    current = [{"beat": "0/1", "bpm": 92.55}]

    assert _validate(profile, audio, checkpoint, tempo_map=current, bpm_input=None,
                     resolved_tempo_map=current) == []

    profile.pop("explicit_tempo_map")
    with pytest.raises(ValueError, match="another tempo map"):
        _validate(profile, audio, checkpoint, tempo_map=current, bpm_input=None,
                  resolved_tempo_map=current)


def test_real_resolved_bpm_change_is_rejected(tmp_path):
    audio = tmp_path / "song.wav"
    checkpoint = tmp_path / "form.pt"
    audio.write_bytes(b"audio")
    checkpoint.write_bytes(b"weights")
    profile = _curve(audio, checkpoint)

    with pytest.raises(ValueError, match="resolves to another tempo map"):
        _validate(profile, audio, checkpoint,
                  resolved_tempo_map=[{"beat": "0", "bpm": 120.001}])
