"""Identity checks for saved song-form workload curves."""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path

from maichart.generation.music_intensity import audio_sha256
from maichart.generation.timeline import normalize_tempo_map, tempo_map_fingerprint


_FIRST_TOLERANCE = 1e-9
_BPM_TOLERANCE = 1e-6


def _tempo_maps_match(left, right) -> bool:
    """Compare exact beat positions and absolute BPM error for saved clocks."""
    try:
        left_map = normalize_tempo_map(left)
        right_map = normalize_tempo_map(right)
    except (TypeError, ValueError):
        return False
    if len(left_map) != len(right_map):
        return False
    return all(
        Fraction(a.beat) == Fraction(b.beat)
        and abs(float(a.bpm) - float(b.bpm)) <= _BPM_TOLERANCE
        for a, b in zip(left_map, right_map)
    )


def validate_saved_curve(profile: dict, *, audio_path, checkpoint_path,
                         checkpoint_identity: dict, first: float,
                         tempo_map, bpm_input: float | None,
                         condition: dict, resolved_tempo_map=None,
                         check_tempo_inputs: bool = True) -> list[str]:
    """Validate immutable curve provenance and report reusable condition changes."""
    from maichart.form.clock import curve_axis

    curve_axis(profile)
    profile = dict(profile)
    # A null optional-map fingerprint was omitted by the editor transport.
    # Rebuild only this optional value; the resolved clock remains required.
    if 'tempo_map_sha256' not in profile:
        profile['tempo_map_sha256'] = tempo_map_fingerprint(profile.get('explicit_tempo_map'))
    required = {
        "audio_sha256", "checkpoint", "checkpoint_identity", "first",
        "tempo_map_sha256", "tempo_changes", "bpm_input", "condition", "time_axis",
    }
    missing = sorted(required - profile.keys())
    if missing:
        raise ValueError("Saved song-form curve lacks identity metadata (" +
                         ", ".join(missing) + "); regenerate the curve")

    if profile["audio_sha256"] != audio_sha256(audio_path):
        raise ValueError("Saved song-form curve belongs to another audio file; regenerate the curve")
    if Path(str(profile["checkpoint"])).resolve() != Path(checkpoint_path).resolve():
        raise ValueError("Saved song-form curve was created with another checkpoint; regenerate the curve")
    saved_identity = profile["checkpoint_identity"]
    if not isinstance(saved_identity, dict) or not isinstance(checkpoint_identity, dict):
        raise ValueError("Saved song-form curve has invalid checkpoint identity; regenerate the curve")
    saved_hash = saved_identity.get("sha256")
    loaded_hash = checkpoint_identity.get("sha256")
    if not isinstance(saved_hash, str) or not isinstance(loaded_hash, str) or saved_hash != loaded_hash:
        raise ValueError("Saved song-form curve checkpoint weights changed; regenerate the curve")
    if saved_identity.get("size") != checkpoint_identity.get("size"):
        raise ValueError("Saved song-form curve checkpoint identity changed; regenerate the curve")

    try:
        same_first = abs(float(profile["first"]) - float(first)) <= _FIRST_TOLERANCE
    except (TypeError, ValueError):
        same_first = False
    if not same_first:
        raise ValueError("Saved song-form curve uses another first/audio offset; regenerate the curve")
    if check_tempo_inputs:
        current_tempo_hash = tempo_map_fingerprint(tempo_map)
        if profile["tempo_map_sha256"] != current_tempo_hash:
            saved_explicit_tempo = profile.get("explicit_tempo_map")
            if (saved_explicit_tempo is None or tempo_map is None
                    or not _tempo_maps_match(saved_explicit_tempo, tempo_map)):
                raise ValueError("Saved song-form curve uses another tempo map; regenerate the curve")
        if profile["tempo_map_sha256"] is None and profile["bpm_input"] != bpm_input:
            raise ValueError("Saved song-form curve uses another BPM input; regenerate the curve")
    if resolved_tempo_map is not None:
        try:
            saved_resolved_tempo = normalize_tempo_map(profile["tempo_changes"])
        except (TypeError, ValueError) as exc:
            raise ValueError("Saved song-form curve has invalid resolved tempo changes; regenerate the curve") from exc
        if not _tempo_maps_match(saved_resolved_tempo, resolved_tempo_map):
            raise ValueError("Saved song-form curve resolves to another tempo map; regenerate the curve")
    if not isinstance(profile["condition"], dict):
        raise ValueError("Saved song-form curve has invalid condition metadata; regenerate the curve")
    if profile["condition"] != condition:
        return ["等级、Radar 或 Style 与曲线预测时不同，将沿用原曲线并按当前生成参数写谱；可按需重新生成曲线。"]
    return []
