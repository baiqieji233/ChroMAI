"""Load-curve construction and song-form intensity profiles."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from maichart.event_timing import seconds_at_beat
from maichart.form.clock import AUDIO_CURVE_AXIS


LOAD_CURVE_CHANNELS = ("event_density", "tap", "sustain", "slide_touch")
LOAD_CURVE_BIN_SECONDS = 2.0


def load_curve_bin_centers(duration_sec, bin_seconds=LOAD_CURVE_BIN_SECONDS):
    duration, width = float(duration_sec), float(bin_seconds)
    if not math.isfinite(duration) or duration < 0:
        raise ValueError('curve duration must be finite and nonnegative')
    if not math.isfinite(width) or width <= 0:
        raise ValueError('curve resolution must be finite and positive')
    starts = np.arange(max(1, int(math.ceil(duration / width))), dtype=np.float64) * width
    return (starts + np.minimum(duration, starts + width)) * .5


def build_load_curve(points, tempo, first, duration_sec, *, bin_seconds=LOAD_CURVE_BIN_SECONDS):
    """Build legacy chart-relative workload bins for compatible model inputs.

    Bin coordinates are seconds from chart beat zero, not raw audio seconds.
    The training frame clock converts with audio_second - first. Public editor
    profiles use a separately versioned absolute-audio time axis.
    """
    duration = float(duration_sec)
    width = float(bin_seconds)
    first = float(first)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration_sec must be finite and positive")
    if not math.isfinite(first):
        raise ValueError("first must be finite")
    if not math.isfinite(width) or width <= 0:
        raise ValueError("bin_seconds must be finite and positive")

    from maichart.generation.event_quality import parse_event_notes

    parsed = parse_event_notes(points, tempo, first=first)
    if not parsed.syntax_valid:
        raise ValueError("cannot build load curve from invalid chart events: " +
                         "; ".join(issue["message"] for issue in parsed.fatal_errors[:3]))
    # The model clock is chart-relative, while duration_sec bounds the source
    # audio clock. Audio end therefore lands at duration - first.
    chart_duration = max(0.0, duration - first)
    count = max(1, int(math.ceil(chart_duration / width)))
    times = np.empty(count, dtype=np.float32)
    values = np.zeros((count, len(LOAD_CURVE_CHANNELS)), dtype=np.float32)
    for index in range(count):
        low = index * width
        high = min(chart_duration, low + width)
        times[index] = (low + high) * 0.5

    # Convert each exact event time once instead of rescanning the chart per bin.
    bin_starts = np.arange(count) * width
    totals = np.zeros(count, dtype=np.int64)
    taps = np.zeros(count, dtype=np.int64)
    sustains = np.zeros(count, dtype=np.int64)
    slide_touches = np.zeros(count, dtype=np.int64)
    for point in parsed.points:
        audio_second = float(seconds_at_beat(point.beat, tempo, first))
        if audio_second < 0.0 or audio_second >= duration:
            continue
        second = audio_second - first
        if second < 0.0 or second >= chart_duration:
            continue
        index = int(np.searchsorted(bin_starts, second, side="right") - 1)
        for note in point.notes:
            totals[index] += 1
            kind = note.note_type
            if kind == "tap":
                taps[index] += 1
            if kind in {"hold", "slide", "touch_hold"}:
                sustains[index] += 1
            if kind in {"slide", "touch", "touch_hold"}:
                slide_touches[index] += 1
    active = totals > 0
    if np.any(active):
        bin_widths = np.minimum(chart_duration, (np.arange(count) + 1) * width) - np.arange(count) * width
        values[active, 0] = np.minimum(totals[active] / bin_widths[active] / 8.0, 1.0)
        values[active, 1] = np.minimum(taps[active] / totals[active], 1.0)
        values[active, 2] = np.minimum(sustains[active] / totals[active], 1.0)
        values[active, 3] = np.minimum(slide_touches[active] / totals[active], 1.0)
    return times, values


def resample_load_curve(times_sec, values, frame_times_sec):
    """Interpolate bin-center values onto an audio-relative frame clock."""
    times = np.asarray(times_sec, dtype=np.float64).reshape(-1)
    curve = np.asarray(values, dtype=np.float64)
    frames = np.asarray(frame_times_sec, dtype=np.float64).reshape(-1)
    if curve.ndim != 2 or curve.shape != (len(times), len(LOAD_CURVE_CHANNELS)):
        raise ValueError("values must have shape [len(times_sec), 4]")
    if not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("times_sec must be finite, nonempty, and strictly increasing")
    if not np.isfinite(curve).all() or np.any(curve < 0) or np.any(curve > 1):
        raise ValueError("load curve values must be finite and in 0..1")
    if not np.isfinite(frames).all():
        raise ValueError("frame_times_sec must be finite")
    return np.stack([np.interp(frames, times, curve[:, channel])
                     for channel in range(curve.shape[1])], axis=-1).astype(np.float32)


def _curve_capability(payload):
    capability = (payload.get("capabilities") or {}).get("load_curve") or {}
    from maichart.form.checkpoint import INPUT_CONTRACT
    expected = INPUT_CONTRACT.get("load_curve", "song_form_load_curve_v1")
    if int(capability.get("steps", 0)) <= 0 or capability.get("contract") != expected:
        raise ValueError("song_form checkpoint has no trained load-curve capability; train a compatible checkpoint first")


def generate_learned_profile(audio_path, checkpoint_path, *, resolution_sec=LOAD_CURVE_BIN_SECONDS,
                             local_context_sec=12.0, condition=None, bpm=None, first=0.0,
                             tempo_mode="constant", tempo_map=None, meter_map=None, device="cpu"):
    """Predict a chart workload curve from a trained song_form_v1 checkpoint."""
    import torch

    from maichart.form.checkpoint import load as load_form_checkpoint
    from maichart.form.data import form_audio, organize_song
    from maichart.form.codec import RationalCodec
    from maichart.generation.conditions import GenerationCondition, GENERATION_DEFAULTS
    from maichart.generation.event_audio import (
        extract_event_audio, _normalize_tempo_map, segment_frame_times,
    )
    from maichart.generation.event_cli import _file_identity, _musical_bpm, _tempo_map
    from maichart.generation.music_intensity import audio_sha256, validate_intensity_profile
    from maichart.generation.timeline import (
        beat_at_audio_end, normalize_tempo_map, serialize_tempo_map, tempo_map_fingerprint,
    )

    resolution = float(resolution_sec)
    if not math.isfinite(resolution) or resolution <= 0:
        raise ValueError("resolution_sec must be finite and positive")
    device = torch.device(device)
    identity_before = _file_identity(checkpoint_path)
    model, planner, _, payload = load_form_checkpoint(checkpoint_path, device=device, inference=False)
    _curve_capability(payload)
    if int(payload.get("optimizer_steps", 0)) <= 0 or int(payload.get("blueprint_steps", 0)) <= 0:
        raise ValueError("song_form draft and whole-song blueprint must be trained before curve inference")
    audio = extract_event_audio(audio_path, cache_dir=Path(checkpoint_path).resolve().parent / "curve_audio_cache")
    bpm_input = float(bpm) if bpm is not None else None
    bpm = bpm_input if bpm_input is not None else _musical_bpm(float(audio.tempo_bpm))
    if not math.isfinite(bpm) or bpm <= 0:
        raise ValueError("model curve prediction needs a finite positive BPM")
    if not math.isfinite(float(first)):
        raise ValueError("first must be finite")
    explicit_tempo_map = serialize_tempo_map(tempo_map) if tempo_map is not None else None
    tempo = (normalize_tempo_map(explicit_tempo_map) if explicit_tempo_map is not None
             else _tempo_map(tempo_mode, audio, bpm, snap=bpm_input is None))
    bpm = float(tempo[0].bpm)
    condition = GenerationCondition.from_dict(condition) if isinstance(condition, dict) else condition
    condition = condition or GenerationCondition(level=13.3, radar=GENERATION_DEFAULTS, style=GENERATION_DEFAULTS)
    song = organize_song(audio, tempo, first, meters=meter_map)
    if not song.spans:
        raise ValueError("Audio is too short to predict a song-form workload curve")
    from maichart.form.generate import to_device
    from maichart.form.data import input_example

    example = input_example(song, 0, [], condition=condition,
                            codec=RationalCodec.from_dict(payload["codec"]))
    planner.eval()
    audio_values = form_audio(song, song.spans[0].start, song.spans[-1].end).unsqueeze(0).to(device)
    form_values = torch.as_tensor(example["form"], dtype=torch.float32, device=device).unsqueeze(0)
    condition_values = condition.to_vector(device=device).unsqueeze(0)
    with torch.inference_mode():
        prediction = planner(audio_values, form_values, condition_values)
    if "load_curve" not in prediction:
        raise ValueError("song_form blueprint has no load-curve output head")
    frame_start = float(seconds_at_beat(song.spans[0].start, tempo, first))
    frame_end = float(seconds_at_beat(song.spans[-1].end, tempo, first))
    frame_times = segment_frame_times(frame_start, frame_end)
    predicted = prediction["load_curve"][0].sigmoid().cpu().numpy()
    if len(predicted) != len(frame_times):
        raise ValueError("blueprint load-curve clock does not align with song audio frames")
    # Keep editor timestamps in float64; a rounded float32 EOF can land past
    # the actual audio duration and produce an extra invalid terminal point.
    points_times = load_curve_bin_centers(audio.duration_sec, resolution)
    profile_values = resample_load_curve(frame_times, predicted, points_times)
    # Inaccessible audio before a positive chart start and virtual pre-audio
    # silence must not inherit an extrapolated nonzero model prediction.
    valid_bins = (points_times >= max(0.,float(first))) & (points_times < float(audio.duration_sec))
    profile_values[~valid_bins] = 0
    density = profile_values[:, 0]
    positive = density[density > 1e-6]
    reference = float(np.mean(positive)) if len(positive) else 1.0
    intensity = np.clip(0.5 * density / max(reference, 1e-6), 0.0, 1.0)
    editor_times = np.unique(np.concatenate(([0.0], points_times, [float(audio.duration_sec)])))
    editor_values = np.interp(editor_times, points_times, intensity).astype(np.float32)
    component_values = resample_load_curve(points_times, profile_values, editor_times)
    editor_values[editor_times < max(0.,float(first))] = 0
    component_values[editor_times < max(0.,float(first))] = 0
    rest_regions = []
    rest_start = None
    for i, value in enumerate(density):
        low, high = i * resolution, min(float(audio.duration_sec), (i + 1) * resolution)
        if value <= 1e-6 and rest_start is None:
            rest_start = low
        if value > 1e-6 and rest_start is not None:
            rest_regions.append({"start_sec": rest_start, "end_sec": low})
            rest_start = None
    if rest_start is not None:
        rest_regions.append({"start_sec": rest_start, "end_sec": float(audio.duration_sec)})
    identity_after = _file_identity(checkpoint_path)
    if identity_before != identity_after:
        raise ValueError("Checkpoint changed during curve inference; retry with a fixed checkpoint")
    profile = {
        "schema": "maichart-intensity-curve-v1", "method": "learned_song_form_v1",
        "time_axis": AUDIO_CURVE_AXIS,
        "duration_sec": float(audio.duration_sec),
        "points": [{"time_sec": float(t), "intensity": float(v)}
                   for t, v in zip(editor_times, editor_values)],
        "components": {name: component_values[:, index].tolist()
                       for index, name in enumerate(LOAD_CURVE_CHANNELS)},
        "rest_regions": rest_regions,
        "audio_sha256": audio_sha256(audio_path),
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_identity": {"size": identity_after["size"],
                                "mtime_ns": str(identity_after["mtime_ns"]),
                                "sha256": identity_after["sha256"]},
        "condition": condition.to_dict(), "level": condition.level,
        "bpm": bpm, "bpm_input": bpm_input, "first": float(first),
        "tempo_mode": tempo_mode, "tempo_changes": serialize_tempo_map(tempo),
        "explicit_tempo_map": explicit_tempo_map,
        "tempo_map_sha256": tempo_map_fingerprint(explicit_tempo_map),
        "meter_map": [meter.to_dict() for meter in song.meters],
        "curve_resolution_sec": resolution, "action_state_context": "none_before_chart_generation",
        "limitations": [
            "The profile is a trained chart-workload prediction, not a musical quality guarantee.",
            "Its training bins are two seconds; editor interpolation adds no training resolution.",
        ],
    }
    return validate_intensity_profile(profile)

