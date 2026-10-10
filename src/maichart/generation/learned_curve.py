"""Generate an editable workload curve using the trained music planner."""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

from maichart.event_timing import seconds_at_beat
from maichart.generation.conditions import GenerationCondition, GENERATION_DEFAULTS
from maichart.generation.event_audio import (
    extract_event_audio, audio_segment_tensor, compress_event_audio,
    _normalize_tempo_map, _time_to_beat,
)
from maichart.generation.music_intensity import generate_intensity_profile, validate_intensity_profile
from maichart.training.event_checkpoints import load_event_model
from maichart.generation.music_plan import CONTINUOUS_COUNT, continuous_keys, is_music_structure
from maichart.generation.timeline import (
    beat_at_audio_end,
    normalize_tempo_map,
    serialize_tempo_map,
    tempo_map_fingerprint,
)


def generate_learned_profile(audio_path, checkpoint_path, *, resolution_sec=2.0,
                             local_context_sec=12.0, condition=None, bpm=None,
                             first=0.0, tempo_mode="constant", tempo_map=None, device="cpu"):
    """Predict authored local workload rather than treating energy as difficulty.

    Prediction resolution is the checkpoint's training window. Interpolation
    only places those predictions on the editor's time axis; it does not imply
    finer model supervision.
    """
    from maichart.generation.event_cli import (_file_identity, _tempo_map, _musical_bpm, _segment_windows,
                                              apply_contract_condition_guard)

    device = torch.device(device)
    identity_before = _file_identity(checkpoint_path)
    model, _, checkpoint = load_event_model(checkpoint_path, device)
    planner_profile = str(getattr(model.config, "planner_profile", "legacy_v1"))
    if not is_music_structure(planner_profile):
        raise ValueError("模型预测曲线需要 music_structure_v2/v3 规划器；请重新训练新模型，或选择音频参考曲线。")
    profile = generate_intensity_profile(audio_path, resolution_sec=resolution_sec,
                                         local_context_sec=local_context_sec)
    contract = checkpoint["data_contract"]
    audio = extract_event_audio(audio_path, cache_dir=Path(checkpoint_path).resolve().parent / "curve_audio_cache",
                                sample_rate=int(contract.get("sample_rate", 22050)),
                                hop_length=int(contract.get("hop_length", 441)))
    bpm_input = float(bpm) if bpm is not None else None
    bpm = bpm_input if bpm_input is not None else _musical_bpm(float(audio.tempo_bpm))
    if not np.isfinite(bpm) or bpm <= 0:
        raise ValueError("模型预测曲线需要有效 BPM")
    if not np.isfinite(float(first)):
        raise ValueError("first must be finite")
    explicit_tempo_map = serialize_tempo_map(tempo_map) if tempo_map is not None else None
    tempo = (normalize_tempo_map(explicit_tempo_map) if explicit_tempo_map is not None
             else _tempo_map(tempo_mode, audio, bpm, snap=bpm_input is None))
    bpm = float(tempo[0].bpm)
    condition = (GenerationCondition.from_dict(condition) if isinstance(condition, dict) else condition)
    if condition is None:
        condition = GenerationCondition(level=13.3, radar=GENERATION_DEFAULTS, style=GENERATION_DEFAULTS)
    condition, _ = apply_contract_condition_guard(condition, contract.get("condition_params") or {},
                                                   default_level=condition.level)
    global_audio = compress_event_audio(audio, song_duration=audio.duration_sec, bpm=bpm,
                                        first=first, tempo_changes=tempo,
                                        frames=model.config.global_audio_tokens).unsqueeze(0).to(device)
    layout = contract.get("family_params") or {}
    windows = _segment_windows(beat_at_audio_end(audio.duration_sec, tempo, first),
                               int(contract.get("segment_beats", 16)),
                               overlap_beats=int(layout.get("overlap_beats", 0)))
    centers, values, descriptors = [], [], []
    names = continuous_keys(planner_profile)
    continuous_count = CONTINUOUS_COUNT[planner_profile]
    with torch.inference_mode():
        for start, end in windows:
            low = max(0.0, float(seconds_at_beat(start, tempo, first)))
            high = min(float(audio.duration_sec), float(seconds_at_beat(end, tempo, first)))
            if high <= low:
                continue
            margin = Fraction(int(layout.get("audio_margin_beats", 0)))
            begin = max(Fraction(0), start - margin)
            finish = end + margin
            begin_sec = max(0.0, float(seconds_at_beat(begin, tempo, first)))
            finish_sec = min(float(audio.duration_sec), float(seconds_at_beat(finish, tempo, first)))
            start_beat = _time_to_beat(begin_sec - float(first), _normalize_tempo_map(tempo, bpm))
            x = audio_segment_tensor(audio, begin_sec, finish_sec,
                                      song_duration=audio.duration_sec, bpm=bpm, start_beat=start_beat,
                                      first=first, tempo_changes=tempo).unsqueeze(0).to(device)
            from maichart.generation.event_audio import FEATURE_RATE_HZ
            sample_times = begin_sec + torch.arange(x.size(1), device=device) / FEATURE_RATE_HZ
            core_mask = ((sample_times < low) | (sample_times >= high)).unsqueeze(0)
            memory, padding = model.encode_audio(x, condition.to_vector(device=device).unsqueeze(0),
                                                 global_audio=global_audio,
                                                 plan_audio_mask=core_mask,
                                                 condition_null=torch.tensor([False], device=device))
            _, _, logits = model.append_plan_memory(memory, padding)
            plan = logits[0, :continuous_count].sigmoid().cpu().tolist()
            configs = logits[0, continuous_count:continuous_count + 4].softmax(-1).cpu().tolist()
            centers.append((low + high) * 0.5)
            values.append(plan[names.index("intensity")])
            descriptors.append({"start_sec": low, "end_sec": high,
                                **dict(zip(names, plan)), "configuration_probabilities": configs})
    if not centers:
        raise ValueError("音频太短，无法预测音乐结构曲线")
    times = np.asarray([point["time_sec"] for point in profile["points"]])
    predicted = np.interp(times, centers, values)
    # Independent evidence still gates silence. A flat heuristic activity
    # trace is not used as an authored difficulty target.
    from maichart.generation.intensity import music_activity
    activity = music_activity(audio)
    silent = (np.interp(times, activity.times, activity.active().astype(float)) <= 0.05
              if activity is not None and len(activity.times) else np.zeros_like(times, dtype=bool))
    predicted[silent] = 0.0
    profile["points"] = [{"time_sec": float(time), "intensity": float(value)}
                          for time, value in zip(times, predicted)]
    profile.pop("components", None)
    profile["method"] = ("learned_music_structure_v3" if planner_profile == "music_structure_v3"
                          else "learned_music_structure_v2")
    profile["planner_windows"] = descriptors
    identity_after = _file_identity(checkpoint_path)
    if identity_before != identity_after:
        raise ValueError("曲线预测期间 checkpoint 文件发生变化，请选择固定 epoch checkpoint 后重新生成")
    profile["checkpoint"] = str(Path(checkpoint_path).resolve())
    profile["checkpoint_identity"] = {"size": identity_after["size"],
                                       "mtime_ns": str(identity_after["mtime_ns"]),
                                       "sha256": identity_after["sha256"]}
    profile["condition"] = condition.to_dict()
    profile["level"] = condition.level
    profile["bpm"] = bpm
    profile["bpm_input"] = bpm_input
    profile["first"] = float(first)
    profile["tempo_mode"] = tempo_mode
    profile["tempo_changes"] = serialize_tempo_map(tempo)
    profile["explicit_tempo_map"] = explicit_tempo_map
    profile["tempo_map_sha256"] = tempo_map_fingerprint(explicit_tempo_map)
    profile["limitations"] = [
        "0.5 is the chart's learned average workload, not a difficulty rating or emotion label.",
        "The trained window determines prediction resolution; editor interpolation adds no supervision.",
        "Training quality and musical intent still require fixed-song generation review.",
    ]
    return validate_intensity_profile(profile)
