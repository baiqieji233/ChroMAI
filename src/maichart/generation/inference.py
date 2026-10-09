"""Inference helpers for audio-conditioned chart generation."""

from __future__ import annotations

import bisect
import math
from fractions import Fraction
from pathlib import Path
from typing import Any

import torch

from maichart.audio import AudioFeatureSet, load_audio_features_json
from maichart.generation.conditions import GenerationCondition
from maichart.generation.model import AUDIO_FEATURE_KEYS, ConditionedChartTransformer
from maichart.generation.simai_compile import PredictedNote, TempoChange
from maichart.training.checkpoints import load_checkpoint


def load_model_checkpoint(path: str | Path, device: torch.device) -> tuple[ConditionedChartTransformer, dict[str, Any]]:
    checkpoint = load_checkpoint(path, map_location=device)
    config_data = checkpoint.get("model_config")
    if not isinstance(config_data, dict):
        raise ValueError("Checkpoint 缺少 model_config；旧版纯权重文件需要先用训练时相同的模型参数加载。")
    model = ConditionedChartTransformer(**config_data).to(device)
    model.note_type_vocab = tuple(checkpoint.get("note_type_vocab", ("none", "tap", "break", "hold", "slide", "touch", "touch_hold")))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def audio_to_frame_tensor(
    features: AudioFeatureSet,
    *,
    bpm: float,
    division: int,
    first: float,
    tempo_changes: list[TempoChange] | None = None,
) -> tuple[torch.Tensor, int]:
    if bpm <= 0 or division <= 0:
        raise ValueError("bpm and division must be positive.")
    changes = _complete_tempo_changes(tempo_changes, bpm)
    frame_seconds = 60.0 / bpm * 4.0 / division
    upper_bound = max(
        1,
        math.ceil(
            max(0.0, features.duration_sec - first)
            * max(change.bpm for change in changes)
            / 60.0
            * division
            / 4.0
        )
        + 3,
    )
    frame_count = 1
    for index in range(upper_bound):
        chart_beat = Fraction(index * 4, division)
        sample_time = _seconds_at_beat(chart_beat, changes, first)
        if sample_time <= features.duration_sec + max(frame_seconds, 0.01):
            frame_count = index + 1
    audio_frames = sorted(features.feature_frames, key=lambda frame: frame.time_sec)
    times = [float(frame.time_sec) for frame in audio_frames]
    vectors: list[list[float]] = []
    for index in range(frame_count):
        chart_beat = Fraction(index * 4, division)
        sample_time = max(0.0, _seconds_at_beat(chart_beat, changes, first))
        frame = audio_frames[_nearest_index(times, sample_time)] if audio_frames else None
        vectors.append([_feature_value(frame, key) if frame is not None else 0.0 for key in AUDIO_FEATURE_KEYS])
    return torch.tensor(vectors, dtype=torch.float32), frame_count


def build_tempo_changes_from_audio(
    features: AudioFeatureSet,
    *,
    bpm: float,
    first: float,
    division: int,
) -> list[TempoChange]:
    """Convert detected audio-time tempo segments to chart-beat events.

    Audio analysis has no knowledge of the chart's ``&first`` value. The
    detected ``time_sec`` boundaries are therefore integrated from that anchor
    into chart beats, then quantized to the selected Simai frame grid. This
    keeps note extraction and emitted ``(BPM)`` directives on the same clock.
    """

    if bpm <= 0 or division <= 0:
        raise ValueError("bpm and division must be positive.")
    detected = sorted(features.tempo_changes, key=lambda item: item.time_sec)
    if not detected:
        return [TempoChange(Fraction(0), bpm)]

    current_bpm = _tempo_at_audio_time(detected, first, bpm)
    changes = [TempoChange(Fraction(0), current_bpm)]
    previous_time = float(first)
    chart_beat = Fraction(0)
    for detected_change in detected:
        boundary_time = float(detected_change.time_sec)
        if boundary_time <= first + 1e-6:
            continue
        chart_beat += Fraction(str(max(0.0, boundary_time - previous_time))) * Fraction(str(current_bpm)) / 60
        quantized_units = round(float(chart_beat) * division / 4.0)
        quantized_beat = Fraction(quantized_units * 4, division)
        is_new_tempo = (
            detected_change.bpm > 0
            and abs(float(detected_change.bpm) - current_bpm) / max(current_bpm, 1e-9) >= 0.01
        )
        if is_new_tempo and quantized_beat > changes[-1].beat:
            changes.append(TempoChange(quantized_beat, float(detected_change.bpm)))
        if detected_change.bpm > 0:
            current_bpm = float(detected_change.bpm)
        previous_time = boundary_time
    return changes


def predict_frame_notes(
    model: ConditionedChartTransformer,
    features: torch.Tensor,
    condition: GenerationCondition,
    *,
    device: torch.device,
    presence_threshold: float = 0.45,
    button_threshold: float = 0.45,
    sample: bool = False,
    guidance_scale: float = 1.0,
) -> dict[int, list[PredictedNote]]:
    vocab = getattr(model, "note_type_vocab", ("none", "tap", "break", "hold", "slide", "touch", "touch_hold"))
    result: dict[int, list[PredictedNote]] = {}
    max_buttons = max(1, min(8, 1 + round(condition.radar["peak"] * 7)))
    window = max(1, int(model.config.max_len))
    condition_vector = condition.to_vector(device=device).unsqueeze(0)
    for start in range(0, features.size(0), window):
        end = min(features.size(0), start + window)
        x = features[start:end].unsqueeze(0).to(device)
        with torch.inference_mode():
            outputs = model(x, condition_vector)
            if abs(float(guidance_scale) - 1.0) > 1e-6:
                unconditional = model(x, torch.zeros_like(condition_vector))
                outputs = {
                    key: unconditional[key] + float(guidance_scale) * (value - unconditional[key])
                    for key, value in outputs.items()
                }
        _decode_window(
            result,
            outputs,
            start=start,
            vocab=vocab,
            max_buttons=max_buttons,
            presence_threshold=presence_threshold,
            button_threshold=button_threshold,
            sample=sample,
        )
    return result


def _decode_window(
    result: dict[int, list[PredictedNote]],
    outputs: dict[str, torch.Tensor],
    *,
    start: int,
    vocab: tuple[str, ...],
    max_buttons: int,
    presence_threshold: float,
    button_threshold: float,
    sample: bool,
) -> None:
    presence = torch.sigmoid(outputs["note_presence_logits"])[0, :, 0].cpu()
    buttons = torch.sigmoid(outputs["button_logits"])[0].cpu()
    types = torch.softmax(outputs["type_logits"], dim=-1)[0].cpu()
    for local_index in range(presence.size(0)):
        present = bool(torch.rand(()) < presence[local_index]) if sample else bool(presence[local_index] >= presence_threshold)
        type_index = int(torch.multinomial(types[local_index], 1)) if sample else int(types[local_index].argmax())
        note_type = vocab[type_index] if type_index < len(vocab) else "tap"
        if not present or note_type == "none":
            continue
        button_probs = buttons[local_index]
        if sample:
            selected = [index for index, probability in enumerate(button_probs) if torch.rand(()) < probability]
        else:
            selected = [index for index, probability in enumerate(button_probs) if probability >= button_threshold]
        if not selected:
            selected = [int(button_probs.argmax())]
        selected = sorted(selected, key=lambda index: float(button_probs[index]), reverse=True)[:max_buttons]
        is_break = note_type == "break"
        if note_type == "break":
            note_type = "tap"
        result[start + local_index] = [
            PredictedNote(note_type=note_type, position=button_index + 1, break_note=is_break)
            for button_index in selected
        ]


def load_or_analyze_audio(audio_path: str | Path | None, features_path: str | Path | None, *, division: int, sample_rate: int) -> AudioFeatureSet:
    if features_path:
        return load_audio_features_json(features_path)
    if not audio_path:
        raise ValueError("Provide --audio or --audio-features.")
    from maichart.audio import analyze_audio_file

    return analyze_audio_file(audio_path, division=division, sample_rate=sample_rate)


def _complete_tempo_changes(tempo_changes: list[TempoChange] | None, bpm: float) -> list[TempoChange]:
    changes = sorted(tempo_changes or [], key=lambda item: item.beat)
    if not changes or changes[0].beat > 0:
        changes.insert(0, TempoChange(Fraction(0), bpm))
    elif changes[0].beat < 0:
        changes = [TempoChange(Fraction(0), changes[0].bpm)] + [item for item in changes if item.beat > 0]
    return changes


def _tempo_at_audio_time(detected: list[Any], time_sec: float, fallback: float) -> float:
    value = float(fallback)
    for change in detected:
        if float(change.time_sec) <= time_sec + 1e-6 and float(change.bpm) > 0:
            value = float(change.bpm)
        elif float(change.time_sec) > time_sec:
            break
    return value


def _seconds_at_beat(beat: Fraction, changes: list[TempoChange], first: float) -> float:
    elapsed = float(first)
    previous = Fraction(0)
    bpm = float(changes[0].bpm)
    for change in changes[1:]:
        if change.beat >= beat:
            break
        elapsed += float(change.beat - previous) * 60.0 / bpm
        previous = change.beat
        bpm = float(change.bpm)
    elapsed += float(beat - previous) * 60.0 / bpm
    return elapsed


def _nearest_index(values: list[float], target: float) -> int:
    if not values:
        return 0
    index = bisect.bisect_left(values, target)
    candidates = [candidate for candidate in (index - 1, index) if 0 <= candidate < len(values)]
    return min(candidates, key=lambda candidate: abs(values[candidate] - target))


def _feature_value(frame: Any, key: str) -> float:
    value = float(getattr(frame, key, 0.0) or 0.0)
    if key in {"spectral_centroid", "spectral_bandwidth"}:
        return value / 10000.0
    return value
