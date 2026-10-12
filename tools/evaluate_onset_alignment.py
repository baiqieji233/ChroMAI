"""Frame-level validation for a trained local chart-onset head.

This runs the encoder and onset head only: it does not decode chart tokens or
inject planner targets. Results diagnose alignment on annotated chart starts;
they are not generation or playability metrics.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from evaluate_event_tokens import shuffle_acoustic_frames  # noqa: E402
from maichart.generation.conditions import (  # noqa: E402
    GENERATION_DEFAULTS, RADAR_KEYS, STYLE_KEYS, GenerationCondition,
)
from maichart.generation.event_audio import (  # noqa: E402
    ACOUSTIC_DIM, CONTEXT_KEYS, FEATURE_RATE_HZ, GLOBAL_MEMORY_FRAMES, INPUT_DIM,
    RHYTHM_CONTEXT_KEYS,
)
from maichart.generation.event_cli import apply_contract_condition_guard  # noqa: E402
from maichart.generation.event_ctx import TARGET_SCHEMA  # noqa: E402
from maichart.event_timing import seconds_at_beat  # noqa: E402
from maichart.generation.event_model import EventVocabulary  # noqa: E402
from maichart.generation.sustain_state import action_state_kwargs  # noqa: E402
from maichart.training.event_checkpoints import (  # noqa: E402
    compare_event_contract, load_event_checkpoint, load_event_model, vocabulary_signature,
)
from maichart.training.event_ctx_dataset import EventCtxDataset  # noqa: E402
from maichart.training.event_dataset import collate_events  # noqa: E402
from maichart.training.event_input_policy import stored_input_policies  # noqa: E402
from maichart.training.train_events import pool_onset_targets  # noqa: E402


THRESHOLDS = (0.3, 0.5, 0.7)


def time_only_audio_features(local_audio: torch.Tensor, global_audio: torch.Tensor):
    """Return cloned tracks with acoustics and derived rhythm zeroed, clocks kept."""
    if local_audio.ndim != 3 or global_audio.ndim != 3:
        raise ValueError("local and global audio must have shape [B, T, F]")
    if local_audio.size(0) != global_audio.size(0):
        raise ValueError("local and global audio batch sizes differ")
    if local_audio.size(-1) != INPUT_DIM or global_audio.size(-1) != INPUT_DIM:
        raise ValueError(f"time-only intervention expects {INPUT_DIM} event-audio columns")
    clock_offset = ACOUSTIC_DIM + len(RHYTHM_CONTEXT_KEYS)
    if clock_offset + len(CONTEXT_KEYS) != INPUT_DIM:
        raise RuntimeError("event-audio clock feature layout has changed")
    local = local_audio.clone()
    global_ = global_audio.clone()
    local[..., :clock_offset] = 0
    global_[..., :clock_offset] = 0
    return local, global_


def weighted_average_precision(scores, labels, weights) -> float | None:
    """Weighted binary AP, grouping tied scores before each precision step."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if scores.shape != labels.shape or scores.shape != weights.shape:
        raise ValueError("scores, labels and weights must have the same shape")
    keep = np.isfinite(scores) & np.isfinite(weights) & (weights > 0)
    scores, labels, weights = scores[keep], labels[keep], weights[keep]
    positive_mass = float(weights[labels].sum())
    if not scores.size or positive_mass <= 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    scores, labels, weights = scores[order], labels[order], weights[order]
    group_ends = np.r_[np.flatnonzero(scores[1:] != scores[:-1]), len(scores) - 1]
    positive_cumulative = np.cumsum(weights * labels)[group_ends]
    total_cumulative = np.cumsum(weights)[group_ends]
    precision = positive_cumulative / np.maximum(total_cumulative, 1e-12)
    recall = positive_cumulative / positive_mass
    previous = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - previous) * precision))


def score_frame_predictions(probabilities, target, quality, valid) -> dict[str, Any]:
    """Score pooled frame predictions with quality/core coverage weights."""
    probability = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    quality = np.asarray(quality, dtype=np.float64).reshape(-1)
    valid = np.asarray(valid, dtype=np.float64).reshape(-1)
    if not (probability.shape == target.shape == quality.shape == valid.shape):
        raise ValueError("frame prediction and target arrays must have matching shapes")
    finite = (np.isfinite(probability) & np.isfinite(target)
              & np.isfinite(quality) & np.isfinite(valid))
    frame_weight = np.where(finite, np.maximum(quality, 0) * np.clip(valid, 0, 1), 0.0)
    keep = frame_weight > 0
    count = int(keep.sum())
    weight_sum = float(frame_weight.sum())
    if weight_sum <= 0:
        return {
            "valid_frame_bins": count, "quality_weighted_frame_mass": 0.0,
            "positive_target_frame_bins": 0, "positive_target_fraction": None,
            "weighted_positive_target_fraction": None, "soft_target_mean": None,
            "soft_bce": None, "brier": None, "constant_sample_mean_brier_oracle": None,
            "binary_target_threshold": 0.5, "average_precision": None,
            "average_precision_note": "undefined: no positive-weight valid frames",
            "mean_probability_at_positive_targets": None,
            "thresholds": {str(t): {"precision": None, "recall": None,
                                     "predicted_positive_frame_bins": 0,
                                     "positive_target_frame_bins": 0,
                                     "true_positive_weight": 0.0,
                                     "false_positive_weight": 0.0,
                                     "false_negative_weight": 0.0}
                           for t in THRESHOLDS},
        }

    p = np.clip(probability[keep], 1e-7, 1.0 - 1e-7)
    y = np.clip(target[keep], 0.0, 1.0)
    w = frame_weight[keep]
    binary = y >= 0.5
    positive_bins = int(binary.sum())
    positive_weight = float(w[binary].sum())
    mean_target = float(np.sum(w * y) / weight_sum)
    bce = float(np.sum(w * (-y * np.log(p) - (1.0 - y) * np.log1p(-p))) / weight_sum)
    brier = float(np.sum(w * np.square(p - y)) / weight_sum)
    oracle_brier = float(np.sum(w * np.square(mean_target - y)) / weight_sum)
    at_positive = (float(np.sum(w[binary] * p[binary]) / positive_weight)
                   if positive_weight > 0 else None)
    threshold_metrics = {}
    for threshold in THRESHOLDS:
        predicted = p >= threshold
        tp = float(w[predicted & binary].sum())
        fp = float(w[predicted & ~binary].sum())
        fn = float(w[~predicted & binary].sum())
        predicted_count = int(predicted.sum())
        threshold_metrics[str(threshold)] = {
            "precision": tp / (tp + fp) if tp + fp > 0 else None,
            "recall": tp / (tp + fn) if tp + fn > 0 else None,
            "predicted_positive_frame_bins": predicted_count,
            "positive_target_frame_bins": positive_bins,
            "true_positive_weight": tp,
            "false_positive_weight": fp,
            "false_negative_weight": fn,
        }
    return {
        "valid_frame_bins": count,
        "quality_weighted_frame_mass": weight_sum,
        "positive_target_frame_bins": positive_bins,
        "positive_target_fraction": positive_bins / count,
        "weighted_positive_target_fraction": positive_weight / weight_sum,
        "soft_target_mean": mean_target,
        "soft_bce": bce,
        "brier": brier,
        "constant_sample_mean_brier_oracle": oracle_brier,
        "binary_target_threshold": 0.5,
        "average_precision": weighted_average_precision(p, binary, w),
        "average_precision_note": (
            "Weighted average precision over target>=0.5 labels; equal scores are grouped "
            "before each precision/recall step. Undefined when this evaluation slice has no positives."
        ),
        "mean_probability_at_positive_targets": at_positive,
        "thresholds": threshold_metrics,
    }


def _build_dataset(checkpoint: dict[str, Any], manifest: Path, cache_dir: Path,
                   quality_annotations: Path | None) -> EventCtxDataset:
    contract = checkpoint["data_contract"]
    if contract.get("target_schema") != TARGET_SCHEMA:
        raise ValueError(f"Expected {TARGET_SCHEMA}, got {contract.get('target_schema')!r}.")
    config = checkpoint["model_config"]
    if config.get("onset_profile") != "chart_onsets_v1":
        raise ValueError("The checkpoint must enable chart_onsets_v1.")
    family = contract.get("family_params") or {}
    difficulty = contract.get("difficulty_filter") or {}
    input_policies = stored_input_policies(checkpoint)
    action_state_profile = str(config.get("action_state_profile", "none"))
    dataset = EventCtxDataset(
        manifest,
        cache_dir=cache_dir,
        segment_beats=int(contract["segment_beats"]),
        vocabulary=EventVocabulary.from_dict(checkpoint["vocabulary"]),
        max_tokens=int(contract["max_tokens"]),
        sample_rate=int(contract["sample_rate"]),
        hop_length=int(contract["hop_length"]),
        eager_audio=False,
        global_frames=int(config.get("global_audio_tokens", GLOBAL_MEMORY_FRAMES)),
        condition_target_version=contract.get("condition_target_version", "legacy_unshifted_v1"),
        condition_mask_prob=0.0,
        condition_default=float((contract.get("condition_params") or {}).get("default", 0.5)),
        condition_seed=0,
        min_difficulty_index=int(difficulty.get("min_index", 4)),
        max_difficulty_index=int(difficulty.get("max_index", 6)),
        min_level=float(difficulty.get("min_level", 12.0)),
        max_level=float(difficulty.get("max_level", 15.0)),
        planner_profile=config.get("planner_profile", "legacy_v1"),
        onset_profile="chart_onsets_v1",
        quality_annotations=quality_annotations,
        context_beats=int(family.get("context_beats", 0)),
        audio_margin_beats=int(family.get("audio_margin_beats", 0)),
        context_tokens=int(family.get("context_tokens", 1)),
        overlap_beats=int(family.get("overlap_beats", 0)),
        context_dropout=0.0,
        context_budget_policy=input_policies["context_budget_policy"],
        chart_filter_policy=input_policies["chart_filter_policy"],
        action_state_profile=action_state_profile,
        slide_oversample=1,
        touch_oversample=1,
        fine_rhythm_oversample=1,
    )
    current = {
        "target_schema": TARGET_SCHEMA,
        "feature_profile": contract["feature_profile"],
        "input_dim": INPUT_DIM,
        "sample_rate": dataset.sample_rate,
        "hop_length": dataset.hop_length,
        "segment_beats": dataset.segment_beats,
        "max_tokens": dataset.max_tokens,
        "vocabulary_signature": vocabulary_signature(dataset.vocabulary),
        "dataset_signature": dataset.signature,
        "condition_keys": contract["condition_keys"],
        "condition_target_version": contract.get("condition_target_version", "legacy_unshifted_v1"),
        "input_policies": input_policies,
        "action_state_profile": action_state_profile,
    }
    stored_for_guard = {**checkpoint, "data_contract": {
        key: contract[key] for key in (
            "target_schema", "feature_profile", "input_dim", "sample_rate", "hop_length",
            "segment_beats", "max_tokens", "vocabulary_signature", "condition_keys",
            "condition_target_version", "input_policies",
        ) if key in contract
    }}
    compare_event_contract(stored_for_guard, current, mode="init")
    for key in ("context_beats", "audio_margin_beats", "context_tokens", "overlap_beats"):
        expected = family.get(key)
        if expected is not None and getattr(dataset, key) != expected:
            raise ValueError(f"Checkpoint window contract mismatch for {key}.")
    return dataset


def _conditions(checkpoint: dict[str, Any], dataset: EventCtxDataset,
                indices: list[int]) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    conditions, rows = [], []
    condition_params = checkpoint["data_contract"].get("condition_params", {})
    defaults = GENERATION_DEFAULTS
    for index in indices:
        sample_index, start, end = dataset.segments[index][:3]
        ref = dataset.samples[sample_index]
        level = float(ref.level)
        difficulty = int(ref.chart.difficulty)
        requested = GenerationCondition(
            level=level,
            difficulty=difficulty,
            radar={key: defaults[key] for key in RADAR_KEYS},
            style={key: defaults[key] for key in STYLE_KEYS},
        )
        effective, guard_note = apply_contract_condition_guard(
            requested, condition_params, default_level=level)
        conditions.append(effective.to_vector())
        rows.append({
            "dataset_index": int(index),
            "song_id": ref.song_id,
            "source_path": str(ref.source_path),
            "audio_path": str(ref.audio_path),
            "difficulty_index": difficulty,
            "level": level,
            "start_beat": str(start),
            "end_beat": str(end),
            "core_start_sec": float(seconds_at_beat(start, ref.chart.tempo_changes, ref.chart.audio_first)),
            "core_end_sec": float(seconds_at_beat(end, ref.chart.tempo_changes, ref.chart.audio_first)),
            "chart_first": float(ref.chart.first),
            "audio_first": float(ref.chart.audio_first),
            "tempo_changes": [{"beat": str(change.beat), "bpm": float(change.bpm)}
                              for change in ref.chart.tempo_changes],
            "requested_condition": requested.to_dict(),
            "effective_condition": effective.to_dict(),
            "condition_guard_note": guard_note,
        })
    return torch.stack(conditions), rows


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(2)
    torch.manual_seed(0)
    checkpoint = load_event_checkpoint(args.checkpoint, device="cpu")
    training = checkpoint.get("onset_training") or {}
    onset_contract = checkpoint["data_contract"].get("onset_params") or {}
    if (training.get("profile") != "chart_onsets_v1" or training.get("trained") is not True
            or int(training.get("optimizer_steps", 0)) <= 0
            or float(onset_contract.get("loss_weight", 0)) <= 0):
        raise ValueError("The checkpoint does not record a trained chart_onsets_v1 head.")
    dataset = _build_dataset(checkpoint, args.manifest, args.cache_dir, args.quality_annotations)
    model, vocabulary, loaded = load_event_model(args.checkpoint, torch.device("cpu"))
    if loaded.get("epoch") != checkpoint.get("epoch"):
        raise RuntimeError("Checkpoint epoch changed between reads.")
    model.eval()
    count = min(int(args.limit), len(dataset))
    if count <= 0:
        raise ValueError("Validation dataset contains no usable windows.")
    indices = np.linspace(0, len(dataset) - 1, num=count, dtype=np.int64).tolist()
    predicted_frames: list[np.ndarray] = []
    target_frames: list[np.ndarray] = []
    quality_frames: list[np.ndarray] = []
    valid_frames: list[np.ndarray] = []
    shuffled_frames: list[np.ndarray] = []
    time_only_frames: list[np.ndarray] = []
    windows: list[dict[str, Any]] = []

    for batch_start in range(0, len(indices), args.batch_size):
        selected = indices[batch_start:batch_start + args.batch_size]
        items = [dataset[index] for index in selected]
        batch = collate_events(items, pad_id=vocabulary.pad_id)
        conditions, metadata = _conditions(checkpoint, dataset, selected)
        condition_null = torch.zeros(len(selected), dtype=torch.bool)
        with torch.inference_mode():
            def predict(local_audio: torch.Tensor, global_audio: torch.Tensor | None = None) -> torch.Tensor:
                memory, _ = model.encode_audio(
                    local_audio, conditions,
                    audio_padding_mask=batch["audio_padding_mask"],
                    global_audio=batch["global_audio"] if global_audio is None else global_audio,
                    condition_null=condition_null,
                    plan_audio_mask=batch["plan_audio_mask"],
                    **action_state_kwargs(batch, batched=True),
                )
                return torch.sigmoid(model.local_onset_logits(memory).float())

            probabilities = predict(batch["audio"])
            if args.shuffle_acoustic:
                shuffled = shuffle_acoustic_frames(
                    batch["audio"], batch["audio_padding_mask"], seed=args.seed + batch_start)
                shuffled_probabilities = predict(shuffled)
            else:
                shuffled_probabilities = None
            if args.time_only:
                local_clock, global_clock = time_only_audio_features(
                    batch["audio"], batch["global_audio"])
                time_only_probabilities = predict(local_clock, global_clock)
            else:
                time_only_probabilities = None
        target, quality, valid = pool_onset_targets(batch, model)
        if probabilities.shape != target.shape:
            raise RuntimeError(f"onset head/target pooling mismatch: {tuple(probabilities.shape)} != {tuple(target.shape)}")
        for local, row in enumerate(metadata):
            p = probabilities[local].cpu().numpy()
            y = target[local].cpu().numpy()
            q = quality[local].cpu().numpy()
            v = valid[local].cpu().numpy()
            row["frame_bins"] = int(p.size)
            row["metrics"] = score_frame_predictions(p, y, q, v)
            predicted_frames.append(p)
            target_frames.append(y)
            quality_frames.append(q)
            valid_frames.append(v)
            if shuffled_probabilities is not None:
                p_shuffled = shuffled_probabilities[local].cpu().numpy()
                row["shuffle_acoustic_metrics"] = score_frame_predictions(p_shuffled, y, q, v)
                shuffled_frames.append(p_shuffled)
            if time_only_probabilities is not None:
                p_time_only = time_only_probabilities[local].cpu().numpy()
                row["time_only_metrics"] = score_frame_predictions(p_time_only, y, q, v)
                time_only_frames.append(p_time_only)
            windows.append(row)

    all_metrics = score_frame_predictions(
        np.concatenate(predicted_frames), np.concatenate(target_frames),
        np.concatenate(quality_frames), np.concatenate(valid_frames))
    shuffle_metrics = None
    if args.shuffle_acoustic:
        shuffle_metrics = score_frame_predictions(
            np.concatenate(shuffled_frames), np.concatenate(target_frames),
            np.concatenate(quality_frames), np.concatenate(valid_frames))
    time_only_metrics = None
    if args.time_only:
        time_only_metrics = score_frame_predictions(
            np.concatenate(time_only_frames), np.concatenate(target_frames),
            np.concatenate(quality_frames), np.concatenate(valid_frames))
    return {
        "schema": "local_chart_onset_alignment_v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
        "onset_training": training,
        "onset_params": onset_contract,
        "model_config": {"onset_profile": model.config.onset_profile,
                         "local_pooling_policy": model.config.local_pooling_policy,
                         "max_audio_tokens": model.config.max_audio_tokens,
                         "action_state_profile": model.config.action_state_profile},
        "dataset_input_policies": stored_input_policies(checkpoint),
        "manifest": str(args.manifest.resolve()),
        "quality_annotations": str(args.quality_annotations.resolve()) if args.quality_annotations else None,
        "evaluation": {
            "mode": "encoder_local_head_only",
            "decoder_used": False,
            "planner_target_injected": False,
            "input_state": {
                "action_state_profile": model.config.action_state_profile,
                "input_policies": stored_input_policies(checkpoint),
            },
            "condition_source": "GENERATION_DEFAULTS radar/style plus known source level/difficulty, after contract guard",
            "window_selection": "uniform deterministic linspace over EventCtxDataset order",
            "windows_requested": int(args.limit),
            "windows_evaluated": len(windows),
            "dataset_windows": len(dataset),
            "thread_count": torch.get_num_threads(),
            "raw_feature_rate_hz": FEATURE_RATE_HZ,
            "binary_label": "soft Gaussian target >= 0.5; Gaussian sigma comes from onset_params",
            "average_precision": "weighted average precision; tied scores are grouped; not trapezoidal PR area",
            "constant_sample_mean_brier": "evaluation-slice oracle baseline fitted on these same labels; descriptive only and unavailable at inference",
            "shuffle_acoustic": ({
                "enabled": True,
                "seed": int(args.seed),
                "method": "evaluate_event_tokens.shuffle_acoustic_frames per window; shuffles local acoustic/rhythm rows while retaining time-context columns, global audio, conditions, and authored action_state",
                "interpretation": "alignment ablation only; not proof of audio autonomy because chart targets, conditions and global memory remain",
            } if args.shuffle_acoustic else None),
            "time_only": ({
                "enabled": True,
                "method": (f"zero local/global feature columns [0:{ACOUSTIC_DIM + len(RHYTHM_CONTEXT_KEYS)}] "
                           f"(acoustic plus audio-derived rhythm); preserve pure clock columns "
                           f"[{ACOUSTIC_DIM + len(RHYTHM_CONTEXT_KEYS)}:{INPUT_DIM}]"),
                "interpretation": "out-of-distribution diagnostic intervention; tests whether chart-clock and authored action-state inputs alone support chart-onset prediction, but does not prove causality or deployment performance",
            } if args.time_only else None),
            "limitations": [
                "Frame metrics measure agreement with author-selected chart starts, not acoustic ground truth or playability.",
                "The constant sample-mean Brier baseline is an oracle over this evaluation slice.",
                "No peak-to-exact-event timing tolerance is reported; the head is evaluated on its pooled frame grid.",
                "Shuffle-acoustic results retain global memory and chart-derived target labels; they do not establish audio-only behavior.",
                "Time-only zeroing is an out-of-distribution intervention; it is not a causal estimate or a deployable baseline.",
            ],
        },
        "metrics": all_metrics,
        "shuffle_acoustic_metrics": shuffle_metrics,
        "time_only_metrics": time_only_metrics,
        "windows": windows,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("cache/events"))
    parser.add_argument("--quality-annotations", type=Path)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--shuffle-acoustic", action="store_true")
    parser.add_argument("--time-only", action="store_true",
                        help="Zero local/global acoustic and audio-derived rhythm features while retaining clock columns (OOD diagnostic).")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.limit < 1 or args.batch_size < 1:
        raise SystemExit("--limit and --batch-size must be positive")
    result = evaluate(args)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(json.dumps({"checkpoint_epoch": result["checkpoint_epoch"],
                          "windows_evaluated": result["evaluation"]["windows_evaluated"],
                          "metrics": result["metrics"],
                          "shuffle_acoustic_metrics": result["shuffle_acoustic_metrics"],
                          "time_only_metrics": result["time_only_metrics"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
