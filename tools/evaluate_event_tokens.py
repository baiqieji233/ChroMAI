"""Teacher-forced token diagnostics for event_ctx checkpoints.

This measures next-token prediction with the chart's true previous tokens in
the decoder prefix. It does not generate charts or assess playability.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from maichart.generation.conditions import (  # noqa: E402
    CONDITION_KEYS, GENERATION_DEFAULTS, RADAR_KEYS, STYLE_KEYS, GenerationCondition,
)
from maichart.generation.event_audio import (  # noqa: E402
    ACOUSTIC_DIM, CONTEXT_KEYS, GLOBAL_MEMORY_FRAMES, INPUT_DIM, RHYTHM_CONTEXT_KEYS,
)
from maichart.generation.event_ctx import TARGET_SCHEMA  # noqa: E402
from maichart.generation.event_cli import apply_contract_condition_guard  # noqa: E402
from maichart.generation.event_model import EVENT_END, EventVocabulary  # noqa: E402
from maichart.generation.sustain_state import action_state_kwargs  # noqa: E402
from maichart.generation.music_plan import continuous_keys, is_music_structure, plan_values  # noqa: E402
from maichart.training.event_checkpoints import (  # noqa: E402
    compare_event_contract,
    load_event_checkpoint,
    load_event_model,
    vocabulary_signature,
)
from maichart.training.event_ctx_dataset import EventCtxDataset  # noqa: E402
from maichart.training.event_dataset import collate_events  # noqa: E402
from maichart.training.event_input_policy import stored_input_policies  # noqa: E402


# These are diagnostic syntax markers, not parsed note-type labels. One chart
# note can contain multiple markers, so counts are character-token counts.
MARKER_CHARS = {
    "hold": frozenset("h"),
    "slide": frozenset("-<>^pqszvwV"),
    "slide_branch": frozenset("*"),
    "chord_separator": frozenset("/"),
    "touch": frozenset("ABCDE"),
}
ALL_MARKERS = frozenset().union(*MARKER_CHARS.values())


def _ratio(correct: int, total: int) -> float | None:
    return correct / total if total else None


def score_token_batch(
    vocabulary: EventVocabulary,
    probabilities: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor,
    padding_mask: torch.Tensor,
) -> dict[str, Any]:
    """Summarize one sequence using only scored, non-padding targets."""
    predicted = probabilities.argmax(dim=-1)
    valid = loss_mask.bool() & ~padding_mask.bool()
    valid_indices = torch.nonzero(valid, as_tuple=False).flatten().tolist()
    token_correct = 0
    chars: dict[str, dict[str, float | int]] = defaultdict(lambda: {"count": 0, "correct": 0, "truth_probability_sum": 0.0})
    positions: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "correct": 0})
    premature = 0
    char_count = 0
    first_position_seen = False
    previous_position: Fraction | None = None

    for index in valid_indices:
        truth_id = int(target[index])
        guess_id = int(predicted[index])
        truth = vocabulary.id_to_token[truth_id]
        guess = vocabulary.id_to_token[guess_id]
        token_correct += guess_id == truth_id

        if truth.startswith("AT:"):
            beat = Fraction(truth[3:])
            if not first_position_seen:
                group = "first"
                first_position_seen = True
            else:
                assert previous_position is not None
                group = str(beat - previous_position)
            bucket = positions[group]
            bucket["count"] += 1
            bucket["correct"] += guess_id == truth_id
            previous_position = beat
            continue

        if truth.startswith("CHAR:"):
            char_count += 1
            char = truth[5:]
            category = next((name for name, chars_set in MARKER_CHARS.items() if char in chars_set), "ordinary_syntax")
            bucket = chars[category]
            bucket["count"] += 1
            bucket["correct"] += guess_id == truth_id
            bucket["truth_probability_sum"] += float(probabilities[index, truth_id])
            premature += guess == EVENT_END
            continue

    total = len(valid_indices)
    return {
        "token_count": total,
        "token_correct": int(token_correct),
        "token_accuracy": _ratio(token_correct, total),
        "position_groups": {
            key: {"count": value["count"], "correct": value["correct"],
                  "top1_accuracy": _ratio(value["correct"], value["count"])}
            for key, value in sorted(positions.items(), key=lambda item: (item[0] != "first", item[0]))
        },
        "character_groups": {
            key: {"count": value["count"], "correct": value["correct"],
                  "top1_accuracy": _ratio(value["correct"], value["count"]),
                  "mean_truth_probability": (value["truth_probability_sum"] / value["count"]
                                              if value["count"] else None)}
            for key, value in sorted(chars.items())
        },
        "note_character_count": char_count,
        "premature_event_end": {"count": int(premature), "among_true_note_characters": char_count,
                                "ratio": _ratio(premature, char_count)},
    }


def _add_metric(total: dict[str, Any], result: dict[str, Any]) -> None:
    total["token_count"] += result["token_count"]
    total["token_correct"] += result["token_correct"]
    total["note_character_count"] += result["note_character_count"]
    total["premature_event_end_count"] += result["premature_event_end"]["count"]
    for group, value in result["position_groups"].items():
        bucket = total["position_groups"].setdefault(group, {"count": 0, "correct": 0})
        bucket["count"] += value["count"]
        bucket["correct"] += value["correct"]
    for group, value in result["character_groups"].items():
        bucket = total["character_groups"].setdefault(group, {"count": 0, "correct": 0, "truth_probability_sum": 0.0})
        bucket["count"] += value["count"]
        bucket["correct"] += value["correct"]
        bucket["truth_probability_sum"] += value["mean_truth_probability"] * value["count"]


def _finalize_metrics(total: dict[str, Any]) -> dict[str, Any]:
    return {
        "token_count": total["token_count"],
        "token_correct": total["token_correct"],
        "token_accuracy": _ratio(total["token_correct"], total["token_count"]),
        "position_groups": {
            key: {"count": value["count"], "correct": value["correct"],
                  "top1_accuracy": _ratio(value["correct"], value["count"])}
            for key, value in sorted(total["position_groups"].items(), key=lambda item: (item[0] != "first", item[0]))
        },
        "character_groups": {
            key: {"count": value["count"], "correct": value["correct"],
                  "top1_accuracy": _ratio(value["correct"], value["count"]),
                  "mean_truth_probability": value["truth_probability_sum"] / value["count"]}
            for key, value in sorted(total["character_groups"].items())
        },
        "premature_event_end": {
            "count": total["premature_event_end_count"],
            "among_true_note_characters": total["note_character_count"],
            "ratio": _ratio(total["premature_event_end_count"], total["note_character_count"]),
        },
    }


def shuffle_acoustic_frames(audio: torch.Tensor, padding_mask: torch.Tensor, *, seed: int) -> torch.Tensor:
    """Shuffle local acoustic/rhythm rows per sample, retaining beat/time columns."""
    result = audio.clone()
    acoustic_end = ACOUSTIC_DIM + len(RHYTHM_CONTEXT_KEYS)
    if acoustic_end + len(CONTEXT_KEYS) != audio.size(-1):
        raise ValueError("event audio feature layout does not match this diagnostic.")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    for row in range(audio.size(0)):
        valid = torch.nonzero(~padding_mask[row].bool(), as_tuple=False).flatten()
        if valid.numel() > 1:
            order = torch.randperm(valid.numel(), generator=generator)
            result[row, valid, :acoustic_end] = audio[row, valid[order], :acoustic_end]
    return result


def _dataset(checkpoint: dict[str, Any], manifest: Path, cache_dir: Path) -> EventCtxDataset:
    contract = checkpoint["data_contract"]
    if contract.get("target_schema") != TARGET_SCHEMA:
        raise ValueError(f"Expected {TARGET_SCHEMA}, got {contract.get('target_schema')!r}.")
    config = checkpoint["model_config"]
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
        "feature_profile": "event_audio_v2_rhythm",
        "input_dim": INPUT_DIM,
        "sample_rate": dataset.sample_rate,
        "hop_length": dataset.hop_length,
        "segment_beats": dataset.segment_beats,
        "max_tokens": dataset.max_tokens,
        "vocabulary_signature": vocabulary_signature(dataset.vocabulary),
        "dataset_signature": dataset.signature,
        "condition_keys": list(CONDITION_KEYS),
        "condition_target_version": contract.get("condition_target_version", "legacy_unshifted_v1"),
        "input_policies": input_policies,
        "action_state_profile": action_state_profile,
    }
    # Match the same strict structural guard used by training, while checking
    # inference-time window geometry explicitly below.
    stored_for_guard = {**checkpoint, "data_contract": {
        key: contract[key] for key in (
            "target_schema", "feature_profile", "input_dim", "sample_rate", "hop_length",
            "segment_beats", "max_tokens", "vocabulary_signature", "condition_keys",
            "condition_target_version", "input_policies",
        ) if key in contract
    }}
    compare_event_contract(stored_for_guard, current, mode="init")
    for key in ("context_beats", "audio_margin_beats", "context_tokens", "overlap_beats"):
        actual = getattr(dataset, key)
        expected = family.get(key)
        if expected is not None and actual != expected:
            raise ValueError(f"Checkpoint window contract mismatch for {key}: {actual} != {expected}.")
    return dataset


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(2)
    torch.manual_seed(0)
    checkpoint = load_event_checkpoint(args.checkpoint, device="cpu")
    dataset = _dataset(checkpoint, args.manifest, args.cache_dir)
    model, vocabulary, loaded = load_event_model(args.checkpoint, torch.device("cpu"))
    if loaded.get("epoch") != checkpoint.get("epoch"):
        raise RuntimeError("Checkpoint epoch changed between reads.")
    model.eval()
    count = min(args.limit, len(dataset))
    if count <= 0:
        raise ValueError("Validation dataset contains no usable windows.")
    indices = np.linspace(0, len(dataset) - 1, num=count, dtype=np.int64).tolist()
    totals: dict[str, Any] = {"token_count": 0, "token_correct": 0, "note_character_count": 0,
                              "premature_event_end_count": 0, "position_groups": {}, "character_groups": {}}
    ablated_totals: dict[str, Any] = {"token_count": 0, "token_correct": 0, "note_character_count": 0,
                                      "premature_event_end_count": 0, "position_groups": {}, "character_groups": {}}
    rows: list[dict[str, Any]] = []
    device = torch.device("cpu")
    for batch_start in range(0, len(indices), args.batch_size):
        selected = indices[batch_start:batch_start + args.batch_size]
        items = [dataset[index] for index in selected]
        batch = collate_events(items, pad_id=vocabulary.pad_id)
        conditions = []
        metadata = []
        for index in selected:
            sample_index, start, end = dataset.segments[index][:3]
            ref = dataset.samples[sample_index]
            difficulty = int(ref.chart.difficulty)
            defaults = dict(GENERATION_DEFAULTS)
            level = float(ref.level)
            condition = GenerationCondition(
                level=level, difficulty=difficulty,
                radar=defaults, style=defaults,
            )
            requested_condition = condition.to_dict()
            condition, condition_note = apply_contract_condition_guard(
                condition, checkpoint["data_contract"].get("condition_params", {}), default_level=level)
            conditions.append(condition.to_vector())
            metadata.append({
                "dataset_index": index,
                "song_id": ref.song_id,
                "source_path": str(ref.source_path),
                "audio_path": str(ref.audio_path),
                "difficulty_index": difficulty,
                "level": level,
                "requested_condition": requested_condition,
                "condition": condition.to_dict(),
                "condition_guard_note": condition_note,
                "start_beat": str(start),
                "end_beat": str(end),
                "tempo_changes": [{"beat": str(change.beat), "bpm": float(change.bpm)}
                                  for change in ref.chart.tempo_changes],
            })
        condition_tensor = torch.stack(conditions).to(device)
        with torch.inference_mode():
            def forward(audio_tensor):
                return model(
                    audio_tensor.to(device), batch["decoder_input"].to(device), condition_tensor,
                    condition_null=torch.zeros(len(selected), dtype=torch.bool, device=device),
                    audio_padding_mask=batch["audio_padding_mask"].to(device),
                    decoder_padding_mask=batch["decoder_padding_mask"].to(device),
                    global_audio=batch["global_audio"].to(device),
                    plan_audio_mask=batch["plan_audio_mask"].to(device),
                    return_plan=True,
                    **action_state_kwargs(batch, batched=True),
                )

            logits, plan_logits = forward(batch["audio"])
            probabilities = logits.float().softmax(dim=-1)
            values = plan_values(plan_logits, profile=model.planner_profile)
            ablated_probabilities = None
            if args.audio_ablation == "shuffle_acoustic":
                shuffled = shuffle_acoustic_frames(batch["audio"], batch["audio_padding_mask"],
                                                   seed=args.seed + batch_start)
                ablated_logits, _ = forward(shuffled)
                ablated_probabilities = ablated_logits.float().softmax(dim=-1)
        for local, item_meta in enumerate(metadata):
            result = score_token_batch(
                vocabulary,
                probabilities[local],
                batch["target"][local],
                batch["loss_mask"][local],
                batch["decoder_padding_mask"][local],
            )
            _add_metric(totals, result)
            ablated_result = None
            if ablated_probabilities is not None:
                ablated_result = score_token_batch(
                    vocabulary, ablated_probabilities[local], batch["target"][local],
                    batch["loss_mask"][local], batch["decoder_padding_mask"][local],
                )
                _add_metric(ablated_totals, ablated_result)
            plan_names = continuous_keys(model.planner_profile)
            continuous_count = len(plan_names)
            item_meta["predicted_plan"] = {
                **{name: float(values[local, position]) for position, name in enumerate(plan_names)},
                "configuration_probabilities": [float(v) for v in values[local, continuous_count:].tolist()],
            } if is_music_structure(model.planner_profile) else None
            item_meta["metrics"] = result
            if ablated_result is not None:
                item_meta["audio_ablation_metrics"] = ablated_result
            rows.append(item_meta)

    result = {
        "schema": "event_token_teacher_forced_diagnostic_v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
        "model_family": checkpoint.get("model_family", checkpoint["data_contract"].get("target_schema")),
        "data_contract": checkpoint["data_contract"],
        "evaluation": {
            "mode": "teacher_forced",
            "history": "Correct chart tokens are supplied in decoder_input; only loss_mask=true, non-padding targets are scored.",
            "window_selection": "Uniform deterministic linspace over validation dataset order.",
            "windows_requested": args.limit,
            "windows_evaluated": len(rows),
            "dataset_windows": len(dataset),
            "dataset_input_policies": stored_input_policies(checkpoint),
            "action_state_profile": model.config.action_state_profile,
            "plan_target_injected": False,
            "input_state": {
                "action_state_profile": model.config.action_state_profile,
                "input_policies": stored_input_policies(checkpoint),
            },
            "conditions": "GENERATION_DEFAULTS radar/style plus source chart level and difficulty metadata.",
            "audio_ablation": (None if args.audio_ablation == "none" else {
                "kind": args.audio_ablation,
                "seed": args.seed,
                "description": "Per-window local frames are permuted for acoustic and audio-derived rhythm columns; pure time/beat columns, global_audio, conditions, decoder history, authored action_state, and targets are retained. The predicted planner is recomputed from ablated input. This breaks local audio-time alignment and is not an audio-autonomy score.",
            }),
            "thread_count": torch.get_num_threads(),
            "limitations": [
                "Character marker metrics are syntax-character token metrics, not complete note-type F1.",
                "Teacher-forced accuracy is not free-running generation quality or playability.",
                "Global metrics exclude context targets and padding through the training loss mask.",
            ],
        },
        "metrics": _finalize_metrics(totals),
        "audio_ablation_metrics": (_finalize_metrics(ablated_totals) if args.audio_ablation != "none" else None),
        "windows": rows,
    }
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("cache/events"))
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--audio-ablation", choices=("none", "shuffle_acoustic"), default="none")
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
                          "audio_ablation_metrics": result["audio_ablation_metrics"]}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
