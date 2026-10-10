"""Evaluate planner variation on fixed validation windows, without decoding notes."""
from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from maichart.training.event_checkpoints import load_event_model
from maichart.training.event_dataset import EventDataset
from maichart.training.event_ctx_dataset import EventCtxDataset
from maichart.generation.music_plan import continuous_keys, plan_values, is_music_structure
from maichart.generation.conditions import (
    GENERATION_DEFAULTS, RADAR_KEYS, STYLE_KEYS, GenerationCondition,
)
from maichart.generation.event_cli import apply_contract_condition_guard


def _condition_for_evaluation(
    source: str,
    *,
    contract: dict,
    level: float,
    difficulty: int,
    authored_vector=None,
) -> tuple[GenerationCondition, GenerationCondition, str]:
    if source == "generation_defaults":
        requested = GenerationCondition(
            level=level,
            difficulty=difficulty,
            radar={key: GENERATION_DEFAULTS[key] for key in RADAR_KEYS},
            style={key: GENERATION_DEFAULTS[key] for key in STYLE_KEYS},
        )
    elif source == "authored":
        if authored_vector is None:
            raise ValueError("authored condition source requires a chart-derived condition vector")
        values = torch.as_tensor(authored_vector).detach().float().reshape(-1).tolist()
        if len(values) != 1 + len(RADAR_KEYS) + len(STYLE_KEYS):
            raise ValueError(f"Unexpected authored condition vector size: {len(values)}")
        requested = GenerationCondition(
            level=values[0] * 15.0,
            difficulty=difficulty,
            radar=dict(zip(RADAR_KEYS, values[1:1 + len(RADAR_KEYS)])),
            style=dict(zip(STYLE_KEYS, values[1 + len(RADAR_KEYS):])),
        )
    else:
        raise ValueError(f"Unknown condition source: {source}")
    effective, guard_note = apply_contract_condition_guard(
        requested, contract, default_level=level)
    return requested, effective, guard_note


def _channel_metrics(predicted: np.ndarray, target: np.ndarray, name: str) -> dict:
    errors = predicted - target
    target_mean = float(target.mean())
    metrics = {
        "mae": float(np.abs(errors).mean()),
        "mse": float(np.square(errors).mean()),
        "prediction_std": float(predicted.std()),
        "target_std": float(target.std()),
        "prediction_mean": float(predicted.mean()),
        "target_mean": target_mean,
        "zero_baseline_mae": float(np.abs(target).mean()),
        "zero_baseline_mse": float(np.square(target).mean()),
        "sample_oracle_mean_baseline_mae": float(np.abs(target - target_mean).mean()),
        "sample_oracle_mean_baseline_mse": float(np.square(target - target_mean).mean()),
        "sample_oracle_median_baseline_mae": float(np.abs(target - np.median(target)).mean()),
        "correlation": (float(np.corrcoef(predicted, target)[0, 1])
                        if predicted.std() > 1e-8 and target.std() > 1e-8 else None),
    }
    if name == "fine_rhythm":
        positive = target > 0
        metrics.update({
            "positive_target_count": int(positive.sum()),
            "positive_target_fraction": float(positive.mean()),
            "positive_target_mae": (float(np.abs(errors[positive]).mean()) if positive.any() else None),
            "positive_target_prediction_mean": (float(predicted[positive].mean()) if positive.any() else None),
            "positive_target_prediction_std": (float(predicted[positive].std()) if positive.any() else None),
            "positive_target_mean": (float(target[positive].mean()) if positive.any() else None),
        })
    return metrics


def evaluate(checkpoints, manifest, cache_dir, output, limit=64, *,
             condition_source="generation_defaults", level=None, difficulty=None):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    reports = []
    dataset = None
    dataset_contract = None
    selected_keys = None
    requested_level = None if level is None else float(level)
    requested_difficulty = None if difficulty is None else int(difficulty)
    if requested_level is not None and not 1.0 <= requested_level <= 15.0:
        raise ValueError("level must be in [1, 15]")
    if requested_difficulty is not None and requested_difficulty not in range(7):
        raise ValueError("difficulty must be a known index in 0..6")
    for path in checkpoints:
        model, vocabulary, checkpoint = load_event_model(path, torch.device("cpu"))
        contract = checkpoint["data_contract"]
        profile = model.config.planner_profile
        if not is_music_structure(profile) or not model.planner_enabled:
            raise ValueError("This tool requires a trained structure planner")
        names = continuous_keys(profile)
        signature = {k: contract.get(k) for k in ("target_schema", "segment_beats", "max_tokens", "family_params", "vocabulary_signature")}
        signature["planner_profile"] = profile
        if dataset is None:
            cls = EventCtxDataset if contract["target_schema"] == "event_ctx_v1" else EventDataset
            family = contract.get("family_params", {})
            kwargs = {k: family[k] for k in ("context_beats", "audio_margin_beats", "context_tokens", "overlap_beats") if k in family}
            if cls is EventCtxDataset:
                kwargs.update(context_dropout=0, slide_oversample=1, touch_oversample=1, fine_rhythm_oversample=1)
            with output.with_suffix(".dataset.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
                dataset = cls(manifest, cache_dir=cache_dir, segment_beats=contract["segment_beats"],
                              max_tokens=contract["max_tokens"], vocabulary=vocabulary, planner_profile=profile,
                              eager_audio=False, condition_mask_prob=0,
                              global_frames=model.config.global_audio_tokens, **kwargs)
            dataset_contract = signature
            selected = np.linspace(0, len(dataset)-1, min(limit, len(dataset)), dtype=int).tolist()
            selected_keys = selected
        elif signature != dataset_contract:
            raise ValueError("Checkpoints have different dataset/layout/target contracts; compare in separate reports")
        targets, predictions, windows = [], [], []
        with torch.inference_mode():
            for index in selected_keys:
                item = dataset[index]
                sample_index = dataset.segments[index][0]
                sample = dataset.samples[sample_index]
                chart_level = float(sample.level)
                chart_difficulty = int(sample.chart.difficulty)
                window_level = requested_level if requested_level is not None else chart_level
                window_difficulty = (requested_difficulty if requested_difficulty is not None
                                     else chart_difficulty)
                if condition_source == "authored":
                    requested_condition, model_condition, guard_note = _condition_for_evaluation(
                        condition_source, contract=checkpoint.get("data_contract", {}).get("condition_params", {}),
                        level=window_level, difficulty=window_difficulty,
                        authored_vector=item["condition"])
                    condition_null = item["condition_null"].reshape(1)
                    effective_source = "validation_chart_authored_controls"
                else:
                    requested_condition, model_condition, guard_note = _condition_for_evaluation(
                        condition_source, contract=checkpoint.get("data_contract", {}).get("condition_params", {}),
                        level=window_level, difficulty=window_difficulty)
                    condition_null = torch.zeros((1,), dtype=torch.bool)
                    effective_source = "generation_defaults_fixed_recipe"
                memory, padding = model.encode_audio(
                    item["audio"].unsqueeze(0), model_condition.to_vector().unsqueeze(0),
                    global_audio=item["global_audio"].unsqueeze(0), condition_null=condition_null,
                    plan_audio_mask=item["plan_audio_mask"].unsqueeze(0) if "plan_audio_mask" in item else None)
                _, _, logits = model.append_plan_memory(memory, padding)
                predicted = plan_values(logits, profile=profile)[0, :len(names)].tolist()
                target = item["plan_target"][:len(names)].tolist()
                predictions.append(predicted)
                targets.append(target)
                windows.append({**item["meta"], "condition_source": effective_source,
                                "requested_condition": requested_condition.to_dict(),
                                "condition": model_condition.to_dict(),
                                "condition_metadata": {
                                    "level_source": ("validation chart condition vector" if condition_source == "authored"
                                                     else "CLI --level override" if requested_level is not None
                                                     else "reference chart level metadata"),
                                    "difficulty_source": ("reference chart difficulty metadata" if condition_source == "authored"
                                                          else "CLI --difficulty override" if requested_difficulty is not None
                                                          else "reference chart difficulty metadata"),
                                    "contract_guard_note": guard_note,
                                },
                                "prediction": dict(zip(names,predicted)), "target": dict(zip(names,target))})
        predicted, target = np.asarray(predictions), np.asarray(targets)
        channels = {}
        for i, name in enumerate(names):
            p, t = predicted[:,i], target[:,i]
            channels[name] = _channel_metrics(p, t, name)
        reports.append({"checkpoint": str(Path(path).resolve()), "epoch": checkpoint["epoch"],
                        "planner_profile":profile, "window_count":len(windows), "channels":channels,
                        "condition": {
                            "source_mode": condition_source,
                            "source": ("GENERATION_DEFAULTS radar/style plus known level/difficulty metadata"
                                       if condition_source == "generation_defaults"
                                       else "validation chart authored controls"),
                            "requested": "per window; see requested_condition in each window",
                            "effective_after_contract_guard": "per window; see condition in each window",
                            "guard_note": "per window; see condition_metadata in each window",
                            "level_strategy": ("CLI --level fixed across windows; otherwise known reference-chart level metadata"
                                               if condition_source == "generation_defaults"
                                               else "level from validation chart condition vector"),
                            "difficulty_strategy": ("CLI --difficulty fixed across windows; otherwise known reference-chart difficulty metadata; metadata only, not encoded in planner condition vector"
                                                    if condition_source == "generation_defaults"
                                                    else "difficulty index is validation chart metadata"),
                        }, "windows":windows})
        if condition_source == "generation_defaults":
            condition_caveat = ("This fixes radar/style to one GENERATION_DEFAULTS recipe and uses known per-window "
                                "level/difficulty metadata; it does not represent every WebUI preset.")
        else:
            condition_caveat = ("Authored mode reads chart-derived radar/style controls from the same reference chart; "
                                "its metrics are not evidence of audio-only planner capability.")
        output.write_text(json.dumps({"schema":"maichart-planner-variation-v2", "manifest":str(Path(manifest).resolve()),
            "condition_source": condition_source,
            "condition_policy": condition_caveat,
            "metric_context":"Continuous planner channels are trained with MSE; MAE/MSE and constant baselines are descriptive regression diagnostics, not chart-quality scores.",
            "sampling":"evenly spaced fixed window indices; same selected indices for every checkpoint; no teacher plan target",
            "limitations":["Small deterministic panel is not the full validation metric or musical quality.",
                            "Sample-mean/median baselines use this evaluation sample's labels and are oracle constant predictors, not deployable baselines.",
                            "Generation-default mode fixes radar/style to one GENERATION_DEFAULTS recipe, not every WebUI preset; level/difficulty are known reference metadata unless explicitly overridden.",
                            "Because level/difficulty can come from the reference chart, this is not a fully audio-autonomous prediction test.",
                            "Authored mode contains reference-chart-derived controls and cannot establish audio-only capability."],
            "models":reports},ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
        print(f"planner_evaluated epoch={checkpoint['epoch']} windows={len(windows)} fine={channels.get('fine_rhythm')}",flush=True)
    return reports


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoints",nargs="+")
    p.add_argument("--manifest",required=True)
    p.add_argument("--cache-dir",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--limit",type=int,default=64)
    p.add_argument("--condition-source", choices=("generation_defaults", "authored"),
                   default="generation_defaults",
                   help="planner input condition source (default: fixed generation defaults; authored uses reference-chart controls)")
    p.add_argument("--level", type=float, default=None,
                   help="fixed known target level; defaults to each validation chart's known level metadata")
    p.add_argument("--difficulty", type=int, default=None,
                   help="fixed known target difficulty index 0..6; defaults to each validation chart's metadata (not encoded in planner condition vector)")
    a=p.parse_args(argv)
    if a.limit < 1: p.error("limit must be positive")
    try:
        evaluate(a.checkpoints,a.manifest,a.cache_dir,a.output,a.limit,
                 condition_source=a.condition_source, level=a.level, difficulty=a.difficulty)
    except ValueError as exc:
        p.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
