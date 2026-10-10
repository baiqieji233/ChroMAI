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


def evaluate(checkpoints, manifest, cache_dir, output, limit=64):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    reports = []
    dataset = None
    dataset_contract = None
    selected_keys = None
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
                memory, padding = model.encode_audio(
                    item["audio"].unsqueeze(0), item["condition"].unsqueeze(0),
                    global_audio=item["global_audio"].unsqueeze(0), condition_null=item["condition_null"].reshape(1),
                    plan_audio_mask=item["plan_audio_mask"].unsqueeze(0) if "plan_audio_mask" in item else None)
                _, _, logits = model.append_plan_memory(memory, padding)
                predicted = plan_values(logits, profile=profile)[0, :len(names)].tolist()
                target = item["plan_target"][:len(names)].tolist()
                predictions.append(predicted)
                targets.append(target)
                windows.append({**item["meta"], "prediction": dict(zip(names,predicted)), "target": dict(zip(names,target))})
        predicted, target = np.asarray(predictions), np.asarray(targets)
        channels = {}
        for i, name in enumerate(names):
            p, t = predicted[:,i], target[:,i]
            channels[name] = {
                "mae": float(np.abs(p-t).mean()), "prediction_std": float(p.std()), "target_std": float(t.std()),
                "prediction_mean": float(p.mean()), "target_mean": float(t.mean()),
                "zero_baseline_mae": float(np.abs(t).mean()),
                "sample_oracle_median_baseline_mae": float(np.abs(t-np.median(t)).mean()),
                "correlation": float(np.corrcoef(p,t)[0,1]) if p.std()>1e-8 and t.std()>1e-8 else None}
        reports.append({"checkpoint": str(Path(path).resolve()), "epoch": checkpoint["epoch"],
                        "planner_profile":profile, "window_count":len(windows), "channels":channels, "windows":windows})
        output.write_text(json.dumps({"schema":"maichart-planner-variation-v1", "manifest":str(Path(manifest).resolve()),
            "sampling":"evenly spaced fixed window indices; conditional inputs from chart-global controls; no teacher target",
            "limitations":["Small deterministic panel is not the full validation metric or musical quality.",
                            "The oracle median uses this evaluation sample's labels and is only a diagnostic lower bound for a constant predictor."],
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
    a=p.parse_args(argv)
    if a.limit < 1: p.error("limit must be positive")
    evaluate(a.checkpoints,a.manifest,a.cache_dir,a.output,a.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
