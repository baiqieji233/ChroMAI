"""Train the audio-conditioned maimai chart generator."""

from __future__ import annotations

import argparse
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from maichart.generation.collate import collate_generator
from maichart.generation.conditions import CONDITION_KEYS
from maichart.generation.dataset import GeneratorDataset
from maichart.generation.model import AUDIO_FEATURE_KEYS, ConditionedChartTransformer, ConditionedChartTransformerConfig
from maichart.training.checkpoints import CHECKPOINT_SCHEMA, compare_checkpoint_contract, load_checkpoint
from maichart.training.losses_v25 import compute_v25_losses


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.resume and args.init_checkpoint:
        raise SystemExit("--resume 与 --init-checkpoint 不能同时使用，请选择一种训练方式。")
    _seed_everything(args.seed)

    dataset = GeneratorDataset(
        args.manifest,
        cache_dir=args.cache_dir,
        segment_frames=args.segment_frames,
        random_crop=True,
        include_unusable=args.include_unusable,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_generator,
    )
    data_contract = _data_contract(dataset)
    checkpoint = None
    if args.resume or args.init_checkpoint:
        checkpoint = load_checkpoint(args.resume or args.init_checkpoint, map_location="cpu")

    config = _build_config(args, dataset, checkpoint)
    if checkpoint is not None:
        errors, warnings = compare_checkpoint_contract(
            checkpoint,
            model_config=config.to_dict(),
            data_contract=data_contract,
            mode="resume" if args.resume else "init",
        )
        if errors:
            raise SystemExit("Checkpoint 不兼容，无法训练：\n- " + "\n- ".join(errors))
        for warning in warnings:
            print(f"warning={warning}", flush=True)

    device = _resolve_device(args.device)
    model = ConditionedChartTransformer(config).to(device)
    if checkpoint is not None:
        _load_model_weights(model, checkpoint)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _epoch: 1.0)
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch = 0
    global_step = 0
    best_metric: float | None = None
    if args.resume:
        assert checkpoint is not None
        if not args.reset_optimizer and checkpoint.get("optimizer_state_dict"):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if not args.reset_optimizer and checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if not args.reset_optimizer and checkpoint.get("amp_scaler_state_dict"):
            scaler.load_state_dict(checkpoint["amp_scaler_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0))
        global_step = int(checkpoint.get("global_step", 0))
        best_metric = checkpoint.get("best_metric")
        if not args.reset_optimizer:
            _restore_rng_state(checkpoint.get("rng_state"))
        print(f"resume_checkpoint={args.resume} start_epoch={start_epoch} global_step={global_step}", flush=True)
    elif args.init_checkpoint:
        print(f"init_checkpoint={args.init_checkpoint} mode=finetune optimizer_reset=true", flush=True)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    total_epochs = start_epoch + args.epochs
    print(
        f"samples={len(dataset)} device={device} parameters={_parameter_count(model)} "
        f"segment_frames={args.segment_frames} condition_dim={config.condition_dim} "
        f"epochs={start_epoch + 1}-{total_epochs}",
        flush=True,
    )
    for skipped_name in getattr(dataset.base, "skipped_missing_samples", []):
        print(f"skipped_missing_cache_sample={skipped_name}", flush=True)
    for skipped in getattr(dataset.base, "skipped_invalid_samples", []):
        print(f"skipped_invalid_sample={skipped}", flush=True)

    for epoch in range(start_epoch + 1, total_epochs + 1):
        metrics, steps = _train_epoch(
            model,
            loader,
            optimizer,
            scaler,
            device,
            grad_accumulation=args.grad_accumulation,
            use_amp=use_amp,
            condition_dropout=args.condition_dropout,
        )
        scheduler.step()
        global_step += steps
        print(
            f"epoch={epoch:03d} loss={metrics['loss']:.6f} "
            f"note={metrics['loss_note']:.6f} buttons={metrics['loss_buttons']:.6f} "
            f"type={metrics['loss_type']:.6f} density={metrics['loss_density']:.6f}",
            flush=True,
        )
        best_metric = metrics["loss"] if best_metric is None else min(best_metric, metrics["loss"])
        if args.save_every > 0 and epoch % args.save_every == 0:
            _save_checkpoint(
                checkpoint_dir / f"generator_epoch_{epoch:03d}.pt",
                model, optimizer, scheduler, scaler, epoch, global_step,
                best_metric, args, dataset, data_contract,
            )
    _save_checkpoint(
        checkpoint_dir / "generator_last.pt",
        model, optimizer, scheduler, scaler, total_epochs, global_step,
        best_metric, args, dataset, data_contract,
    )
    return 0


def _train_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    *,
    grad_accumulation: int,
    use_amp: bool,
    condition_dropout: float,
) -> tuple[dict[str, float], int]:
    model.train()
    totals = {"loss": 0.0, "loss_note": 0.0, "loss_buttons": 0.0, "loss_type": 0.0, "loss_density": 0.0}
    batches = 0
    optimizer.zero_grad(set_to_none=True)
    accumulation = max(1, grad_accumulation)
    dropout = max(0.0, min(1.0, condition_dropout))
    for batch_index, batch in enumerate(loader):
        x = batch["x"].to(device)
        condition = batch["condition"].to(device)
        if dropout > 0.0:
            mask = torch.rand(condition.size(0), device=device) < dropout
            condition = condition.masked_fill(mask.unsqueeze(1), 0.0)
        padding_mask = batch["padding_mask"].to(device)
        loss_mask = batch["loss_mask"].to(device)
        targets = {key: value.to(device) for key, value in batch["y"].items()}
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            outputs = model(x, condition, padding_mask=padding_mask)
            losses = compute_v25_losses(outputs, targets, loss_mask)
            loss = losses["loss"] / accumulation
        scaler.scale(loss).backward()
        if (batch_index + 1) % accumulation == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        batches += 1
        for key in totals:
            totals[key] += float(losses[key].detach().cpu())
    if batches and batches % accumulation:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    return {key: value / max(1, batches) for key, value in totals.items()}, batches


def _save_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    global_step: int,
    best_metric: float | None,
    args: argparse.Namespace,
    dataset: GeneratorDataset,
    data_contract: dict[str, Any],
) -> None:
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "checkpoint_kind": "training_checkpoint",
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_metric": best_metric,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "amp_scaler_state_dict": scaler.state_dict(),
        "model_config": asdict(model.config),
        "data_contract": data_contract,
        "note_type_vocab": list(dataset.note_type_vocab),
        "audio_feature_keys": list(AUDIO_FEATURE_KEYS),
        "condition_keys": list(CONDITION_KEYS),
        "args": vars(args),
        "rng_state": _capture_rng_state(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _build_config(args: argparse.Namespace, dataset: GeneratorDataset, checkpoint: dict[str, Any] | None) -> ConditionedChartTransformerConfig:
    values = {
        "input_dim": dataset.input_dim,
        "condition_dim": len(CONDITION_KEYS),
        "num_note_types": dataset.num_note_types,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_layers": args.num_layers,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
        "max_len": max(args.max_len, args.segment_frames),
    }
    stored = checkpoint.get("model_config") if checkpoint else None
    if isinstance(stored, dict):
        # The data-dependent dimensions belong to the current manifest. Keep
        # them here so the contract comparison can reject a 7->77 feature
        # change before model construction; architecture dimensions can be
        # adopted from an existing model for resume/finetune.
        for key in ("d_model", "nhead", "num_layers", "dim_feedforward", "dropout", "max_len"):
            if key in stored:
                values[key] = stored[key]
        if "positional_encoding" in stored:
            values["positional_encoding"] = stored["positional_encoding"]
    return ConditionedChartTransformerConfig(**values)


def _data_contract(dataset: GeneratorDataset) -> dict[str, Any]:
    preprocessing = dataset.base.manifest.get("preprocessing") or {}
    return {
        "division": preprocessing.get("division"),
        "sample_rate": preprocessing.get("sample_rate"),
        "feature_profile": "basic_v1",
        "audio_feature_dim": dataset.input_dim,
        "audio_feature_keys": list(AUDIO_FEATURE_KEYS),
        "condition_keys": list(CONDITION_KEYS),
        "condition_dim": len(CONDITION_KEYS),
        "note_type_vocab": list(dataset.note_type_vocab),
        "target_schema": "frame_v2",
    }


def _load_model_weights(model: torch.nn.Module, checkpoint: dict[str, Any]) -> None:
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except RuntimeError as exc:
        raise SystemExit(f"模型权重与当前模型结构不兼容：{exc}") from exc


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"python": random.getstate(), "torch": torch.get_rng_state()}
    try:
        import numpy as np
        state["numpy"] = np.random.get_state()
    except ImportError:
        pass
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Any) -> None:
    if not isinstance(state, dict):
        return
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("torch") is not None:
        torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])
    try:
        import numpy as np
        if state.get("numpy") is not None:
            np.random.set_state(state["numpy"])
    except ImportError:
        pass


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="V2.5 training manifest JSON path")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--checkpoint-dir", default="checkpoints/generator")
    parser.add_argument("--resume", help="完整 checkpoint：恢复模型、优化器、调度器和 epoch")
    parser.add_argument("--init-checkpoint", help="已有模型：只加载模型权重并重新开始优化")
    parser.add_argument("--reset-optimizer", action="store_true", help="断点模式下也重置优化器和调度器")
    parser.add_argument("--epochs", type=int, default=20, help="本次额外训练的 epoch 数")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--segment-frames", type=int, default=512)
    parser.add_argument("--max-len", type=int, default=4096)
    parser.add_argument("--grad-accumulation", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--condition-dropout", type=float, default=0.15, help="训练时随机隐藏条件，支持生成阶段提高条件强度")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-unusable", action="store_true", help="include samples rejected by parser/QC filters")
    return parser


def _resolve_device(value: str) -> torch.device:
    if value == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


if __name__ == "__main__":
    raise SystemExit(main())
