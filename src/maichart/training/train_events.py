"""Train the exact-event audio-conditioned Transformer, with real validation."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import threading
import time
from functools import partial
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from maichart.generation.conditions import CONDITION_KEYS
from maichart.generation.event_audio import GLOBAL_MEMORY_FRAMES, INPUT_DIM
from maichart.generation.event_ctx import (
    DEFAULT_AUDIO_MARGIN_BEATS,
    DEFAULT_CONTEXT_BEATS,
    DEFAULT_CONTEXT_DROPOUT,
    DEFAULT_CONTEXT_TOKENS,
)
from maichart.generation.event_model import EventConditionedTransformer, EventTransformerConfig, EventVocabulary
from maichart.training.event_checkpoints import EVENT_CHECKPOINT_SCHEMA, atomic_save_checkpoint, compare_event_contract, load_event_checkpoint, vocabulary_signature
from maichart.training.event_dataset import EventDataset, collate_events


# Model family.  `event_v1` is the default; the cross-window family ships as
# `maichart.training.train_event_ctx`, which swaps these three module-level
# hooks before calling `main`.  Everything else -- the loop, the checkpoint
# format, the acceptance integration -- is shared, so a family only has to
# describe what is different about its data.
DATASET_CLASS: type[EventDataset] = EventDataset
TARGET_SCHEMA = "event_v1"
FAMILY_LABEL = "event_v1"
from maichart.training.prewarm_audio import prewarm_audio, resolve_workers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--validation-manifest")
    parser.add_argument("--cache-dir", default="cache/events")
    parser.add_argument("--checkpoint-dir", default="checkpoints/events")
    parser.add_argument("--resume")
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--reset-optimizer", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--segment-beats", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grad-accumulation", type=int, default=4)
    parser.add_argument("--plan-loss-weight", type=float, default=0.1,
                         help="Weight of the window-level planner auxiliary loss.")
    parser.add_argument("--planner-profile", choices=("auto", "legacy_v1", "music_structure_v2"),
                        default="auto", help="Auto: structure v2 for a new model, checkpoint profile for continued training.")
    parser.add_argument("--plan-teacher-mix", type=float, default=0.5,
                        help="Probability-space mix of authored plan targets during structure-v2 training.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--max-steps", type=int, default=0, help="Optional per-epoch smoke-test limit, zero means all batches")
    parser.add_argument("--audio-cache-gb", type=float, default=4.0,
                        help="Resident audio-feature cache budget in GiB (about 3 MB per song, so 4 GiB holds ~1300 "
                             "songs).  Negative means unbounded, 0 disables the cache.  Spilled songs are reloaded "
                             "from their npz cache, so a smaller budget costs I/O rather than recomputation.")
    parser.add_argument("--log-audio-loads", action="store_true",
                        help="Print one line per song the first time its features enter the resident cache.  Off by "
                             "default: the first epoch loads every song, which would flood the log with ~1600 lines "
                             "and push the epoch summaries out of the WebUI's 300-line window.  The default instead "
                             "prints an aggregate line every 25 loads, and each epoch prints the cache state.")
    parser.add_argument("--no-length-bucketing", action="store_true",
                        help="Batch random windows instead of length-bucketed ones.  A batch is padded to its "
                             "longest item and windows span 13..756 tokens, so random batches make the GPU compute "
                             "~1.8x the real tokens; bucketing (the default) groups length-neighbours inside "
                             "shuffled chunks and cuts that to ~1.2x.")
    parser.add_argument("--prefetch-batches", type=int, default=2,
                        help="Assemble batches on a background thread this many batches ahead (0 disables).  With "
                             "num_workers=0 the loader's CPU work is otherwise serialised with the GPU: measured on "
                             "the full corpus the GPU sat at 2-50%% utilisation while it waited.")
    parser.add_argument("--cuda-memory-budget-gb", type=float, default=0.0,
                        help="Return cached blocks to the driver once the peak reservation exceeds this budget, so "
                             "the allocator's cache cannot grow past dedicated memory and spill into shared GPU "
                             "memory (which runs over PCIe: measured 425 ms/batch while spilling versus 228 ms "
                             "healthy, and a 17.2 GiB peak on a 12 GiB card).  0 = auto, 60%% of device memory.")
    parser.add_argument("--cuda-cache-flush-batches", type=int, default=100,
                        help="Check the peak reservation and release cached blocks every N micro-batches (0 "
                             "disables).  Every batch shape gets its own cached block, so without a periodic flush "
                             "the reservation grows monotonically over an epoch.")
    parser.add_argument("--validation-max-steps", type=int, default=0,
                        help="Optional validation smoke-test limit, zero means all validation batches")
    parser.add_argument("--global-frames", type=int, default=GLOBAL_MEMORY_FRAMES,
                        help="Frames in the full-song memory (default %(default)s).  Higher values let the model "
                             "compare the current phrase against earlier parts of the song; the cost is encoder "
                             "attention, since global memory is concatenated to the segment.")
    parser.add_argument("--max-audio-tokens", type=int, default=1024,
                         help="Maximum local audio frames retained by the encoder; larger values preserve timing resolution.")
    parser.add_argument("--prewarm-workers", default="auto",
                         help="Extract missing audio features in this many processes before training "
                              "(default: auto = min(8, cpus-2); 'off' or 0 disables).  Extraction costs "
                              "about 1.4 s per song and would otherwise be paid inside the first epoch "
                              "while the GPU waits.")
    parser.add_argument("--condition-mask-prob", type=float, default=0.5,
                        help="Probability of replacing the chart-derived condition controls with the values "
                             "generation will actually supply (default 0.5 = balanced control/audio training).  Those "
                             "controls are computed from the target chart, so leaving them in lets the model "
                             "read the answer off the condition instead of following the audio.")
    parser.add_argument("--condition-default", type=float, default=0.5,
                        help="The value masked condition controls are set to (0.5 = no information).  It is "
                             "recorded in the data contract: a generation side must feed exactly this value "
                             "in the masked positions, and a resume must use the same one.")
    parser.add_argument("--condition-dropout", type=float, default=0.0,
                        help="Additional CFG dropout during training; masks only condition controls [1:].")
    parser.add_argument("--quality-annotations",
                        help="WebUI-managed chart quality annotations; when set, positive SFT and planner losses are weighted.")
    parser.add_argument("--preference-enabled", action="store_true",
                        help="Use corrected negative-chart examples for a small DPO preference pass after each SFT epoch.")
    parser.add_argument("--preference-weight", type=float, default=0.05,
                        help="Scale of the DPO preference term; chosen-chart SFT anchor stays fixed at 0.1.")
    parser.add_argument("--preference-beta", type=float, default=0.1,
                        help="DPO preference temperature (default 0.1).")
    parser.add_argument("--preference-max-pairs", type=int, default=32,
                        help="Maximum corrected/rejected windows per epoch (default 32).")
    parser.add_argument("--min-difficulty-index", type=int, default=4)
    parser.add_argument("--max-difficulty-index", type=int, default=6)
    parser.add_argument("--min-level", type=float, default=12.0)
    parser.add_argument("--max-level", type=float, default=15.0)
    # ignores these entirely: only a family module that asks for them (through
    # `dataset_kwargs`) turns them into dataset arguments.
    parser.add_argument("--context-beats", type=int, default=DEFAULT_CONTEXT_BEATS,
                        help="Lead-in notes given to each window, in beats (cross-window family only).")
    parser.add_argument("--audio-margin-beats", type=int, default=DEFAULT_AUDIO_MARGIN_BEATS,
                        help="Extra audio sampled before and after each window (cross-window family only).")
    parser.add_argument("--context-tokens", type=int, default=DEFAULT_CONTEXT_TOKENS,
                        help="Token budget for the lead-in block (cross-window family only).")
    parser.add_argument("--overlap-beats", type=int, default=0,
                        help="Shorten the window stride by this many beats, sweeping the song with "
                             "overlapping windows (cross-window family only).")
    parser.add_argument("--slide-oversample", type=int, default=1,
                        help="Repeat slide-containing windows without touch notes this many times "
                             "(cross-window family only). 1 disables oversampling. Windows containing "
                             "touch/touch_hold use --touch-oversample even if they also contain slides. "
                             "Change sampling via --init-checkpoint, not --resume.")
    parser.add_argument("--touch-oversample", type=int, default=1,
                        help="Repeat windows with parsed touch/touch_hold notes this many times "
                             "(cross-window family only). Default 1 keeps their natural frequency. "
                             "This value takes priority in mixed slide/touch windows; factors never "
                             "multiply. Change sampling via --init-checkpoint, not --resume.")
    parser.add_argument("--context-dropout", type=float, default=DEFAULT_CONTEXT_DROPOUT,
                        help="Probability of training a window with an empty lead-in (cross-window family "
                             "only).  A fixed, always-present context teaches the model to copy it; "
                             "randomly removing it is the cheap form of progressive context training.")
    add_family_arguments(parser)
    return parser


def add_family_arguments(parser: argparse.ArgumentParser) -> None:
    """Hook for model families that need extra flags."""


def dataset_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Hook for model families that need extra dataset arguments."""

    return {}


def device_for(value: str) -> torch.device:
    if value == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; choose CPU or install matching CUDA PyTorch.")
    return torch.device("cuda" if value == "auto" and torch.cuda.is_available() else "cpu" if value == "auto" else value)


def _audio_cache_bytes(gigabytes: float) -> int | None:
    """Translate --audio-cache-gb into the dataset's byte budget."""

    if gigabytes is None or gigabytes < 0:
        return None  # unbounded: the pre-LRU behaviour
    return int(gigabytes * 1024 ** 3)


def prewarm_cache_root(dataset) -> Path:
    """Cache directory ``prewarm_audio`` must receive to warm what the dataset reads.

    ``prewarm_audio`` appends the canonical ``audio/`` folder itself, so it takes
    the dataset's cache *root* (see ``test_prewarm_writes_where_training_reads``).
    Passing ``cache_dir / "audio"`` nested a second ``audio/``: prewarm reported
    ``extracted=N`` into an orphan folder, and every song stayed a miss that
    training then extracted inline, one at a time, while the GPU waited -- the
    ``event_audio_ready`` trickle.  Because prewarm's own "already cached?" check
    reads the same wrong path, it reported the corpus as warm on every later run
    and never converged.
    """

    return dataset.cache_dir


def _quality_weighted_mean(values: torch.Tensor, weights: torch.Tensor,
                           normalizer: float = 1.0,
                           valid: torch.Tensor | None = None) -> torch.Tensor:
    """Apply fixed dataset-level quality weights without batch cancellation."""
    weighted = values * weights
    if valid is None:
        denominator = max(values.numel(), 1)
        total = weighted.sum()
    else:
        denominator = valid.sum().clamp_min(1)
        total = weighted[valid].sum()
    return total / denominator / max(float(normalizer), 1e-8)


def run_epoch(model, loader, device, *, optimizer=None, scaler=None, accumulation=1,
              condition_dropout=0.0, condition_default=0.5, plan_loss_weight=0.1, max_steps=0,
              vram_budget_bytes: int = 0, vram_flush_batches: int = 100,
              quality_token_normalizer: float = 1.0, quality_plan_normalizer: float = 1.0,
              plan_teacher_mix: float = 0.0):
    train = optimizer is not None
    model.train(train)
    if train:
        optimizer.zero_grad(set_to_none=True)
    # Statistics stay on the device and are read back once, at the end of the
    # epoch.  Reading them per micro-batch (four GPU->CPU syncs each: isfinite,
    # loss sum, token count, match count) drains the launch queue 4,790 times per
    # epoch, so the GPU never runs ahead of the Python loop.  The sums are kept in
    # float64 on the device, which is the same accumulation the CPU used before.
    loss_total = None
    weighted_loss_total = None
    weighted_token_total = None
    planner_error_total = None
    planner_window_total = None
    token_total = None
    correct_total = None
    position_total = None
    position_correct_total = None
    batches = 0
    # Where did the epoch actually spend its time?  `data_seconds` is the loader
    # (CPU, and with num_workers=0 it does not overlap the GPU); `gpu_seconds`
    # comes from CUDA events recorded around each micro-batch, so the pair tells
    # a stalled GPU apart from a starved one without guessing.  Events are only
    # read after the loop: synchronising per batch would destroy the pipelining
    # this measures.
    data_seconds = 0.0
    gpu_events: list[tuple[Any, Any]] = []
    vram_freed = 0.0
    timing = device.type == "cuda"
    iterator = iter(loader)
    for batch_index in range(len(loader)):
        if max_steps and batch_index >= max_steps:
            break
        data_begin = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            break
        data_seconds += time.perf_counter() - data_begin
        start_event = end_event = None
        if timing:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        batch = {key: value.to(device) for key, value in batch.items()}
        condition = batch["condition"]
        condition_null = batch["condition_null"]
        if train and condition_dropout:
            mask = torch.rand(condition.size(0), device=device) < condition_dropout
            condition = condition.clone()
            condition[mask, 1:] = float(condition_default)
            # Keep the explicit null flag synchronized with the numeric mask.
            # Otherwise dropout creates a third condition state (default-valued
            # controls with condition_null=False) that is absent from both the
            # dataset contract and generation-time CFG.
            condition_null = condition_null.clone()
            condition_null[mask] = True
        with torch.set_grad_enabled(train), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=bool(train and scaler and scaler.is_enabled())):
            structure_v2 = getattr(model.config, "planner_profile", "legacy_v1") == "music_structure_v2"
            planner_kwargs = {}
            if train and structure_v2 and plan_teacher_mix:
                plan_weights = batch.get("quality_plan_weights", torch.ones(condition.size(0), device=device))
                planner_kwargs = {"plan_target": batch["plan_target"],
                                  "plan_teacher_mix": (plan_weights > 0).float() * float(plan_teacher_mix)}
            output = model(batch["audio"], batch["decoder_input"], condition,
                           audio_padding_mask=batch["audio_padding_mask"], decoder_padding_mask=batch["decoder_padding_mask"],
                           global_audio=batch["global_audio"], condition_null=condition_null,
                           plan_audio_mask=batch.get("plan_audio_mask"),
                           return_plan=True, **planner_kwargs)
            logits, plan_pred = output
            valid = ~batch["decoder_padding_mask"] & batch["loss_mask"]
            raw_loss = F.cross_entropy(logits.transpose(1, 2), batch["target"], reduction="none")
            token_weights = batch.get("quality_token_weights")
            if token_weights is None:
                token_weights = torch.ones_like(raw_loss)
            weighted_raw_loss = raw_loss * token_weights
            loss = _quality_weighted_mean(
                raw_loss, token_weights, quality_token_normalizer, valid=valid)
            plan_target = batch["plan_target"]
            continuous = 8 if structure_v2 else 5
            plan_per_window = F.mse_loss(torch.sigmoid(plan_pred[:, :continuous]), plan_target[:, :continuous], reduction="none").mean(dim=-1)
            config_target = torch.clamp((plan_target[:, continuous] * 3.0).round().long(), 0, 3)
            config_per_window = (F.cross_entropy(plan_pred[:, continuous:continuous + 4], config_target, reduction="none")
                                 if plan_pred.size(-1) >= continuous + 4 else
                                 (torch.sigmoid(plan_pred[:, 5]) - plan_target[:, 5]).square())
            sample_weights = batch.get("quality_plan_weights", batch.get("quality_sample_weights"))
            if sample_weights is None:
                sample_weights = torch.ones_like(plan_per_window)
            weighted_plan = _quality_weighted_mean(
                plan_per_window, sample_weights, quality_plan_normalizer)
            weighted_config = _quality_weighted_mean(
                config_per_window, sample_weights, quality_plan_normalizer)
            loss = loss + float(plan_loss_weight) * (weighted_plan + 0.25 * weighted_config)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite event loss; no successful checkpoint will be saved.")
        if train:
            # Normalize the last incomplete accumulation group correctly.
            batch_limit = min(len(loader), max_steps) if max_steps else len(loader)
            group_start = (batch_index // accumulation) * accumulation
            divisor = min(accumulation, batch_limit - group_start)
            scaler.scale(loss / divisor).backward()
            if (batch_index + 1) % accumulation == 0 or batch_index + 1 == batch_limit:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        batch_loss = raw_loss[valid].sum().detach().double()
        loss_total = batch_loss if loss_total is None else loss_total + batch_loss
        valid_count = valid.sum()
        token_total = valid_count if token_total is None else token_total + valid_count
        batch_weighted_loss = weighted_raw_loss[valid].sum().detach().double()
        weighted_loss_total = batch_weighted_loss if weighted_loss_total is None else weighted_loss_total + batch_weighted_loss
        valid_weight = token_weights[valid].sum().detach().double()
        weighted_token_total = valid_weight if weighted_token_total is None else weighted_token_total + valid_weight
        predictions = logits.argmax(-1)
        matches = ((predictions == batch["target"]) & valid).sum()
        correct_total = matches if correct_total is None else correct_total + matches
        plan_valid = sample_weights > 0
        channel_errors = ((torch.sigmoid(plan_pred[:, :continuous]) - plan_target[:, :continuous]).abs()
                          * plan_valid.unsqueeze(-1)).sum(dim=0).detach().double()
        planner_error_total = channel_errors if planner_error_total is None else planner_error_total + channel_errors
        plan_count = plan_valid.sum().detach()
        planner_window_total = plan_count if planner_window_total is None else planner_window_total + plan_count
        if hasattr(model, "position_ids"):
            # A lookup table turns this from a sort-and-search per batch into a
            # single gather, and the ids are a fixed set for the whole run.
            lookup = getattr(model, "position_lookup", None)
            if lookup is None or lookup.numel() != getattr(model.token_embedding, "num_embeddings", 0):
                lookup = torch.zeros(model.token_embedding.num_embeddings, dtype=torch.bool,
                                     device=batch["target"].device)
                lookup[model.position_ids] = True
                model.position_lookup = lookup
            positions = lookup[batch["target"]]
            position_valid = positions & valid
            position_count = position_valid.sum()
            position_total = position_count if position_total is None else position_total + position_count
            position_matches = ((predictions == batch["target"]) & position_valid).sum()
            position_correct_total = (position_matches if position_correct_total is None
                                      else position_correct_total + position_matches)
        if end_event is not None:
            end_event.record()
            gpu_events.append((start_event, end_event))
        batches += 1
        # Keep the allocator's cache inside the budget while the epoch runs: every
        # batch shape gets its own cached block, so without this the reservation
        # grows monotonically until it spills into shared GPU memory.
        if vram_flush_batches and batch_index and batch_index % vram_flush_batches == 0:
            vram_freed += enforce_vram_budget(device, vram_budget_bytes, reset_peak=True)
    # One synchronisation for the whole epoch instead of four per micro-batch.
    total_loss = float(loss_total.item()) if loss_total is not None else 0.0
    total_weighted_loss = float(weighted_loss_total.item()) if weighted_loss_total is not None else 0.0
    total_weighted_tokens = float(weighted_token_total.item()) if weighted_token_total is not None else 0.0
    total_tokens = int(token_total.item()) if token_total is not None else 0
    correct = int(correct_total.item()) if correct_total is not None else 0
    position_correct = int(position_correct_total.item()) if position_correct_total is not None else 0
    position_count = int(position_total.item()) if position_total is not None else 0
    gpu_seconds = 0.0
    if gpu_events and timing:
        torch.cuda.synchronize()
        gpu_seconds = sum(start.elapsed_time(end) for start, end in gpu_events) / 1000.0
    plan_names = ("count", "density", "chord", "slide", "last", "touch", "hold", "intensity")
    plan_errors = (planner_error_total.cpu().tolist() if planner_error_total is not None else [])
    plan_count = int(planner_window_total.item()) if planner_window_total is not None else 0
    return {"loss": total_loss / max(total_tokens, 1),
            "quality_weighted_token_loss": total_weighted_loss / max(total_weighted_tokens, 1e-8),
            "quality_weighted_tokens": total_weighted_tokens,
            "planner_mae": {key: float(value) / max(plan_count, 1)
                            for key, value in zip(plan_names, plan_errors)},
            "planner_supervised_windows": plan_count,
            "token_accuracy": correct / max(total_tokens, 1),
            "position_accuracy": position_correct / max(position_count, 1), "tokens": total_tokens,
            "batches": batches, "data_seconds": round(data_seconds, 3),
            "gpu_seconds": round(gpu_seconds, 3), "vram_freed_gib": round(vram_freed, 2)}


class PrefetchLoader:
    """Iterate a loader on a background thread so CPU prep overlaps the GPU.

    ``num_workers`` cannot be used here: on Windows each worker is spawned and
    rebuilds the dataset (charts, conditions, and the resident audio cache) from
    scratch, which costs minutes and gigabytes per worker.  The per-sample work is
    numpy, which releases the GIL, so one prefetch thread is enough to keep the
    GPU fed while the next batch is assembled.
    """

    def __init__(self, loader: DataLoader, depth: int = 2) -> None:
        self.loader = loader
        self.depth = max(1, int(depth))

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self):
        queue: Any = __import__("queue").Queue(maxsize=self.depth)
        sentinel = object()
        # Per-iteration stop flag: keeping it on the instance meant the second
        # epoch saw it already set and produced nothing at all (batches=0), i.e.
        # training silently stopped doing work after the first epoch.
        stop = threading.Event()

        def produce() -> None:
            try:
                for item in self.loader:
                    while not stop.is_set():
                        try:
                            queue.put(item, timeout=0.1)
                            break
                        except Exception:  # queue.Full
                            continue
                    if stop.is_set():
                        return
            finally:
                try:
                    queue.put(sentinel, timeout=2.0)
                except Exception:
                    pass

        thread = threading.Thread(target=produce, name="maichart-prefetch", daemon=True)
        thread.start()
        try:
            while True:
                item = queue.get()
                if item is sentinel:
                    return
                yield item
        finally:
            stop.set()
            thread.join(timeout=5.0)


def vram_budget_bytes(requested_gb: float, device: Any) -> int:
    """Bytes the allocator may keep reserved before cached blocks are returned.

    The caching allocator never gives memory back on its own, and with variable
    batch shapes it keeps a block for every distinct shape it has seen.  On a
    laptop GPU that cache grows until it passes dedicated memory and Windows
    starts backing it with shared GPU memory over PCIe: throughput then collapses
    (measured 3.1 s per micro-batch while spilling versus 0.22 s healthy).
    """

    if requested_gb and requested_gb > 0:
        return int(float(requested_gb) * 1024 ** 3)
    if getattr(device, "type", "") != "cuda" or not torch.cuda.is_available():
        return 0
    try:
        total = int(torch.cuda.get_device_properties(device).total_memory)
    except Exception:  # noqa: BLE001 - a guard is not worth failing the run
        return 0
    # 60%: the desktop and other applications hold VRAM too, and the card's
    # dedicated memory is what must not be exceeded.
    return int(total * 0.6)


def device_memory_report(device: Any) -> dict[str, float]:
    """Allocated/reserved/peak VRAM in GiB (zeros off CUDA)."""

    if getattr(device, "type", "") != "cuda" or not torch.cuda.is_available():
        return {"allocated_gib": 0.0, "reserved_gib": 0.0, "peak_gib": 0.0}
    return {
        "allocated_gib": round(torch.cuda.memory_allocated() / 1024 ** 3, 2),
        "reserved_gib": round(torch.cuda.memory_reserved() / 1024 ** 3, 2),
        "peak_gib": round(torch.cuda.max_memory_reserved() / 1024 ** 3, 2),
    }


def enforce_vram_budget(device: Any, budget_bytes: int, *, reset_peak: bool = False) -> float:
    """Release cached blocks once the *peak* reservation passes the budget.

    Checking ``memory_reserved()`` alone is not enough: every batch has a
    different shape and the caching allocator keeps a block for each one, but the
    instantaneous value sampled between steps sits far below the peak reached
    inside forward/backward.  Measured: checks saw 4.33 GiB while
    ``max_memory_reserved`` hit 17.21 GiB on a 12 GiB card, and Windows then
    backed the excess with shared GPU memory over PCIe (425 ms/batch instead of
    228 ms, and the original run's 38-minute first epoch).
    """

    if not budget_bytes or getattr(device, "type", "") != "cuda" or not torch.cuda.is_available():
        return 0.0
    peak = torch.cuda.max_memory_reserved()
    if peak <= budget_bytes:
        # Under budget: leave the peak window running so the next check still
        # measures the highest reservation since the last successful flush.
        return 0.0
    reserved = torch.cuda.memory_reserved()
    torch.cuda.empty_cache()
    if reset_peak:
        # After the flush, so the next window starts from live usage rather than
        # the stale pre-flush value.
        torch.cuda.reset_peak_memory_stats()
    return round((reserved - torch.cuda.memory_reserved()) / 1024 ** 3, 2)


def apply_vram_hard_cap(device: Any, budget_bytes: int) -> int:
    """Forbid the allocator from going past the budget (shared memory instead).

    Releasing cached blocks is only a best effort: if one batch's peak already
    exceeds the budget, the cache is legitimately that large and a flush cannot
    help.  A hard cap turns that silent disaster -- Windows backing the overflow
    with shared GPU memory over PCIe, measured 425 ms/micro-batch instead of 228,
    and a 17.2 GiB reservation on a 12 GiB card -- into a loud, actionable
    OutOfMemoryError telling the user to lower the batch size.
    """

    if not budget_bytes or getattr(device, "type", "") != "cuda" or not torch.cuda.is_available():
        return 0
    try:
        total = int(torch.cuda.get_device_properties(device).total_memory)
        if total <= 0:
            return 0
        fraction = min(0.95, max(0.05, budget_bytes / total))
        # set_per_process_memory_fraction rejects a bare `cuda` device: it needs an
        # index, and silently skipping it here left the run uncapped.
        index = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.set_per_process_memory_fraction(fraction, index)
    except Exception as error:  # noqa: BLE001 - the guard must not break the run
        print(f"cuda_memory_cap skipped error={type(error).__name__}: {error}", flush=True)
        return 0
    return int(total * fraction)


def segment_lengths(dataset: EventDataset) -> list[int]:
    """Per-window token length, cheap enough to compute before the first epoch.

    Both families store their token material in the segment tuple, so no audio is
    touched.  The cross-window family's composite target is
    ``<bos> + lead-in + window``; the plain family's is the window itself.
    """

    lengths: list[int] = []
    for segment in dataset.segments:
        if len(segment) >= 7:
            _sample, _start, _end, window, prefix, _prefix_len, _offset = segment[:7]
            lengths.append(len(prefix) + len(window))
        else:
            lengths.append(len(segment[3]))
    return lengths


class LengthBucketedBatchSampler(torch.utils.data.Sampler[list[int]]):
    """Batch windows of similar length together.

    A batch is padded to its longest item, and measured windows span 13..756
    tokens, so uniform random batching makes the GPU compute ~2.2x the real
    tokens -- and sets the peak activation memory from the unluckiest mix.  Here
    the indices are sorted by length, cut into chunks of ``chunk_batches``
    batches, and the chunks are shuffled (items shuffled inside each chunk): a
    batch only ever mixes length-neighbours, while the order over a chunk stays
    random.
    """

    def __init__(self, lengths: Sequence[int], batch_size: int, *, shuffle: bool,
                 seed: int, chunk_batches: int = 64) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.lengths = list(lengths)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.chunk_batches = max(1, int(chunk_batches))

    def __len__(self) -> int:
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        order = list(range(len(self.lengths)))
        count = len(order)
        if count == 0:
            return
        # The chunk must stay smaller than the corpus or bucketing degenerates
        # into plain random batching (one chunk = nothing to shuffle), and small
        # corpora would never benefit: keep at least ~8 chunks while capping the
        # chunk so lengths inside it stay close.
        chunk = min(self.batch_size * self.chunk_batches,
                    max(self.batch_size * 2, count // 8))
        order.sort(key=lambda index: self.lengths[index])
        if not self.shuffle:
            for start in range(0, count, self.batch_size):
                yield order[start:start + self.batch_size]
            return
        chunks = [order[start:start + chunk] for start in range(0, count, chunk)]
        # Draw the per-epoch order from the global RNG: the trainer seeds it and
        # checkpoints it, so every epoch reshuffles and --resume continues the
        # sequence instead of replaying an identical order forever.
        rng = random.Random(random.getrandbits(64))
        rng.shuffle(chunks)
        for part in chunks:
            rng.shuffle(part)
            for start in range(0, len(part), self.batch_size):
                yield part[start:start + self.batch_size]


def build_loader(dataset: EventDataset, *, batch_size: int, shuffle: bool, seed: int,
                 collate_fn: Any, bucket: bool = True, prefetch: int = 2) -> Any:
    """DataLoader that optionally batches by window length, with CPU/GPU overlap.

    Bucketing shortens padding (measured 1.5x less padded work on the full
    corpus); the prefetch thread then keeps the GPU busy while the next batch is
    assembled.  Neither changes the sample set an epoch sees.
    """

    if bucket:
        sampler = LengthBucketedBatchSampler(segment_lengths(dataset), batch_size,
                                             shuffle=shuffle, seed=seed)
        loader: DataLoader = DataLoader(dataset, batch_sampler=sampler, num_workers=0,
                                        collate_fn=collate_fn)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            num_workers=0, collate_fn=collate_fn)
    return PrefetchLoader(loader, depth=prefetch) if prefetch > 0 else loader


def atom_catalog(dataset: EventDataset) -> list[str]:
    from maichart.notes import _split_top_level_slashes
    return sorted({atom for ref in dataset.samples for point in ref.chart.points for atom in _split_top_level_slashes("".join(point.note_text.split()))})


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.resume and args.init_checkpoint:
        raise ValueError("Choose either resume or init-checkpoint.")
    if args.epochs <= 0 or args.batch_size <= 0 or args.grad_accumulation <= 0:
        raise ValueError("epochs, batch-size and grad-accumulation must be positive.")
    if args.preference_enabled and not args.quality_annotations:
        raise ValueError("preference training requires WebUI-managed --quality-annotations.")
    if (not math.isfinite(args.preference_weight) or not 0.0 <= args.preference_weight <= 1.0
            or not math.isfinite(args.preference_beta) or args.preference_beta <= 0
            or args.preference_max_pairs <= 0):
        raise ValueError("preference weight must be 0..1, beta positive and max pairs positive.")
    if not math.isfinite(args.plan_teacher_mix) or not 0.0 <= args.plan_teacher_mix <= 1.0:
        raise ValueError("plan-teacher-mix must be a finite value in 0..1")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    checkpoint = load_event_checkpoint(args.resume or args.init_checkpoint) if args.resume or args.init_checkpoint else None
    stored_planner = checkpoint.get("model_config", {}).get("planner_profile", "legacy_v1") if checkpoint else None
    planner_profile = (stored_planner or "music_structure_v2") if args.planner_profile == "auto" else args.planner_profile
    if checkpoint and planner_profile != stored_planner:
        raise ValueError("音乐结构规划器的训练目标和模型头已改变；请选择 fresh 重新训练，原 checkpoint 可继续用其原规划器。")
    vocabulary = EventVocabulary.from_dict(checkpoint["vocabulary"]) if checkpoint else None
    quality_payload = None
    quality_normalizers = {"token": 1.0, "planner": 1.0, "positive_windows": 0.0}
    if args.quality_annotations:
        from maichart.training.quality_annotations import annotations_fingerprint, load_quality_annotations

        quality_payload = load_quality_annotations(args.quality_annotations)
    # Audio features are content-addressed and can be extracted lazily.  This
    # avoids decoding every song before the first optimizer step, which is
    # especially important for a large manifest and for smoke runs with
    # --max-steps.
    family = dataset_kwargs(args)
    shared = {"condition_mask_prob": args.condition_mask_prob,
              "condition_default": args.condition_default,
              "min_difficulty_index": args.min_difficulty_index,
              "max_difficulty_index": args.max_difficulty_index,
              "min_level": args.min_level, "max_level": args.max_level}
    dataset = DATASET_CLASS(args.manifest, cache_dir=args.cache_dir, segment_beats=args.segment_beats,
                            max_tokens=args.max_tokens, vocabulary=vocabulary, eager_audio=False,
                            audio_cache_bytes=_audio_cache_bytes(args.audio_cache_gb),
                            global_frames=args.global_frames, log_audio_loads=args.log_audio_loads,
                            audio_cache_label="train",
                            quality_annotations=args.quality_annotations,
                            planner_profile=planner_profile,
                            **shared, **family)
    if quality_payload is not None:
        quality_normalizers = dataset.quality_normalizers()
        if quality_normalizers["positive_windows"] <= 0:
            raise ValueError("质量标注把所有训练窗口都排除或标成反例；请保留正向训练内容。")
    vocabulary = dataset.vocabulary
    preference_pairs, preference_audit, reference_scores = [], {"pairs": 0}, {}
    if args.preference_enabled:
        if checkpoint is None:
            raise ValueError("反面偏好训练需要已训练 checkpoint 作为参考；fresh 阶段先做质量加权正向训练，再以 finetune 启用修正版配对。")
        from maichart.training.event_preferences import build_preference_pairs

        preference_pairs, preference_audit = build_preference_pairs(
            dataset, quality_payload, max_pairs=max(1, args.preference_max_pairs) * 4,
            seed=int(checkpoint.get("args", {}).get("seed", args.seed)) if args.resume else args.seed,
        )
        if not preference_pairs:
            skipped = preference_audit.get("skipped", [])
            first_reason = skipped[0].get("reason") if skipped else "没有反例配对窗口或没有已保存修正版"
            raise ValueError(
                "反面偏好训练已启用，但当前素材标注无法形成可训练的修正版配对。"
                f" 未配对反例={preference_audit.get('negative_without_correction', 0)}；"
                f"跳过窗口={len(skipped)}；示例原因={first_reason}。"
                "请在 WebUI 的‘训练素材’中检查并重新保存修正版配对。"
            )
    # The validation dataset must be deterministic: the context-family
    # constructor draws its own RNG for context_dropout, and leaving it enabled
    # here silently drops the lead-in on ~15% of the validation windows, which
    # both inflates the reported loss and lets the early-stopping score wander
    # between otherwise identical measurements.
    validation_family = dict(family)
    validation_family["context_dropout"] = 0.0
    # Evaluate the source distribution, rather than duplicating the very note
    # types the training sampler promotes. These counts are logged separately.
    for sampling_key in ("slide_oversample", "touch_oversample"):
        if sampling_key in validation_family:
            validation_family[sampling_key] = 1
    validation = DATASET_CLASS(args.validation_manifest, cache_dir=args.cache_dir, segment_beats=args.segment_beats,
                               max_tokens=args.max_tokens, vocabulary=vocabulary, eager_audio=False,
                               audio_cache_bytes=_audio_cache_bytes(args.audio_cache_gb),
                               global_frames=args.global_frames, log_audio_loads=args.log_audio_loads,
                               audio_cache_label="validation",
                               planner_profile=planner_profile,
                               **shared,
                               **validation_family) if args.validation_manifest else None
    if validation:
        train_sources = {str(ref.audio_path) for ref in dataset.samples}
        overlap = train_sources & {str(ref.audio_path) for ref in validation.samples}
        if overlap:
            raise ValueError(f"训练与验证集存在歌曲泄漏：{list(overlap)[:3]}")
    # Pay the (one-off, CPU-only) feature extraction up front, in parallel and
    # while the GPU is idle, instead of letting the data loader pay it one song
    # at a time during the first epoch.  The cache is content-addressed, so this
    # is a no-op once it is warm.
    prewarm_workers = resolve_workers(args.prewarm_workers)
    if prewarm_workers > 0:
        paths = [str(ref.audio_path) for ref in dataset.samples]
        if validation is not None:
            paths += [str(ref.audio_path) for ref in validation.samples]
        print(f"prewarm_audio_start songs={len(set(paths))} workers={prewarm_workers}", flush=True)
        prewarm = prewarm_audio(
            paths, cache_dir=prewarm_cache_root(dataset), sample_rate=dataset.sample_rate,
            hop_length=dataset.hop_length, workers=prewarm_workers,
            log=lambda text: print(text, flush=True),
        )
        print(f"prewarm_audio_done extracted={prewarm['extracted']} cached={prewarm['cached']} "
              f"failed={prewarm['failed']} seconds={prewarm['seconds']:.1f}", flush=True)
        for failure in prewarm["failures"][:5]:
            print(f"prewarm_audio_failure {failure['path']}: {failure['error']}",
                  file=sys.stderr, flush=True)
    contract = {"target_schema": TARGET_SCHEMA, "feature_profile": "event_audio_v2_rhythm", "input_dim": INPUT_DIM,
                "sample_rate": dataset.sample_rate, "hop_length": dataset.hop_length, "segment_beats": args.segment_beats,
                "max_tokens": args.max_tokens, "vocabulary_signature": vocabulary_signature(vocabulary),
                "dataset_signature": dataset.signature, "condition_keys": list(CONDITION_KEYS),
                "max_audio_tokens": args.max_audio_tokens,
                "difficulty_filter": {"min_index": args.min_difficulty_index,
                                      "max_index": args.max_difficulty_index,
                                      "min_level": args.min_level, "max_level": args.max_level}}
    if family:
        # Recorded and compared: two runs of the same family with different
        # context settings are not the same experiment, and resuming across them
        # would silently mix window layouts.
        contract["family_params"] = dataset.family_params()
    # How much of the condition vector was withheld is part of the experiment:
    # a model trained with the answer's statistics is a different model.
    contract["condition_params"] = {"mask_prob": float(args.condition_mask_prob),
                                    "dropout": float(args.condition_dropout),
                                    "default": float(args.condition_default),
                                    "null_embedding": True,
                                    "null_policy": "masked_condition_controls"}
    if planner_profile == "music_structure_v2":
        contract["planner_params"] = {"profile": planner_profile, "plan_dim": 12,
                                      "teacher_mix": args.plan_teacher_mix,
                                      "target": "chart_authored_local_workload_v2"}
    if quality_payload is not None:
        contract["quality_params"] = {
            "schema": quality_payload.get("schema"),
            "annotations_fingerprint": dataset.quality_annotation_fingerprint,
            "loss_unit": "token_fixed_normalizer_partial_planner_mask_v2",
            "preference_enabled": bool(args.preference_enabled),
            "preference_weight": float(args.preference_weight),
            "preference_beta": float(args.preference_beta),
            "preference_max_pairs": int(args.preference_max_pairs),
        }
    if checkpoint:
        compare_event_contract(checkpoint, contract, mode="resume" if args.resume else "init")
    training_origin = checkpoint.get("training_origin") if checkpoint and args.resume else None
    if args.init_checkpoint:
        old_family = checkpoint.get("data_contract", {}).get("family_params", {})
        new_family = contract.get("family_params", {})
        sampling_changes = {
            key: {"source": old_family.get(key), "current": new_family.get(key)}
            for key in ("slide_oversample", "touch_oversample", "oversample_policy")
            if old_family.get(key) != new_family.get(key)
        }
        training_origin = {
            "mode": "finetune", "checkpoint": str(Path(args.init_checkpoint).resolve()),
            "source_epoch": checkpoint.get("epoch"),
            "source_global_step": checkpoint.get("global_step"),
            "optimizer_reset": True, "sampling_changes": sampling_changes,
        }
        print(f"init_checkpoint={args.init_checkpoint} mode=finetune source_epoch={checkpoint.get('epoch')} "
              f"optimizer_reset=true sampling_changes={json.dumps(sampling_changes)}", flush=True)
    config = EventTransformerConfig(**checkpoint["model_config"]) if checkpoint else EventTransformerConfig(
        input_dim=INPUT_DIM, vocab_size=len(vocabulary), d_model=args.d_model, nhead=args.nhead,
        num_layers=args.num_layers, dim_feedforward=args.dim_feedforward, dropout=args.dropout,
        max_tokens=args.max_tokens, max_audio_tokens=args.max_audio_tokens,
        global_audio_tokens=args.global_frames, planner_profile=planner_profile,
        plan_dim=12 if planner_profile == "music_structure_v2" else 9)
    device = device_for(args.device)
    model = EventConditionedTransformer(config).to(device)
    model.position_ids = torch.tensor([index for index, token in enumerate(vocabulary.tokens) if token.startswith("AT:")], device=device)
    if checkpoint:
        compatibility = model.load_weights_compatible(checkpoint["model_state_dict"])
        # A finetune/init run should learn the current planner even when the
        # source checkpoint predates planner injection; generation of that old
        # checkpoint remains planner-disabled in load_event_model().
        model.planner_enabled = True
        if compatibility["legacy"]:
            print("checkpoint_condition_fusion=per_frame_bias（旧结构 checkpoint：按它训练时的条件注入方式继续；"
                  "下一轮完整重训后才会使用 condition token 结构）", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and not args.no_amp)
    start_epoch, global_step, best, stale = 0, 0, float("inf"), 0
    if args.resume:
        start_epoch, global_step = checkpoint["epoch"], checkpoint["global_step"]
        best = checkpoint.get("best_metric", best)
        stale = checkpoint.get("stale_epochs", 0)
        if not args.reset_optimizer:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            scaler.load_state_dict(checkpoint["amp_scaler_state_dict"])
        rng = checkpoint.get("rng_state", {})
        if rng:
            random.setstate(rng["python"]); np.random.set_state(rng["numpy"]); torch.set_rng_state(rng["torch"])
            if device.type == "cuda" and rng.get("cuda"):
                torch.cuda.set_rng_state_all(rng["cuda"])
    checkpoint_dir = Path(args.checkpoint_dir); checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "dataset_audit.json").write_text(json.dumps({"train": dataset.audit(), "validation": validation.audit() if validation else None}, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.preference_enabled:
        from maichart.training.event_preferences import precompute_reference_scores

        saved_reference = checkpoint.get("preference_reference_scores", {}) if checkpoint and args.resume else {}
        if args.resume and saved_reference:
            reference_scores = {pair.key: saved_reference[pair.key]
                                for pair in preference_pairs if pair.key in saved_reference}
            if len(reference_scores) != len(preference_pairs):
                raise ValueError("断点保存的偏好参考分数与当前训练配对不完整；请使用 finetune 重建参考模型。")
        elif args.resume:
            raise ValueError("偏好模式无法从旧断点恢复冻结参考分数；请使用 finetune 启动新的偏好阶段。")
        else:
            reference_scores = precompute_reference_scores(model, preference_pairs, dataset, device=device)
        print(f"quality_preference_preflight pairs={preference_audit['pairs']} "
              f"negative_without_correction={preference_audit['negative_without_correction']} "
              f"skipped={len(preference_audit['skipped'])} reference_scores={len(reference_scores)}", flush=True)
        (checkpoint_dir / "preference_audit.json").write_text(
            json.dumps({"audit": preference_audit, "reference_scores": reference_scores}, ensure_ascii=False, indent=2),
            encoding="utf-8")
    else:
        preference_pairs, preference_audit, reference_scores = [], {"pairs": 0}, {}
    loader = build_loader(dataset, batch_size=args.batch_size, shuffle=True, seed=args.seed,
                          collate_fn=partial(collate_events, pad_id=vocabulary.pad_id),
                          bucket=not args.no_length_bucketing, prefetch=args.prefetch_batches)
    val_loader = build_loader(validation, batch_size=args.batch_size, shuffle=False, seed=args.seed,
                              collate_fn=partial(collate_events, pad_id=vocabulary.pad_id),
                              bucket=not args.no_length_bucketing,
                              prefetch=args.prefetch_batches) if validation else None
    catalog = sorted(set(atom_catalog(dataset)) | set(checkpoint.get("atom_catalog", []) if checkpoint else []))
    print(f"samples={len(dataset.samples)} segments={len(dataset)} validation_segments={len(validation) if validation else 0} device={device} parameters={sum(p.numel() for p in model.parameters())} target_schema={TARGET_SCHEMA}", flush=True)
    # Print the effective seed so a good draw can be replayed; the WebUI draws a
    # fresh one for every run unless the seed field names a specific value.
    print(f"seed={args.seed} lr={args.lr} batch={args.batch_size} accumulation={args.grad_accumulation} "
          f"dropout={args.dropout} weight_decay={args.weight_decay} max_tokens={args.max_tokens} "
          f"segment_beats={args.segment_beats} input_dim={INPUT_DIM} global_frames={args.global_frames} "
          f"d_model={config.d_model} layers={config.num_layers} heads={config.nhead}", flush=True)
    print(f"dataset_audio_cache {dataset.audio_cache_summary()}", flush=True)
    budget_bytes = vram_budget_bytes(args.cuda_memory_budget_gb, device)
    if budget_bytes:
        applied = apply_vram_hard_cap(device, budget_bytes)
        print(f"cuda_memory_budget bytes={budget_bytes} "
              f"({budget_bytes / 1024 ** 3:.1f} GiB, requested={args.cuda_memory_budget_gb or 'auto'}) "
              f"hard_cap={applied / 1024 ** 3:.1f} GiB "
              f"flush_every={args.cuda_cache_flush_batches}", flush=True)
    if validation is None:
        print("warning=未提供独立验证集；event_best.pt仅按训练loss保存，无法证明泛化质量。", flush=True)
    for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
        epoch_start = time.perf_counter()
        metrics = run_epoch(model, loader, device, optimizer=optimizer, scaler=scaler,
                            accumulation=args.grad_accumulation, condition_dropout=args.condition_dropout,
                            condition_default=args.condition_default, plan_loss_weight=args.plan_loss_weight,
                            max_steps=args.max_steps,
                            vram_budget_bytes=budget_bytes, vram_flush_batches=args.cuda_cache_flush_batches,
                            quality_token_normalizer=quality_normalizers["token"],
                            quality_plan_normalizer=quality_normalizers["planner"],
                            plan_teacher_mix=args.plan_teacher_mix)
        preference_metrics = {"pairs": 0.0, "loss": 0.0, "win_rate": 0.0}
        if preference_pairs and args.preference_enabled:
            from maichart.training.event_preferences import run_preference_epoch

            preference_metrics = run_preference_epoch(
                model, preference_pairs, dataset, device, optimizer, scaler,
                max_pairs=args.preference_max_pairs, beta=args.preference_beta,
                weight=args.preference_weight, reference_scores=reference_scores, seed=args.seed + epoch,
            )
        train_seconds = time.perf_counter() - epoch_start
        validation_start = time.perf_counter()
        val_metrics = run_epoch(model, val_loader, device, plan_loss_weight=args.plan_loss_weight,
                                max_steps=args.validation_max_steps, vram_budget_bytes=budget_bytes,
                                vram_flush_batches=args.cuda_cache_flush_batches) if val_loader else None
        validation_seconds = time.perf_counter() - validation_start
        score = (val_metrics or metrics)["loss"]
        previous_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(score)
        current_lr = optimizer.param_groups[0]["lr"]
        lr_reduced = current_lr < previous_lr
        if lr_reduced:
            stale = 0
        global_step += math.ceil(metrics["batches"] / args.grad_accumulation)
        global_step += int(preference_metrics.get("pairs", 0.0))
        improved = score < best
        if improved: best, stale = score, 0
        else: stale += 1
        per_batch_ms = train_seconds / max(metrics["batches"], 1) * 1000
        record = {"epoch": epoch, "wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                  "train": metrics, "preference": preference_metrics,
                  "validation": val_metrics, "lr": optimizer.param_groups[0]["lr"],
                  "selection_metric": "validation_loss" if validation else "training_loss",
                  "lr_reduced": bool(lr_reduced),
                  "train_seconds": round(train_seconds, 2),
                  "validation_seconds": round(validation_seconds, 2),
                  "ms_per_batch": round(per_batch_ms, 2)}
        if args.quality_annotations:
            record["quality"] = {"annotations_fingerprint": dataset.quality_annotation_fingerprint,
                                  "token_normalizer": quality_normalizers["token"],
                                  "planner_normalizer": quality_normalizers["planner"],
                                  "positive_windows": quality_normalizers["positive_windows"]}
        # The per-batch time is the throughput readout, and data/gpu split says
        # whether the GPU is starved (data_seconds >> gpu_seconds) or stalled
        # (gpu_seconds ~= train_seconds with low utilisation).
        print(f"epoch={epoch:03d} loss={metrics['loss']:.6f} token_accuracy={metrics['token_accuracy']:.4f} "
              f"val_loss={val_metrics['loss'] if val_metrics else 'unavailable'} "
              f"seconds={train_seconds:.1f} ms_per_batch={per_batch_ms:.1f} batches={metrics['batches']} "
              f"data_s={metrics.get('data_seconds', 0.0):.1f} gpu_s={metrics.get('gpu_seconds', 0.0):.1f}",
              flush=True)
        if planner_profile == "music_structure_v2":
            planner_metrics = (val_metrics or metrics).get("planner_mae", {})
            print("planner_epoch=" + str(epoch) + " source=" + ("validation" if val_metrics else "training")
                  + " " + " ".join(f"{key}_mae={value:.4f}" for key, value in planner_metrics.items()), flush=True)
        if args.quality_annotations:
            print(f"quality_epoch={epoch:03d} weighted_token_loss={metrics['quality_weighted_token_loss']:.6f} "
                  f"weighted_tokens={metrics['quality_weighted_tokens']:.0f} "
                  f"preference_pairs={preference_metrics.get('pairs', 0):.0f} "
                  f"preference_win_rate={preference_metrics.get('win_rate', 0):.3f}", flush=True)
        # The audio cache is the one lazy cost that can silently dominate an epoch
        # (a spilled song is re-decoded from its npz mid-epoch).  Printing it per
        # epoch makes thrash visible: rising `spills` means --audio-cache-gb is too
        # small for this corpus.
        print(f"dataset_audio_cache epoch={epoch:03d} {dataset.audio_cache_summary()}", flush=True)
        # VRAM is the other silent killer: once the allocator's cache passes
        # dedicated memory, Windows backs it with shared GPU memory over PCIe and
        # every kernel slows down by an order of magnitude.
        freed = enforce_vram_budget(device, budget_bytes)
        memory = device_memory_report(device)
        print(f"cuda_memory epoch={epoch:03d} allocated={memory['allocated_gib']:.2f} "
              f"reserved={memory['reserved_gib']:.2f} peak={memory['peak_gib']:.2f} GiB "
              f"budget={budget_bytes / 1024 ** 3:.1f} freed={freed:.2f}", flush=True)
        record["cuda_memory"] = {**memory, "budget_gib": round(budget_bytes / 1024 ** 3, 2),
                                 "freed_gib": freed}
        with (checkpoint_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle: handle.write(json.dumps(record) + "\n")
        payload = {"schema": EVENT_CHECKPOINT_SCHEMA, "checkpoint_kind": "training_checkpoint", "epoch": epoch, "global_step": global_step,
                   "best_metric": best, "stale_epochs": stale, "metrics": record,
                   "model_config": config.to_dict(), "data_contract": contract, "vocabulary": vocabulary.to_dict(), "atom_catalog": catalog,
                   "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
                   "training_origin": training_origin,
                   "amp_scaler_state_dict": scaler.state_dict(), "args": vars(args),
                   "preference_reference_scores": reference_scores if args.preference_enabled else {},
                   "rng_state": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}}
        atomic_save_checkpoint(payload, checkpoint_dir / "event_last.pt")
        if improved: atomic_save_checkpoint(payload, checkpoint_dir / "event_best.pt")
        if args.save_every and epoch % args.save_every == 0: atomic_save_checkpoint(payload, checkpoint_dir / f"event_epoch_{epoch:03d}.pt")
        if validation and args.patience and stale >= args.patience:
            print(f"early_stop=epoch_{epoch} patience={args.patience}", flush=True); break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
