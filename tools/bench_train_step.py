"""Measure milliseconds per training micro-batch, in about a minute.

The full-corpus run needs ~4,790 micro-batches per epoch, so a single change that
shaves 20 ms per step is worth ~1.6 minutes per epoch and hours over a run.  This
tool reproduces one step at the real shapes and reports ms/step plus the implied
epoch time, so a configuration change can be judged in a minute instead of after
40 minutes of real training.

It needs the GPU to itself: run it while no training task is running.

Example::

    python tools/bench_train_step.py --steps 30
    python tools/bench_train_step.py --batch-size 8 --decoder-len 488 --encoder-len 736
    python tools/bench_train_step.py --no-sync-stats      # A/B the per-step syncs
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from maichart.generation.event_model import EventConditionedTransformer, EventTransformerConfig  # noqa: E402

MICRO_BATCHES_PER_EPOCH = 4790  # full corpus: 76,643 windows / batch 16

# Measured on the corpus: target lengths p50 218 / p90 344 / max 756, and audio
# window frames p50 533 / p90 701 / p99 960 / max 2193 (the model pools the
# encoder to max_audio_tokens=512, so long windows still cost a pooling pass).
TOKEN_SAMPLES = (13, 60, 120, 180, 218, 260, 300, 344, 420, 488, 600, 756)
FRAME_SAMPLES = (232, 417, 480, 533, 600, 701, 800, 960, 1200, 1600, 2193)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ffn", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--decoder-len", type=int, default=488, help="mean padded target length (measured 488)")
    parser.add_argument("--encoder-len", type=int, default=480, help="window audio frames incl. margins (measured 480)")
    parser.add_argument("--global-frames", type=int, default=256)
    parser.add_argument("--vocab", type=int, default=2037)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-sync-stats", action="store_true",
                        help="skip the per-micro-batch statistics read-back instead of batching it per epoch")
    parser.add_argument("--padding-mask", action="store_true",
                        help="reproduce training's masks: a causal decoder mask plus padding masks.  PyTorch "
                             "merges those into an additive float mask, which pushes attention off the flash "
                             "backend -- measure with and without to see what that costs.")
    parser.add_argument("--mask-fraction", type=float, default=0.25,
                        help="fraction of the last sample's frames/tokens marked as padding in --padding-mask mode")
    parser.add_argument("--shape-jitter", action="store_true",
                        help="redraw the batch shapes every step, the way a real dataloader does (measured windows "
                             "span 13..756 tokens and 232..2193 audio frames).  Fixed shapes let the caching "
                             "allocator reuse one set of blocks, which hides fragmentation growth -- exactly what a "
                             "saturated-VRAM run suffers from.")
    parser.add_argument("--micro-batches", type=int, default=MICRO_BATCHES_PER_EPOCH)
    parser.add_argument("--trace-every", type=int, default=0,
                        help="print allocated/reserved memory every N measured steps, to expose allocator growth")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    config = EventTransformerConfig(
        input_dim=108, vocab_size=args.vocab, d_model=args.d_model, nhead=args.heads,
        num_layers=args.layers, dim_feedforward=args.ffn, dropout=0.15,
        max_tokens=1024, global_audio_tokens=args.global_frames,
    )
    model = EventConditionedTransformer(config).to(device)
    model.position_ids = torch.arange(args.vocab, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.02)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    parameters = sum(p.numel() for p in model.parameters())

    hidden = args.d_model
    rng = __import__("random").Random(7)
    audio = torch.randn(args.batch_size, args.encoder_len, config.input_dim, device=device)
    global_audio = torch.randn(args.batch_size, args.global_frames, config.input_dim, device=device)
    decoder_input = torch.randint(0, args.vocab, (args.batch_size, args.decoder_len), device=device)
    target = torch.randint(0, args.vocab, (args.batch_size, args.decoder_len), device=device)
    condition = torch.rand(args.batch_size, 16, device=device) * 0.5 + 0.25
    valid = torch.ones_like(target, dtype=torch.bool)

    print(f"device={device} params={parameters / 1e6:.2f}M d_model={hidden} layers={args.layers} "
          f"batch={args.batch_size} decoder_len={args.decoder_len} encoder_len={args.encoder_len} "
          f"global_frames={args.global_frames}")
    timings: list[float] = []
    reserved_trace: list[float] = []
    for step in range(args.steps + args.warmup):
        # Real batches are padded to their own longest item, and every step has a
        # different shape; that is what makes the allocator grow.
        if args.shape_jitter:
            decoder_len = int(rng.choice(TOKEN_SAMPLES))
            encoder_len = int(rng.choice(FRAME_SAMPLES))
            decoder_input = torch.randint(0, args.vocab, (args.batch_size, decoder_len), device=device)
            target = torch.randint(0, args.vocab, (args.batch_size, decoder_len), device=device)
            audio = torch.randn(args.batch_size, encoder_len, config.input_dim, device=device)
            valid = torch.ones_like(target, dtype=torch.bool)
        else:
            decoder_len, encoder_len = args.decoder_len, args.encoder_len
        # Training always has padding: windows have different lengths within a
        # batch, so both the audio and the target carry padding masks.
        audio_mask = decoder_mask = None
        if args.padding_mask:
            audio_mask = torch.zeros(args.batch_size, audio.size(1), dtype=torch.bool, device=device)
            audio_mask[:, int(audio.size(1) * (1 - args.mask_fraction)):] = True
            audio_mask[0, :] = False  # at least one full-length sample, as in a real bucket
            decoder_mask = torch.zeros(args.batch_size, decoder_len, dtype=torch.bool, device=device)
            decoder_mask[:, int(decoder_len * (1 - args.mask_fraction)):] = True
            decoder_mask[0, :] = False
        if device.type == "cuda":
            torch.cuda.synchronize()
        begin = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(audio, decoder_input, condition,
                           audio_padding_mask=audio_mask, decoder_padding_mask=decoder_mask,
                           global_audio=global_audio)
            raw_loss = F.cross_entropy(logits.transpose(1, 2), target, reduction="none")
            loss = raw_loss[valid].mean()
        if not args.no_sync_stats:
            torch.isfinite(loss).item()
            raw_loss[valid].sum().detach().item()
            (logits.argmax(-1) == target).sum().item()
        scaler.scale(loss / args.accumulation).backward()
        if (step + 1) % args.accumulation == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - begin) * 1000
        if step >= args.warmup:
            timings.append(elapsed)
            if device.type == "cuda":
                reserved_trace.append(torch.cuda.memory_reserved() / 1024 ** 3)
                if args.trace_every and len(timings) % args.trace_every == 0:
                    print(f"  step {len(timings):>4}: ms={elapsed:7.1f} "
                          f"allocated={torch.cuda.memory_allocated() / 1024 ** 3:.2f} "
                          f"reserved={torch.cuda.memory_reserved() / 1024 ** 3:.2f} GiB", flush=True)

    median = statistics.median(timings)
    print(f"ms/micro-batch: median={median:.1f} p10={min(timings):.1f} p90={max(timings):.1f} "
          f"(steps={len(timings)})")
    if reserved_trace:
        print(f"reserved trace: first={reserved_trace[0]:.2f} last={reserved_trace[-1]:.2f} "
              f"max={max(reserved_trace):.2f} GiB over {len(reserved_trace)} steps")
        first_half = statistics.median(timings[: max(1, len(timings) // 2)])
        second_half = statistics.median(timings[max(1, len(timings) // 2):])
        print(f"drift: first-half median={first_half:.1f} second-half={second_half:.1f} "
              f"({second_half / max(first_half, 1e-9):.2f}x)")
    epoch_seconds = median * args.micro_batches / 1000
    print(f"implied epoch ({args.micro_batches} micro-batches): {epoch_seconds / 60:.1f} min")
    if device.type == "cuda":
        print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GiB "
              f"reserved={torch.cuda.max_memory_reserved() / 1024 ** 3:.2f} GiB")
        flops = estimate_flops(args, config)
        print(f"effective throughput: {flops / (median / 1000) / 1e12:.2f} TFLOPS "
              f"(rough estimate {flops / 1e9:.1f} GFLOP/micro-batch)")
    return 0


def estimate_flops(args, config) -> float:
    """Forward+backward FLOPs for one micro-batch (attention + FFN + logits)."""

    batch, decoder, encoder = args.batch_size, args.decoder_len, args.encoder_len + args.global_frames
    heads, head_dim, d_model, ffn = config.nhead, config.d_model // config.nhead, config.d_model, config.dim_feedforward
    layers = config.num_layers
    attention = lambda L, S: 2 * batch * heads * L * S * head_dim  # noqa: E731
    ff = lambda tokens: 2 * tokens * d_model * ffn * 2  # noqa: E731
    forward = layers * (attention(encoder, encoder) + attention(decoder, decoder)
                        + attention(decoder, encoder) + ff(batch * encoder) + ff(batch * decoder))
    forward += 2 * batch * decoder * args.vocab * d_model  # output projection
    return forward * 3  # forward + backward


if __name__ == "__main__":
    raise SystemExit(main())
