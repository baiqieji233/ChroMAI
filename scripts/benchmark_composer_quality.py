"""Bounded synthetic cooperative_v1 on/off forward/backward timing.

This excludes CPU preparation, transfer, optimizer, recovery and generation.
It measures architecture overhead, not end-to-end samples/s or chart quality.
Examples (run only after implementation is complete):
  python scripts/benchmark_composer_quality.py --output result.json
  python scripts/benchmark_composer_quality.py --profile representative --device cuda --output gpu.json
  python scripts/benchmark_composer_quality.py --profile representative --checkpoint checkpoint.pt --output vocab.json
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import statistics
import time

import torch

from maichart.composer.model import ComposerConfig, GROUNDED_PROFILE, build_model
from maichart.composer.objectives import RECOVERY_ACTION_OBJECTIVE, action_group_loss_kwargs
from maichart.composer.train import _loss


SMALL_ATOMS = ['1', '2', '3', '4', '5', '8', '1h[4:1]', '1-5[8:1]', 'A1', 'C']


def read_catalog(args):
    if args.atom_catalog:
        value = json.loads(Path(args.atom_catalog).read_text(encoding='utf-8'))
        return value['atom_catalog'] if isinstance(value, dict) else value
    if args.checkpoint:
        # Weights are read solely to obtain the vocabulary, never overwritten.
        value = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        return list(value['atom_catalog'])
    return SMALL_ATOMS


def make_batch(config, batch_size, device):
    b, c, h, s = batch_size, config.candidate_count, config.history_events, config.action_slots
    probes, phrases = c + 16, 32
    rand = lambda *shape: torch.rand(*shape, device=device)
    zeros = lambda *shape: torch.zeros(*shape, dtype=torch.bool, device=device)
    features = rand(b, c, config.candidate_feature_dim)
    features[..., 120] = torch.linspace(.125, 16, c, device=device)
    features[..., 121] = features[..., 120] * .5
    history = torch.zeros(b, h, s, dtype=torch.long, device=device)
    history[..., :2] = torch.randint(1, config.atom_vocab_size, (b, h, 2), device=device)
    targets = torch.zeros(b, s, dtype=torch.long, device=device)
    targets[:, :2] = torch.randint(1, config.atom_vocab_size, (b, 2), device=device)
    candidate_target = torch.arange(b, device=device) % c
    indices = (torch.arange(c, device=device) * phrases // c).clamp_max(phrases - 1)
    quality = dict(
        quality_lifecycle=rand(b, c, 88), quality_anchor_audio=rand(b, 8, 120),
        quality_candidate_offsets=features[..., 120].clone(),
        quality_anchor_ids=history[:, -8:], quality_anchor_ages=rand(b, 8) * 16,
        quality_anchor_mask=~zeros(b, 8), quality_phrase_audio=rand(b, phrases, 104),
        quality_phrase_indices=indices[None].expand(b, -1),
        quality_phrase_target=rand(b, phrases, 8), quality_phrase_weight=rand(b, phrases),
        quality_gap_target=torch.randint(0, 7, (b,), device=device), quality_gap_weight=rand(b),
    )
    data = dict(candidate_features=features, history=history,
        history_features=rand(b, h, 8), history_padding_mask=zeros(b, h),
        history_beat_ages=torch.arange(h, device=device).flip(0)[None].expand(b, -1).float() * .5,
        condition=rand(b, 16), action_state=rand(b, 50), head_timing=rand(b, c, 99),
        candidate_mask=zeros(b, c), workload_audio=rand(b, probes, 104),
        ledger=rand(b, c, 6), interval_audio=rand(b, c, 16), trailing_audio=rand(b, probes, 52),
        audio_lens=rand(b, probes, 31, 26), audio_lens_mask=~zeros(b, probes, 31),
        planner_audio=rand(b, phrases, 104), planner_mask=zeros(b, phrases),
        planner_indices=torch.stack((indices, (indices + 1).clamp_max(phrases - 1)), -1)[None].expand(b, -1, -1),
        planner_mix=rand(b, c), candidate_indices=candidate_target,
        action_targets=targets, packed_actions=True, quality_inputs=quality)
    labels = dict(candidate_target=candidate_target, action_target=targets,
        future_target=rand(b, 8), eos_target=torch.zeros(b, device=device),
        workload_target=rand(b, probes, 5), workload_weight=rand(b, probes),
        workload_seconds=torch.arange(probes, device=device)[None].expand(b, -1).float(),
        trailing_target=rand(b, probes, 4), trailing_weight=rand(b, probes, 4),
        placement_target=rand(b, probes, 5), placement_weight=rand(b, probes, 5),
        planner_target=rand(b, phrases, 6), planner_weight=rand(b, phrases),
        sample_weight=torch.ones(b, device=device), **quality)
    return data, labels


def measure(model, data, labels, args, device):
    cuda = device.type == 'cuda'
    cast = (lambda: torch.autocast(device_type='cuda', dtype=torch.bfloat16)) if cuda and args.precision == 'bf16' else nullcontext
    kwargs = action_group_loss_kwargs(model)

    def forward_loss():
        with cast():
            output = model(**data)
            loss, _ = _loss(output, labels, model.config.atom_vocab_size,
                            action_objective_policy=RECOVERY_ACTION_OBJECTIVE, **kwargs)
        return loss

    for _ in range(args.warmup):
        model.zero_grad(set_to_none=True)
        forward_loss().backward()
    model.zero_grad(set_to_none=True)
    if cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        baseline_memory = torch.cuda.memory_allocated(device)
    else:
        baseline_memory = 0
    forward_times, backward_times, events = [], [], []
    started = time.perf_counter()
    for _ in range(args.steps):
        model.zero_grad(set_to_none=True)
        if cuda:
            start, middle, end = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
            start.record()
            loss = forward_loss()
            middle.record()
            loss.backward()
            end.record()
            events.append((start, middle, end))
        else:
            start = time.perf_counter()
            loss = forward_loss()
            middle = time.perf_counter()
            loss.backward()
            end = time.perf_counter()
            forward_times.append((middle - start) * 1000)
            backward_times.append((end - middle) * 1000)
    if cuda:
        torch.cuda.synchronize(device)
        forward_times = [start.elapsed_time(middle) for start, middle, _ in events]
        backward_times = [middle.elapsed_time(end) for _, middle, end in events]
    elapsed = time.perf_counter() - started
    return dict(wall_sec=elapsed, wall_step_ms=elapsed * 1000 / args.steps,
        synthetic_samples_per_sec=data['history'].size(0) * args.steps / elapsed,
        forward_ms_median=statistics.median(forward_times),
        backward_ms_median=statistics.median(backward_times),
        baseline_allocated_mib=baseline_memory / 1024 ** 2,
        peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 1024 ** 2 if cuda else None,
        extra_peak_allocated_mib=(torch.cuda.max_memory_allocated(device) - baseline_memory) / 1024 ** 2 if cuda else None,
        final_loss=float(loss.detach().float().cpu()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=('small', 'representative'), default='small')
    parser.add_argument('--quality-policy', choices=('cooperative_v1','cooperative_v2'), default='cooperative_v2')
    parser.add_argument('--batch-size',type=int,default=None)
    parser.add_argument('--order',choices=('ab','abba'),default='abba')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--precision', choices=('fp32', 'bf16'), default='bf16')
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--cpu-threads', type=int, default=2)
    parser.add_argument('--output', required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--atom-catalog', help='JSON atom list or object containing atom_catalog')
    source.add_argument('--checkpoint', help='Read only: take atom_catalog from an existing checkpoint')
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 0:
        parser.error('steps must be positive and warmup nonnegative')
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(123)
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    atoms = read_catalog(args)
    representative = args.profile == 'representative'
    batch_size = args.batch_size or (8 if representative else 2)
    if batch_size<1:parser.error('batch size must be positive')
    config = ComposerConfig(atom_vocab_size=len(atoms) + 1, d_model=384 if representative else 32,
        nhead=8 if representative else 4, layers=6 if representative else 1,
        dim_feedforward=1536 if representative else 64, dropout=0.,
        history_events=128 if representative else 8, candidate_count=96 if representative else 8,
        action_slots=33 if representative else 4, architecture_profile=GROUNDED_PROFILE,
        head_timing_dim=99, lens_width=64 if representative else 8,
        history_dropout=0., ledger_dropout=0., symbolic_dropout=0.,
        cross_event_relation_policy='relative_heads_v1', future_horizon_policy='candidate_horizon_v1',
        action_head_policy='conserved_grouped_v2', configuration_quality_policy=args.quality_policy)
    model = build_model(config, atoms).to(device).train()
    data, labels = make_batch(config, batch_size, device)
    quality, geometry = model.quality_context, model.cooperative_geometry
    rounds = []
    for enabled in ((False,True,True,False) if args.order=='abba' else (False,True)):
        model.config.configuration_quality_policy=args.quality_policy if enabled else 'off'
        model.quality_context = quality if enabled else None
        model.cooperative_geometry = geometry if enabled else None
        # Data and all original model weights are identical; no optimizer step.
        rounds.append({'variant':'on' if enabled else 'off',**measure(model,data,labels,args,device)})
    variant_results={}
    for name in ('off','on'):
        matches=[row for row in rounds if row['variant']==name]
        variant_results[name]={key:(max(row[key] for row in matches) if 'peak' in key else
                                  sum(row[key] for row in matches)/len(matches))
                              if matches[0][key] is not None else None
                              for key in matches[0] if key!='variant'}
    model.config.configuration_quality_policy=args.quality_policy
    report = dict(scope='synthetic forward/backward; excludes data/transfer/optimizer/recovery/generation',
        torch_version=torch.__version__, device=str(device),
        device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
        precision=args.precision if device.type == 'cuda' else 'fp32', profile=args.profile,
        batch_size=batch_size, vocab_size=len(atoms) + 1,
        catalog_source=args.atom_catalog or args.checkpoint or '10-atom fixture',
        config=config.to_dict(), steps=args.steps, warmup=args.warmup,
        quality_parameter_count=sum(p.numel() for p in quality.parameters()),
        geometry_parameter_count=sum(p.numel() for p in geometry.parameters()),
        memory_note='Both variants retain one model and Q1/Q2 parameter storage; extra_peak reports temporary/gradient memory above that baseline.',
        variants=variant_results,rounds=rounds,order=args.order,
        wall_overhead_ratio=variant_results['on']['wall_step_ms'] / variant_results['off']['wall_step_ms'] - 1.)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
