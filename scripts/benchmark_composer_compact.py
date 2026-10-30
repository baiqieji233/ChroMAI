"""Synthetic full-loss forward/backward ABBA: conserved V2 versus compact V3.

Random initial weights, synthetic inputs and changed architectural semantics:
this measures compute/memory, never chart quality or checkpoint accuracy.
No optimizer, real audio/cache access, parameter updates or checkpoint writes.
Example: python scripts/benchmark_composer_compact.py --batch-size 8 --output compact_abba.json
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import statistics

import torch

from benchmark_composer_quality import make_batch, measure
from maichart.composer.checkpoint import read_payload
from maichart.composer.model import ComposerConfig, GROUNDED_PROFILE, build_model


def make_config(vocab_size, policy):
    return ComposerConfig(
        atom_vocab_size=vocab_size, architecture_profile=GROUNDED_PROFILE,
        d_model=384, nhead=8, layers=6, dim_feedforward=1536,
        history_events=128, candidate_count=96, action_slots=8,
        dropout=0., history_dropout=0., ledger_dropout=0., symbolic_dropout=0.,
        condition_mask_prob=0., lens_width=64, head_timing_dim=123,
        head_timing_policy='separate_pads_v2',
        cross_event_relation_policy='relative_heads_v1',
        future_horizon_policy='candidate_horizon_v1',
        configuration_quality_policy='song_paired_v3',
        echo_prior_policy='phase_echo_v1', timing_distribution_policy='survival_v1',
        button_load_policy='per_button_v1', action_head_policy=policy)


def make_v3_batch(config, batch_size, device, notes_per_event=2):
    data, labels = make_batch(config, batch_size, device)
    b, c, h = batch_size, config.candidate_count, config.history_events
    phrases = data['planner_audio'].size(1)
    probes = data['workload_audio'].size(1)
    # Packed training evaluates real notes plus the first STOP, rather than
    # padding all S slots. Expose this workload dimension in the CLI/report.
    targets = torch.zeros_like(data['action_targets'])
    targets[:, :notes_per_event] = torch.randint(
        1, config.atom_vocab_size, (b, notes_per_event), device=device)
    data['action_targets'] = targets
    labels['action_target'] = targets
    rand = lambda *shape: torch.rand(*shape, device=device)
    clock_beats = torch.arange(phrases, device=device).float() * 4
    plan_clock = torch.stack((clock_beats, clock_beats * .5), -1)[None].expand(b, -1, -1)
    history_beats = data['history_beat_ages']
    history_clock = torch.stack((128. - history_beats, 64. - history_beats * .5), -1)
    history_indices = ((history_clock[..., 0] / 4).long()).clamp(0, phrases - 1)
    phrase_indices = torch.arange(phrases, device=device)[None].expand(b, -1)
    q = data['quality_inputs']
    q.update(
        quality_plan_clock=plan_clock,
        quality_history_audio=rand(b, h, 104),
        quality_history_clock=history_clock,
        quality_history_plan_indices=torch.stack((history_indices, (history_indices + 1).clamp_max(phrases - 1)), -1),
        quality_history_plan_mix=rand(b, h),
        quality_phrase_plan_indices=torch.stack((phrase_indices, phrase_indices), -1),
        quality_phrase_plan_mix=torch.zeros(b, phrases, device=device),
        quality_phrase_target=rand(b, phrases, 9),
        quality_future_audio=rand(b, c, 10, 104),
        quality_future_mask=torch.ones(b, c, 10, dtype=torch.bool, device=device),
        quality_future_offsets=torch.tensor([.25, .5, .75, 1., 1.5, 2., 3., 4., 6., 8.], device=device)[None].expand(b, -1))
    data.update(
        head_timing=rand(b, c, 123), history_second_ages=history_beats * .5,
        survival_start=torch.zeros(b, device=device), empty_gap_audio=rand(b, 8, 104))
    # The first C workload probes coincide with candidate timestamps. Remaining
    # probes represent independent synthetic Poisson-count supervision.
    seconds = torch.cat((data['candidate_features'][..., 121],
                         torch.linspace(8.5, 16., probes - c, device=device)[None].expand(b, -1)), -1)
    labels.update(q)
    labels.update(
        workload_seconds=seconds, event_count_target=torch.randint(0, 4, (b, probes), device=device).float(),
        event_exposure=torch.full((b, probes), .5, device=device),
        event_rate_weight=torch.ones(b, probes, device=device),
        empty_gap_exposure=torch.full((b, 8), .125, device=device),
        empty_gap_weight=torch.ones(b, 8, device=device))
    return data, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', help='Read only atom_catalog; defaults to v25 inference, then last')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--notes-per-event', type=int, default=2,
                        help='1..7; packed slots per event are this plus one STOP (default 3)')
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--precision', choices=('bf16', 'fp32'), default='bf16')
    parser.add_argument('--cpu-threads', type=int, default=2)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.batch_size < 1 or args.steps < 1 or args.warmup < 0 or args.cpu_threads < 1:
        parser.error('batch-size, steps and cpu-threads must be positive; warmup nonnegative')
    if not 1 <= args.notes_per_event <= 7:
        parser.error('notes-per-event must be in 1..7')
    root = Path(__file__).resolve().parents[1]
    if args.checkpoint:
        checkpoint = Path(args.checkpoint).resolve()
    else:
        candidates = [root / 'checkpoints/prism_composer_v25' / name
                      for name in ('composer_inference.pt', 'composer_last.pt')]
        checkpoint = next((path for path in candidates if path.is_file()), None)
        if checkpoint is None:
            parser.error('No v25 inference/last checkpoint; supply --checkpoint')
    payload = read_payload(checkpoint)
    atoms = list(payload['atom_catalog'])
    del payload
    gc.collect()
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    if device.type == 'cuda' and args.precision == 'bf16' and not torch.cuda.is_bf16_supported():
        parser.error('Selected CUDA device does not support BF16; use --precision fp32')
    # Fixed data survives the four rounds. Model initializations use a separate
    # seed, reset each round; repeated instances of a variant are identical.
    torch.manual_seed(args.seed)
    configs = {name: make_config(len(atoms) + 1, name) for name in
               ('conserved_grouped_v2', 'compact_grouped_v3')}
    data, labels = make_v3_batch(configs['compact_grouped_v3'], args.batch_size, device, args.notes_per_event)
    order = ('conserved_grouped_v2', 'compact_grouped_v3', 'compact_grouped_v3', 'conserved_grouped_v2')
    rounds = []
    for number, variant in enumerate(order, 1):
        torch.manual_seed(args.seed + 1)
        model = build_model(configs[variant], atoms).to(device).train()
        parameters = sum(parameter.numel() for parameter in model.parameters())
        print(f'Round {number}/4: {variant}; B={args.batch_size}; parameters={parameters}', flush=True)
        result = measure(model, data, labels, args, device)
        rounds.append(dict(round=number, variant=variant, parameter_count=parameters, **result))
        print(json.dumps(rounds[-1]), flush=True)
        del model
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            # Only between model instances, never inside warmup/measured steps.
            torch.cuda.empty_cache()
    variants = {}
    for variant in configs:
        matches = [row for row in rounds if row['variant'] == variant]
        variants[variant] = {
            key: None if matches[0][key] is None else
            max(row[key] for row in matches) if 'peak' in key else
            statistics.median(row[key] for row in matches)
            for key in matches[0] if key not in ('round', 'variant')}
    before, after = (variants[name] for name in configs)
    report = dict(
        scope='Synthetic full training-loss forward/backward only; excludes data preparation, transfer, optimizer, recovery and generation',
        interpretation='Random initial weights; architectural semantics differ. This is not a quality, accuracy or end-to-end throughput measurement.',
        seed=args.seed, initialization='Same reset seed for each model; shape/path changes can shift subsequent random draws across variants.',
        torch_version=torch.__version__, device=str(device),
        device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
        precision=args.precision if device.type == 'cuda' else 'fp32',
        catalog_source=str(checkpoint), vocab_size=len(atoms) + 1,
        checkpoint_use='atom_catalog only, lazy read_payload; no checkpoint weights or optimizer restored',
        batch_size=args.batch_size, steps=args.steps, warmup=args.warmup,
        notes_per_event=args.notes_per_event, packed_slots_per_event=args.notes_per_event + 1,
        order='ABBA', configs={name: config.to_dict() for name, config in configs.items()},
        memory_note='Exactly one model resident per round; fixed shared synthetic batch remains allocated. Peaks include model, batch, activations and gradients.',
        variants=variants, rounds=rounds,
        compact_wall_ratio=after['wall_step_ms'] / before['wall_step_ms'],
        compact_gpu_step_ratio=(after['forward_ms_median'] + after['backward_ms_median']) /
                               (before['forward_ms_median'] + before['backward_ms_median']))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
