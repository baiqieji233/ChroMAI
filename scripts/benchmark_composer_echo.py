"""Synthetic component microbenchmark; no training or end-to-end throughput claims.

Example: python scripts/benchmark_composer_echo.py --device cuda --steps 30
    --warmup 5 --output echo_benchmark.json
Dense echo reference is loaded from the standalone regression test, not scratch.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import torch
from maichart.composer.phase_echo import phase_echo_features
from maichart.composer.quality_context import song_relative_time_scores


def dense_time_scores(query, offsets, ages):
    delta = offsets.float()[..., None] + ages.float()[:, None]
    positive = delta.clamp_min(0.)
    features = [torch.log1p(positive), torch.exp(-positive / 4.)]
    for period in (2., 4., 1., 3.):
        angle = delta * (2. * math.pi / period)
        features.extend((angle.sin(), angle.cos()))
    return torch.einsum('bchd,bcd->bch', torch.stack(features, -1), query.float()) / math.sqrt(10.)


def load_dense_echo():
    spec = importlib.util.spec_from_file_location(
        '_echo_benchmark_reference', ROOT / 'tests/test_composer_echo_efficiency.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._dense_reference


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def compare_abba(operations, device, steps, warmup):
    """Each operation prepares work then returns a zero-argument timed call.

    ABBA blocks avoid systematically favoring the second implementation.
    CUDA event pairs time each iteration with synchronization once per block.
    Peak allocation includes preparation (e.g. backward's forward graph), while
    backward timing excludes preparation. No empty_cache or optimizer is used.
    """
    order = ('dense', 'optimized', 'optimized', 'dense')
    for name in order:
        for _ in range(warmup):
            result = operations[name]()()
            del result
    synchronize(device)
    times = {name: [] for name in operations}
    peaks = {name: [] for name in operations}
    for name in order:
        synchronize(device)
        baseline = torch.cuda.memory_allocated(device) if device.type == 'cuda' else None
        if baseline is not None:
            torch.cuda.reset_peak_memory_stats(device)
        events = []
        for _ in range(steps):
            call = operations[name]()
            if device.type == 'cuda':
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                start.record()
                result = call()
                end.record()
                events.append((start, end))
            else:
                start = time.perf_counter()
                result = call()
                times[name].append((time.perf_counter() - start) * 1000.)
            del result, call
        synchronize(device)
        if baseline is not None:
            times[name].extend(start.elapsed_time(end) for start, end in events)
            peaks[name].append(max(0, torch.cuda.max_memory_allocated(device) - baseline))
    return {
        name: {'median_ms': statistics.median(times[name]),
               'timed_iterations': len(times[name]),
               'incremental_peak_allocated_bytes': max(peaks[name]) if peaks[name] else None}
        for name in operations
    }


def errors(a, b):
    difference = (a.float() - b.float()).abs()
    return {'max_abs': difference.max().item(),
            'relative_l2': (difference.norm() / b.float().norm().clamp_min(1e-12)).item()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--candidates', type=int, default=96)
    parser.add_argument('--history', type=int, nargs='+', default=[128, 512])
    parser.add_argument('--dtype', choices=['float32', 'bfloat16'], default='float32')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 0 or min(args.history + [args.batch, args.candidates]) < 1:
        parser.error('steps and dimensions must be positive; warmup must be nonnegative')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA requested but unavailable')
    # Keep FP32 comparisons independent of machine-specific TF32 defaults.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(1729)
    dtype = getattr(torch, args.dtype)
    dense_echo = load_dense_echo()
    report = {
        'scope': 'synthetic components only; excludes model, data loading, optimizer and end-to-end training',
        'device': str(device), 'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
        'torch_version': str(torch.__version__), 'dtype': args.dtype,
        'tf32': False, 'steps_per_block': args.steps, 'warmup_per_block': args.warmup,
        'order': 'ABBA; A=dense B=optimized',
        'memory': 'max allocated minus pre-block allocated; excludes reserved memory; backward includes forward graph preparation',
        'timed_gradients': 'projected query only; clocks are fixed data as in training; error probe also checks clock derivatives',
        'cases': [],
    }
    for h in args.history:
        b, c = args.batch, args.candidates
        lanes = torch.rand(b, h, 8, device=device) < .15
        lanes[:, ::4] = False  # Touch-only events.
        valid = torch.rand(b, h, device=device) > .12
        ages = torch.arange(h - 1, -1, -1, device=device).float()[None].expand(b, h) * .25
        ages = ages.masked_fill(~valid, 0.)
        offsets = torch.rand(b, c, device=device) * 16.
        echo_args = lanes, ages, valid, offsets
        with torch.no_grad():
            expected, actual = dense_echo(*echo_args), phase_echo_features(*echo_args)
            exact = {key: torch.equal(actual[key], expected[key]) for key in expected}
            if not all(exact.values()):
                raise AssertionError(f'echo mismatch H={h}: {exact}')
        del expected, actual
        echo_ops = {
            'dense': lambda: lambda: dense_echo(*echo_args),
            'optimized': lambda: lambda: phase_echo_features(*echo_args),
        }
        case = {'batch': b, 'candidates': c, 'history': h, 'echo_exact_by_field': exact,
                'echo_forward': compare_abba(echo_ops, device, args.steps, args.warmup)}
        query = torch.randn(b, c, 10, device=device, dtype=dtype, requires_grad=True)
        # Test gradients for every differentiable input, although training
        # generally needs gradients only for the projected query.
        relative_offsets = offsets.detach().to(dtype).requires_grad_(True)
        relative_ages = ages.detach().to(dtype).requires_grad_(True)
        inputs = query, relative_offsets, relative_ages
        gradient = torch.randn(b, c, h, device=device) / math.sqrt(b*c*h)
        out_dense = dense_time_scores(*inputs)
        out_optimized = song_relative_time_scores(*inputs)
        grad_dense = torch.autograd.grad(out_dense, inputs, gradient)
        grad_optimized = torch.autograd.grad(out_optimized, inputs, gradient)
        case['relative_time_errors'] = {
            'output': errors(out_optimized, out_dense),
            'gradients': {key: errors(a, e) for key, a, e in zip(
                ('query', 'offsets', 'ages'), grad_optimized, grad_dense)},
        }
        del out_dense, out_optimized, grad_dense, grad_optimized
        # Clocks are dataset inputs, not learned parameters. Time the real
        # training graph, while retaining the stronger derivative probe above.
        timed_inputs=(query,relative_offsets.detach(),relative_ages.detach())
        functions = {'dense': dense_time_scores, 'optimized': song_relative_time_scores}
        def prepare(fn, mode):
            if mode == 'forward':
                return lambda: fn(*timed_inputs)
            if mode == 'backward':
                out = fn(*timed_inputs)
                return lambda: torch.autograd.grad(out, query, gradient)
            def both():
                out = fn(*timed_inputs)
                return torch.autograd.grad(out, query, gradient)
            return both
        for mode in ('forward', 'backward', 'forward_backward'):
            operations = {key: (lambda fn=fn, mode=mode: prepare(fn, mode))
                          for key, fn in functions.items()}
            case['relative_time_' + mode] = compare_abba(
                operations, device, args.steps, args.warmup)
        report['cases'].append(case)
    payload = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + '\n', encoding='utf-8')
    print(payload)


if __name__ == '__main__':
    main()
