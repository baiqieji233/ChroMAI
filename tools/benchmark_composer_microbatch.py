"""Fixed-input, isolated Composer update benchmark. Never writes the checkpoint.

Prepare on CPU before stopping training, then run with the GPU exclusively.
The optional synthetic capacity arm is a memory stress test, not a chart.
"""
import argparse
import copy
import hashlib
import json
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from maichart.composer.checkpoint import load
from maichart.composer.data import ComposerDataset, collate_composer
from maichart.composer.train import _forward_batch, _loss, _nonfinite_tensors, _assert_model_finite,_forward_finite_check


def chunks(items, size):
    if size <= 0 or len(items) % size:
        raise ValueError('microbatch must divide effective batch')
    return [items[i:i + size] for i in range(0, len(items), size)]


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def prepare(args):
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    a = payload['args']
    ds = ComposerDataset(a['manifest'], cache_dir=a['cache_dir'],
        candidate_count=a['candidate_count'], history_events=a['history_events'],
        action_slots=a['action_slots'], preparation_workers=args.workers,
        audio_cache_gb=1., limit_charts=args.limit_charts, seed=a['seed'],
        architecture_profile=a['architecture_profile'], action_state_policy=a['action_state_policy'],
        motif_memory_policy=a['motif_memory_policy'], motif_max_events=a['motif_max_events'],
        head_timing_policy=a['head_timing_policy'], structure_policy=a['structure_policy'],
        structure_backend=a['structure_backend'], atom_catalog=payload['atom_catalog'])
    ds.touch_layout_policy = a['touch_layout_policy']
    ds.contact_policy = a['contact_policy']
    if len(ds) < 256:
        raise ValueError('dataset has fewer than 256 samples')
    indices = random.Random(args.seed).sample(range(len(ds)), 256)
    items = [ds[i] for i in indices]
    metadata = {'manifest': a['manifest'], 'limit_charts': args.limit_charts,
                'indices': indices, 'dataset_size': len(ds), 'seed': args.seed,
                'catalog_sha256': hashlib.sha256(json.dumps(payload['atom_catalog'], ensure_ascii=False).encode()).hexdigest(),
                'max_target_atoms': max(int(x['action_target'].ne(0).sum()) for x in items),
                'max_history_width': max(int(x['history'].ne(0).any(0).sum()) for x in items)}
    torch.save({'items': items, 'metadata': metadata}, args.output)
    print(json.dumps({'input_file': str(args.output), 'sha256': digest(args.output), **metadata}), flush=True)


def stress_items(items):
    result = list(items)
    sample = copy.deepcopy(result[0])
    target = sample['action_target']
    atom = next((int(v) for v in target if int(v) > 0), 1)
    target[:] = atom
    if sample['history'].ndim == 2 and sample['history'].shape[-1] >= 33 and sample['history'].shape[0]:
        sample['history'][-1, :33] = atom
    result[0] = sample
    return result


def arm(args):
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    bundle = torch.load(args.inputs, map_location='cpu', weights_only=False)
    items = bundle['items']
    if args.stress:
        items = stress_items(items)
    model, payload = load(args.checkpoint, device='cpu', include_optimizer=True)
    a = payload['args']
    catalog_hash = hashlib.sha256(json.dumps(payload['atom_catalog'], ensure_ascii=False).encode()).hexdigest()
    if catalog_hash != bundle['metadata']['catalog_sha256'] or a['manifest'] != bundle['metadata']['manifest']:
        raise ValueError('checkpoint catalog or training manifest mismatch')
    model.to('cuda').train()
    optimizer = torch.optim.AdamW(model.parameters(), fused=True)
    optimizer.load_state_dict(payload['optimizer_state_dict'])
    for group in optimizer.param_groups:
        group['fused'] = True
        group['foreach'] = False
    del payload
    from torch.utils.data._utils.pin_memory import pin_memory
    prepared=[pin_memory(collate_composer(group),'cuda') for group in chunks(items,args.microbatch)]
    losses, norms, wall, gpu = [], [], [], []
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    for update in range(args.warmup + args.steps):
        start = time.perf_counter()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        optimizer.zero_grad(set_to_none=True)
        total = torch.zeros((), device='cuda')
        for batch in prepared:
            tensors = {k: v.to('cuda', non_blocking=True) for k, v in batch.items()
                       if torch.is_tensor(v) and k not in {'song_index', 'event_index'}}
            with torch.autocast('cuda', dtype=torch.bfloat16):
                output = _forward_batch(model, tensors, audio_auxiliary=a['audio_pointer_weight'] > 0)
                bad = _nonfinite_tensors(output) if args.check_mode=='legacy' else []
                loss, _ = _loss(output, tensors, model.config.atom_vocab_size,
                    audio_pointer_weight=a['audio_pointer_weight'], placement_weight=a['placement_weight'],
                    planner_weight=a['planner_weight'], action_objective_policy=a['action_objective_policy'])
            backward = loss / (256 // args.microbatch)
            if args.check_mode=='legacy':
                loss_finite=bool(torch.isfinite(loss));backward_finite=bool(torch.isfinite(backward))
            else:bad,loss_finite,backward_finite=_forward_finite_check(output,loss,backward)
            if bad or not loss_finite or not backward_finite:
                raise FloatingPointError(f'nonfinite forward: {bad}')
            backward.backward()
            total = total + backward.detach()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        _assert_model_finite(model)
        end.record()
        torch.cuda.synchronize()
        if update >= args.warmup:
            wall.append((time.perf_counter() - start) * 1000)
            gpu.append(begin.elapsed_time(end))
            losses.append(float(total))
            norms.append(float(norm))
    free, total_memory = torch.cuda.mem_get_info()
    print('BENCH_RESULT ' + json.dumps({'status': 'ok', 'microbatch': args.microbatch,
        'accumulation': 256 // args.microbatch, 'stress': args.stress,'check_mode':args.check_mode, 'wall_ms': wall,
        'cuda_event_ms': gpu, 'mean_ms': statistics.mean(wall),
        'samples_per_second': 256000 / statistics.mean(wall), 'losses': losses, 'gradient_norms': norms,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
        'driver_free_bytes': free, 'driver_total_bytes': total_memory}), flush=True)


def run(args):
    results = []
    for size, stress,check_mode in [(64,False,'legacy'),(64,False,'merged'),(128,False,'merged'),(128,True,'merged')]:
        command = [sys.executable, __file__, 'arm', '--checkpoint', args.checkpoint, '--inputs', args.inputs,
                   '--microbatch', str(size), '--warmup', str(args.warmup), '--steps', str(args.steps), '--seed', str(args.seed),
                   '--check-mode',check_mode]
        if stress: command.append('--stress')
        process = subprocess.run(command, text=True, capture_output=True)
        line = next((x for x in process.stdout.splitlines() if x.startswith('BENCH_RESULT ')), None)
        result = json.loads(line[len('BENCH_RESULT '):]) if line else {'status': 'oom' if 'out of memory' in process.stderr.lower() else 'error', 'microbatch': size, 'stress': stress,'check_mode':check_mode, 'exit_code': process.returncode, 'stderr': process.stderr[-5000:]}
        results.append(result)
        print(json.dumps(result), flush=True)
    report = {'checkpoint': args.checkpoint, 'checkpoint_sha256': digest(args.checkpoint),
              'inputs': args.inputs, 'inputs_sha256': digest(args.inputs), 'results': results,
              'caveats': 'Fixed prepared pinned inputs: CPU feature extraction excluded. Synthetic 33-slot stress is not a valid chart or quality test. Per-microbatch weighted loss reductions, EOS weights, dropout and padding differ: gradients and trajectories are not equivalent. Recovery updates excluded.'}
    Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='mode', required=True)
    for name in ('prepare', 'run', 'arm'):
        s = sub.add_parser(name)
        s.add_argument('--checkpoint', required=True)
        s.add_argument('--seed', type=int, default=101)
        if name == 'prepare':
            s.add_argument('--output', required=True); s.add_argument('--limit-charts', type=int, default=32)
            s.add_argument('--workers', type=int, default=2)
        else:
            s.add_argument('--inputs', required=True); s.add_argument('--warmup', type=int, default=2)
            s.add_argument('--steps', type=int, default=4)
            if name == 'run': s.add_argument('--output', required=True)
            else:
                s.add_argument('--microbatch', type=int, choices=(64, 128, 256), required=True)
                s.add_argument('--stress', action='store_true')
                s.add_argument('--check-mode',choices=('legacy','merged'),default='merged')
    args = p.parse_args()
    {'prepare': prepare, 'run': run, 'arm': arm}[args.mode](args)


if __name__ == '__main__':
    main()
