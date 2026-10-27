"""Compare Touch probability on identical teacher-forced Composer inputs.

The selected inputs are fixed once and reused across checkpoints. This probe
measures probabilities before grammar and playability masks; it does not run
free generation or establish a cause for any observed difference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import torch
from maichart.composer.action_guidance import action_logits_with_touch_guard, TOUCH_HISTORY_LOG_ODDS_BOUND
from maichart.composer.checkpoint import load
from maichart.composer.data import ComposerDataset, collate_composer
from maichart.composer.evaluation import file_hash
from maichart.composer.train import _forward_batch


def require_compatible(records):
    """Do not compare incompatible atom order or input contract."""
    if not records:
        raise ValueError('at least one checkpoint is required')
    catalog = records[0][1].get('atom_catalog')
    config = records[0][0].config.to_dict()
    if not isinstance(catalog, list) or not catalog:
        raise ValueError('checkpoint has no atom catalog')
    for model, payload in records[1:]:
        if payload.get('atom_catalog') != catalog:
            raise ValueError('checkpoint atom catalogs differ; compare them separately')
        if model.config.to_dict() != config:
            raise ValueError('checkpoint model configurations differ; fixed-input comparison is not valid')
    return catalog


def touch_mask(catalog, device='cpu'):
    return torch.tensor([False] + [bool(atom) and atom[0] in 'ABCDE' for atom in catalog],
                        dtype=torch.bool, device=device)


def accumulate(output, batch, mask, model):
    """Return weighted probability sums and weights for one fixed batch."""
    logits = output['action_logits'].float()
    audio = output.get('audio_action_logits')
    if audio is None or 'audio_action_context' not in output:
        raise ValueError('checkpoint does not expose audio action logits and contexts')
    audio = audio.float()
    target = batch['action_target']
    if logits.ndim != 3 or audio.shape != logits.shape or logits.shape[:2] != target.shape:
        raise ValueError('unexpected selected-candidate action logits shape')
    if logits.shape[-1] != mask.numel():
        raise ValueError('checkpoint vocabulary does not match logits')
    w = batch.get('sample_weight', torch.ones(target.shape[0], device=target.device))
    w = w * batch.get('transition_weight', torch.ones_like(w)) * batch.get('action_weight', torch.ones_like(w))
    active = torch.ones_like(target, dtype=torch.bool)
    if target.shape[1] > 1:
        active[:, 1:] = target[:, :-1].ne(0).long().cumprod(-1).bool()
    full_p = logits.softmax(-1)[..., mask].sum(-1)
    audio_p = audio.softmax(-1)[..., mask].sum(-1)
    contexts = output['action_context']
    audio_contexts = output['audio_action_context']
    if contexts.ndim != 2 or audio_contexts.shape != contexts.shape:
        raise ValueError('expected selected action contexts for each sample')
    guided = []
    changed = 0
    for i in range(target.shape[0]):
        diag = {}
        previous_heads = output.get('previous_heads')
        previous = previous_heads[i] if previous_heads is not None else None
        result = action_logits_with_touch_guard(model, contexts[i], 0, [], audio_context=audio_contexts[i],
            enabled=True, touch_mask=mask, diagnostics=diag, previous_heads=previous)
        guided.append(result.float().softmax(-1)[mask].sum())
        changed += bool(diag.get('applied'))
    guided = torch.stack(guided)
    results = {}
    for label, slot_slice in (('first', slice(0, 1)), ('later', slice(1, None))):
        weights = w[:, None] * active[:, slot_slice]
        mass = float(weights.sum().item())
        count = int(((weights > 0)).sum().item())
        results[label] = {'weight_sum': mass, 'valid_decisions': count,
            'main_probability_sum': float((full_p[:, slot_slice] * weights).sum().item()),
            'audio_probability_sum': float((audio_p[:, slot_slice] * weights).sum().item())}
        if label == 'first':
            results[label]['guided_probability_sum'] = float((guided * weights[:, 0]).sum().item())
            results[label]['guidance_changed_valid_decisions'] = sum(
                1 for i in range(len(w)) if bool(w[i] > 0) and
                not torch.allclose(full_p[i, 0], guided[i], atol=1e-7, rtol=1e-6))
            results[label]['guidance_applied_raw_samples'] = changed
    return results


def combine(total, batch_result):
    for slot, values in batch_result.items():
        row = total.setdefault(slot, {})
        for key, value in values.items(): row[key] = row.get(key, 0) + value


def finalize(total):
    for row in total.values():
        mass = row['weight_sum']
        for key in ('main', 'audio', 'guided'):
            field = key + '_probability_sum'
            if field in row: row[key + '_mean_probability'] = row[field] / mass if mass else None
    return total


def run(args):
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else
                          'cpu' if args.device == 'auto' else args.device)
    if args.batches < 1 or args.batch_size < 1: raise ValueError('batches and batch-size must be positive')
    records = [load(path, device=device) for path in args.checkpoints]
    catalog = require_compatible(records)
    config = records[0][0].config
    dataset = ComposerDataset(args.manifest, cache_dir=args.cache_dir,
        candidate_count=config.candidate_count, history_events=config.history_events,
        action_slots=config.action_slots, architecture_profile=config.architecture_profile,
        action_state_policy=config.action_state_policy, head_timing_policy=config.head_timing_policy,
        motif_memory_policy=config.motif_memory_policy, motif_max_events=config.motif_max_events,
        preparation_workers=1, audio_cache_gb=.25, seed=0, atom_catalog=catalog,
        structure_policy=config.structure_policy, structure_backend=config.structure_backend)
    if dataset.action_slots != config.action_slots:
        raise ValueError('dataset action slots differ from checkpoint')
    n = min(len(dataset), args.batches * args.batch_size)
    if not n: raise ValueError('dataset has no eligible samples')
    # Equally spaced deterministic subset includes the entire manifest rather
    # than only the first song. Materialize once to freeze every tensor.
    indices = [i * (len(dataset) - 1) // max(1, n - 1) for i in range(n)]
    items = [dataset[i] for i in indices]
    batches = [collate_composer(items[i:i + args.batch_size]) for i in range(0, n, args.batch_size)]
    snapshot = hashlib.sha256()
    for batch in batches:
        for key, value in sorted(batch.items()):
            if torch.is_tensor(value):
                tensor = value.detach().cpu().contiguous()
                snapshot.update(key.encode()); snapshot.update(str((tensor.dtype, tuple(tensor.shape))).encode())
                snapshot.update(tensor.numpy().tobytes())
    mask = touch_mask(catalog, device)
    reports = []
    with torch.inference_mode():
        for path, (model, payload) in zip(args.checkpoints, records):
            model.eval(); aggregate = {}
            for batch in batches:
                tensors = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
                outputs = _forward_batch(model, tensors)
                combine(aggregate, accumulate(outputs, tensors, mask, model))
            reports.append({'checkpoint': str(Path(path).resolve()), 'sha256': file_hash(path),
                'optimizer_steps': payload.get('optimizer_steps'), 'epoch': payload.get('epoch'),
                'slots': finalize(aggregate)})
    steps = {r['optimizer_steps'] for r in reports if r['optimizer_steps'] is not None}
    report = {'schema': 'composer-touch-teacher-forced-probe-v1',
        'manifest': str(Path(args.manifest).resolve()), 'manifest_sha256': file_hash(args.manifest),
        'sample_indices': indices, 'sample_count': n, 'input_snapshot_sha256': snapshot.hexdigest(),
        'catalog_coverage': dataset.catalog_coverage,
        'atom_catalog_sha256': hashlib.sha256(json.dumps(catalog, ensure_ascii=False).encode()).hexdigest(),
        'touch_atom_count': int(mask.sum().item()), 'touch_guard_log_odds_bound': TOUCH_HISTORY_LOG_ODDS_BOUND,
        'checkpoints': reports, 'expected_steps_not_provided': [step for step in (1000, 8000) if step not in steps],
        'scope': 'authored history and within-event prefixes; probabilities before legal masks; first-slot guidance only; not generated-history probing or causal proof'}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoints', nargs='+', required=True); p.add_argument('--manifest', required=True)
    p.add_argument('--cache-dir', default='cache/chart_composer_v1'); p.add_argument('--output', required=True)
    p.add_argument('--batches', type=int, default=4); p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--device', default='auto')
    args = p.parse_args(argv); result = run(args)
    print(json.dumps({'output': args.output, 'samples': result['sample_count'],
                      'expected_steps_not_provided': result['expected_steps_not_provided']}))
    return 0

if __name__ == '__main__': raise SystemExit(main())
