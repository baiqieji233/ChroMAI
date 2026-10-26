"""Frozen, CPU-only teacher-forced configuration diagnostics.

This is an offline audit, not free-running generation or an automatic model
selection gate. Trusted local torch fixture files are required, together with
their codec provenance. No optimizer or training task is modified.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_checkpoint(source, directory):
    """Copy one atomic checkpoint generation; never overwrite an old snapshot."""
    source = Path(source).resolve()
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    # Unique names avoid overwriting earlier experiment baselines. In-place
    # source writers are rejected; production Composer publishes atomically.
    with tempfile.NamedTemporaryFile(prefix='composer_', suffix='.pt', dir=directory,
                                     delete=False) as handle:
        frozen = Path(handle.name)
    try:
        before = source.stat()
        shutil.copyfile(source, frozen)
        after = source.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (
                after.st_size, after.st_mtime_ns, after.st_ino):
            raise RuntimeError('Checkpoint changed during snapshot; retry with a stable saved point')
        return frozen, file_hash(frozen)
    except BaseException:
        frozen.unlink(missing_ok=True)
        raise


def verify_fixture(payload, provenance, *, allow_missing_motif=False):
    """Reject same-sized but differently ordered vocabularies before inference."""
    catalog = payload['atom_catalog']
    actual = hashlib.sha256(json.dumps(catalog, ensure_ascii=False).encode()).hexdigest()
    expected = provenance.get('parent_catalog_sha256', provenance.get('catalog_sha256'))
    if not expected or actual != expected:
        raise ValueError('Fixture catalog identity mismatch or missing catalog SHA-256')
    config = payload['model_config']
    fixture_config = provenance.get('parent_model_config', provenance.get('config', {}))
    for name in ('architecture_profile', 'action_state_policy', 'history_events',
                 'candidate_count', 'action_slots'):
        if name not in fixture_config or fixture_config[name] != config.get(name):
            raise ValueError(f'Fixture input contract mismatch: {name}')
    fixture_motif = fixture_config.get('motif_memory_policy', 'off')
    model_motif = config.get('motif_memory_policy', 'off')
    if fixture_motif != model_motif:
        if not (allow_missing_motif and fixture_motif == 'off' and model_motif != 'off'):
            raise ValueError('Fixture motif contract mismatch; missing motif needs explicit --allow-missing-motif')
    elif fixture_motif != 'off' and fixture_config.get('motif_max_events') != config.get('motif_max_events'):
        raise ValueError('Fixture motif contract mismatch: motif_max_events')
    return actual


@contextmanager
def intervention(model, variant):
    """Remove one actual projected term; hooks are always removed on exit.

    Zeroed branches are out-of-distribution sensitivity tests, not proposed
    deployment settings or proof of an optimal training architecture.
    """
    import torch

    targets = {'relation_zero': 'action_relation', 'placement_zero': 'placement_projection'}
    handle = None
    try:
        if variant in targets:
            layer = getattr(model, targets[variant])
            handle = layer.register_forward_hook(lambda _module, _inputs, output: torch.zeros_like(output))
        yield
    finally:
        if handle is not None:
            handle.remove()


def audit(args):
    import torch
    from maichart.composer.capability_metrics import CapabilityMetrics
    from maichart.composer.checkpoint import load
    from maichart.composer.train import _forward_batch

    torch.set_num_threads(args.threads)
    frozen, checkpoint_hash = freeze_checkpoint(args.checkpoint, args.snapshot_dir)
    model, payload = load(frozen, device='cpu')  # strict model-state validation
    model.eval()
    if not hasattr(model, 'action_factors'):
        raise ValueError('Configuration audit requires audio_grounded_v3')
    provenance = json.loads(Path(args.provenance).read_text(encoding='utf-8'))
    catalog_hash = verify_fixture(payload, provenance, allow_missing_motif=args.allow_missing_motif)
    fixture_hash = file_hash(args.batch_file)
    expected_fixture = provenance.get('fixture_sha256')
    if expected_fixture and expected_fixture != fixture_hash:
        raise ValueError('Fixture file SHA-256 mismatch')
    pack = torch.load(args.batch_file, map_location='cpu', weights_only=False, mmap=True)
    selected = pack[args.split]
    batches = [selected] if isinstance(selected, dict) else selected
    if args.max_batches is not None:
        batches = batches[:args.max_batches]
    if not batches:
        raise ValueError('Empty fixture selection')
    has_motif = any('motif_ids' in batch and bool(batch['motif_mask'].any()) for batch in batches)
    if any(name.startswith('motif_') for name in args.variants):
        if not hasattr(model, 'motif_memory') or not has_motif:
            raise ValueError('Motif interventions require enabled memory and a nonempty matching fixture')
        if 'motif_shuffled' in args.variants and any(batch['candidate_target'].numel() < 2 for batch in batches):
            raise ValueError('Motif shuffling requires at least two rows in every batch')
    report = {
        'schema': 'composer_configuration_capability_audit_v1',
        'checked_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'scope': 'teacher-forced at authored target candidates; not sampled output or musical quality',
        'intervention_scope': 'single projected branch removal or archive permutation; OOD sensitivity only',
        'checkpoint': {'source': str(Path(args.checkpoint).resolve()), 'snapshot': str(frozen),
                       'sha256': checkpoint_hash, 'epoch': payload.get('epoch'),
                       'optimizer_steps': payload.get('optimizer_steps'),
                       'model_config': payload['model_config']},
        'fixture': {'path': str(Path(args.batch_file).resolve()), 'sha256': fixture_hash,
                    'provenance': str(Path(args.provenance).resolve()), 'catalog_sha256': catalog_hash,
                    'fixture_hash_verified': bool(expected_fixture),
                    'fixture_motif_policy': provenance.get('parent_model_config', provenance.get('config', {})).get('motif_memory_policy', 'off'),
                    'checkpoint_motif_policy': payload['model_config'].get('motif_memory_policy', 'off'),
                    'missing_motif_allowed': args.allow_missing_motif,
                    'split': args.split, 'batches': len(batches), 'nonempty_motif': has_motif,
                    'split_semantics': provenance.get('split_semantics', provenance.get('scope'))},
        'evaluations': {},
    }
    for variant in args.variants:
        meter = CapabilityMetrics(model.action_factors)
        with intervention(model, variant), torch.no_grad():
            for batch in batches:
                batch = {key: value for key, value in batch.items() if torch.is_tensor(value)}
                overrides = {'motif_strength': 0.} if variant == 'motif_off' else {}
                if variant == 'motif_shuffled':
                    # Deterministic nonidentity row permutation, targets stay fixed.
                    permutation = torch.arange(batch['candidate_target'].numel()).roll(1)
                    batch = {**batch, **{key: batch[key][permutation] for key in (
                        'motif_ids', 'motif_relative_beats', 'motif_mask')}}
                output = _forward_batch(model, batch, **overrides)
                meter.update(output['action_logits'], batch)
        report['evaluations'][variant] = meter.report()
        print(json.dumps({'variant_complete': variant, 'output': str(args.output)}), flush=True)
    destination = Path(args.output)
    protected = {Path(args.checkpoint).resolve(), frozen, Path(args.batch_file).resolve(),
                 Path(args.provenance).resolve()}
    if destination.resolve() in protected:
        raise ValueError('Output report must not overwrite checkpoint, fixture or provenance')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--snapshot-dir', required=True)
    parser.add_argument('--batch-file', required=True)
    parser.add_argument('--provenance', required=True)
    parser.add_argument('--split', default='heldout')
    parser.add_argument('--output', required=True)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--max-batches', type=int)
    parser.add_argument('--allow-missing-motif', action='store_true',
                        help='Explicit no-motif-input diagnostic for an enabled-motif checkpoint; not its full-path evaluation')
    parser.add_argument('--variants', nargs='+', default=['normal', 'relation_zero', 'placement_zero'],
                        choices=['normal', 'relation_zero', 'placement_zero', 'motif_off', 'motif_shuffled'])
    args = parser.parse_args(argv)
    if not 1 <= args.threads <= 4 or (args.max_batches is not None and args.max_batches < 1):
        parser.error('threads must be in [1,4]; max-batches must be positive')
    audit(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
