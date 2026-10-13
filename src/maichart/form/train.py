"""Train song-form planning and layout with observable, resumable updates."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F

from maichart.form.data import FormDataset, collate_form, form_audio
from maichart.form.model import FormConfig, FormTransformer
from maichart.form import checkpoint
from maichart.form.training_runtime import (
    TRAINING_SEMANTICS, EpochBatches, Progress, UpdateEngine, capture_rng, emit,
    resolve_precision, restore_rng, validate_cpu_batch,
)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--checkpoint-dir', required=True)
    p.add_argument('--cache-dir', default='cache/song_form')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--prepare-workers', type=int, default=1)
    p.add_argument('--device', default='cpu')
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--lr', type=float, default=.0002)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max-steps', type=int, default=0)
    p.add_argument('--limit-charts', type=int, default=0)
    p.add_argument('--d-model', type=int, default=256)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--nhead', type=int, default=8)
    p.add_argument('--max-tokens', type=int, default=2048)
    p.add_argument('--dropout', type=float, default=.1)
    p.add_argument('--weight-decay', type=float, default=.01)
    p.add_argument('--grad-accumulation', type=int, default=1)
    p.add_argument('--save-every', type=int, default=0)
    p.add_argument('--checkpoint-interval-steps', type=int, default=500,
                   help='save resumable form_last.pt every N successful updates; 0 disables interval saves')
    p.add_argument('--condition-drop-prob', type=float, default=.1)
    p.add_argument('--blueprint-loss-weight', type=float, default=1.)
    p.add_argument('--load-curve-loss-weight', type=float, default=1.)
    p.add_argument('--relation-loss-weight', type=float, default=.1)
    p.add_argument('--quality-enabled', dest='quality_enabled', action='store_true', default=True)
    p.add_argument('--no-quality-enabled', dest='quality_enabled', action='store_false')
    p.add_argument('--max-memory-tokens', type=int, default=512)
    p.add_argument('--max-region-seconds', type=float, default=14.)
    p.add_argument('--meter-map', help='per-song music_structure.json may override this map')
    p.add_argument('--init-checkpoint')
    p.add_argument('--resume')
    p.add_argument('--precision', choices=('auto', 'fp32', 'bf16', 'fp16'), default='auto')
    p.add_argument('--loader-workers', type=int, default=0)
    p.add_argument('--prefetch-factor', type=int, default=2)
    p.add_argument('--audio-cache-gb', type=float, default=1.)
    p.add_argument('--sample-cache-mb', type=int, default=256)
    p.add_argument('--log-interval', type=float, default=10.)
    return p


def _validate(args):
    positive = ('epochs', 'batch_size', 'lr', 'grad_accumulation', 'prefetch_factor', 'log_interval')
    nonnegative = ('max_steps', 'limit_charts', 'weight_decay', 'prepare_workers', 'loader_workers',
                   'audio_cache_gb', 'sample_cache_mb', 'checkpoint_interval_steps', 'save_every',
                   'blueprint_loss_weight', 'load_curve_loss_weight', 'relation_loss_weight')
    for name in positive + nonnegative:
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0 or (name in positive and value == 0):
            raise ValueError(f'Invalid training setting: {name}')
    if not math.isfinite(args.condition_drop_prob) or not 0 <= args.condition_drop_prob <= 1:
        raise ValueError('condition_drop_prob must be in [0, 1]')
    if args.resume and args.init_checkpoint:
        raise ValueError('Choose resume or finetune, not both')


def layout_loss(prediction, batch, relation_weight):
    """Per-example objectives. Their sum can be accumulated across batch sizes."""
    logits = prediction['logits'].float()
    valid = batch['target'] != 0
    ce = F.cross_entropy(logits.transpose(1, 2), batch['target'], ignore_index=0, reduction='none')
    weights = batch['weight'].float()
    per_token = (ce * valid).sum(1) / valid.sum(1).clamp_min(1)
    relation = F.binary_cross_entropy_with_logits(
        prediction['relation_logits'].float(), batch['relation_target'].float(), reduction='none').mean(1)
    relation = relation * batch['relation_valid'].float() * weights
    token = per_token * weights
    loss_sum = (token + relation_weight * relation).sum()
    correct = ((logits.detach().argmax(-1) == batch['target']) & valid).sum()
    return loss_sum, {'token_loss': token.detach().mean(), 'relation_loss': relation.detach().mean(),
                      'loss': loss_sum.detach() / len(weights),
                      'token_accuracy': correct.float() / valid.sum().clamp_min(1)}


def _atomic_save(value, path):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def _blueprint_input(dataset, index, first_index):
    """CPU targets retain exact tempo/first; excluded spans get zero loss weight."""
    from maichart.form.blueprint import build_blueprint_targets
    from maichart.form.curve import resample_load_curve
    from maichart.event_timing import seconds_at_beat
    from maichart.generation.event_audio import segment_frame_times
    song, points, level, weight, _, condition, *rest = dataset.songs[index]
    audio = form_audio(song, song.spans[0].start, song.spans[-1].end)
    times = segment_frame_times(seconds_at_beat(song.spans[0].start, song.tempo, song.first),
                                seconds_at_beat(song.spans[-1].end, song.tempo, song.first))
    targets = build_blueprint_targets(points, song.tempo, song.first, times)
    curve = resample_load_curve(song.load_curve_times_sec, song.load_curve, times-float(song.first))
    targets['load_curve_target'] = torch.from_numpy(curve)
    targets['load_curve_mask'] = torch.ones_like(targets['load_curve_target'], dtype=torch.bool)
    frame_weight = np.ones(len(times), dtype=np.float32)
    record = rest[1] if len(rest) > 1 else None
    for region in (record or {}).get('regions', []):
        selected = (times >= float(region['start_sec'])) & (times < float(region['end_sec']))
        frame_weight[selected] *= float(region.get('weight', 1.))
    targets['frame_weight'] = torch.from_numpy(frame_weight)
    inputs = {'audio': audio[None], 'form': dataset[first_index]['form'][None],
              'condition': condition.to_vector()[None]}
    validate_cpu_batch(inputs)
    validate_cpu_batch(targets)
    return inputs, {k: v[None] for k, v in targets.items()}, float(weight)


def main(argv=None):
    args = parser().parse_args(argv)
    _validate(args)
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if args.device == 'auto' else args.device)
    meters = json.loads(Path(args.meter_map).read_text(encoding='utf-8-sig')) if args.meter_map else None
    old = None
    if args.resume or args.init_checkpoint:
        model, blueprint, codec, old = checkpoint.load(args.resume or args.init_checkpoint, device='cpu')
        for key in ('max_tokens', 'max_memory_tokens', 'd_model', 'layers', 'nhead', 'dropout'):
            setattr(args, key, getattr(model.config, key))
        if args.resume:
            saved_args = old.get('args') or {}
            for name in ('seed', 'weight_decay', 'grad_accumulation', 'condition_drop_prob',
                         'blueprint_loss_weight', 'load_curve_loss_weight', 'relation_loss_weight',
                         'quality_enabled', 'max_region_seconds', 'limit_charts', 'batch_size', 'max_steps'):
                if name in saved_args:
                    setattr(args, name, saved_args[name])
            # Resume retains the optimizer's numerical regime. Old trainers
            # were FP32; performance options remain adjustable during finetune.
            args.precision = old.get('precision', saved_args.get('precision', 'fp32'))
            meters = (old.get('dataset_contract') or {}).get('meter_map', meters)
    _validate(args)
    precision = resolve_precision(args.precision, device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    output = Path(args.checkpoint_dir)
    if old and args.init_checkpoint and Path(args.init_checkpoint).resolve().parent == output.resolve():
        raise ValueError('Finetune uses a separate output directory to preserve the source model')
    output.mkdir(parents=True, exist_ok=True)
    emit('form_runtime', training_semantics=TRAINING_SEMANTICS, device=str(device), precision=precision,
         batch_size=args.batch_size, grad_accumulation=args.grad_accumulation,
         effective_batch_size=args.batch_size*args.grad_accumulation, loader_workers=args.loader_workers,
         prepare_workers=args.prepare_workers, prefetch_factor=args.prefetch_factor,
         cache_budget_scope='per_process', audio_cache_gb=args.audio_cache_gb,
         sample_cache_mb=args.sample_cache_mb, checkpoint_interval_steps=args.checkpoint_interval_steps)
    if args.loader_workers:
        emit('form_warning', message='Loader workers copy dataset metadata and each has its own cache; memory grows with worker count.')
    if old and old.get('training_semantics') != TRAINING_SEMANTICS:
        emit('form_warning', message='Checkpoint predates accumulation/curve-loss corrections. Weights remain compatible; this run uses corrected training semantics.',
             optimizer_restored=bool(args.resume))
    emit('form_prepare', stage='dataset', manifest=str(Path(args.manifest).resolve()),
         limit_charts=args.limit_charts, max_steps=args.max_steps, prepare_workers=args.prepare_workers)
    dataset = FormDataset(args.manifest, cache_dir=args.cache_dir, max_tokens=args.max_tokens,
                          max_memory_tokens=args.max_memory_tokens, meters=meters, seed=args.seed,
                          limit_charts=args.limit_charts, max_region_seconds=args.max_region_seconds,
                          quality_enabled=args.quality_enabled, preparation_workers=args.prepare_workers,
                          audio_cache_gb=args.audio_cache_gb, sample_cache_mb=args.sample_cache_mb)
    contract = {'manifest_sha256': hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
                'meter_map': meters, 'max_region_seconds': args.max_region_seconds, 'charts': dataset.audit['charts']}
    if args.resume and old.get('dataset_contract') != contract:
        raise ValueError('Resume data changed; use --init-checkpoint')
    from maichart.form.blueprint import BlueprintConfig, SongBlueprint, compute_blueprint_loss
    if old is None:
        codec = dataset.codec
        model = FormTransformer(FormConfig(vocab_size=len(codec), d_model=args.d_model, nhead=args.nhead,
                    dropout=args.dropout, layers=args.layers, dim_feedforward=4*args.d_model,
                    max_tokens=args.max_tokens, max_memory_tokens=args.max_memory_tokens))
        blueprint = SongBlueprint(BlueprintConfig(width=max(16, args.d_model//2)))
    model.to(device); blueprint.to(device)
    # Inputs are checked on CPU before copying. Shapes are always checked in
    # the model; avoid per-forward GPU -> host scalar synchronizations.
    model.validate_inputs = False
    blueprint.validate_inputs = False
    parameters = list(model.parameters()) + list(blueprint.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    if args.resume:
        optimizer.load_state_dict(old['optimizer_state_dict'])
    engine = UpdateEngine(optimizer, parameters, device, precision)
    if args.resume and old.get('grad_scaler_state') and engine.scaler.is_enabled():
        engine.scaler.load_state_dict(old['grad_scaler_state'])
    steps = int(old.get('optimizer_steps', 0)) if old else 0
    plan_steps = int(old.get('blueprint_steps', 0)) if old else 0
    task_training = dict(old.get('task_training', {})) if old else {'draft': 0, 'revision': 0}
    capabilities = dict(old.get('capabilities', {})) if old and old.get('training_semantics') == TRAINING_SEMANTICS else {}
    history = list(old.get('training_history', [])) if args.resume else []
    start_epoch = int(old.get('epoch', 0)) if args.resume else 0
    cursor = dict(old.get('training_position') or {}) if args.resume else {}
    first_items = {}
    for i, item in enumerate(dataset.items):
        first_items.setdefault(item[0], i)
    plan_indices = [i for i in range(len(dataset.songs)) if i in first_items]
    if args.max_steps:
        plan_indices = plan_indices[:args.max_steps]
    plan_total = len(plan_indices)
    layout_total = min(math.ceil(len(dataset)/args.batch_size), args.max_steps or math.ceil(len(dataset)/args.batch_size))
    (output/'dataset_audit.json').write_text(json.dumps(dataset.audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'form_training charts={len(dataset.songs)} examples={len(dataset)} device={device} parameters={sum(p.numel() for p in parameters)}', flush=True)
    emit('form_dataset', charts=len(dataset.songs), examples=len(dataset), draft_examples=sum(item[3] == 0 for item in dataset.items),
         revision_examples=sum(item[3] == 1 for item in dataset.items), layout_batches=layout_total,
         blueprint_charts=plan_total, optimizer_updates_per_layout_epoch=math.ceil(layout_total/args.grad_accumulation),
         skipped_reasons=dict(Counter(str(item.get('reason', 'unknown')) for item in dataset.audit.get('skipped', []))),
         prepare_elapsed_sec=round(time.perf_counter()-started, 2))
    atoms = sorted({atom for _, points, *_ in dataset.songs for point in points for atom in _atoms(point.note_text)}
                   | set(old.get('atom_catalog', []) if old else []))
    last_save_steps = steps + plan_steps
    progress = None
    metrics = {}

    def save(epoch, phase, completed, *, complete=False, reason='interval'):
        nonlocal last_save_steps
        if engine.samples:
            raise ValueError('Checkpoint requested inside an accumulation group')
        tick = time.perf_counter()
        saved = checkpoint.payload(model, blueprint, codec, epoch=epoch if complete else epoch-1,
            optimizer_steps=steps, blueprint_steps=plan_steps, task_training=dict(task_training),
            capabilities={k:dict(v) for k,v in capabilities.items()}, optimizer_state_dict=optimizer.state_dict(),
            grad_scaler_state=engine.scaler.state_dict(), dataset_contract=contract, args=vars(args),
            atom_catalog=atoms, metrics=metrics, training_history=history,
            condition_supervision='event_derived_radar_style_v1', training_semantics=TRAINING_SEMANTICS,
            precision=precision,
            training_position={'epoch':epoch, 'phase':'complete' if complete else phase, 'completed':completed,
                               'layout_total':layout_total, 'plan_total':plan_total, 'batch_size':args.batch_size,
                               'max_steps':args.max_steps}, rng_state=capture_rng(device),
            supervision='authored_draft_and_masked_reconstruction_not_aesthetic_reward')
        _atomic_save(saved, output/'form_last.pt')
        if complete and args.save_every and epoch % args.save_every == 0:
            _atomic_save(saved, output/f'form_epoch_{epoch:03d}.pt')
        elapsed = time.perf_counter()-tick
        if progress:
            progress.timing['checkpoint_sec'] += elapsed
        last_save_steps = steps + plan_steps
        emit('form_checkpoint', path=str(output/'form_last.pt'), epoch=epoch, phase=phase,
             batch=completed, optimizer_steps=steps, blueprint_steps=plan_steps,
             epoch_complete=complete, resumable=True, reason=reason, elapsed_sec=round(elapsed, 3))

    def increment_cap(name, amount, contract_name):
        value = capabilities.setdefault(name, {'steps': 0, 'contract': contract_name})
        value['steps'] += amount
        value['contract'] = contract_name

    if args.resume:
        restore_rng(old.get('rng_state'), device)
        if cursor and cursor.get('phase') != 'complete':
            if cursor.get('layout_total') != layout_total or cursor.get('plan_total') != plan_total or cursor.get('max_steps') != args.max_steps:
                raise ValueError('Mid-epoch resume requires identical batch/step limits; use finetune for a new run')
            emit('form_resume', **cursor, optimizer_steps=steps, blueprint_steps=plan_steps)
    for epoch in range(start_epoch+1, start_epoch+args.epochs+1):
        dataset.seed = args.seed + epoch*100003
        model.train(); blueprint.train()
        progress = Progress(epoch, plan_total, layout_total, args.log_interval, device, precision)
        plan_start = layout_start = 0
        if cursor.get('epoch') == epoch and cursor.get('phase') != 'complete':
            if cursor.get('phase') == 'blueprint':
                plan_start = int(cursor['completed'])
            elif cursor.get('phase') == 'layout_revision':
                plan_start = plan_total
                layout_start = int(cursor['completed'])
        phase_loss_sums = dict(plan_loss=0., layout_loss=0.)
        phase_counts = dict(plan_loss=0, layout_loss=0)
        progress.begin('blueprint', plan_start)
        for position in range(plan_start, plan_total):
            index = plan_indices[position]
            before = time.perf_counter()
            inputs, targets, weight = _blueprint_input(dataset, index, first_items[index])
            progress.timing['data_wait_sec'] += time.perf_counter()-before
            tick = time.perf_counter()
            inputs = {k:v.to(device, non_blocking=True) for k,v in inputs.items()}
            targets = {k:v.to(device, non_blocking=True) for k,v in targets.items()}
            progress.timing['h2d_sec'] += time.perf_counter()-tick
            tick = time.perf_counter()
            with engine.autocast():
                prediction = blueprint(inputs['audio'], inputs['form'], inputs['condition'])
            parts = compute_blueprint_loss({k:v.float() for k,v in prediction.items()}, targets,
                                           load_curve_weight=args.load_curve_loss_weight)
            loss = parts['loss'] * weight * args.blueprint_loss_weight
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite blueprint loss')
            if args.blueprint_loss_weight > 0 and weight > 0 and bool(targets['frame_weight'].sum() > 0):
                engine.backward(loss, 1)
                update = engine.step()
                if update['applied']:
                    plan_steps += 1
                    if args.load_curve_loss_weight > 0:
                        increment_cap('load_curve', 1, 'load_curve_v1')
                else:
                    emit('form_warning', message='AMP overflow: blueprint update skipped', epoch=epoch, chart=position+1)
            progress.timing['compute_sec'] += time.perf_counter()-tick
            progress.add(parts)
            phase_loss_sums['plan_loss'] += loss.detach()
            phase_counts['plan_loss'] += 1
            progress.phase_samples += 1
            if args.checkpoint_interval_steps and steps+plan_steps-last_save_steps >= args.checkpoint_interval_steps:
                save(epoch, 'blueprint', position+1)
            progress.log(position+1, plan_total, optimizer_steps=steps, blueprint_steps=plan_steps,
                         lr=optimizer.param_groups[0]['lr'])
        if plan_start < plan_total:
            save(epoch, 'layout_revision', 0, reason='blueprint_complete')
        progress.begin('layout_revision', layout_start)
        sampler = EpochBatches(len(dataset), args.batch_size, args.seed, epoch, start=layout_start, limit=args.max_steps)
        loader_options = dict(batch_sampler=sampler, collate_fn=collate_form, num_workers=args.loader_workers,
                              pin_memory=device.type=='cuda', generator=torch.Generator().manual_seed(args.seed+epoch))
        if args.loader_workers:
            loader_options['prefetch_factor'] = args.prefetch_factor
            loader_options['multiprocessing_context'] = 'spawn'
        loader = DataLoader(dataset, **loader_options)
        iterator = iter(loader)
        pending_tasks = Counter()
        pending_cap_tasks = Counter()
        last_update = {}
        for batch_index in range(layout_start, layout_total):
            tick = time.perf_counter()
            batch = next(iterator)
            validate_cpu_batch(batch)
            batch_size = len(batch['task'])
            tokens = int((batch['target'] != 0).sum())
            task_counts = Counter(batch['task'].tolist())
            progress.timing['data_wait_sec'] += time.perf_counter()-tick
            tick = time.perf_counter()
            batch = {k:v.to(device, non_blocking=device.type=='cuda') if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
            progress.timing['h2d_sec'] += time.perf_counter()-tick
            tick = time.perf_counter()
            if args.condition_drop_prob:
                batch['condition_null'] = torch.rand(batch_size, device=device) < args.condition_drop_prob
            with engine.autocast():
                prediction = model(batch)
            loss_sum, parts = layout_loss(prediction, batch, args.relation_loss_weight)
            if not torch.isfinite(loss_sum):
                raise ValueError('Nonfinite layout loss')
            engine.backward(loss_sum, batch_size)
            pending_tasks.update(task_counts)
            for task_id in (0, 1):
                mask = (batch['task'] == task_id) & ~batch['condition_null'] & (batch['weight'] > 0)
                pending_cap_tasks[task_id] += mask.sum().detach()
            update = None
            if engine.microbatches >= args.grad_accumulation or batch_index+1 == layout_total:
                update = engine.step()
                last_update = update
                if update['applied']:
                    steps += 1
                    for task_id, name in ((0, 'draft'), (1, 'revision')):
                        task_training[name] = task_training.get(name, 0) + pending_tasks[task_id]
                        conditioned_count = int(pending_cap_tasks[task_id])
                        if conditioned_count:
                            increment_cap(name, conditioned_count, 'form_condition_v2')
                else:
                    emit('form_warning', message='AMP overflow: accumulated layout update skipped', epoch=epoch, batch=batch_index+1)
                pending_tasks.clear(); pending_cap_tasks.clear()
            progress.timing['compute_sec'] += time.perf_counter()-tick
            progress.phase_samples += batch_size
            progress.phase_tokens += tokens
            progress.add(parts, batch_size, weights={'token_accuracy': tokens})
            phase_loss_sums['layout_loss'] += parts['loss']*batch_size
            phase_counts['layout_loss'] += batch_size
            if update and args.checkpoint_interval_steps and steps+plan_steps-last_save_steps >= args.checkpoint_interval_steps:
                save(epoch, 'layout_revision', batch_index+1)
            progress.log(batch_index+1, layout_total, optimizer_steps=steps, blueprint_steps=plan_steps,
                         accumulation_pending=engine.microbatches, grad_accumulation=args.grad_accumulation,
                         lr=optimizer.param_groups[0]['lr'], cache=dataset.cache_stats(),
                         cache_scope='main_process_only' if args.loader_workers else 'training_process',
                         grad_norm=last_update.get('grad_norm'), amp_scale=last_update.get('amp_scale'))
        del iterator, loader
        metrics = {'epoch':epoch,
                   'draft_revision_loss':float(phase_loss_sums['layout_loss'])/max(1,phase_counts['layout_loss']),
                   'whole_song_plan_loss':float(phase_loss_sums['plan_loss'])/max(1,phase_counts['plan_loss']),
                   'optimizer_steps':steps, 'blueprint_steps':plan_steps, 'task_training':dict(task_training),
                   'elapsed_sec':round(time.perf_counter()-progress.started,2),
                   'partial_epoch_metrics':bool(plan_start or layout_start)}
        history.append(metrics)
        save(epoch, 'layout_revision', layout_total, complete=True, reason='epoch_complete')
        (output/'training_history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        emit('form_epoch', **metrics)
        cursor = {}
    return 0


def _atoms(text):
    from maichart.notes import _split_top_level_slashes
    return _split_top_level_slashes(text)


if __name__ == '__main__':
    raise SystemExit(main())
