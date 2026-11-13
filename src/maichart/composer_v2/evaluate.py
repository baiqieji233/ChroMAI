"""Read-only, CPU diagnostics for Composer v2; never starts generation or training."""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
import statistics
from types import SimpleNamespace
from pathlib import Path
import shutil
import tempfile
import time
from unittest.mock import patch


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def stable_checkpoint(path):
    """Detach tensors from a live atomic-replaced checkpoint and verify the copy."""
    from .checkpoint import inspect_payload
    import torch
    path = Path(path).resolve()
    before = file_sha256(path)
    with tempfile.TemporaryDirectory(prefix='composer-v2-eval-') as directory:
        snapshot = Path(directory) / 'checkpoint.pt'
        shutil.copyfile(path, snapshot)
        copied = file_sha256(snapshot)
        payload = torch.load(snapshot, map_location="cpu", weights_only=True)
        after = file_sha256(path)
        if before != copied or copied != after:
            raise RuntimeError('Checkpoint changed while reading; evaluate an epoch snapshot or retry')
    inspection = inspect_payload(payload)
    if not inspection['valid']:
        raise ValueError(inspection['error'])
    return payload, dict(path=str(path), sha256=copied, sha256_before=before, sha256_after=after,
                         stable_snapshot=True,epoch=inspection.get('epoch'),optimizer_steps=inspection.get('optimizer_steps'))


def overlap_audit(manifest, training_manifests):
    from maichart.composer.evaluation import assert_unseen
    results = []
    for training in training_manifests:
        item = dict(path=str(Path(training).resolve()), sha256=file_sha256(training))
        try:
            assert_unseen(training, manifest)
            item['status'] = 'disjoint_from_supplied_manifest'
        except ValueError as error:
            item.update(status='overlap_or_invalid', detail=str(error))
        results.append(item)
    return dict(status='unseen_not_verified', supplied_training_manifests=results,
                limitation='Supplied manifests do not prove coverage of all parent checkpoint training data')


def _summary(values):
    count = values['count']
    return dict(token_count=count, nll=values['nll_sum'] / count if count else None,
                accuracy=values['correct'] / count if count else None)


def sample_window_indices(count, budget, sampling='spread'):
    """Stable inclusive coverage; a zero budget means every window."""
    import numpy as np
    size = min(count, budget) if budget else count
    if not size:
        return []
    return list(range(size)) if sampling == 'prefix' else np.linspace(0, count - 1, size, dtype=int).tolist()


def audio_macro_summary(per_song):
    """Each measured audio identity has equal weight, regardless of note count."""
    result = {}
    names = {name for metrics in per_song.values() for name in metrics}
    for name in sorted(names):
        values = [_summary(metrics[name]) for metrics in per_song.values()
                  if name in metrics and metrics[name]['count'] > 0]
        result[name] = dict(audio_count=len(values),
            nll=statistics.mean(value['nll'] for value in values) if values else None,
            accuracy=statistics.mean(value['accuracy'] for value in values) if values else None)
    return result


def timing_rollout(model, dataset, song, begin):
    """Sample only the absolute time grammar; never decode note actions."""
    import torch
    from .stream_generate import EventSampler
    codec = model.codec
    prefix = list(song.points[:begin])
    after = prefix[-1].beat if prefix else Fraction(-1)
    truth = song.points[begin].beat if begin < len(song.points) else None
    args = SimpleNamespace(temperature=0., top_p=1., action_temperature=None, action_top_p=None,
                           onset_strength=None, intensity_strength=1.)
    sentinel = object(); old_profile = getattr(dataset, 'intensity_profile', sentinel)
    if old_profile is not sentinel:
        del dataset.intensity_profile
    wall = time.perf_counter(); cpu = time.process_time(); tokens = []
    try:
        # EventSampler builds training=False with exactly this committed prefix;
        # build_stream reconstructs sustain from prefix, never song.sustain.
        sampler = EventSampler(model, dataset, song, prefix, torch.device('cpu'),
                               torch.Generator(device='cpu').manual_seed(0), args)
        for _ in range(model.config.max_tokens // 2):
            allowed = codec.time_choices(tokens, after, song.end_beat)
            token = sampler.next(tokens, allowed)
            tokens.append(token)
            if token == codec.eos_id or token in codec.phases:
                break
        else:
            raise ValueError('Time grammar exceeded its finite token budget')
        predicted = None
        if tokens[-1] != codec.eos_id:
            split = tokens.index(codec.phase_id)
            integer = int(''.join(codec.digits[token] for token in tokens[1:split]))
            predicted = Fraction(integer) + codec.phases[tokens[-1]]
        whole_correct = (predicted.numerator // predicted.denominator == truth.numerator // truth.denominator
                         if predicted is not None and truth is not None else None)
        return dict(status='measured', true_next_beat=str(truth) if truth is not None else None,
            predicted_beat=str(predicted) if predicted is not None else None, predicted_eos=predicted is None,
            true_chart_end=truth is None, integer_beat_correct=whole_correct,
            exact_phase_correct=(predicted % 1 == truth % 1) if predicted is not None and truth is not None else None,
            joint_exact_beat=(predicted == truth),
            signed_error_sec=(song.clock.at(predicted) - song.clock.at(truth))
                if predicted is not None and truth is not None else None,
            skipped_true_next=bool(predicted is not None and truth is not None and predicted > truth),
            early_next=bool(predicted is not None and truth is not None and predicted < truth),
            premature_eos=bool(predicted is None and truth is not None),
            time_token_ids=tokens, committed_prefix_events=len(prefix), after_beat=str(after),
            audio_end_beat=str(song.end_beat), cpu_sec=time.process_time() - cpu,
            wall_sec=time.perf_counter() - wall, temperature=0., top_p=1., seed=0)
    finally:
        if old_profile is sentinel:
            if hasattr(dataset, 'intensity_profile'): del dataset.intensity_profile
        else:
            dataset.intensity_profile = old_profile


def teacher_forced(args):
    import torch
    from torch.nn import functional as F
    from .audio import AudioFeatureStore
    from .codec import ComponentCodec
    from .conditions import resolve_condition
    from .data import _beats_at_seconds
    from .model import ComposerV2, ComposerV2Config
    from .stream_data import StreamDataset, collate_stream
    from .timing_metrics import phase_family,rhythm_profile

    torch.set_num_threads(args.cpu_threads)
    payload, checkpoint = stable_checkpoint(args.checkpoint)
    config = ComposerV2Config.from_dict(payload['model_config'])
    if config.sequence_policy != 'continuous_events_v1':
        raise ValueError('This evaluator requires the continuous-events StreamDataset contract')
    codec = ComponentCodec.from_dict(payload['codec'])
    store = AudioFeatureStore(args.cache_dir, config.audio_backend, config.mert_model, 'cpu')
    # Bypass writable source-digest SQLite even when an existing cache is present.
    store._source_digests = SimpleNamespace(digest=file_sha256)
    contract = payload.get('training_contract') or {}
    options = dict(cache_dir=args.cache_dir, codec=codec, prepare=False, audio_store=store,
                   limit_charts=args.limit_charts, timing_recovery_policy='off', reference_augmentation=False,
                   quality_enabled=contract.get('quality_enabled', True))
    for key in ('window_beats', 'history_beats', 'future_beats', 'ticks_per_beat', 'max_tokens',
                'audio_backend', 'mert_model', 'condition_policy', 'refresh_events',
                'onset_readout_policy', 'structure_policy', 'intensity_policy','music_context_policy'):
        options[key] = getattr(config, key)
    from . import chart_dialect
    options['dialect_policy'] = contract.get('chart_dialect', chart_dialect.LEGACY_STRICT_POLICY)
    from .difficulty import FITTED_SOURCES, make_scorer, resolve_selection
    selection = resolve_selection(parent=contract, resume=True)
    options.update(selection.dataset_options())
    if selection.source in FITTED_SOURCES:
        scorer = make_scorer(selection.source,None,args.cache_dir,read_only=True)
        if selection.contract(scorer) != contract.get('difficulty_selection'):
            raise ValueError('当前版本无法读取此检查点使用的拟合定数缓存')
        options['difficulty_scorer'] = scorer
    # Parsing is permitted, but evaluation must not populate any persistent chart cache.
    with patch('maichart.composer_v2.chart_cache.ChartCache.save', return_value=None):
        dataset = StreamDataset(args.manifest, **options)
    for song in dataset.songs:
        song.audio = store.load(song.audio_path)  # No prepare/build/fallback.
        duration = song.audio.duration_sec
        if any(song.clock.at(point.beat) >= duration for point in song.points):
            raise ValueError(f'Authored onset outside audio extent: {song.source_path}')
        song.summary = dict(song.audio.summary(160))
        song.summary['beats'] = _beats_at_seconds(song.clock, song.summary['times_sec'])
        song.end_beat = Fraction(str(float(_beats_at_seconds(song.clock, [duration])[0])))
        song.condition, song.condition_known = resolve_condition(
            song.level, policy=config.condition_policy, max_level=dataset.max_condition_level)
    model = ComposerV2(config, codec).cpu().eval()
    model.load_state_dict(payload['model_state_dict'], strict=True)
    totals = {key: Counter() for key in ('all', 'time', 'phase', 'time_digits_true_prefix')}
    buckets = {}; target_buckets={};confusion = Counter(); selection = []; per_song = {}; rollouts = []
    from .validation_sampling import balanced_window_indices,coverage_report,POLICY as COVERAGE_POLICY
    indices = (balanced_window_indices(dataset,args.max_batches) if args.window_sampling=='spread'
               else sample_window_indices(len(dataset.windows),args.max_batches,'prefix'))
    rollout_positions = set(sample_window_indices(len(indices), args.timing_rollouts, 'spread')) if args.timing_rollouts else set()
    # A batch is exactly one deterministic computational slice, never a song replicate.
    try:
        with torch.inference_mode():
            for sample_position, index in enumerate(indices):
                song_index, begin = dataset.windows[index]
                song = dataset.songs[song_index]; end = dataset.ranges[index]
                cursor = song.points[begin - 1].beat if begin else Fraction(0)
                item = dataset.build_stream(song, song.points, cursor, target=song.points[begin:end],
                    ended=end == len(song.points), training=True,
                    reference_start=dataset._reference_argument(index), _prefix_end=begin)
                batch = collate_stream([item])
                logits = model(batch)['logits'].float(); labels = batch['labels']
                if not torch.isfinite(logits).all():
                    raise ValueError('Non-finite evaluation logits; refusing an invalid metric report')
                # Zero-supervision regions remain context, never metric targets.
                active = labels.ne(-100) & batch['token_weights'].gt(0)
                losses = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100,
                                         reduction='none').reshape_as(labels)
                predicted = logits.argmax(-1); safe = labels.clamp_min(0)
                phase = model.phase_targets[safe] & active
                # NUM tokens are time digits only between [AT] and [PHASE].
                digit = torch.zeros_like(active)
                for row in range(labels.size(0)):
                    reading = False
                    for column, token in enumerate(labels[row].tolist()):
                        if token == codec.at_id: reading = True
                        elif token == codec.phase_id: reading = False
                        elif reading and token in codec.digits: digit[row, column] = True
                timing = digit | phase | labels.eq(codec.at_id) | labels.eq(codec.phase_id) | labels.eq(codec.eos_id)
                song_key = song.audio.identity['audio_sha256']
                song_metrics = per_song.setdefault(song_key, {key: Counter() for key in totals})
                for key, mask in (('all', active), ('time', timing & active),
                                  ('phase', phase), ('time_digits_true_prefix', digit & active)):
                    values = dict(count=int(mask.sum()), nll_sum=float(losses[mask].sum()),
                                  correct=int((predicted.eq(labels) & mask).sum()))
                    totals[key].update(values); song_metrics[key].update(values)
                visible = item['input_ids'].tolist()
                prefix_end = visible.index(codec.tgt_id)
                families = {phase_family(codec.phases[token]) for token in visible[:prefix_end]
                            if token in codec.phases}
                bucket = 'binary_ternary_mixed' if {'binary', 'ternary'} <= families else 'other_context'
                counts = buckets.setdefault(bucket, Counter())
                counts.update(count=int(phase.sum()), nll_sum=float(losses[phase].sum()),
                              correct=int((predicted.eq(labels) & phase).sum()))
                target_profile=rhythm_profile(song.points[begin:end])
                target_bucket='mixed_16_24' if target_profile['mixed_16_24_windows'] else 'other_target'
                target_counts=target_buckets.setdefault(target_bucket,Counter())
                target_counts.update(count=int(phase.sum()),nll_sum=float(losses[phase].sum()),
                    correct=int((predicted.eq(labels)&phase).sum()))
                for truth, guess in zip(labels[phase].tolist(), predicted[phase].tolist()):
                    left = phase_family(codec.phases[truth])
                    right = phase_family(codec.phases[guess]) if guess in codec.phases else 'non_phase_token'
                    confusion[(left, right)] += 1
                if sample_position in rollout_positions:
                    phase_positions = labels[0].ne(-100) & model.phase_targets[labels[0].clamp_min(0)]
                    approved = bool(active[0][phase_positions][0]) if bool(phase_positions.any()) else not song.points
                    if approved:
                        rollouts.append(dict(window_index=index, song_id=song.song_id, source=str(song.source_path),
                            begin_event=begin, **timing_rollout(model, dataset, song, begin)))
                    else:
                        rollouts.append(dict(window_index=index, source=str(song.source_path), begin_event=begin,
                                             status='not_measured', reason='true_next_phase_has_zero_supervision_weight'))
                selection.append(dict(window_index=index, song_id=song.song_id, source=str(song.source_path),
                    chart_sha256=song.chart_sha256, difficulty=song.difficulty, begin_event=begin, end_event=end,
                    context_bucket=bucket,target_bucket=target_bucket))
    finally:
        dataset.close()
    return dict(schema='composer_v2_timing_evaluation_v1', mode='teacher_forced', device='cpu',
        checkpoint=checkpoint, model_config=config.to_dict(),
        manifest=dict(path=str(Path(args.manifest).resolve()), sha256=dataset.manifest_sha256),
        overlap=overlap_audit(args.manifest, args.training_manifest), condition_known='level_only',
        supervision='Unweighted positive-supervision tokens only; zero-weight reviewed/unmarked regions excluded',
        augmentation='off', ordering='manifest_chart_order_then_event_order', batch_size=1,
        budget=dict(limit_charts=args.limit_charts, max_batches=args.max_batches, timing_rollouts=args.timing_rollouts),
        window_sampling=args.window_sampling, sample_selection=selection,
        sample_scope='Small diagnostic over the selected charts; chart limits restrict the manifest prefix before audio-balanced window sampling',
        coverage=coverage_report(dataset,indices,COVERAGE_POLICY if args.window_sampling=='spread' else 'manifest_prefix_v1'),
        metrics={key: _summary(value) for key, value in totals.items()},
        audio_macro_metrics=audio_macro_summary(per_song),
        per_song=[dict(audio_sha256=key, metrics={name: _summary(value) for name, value in metrics.items()})
                  for key, metrics in sorted(per_song.items())],
        phase_context_buckets={key: _summary(value) for key, value in buckets.items()},
        phase_target_buckets={key: _summary(value) for key,value in target_buckets.items()},
        target_bucket_scope='Authored target rhythms used only to stratify evaluation, never passed as a new model input',
        context_bucket_scope='Exact F phase tokens in the visible carry/history/reference prefix before TGT',
        phase_family_confusion=[dict(true=a, predicted=b, count=n) for (a,b),n in sorted(confusion.items())],
        phase_scope='F token under true integer beat and full true prefix; unconstrained vocabulary argmax',
        time_digits_scope='Single token accuracy under true prefix; not whole-event timing accuracy',
        joint_timing=dict(status='measured' if any(row.get('status') == 'measured' for row in rollouts) else 'not_measured',
            samples=rollouts, scope='Greedy next-time head under real past only; not full-chart generation or musical quality'),
        aggregation='token micro averages plus equal-audio macro averages; slices and copied charts are not independent songs',
        dataset_audit=dataset.audit)


def _read_chart(path, difficulty):
    from maichart.maidata import decode_text_with_encoding, parse_maidata_metadata
    from maichart.event_timing import _parse_event_metadata
    from maichart.tempo_clock import TempoClock
    text, _ = decode_text_with_encoding(Path(path).read_bytes())
    chart = _parse_event_metadata(parse_maidata_metadata(text), difficulty, strict_notes=False)
    return chart, TempoClock(chart.tempo_changes, chart.audio_first)


def seed_variance(results, reference_sha256, difficulty):
    """Group only complete recorded provenance; never infer identity from file paths."""
    groups = {}; rejected = []
    for row in results:
        meta = row.get('composer_v2') or {}
        settings = meta.get('generation_settings')
        checkpoint_hash = meta.get('checkpoint_sha256') or (meta.get('checkpoint_info') or {}).get('sha256')
        audio_hash = (meta.get('audio_identity') or {}).get('audio_sha256')
        level = meta.get('level', (settings or {}).get('level'))
        condition = meta.get('condition'); timing = meta.get('timing'); seed = meta.get('seed')
        missing = [key for key, value in dict(audio_sha256=audio_hash, checkpoint_sha256=checkpoint_hash,
            level=level, condition=condition, generation_settings=settings, timing=timing,
            seed=seed, tempo_map=row.get('parsed_tempo_map'), first=row.get('parsed_first')).items()
            if value is None]
        if not isinstance(settings, dict): missing.append('generation_settings_dict')
        else:
            missing.extend('generation_settings.' + key for key in ('temperature', 'top_p', 'candidates')
                           if settings.get(key) is None)
        if not condition: missing.append('condition_values')
        if not timing: missing.append('timing_source')
        if missing:
            rejected.append(dict(path=row['path'], status='not_verified', missing=sorted(set(missing))))
            continue
        controls = {key: value for key, value in settings.items() if key != 'seed'}
        # Cache hits, residual summaries and elapsed timings do not define a
        # different tempo condition. The actual exported clock is also checked.
        timing_identity={key:timing.get(key) for key in ('tempo_source','first_source','tempo_detection',
            'algorithm_policy','effective_bpm','effective_first','effective_tempo_map')}
        signature = dict(reference_sha256=reference_sha256, difficulty=difficulty,
            audio_sha256=audio_hash, checkpoint_sha256=checkpoint_hash, level=level,
            condition=condition, sampling_controls=controls, timing_source=timing_identity,
            tempo_map=row['parsed_tempo_map'], first=row['parsed_first'])
        key = json.dumps(signature, sort_keys=True, allow_nan=False)
        groups.setdefault(key, []).append(row)
    output = []
    for signature, rows in groups.items():
        seeds = [row['composer_v2']['seed'] for row in rows]
        entry = dict(provenance=json.loads(signature), paths=[row['path'] for row in rows], seeds=seeds,
                     variance='not_measured')
        # Repeated renders of a seed are not independent random trials.
        if len(seeds) >= 2 and len(set(seeds)) == len(seeds):
            f1 = {key: [row['metrics']['whole']['matching'][key]['f1'] for row in rows]
                  for key in ('10ms', '25ms')}
            counts = [row['metrics']['whole']['matching']['10ms']['generated_onsets'] for row in rows]
            entry['variance'] = dict(status='measured', variance_estimator='population',
                macro_f1={key: dict(mean=statistics.mean(values), variance=statistics.pvariance(values))
                          if all(value is not None for value in values) else dict(mean=None, variance=None)
                          for key, values in f1.items()},
                event_count=dict(mean=statistics.mean(counts), variance=statistics.pvariance(counts)))
        else:
            entry['reason'] = 'Requires at least two distinct seeds, with one artifact per seed'
        output.append(entry)
    return dict(status='grouped_complete_provenance' if groups else 'not_verified', groups=output,
                excluded=rejected, limitation='Provenance metadata is not independent verification; agreement is not musical quality')


def compare(args):
    from .timing_metrics import compare_charts
    reference, reference_clock = _read_chart(args.reference, args.difficulty)
    audio=None
    if getattr(args,'audio',None):
        from .audio import AudioFeatureStore
        store=AudioFeatureStore(args.cache_dir,args.audio_backend,None,'cpu')
        store._source_digests=SimpleNamespace(digest=file_sha256)
        try:audio=store.load(Path(args.audio).resolve())
        finally:store.close()
    duration = args.duration_sec
    if duration is None:
        duration = audio.duration_sec if audio is not None else float(reference_clock.at(reference.end_beat))
    if not math.isfinite(duration) or duration < 0:
        raise ValueError('Comparison duration must be finite and nonnegative; provide --duration-sec')
    results = []
    for path in args.generated:
        generated, clock = _read_chart(path, args.difficulty)
        sidecar = Path(path).with_suffix('.composer_v2.json')
        metadata = json.loads(sidecar.read_text(encoding='utf-8-sig')) if sidecar.is_file() else None
        identity=(metadata or {}).get('audio_identity') or {}
        if audio is not None and identity.get('audio_sha256') and identity['audio_sha256']!=audio.identity['audio_sha256']:
            raise ValueError('Generated chart metadata refers to a different audio file')
        results.append(dict(path=str(Path(path).resolve()), sha256=file_sha256(path),
            composer_v2=metadata, parsed_first=generated.audio_first,
            parsed_tempo_map=[dict(beat=str(change.beat), bpm=change.bpm) for change in generated.tempo_changes],
            metrics=compare_charts(reference.points, reference_clock,
                generated.points, clock, duration,audio=audio)))
    seed_report = seed_variance(results, file_sha256(args.reference), args.difficulty)
    return dict(schema='composer_v2_timing_evaluation_v1', mode='compare',
        seed_comparability=seed_report,
        reference=dict(path=str(Path(args.reference).resolve()), sha256=file_sha256(args.reference)),
        difficulty=args.difficulty, duration_sec=duration,
        duration_source='explicit' if args.duration_sec is not None else 'cached_audio' if audio is not None else 'reference_chart_end',
        audio_identity=audio.identity if audio is not None else None,
        generated=results, song_count=1, aggregation='one reference song; seed variance only within complete matching provenance groups',
        interpretation='Reference agreement only; these metrics do not establish musical quality or playability',
        overlap=dict(status='unseen_not_verified'))


def build_parser():
    from .defaults import EVALUATION_DEFAULTS as defaults
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    teacher = sub.add_parser('teacher-forced')
    teacher.add_argument('--checkpoint', required=True)
    teacher.add_argument('--manifest', required=True)
    teacher.add_argument('--cache-dir', default='cache/chart_composer_v2')
    teacher.add_argument('--training-manifest', action='append', default=[])
    teacher.add_argument('--limit-charts', type=int, default=defaults['limit_charts'])
    teacher.add_argument('--max-batches', type=int, default=defaults['max_batches'])
    teacher.add_argument('--window-sampling', choices=('spread', 'prefix'), default=defaults['window_sampling'])
    teacher.add_argument('--cpu-threads', type=int, default=defaults['cpu_threads'])
    teacher.add_argument('--timing-rollouts', type=int, default=defaults['timing_rollouts'])
    teacher.add_argument('--output', required=True)
    comparison = sub.add_parser('compare')
    comparison.add_argument('--reference', required=True)
    comparison.add_argument('--generated', action='append', required=True)
    comparison.add_argument('--difficulty', type=int, default=5)
    comparison.add_argument('--duration-sec', type=float)
    comparison.add_argument('--audio',help='Optional audio identity; only an existing feature bank is read')
    comparison.add_argument('--cache-dir',default='cache/chart_composer_v2')
    comparison.add_argument('--audio-backend',choices=('mert','native'),default='mert')
    comparison.add_argument('--output', required=True)
    return parser


def main(argv=None):
    parser = build_parser(); args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    inputs = ([args.checkpoint, args.manifest, *args.training_manifest] if args.command == 'teacher-forced'
              else [args.reference, *args.generated,*([args.audio] if args.audio else [])])
    if output in {Path(path).resolve() for path in inputs}:
        parser.error('output must not overwrite an evaluation input')
    if args.command == 'teacher-forced':
        if min(args.limit_charts, args.max_batches, args.timing_rollouts) < 0 or args.cpu_threads < 1:
            parser.error('budgets must be nonnegative and cpu-threads positive')
        result = teacher_forced(args)
    else:
        result = compare(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    from .timing_metrics import _finite_json
    result = _finite_json(result)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    main()
