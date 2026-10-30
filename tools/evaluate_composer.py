"""Generate a fixed-seed Composer panel and compare it to a same-song chart.

A holdout is called unseen only when both manifests are supplied and proven
song/audio-disjoint. Reference and generated statistics are descriptive.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tools'))
from audit_generated_chart import _events_from_inote
from maichart.composer.configuration_observation import configuration_observation
from maichart.composer.structure_metrics import structure_metrics
from maichart.composer.evaluation import assert_unseen, file_hash,compare_chart_reports
from maichart.generation.event_quality import assess_event_chart
from maichart.maidata import parse_maidata_file
from maichart.notes import parse_timing_point_notes
from maichart.timing import RawTimingPoint
from fractions import Fraction


def chart_report(path, difficulty, *, fallback_bpm=None):
    chart = parse_maidata_file(path)
    blocks = [x for x in chart.difficulties if x.inote and (difficulty is None or x.index == difficulty)]
    if len(blocks) != 1:
        raise ValueError(f'{path}: specify a single existing difficulty')
    # A nominal tempo permits beat-domain configuration counts when BPM
    # metadata is absent; time-based judge results then remain unavailable.
    bpm_available = bool(chart.wholebpm) or fallback_bpm is not None
    points, tempo = _events_from_inote(blocks[0].inote, float(chart.wholebpm or fallback_bpm or 120))
    events = [p for p in points if p.note_text.strip()]
    counts = dict(events=len(events), heads=0, touch_heads=0, touch_events=0,
                  events_3plus_heads=0, break_heads=0, ex_heads=0, break_events=0, ex_events=0, unsupported_events=0)
    for p in events:
        parsed = parse_timing_point_notes(RawTimingPoint(index=0, raw=p.note_text, note_text=p.note_text,
            beat=Fraction(0), tick=0, time_sec=0., division=4, bpm=120.))
        if parsed.unknown_tokens or not parsed.notes:
            counts['unsupported_events'] += 1
            continue
        heads = {str(n.position) for n in parsed.notes}
        touches = [n for n in parsed.notes if n.note_type in ('touch', 'touch_hold')]
        counts['heads'] += len(heads)
        counts['touch_heads'] += len({str(n.position) for n in touches})
        counts['touch_events'] += bool(touches)
        counts['events_3plus_heads'] += len(heads) >= 3
        break_heads = {str(n.position) for n in parsed.notes if n.is_break}
        ex_heads = {str(n.position) for n in parsed.notes if n.modifiers.get('ex', False)}
        counts['break_heads'] += len(break_heads)
        counts['ex_heads'] += len(ex_heads)
        counts['break_events'] += bool(break_heads)
        counts['ex_events'] += bool(ex_heads)
    valid_events = counts['events'] - counts['unsupported_events']
    def ratio(num, den): return {'numerator': num, 'denominator': den, 'rate': num / den if den else None}
    return {'path': str(Path(path).resolve()), 'sha256': file_hash(path), 'difficulty': blocks[0].index,
            'tempo_changes': [{'beat': str(beat), 'bpm': float(bpm)} for beat, bpm in tempo],
            'counts': counts, 'rates': {
                'touch_heads': ratio(counts['touch_heads'], counts['heads']),
                'touch_events': ratio(counts['touch_events'], valid_events),
                'events_3plus_heads': ratio(counts['events_3plus_heads'], valid_events),
                'break_heads': ratio(counts['break_heads'], counts['heads']),
                'ex_heads': ratio(counts['ex_heads'], counts['heads']),
                'break_events': ratio(counts['break_events'], valid_events),
                'ex_events': ratio(counts['ex_events'], valid_events)},
            'configuration': configuration_observation(events),
            'structure': structure_metrics(events),
            'judge': assess_event_chart(points, tempo, first=float(chart.first or 0)) if bpm_available else None,
            'judge_unavailable_reason': None if bpm_available else 'wholebpm metadata missing',
            'scope': 'unique positions per event; Break and EX may overlap; unsupported events excluded from rates'}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True); p.add_argument('--audio', required=True)
    p.add_argument('--reference', required=True); p.add_argument('--difficulty', type=int, required=True)
    p.add_argument('--output-dir', required=True); p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--device', default='auto'); p.add_argument('--bpm', type=float, required=True)
    p.add_argument('--first', type=float, default=0); p.add_argument('--level', type=float, default=13.)
    p.add_argument('--training-manifest'); p.add_argument('--holdout-manifest')
    p.add_argument('--teacher-forced', action='store_true', help='also evaluate representable held-out targets')
    p.add_argument('--cache-dir', default='cache/chart_composer_v1')
    p.add_argument('--eval-batch-size', type=int, default=16); p.add_argument('--eval-batches', type=int, default=0)
    a, extra = p.parse_known_args(argv)
    if bool(a.training_manifest) != bool(a.holdout_manifest): p.error('provide both manifests for an unseen claim')
    if a.teacher_forced and not a.holdout_manifest: p.error('--teacher-forced requires --holdout-manifest and --training-manifest')
    unseen = False
    if a.training_manifest:
        assert_unseen(a.training_manifest, a.holdout_manifest)
        from maichart.training.dataset_catalog import _source_asset
        manifest = Path(a.holdout_manifest).resolve()
        data = json.loads(manifest.read_text(encoding='utf-8-sig'))
        audio_hash = file_hash(a.audio)
        matched = [s for s in data['songs'] if file_hash(_source_asset(s['audio_path'], manifest, data)) == audio_hash]
        if not matched:
            raise ValueError('evaluation audio does not belong to holdout manifest')
        reference_path = Path(a.reference).resolve()
        if not any(s.get('maidata_path') and _source_asset(s['maidata_path'], manifest, data) == reference_path for s in matched):
            raise ValueError('reference chart does not match holdout audio entry')
        import torch
        payload = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
        recorded = payload.get('manifest')
        if not recorded or not Path(recorded).is_file() or file_hash(recorded) != file_hash(a.training_manifest):
            raise ValueError('training manifest is not verifiably the one recorded by the checkpoint')
        unseen = True
    teacher_forced = None
    if a.teacher_forced:
        import torch
        from maichart.composer.checkpoint import load
        from maichart.composer.data import ComposerDataset
        from maichart.composer.evaluation import evaluate_teacher_forced
        device = torch.device('cuda' if a.device == 'auto' and torch.cuda.is_available() else
                              'cpu' if a.device == 'auto' else a.device)
        model, payload = load(a.checkpoint, device=device); config = model.config
        dataset = ComposerDataset(a.holdout_manifest, cache_dir=a.cache_dir,
            candidate_count=config.candidate_count, history_events=config.history_events,
            action_slots=config.action_slots, architecture_profile=config.architecture_profile,
            action_state_policy=config.action_state_policy, head_timing_policy=config.head_timing_policy,
            motif_memory_policy=config.motif_memory_policy, motif_max_events=config.motif_max_events,
            structure_policy=config.structure_policy, structure_backend=config.structure_backend,
            preparation_workers=1, audio_cache_gb=.25, atom_catalog=payload['atom_catalog'])
        saved_args = payload.get('args') or {}
        loss_kwargs = {key: saved_args[key] for key in ('audio_pointer_weight', 'placement_weight',
                       'planner_weight', 'action_objective_policy') if key in saved_args}
        teacher_forced = evaluate_teacher_forced(model, dataset, device=device, batch_size=a.eval_batch_size,
                                                  max_batches=a.eval_batches, loss_kwargs=loss_kwargs)
        teacher_forced['catalog_coverage'] = dataset.catalog_coverage
    out = Path(a.output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    reference = chart_report(a.reference, a.difficulty, fallback_bpm=a.bpm)
    user_tempo = next((i for i, value in enumerate(extra) if value == '--tempo-map' or value.startswith('--tempo-map=')), None)
    if user_tempo is None:
        tempo_path = out / 'reference_tempo_map.json'
        tempo_path.write_text(json.dumps(reference['tempo_changes'], ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        extra = [*extra, '--tempo-map', str(tempo_path)]
        tempo_source = 'reference'
    else:
        token = extra[user_tempo]
        if token == '--tempo-map':
            if user_tempo + 1 >= len(extra): p.error('--tempo-map requires a path')
            tempo_path = Path(extra[user_tempo + 1]).resolve()
        else:
            tempo_path = Path(token.split('=', 1)[1]).resolve()
        tempo_source = 'explicit'
    runs = []
    for seed in a.seeds:
        target = out / f'seed_{seed}.txt'
        cmd = [sys.executable, '-m', 'maichart.composer.generate', '--checkpoint', str(Path(a.checkpoint).resolve()),
               '--audio', str(Path(a.audio).resolve()), '--output', str(target), '--bpm', str(a.bpm),
               '--first', str(a.first), '--level', str(a.level), '--device', a.device, '--seed', str(seed), *extra]
        import os
        env = os.environ.copy(); env['PYTHONPATH'] = str(ROOT / 'src') + os.pathsep + env.get('PYTHONPATH', '')
        completed = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, encoding='utf-8', errors='replace')
        if completed.returncode: raise RuntimeError(f'generation failed for seed {seed}: {completed.stderr}\n{completed.stdout}')
        generated_report=chart_report(target,None,fallback_bpm=a.bpm)
        runs.append({'seed': seed, 'command': cmd, 'chart': generated_report,
                     'reference_comparison':compare_chart_reports(reference,generated_report),
                     'generation': json.loads(target.with_suffix('.composer.json').read_text(encoding='utf-8'))})
    report = {'schema': 'composer-generation-evaluation-v1', 'checkpoint': str(Path(a.checkpoint).resolve()),
              'checkpoint_sha256': file_hash(a.checkpoint), 'audio': str(Path(a.audio).resolve()),
              'audio_sha256': file_hash(a.audio), 'unseen_manifest_disjoint_verified': unseen,
              'training_manifest': a.training_manifest, 'holdout_manifest': a.holdout_manifest,
              'reference': reference, 'runs': runs, 'teacher_forced': teacher_forced,
              'generation_tempo_map': {'source': tempo_source, 'path': str(tempo_path),
                                       'sha256': file_hash(tempo_path)},
              'scope': 'manifest check covers recorded training manifest only, not parent checkpoint lineage; no downbeat or human playability claim'}
    (out / 'evaluation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')
    print(str(out / 'evaluation.json')); return 0

if __name__ == '__main__': raise SystemExit(main())
