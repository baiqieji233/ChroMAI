"""Read-only checkpoint diagnostics; training charts are not quality validation."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from fractions import Fraction
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from maichart.form.checkpoint import load
from maichart.form.data import FormDataset, collate_form
from maichart.form.generate import arrange_span, to_device


def scores(logits, batch, codec):
    target = batch['target']
    loss = F.cross_entropy(logits.float().transpose(1, 2), target,
                           ignore_index=codec.pad_id, reduction='none')
    correct = logits.argmax(-1) == target
    note_ids = [index for index, token in enumerate(codec.tokens) if token.startswith('N:')]
    note = (target[..., None] == target.new_tensor(note_ids)).any(-1) | (target == codec.end_id)
    groups = {
        'all': target != codec.pad_id,
        'configuration': note,
        'position_structure': (target != codec.pad_id) & ~note,
        'first_head': batch['decoder_input'] == codec.note_id,
    }
    return {name: {'tokens': int(mask.sum()),
                   'ce': float(loss[mask].mean()) if mask.any() else None,
                   'accuracy': float(correct[mask].float().mean()) if mask.any() else None}
            for name, mask in groups.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--charts', type=int, default=3)
    parser.add_argument('--samples', type=int, default=8)
    parser.add_argument('--rollout-onsets', type=int, default=16)
    parser.add_argument('--output')
    args = parser.parse_args()
    torch.set_num_threads(4)
    model, _, codec, payload = load(args.checkpoint, device=args.device)
    dataset = FormDataset(args.manifest, cache_dir=args.cache_dir,
        max_tokens=model.config.max_tokens, max_memory_tokens=model.config.max_memory_tokens,
        max_region_seconds=payload.get('args', {}).get('max_region_seconds', 14.),
        limit_charts=args.charts, quality_enabled=True, sample_cache_mb=0)
    groups = {}
    for index, item in enumerate(dataset.items):
        if not 24 <= len(item[2])-1 <= 800:
            continue
        groups.setdefault(item[0], {}).setdefault(item[1], []).append(index)
    per_song = max(1, (args.samples+2*len(groups)-1)//max(1, 2*len(groups)))
    selected = []
    for rank in range(per_song):
        for spans in groups.values():
            keys = sorted(spans)
            selected.extend(spans[keys[min(len(keys)-1, rank*len(keys)//per_song)]])
    rows = []
    for index in selected:
        if len(rows) >= args.samples:
            break
        example = dataset[index]
        if len(example['target']) < 24 or len(example['target']) > 800:
            continue
        batch = to_device(collate_form([example]), args.device)
        modes = {}
        with torch.inference_mode():
            output = model(batch)['logits']
            modes['authored_inputs'] = scores(output, batch, codec)
            hidden_curve = dict(batch)
            hidden_curve['load_curve_condition_mask'] = torch.zeros_like(
                hidden_curve['load_curve'], dtype=torch.bool)
            modes['no_curve'] = scores(model(hidden_curve)['logits'], batch, codec)
            if len(rows) == 0:
                memory, mask = model.encode(batch)
                prefix = batch['decoder_input'][:, :96]
                full = model.decode(prefix, memory, mask)
                cache = None
                steps = []
                for position in range(prefix.size(1)):
                    logits, cache = model.decode_step(prefix[:, position:position+1],
                                                      position, memory, mask, cache)
                    steps.append(logits)
                incremental = torch.cat(steps, 1)
                modes['cache_parity'] = {'maximum_absolute_error': float((full-incremental).abs().max()),
                    'argmax_agreement': float((full.argmax(-1)==incremental.argmax(-1)).float().mean())}
        row = {'dataset_item': index, 'song_index': dataset.items[index][0],
               'span_index': dataset.items[index][1], 'task': int(example['task']),
               'curve_rows': len(example['load_curve']), 'metrics': modes}
        if not rows and args.rollout_onsets:
            origin = Fraction(example['meta']['span']['start'])
            target_points = codec.decode([codec.bos_id]+example['target'].tolist(), origin=origin)
            target_points = target_points[:args.rollout_onsets]
            rollout_example = copy.deepcopy(example)
            rollout_example['load_curve_condition_mask'].zero_()
            generated, fallbacks = arrange_span(model, codec, rollout_example,
                [point.beat for point in target_points], payload['atom_catalog'],
                origin=origin, device=args.device, temperature=0., seed=0)
            row['rollout'] = {'authored': [p.note_text for p in target_points],
                             'generated': [p.note_text for p in generated],
                             'fallbacks': fallbacks,
                             'exact_cells': sum(a.note_text == b.note_text for a, b in zip(target_points, generated)),
                             'generated_counts': dict(Counter(p.note_text for p in generated))}
        rows.append(row)
        print('form_quality_probe '+json.dumps(row, ensure_ascii=False), flush=True)
    aggregate = {}
    for task in (0, 1):
        aggregate[str(task)] = {}
        for mode in ('authored_inputs', 'no_curve'):
            aggregate[str(task)][mode] = {}
            for category in ('all', 'configuration', 'position_structure', 'first_head'):
                items = [row['metrics'][mode][category] for row in rows if row['task'] == task]
                count = sum(item['tokens'] for item in items)
                aggregate[str(task)][mode][category] = {'tokens': count,
                    **{metric: sum((item[metric] or 0.)*item['tokens'] for item in items)/max(1, count)
                       for metric in ('ce', 'accuracy')}}
    report = {'checkpoint': str(Path(args.checkpoint).resolve()),
              'completed_epoch': payload.get('epoch'), 'training_semantics': payload.get('training_semantics'),
              'source': 'training_chart_diagnostics_not_validation_or_playability_acceptance',
              'aggregate': aggregate, 'examples': rows}
    print('form_quality_probe_aggregate '+json.dumps(aggregate), flush=True)
    if args.output:
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
