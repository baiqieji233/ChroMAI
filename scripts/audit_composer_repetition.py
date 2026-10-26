"""Parse complete Simai charts and report reproducible event repetition counts.

No checkpoint, model inference, GPU or training task is used. Compare charts
with matched music/conditions before interpreting these statistics as changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from maichart.composer.configuration_observation import configuration_observation
from maichart.event_timing import parse_event_chart


def audit_chart(path, difficulty):
    path = Path(path).resolve()
    chart = parse_event_chart(path, difficulty, strict_notes=True)
    record = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'normalized_text_sha256': hashlib.sha256(path.read_text(encoding='utf-8-sig').encode('utf-8')).hexdigest(),
              'difficulty': difficulty, 'title': chart.title,
              'configuration_observation': configuration_observation(chart.points)}
    sidecar_path = path.with_suffix('.composer.json')
    if sidecar_path.exists():
        sidecar = json.loads(sidecar_path.read_text(encoding='utf-8'))
        record['generation_provenance'] = {key: sidecar.get(key) for key in (
            'optimizer_steps', 'weights_sha256', 'chart_sha256', 'event_count', 'seed', 'pointer_temperature',
            'action_temperature', 'pointer_history_strength', 'action_history_strength',
            'action_type_temperature', 'action_type_min_p', 'touch_history_guard', 'touch_history_log_odds_bound')}
        record['sidecar_event_count_matches'] = sidecar.get('event_count') == len(chart.points)
        record['sidecar_chart_hash_matches'] = (sidecar['chart_sha256'] == record['normalized_text_sha256']
                                                if 'chart_sha256' in sidecar else None)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('charts', nargs='+', type=Path)
    parser.add_argument('--difficulty', type=int, default=5)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    report = {'schema': 'composer_repetition_audit_v1',
              'limitation': 'Descriptive counts, not a quality score; different songs or conditions are not a matched comparison.',
              'charts': [audit_chart(path, args.difficulty) for path in args.charts]}
    text = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding='utf-8')
        print(args.output)
    else:
        print(text, end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
