"""First-pass review coverage for a planned WebUI generation batch."""
from collections import Counter
import hashlib
import json
from pathlib import Path


def _feedback(output, digest, slot, audio_digest):
    feedback = output.with_name('generation_feedback.json')
    if not feedback.is_file():
        return 'unreviewed'
    try:
        data = json.loads(feedback.read_text(encoding='utf-8-sig'))
    except (ValueError,OSError):
        return 'unverified'
    if data.get('schema') != 'maichart-generation-feedback-v1':
        return 'unverified'
    for row in reversed(data.get('records', [])):
        if (row.get('scope') == 'whole_chart' and 'training_use' not in row
                and row.get('difficulty_index') == slot and row.get('file_sha256') == digest
                and row.get('audio_sha256') == audio_digest and audio_digest
                and row.get('label') in ('good','problem','uncertain')):
            return 'unreviewed' if row['label']=='good' and row.get('reason') else row['label']
    return 'unreviewed'


def batch_summary(path):
    source = Path(path).resolve()
    data = json.loads(source.read_text(encoding='utf-8-sig'))
    if data.get('schema') != 'composer-v2-batch-v1':
        raise ValueError('请选择批量生成目录中的 batch.composer_v2_batch.json')
    count = int(data['count'])
    if count < 1:
        raise ValueError('批量清单缺少计划生成张数')
    entries = {row['index']:row for row in data['charts']}
    if len(entries) != len(data['charts']) or any(i < 1 or i > count for i in entries):
        raise ValueError('批量清单中有重复或超出范围的版本号')
    progress = {row['index']:row for row in data.get('chart_progress', [])}
    rows, counts = [], Counter()
    for index in range(1, count+1):
        item = entries.get(index, {})
        state = item.get('status', progress.get(index, {}).get('status')) or 'pending'
        result = 'failed' if state in ('failed','cancelled') else 'pending'
        seed = item.get('seed', progress.get(index, {}).get('seed'))
        if state == 'completed':
            # Batch outputs belong to the manifest's directory, including when
            # the bundle was moved from another device.
            name = str(item.get('output','')).replace('\\','/').rsplit('/',1)[-1]
            output = source.parent / f'{index:02d}_seed_{seed}' / name
            result = 'unverified'
            if (output.resolve().is_relative_to(source.parent) and output.is_file()
                    and hashlib.sha256(output.read_bytes()).hexdigest() == item.get('sha256')):
                report = output.with_suffix('.composer_v2.json')
                if report.is_file():
                    try:
                        info = json.loads(report.read_text(encoding='utf-8-sig'))
                    except (ValueError,OSError):
                        info = {}
                    if info.get('seed') == seed and info.get('family') == 'chart_composer_v2':
                        result = _feedback(output,item['sha256'],info.get('difficulty',5),
                                           (info.get('audio_identity') or {}).get('audio_sha256'))
        counts[result] += 1
        rows.append(dict(index=index, seed=seed, result=result,
                         admitted_conflicts=item.get('accepted_with_conflicts_count')))
    decisive = counts['good'] + counts['problem']
    unresolved = count - decisive - counts['failed']
    return dict(batch=str(source), audio=data.get('audio'), checkpoint_sha256=data.get('checkpoint_sha256'),
                level=data.get('level'), planned=count, counts=dict(counts), rows=rows,
                decisive_review_fraction=decisive/count, confirmed_good_fraction=counts['good']/count,
                possible_good_fraction=(counts['good']+unresolved)/count,
                good_among_decisive=counts['good']/decisive if decisive else None)
