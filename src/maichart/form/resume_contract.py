"""Explicit resume checks and source-preserving preparation migrations."""
import hashlib
import os
from pathlib import Path


HEADER_KEYS = ('manifest_sha256', 'meter_map', 'max_region_seconds')
SOURCE_KEYS = ('chart_sha256', 'audio_sha256', 'meter_map', 'quality', 'weight', 'first_override',
               'training_inputs')


def contract_header(manifest, meter_map, max_region_seconds):
    return {'manifest_sha256': hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
            'meter_map': meter_map, 'max_region_seconds': max_region_seconds}


def preflight_resume(saved, current):
    changed = [key for key in HEADER_KEYS if key in saved and saved[key] != current.get(key)]
    if changed:
        raise ValueError('续训数据合同已变化：' + '、'.join(changed) +
                         '。请用“已有模型加强训练（finetune）”；这项检查已在准备数据前完成。')


def verify_saved_chart_sources(saved):
    """Catch edited or missing previously trained files before costly parsing."""
    seen = {}
    for chart in saved.get('charts') or []:
        expected = chart.get('chart_sha256')
        if not expected or not chart.get('source'):
            raise ValueError('已训练谱面缺失源文件身份 hash，无法验证续训；请用 finetune。')
        source = Path(chart['source'])
        if source not in seen:
            try: seen[source] = hashlib.sha256(source.read_bytes()).hexdigest()
            except OSError: seen[source] = None
        if seen[source] != expected:
            raise ValueError(f'已训练谱面源文件变化或缺失：{source}。请用 finetune；已在准备数据前检查。')


def _key(chart):
    return os.path.normcase(os.path.normpath(str(chart.get('source', '')))), chart.get('difficulty')


def compare_resume_contract(saved, current):
    before, after = saved.get('charts') or [], current.get('charts') or []
    old = {_key(chart): chart for chart in before}
    new = {_key(chart): chart for chart in after}
    common = old.keys() & new.keys()
    added = [chart for chart in after if _key(chart) not in old]
    removed = [chart for chart in before if _key(chart) not in new]
    changed = []
    source_changes = []
    for key in sorted(common, key=str):
        keys = [name for name in old[key].keys() | new[key].keys()
                if old[key].get(name) != new[key].get(name)]
        if not keys:
            continue
        item = {'source': new[key].get('source'), 'difficulty': key[1], 'fields': sorted(keys)}
        changed.append(item)
        if any(name in SOURCE_KEYS for name in keys): source_changes.append(item)
    headers = [key for key in HEADER_KEYS if saved.get(key) != current.get(key)]
    duplicates = len(old) != len(before) or len(new) != len(after)
    unverifiable = [
        {'side': side, 'source': chart.get('source'), 'difficulty': chart.get('difficulty'),
         'fields': [name for name in ('source', 'chart_sha256', 'audio_sha256') if not chart.get(name)]
                   + (['difficulty'] if chart.get('difficulty') is None else [])}
        for side, charts in (('saved', before), ('current', after)) for chart in charts
        if not chart.get('source') or chart.get('difficulty') is None
        or not chart.get('chart_sha256') or not chart.get('audio_sha256')
    ]
    safe = (not headers and not source_changes and not removed and not duplicates
            and not unverifiable and bool(before) and bool(after))
    return {
        'schema': 'song_form_resume_contract_diff_v1', 'compatible': saved == current,
        'old_chart_count': len(before), 'new_chart_count': len(after),
        'header_changes': headers, 'added': added, 'removed': removed,
        'changed': changed, 'source_changes': source_changes,
        'duplicate_chart_identities': duplicates,
        'unverifiable_identities': unverifiable,
        'can_restart_epoch': safe,
        'order_changed': [_key(chart) for chart in before] != [_key(chart) for chart in after],
    }


def resolve_resume_contract(saved, current, policy='strict'):
    if policy not in {'strict', 'restart_epoch'}:
        raise ValueError('resume_data_policy must be strict or restart_epoch')
    diff = compare_resume_contract(saved or {}, current)
    if diff['compatible']:
        return diff
    summary = (f"续训数据合同不兼容：可用谱面 {diff['old_chart_count']} → {diff['new_chart_count']}，"
               f"新增 {len(diff['added'])}、移除 {len(diff['removed'])}，"
               f"已有谱面记录变化 {len(diff['changed'])} 项")
    if policy == 'restart_epoch' and diff['can_restart_epoch']:
        return diff
    if diff['can_restart_epoch']:
        action = '请选择“解析结果变化时，保留状态并重跑当前轮”，或使用 finetune（--init-checkpoint）。'
    else:
        fields = diff['header_changes'] or sorted({key for item in diff['source_changes'] for key in item['fields']})
        action = '源数据、标注或布局设置变化，不能迁移旧游标；请用 finetune（--init-checkpoint）。'
        if fields: summary += '；变化字段：' + '、'.join(fields)
    raise ValueError(summary + '。' + action + ' 详细差异见 resume_contract_diff.json。')


def validate_resume_cursor(cursor, *, epoch, layout_total, plan_total, batch_size, max_steps,
                           blueprint_batch_size=None):
    """An exact contract still needs a valid completed/unfinished batch position."""
    if not cursor:
        return
    phase = cursor.get('phase')
    expected_epoch = epoch if phase == 'complete' else epoch + 1
    limits = {'epoch': expected_epoch, 'layout_total': layout_total, 'plan_total': plan_total,
              'batch_size': batch_size, 'max_steps': max_steps}
    # Old checkpoints did not serialize this counter. Their trainer still
    # inherits the saved argument (or its historical default of one).
    if 'blueprint_batch_size' in cursor:
        limits['blueprint_batch_size'] = blueprint_batch_size
    if phase not in {'blueprint', 'layout_revision', 'complete'}:
        raise ValueError('续训游标阶段无效；请用 finetune。')
    for name, expected in limits.items():
        # Older completed checkpoints may lack counters; unfinished cursors may not.
        if name not in cursor and phase == 'complete':
            continue
        if type(cursor.get(name)) is not int or cursor[name] != expected:
            raise ValueError(f'续训游标 {name} 不匹配；请用 finetune。')
    completed = cursor.get('completed')
    total = plan_total if phase == 'blueprint' else layout_total
    if type(completed) is not int or not 0 <= completed <= total:
        raise ValueError('续训游标 completed 超出阶段范围；请用 finetune。')
    if phase == 'complete' and completed != layout_total:
        raise ValueError('已完成轮次的续训游标不完整；请用 finetune。')
