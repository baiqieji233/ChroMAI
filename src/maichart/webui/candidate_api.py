"""Candidate review operations; scanning runs in a separate CPU-only task."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import threading
import uuid

_START_LOCK = threading.RLock()


def _write(path, value):
    from maichart.training.dataset_catalog import _replace_file

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
        _replace_file(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _is_pool(manifest):
    from .dataset_api import _json_data

    return bool(_json_data(Path(manifest)).get('candidate_pool'))


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def remember_undo(manifest, chart_id, before, after):
    """Persist a chart-scoped undo; unrelated chart reviews are never replaced."""
    path = Path(manifest).parent / 'review_undo.json'
    values = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    token = uuid.uuid4().hex
    policy = json.loads((Path(manifest).parent / 'pool.json').read_text(encoding='utf-8')).get('scan_policy')
    values[token] = dict(chart_id=chart_id, before=before, after=_digest(after), scan_policy=policy)
    values = dict(list(values.items())[-500:])
    _write(path, values)
    return token


def _undo(manifest, chart_id, token):
    from .quality_api import _AnnotationLookup, _catalog_row, _material_write
    from maichart.training.candidate_pool import review_guard
    from maichart.training.quality_annotations import load_annotations, save_annotations, zero_unmarked_region_weights
    manifest = Path(manifest).resolve()
    with _material_write(manifest):
        path = Path(manifest).parent / 'review_undo.json'
        values = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        item = values.get(token)
        if not item or item['chart_id'] != chart_id:
            raise ValueError('这条操作已经撤销，或不属于当前谱面')
        policy = json.loads((Path(manifest).parent / 'pool.json').read_text(encoding='utf-8')).get('scan_policy')
        if item.get('scan_policy') != policy:
            raise ValueError('检查方式已变化，请重新审阅；旧撤销不会恢复旧的学习范围')
        positive = any(any(zero_unmarked_region_weights(r)) for r in (item.get('before') or {}).get('regions', []))
        row = _catalog_row(manifest, chart_id)
        review_guard(manifest, chart_id, positive=positive, row=row)
        raw_records = load_annotations(manifest)['records']
        lookup = _AnnotationLookup(raw_records)
        current = lookup.find(row) if row is not None else next((r for r in raw_records if r.get('chart_id') == chart_id), None)
        if _digest(current) != item['after']:
            raise ValueError('这张谱面已有更新评价，请先重新载入；旧撤销不会覆盖新评价')
        if row is not None:
            if item['before'] is not None:
                lookup.replace_for_row(row, chart_id, item['before'])
            else:
                lookup.remove_for_row(row, chart_id)
            records = lookup.active_records()
        else:
            records = [r for r in raw_records if r.get('chart_id') != chart_id]
            if item['before'] is not None:
                records.append(item['before'])
        save_annotations(manifest, records)
        values.pop(token)
        _write(path, values)
    return dict(undone=True, chart_id=chart_id, records=records)


def _registered(runtime):
    api = runtime._dataset_api()
    manifests = set()
    for entry in api._roots():
        root = api._resolve(entry['path'])
        paths = [api._resolve(entry['manifest'])] if entry.get('manifest') else [root / name for name in ('training_manifest.json', 'pretrain_manifest.json')]
        manifests.update(path.resolve() for path in paths if path.is_file())
    return manifests


def _pool(payload, runtime, *, pending=False):
    base = (runtime.WEBUI_ROOT / 'candidate_pools').resolve()
    value = payload.get('pool_id') or payload.get('pool')
    if value:
        if not isinstance(value, str) or not re.fullmatch(r'[a-f0-9]{12}', value):
            raise ValueError('候选库标识无效')
        path = base / value
    else:
        manifest = Path(str(payload.get('manifest') or '')).resolve()
        path = manifest.parent
        if manifest.name != 'training_manifest.json':
            raise ValueError('请先选择候选库')
    if path.parent != base or not re.fullmatch(r'[a-f0-9]{12}', path.name) or not (
            (path / 'pool.json').is_file() or pending and (path / 'scan_state.json').is_file()):
        raise ValueError('候选库不存在，请先创建并等待检查完成')
    return path


def _start(runtime, pool_dir, args):
    from .quality_api import _material_write
    with _START_LOCK, _material_write(pool_dir / 'training_manifest.json'):
        for task in runtime._all_tasks():
            if task.kind == 'candidate_scan' and task.status in ('queued', 'running', 'pending') and task.result.get('pool_dir') == str(pool_dir):
                return dict(task_id=task.task_id, pool_dir=str(pool_dir), pool_id=pool_dir.name,
                            manifest=str(pool_dir / 'training_manifest.json'))
        from .material_audit_status import task_state

        state_path = pool_dir / 'scan_state.json'
        if state_path.is_file() and task_state(json.loads(state_path.read_text(encoding="utf-8"))).get("status") in {"queued", "running"}:
            raise ValueError("候选库已有扫描在运行")
        task = runtime._create_task('candidate_scan')
        task.result.update(pool_dir=str(pool_dir), pool_id=pool_dir.name)
        _write(state_path, dict(status='queued', task_id=task.task_id, pid=os.getpid()))

        def finished():
            state_path = pool_dir / 'scan_state.json'
            state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.is_file() else {}
            if state.get("task_id") == task.task_id:
                state.update(status=task.status, error=task.error)
                _write(state_path, state)

        try:
            runtime._run_subprocess_task(task, args, expected_epoch=None,
                result_paths=[pool_dir / 'pool.json', pool_dir / 'training_manifest.json', pool_dir / 'scan/results.json'],
                on_finish=finished)
        except Exception as exc:
            task.status, task.error = "failed", str(exc)
            finished()
            raise
    return dict(task_id=task.task_id, pool_dir=str(pool_dir), pool_id=pool_dir.name,
                manifest=str(pool_dir / 'training_manifest.json'))


def post(endpoint, payload, *, runtime):
    from maichart.training import candidate_pool as pools
    operation = endpoint.rsplit('/', 1)[-1]
    if operation == 'create':
        manifest = Path(str(payload.get('manifest') or '')).resolve()
        registered = _registered(runtime)
        if manifest not in registered:
            raise ValueError('请先在训练素材中登记并预处理这个数据集')
        source = json.loads(manifest.read_text(encoding='utf-8-sig'))
        if source.get('candidate_pool') or source.get('reviewed_training'):
            raise ValueError('这个来源已经是候选库或人工确认示范库')
        identity = pools.pool_identity(manifest, runtime.WEBUI_ROOT)
        pool_dir = Path(identity['pool_dir'])
        references = []
        for path in sorted(registered - {manifest}):
            data = json.loads(path.read_text(encoding='utf-8-sig'))
            if not data.get('candidate_pool') and not data.get('reviewed_training'):
                references.append(path)
        args = ['-m', 'maichart.training.candidate_pool', 'build', str(manifest),
                '--workspace-root', str(runtime.WEBUI_ROOT), '--threads', '1']
        name = str(payload.get('name') or '').strip()
        if name:
            args += ['--name', name[:120]]
        for path in references:
            args += ['--reference-manifest', str(path)]
        result = _start(runtime, pool_dir, args)
        from .dataset_api import _REGISTRY_LOCK
        api = runtime._dataset_api()
        with _REGISTRY_LOCK:
            entries = [entry for entry in api._registry() if api._resolve(entry['path']) != pool_dir]
            entries.append(dict(path=str(pool_dir), manifest=result['manifest'], name=name or '候选库',
                                candidate_pool=dict(pool_id=result['pool_id'], scan_state='queued')))
            api._save_registry(entries)
        return 200, result
    if operation == 'budget':
        values = payload.get('manifests')
        if not isinstance(values, list) or not values:
            raise ValueError('请先选择训练来源')
        if any(not isinstance(value,str) or not Path(value).is_file() for value in values):
            raise ValueError('请选择有效的训练清单')
        values=list(dict.fromkeys(str(Path(value).resolve()) for value in values))
        roles=payload.get('mechanics_manifests', [])
        if not isinstance(roles,list) or any(not isinstance(v,str) or not v.strip() for v in roles):
            raise ValueError('仅练动作的来源必须是清单路径列表')
        selected={Path(v) for v in values}
        mechanics={Path(v).resolve() for v in roles}
        if mechanics-selected:
            raise ValueError('估算的动作来源必须已选中')
        request_data=dict(manifests=sorted(values),mechanics_manifests=sorted(str(p) for p in selected & mechanics))
        key=_digest(request_data)
        with _START_LOCK:
            for task in runtime._all_tasks():
                if task.kind=='candidate_budget' and task.status in ('queued','running') and task.result.get('request_key')==key:
                    return 200,dict(task_id=task.task_id,status=task.status)
            root=runtime.WEBUI_ROOT/'supervision_budgets'/uuid.uuid4().hex
            request=root/'request.json';output=root/'result.json'
            _write(request,request_data)
            task=runtime._create_task('candidate_budget')
            task.result.update(request_key=key,budget_output=str(output))
            runtime._run_subprocess_task(task,['-m','maichart.training.candidate_budget','--request',str(request),
                '--output',str(output),'--cache-dir',str(runtime.WEBUI_ROOT/'supervision_budgets'/'cache')],
                expected_epoch=None,result_paths=[output])
        return 200,dict(task_id=task.task_id,status='queued')
    path = _pool(payload, runtime, pending=operation == 'status')
    if operation == 'status':
        result = pools.pool_status(path, include_supervision=not payload.get('summary_only')) if (path / 'pool.json').is_file() else dict(
            pool_id=path.name, pool_dir=str(path), manifest=str(path / 'training_manifest.json'), counts={})
        state = path / 'scan_state.json'
        if state.is_file():
            from .material_audit_status import task_state

            data = task_state(json.loads(state.read_text(encoding='utf-8')))
            result['scan_state'] = data.get('status')
            result['scan_task'] = data
            result['updated_at'] = data.get("updated_at") or result.get("updated_at")
        return 200, result
    if operation == 'queue':
        offset, limit = int(payload.get('offset', 0)), int(payload.get('limit', 1))
        status=payload.get('candidate_status') or None
        if status not in (None,'error','suspicious','ok','pending','stale'):raise ValueError('未知自动检查状态')
        if offset < 0 or not 1 <= limit <= 100:
            raise ValueError('审阅位置或数量无效')
        seen = payload.get('seen_segments', [])
        if not isinstance(seen, list):
            raise ValueError('已浏览片段应为列表')
        seen_segments = []
        for item in seen:
            if (not isinstance(item, list) or len(item) != 3 or not isinstance(item[0], str)
                    or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in item[1:])):
                raise ValueError('已浏览片段无效')
            seen_segments.append(tuple(item))
        return 200, pools.review_queue(path, offset=offset, limit=limit, status=status,
                                       exclude_chart_id=payload.get('exclude_chart_id'),
                                       prefer_chart_id=payload.get('prefer_chart_id'), seen_segments=seen_segments)
    if operation == 'excluded':
        data = json.loads((path / 'pool.json').read_text(encoding='utf-8'))
        return 200, dict(items=data.get('excluded', []))
    if operation == 'sync':
        meta = json.loads((path / 'pool.json').read_text(encoding='utf-8')) if (path / 'pool.json').is_file() else {}
        source_manifest = Path(meta['source_manifest']).resolve() if meta.get('source_manifest') else None
        sync_args = ['-m', 'maichart.training.candidate_pool', 'scan', str(path), '--threads', '1']
        for ref_path in sorted(_registered(runtime)):
            if ref_path == source_manifest or ref_path == (path / 'training_manifest.json').resolve():
                continue
            try:
                ref_data = json.loads(ref_path.read_text(encoding='utf-8-sig'))
            except (OSError, ValueError):
                continue
            if not ref_data.get('candidate_pool') and not ref_data.get('reviewed_training'):
                sync_args += ['--reference-manifest', str(ref_path)]
        return 200, _start(runtime, path, sync_args)
    if operation in ('auto_screen', 'auto-screen'):
        from contextlib import nullcontext
        from .quality_api import _material_write
        slots = payload.get('difficulty_slots')
        if slots is None:
            slots = [4, 5, 6] if payload.get('include_expert') is True else [5, 6]
        if not isinstance(slots, list) or not slots:
            raise ValueError('请至少选择一个难度槽位')
        dry_run = bool(payload.get('dry_run', False))
        preserve_manual = payload.get('preserve_manual', True)
        if not isinstance(preserve_manual, bool):
            raise ValueError('preserve_manual 必须是布尔值')
        manifest_file = (path / 'training_manifest.json').resolve()
        with (_material_write(manifest_file) if not dry_run else nullcontext()):
            result = pools.auto_screen_pool(
                path,
                difficulty_slots=[int(s) for s in slots if not isinstance(s, bool)],
                min_level=float(payload.get('min_level', 12.0)),
                max_level=float(payload.get('max_level', 15.0)),
                tap_on_slide_mode=str(payload.get('tap_on_slide_mode', 'mask_hard')),
                tap_on_slide_ms=float(payload.get('tap_on_slide_ms', 200.0)),
                inner_muri_mode=str(payload.get('inner_muri_mode', 'allow')),
                max_offset_ms=float(payload.get('max_offset_ms', 35.0)),
                max_problem_fraction=float(payload.get('max_problem_fraction', 0.35)),
                preserve_manual=preserve_manual,
                dry_run=dry_run,
            )
        return 200, result
    if operation == 'undo':
        return 200, _undo(path / 'training_manifest.json', str(payload.get('chart_id') or ''), str(payload.get('undo_token') or ''))
    return 404, dict(error='未知候选库操作')
