"""Prepare and run an ordered set of independent Composer v2 song jobs."""
from __future__ import annotations

import json
from pathlib import Path
import re
import secrets
import threading
from datetime import datetime, timezone

from maichart.training.dataset_catalog import _write_json
from .composer_v2_api import FAMILY, SCHEMA, prepare_generation

KIND = 'composer_v2_generation_queue'
TERMINAL = {'completed', 'failed', 'partial', 'cancelled'}


def _directory(runtime, queue_id):
    return runtime.WEBUI_ROOT / 'generated' / f'queue_{queue_id}'


def _checkpoint_version(path):
    stat = Path(path).stat()
    return dict(bytes=stat.st_size, modified_ns=stat.st_mtime_ns)


def _prepare(payload, runtime, *, root=None, inspections=None):
    rows = payload.get('songs')
    if not isinstance(rows, list) or not rows:
        raise ValueError('请选择至少一首歌')
    prepared, jobs = [], []
    if inspections is None:
        inspections = {}
    seen = set()
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict) or not isinstance(row.get('settings'), dict):
            raise ValueError(f'第 {index} 首歌缺少生成参数')
        key = row.get('id')
        if not isinstance(key, str) or not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}', key) or key in seen:
            raise ValueError(f'第 {index} 首歌的队列编号无效')
        seen.add(key)
        config = dict(row['settings'])
        audio = config.get('audio')
        title = row.get('name')
        if not isinstance(title, str) or not title.strip():
            title = Path(audio).stem if isinstance(audio, str) else f'第 {index} 首'
        info = dict(id=key, name=title.strip()[:160], status='queued')
        sources = row.get('sources')
        if isinstance(sources, dict):
            info['sources'] = {field: owner[:160] for field, owner in sources.items()
                               if isinstance(owner, str)}
        # The single-song adapter owns the full checkpoint and option contract.
        config.pop('output', None)
        if root is not None:
            config['output'] = str(root / f'{index:02d}' / 'maidata.txt')
        if config.get('seed') is None:
            config['seed'] = secrets.randbelow(2**31-32) if root is not None else 0
            info['automatic_seed'] = True
        try:
            args, settings, paths = prepare_generation(config, runtime=runtime,
                inspections=inspections, preview=root is None)
            signature = _checkpoint_version(settings['checkpoint'])
        except (ValueError, OSError, TypeError, RuntimeError, AttributeError, KeyError, ZeroDivisionError) as exc:
            info.update(status='invalid', error=str(exc))
        else:
            if root is None:
                settings.pop('output', None)
            info.update(settings=settings, checkpoint_version=signature)
            prepared.append((args, settings, paths))
        jobs.append(info)
    return jobs, prepared


def preview(handler, payload, *, runtime):
    rows, _ = _prepare(payload, runtime)
    return handler._json_response(dict(valid=all(row['status'] == 'queued' for row in rows), jobs=rows))


def _update(task, index, **fields):
    # Replace the list and row so readers holding a prior snapshot see stable data.
    with task.lock:
        rows = task.result['jobs'].copy()
        rows[index] = {**rows[index], **fields}
        task.result['jobs'] = rows


def start(handler, payload, *, runtime):
    inspections = {}
    rows, _ = _prepare(payload, runtime, inspections=inspections)
    if any(row['status'] == 'invalid' for row in rows):
        return handler._json_response(dict(valid=False, jobs=rows))

    task = runtime._create_task(KIND)
    root = _directory(runtime, task.task_id)
    manifest = root / 'queue.json'
    original = rows
    rows, commands = _prepare(payload, runtime, root=root, inspections=inspections)
    for before, after in zip(original, rows):
        if after.get('checkpoint_version') != before.get('checkpoint_version'):
            after.update(status='invalid', error='模型在提交期间已变化，请重新核对')
    if any(row['status'] == 'invalid' for row in rows):
        with task.lock:
            task.status = 'failed'
            task.error = '提交期间输入已变化'
        return handler._json_response(dict(valid=False, jobs=rows))
    with task.lock:
        task.result.update(jobs=rows, model_family=FAMILY, manifest=str(manifest),
                           created_at=datetime.now(timezone.utc).isoformat())

    def persist():
        record = task.to_dict()
        record.pop('logs', None)
        _write_json(manifest, record)

    def run():
        finished = 0
        try:
            with task.lock:
                if not task.cancel_requested:
                    task.status = 'running'
                    task.message = '按队列顺序生成'
            for index, (args, settings, paths) in enumerate(commands):
                try:
                    checked = _checkpoint_version(settings['checkpoint']) == rows[index]['checkpoint_version']
                except OSError:
                    checked = False
                with task.lock:
                    if task.cancel_requested:
                        break
                    if checked:
                        child = runtime._create_task('composer_v2_generate')
                        child.result.update(model_family=FAMILY, family=FAMILY, schema=SCHEMA, effective_settings=settings)
                        task.result['active_task_id'] = child.task_id
                if not checked:
                    _update(task, index, status='failed', error='模型文件在排队期间已变化或移动，请用固定检查点重跑此曲')
                    finished += 1
                    with task.lock:
                        task.progress = finished / len(rows)
                    persist()
                    continue
                ready = threading.Event()
                try:
                    _update(task, index, status='running', task_id=child.task_id)
                    runtime._run_subprocess_task(child, args, expected_epoch=None,
                                                 result_paths=paths, on_finish=ready.set)
                    try:
                        persist()
                    except OSError as exc:
                        with task.lock:
                            task.result['persistence_error'] = str(exc)
                    ready.wait()
                except Exception as exc:
                    with child.lock:
                        child.status = 'failed'
                        child.error = str(exc)
                with child.lock:
                    status, error = child.status, child.error
                    batch = child.result.get('composer_v2_batch') or {}
                    missing = child.result.get('missing_artifacts')
                if status == 'completed' and (missing or batch.get('failed')):
                    status = 'partial' if batch.get('completed') else 'failed'
                    error = '部分谱面生成失败' if status == 'partial' else '谱面输出缺失或生成失败'
                finished += 1
                _update(task, index, status=status, error=error, produced=batch.get('completed', int(status == 'completed')))
                with task.lock:
                    task.result.pop('active_task_id', None)
                    task.progress = finished / len(rows)
                persist()
            with task.lock:
                cancelled = task.cancel_requested
                if cancelled:
                    task.result['jobs'] = [{**row, 'status': 'cancelled'} if row['status'] == 'queued' else row
                                           for row in task.result['jobs']]
                succeeded = any(row['status'] in ('completed', 'partial') for row in task.result['jobs'])
                task.status = 'cancelled' if cancelled else 'completed' if succeeded else 'failed'
                task.message = '队列已停止' if cancelled else '队列已处理完毕'
        except Exception as exc:
            with task.lock:
                task.status = 'failed'
                task.error = str(exc)
                task.message = '队列运行失败'
                task.result.pop('active_task_id', None)
                task.result['jobs'] = [{**row, 'status': 'interrupted'} if row['status'] == 'queued' else row
                                       for row in task.result['jobs']]
        finally:
            try:
                persist()
            except OSError as exc:
                with task.lock:
                    task.result['persistence_error'] = str(exc)

    try:
        persist()
        threading.Thread(target=run, name=f'generation-queue-{task.task_id}', daemon=True).start()
    except Exception as exc:
        with task.lock:
            task.status = 'failed'
            task.error = str(exc)
        raise
    return handler._json_response(dict(valid=True, task_id=task.task_id))


def status(queue_id, *, runtime):
    if not re.fullmatch(r'[a-f0-9]{12}', queue_id):
        return None
    task = runtime._get_task(queue_id)
    if task is None:
        try:
            state = json.loads((_directory(runtime, queue_id) / 'queue.json').read_text(encoding='utf-8'))
        except (FileNotFoundError, ValueError):
            return None
        if state['status'] not in TERMINAL:
            state.update(status='interrupted', message='服务已重启，保留已完成的结果')
            for row in state['result']['jobs']:
                if row['status'] not in TERMINAL:
                    row['status'] = 'interrupted'
        return state
    if task.kind != KIND:
        return None
    state = task.to_dict()
    active_id = state['result'].get('active_task_id')
    child = runtime._get_task(active_id) if any(
        row.get('task_id') == active_id and row['status'] == 'running' for row in state['result']['jobs']) else None
    if child is not None:
        with child.lock:
            progress, message, error = child.progress, child.message, child.cancel_error
        state['result']['active_job'] = dict(progress=progress, message=message)
        if error and state['status'] == 'cancelling':
            state.update(cancel_error=error, cancel_retryable=True)
    return state
