"""Read-only preview of generation artifacts and their original task audio."""
from __future__ import annotations

from collections import OrderedDict
from datetime import datetime
import hashlib
import json
import math
import mimetypes
from pathlib import Path
import re
import secrets
import shutil
import tempfile
import threading
from urllib.parse import quote

GENERATION_KINDS = {'generate','form_generate','composer_generate','composer_v2_generate','chart_stream_generate'}
_resources = OrderedDict()
_hashes = OrderedDict()
_lock = threading.Lock()


def _path(value, project):
    path = Path(str(value))
    return (path if path.is_absolute() else Path(project)/path).resolve()


def _argument(argv, flag):
    for index,value in enumerate(argv):
        if value==flag and index+1<len(argv):return argv[index+1]
        if value.startswith(flag+'='):return value[len(flag)+1:]
    return None


def _resource(path):
    stat=path.stat()
    key=(str(path),stat.st_mtime_ns,stat.st_size)
    with _lock:
        for token,(previous,_) in _resources.items():
            if previous==key:return '/api/preview/resource?id='+token
        token=secrets.token_urlsafe(24)
        _resources[token]=(key,path)
        while len(_resources)>512:_resources.popitem(last=False)
    return '/api/preview/resource?id='+token


def resource_path(token):
    with _lock:
        entry=_resources.get(token)
        if entry is not None:_resources.move_to_end(token)
    if entry is None:raise FileNotFoundError('预览资源已失效，请重新载入谱面')
    key,path=entry
    stat=path.stat()
    if (str(path),stat.st_mtime_ns,stat.st_size)!=key:
        raise FileNotFoundError('预览源文件已改变，请重新载入谱面')
    return path


def _identities(path, difficulty_index=None):
    """Hash immutable preview inputs once, rechecking their file stamp."""
    stat=path.stat()
    key=(str(path),stat.st_mtime_ns,stat.st_size,difficulty_index)
    with _lock:
        cached=_hashes.get(key)
        if cached is not None:
            _hashes.move_to_end(key)
            return cached
    digest=hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda:source.read(1024*1024),b''):digest.update(block)
    value={'file_sha256':digest.hexdigest()}
    if difficulty_index is not None:
        from maichart.training.quality_annotations import chart_sha256
        value['chart_sha256']=chart_sha256(path,difficulty_index)
    current=path.stat()
    if (current.st_mtime_ns,current.st_size)!=(stat.st_mtime_ns,stat.st_size):
        raise ValueError('预览源文件已改变，请重新载入谱面')
    with _lock:
        _hashes[key]=value
        while len(_hashes)>256:_hashes.popitem(last=False)
    return value


def _history_path(task_id, log_root):
    if not re.fullmatch(r'[0-9a-f]{6,64}',str(task_id)):
        raise FileNotFoundError('生成任务不存在')
    return Path(log_root).parent/'generation_results'/task_id/'history.json'


def _saved(task_id, log_root):
    path = _history_path(task_id, log_root)
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        pass
    matches=sorted(Path(log_root).glob('*_'+task_id+'.meta.json'),reverse=True)
    if not matches:raise FileNotFoundError('生成任务记录不存在')
    return json.loads(matches[0].read_text(encoding='utf-8'))


def save_history(record, *, log_root, log_path):
    """Keep completed artifacts discoverable independently of log retention."""
    target = _history_path(record['task_id'], log_root)
    result = record['result']
    archive = result['generation_archive']
    if not archive['charts'] or target.is_file():
        return
    value = {key: record[key] for key in ('task_id', 'kind', 'status', 'argv', 'started', 'ended')}
    value['log'] = str(log_path) if log_path else ''
    value['result'] = {key: result[key] for key in ('generation_archive', 'paths', 'files', 'download_urls')
                       if key in result}
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix='.tmp', delete=False,
                                     mode='w', encoding='utf-8') as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, ensure_ascii=False)
            stream.close()
            temporary.replace(target)
        finally:
            stream.close()
            temporary.unlink(missing_ok=True)


def describe(task_id, *, project, log_root, task=None):
    if task is None:
        record=_saved(task_id,log_root)
    elif isinstance(task,dict):
        record=task
    else:
        record=task.to_dict()
        record['argv']=list(task.argv)
        record['started']=task.started_at
        record['ended']=task.ended_at
    if record.get('kind') not in GENERATION_KINDS or record.get('status')!='completed':
        raise ValueError('仅能预览已完成的生成任务')
    result=record.get('result') or {}
    if result.get('generation_archive_error'):
        raise FileNotFoundError(result['generation_archive_error'])
    archive=result.get('generation_archive')
    archived_charts={Path(chart['path']):chart for chart in archive['charts']} if archive is not None else {}
    argv=record.get('argv') or []
    settings=result.get('effective_settings') or {}
    audio_duration=None
    for metadata in [archive,result,result.get('composer_v2_report'),result.get('composer_report'),result.get('stream_report')]:
        if not isinstance(metadata,dict):continue
        try:duration=float(metadata.get('duration_sec'))
        except (TypeError,ValueError):continue
        if math.isfinite(duration) and duration>0:
            audio_duration=duration
            break
    # Actual child-process arguments win over editable current UI fields.
    output=_argument(argv,'--output') or result.get('output') or settings.get('output')
    audio=_argument(argv,'--audio') or settings.get('audio') or result.get('audio')
    result_paths=list(result.get('paths') or [])
    # Persisted task logs contain argv even after the server has restarted.
    # The batch manifest keeps every chart's seed and original, isolated path.
    batch_path=_argument(argv,'--batch-manifest') or settings.get('batch_manifest')
    batch_labels={}
    batch=result.get('composer_v2_batch')
    if batch_path and archive is None:
        manifest=_path(batch_path,project)
        if manifest.is_file():batch=json.loads(manifest.read_text(encoding='utf-8-sig'))
    candidates=[]
    if archive is not None:
        candidates.extend(archived_charts)
    elif isinstance(batch,dict):
        audio=audio or batch.get('audio')
        for item in batch.get('charts', []):
            if item.get('status')!='completed' or not item.get('output'):continue
            path=_path(item['output'],project)
            candidates.append(path)
            batch_labels[path]=f"本次第 {item.get('index',len(candidates))} 张" + (f" · seed {item['seed']}" if 'seed' in item else '')
    elif output:
        candidates.append(_path(output,project))
    # Composer v2's task output or completed batch entries are authoritative.
    # Phrase-search snapshots have only a prefix and a trial, never a complete
    # alternative continuation. Do not open their reports or offer them here.
    if archive is None and (record.get('kind')!='composer_v2_generate' or (not output and not isinstance(batch,dict))):
        candidates.extend(result_paths)
    explicit_paths={_path(value,project) for value in result_paths}
    discovered_repairs=set()
    repair_originals={}
    for value in list(candidates) if archive is None else ():
        target=_path(value,project)
        if target.suffix.lower() not in {'.txt','.maidata'} or '.repaired.' in target.name:continue
        repair=target.with_name(target.stem+'.repaired.txt')
        repair_originals[repair]=target
        if repair not in explicit_paths:discovered_repairs.add(repair)
        candidates.append(repair)
    def timestamp(key):
        try:return datetime.strptime(record.get(key,''),'%Y-%m-%d %H:%M:%S').timestamp()
        except (ValueError,TypeError):return None
    started,ended=timestamp('started'),timestamp('ended')
    audio_path=_path(audio,project) if audio else None
    audio_hash=_identities(audio_path)['file_sha256'] if audio_path is not None and audio_path.is_file() else None
    if archive is not None and audio_hash != archive['audio_sha256']:
        audio_hash=None
    audio_url=_resource(audio_path) if audio_hash else None
    charts=[]
    seen=set()
    for value in candidates:
        path=_path(value,project)
        if 'phrase_candidates' in {part.casefold() for part in path.parts}:continue
        if path in seen or path.suffix.lower() not in {'.txt','.maidata'} or not path.is_file():continue
        seen.add(path)
        # Persisted logs may refer to old non-isolated output paths. Never pair
        # a subsequently overwritten chart with that old task's music.
        file_stat=path.stat()
        modified=file_stat.st_mtime
        # Log timestamps have one-second precision; allow its final fraction,
        # without accepting files left by an earlier generation.
        if archive is None:
            if ended is not None and modified>ended+1:continue
            if started is not None and modified<started:continue
        if path in discovered_repairs and (started is None or ended is None):continue
        from maichart.maidata import parse_maidata_file
        metadata=parse_maidata_file(path)
        difficulty=next((item for item in metadata.difficulties if (item.inote or '').strip()),None)
        if difficulty is None:continue
        index=difficulty.index
        hashes=_identities(path,index)
        saved=archived_charts.get(path)
        if saved and hashes['file_sha256']!=saved['file_sha256']:continue
        current=path.stat()
        if (current.st_mtime_ns,current.st_size)!=(file_stat.st_mtime_ns,file_stat.st_size):
            raise ValueError('预览源文件已改变，请重新载入谱面')
        repaired='.repaired.' in path.name
        original=repair_originals.get(path,path.with_name(path.name.replace('.repaired.','.'))) if repaired else path
        label=batch_labels.get(original) or '完整谱面'
        if repaired:label+=' · 修复版'
        if saved:label=saved['label']
        identity=dict(task_id=str(task_id),path=str(path),difficulty_index=index,**hashes,audio_sha256=audio_hash)
        version_id=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:24]
        charts.append(dict(label=label,content_kind='complete_chart',version_id=version_id,**hashes,
            title=metadata.title or path.stem,path=str(path),url=_resource(path),
            audio_url=audio_url,audio_name=audio_path.name if audio_url else None, audio_path=str(audio_path) if audio_path else None,
            audio_sha256=audio_hash,audio_duration_sec=audio_duration,difficulty_index=index,
            timeline_url='/api/preview/timeline?id='+quote(str(task_id))+'&version='+version_id))
    if not charts:raise FileNotFoundError('本任务没有可预览的谱面文件，或原文件已被覆盖')
    if archive is None and not batch_labels:
        def original_path(chart):
            path=Path(chart['path'])
            return repair_originals.get(path,path.with_name(path.name.replace('.repaired.','.')))
        originals=list(dict.fromkeys(original_path(chart) for chart in charts))
        if len(originals)>1:
            for chart in charts:
                number=originals.index(original_path(chart))+1
                chart['label']=f'本次第 {number} 张'+(' · 修复版' if '.repaired.' in Path(chart['path']).name else '')
    revision=hashlib.sha256(json.dumps(charts,sort_keys=True).encode()).hexdigest()[:20]
    return dict(task_id=task_id,revision=revision,charts=charts,
        message='' if audio_url else '未找到本任务的对应音频，可静音预览谱面。')


def generation_stamps(paths):
    """Remember outputs already present before the child starts."""
    stamps={}
    for path in paths:
        options=[path]
        if path.suffix.lower() in {'.txt','.maidata'} and '.repaired.' not in path.name:
            options.append(path.with_name(path.stem+'.repaired.txt'))
        for option in options:
            try:
                stat=option.stat()
            except FileNotFoundError:
                continue
            stamps[option.resolve()]=(stat.st_mtime_ns,stat.st_size)
    return stamps


def archive_generation(record, *, project, log_root, prior=None):
    """Save this run's files before another run may reuse the output path."""
    result=record['result']
    try:
        charts=describe(record['task_id'],project=project,log_root=log_root,task=record)['charts']
    except FileNotFoundError:
        charts=[]
    paths={_path(path,project):label for path,label in zip(result['paths'],result['files'])}
    batch_path = _argument(record.get('argv') or (), '--batch-manifest')
    batch_root = _path(batch_path, project).parent if batch_path else None
    report=result.get('composer_v2_report') or {}
    conflict=_path(report['conflict_report'],project) if report.get('conflict_report') else None
    if conflict is not None and conflict.is_file():
        paths.setdefault(conflict,conflict.name)
    for chart in charts:
        path=Path(chart['path'])
        paths.setdefault(path,chart['label'])
    start=datetime.strptime(record['started'],'%Y-%m-%d %H:%M:%S').timestamp()
    root=Path(log_root).parent/'generation_results'
    root.mkdir(parents=True,exist_ok=True)
    dest=root/record['task_id']
    saved={}
    with tempfile.TemporaryDirectory(dir=root) as work:
        stage=Path(work)/'files'
        stage.mkdir()
        for i,path in enumerate(paths):
            before=path.stat()
            if before.st_mtime<start or (prior or {}).get(path)==(before.st_mtime_ns,before.st_size):
                continue
            # The batch review reads the manifest and its results by relative
            # path, including after this archive has moved.
            relative = (Path('batch') / path.relative_to(batch_root)
                        if batch_root is not None and path.is_relative_to(batch_root)
                        else Path(str(i)) / path.name)
            target=stage/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path,target)
            after=path.stat()
            if (after.st_mtime_ns,after.st_size)!=(before.st_mtime_ns,before.st_size):
                raise OSError(f'归档过程中生成文件被更改：{path.name}')
            saved[path]=dest/relative
        stage.rename(dest)
    selected=[dict(path=str(saved[Path(c['path'])]), label=c['label'], file_sha256=c['file_sha256'])
              for c in charts if Path(c['path']) in saved]
    if conflict is not None:
        report['conflict_report']=str(saved[conflict]) if conflict in saved else None
    result.update(
        generation_archive=dict(charts=selected,
            title=charts[0]['title'] if charts else None,
            audio_sha256=charts[0]['audio_sha256'] if charts else None,
            duration_sec=charts[0]['audio_duration_sec'] if charts else None),
        paths=[str(path) for path in saved.values()],
        files=[paths[path] for path in saved],
        download_urls=['/api/download?path='+quote(str(path)) for path in saved.values()],
    )


def historical_task(task_id, *, project, log_root):
    """Reopen completed generation from disk, including logs predating archives."""
    record=_saved(task_id,log_root)
    if record.get('kind') not in GENERATION_KINDS or record.get('status')!='completed':
        raise FileNotFoundError('没有已完成的生成任务')
    result=record.get('result') or {}
    if 'generation_archive' not in result and 'generation_archive_error' not in result:
        try:
            charts=describe(task_id,project=project,log_root=log_root,task=record)['charts']
        except (OSError, ValueError) as exc:
            result={'generation_archive_error':str(exc)}
        else:
            result=dict(paths=[c['path'] for c in charts],files=[Path(c['path']).name for c in charts],
                        download_urls=[c['url'] for c in charts])
    path=Path(record.get('log') or '')
    logs=[]
    try:
        with path.open('rb') as stream:
            size=stream.seek(0,2)
            stream.seek(max(0,size-256*1024))
            logs=stream.read().decode('utf-8',errors='replace').splitlines()[-300:]
    except OSError:
        pass
    return dict(task_id=task_id,kind=record['kind'],status='completed',progress=1,
                message='历史生成任务',logs=logs,result=result)


def describe_timeline(task_id, version_id, *, project, log_root, task=None):
    """Resolve one complete, unchanged task output before reading its notes."""
    from maichart.webui.chart_timeline import timeline
    value=describe(task_id,project=project,log_root=log_root,task=task)
    chart=next((item for item in value['charts'] if item['version_id']==version_id),None)
    if chart is None:raise FileNotFoundError('完整谱面版本已改变或不存在，请重新载入')
    result=timeline(chart['path'],chart['difficulty_index'],audio_duration_sec=chart.get('audio_duration_sec'))
    # The timeline parser is cached by the exact same file stamp as resources.
    # Revalidate after parsing so a concurrent replacement cannot mix versions.
    token=chart['url'].split('id=',1)[1]
    resource_path(token)
    return dict(result,task_id=task_id,version_id=version_id)


def _history_records(log_root):
    logs = Path(log_root)
    records = {}
    paths = list((logs.parent/'generation_results').glob('*/history.json'))
    paths.extend(path for path in logs.glob('*.meta.json')
                 if any(f'_{kind}_' in path.name for kind in GENERATION_KINDS))
    for path in paths:
        try:
            record = json.loads(path.read_text(encoding='utf-8'))
            if record['status'] != 'completed' or record['kind'] not in GENERATION_KINDS:
                continue
            task_id = record['task_id']
            if task_id not in records:
                records[task_id] = (record.get('ended') or '', path.stat().st_mtime_ns, record)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return (record for _, _, record in sorted(records.values(), key=lambda item: item[:2], reverse=True))


def history(*, project, log_root):
    runs = []
    for record in _history_records(log_root):
        argv = record.get('argv') or []
        result = record.get('result') or {}
        archive = result.get('generation_archive')
        try:
            if archive is not None:
                chart_count = sum(_path(chart['path'], project).is_file() for chart in archive['charts'])
                if not chart_count:
                    continue
            else:
                # Legacy shared outputs may now belong to another run.
                if result.get('generation_archive_error'):
                    continue
                output = _argument(argv, '--batch-manifest') or _argument(argv, '--output')
                if not output:
                    continue
                modified = _path(output, project).stat().st_mtime
                start = datetime.fromisoformat(record['started']).timestamp() if record.get('started') else None
                end = datetime.fromisoformat(record['ended']).timestamp() if record.get('ended') else None
                if (start is not None and modified < start) or (end is not None and modified > end + 1):
                    continue
                chart_count = None
        except (OSError, ValueError, KeyError, TypeError):
            continue
        audio = Path(_argument(argv, '--audio') or '').stem
        audio = re.sub(r'^[0-9a-f]{12}_', '', audio)
        title = (archive or {}).get('title')
        if not title or title == 'Generated maimai chart':
            title = audio or '生成谱面'
        runs.append(dict(task_id=record['task_id'], kind=record['kind'], ended=record.get('ended'),
                         title=title, chart_count=chart_count, seed=_argument(argv, '--seed'),
                         level=_argument(argv, '--level')))
    return runs


def latest(*, project, log_root):
    for record in _history_records(log_root):
        try:
            return describe(record['task_id'],project=project,log_root=log_root,task=record)
        except (OSError,ValueError,KeyError,TypeError):continue
    return None


def serve_file(handler, path, *, head=False, immutable=False):
    """Bounded streaming, correct WASM MIME and media Range/HEAD support."""
    path=Path(path)
    if not path.is_file():return handler._json_response({'error':'文件不存在'},404)
    mime={'.wasm':'application/wasm','.mjs':'text/javascript','.js':'text/javascript',
        '.txt':'text/plain; charset=utf-8','.maidata':'text/plain; charset=utf-8'}.get(path.suffix.lower())
    mime=mime or mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
    compressed=False
    if immutable and not handler.headers.get('Range') and 'gzip' in handler.headers.get('Accept-Encoding','').lower() and path.with_name(path.name+'.gz').is_file():
        path=path.with_name(path.name+'.gz');compressed=True
    size=path.stat().st_size
    start,end=0,size-1
    requested=handler.headers.get('Range')
    if requested:
        match=re.fullmatch(r'bytes=(\d*)-(\d*)',requested.strip())
        if match and any(match.groups()) and size:
            if match.group(1):
                start=int(match.group(1));end=min(size-1,int(match.group(2))) if match.group(2) else size-1
            else:start=max(0,size-int(match.group(2)))
        if not match or not any(match.groups()) or not size or start>=size or end<start:
            handler.send_response(416);handler.send_header('Content-Range',f'bytes */{size}')
            handler.send_header('Content-Length','0');handler.end_headers();return
    length=max(0,end-start+1)
    handler.send_response(206 if requested else 200)
    handler.send_header('Content-Type',mime)
    handler.send_header('Content-Length',str(length))
    handler.send_header('Accept-Ranges','bytes')
    handler.send_header('Cache-Control','public, max-age=31536000, immutable' if immutable else 'private, no-store')
    if compressed:
        handler.send_header('Content-Encoding','gzip');handler.send_header('Vary','Accept-Encoding')
    handler.send_header('Content-Disposition',"inline; filename*=UTF-8''"+quote(path.name))
    if requested:handler.send_header('Content-Range',f'bytes {start}-{end}/{size}')
    handler.end_headers()
    if head:return
    try:
        with path.open('rb') as source:
            source.seek(start)
            remaining=length
            while remaining:
                block=source.read(min(1024*1024,remaining))
                if not block:break
                handler.wfile.write(block);remaining-=len(block)
    except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):pass
