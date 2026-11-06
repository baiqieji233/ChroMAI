"""Read-only preview of generation artifacts and their original task audio."""
from __future__ import annotations

from collections import OrderedDict
from datetime import datetime
import hashlib
import json
import mimetypes
from pathlib import Path
import re
import secrets
import threading
from urllib.parse import quote

GENERATION_KINDS = {'generate','form_generate','composer_generate','composer_v2_generate','chart_stream_generate'}
_resources = OrderedDict()
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


def _saved(task_id, log_root):
    if not re.fullmatch(r'[0-9a-f]{6,64}',str(task_id)):
        raise FileNotFoundError('生成任务不存在')
    matches=sorted(Path(log_root).glob('*_'+task_id+'.meta.json'),reverse=True)
    if not matches:raise FileNotFoundError('生成任务记录不存在')
    return json.loads(matches[0].read_text(encoding='utf-8'))


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
    argv=record.get('argv') or []
    settings=result.get('effective_settings') or {}
    # Actual child-process arguments win over editable current UI fields.
    output=_argument(argv,'--output') or result.get('output') or settings.get('output')
    audio=_argument(argv,'--audio') or settings.get('audio') or result.get('audio')
    candidates=list(result.get('paths') or [])
    discovered_repair=None
    if output:
        target=_path(output,project)
        discovered_repair=target.with_name(target.stem+'.repaired.txt')
        candidates.extend([target,discovered_repair])
    def timestamp(key):
        try:return datetime.strptime(record.get(key,''),'%Y-%m-%d %H:%M:%S').timestamp()
        except (ValueError,TypeError):return None
    started,ended=timestamp('started'),timestamp('ended')
    audio_path=_path(audio,project) if audio else None
    audio_url=_resource(audio_path) if audio_path is not None and audio_path.is_file() else None
    charts=[]
    seen=set()
    for value in candidates:
        path=_path(value,project)
        if path in seen or path.suffix.lower() not in {'.txt','.maidata'} or not path.is_file():continue
        seen.add(path)
        # Persisted logs may refer to old non-isolated output paths. Never pair
        # a subsequently overwritten chart with that old task's music.
        modified=path.stat().st_mtime
        # Log timestamps have one-second precision; allow its final fraction,
        # without accepting files left by an earlier generation.
        if ended is not None and modified>ended+1:continue
        if started is not None and modified<started:continue
        if path==discovered_repair and (started is None or ended is None):continue
        with path.open('r',encoding='utf-8-sig',errors='replace') as stream:header=stream.read(65536)
        if not re.search(r'(?im)^\s*&inote_\d+\s*=',header):continue
        title=re.search(r'(?im)^\s*&title\s*=([^\r\n]*)',header)
        charts.append(dict(label=('修正版 · ' if '.repaired.' in path.name else '生成谱面 · ')+path.name,
            title=title.group(1).strip() if title else path.stem,path=str(path),url=_resource(path),
            audio_url=audio_url,audio_name=audio_path.name if audio_url else None))
    if not charts:raise FileNotFoundError('本任务没有可预览的谱面文件，或原文件已被覆盖')
    revision=hashlib.sha256(json.dumps(charts,sort_keys=True).encode()).hexdigest()[:20]
    return dict(task_id=task_id,revision=revision,charts=charts,
        message='' if audio_url else '未找到本任务的对应音频，可静音预览谱面。')


def latest(*, project, log_root):
    for path in sorted(Path(log_root).glob('*.meta.json'),reverse=True)[:250]:
        try:
            record=json.loads(path.read_text(encoding='utf-8'))
            if record.get('status')=='completed' and record.get('kind') in GENERATION_KINDS:
                return describe(record['task_id'],project=project,log_root=log_root)
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
