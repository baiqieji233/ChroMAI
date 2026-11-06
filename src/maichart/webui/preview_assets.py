"""Install a complete MiaView build locally; no Node/Qt or model imports needed."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
import threading
import urllib.request

UPSTREAM = 'https://team-miacode.github.io/MiaView_Web/'
REPOSITORY = 'https://github.com/Team-MiaCode/MiaView_Web'
SDK_FILES = ('index.mjs', 'resolution.mjs', 'protocol.mjs', 'layout.mjs', 'frame-rate.mjs')
_lock = threading.Lock()
_state = dict(installing=False, message='', error=None)


def _relative(value):
    part = PurePosixPath(value)
    if part.is_absolute() or any(p in {'..', '.'} for p in part.parts) or '\\' in value or ':' in value:
        raise ValueError('Invalid player asset path')
    return str(part)


def _bytes(name):
    with urllib.request.urlopen(UPSTREAM+_relative(name), timeout=60) as response:
        content = response.read(2*1024*1024+1)
    if len(content) > 2*1024*1024:
        raise ValueError('Player metadata is too large')
    return content


def _download(name, target, expected, progress):
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    count = 0
    with urllib.request.urlopen(UPSTREAM+_relative(name), timeout=60) as response, target.open('wb') as output:
        while block := response.read(1024*1024):
            count += len(block)
            if count > expected['byteSize']:
                raise ValueError('Player asset exceeds its declared size: '+name)
            digest.update(block)
            output.write(block)
    if count != expected['byteSize'] or digest.hexdigest() != expected['sha256']:
        raise ValueError('Player asset checksum mismatch: '+name)
    progress('已下载 '+name.rsplit('/',1)[-1])


def _verify(path, expected):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    if path.stat().st_size != expected['byteSize'] or digest.hexdigest() != expected['sha256']:
        raise ValueError('Player asset checksum mismatch: '+str(path))


def install(root, progress=lambda message: None):
    """Stage and verify the build, then atomically publish its version pointer."""
    root = Path(root).resolve()
    releases = root/'releases'
    releases.mkdir(parents=True, exist_ok=True)
    manifest_bytes = _bytes('renderer/artifact-manifest.json')
    files = json.loads(manifest_bytes)['files']
    specs = {}
    for spec in files:
        name = _relative(spec['path'])
        if not re.fullmatch(r'[0-9a-f]{64}',spec['sha256']) or not isinstance(spec['byteSize'],int) or spec['byteSize'] < 0:
            raise ValueError('Invalid player artifact manifest')
        specs[name] = spec
    required = {'index.html','build-identity.json','miacode-preview.js','miacode-preview.wasm','LICENSE','LICENSE_SCOPE.md','THIRD_PARTY_NOTICES.md'}
    if not required.issubset(specs):
        raise ValueError('Incomplete MiaView artifact manifest')
    stage = Path(tempfile.mkdtemp(prefix='.install-', dir=releases)).resolve()
    try:
        sdk = {name:_bytes('sdk/'+name) for name in SDK_FILES}
        for name,content in sdk.items():
            target=stage/'sdk'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
        jobs = [(name,spec) for name,spec in specs.items() if not name.endswith(('.br','.gz'))]
        def download(item):
            name,spec=item
            target=stage/'renderer'/name
            if name=='miacode-preview.wasm' and name+'.gz' in specs:
                compressed=target.with_suffix('.wasm.gz')
                _download('renderer/'+name+'.gz',compressed,specs[name+'.gz'],progress)
                with gzip.open(compressed,'rb') as source,target.open('wb') as output:
                    shutil.copyfileobj(source,output,1024*1024)
                _verify(target,spec)
            else:
                _download('renderer/'+name,target,spec,progress)
        with ThreadPoolExecutor(max_workers=4,thread_name_prefix='miaview-download') as executor:
            list(executor.map(download,jobs))
        # Pages publishes SDK and renderer together. Do not activate a build
        # if deployment changed while downloading its files.
        if _bytes('renderer/artifact-manifest.json') != manifest_bytes:
            raise ValueError('Upstream player changed during download; retry installation')
        if any(_bytes('sdk/'+name) != content for name,content in sdk.items()):
            raise ValueError('Upstream SDK changed during download; retry installation')
        (stage/'renderer'/'artifact-manifest.json').write_bytes(manifest_bytes)
        signature=hashlib.sha256(manifest_bytes)
        for name in SDK_FILES:signature.update(sdk[name])
        release=signature.hexdigest()[:20]
        identity=json.loads((stage/'renderer'/'build-identity.json').read_text(encoding='utf-8'))
        record=dict(release=release,version=identity['version'],source=REPOSITORY,
            upstream_identity=identity,renderer_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
            sdk_sha256={name:hashlib.sha256(content).hexdigest() for name,content in sdk.items()})
        (stage/'installation.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
        destination=releases/release
        if not destination.exists():
            stage.rename(destination)
        descriptor,temporary=tempfile.mkstemp(dir=root,suffix='.json.tmp')
        try:
            with os.fdopen(descriptor,'w',encoding='utf-8') as stream:json.dump(record,stream,ensure_ascii=False)
            os.replace(temporary,root/'current.json')
        finally:
            if os.path.exists(temporary):os.unlink(temporary)
        progress('预览播放器已就绪')
        return record
    finally:
        # Only remove the freshly created staging directory, never a release
        # currently used by a browser or another process.
        if stage.exists() and stage.parent==releases and stage.name.startswith('.install-'):
            shutil.rmtree(stage)


def status(root):
    with _lock:result=dict(_state)
    result['available']=False
    try:
        record=json.loads((Path(root)/'current.json').read_text(encoding='utf-8'))
        release=record['release']
        if not re.fullmatch(r'[0-9a-f]{20}',release):return result
        base=Path(root)/'releases'/release
        if not all((base/p).is_file() for p in ('sdk/index.mjs','renderer/miacode-preview.wasm','renderer/build-identity.json')):return result
        result.update(available=True,version=record['version'],release=release,
            sdk_url=f'/miaview/{release}/sdk/index.mjs',renderer_url=f'/miaview/{release}/renderer/index.html')
    except (OSError,ValueError,KeyError,TypeError):pass
    return result


def start_install(root):
    with _lock:
        if _state['installing']:return
        _state.update(installing=True,error=None,message='正在下载预览播放器，首次约 35 MB')
    def update(message):
        with _lock:_state['message']=message
    def worker():
        try:install(root,update)
        except Exception as exc:
            with _lock:_state.update(error=str(exc),message='预览播放器安装失败')
        finally:
            with _lock:_state['installing']=False
    threading.Thread(target=worker,name='miaview-install',daemon=True).start()


def asset_path(root, relative):
    relative=_relative(relative)
    parts=PurePosixPath(relative).parts
    if len(parts)<3 or not re.fullmatch(r'[0-9a-f]{20}',parts[0]) or parts[1] not in {'sdk','renderer'}:
        raise ValueError('Invalid player asset URL')
    base=(Path(root)/'releases').resolve()
    path=(base/relative).resolve()
    if not path.is_relative_to(base):raise ValueError('Invalid player asset URL')
    return path


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description='Install the local MiaView preview player')
    parser.add_argument('--root',default='webui_workspace/preview_runtime/miaview')
    args=parser.parse_args(argv)
    record=install(args.root,lambda message:print(message,flush=True))
    print(json.dumps(record,ensure_ascii=False,indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
