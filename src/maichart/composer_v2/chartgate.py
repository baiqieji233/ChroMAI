"""Run the pinned Chart;Gate release locally for Composer v2."""
from __future__ import annotations

import argparse
from contextlib import suppress
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from xml.sax.saxutils import escape
from .difficulty import CHARTGATE_SLOTS


REPOSITORY = 'https://github.com/inm2002/chartgate.git'
REVISION = '44d07804de32e66fc66f93c98ab67e2a981b7753'
MODEL_SHA256 = '8a33b3c7b31c99379f61a43519be19712d9de71892a3300fc7743032e92d5012'
IDENTITY = f'chartgate:{REVISION}:v1'
_SOURCE_SHA256 = 'ef670d62b104f267ad014c751e1fbd96c9f1272d0f22a6b7c2ab68196807e31c'
_FLAGS = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
_READER = r"""
using System.Globalization;
using System.Text.Json;
using MajSimaiBrowser;

CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;
var output = Console.Out;
Console.SetOut(Console.Error);
while (Console.ReadLine() is string line)
{
    try
    {
        var request = JsonSerializer.Deserialize<Request>(line)!;
        var charts = BrowserMaidataParser.Parse(request.Chart, [request.Slot]).Charts;
        if (charts.Length != 1) throw new InvalidDataException("未读到指定难度的谱面");
        output.WriteLine(JsonSerializer.Serialize(charts[0], BrowserMaidataParser.JsonOptions));
    }
    catch (Exception error)
    {
        output.WriteLine(JsonSerializer.Serialize(new { error = error.Message }));
    }
}
record Request(string Chart, int Slot);
"""


def _text_hash(path):
    return sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def _validate_checkout(root):
    source = [root/'BrowserMaidataParser.cs', *root.glob('third_party/MajSimai/Runtime/**/*.cs'),
              *root.glob('wwwroot/python/maimai_style/**/*.py')]
    try:
        fingerprints = sorted((file.relative_to(root).as_posix(), _text_hash(file)) for file in source)
        combined = sha256('\n'.join(f'{path} {digest}' for path,digest in fingerprints).encode()).hexdigest()
        valid = combined == _SOURCE_SHA256 and _text_hash(root/'wwwroot/model/model.json') == MODEL_SHA256
    except OSError:
        valid = False
    if not valid:
        raise ValueError(f'Chart;Gate 文件不完整或版本不同。请使用 {REVISION[:12]}。')
    return root


def locate_checkout(cache_dir, path=None, *, download=False):
    project = Path(__file__).resolve().parents[3]
    destination = Path(cache_dir).resolve() / 'chartgate' / 'source'
    path = path or os.environ.get('MAICHART_CHARTGATE_DIR')
    if path:
        return _validate_checkout(Path(path).expanduser().resolve())
    for candidate in (destination, project/'.scratch/chartgate',
                      project/'.scratch/difficulty_compare_20261112/chartgate'):
        if candidate.is_dir():
            return _validate_checkout(candidate)
    if not download:
        raise ValueError('请先在训练界面点击「初始化 Chart;Gate」。')
    if not shutil.which('git'):
        raise RuntimeError('下载 Chart;Gate 需要 Git。请安装 Git 后再初始化。')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='download-', dir=destination.parent) as temp:
        root = Path(temp)/'source'
        root.mkdir()
        _run(['git','init','--quiet'], cwd=root)
        _run(['git','fetch','--quiet','--depth=1',REPOSITORY,REVISION], cwd=root, timeout=180)
        _run(['git','checkout','--quiet','--detach','FETCH_HEAD'], cwd=root)
        _validate_checkout(root)
        _publish_directory(root, destination)
    return _validate_checkout(destination)


def _publish_directory(source, destination):
    try:
        source.rename(destination)
    except OSError:
        if not destination.is_dir():
            raise


def _run(argv, *, cwd=None, timeout=90):
    result = subprocess.run(argv,cwd=cwd,timeout=timeout,capture_output=True,
                            encoding='utf-8',errors='replace',creationflags=_FLAGS)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip()[-2400:] or 'Chart;Gate 初始化失败')
    return result.stdout


class ChartGateScorer:
    source = 'chartgate'
    identity = IDENTITY
    supported_slots = CHARTGATE_SLOTS

    def __init__(self, root, cache_dir, *, read_only=False):
        self.root = root
        self.cache = Path(cache_dir).resolve()/'chartgate'/f'{REVISION[:12]}_v1'
        self.read_only = read_only
        self.worker = None
        self.hits = self.computed = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self.worker is None:
            return
        with suppress(BrokenPipeError):
            self.worker.stdin.close()
        try:
            self.worker.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.worker.kill()
            self.worker.wait()
        self.worker.stdout.close()
        self.worker = None

    def _reader(self):
        if not shutil.which('dotnet'):
            raise RuntimeError('初始化 Chart;Gate 需要 .NET SDK 9 或更新版本。')
        sdks = [int(line.split('.')[0]) for line in _run(['dotnet','--list-sdks']).splitlines()
                if line[:1].isdigit()]
        if max(sdks, default=0) < 9:
            raise RuntimeError('初始化 Chart;Gate 需要 .NET SDK 9 或更新版本。')
        framework = f'net{max(sdks)}.0'
        output = self.cache/framework
        dll = output/'ChartGateReader.dll'
        if dll.is_file():
            return dll
        with tempfile.TemporaryDirectory(prefix='build-',dir=self.cache) as folder:
            folder = Path(folder)
            parser = escape(str(self.root/'BrowserMaidataParser.cs'))
            runtime = escape(str(self.root/'third_party/MajSimai/Runtime'/'**'/'*.cs'))
            project = folder/'ChartGateReader.csproj'
            project.write_text(f'''<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType><TargetFramework>{framework}</TargetFramework>
    <ImplicitUsings>enable</ImplicitUsings><Nullable>enable</Nullable>
    <AllowUnsafeBlocks>true</AllowUnsafeBlocks><NuGetAudit>false</NuGetAudit>
  </PropertyGroup>
  <ItemGroup><Compile Include="{parser}" /><Compile Include="{runtime}" /></ItemGroup>
</Project>''',encoding='utf-8')
            (folder/'Program.cs').write_text(_READER,encoding='utf-8')
            compiled = folder/'compiled'
            _run(['dotnet','build',str(project),'-c','Release','-o',str(compiled),'--nologo','-v','quiet'],
                 cwd=folder,timeout=120)
            _publish_directory(compiled,output)
        return dll

    def _dependencies(self):
        from importlib.util import find_spec
        target = self.cache/f'python{sys.version_info.major}{sys.version_info.minor}'
        if find_spec('pandas') is not None or (target/'pandas').is_dir():
            return target
        with tempfile.TemporaryDirectory(prefix='dependencies-',dir=self.cache) as temp:
            output = Path(temp)/'installed'
            _run([sys.executable,'-m','pip','install','--disable-pip-version-check','--no-deps',
                  '--target',str(output),'pandas==2.3.3','python-dateutil==2.9.0.post0',
                  'six==1.17.0','pytz','tzdata'],timeout=180)
            _publish_directory(output,target)
        return target

    def _receive(self):
        line = self.worker.stdout.readline()
        if not line:
            raise RuntimeError('Chart;Gate 进程意外退出，请查看任务日志。')
        return json.loads(line)

    def start(self):
        if self.worker is not None:
            return
        self.cache.mkdir(parents=True,exist_ok=True)
        reader, dependencies = self._reader(), self._dependencies()
        self.worker = subprocess.Popen([sys.executable,'-I','-B','-u',
            str(Path(__file__).with_name('_chartgate_worker.py')),str(self.root),str(reader),str(dependencies)],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,encoding='utf-8',creationflags=_FLAGS,
            env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
        if self._receive() != {'ready': True}:
            raise RuntimeError('Chart;Gate 初始化失败，请查看任务日志。')

    def score_path(self, digest, slot):
        return self.cache/'scores'/digest[:2]/f'{digest}_{slot}.json'

    def save_score(self, digest, slot, level):
        path = self.score_path(digest,slot)
        path.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,
                                         suffix='.tmp',delete=False) as pending:
            json.dump({'level':level},pending,allow_nan=False)
        try:
            Path(pending.name).replace(path)
        finally:
            Path(pending.name).unlink(missing_ok=True)

    def score(self, digest, slot, source, chart_bytes):
        if slot not in self.supported_slots:
            raise ValueError('Chart;Gate 支持 MASTER 和 Re:MASTER。')
        from .open_mdp import _cached_score
        value = _cached_score(self.score_path(digest,slot))
        if value is not None:
            self.hits += 1
            return value
        if self.read_only:
            raise RuntimeError(f'缺少 Chart;Gate 估分缓存：{source}（槽位 {slot}）')
        from maichart.maidata import decode_text_with_encoding
        self.start()
        try:
            self.worker.stdin.write(json.dumps(dict(Chart=decode_text_with_encoding(chart_bytes)[0],Slot=slot))+'\n')
            self.worker.stdin.flush()
        except BrokenPipeError as error:
            raise RuntimeError('Chart;Gate 进程意外退出，请查看任务日志。') from error
        result = self._receive()
        if 'error' in result:
            raise ValueError(f'Chart;Gate 无法估分：{result["error"]}')
        level = float(result['level'])
        if not math.isfinite(level):
            raise ValueError('Chart;Gate 返回了无效的定数。')
        self.save_score(digest,slot,level)
        self.computed += 1
        return level


def main(argv=None):
    parser = argparse.ArgumentParser(description='初始化 Chart;Gate')
    parser.add_argument('--cache-dir',default='cache/chart_composer_v2')
    parser.add_argument('--chartgate-dir')
    args = parser.parse_args(argv)
    root = locate_checkout(args.cache_dir,args.chartgate_dir,download=True)
    print('正在初始化 Chart;Gate…',flush=True)
    with ChartGateScorer(root,args.cache_dir) as scorer:
        scorer.start()
    print('Chart;Gate 已就绪。',flush=True)


if __name__ == '__main__':
    main()
