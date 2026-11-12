"""Local, CPU-only open-mdp adapter with per-chart score caching."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import uuid
from xml.sax.saxutils import escape


REVISION = 'aac60d3b7ffbb41e23bc16724d92386c091969c7'
IDENTITY = f'open-mdp:{REVISION}:v1'
_ASSETS = {
    'serializer/src/SimaiProcess.cs': '28a819dee22507f2e467c846b2201f1555f8063b1f275973343ceaa45999894e',
    'inference/src/inference.py': '4c18972dfbb6e88e40d34999ff3050b97fb0239438e4564e38f54b0d8aaeb152',
    'inference/trained_models/lstm_model.pth': '2b448c169552c8ce1cd14742d81260fbf5876124f2410592bf367fc72305c1fb',
}
_FLAGS = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
_SERIALIZER = r"""
using System.Globalization;
using System.Text.Json;
using MajdataEdit;

CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;
var slot = int.Parse(args[1]) - 1;
if (!SimaiProcess.ReadData(args[0]) || string.IsNullOrWhiteSpace(SimaiProcess.fumens[slot]))
    return 2;
SimaiProcess.Serialize(SimaiProcess.fumens[slot]);
var points = SimaiProcess.notelist.Select(point => new {
    Time = point.time,
    Notes = point.getNotes().Select(n => new {
        n.holdTime, n.isBreak, n.isEx, n.isFakeRotate, n.isForceStar, n.isHanabi,
        n.isSlideBreak, n.isSlideNoHead, noteType = n.noteType.ToString(),
        n.slideStartTime, n.slideTime, n.startPosition, n.touchArea
    }).ToArray()
}).ToArray();
if (points.Length == 0 || points.Any(p => p.Notes.Length == 0))
    return 2;
File.WriteAllText(args[2], JsonSerializer.Serialize(points));
return 0;
"""


def locate_checkout(path=None):
    path = path or os.environ.get('MAICHART_OPEN_MDP_DIR')
    if not path:
        project = Path(__file__).resolve().parents[3]
        candidates = (project / '.scratch' / 'open-mdp',
                      project / '.scratch' / 'difficulty_compare_20261112' / 'open-mdp')
        path = next((candidate for candidate in candidates if candidate.is_dir()), None)
    if not path:
        raise ValueError('请选择 open-mdp 目录，或设置 MAICHART_OPEN_MDP_DIR')
    root = Path(path).expanduser().resolve()
    for name, expected in _ASSETS.items():
        file = root / name
        if not file.is_file():
            raise ValueError(f'open-mdp 缺少文件：{file}')
        content = file.read_bytes()
        if file.suffix in ('.cs', '.py'):
            content = content.replace(b'\r\n', b'\n')
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f'open-mdp 文件版本不同：{file}。请使用 {REVISION[:12]}。')
    return root


def _cached_score(path):
    try:
        value = json.loads(path.read_text(encoding='utf-8'))['level']
        if type(value) in (int, float) and math.isfinite(value):
            return float(value)
    except (OSError, ValueError, TypeError, KeyError):
        pass
    return None


class OpenMdpScorer:
    identity = IDENTITY

    def __init__(self, root, cache_dir, *, read_only=False):
        self.root = root
        self.cache = Path(cache_dir).resolve() / 'open_mdp' / f'{REVISION[:12]}_v1'
        self.read_only = read_only
        self.worker = None
        self.temp = None
        self.dll = None
        self.hits = self.computed = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self.worker is not None:
            self.worker.stdin.close()
            try:
                self.worker.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.worker.kill()
                self.worker.wait()
            self.worker.stdout.close()
        if self.temp is not None:
            self.temp.cleanup()

    def _start(self):
        if self.worker is not None:
            return
        if not shutil.which('dotnet'):
            raise RuntimeError('首次计算拟合定数需要安装 .NET SDK（8 或更新）')
        self.cache.mkdir(parents=True, exist_ok=True)
        self.dll = self._build()
        self.temp = tempfile.TemporaryDirectory(prefix='score-', dir=self.cache)
        self.worker = subprocess.Popen(
            [sys.executable, '-I', '-B', '-u', str(Path(__file__).with_name('_open_mdp_worker.py'))],
            cwd=self.root / 'inference', env={**os.environ, 'CUDA_VISIBLE_DEVICES': '-1',
                                               'PYTHONIOENCODING': 'utf-8'},
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, encoding='utf-8', creationflags=_FLAGS)
        if not self._response().get('ready'):
            raise RuntimeError('open-mdp 无法加载估分模型')

    def _response(self):
        line = self.worker.stdout.readline()
        if not line:
            raise RuntimeError('open-mdp 估分进程意外退出，请查看任务日志')
        return json.loads(line)

    def _build(self):
        result = subprocess.run(['dotnet', '--list-sdks'], capture_output=True, text=True,
                                timeout=30, check=True, creationflags=_FLAGS)
        majors = [int(line.split('.')[0]) for line in result.stdout.splitlines() if line[:1].isdigit()]
        if max(majors, default=0) < 8:
            raise RuntimeError('首次计算拟合定数需要安装 .NET SDK（8 或更新）')
        framework = f'net{max(majors)}.0'
        output = self.cache / framework
        dll = output / 'OpenMdpChartReader.dll'
        if dll.is_file():
            return dll
        with tempfile.TemporaryDirectory(prefix='build-', dir=self.cache) as temporary:
            source = Path(temporary)
            parser_path = escape(str(self.root / 'serializer/src/SimaiProcess.cs'))
            project = source / 'OpenMdpChartReader.csproj'
            project.write_text(f'''<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>{framework}</TargetFramework>
    <ImplicitUsings>enable</ImplicitUsings>
    <NuGetAudit>false</NuGetAudit>
  </PropertyGroup>
  <ItemGroup><Compile Include="{parser_path}" /></ItemGroup>
</Project>''', encoding='utf-8')
            (source / 'Program.cs').write_text(_SERIALIZER, encoding='utf-8')
            compiled = source / 'compiled'
            result = subprocess.run(['dotnet', 'build', str(project), '-c', 'Release', '-o', str(compiled),
                                     '--nologo', '-v', 'quiet'], cwd=source, capture_output=True,
                                    encoding='utf-8', errors='replace', timeout=120, creationflags=_FLAGS)
            if result.returncode:
                raise RuntimeError(f'open-mdp 解析器构建失败：{result.stdout}\n{result.stderr}')
            try:
                compiled.rename(output)
            except OSError:
                if not dll.is_file():
                    raise
        return dll

    def score(self, chart_sha, slot, source, chart_bytes):
        cache_path = self.cache / 'scores' / chart_sha[:2] / f'{chart_sha}_{slot}.json'
        value = _cached_score(cache_path)
        if value is not None:
            self.hits += 1
            return value
        if self.read_only:
            raise RuntimeError(f'只读评估缺少拟合定数缓存：{source}（难度槽 {slot}）')
        self._start()
        temporary = Path(self.temp.name)
        raw = temporary / 'notes.json'
        raw.unlink(missing_ok=True)
        # Match the bytes used for the training identity, including on a concurrent edit.
        from maichart.maidata import decode_text_with_encoding
        chart = temporary / 'maidata.txt'
        chart.write_text(decode_text_with_encoding(chart_bytes)[0], encoding='utf-8')
        try:
            result = subprocess.run(['dotnet', str(self.dll), str(chart), str(slot), str(raw)],
                                    capture_output=True, encoding='utf-8', errors='replace',
                                    timeout=30, creationflags=_FLAGS)
        except subprocess.TimeoutExpired as error:
            raise ValueError('open-mdp 解析谱面超时') from error
        if result.returncode == 2 or (result.returncode == 0 and not raw.is_file()):
            raise ValueError('open-mdp 无法完整解析这个难度槽')
        if result.returncode:
            raise RuntimeError(f'open-mdp 解析器运行失败：{result.stderr or result.stdout}')
        try:
            self.worker.stdin.write(json.dumps(str(raw)) + '\n')
            self.worker.stdin.flush()
        except BrokenPipeError as error:
            raise RuntimeError('open-mdp 估分进程已退出，请查看任务日志') from error
        response = self._response()
        if 'error' in response:
            raise ValueError(f'open-mdp 估分失败：{response["error"]}')
        value = float(response['level'])
        if not math.isfinite(value):
            raise ValueError('open-mdp 估分不是有限数字')
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pending = cache_path.with_suffix(f'.{uuid.uuid4().hex}.tmp')
        try:
            pending.write_text(json.dumps({'level': value}), encoding='utf-8')
            pending.replace(cache_path)
        finally:
            pending.unlink(missing_ok=True)
        self.computed += 1
        return value
