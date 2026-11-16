"""Select published official CUDA wheels before modifying an environment."""
import csv
import io
import os
import re
import shutil
import subprocess
from urllib.parse import unquote, urlparse
from urllib.request import urlopen, Request
from urllib.error import HTTPError


def detect_nvidia(run=subprocess.run):
    executable=shutil.which('nvidia-smi')
    if not executable:return None
    try:
        result=run([executable],capture_output=True,text=True,timeout=15)
        match=re.search(r'CUDA Version:\s*(\d+)\.(\d+)',result.stdout)
        if result.returncode or not match:return None
        query=run([executable,'--query-gpu=name,driver_version,compute_cap','--format=csv,noheader'],
                  capture_output=True,text=True,timeout=15)
        capabilities=[];devices=[]
        if query.returncode==0:
            for row in csv.reader(io.StringIO(query.stdout)):
                if len(row)!=3:continue
                devices.append(dict(name=row[0].strip(),driver=row[1].strip()))
                try:capabilities.append(float(row[2]))
                except ValueError:pass
        return dict(cuda_max=tuple(map(int,match.groups())),devices=devices,capabilities=capabilities)
    except (OSError,subprocess.TimeoutExpired):return None


def cuda_candidates(hardware):
    maximum=tuple(hardware['cuda_max']);caps=hardware['capabilities']
    # Stay within the driver's advertised runtime, including PTX support.
    options=[(12,8),(12,6),(12,4),(12,1),(11,8),(13,0)]
    if caps and max(caps)>=10:options=[v for v in options if v>=(12,8)]
    if caps and min(caps)<7.5:options=[v for v in options if v<=(12,6)]
    return ['cu'+str(a)+str(b) for a,b in options if (a,b)<=maximum]


def fetch_index(url):
    # No third-party package index, package execution, or large wheel download.
    with urlopen(Request(url,headers={'User-Agent':'maichart-gpu-setup/1'}),timeout=20) as response:
        return response.read(16*1024*1024).decode('utf-8')


def _import_packaging():
    """Import packaging helpers, falling back to pip's vendored copy if needed."""
    try:
        from packaging.tags import Tag, sys_tags
        from packaging.utils import parse_wheel_filename
        from packaging.version import Version
    except ImportError:
        from pip._vendor.packaging.tags import Tag, sys_tags
        from pip._vendor.packaging.utils import parse_wheel_filename
        from pip._vendor.packaging.version import Version
    return Tag, sys_tags, parse_wheel_filename, Version


def wheel_versions(html, package, cuda, tags):
    Tag, _, parse_wheel_filename, _ = _import_packaging()
    tag_set={tag if isinstance(tag,Tag) else Tag(tag.interpreter,tag.abi,tag.platform) for tag in tags}
    versions={}
    for link in re.findall(r'href=[\"\']([^\"\']+)',html):
        name=unquote(urlparse(link).path.rsplit('/',1)[-1])
        if not name.endswith('.whl'):continue
        try:dist,version,_,supported=parse_wheel_filename(name)
        except ValueError:continue
        if str(dist)!=package or version.is_prerelease or not (supported & tag_set):continue
        if version.local!=cuda:continue
        versions[version.base_version]=version
    return versions


def select_wheels(hardware, installed=None, fetch=fetch_index, tags=None):
    _, sys_tags, _, Version = _import_packaging()
    if not hardware.get('capabilities'):
        raise RuntimeError('NVIDIA GPU compute capability could not be detected. No packages changed; check nvidia-smi/driver installation.')
    tags=set(sys_tags()) if tags is None else set(tags)
    failures=[]
    for cuda in cuda_candidates(hardware):
        index='https://download.pytorch.org/whl/'+cuda
        try:
            torch=wheel_versions(fetch(index+'/torch/'),'torch',cuda,tags)
            audio=wheel_versions(fetch(index+'/torchaudio/'),'torchaudio',cuda,tags)
        except (OSError,ValueError) as exc:
            if isinstance(exc,HTTPError) and exc.code==404:
                failures.append(f'{cuda}: {exc}');continue
            raise RuntimeError(f'Official wheel index unavailable: {exc}. No packages changed; check the network and retry.') from exc
        common=[v for v in torch.keys() & audio.keys() if Version(v)>=Version('2.4')]
        caps=hardware['capabilities']
        if caps and min(caps)<7.5:
            common=[v for v in common if Version(v)<Version('2.7')]
        if not common:continue
        previous=installed.split('+')[0] if installed else None
        preferred=previous if previous in common else '2.7.1' if '2.7.1' in common else max(common,key=Version)
        return dict(torch=str(torch[preferred]),torchaudio=str(audio[preferred]),index=index,
                    hardware=hardware)
    raise RuntimeError('No official matching torch/torchaudio wheels for this Python, OS, GPU and driver. '
                       'No packages changed. Check Python/driver support and network. '+ '; '.join(failures))


def gpu_install_command(selection,python):
    return [python,'-m','pip','install','--upgrade','torch=='+selection['torch'],
            'torchaudio=='+selection['torchaudio'],'--index-url',selection['index']]


def device_hidden():
    return os.environ.get('CUDA_VISIBLE_DEVICES') in ('','-1')
