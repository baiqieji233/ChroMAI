"""Check the actual WebUI interpreter before accepting expensive tasks."""
import argparse
import importlib
import os
import shlex
import subprocess
import sys
import re


def require_cuda(torch):
    if getattr(torch.version,'cuda',None) is None:
        raise RuntimeError(f'CPU-only PyTorch in {sys.executable}; CUDA was requested. Select CPU/auto or install a CUDA-enabled PyTorch and matching torchaudio.')
    if not torch.cuda.is_available():
        raise RuntimeError(f'CUDA PyTorch is installed in {sys.executable}, but no CUDA device is available. Check the NVIDIA driver and CUDA_VISIBLE_DEVICES; reinstalling CPU packages will not fix this.')
    try:
        for device in range(torch.cuda.device_count()):
            x=torch.ones((8,8),device=f'cuda:{device}')
            if (x@x).sum().item()!=512:raise RuntimeError('CUDA matrix result mismatch')
            torch.cuda.synchronize(device)
    except Exception as exc:
        raise RuntimeError(f'CUDA kernel check failed in {sys.executable}: {exc}') from exc


def inspect_environment(profile='composer-v2', importer=importlib.import_module, *, cuda_required=False):
    names=['numpy','scipy','soundfile','librosa','torch']
    if profile=='composer-v2':names+=['torchaudio','transformers','huggingface_hub']
    modules={};errors=[]
    for name in names:
        try:modules[name]=importer(name)
        except Exception as exc:errors.append(f'{name}: {type(exc).__name__}: {exc}')
    torch=modules.get('torch');audio=modules.get('torchaudio')
    version=str(torch.__version__) if torch is not None else None
    gpu_ok=False
    if cuda_required and torch is not None:
        try:require_cuda(torch);gpu_ok=True
        except RuntimeError as exc:errors.append(str(exc))
    if torch is not None and audio is not None:
        tv=version.split('+')[0];av=str(audio.__version__).split('+')[0]
        try:abi=int(av.split('.')[0])==2 and int(av.split('.')[1])>=11 and tuple(map(int,tv.split('.')[:2]))>=tuple(map(int,av.split('.')[:2]))
        except ValueError:abi=False
        if tv!=av and not abi:errors.append(f'torchaudio {audio.__version__} does not match torch {version}')
        tb=version.partition('+')[2];ab=str(audio.__version__).partition('+')[2]
        if tb and ab and tb!=ab and all(x=='cpu' or x.startswith(('cu','rocm')) for x in (tb,ab)):
            errors.append(f'torchaudio build {ab} does not match torch build {tb}')
        try:
            # Exercises the exact API used by audio preparation; no GPU/model download.
            audio.functional.resample(torch.zeros(480),48000,24000)
        except Exception as exc:errors.append(f'torchaudio resample: {type(exc).__name__}: {exc}')
    if 'transformers' in modules:
        try:
            v=tuple(map(int,modules['transformers'].__version__.split('.')[:2]))
            if not (4,38)<=v<(4,40):errors.append('MERT requires transformers>=4.38,<4.40 (project dependency contract)')
            getattr(modules['transformers'],'AutoModel');getattr(modules['transformers'],'Wav2Vec2FeatureExtractor')
        except Exception as exc:errors.append(f'MERT imports: {type(exc).__name__}: {exc}')
    if profile=='composer-v2' and hasattr(modules.get('librosa'),'__version__'):
        match=re.match(r'(\d+)\.(\d+)\.(\d+)',modules['librosa'].__version__)
        if match and tuple(map(int,match.groups()))<(0,10,2):
            errors.append('Dynamic tempo tracking requires librosa>=0.10.2')
    return dict(ok=not errors,python=sys.executable,profile=profile,torch=version,
        torchaudio=str(audio.__version__) if audio is not None else None,
        cuda=getattr(getattr(torch,'version',None),'cuda',None),
        xpu_available=bool(torch is not None and hasattr(torch,'xpu') and torch.xpu.is_available()),
        cuda_required=cuda_required,gpu_ok=gpu_ok,errors=errors)


def install_command(report):
    extras='composer-v2-music' if report['profile']=='composer-v2' else 'audio,training'
    command=[sys.executable,'-m','pip','install','-e',f'.[{extras}]']
    if report['torch']:
        # Preserve the installed Torch build; install its matching audio wheel.
        command.append('torch=='+report['torch'])
        if report['profile']=='composer-v2':
            command.append('torchaudio=='+report['torch'])
        tag='cu'+str(report['cuda']).replace('.','') if report['cuda'] else 'cpu'
        index=os.environ.get('MAICHART_TORCH_INDEX_URL')
        if index or ('+' in report['torch'] and ('cu' in report['torch'].split('+')[1] or report['torch'].endswith('+cpu'))):
            command+=['--extra-index-url',index or 'https://download.pytorch.org/whl/'+tag]
    elif os.environ.get('MAICHART_TORCH_INDEX_URL'):
        command+=['--extra-index-url',os.environ['MAICHART_TORCH_INDEX_URL']]
    return command


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=('basic','composer-v2'),default='composer-v2')
    parser.add_argument('--install',action='store_true')
    parser.add_argument('--require-cuda',action='store_true')
    parser.add_argument('--auto-device',action='store_true',help='Use detected NVIDIA hardware unless CPU was explicitly selected')
    args=parser.parse_args(argv)
    from .gpu_setup import detect_nvidia,device_hidden,select_wheels,gpu_install_command
    hardware=detect_nvidia() if args.auto_device or args.require_cuda else None
    required=args.require_cuda or bool(args.auto_device and hardware and not device_hidden())
    report=inspect_environment(args.profile,cuda_required=required)
    if hardware:print('NVIDIA hardware:',hardware,flush=True)
    print('Python:',report['python'],flush=True)
    print('PyTorch:',report['torch'],'TorchAudio:',report['torchaudio'],'CUDA build:',report['cuda'],flush=True)
    if report.get('xpu_available'):
        print('Intel XPU: available for Composer v2 training',flush=True)
    elif report['torch'] and not report['cuda']:
        print('CPU-only PyTorch: choose CPU/auto in WebUI. GPU setup: START_WEBUI.bat --cuda',flush=True)
    from maichart.composer_v2.acceleration import openvino_devices
    devices=openvino_devices()
    if devices:print('OpenVINO inference devices:',', '.join(devices),flush=True)
    if report['ok']:
        print('Startup dependencies: OK',flush=True);return 0
    for error in report['errors']:print('Dependency error:',error,file=sys.stderr,flush=True)
    if any('dll' in error.lower() or 'winerror 126' in error.lower() for error in report['errors']):
        print('A native DLL could not be loaded. On Windows, check the Visual C++ runtime and that PyTorch and TorchAudio use compatible builds.',file=sys.stderr,flush=True)
    switching=required and not report['gpu_ok']
    if switching:
        if device_hidden() or hardware is None:
            print('GPU setup stopped: no visible NVIDIA device/driver could be detected. No packages changed.',file=sys.stderr)
            return 1
        if report['cuda']:
            runtime=tuple(map(int,str(report['cuda']).split('.')[:2]))
            architecture_error=any(word in ' '.join(report['errors']).lower()
                for word in ('no kernel image','not compatible','invalid device function'))
            if runtime<=tuple(hardware['cuda_max']) and not architecture_error:
                print('CUDA runtime/device check failed. Preserve installed packages; resolve driver, visibility or resource errors above before retrying.',file=sys.stderr)
                return 1
        try:
            selection=select_wheels(hardware,report['torch'])
            command=gpu_install_command(selection,sys.executable)
            print('Selected official CUDA wheels:',selection,flush=True)
        except RuntimeError as exc:
            print(str(exc),file=sys.stderr);return 1
    else:command=install_command(report)
    repair = ('& ' + ' '.join("'"+arg.replace("'","''")+"'" for arg in command)
              if os.name=='nt' else shlex.join(command))
    print('Run in the project directory with this interpreter:',repair,flush=True)
    if args.install:
        result=subprocess.run(command,check=False)
        if result.returncode:return result.returncode
        if switching:
            extras='composer-v2-music' if args.profile=='composer-v2' else 'audio,training'
            result=subprocess.run([sys.executable,'-m','pip','install','-e',f'.[{extras}]'],check=False)
            if result.returncode:return result.returncode
        # Recheck in a fresh interpreter after pip; imported DLLs cannot be unloaded safely.
        follow=[sys.executable,'-m',__spec__.name,'--profile',args.profile]
        if required:follow.append('--require-cuda')
        return subprocess.call(follow)
    return 1


if __name__=='__main__':raise SystemExit(main())
