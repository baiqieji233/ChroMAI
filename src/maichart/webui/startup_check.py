"""Check the actual WebUI interpreter before accepting expensive tasks."""
import argparse
import importlib
import os
import shlex
import subprocess
import sys


def require_cuda(torch):
    if getattr(torch.version,'cuda',None) is None:
        raise RuntimeError(f'CPU-only PyTorch in {sys.executable}; CUDA was requested. Select CPU/auto or install a CUDA-enabled PyTorch and matching torchaudio.')
    if not torch.cuda.is_available():
        raise RuntimeError(f'CUDA PyTorch is installed in {sys.executable}, but no CUDA device is available. Check the NVIDIA driver and CUDA_VISIBLE_DEVICES; reinstalling CPU packages will not fix this.')


def inspect_environment(profile='composer-v2', importer=importlib.import_module, *, cuda_required=False):
    names=['numpy','scipy','soundfile','librosa','torch']
    if profile=='composer-v2':names+=['torchaudio','transformers','huggingface_hub']
    modules={};errors=[]
    for name in names:
        try:modules[name]=importer(name)
        except Exception as exc:errors.append(f'{name}: {type(exc).__name__}: {exc}')
    torch=modules.get('torch');audio=modules.get('torchaudio')
    version=str(torch.__version__) if torch is not None else None
    if cuda_required and torch is not None:
        try:require_cuda(torch)
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
    return dict(ok=not errors,python=sys.executable,profile=profile,torch=version,
        torchaudio=str(audio.__version__) if audio is not None else None,
        cuda=getattr(getattr(torch,'version',None),'cuda',None),cuda_required=cuda_required,errors=errors)


def install_command(report):
    if report.get('cuda_required') and not report['cuda']:
        return [sys.executable,'-m','pip','install','--upgrade','torch','torchaudio',
                '--index-url',os.environ.get('MAICHART_TORCH_INDEX_URL') or 'https://download.pytorch.org/whl/cu128']
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
    args=parser.parse_args(argv)
    report=inspect_environment(args.profile,cuda_required=args.require_cuda)
    print('Python:',report['python'],flush=True)
    print('PyTorch:',report['torch'],'TorchAudio:',report['torchaudio'],'CUDA build:',report['cuda'],flush=True)
    if report['torch'] and not report['cuda']:
        print('CPU-only PyTorch: choose CPU/auto in WebUI. GPU setup: START_WEBUI.bat --cuda',flush=True)
    if report['ok']:
        print('Startup dependencies: OK',flush=True);return 0
    for error in report['errors']:print('Dependency error:',error,file=sys.stderr,flush=True)
    command=install_command(report)
    repair = ('& ' + ' '.join("'"+arg.replace("'","''")+"'" for arg in command)
              if os.name=='nt' else shlex.join(command))
    print('Run in the project directory with this interpreter:',repair,flush=True)
    if args.install:
        switching=args.require_cuda and not report['cuda']
        if switching and not os.environ.get('MAICHART_TORCH_INDEX_URL'):
            print('Set MAICHART_TORCH_INDEX_URL to the official CUDA wheel index compatible with this device before automatic GPU installation. The command above uses cu128 as an example.',file=sys.stderr)
            return 1
        if args.require_cuda and report['cuda'] and any('no CUDA device' in e for e in report['errors']):return 1
        result=subprocess.run(command,check=False)
        if result.returncode:return result.returncode
        if switching:
            extras='composer-v2-music' if args.profile=='composer-v2' else 'audio,training'
            result=subprocess.run([sys.executable,'-m','pip','install','-e',f'.[{extras}]'],check=False)
            if result.returncode:return result.returncode
        # Recheck in a fresh interpreter after pip; imported DLLs cannot be unloaded safely.
        follow=[sys.executable,'-m',__spec__.name,'--profile',args.profile]
        if args.require_cuda:follow.append('--require-cuda')
        return subprocess.call(follow)
    return 1


if __name__=='__main__':raise SystemExit(main())
