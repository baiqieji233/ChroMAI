from types import SimpleNamespace as NS
import pytest
from maichart.webui.startup_check import inspect_environment,install_command


def imports(*,missing=None,broken=None,audio_version='2.7.0+cu128'):
    torch=NS(__version__='2.7.0+cu128',version=NS(cuda='12.8'),zeros=lambda n:[0]*n)
    audio=NS(__version__=audio_version,functional=NS(resample=lambda *a:None))
    modules={'torch':torch,'torchaudio':audio,'transformers':NS(__version__='4.39.3',AutoModel=object,Wav2Vec2FeatureExtractor=object)}
    def load(name):
        if name==missing:raise ModuleNotFoundError('No module named '+name)
        if name==broken:raise OSError('DLL load failed')
        return modules.get(name,NS())
    return load


@pytest.mark.parametrize('module',['torchaudio','transformers','huggingface_hub'])
def test_music_missing_dependencies_fail_early(module):
    report=inspect_environment(importer=imports(missing=module))
    assert not report['ok'] and module in ' '.join(report['errors'])


def test_native_dll_mismatch_is_reported_and_repair_preserves_torch():
    report=inspect_environment(importer=imports(broken='torchaudio'))
    assert not report['ok'] and 'DLL load failed' in ' '.join(report['errors'])
    command=install_command(report)
    assert 'torch==2.7.0+cu128' in command and 'torchaudio==2.7.0+cu128' in command
    assert 'https://download.pytorch.org/whl/cu128' in command


@pytest.mark.parametrize('version',['2.6.0+cu128','2.7.0+cpu'])
def test_version_or_cuda_build_mismatch(version):
    assert not inspect_environment(importer=imports(audio_version=version))['ok']


def test_valid_environment_and_explicit_basic_profile():
    assert inspect_environment(importer=imports())['ok']
    assert inspect_environment('basic',importer=imports(missing='torchaudio'))['ok']


def test_cpu_build_is_valid_for_cpu_but_not_explicit_cuda(monkeypatch):
    from maichart.webui.startup_check import require_cuda
    cpu=NS(version=NS(cuda=None))
    with pytest.raises(RuntimeError,match='CPU-only'):require_cuda(cpu)
    with pytest.raises(RuntimeError,match='NVIDIA driver'):
        require_cuda(NS(version=NS(cuda='12.8'),cuda=NS(is_available=lambda:False)))


def test_generation_rejects_cpu_build_before_checkpoint_or_audio(monkeypatch):
    from maichart.composer_v2 import generate
    monkeypatch.setattr(generate.torch.version,'cuda',None)
    with pytest.raises(RuntimeError,match='CPU-only'):
        generate.generate(NS(device='cuda'))


def test_server_does_not_bind_port_when_dependencies_fail(monkeypatch):
    from maichart.webui import server,startup_check
    monkeypatch.setattr(startup_check,'main',lambda _:1)
    monkeypatch.setattr(server,'ThreadingHTTPServer',lambda *a:pytest.fail('port opened before dependency check'))
    assert server.main(['--port','17864'])==1
