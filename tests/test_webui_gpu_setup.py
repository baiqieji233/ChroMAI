from pip._vendor.packaging.tags import Tag
import pytest
from maichart.webui.gpu_setup import cuda_candidates,select_wheels,gpu_install_command


def hardware(cuda=(12,8),caps=(8.6,)):
    return dict(cuda_max=cuda,capabilities=list(caps),devices=[dict(name='test GPU',driver='570')])


def index(package,versions,cuda='cu128',python='cp312',platform='win_amd64'):
    return '\n'.join(f'<a href="{package}-{v}%2B{cuda}-{python}-{python}-{platform}.whl">wheel</a>' for v in versions)


def test_pairs_are_exact_and_preserve_installed_release_when_available():
    def fetch(url):return index('torchaudio' if '/torchaudio/' in url else 'torch',['2.7.0','2.7.1','2.9.0'])
    choice=select_wheels(hardware(), '2.7.0+cpu',fetch,{Tag('cp312','cp312','win_amd64')})
    assert choice['torch']==choice['torchaudio']=='2.7.0+cu128'
    assert 'torch==2.7.0+cu128' in gpu_install_command(choice,'python')


def test_unmatched_audio_and_wrong_python_wheels_cannot_be_selected():
    def fetch(url):
        audio='/torchaudio/' in url
        return index('torchaudio' if audio else 'torch',['2.7.1'] if audio else ['2.9.0'])
    with pytest.raises(RuntimeError,match='No packages changed'):
        select_wheels(hardware(),None,fetch,{Tag('cp312','cp312','win_amd64')})
    with pytest.raises(RuntimeError):
        select_wheels(hardware(),None,lambda u:index('torch',['2.7.1'],python='cp313'),{Tag('cp312','cp312','win_amd64')})


def test_old_driver_and_blackwell_and_legacy_gpu_constraints():
    assert cuda_candidates(hardware((12,1)))==['cu121','cu118']
    assert cuda_candidates(hardware((12,6),(12.0,)))==[]
    assert 'cu128' not in cuda_candidates(hardware(caps=(6.1,)))
    assert 'cu128' not in cuda_candidates(hardware(caps=(6.1,12.0)))


def test_network_failure_does_not_produce_an_install_plan():
    def failed(url):raise OSError('offline')
    with pytest.raises(RuntimeError,match='offline'):select_wheels(hardware(),fetch=failed)


def test_working_environment_never_selects_or_installs_packages(monkeypatch):
    from maichart.webui import gpu_setup,startup_check
    monkeypatch.setattr(gpu_setup,'detect_nvidia',lambda:hardware())
    monkeypatch.setattr(startup_check,'inspect_environment',lambda *a,**kw:dict(ok=True,python='python',torch='2.7.0+cu128',torchaudio='2.7.0+cu128',cuda='12.8',gpu_ok=True))
    monkeypatch.setattr(gpu_setup,'select_wheels',lambda *a,**kw:pytest.fail('working environment queried wheels'))
    monkeypatch.setattr(startup_check.subprocess,'run',lambda *a,**kw:pytest.fail('working environment modified'))
    assert startup_check.main(['--auto-device','--install'])==0


def test_resource_failure_preserves_existing_packages(monkeypatch):
    from maichart.webui import gpu_setup,startup_check
    monkeypatch.setattr(gpu_setup,'detect_nvidia',lambda:hardware())
    monkeypatch.setattr(gpu_setup,'device_hidden',lambda:False)
    monkeypatch.setattr(startup_check,'inspect_environment',lambda *a,**kw:dict(ok=False,python='python',torch='2.7.0+cu128',torchaudio='2.7.0+cu128',cuda='12.8',gpu_ok=False,errors=['CUDA out of memory']))
    monkeypatch.setattr(gpu_setup,'select_wheels',lambda *a,**kw:pytest.fail('resource error triggered replacement'))
    assert startup_check.main(['--auto-device','--install'])==1


def test_cpu_to_cuda_installs_pinned_pair_then_rechecks_in_fresh_process(monkeypatch):
    from types import SimpleNamespace
    from maichart.webui import gpu_setup,startup_check
    monkeypatch.setattr(gpu_setup,'detect_nvidia',lambda:hardware())
    monkeypatch.setattr(gpu_setup,'device_hidden',lambda:False)
    monkeypatch.setattr(startup_check,'inspect_environment',lambda *a,**kw:dict(ok=False,python='python',profile='composer-v2',torch='2.7.0+cpu',torchaudio='2.7.0+cpu',cuda=None,gpu_ok=False,errors=['CPU-only']))
    monkeypatch.setattr(gpu_setup,'select_wheels',lambda *a:dict(torch='2.7.0+cu128',torchaudio='2.7.0+cu128',index='https://download.pytorch.org/whl/cu128'))
    calls=[]
    monkeypatch.setattr(startup_check.subprocess,'run',lambda args,**kw:calls.append(args) or SimpleNamespace(returncode=0))
    monkeypatch.setattr(startup_check.subprocess,'call',lambda args:calls.append(args) or 0)
    assert startup_check.main(['--auto-device','--install'])==0
    assert 'torch==2.7.0+cu128' in calls[0] and 'torchaudio==2.7.0+cu128' in calls[0]
    assert '.[composer-v2-music]' in calls[1]
    assert '--require-cuda' in calls[2] and '--install' not in calls[2]
