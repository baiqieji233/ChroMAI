from collections import OrderedDict
from datetime import datetime
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import threading
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

import pytest
from maichart.webui import chart_preview as preview, preview_assets as assets


@pytest.fixture(autouse=True)
def catalog(monkeypatch):
    monkeypatch.setattr(preview,'_resources',OrderedDict())


def record(tmp_path,audio=True):
    chart=tmp_path/'maidata.txt'
    chart.write_text('&title=测试谱面\n&first=0\n&inote_5=(120){4}1,2,3,E',encoding='utf-8')
    music=tmp_path/'original.wav'
    if audio:music.write_bytes(b'original task audio')
    return dict(task_id='aabbcc123456',kind='composer_v2_generate',status='completed',
        argv=['python','-m','maichart.composer_v2.generate','--output',str(chart),'--audio',str(music)],
        result=dict(paths=[str(chart)],effective_settings=dict(audio=str(tmp_path/'other.wav'))))


def test_exact_task_audio_and_chart_are_used_not_editable_settings(tmp_path):
    task=record(tmp_path)
    value=preview.describe(task['task_id'],project=tmp_path,log_root=tmp_path,task=task)
    chart=value['charts'][0]
    token=parse_qs(urlparse(chart['audio_url']).query)['id'][0]
    assert preview.resource_path(token).name=='original.wav'
    assert chart['title']=='测试谱面'
    again=preview.describe(task['task_id'],project=tmp_path,log_root=tmp_path,task=task)
    assert again['revision']==value['revision']


def test_no_audio_still_allows_visual_preview(tmp_path):
    task=record(tmp_path,audio=False)
    value=preview.describe(task['task_id'],project=tmp_path,log_root=tmp_path,task=task)
    assert value['charts'][0]['audio_url'] is None
    assert '静音' in value['message']


def test_repaired_chart_from_an_older_task_is_not_paired_with_new_audio(tmp_path):
    import os,time
    task=record(tmp_path)
    task['started']=task['ended']=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    stale=tmp_path/'maidata.repaired.txt'
    stale.write_text('&title=old song\n&inote_5=(120)1,E',encoding='utf-8')
    os.utime(stale,(time.time()-3600,time.time()-3600))
    value=preview.describe(task['task_id'],project=tmp_path,log_root=tmp_path,task=task)
    assert len(value['charts'])==1 and value['charts'][0]['title']=='测试谱面'


def test_modified_file_invalidates_old_resource_instead_of_silently_switching(tmp_path):
    path=tmp_path/'audio.wav';path.write_bytes(b'old')
    url=preview._resource(path);token=parse_qs(urlparse(url).query)['id'][0]
    path.write_bytes(b'a different song')
    with pytest.raises(FileNotFoundError,match='改变'):preview.resource_path(token)


def test_saved_task_recovers_after_server_restart_and_rejects_overwritten_chart(tmp_path):
    task=record(tmp_path)
    task['ended']=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    (tmp_path/('20261105_composer_v2_generate_'+task['task_id']+'.meta.json')).write_text(json.dumps(task),encoding='utf-8')
    assert preview.latest(project=tmp_path,log_root=tmp_path)['task_id']==task['task_id']
    import os,time
    os.utime(tmp_path/'maidata.txt',(time.time()+30,time.time()+30))
    assert preview.latest(project=tmp_path,log_root=tmp_path) is None


def test_running_task_cannot_offer_partial_chart_and_arbitrary_paths_are_not_resources(tmp_path):
    task=record(tmp_path);task['status']='running'
    with pytest.raises(ValueError):preview.describe(task['task_id'],project=tmp_path,log_root=tmp_path,task=task)
    with pytest.raises(FileNotFoundError):preview.resource_path(str(tmp_path/'maidata.txt'))
    with pytest.raises(FileNotFoundError):preview.describe('../secret',project=tmp_path,log_root=tmp_path)


@pytest.mark.parametrize('name',['../secret','sdk/../../secret','a'*20+'/renderer/../../secret','C:/secret','a'*20+'\\renderer\\secret'])
def test_player_asset_route_blocks_escape(tmp_path,name):
    with pytest.raises(ValueError):assets.asset_path(tmp_path,name)


@pytest.fixture
def http(tmp_path,monkeypatch):
    from maichart.webui import server
    from http.server import ThreadingHTTPServer
    monkeypatch.setattr(server,'WEBUI_ROOT',tmp_path)
    monkeypatch.setattr(server,'LOG_ROOT',tmp_path/'logs')
    instance=ThreadingHTTPServer(('127.0.0.1',0),server.WebUIHandler)
    thread=threading.Thread(target=lambda:instance.serve_forever(poll_interval=.01),daemon=True)
    thread.start()
    yield f'http://127.0.0.1:{instance.server_port}'
    instance.shutdown();instance.server_close();thread.join(timeout=2)


def test_real_http_audio_ranges_head_and_opaque_resource_ids(tmp_path,http):
    audio=tmp_path/'song.wav';audio.write_bytes(bytes(range(100)))
    url=http+preview._resource(audio)
    with urlopen(Request(url,headers={'Range':'bytes=10-19'})) as response:
        assert response.status==206 and response.read()==bytes(range(10,20))
        assert response.headers['Content-Range']=='bytes 10-19/100'
    with urlopen(Request(url,method='HEAD')) as response:
        assert response.headers['Content-Length']=='100' and response.read()==b''
    with pytest.raises(HTTPError) as error:urlopen(Request(url,headers={'Range':'bytes=1000-'}))
    assert error.value.code==416
    with pytest.raises(HTTPError) as error:urlopen(http+'/api/preview/resource?id='+str(audio))
    assert error.value.code==404


def test_real_http_wasm_mime_and_transport_compression(tmp_path,http):
    import gzip
    folder=tmp_path/'preview_runtime/miaview/releases'/('a'*20)/'renderer';folder.mkdir(parents=True)
    wasm=b'\x00asm'+b'payload'*30
    (folder/'miacode-preview.wasm').write_bytes(wasm)
    (folder/'miacode-preview.wasm.gz').write_bytes(gzip.compress(wasm))
    with urlopen(Request(http+'/miaview/'+('a'*20)+'/renderer/miacode-preview.wasm',headers={'Accept-Encoding':'gzip'})) as response:
        assert response.headers['Content-Type']=='application/wasm'
        assert response.headers['Content-Encoding']=='gzip'
        assert gzip.decompress(response.read())==wasm
        assert 'immutable' in response.headers['Cache-Control']


def test_installer_checks_hashes_and_does_not_replace_working_release_on_failure(tmp_path,monkeypatch):
    import gzip
    payload={'renderer/index.html':b'player','renderer/miacode-preview.js':b'js',
        'renderer/miacode-preview.wasm':b'\x00asm example','renderer/build-identity.json':b'{"version":"test-version"}',
        'renderer/LICENSE':b'MIT','renderer/LICENSE_SCOPE.md':b'asset scope','renderer/THIRD_PARTY_NOTICES.md':b'notices'}
    payload['renderer/miacode-preview.wasm.gz']=gzip.compress(payload['renderer/miacode-preview.wasm'])
    specs=[dict(path=name.removeprefix('renderer/'),byteSize=len(data),sha256=hashlib.sha256(data).hexdigest()) for name,data in payload.items()]
    payload['renderer/artifact-manifest.json']=json.dumps({'files':specs}).encode()
    payload.update({'sdk/'+name:b'// same complete SDK' for name in assets.SDK_FILES})
    monkeypatch.setattr(assets.urllib.request,'urlopen',lambda url,**kwargs:BytesIO(payload[url.removeprefix(assets.UPSTREAM)]))
    saved=assets.install(tmp_path)
    assert assets.status(tmp_path)['available']
    before=(tmp_path/'current.json').read_bytes()
    payload['renderer/miacode-preview.js']=b'corrupt source'
    with pytest.raises(ValueError):assets.install(tmp_path)
    assert (tmp_path/'current.json').read_bytes()==before
    assert assets.status(tmp_path)['release']==saved['release']
    assert not list((tmp_path/'releases').glob('.install-*'))


def test_preview_is_above_generation_inputs_and_collapsed_by_default():
    root=Path(__file__).resolve().parents[1]
    html=(root/'src/maichart/webui/static/index.html').read_text(encoding='utf-8')
    tag=re.search(r'<details id="generationPreviewPanel"([^>]*)>',html)
    assert tag and not re.search(r'\bopen\b',tag.group(1))
    assert html.index('class="panel generation-panel"') < tag.start() < html.index('id="audioPath"')


def test_install_endpoint_rejects_cross_site_requests_and_returns_immediately(http,monkeypatch):
    called=[]
    monkeypatch.setattr(assets,'start_install',lambda root:called.append(root))
    with pytest.raises(HTTPError) as error:
        urlopen(Request(http+'/api/preview/install',data=b'{}',headers={'Content-Type':'application/json','Origin':'https://unrelated.invalid'}))
    assert error.value.code==400 and not called
    with urlopen(Request(http+'/api/preview/install',data=b'{}',headers={'Content-Type':'application/json','Origin':http})) as response:
        assert response.status==200
    assert len(called)==1


def test_install_runs_in_background_and_duplicate_clicks_share_the_download(tmp_path,monkeypatch):
    import time
    entered=threading.Event();finish=threading.Event();completed=threading.Event()
    calls=[]
    def install(root,progress):
        calls.append(root);entered.set()
        finish.wait(timeout=2)
        completed.set()
    monkeypatch.setattr(assets,'install',install)
    monkeypatch.setattr(assets,'_state',dict(installing=False,error=None,message=''))
    started=time.perf_counter();assets.start_install(tmp_path)
    assert time.perf_counter()-started < .5
    assert entered.wait(timeout=1)
    assets.start_install(tmp_path)
    assert len(calls)==1 and assets.status(tmp_path)['installing']
    finish.set();assert completed.wait(timeout=1)
    deadline=time.perf_counter()+1
    while assets.status(tmp_path)['installing'] and time.perf_counter()<deadline:time.sleep(.005)
    assert not assets.status(tmp_path)['installing']
