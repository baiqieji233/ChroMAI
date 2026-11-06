import json
from pathlib import Path
from maichart.composer_v2.train import _hash, _same_data_identity

def make(source):
    x={'manifest_sha256':'manifest','charts':[{'source':source,'chart_sha256':'chart','quality':{'weight':1.4}}], 'audio':[{'identity':{'audio_sha256':'audio'}}], 'codec_sha256':'codec','windows':2}
    return {'train':dict(x,sha256=_hash(x)), 'validation':None}

def test_v2_resume_mapping_and_integrity(tmp_path,monkeypatch):
    target=tmp_path/'data';target.mkdir()
    mapping=tmp_path/'map.json';mapping.write_text(json.dumps({'E:/dataset':str(target)}))
    monkeypatch.setenv('MAICHART_PATH_MAP',str(mapping))
    old=make('E:/dataset/chart.txt');new=make(str(target/'chart.txt'))
    assert _same_data_identity(old,new)
    changed=make(str(target/'chart.txt'));changed['train']['charts'][0]['chart_sha256']='changed'
    assert not _same_data_identity(old,changed)
    changed=make(str(target/'chart.txt'));changed['train']['windows']=3;changed['train']['sha256']=_hash({k:v for k,v in changed['train'].items() if k!='sha256'})
    assert not _same_data_identity(old,changed)
