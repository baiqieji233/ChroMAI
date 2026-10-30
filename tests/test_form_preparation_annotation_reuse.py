import hashlib
import json
import pytest
import torch
from maichart.form import data
from maichart.form.preparation_cache import PreparationCache,_json_bytes
from test_form_preparation_cache import cached_source


def test_namespaces_are_not_preparation_inputs_but_labels_weights_and_regions_are():
    identity={'schema':'test','quality':{'label':'excellent','weight':1.4,'source_path':'first',
        'song_id':'42','chart_id':'old','regions':[{'start_sec':1,'end_sec':2,'weight':.5,'reason':'old'}]}}
    changed={**identity,'quality':{**identity['quality'],'source_path':'second','song_id':'dataset::42',
        'chart_id':'new','source_dataset':{'id':'dataset'},'regions':[{'start_sec':1.,'end_sec':2.,'weight':.5,'reason':'new'}]}}
    assert PreparationCache._normalized_identity(identity)==PreparationCache._normalized_identity(changed)
    for override in ({'label':'negative'},{'weight':1.5},{'regions':[]},
                     {'regions':[{'start_sec':1,'end_sec':2,'weight':0}]}):
        different={**changed,'quality':{**changed['quality'],**override}}
        assert PreparationCache._normalized_identity(identity)!=PreparationCache._normalized_identity(different)


def test_old_namespace_cache_reused_and_rebound_without_repreparing(cached_source,monkeypatch):
    build,chart,audio,calls=cached_source
    from maichart.training.quality_annotations import SCHEMA,chart_sha256
    annotation_file=chart.parent.parent/'quality_annotations.json'
    record={'difficulty_index':5,'label':'excellent','weight':1.4,'source_path':str(chart),
        'audio_path':str(audio),'chart_sha256':chart_sha256(chart,5),'song_id':'fixture','chart_id':'original'}
    annotation_file.write_text(json.dumps({'schema':SCHEMA,'records':[record]}),encoding='utf-8')
    def old_identity(identity):
        value=dict(identity);[value.pop(k,None) for k in ('code_sha256','source','audio')]
        value.setdefault('semantics','song_form_preparation_exact_geometry_v1')
        return _json_bytes(value)
    with monkeypatch.context() as legacy:
        legacy.setattr(PreparationCache,'_normalized_identity',staticmethod(old_identity))
        cold=build()
    assert cold.preparation_cache_stats['rebuilt']==1
    changed={**record,'song_id':'dataset::fixture','chart_id':'relabelled',
        'source_dataset':{'id':'dataset','name':'copied dataset'},'reason':'display metadata changed'}
    annotation_file.write_text(json.dumps({'schema':SCHEMA,'records':[changed]}),encoding='utf-8')
    def unexpected(*args,**kwargs):raise AssertionError('Equivalent annotations must reuse prepared artifacts')
    import inspect
    unexpected.__signature__=inspect.signature(data.organize_song)
    monkeypatch.setattr(data,'organize_song',unexpected)
    monkeypatch.setattr(data,'parse_event_chart',unexpected)
    calls.clear();warm=build()
    assert warm.preparation_cache_stats['hits']==1 and not calls
    assert warm.songs[0][7]['chart_id']=='relabelled'
    assert warm.audit['charts'][0]['quality']['source_dataset']==changed['source_dataset']
    assert warm.items==cold.items
    assert warm.songs[0][1]==cold.songs[0][1]
    assert warm.songs[0][3]==cold.songs[0][3]==1.4
    torch.testing.assert_close(warm.songs[0][5].to_vector(),cold.songs[0][5].to_vector(),rtol=0,atol=0)
    assert hashlib.sha256(warm.songs[0][0].relations.descriptors.tobytes()).digest()==hashlib.sha256(cold.songs[0][0].relations.descriptors.tobytes()).digest()
    calls.clear();assert build().preparation_cache_stats['hits']==1 and not calls
