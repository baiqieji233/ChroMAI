import json
from fractions import Fraction
from types import SimpleNamespace

import torch

from maichart.composer.stream_data_identity import SCHEMA, dataset_content_identity


def _dataset(*, text='1', weight=1., condition=0., chart_weight=1.):
    step=SimpleNamespace(kind=2,start=Fraction(0),end=Fraction(1,2),
        start_second=0.,end_second=.25,text=text,weight=weight)
    return SimpleNamespace(streams=[[step]],conditions=[torch.tensor([condition])],
        chart_weights=[chart_weight],target_window_weights=[[1.]])


def _manifest(tmp_path, *, split_source=None):
    source=tmp_path/'source';source.mkdir()
    chart=source/'maidata.txt';chart.write_text('chart A',encoding='utf-8')
    audio=source/'track.wav';audio.write_bytes(b'audio A')
    payload={'source_root':str(source),'songs':[
        {'maidata_path':'maidata.txt','audio_path':'track.wav','difficulties':[{'difficulty_index':5}]},
        {'maidata_path':'maidata.txt','audio_path':'track.wav','difficulties':[{'difficulty_index':4}]}]}
    if split_source is not None:
        payload['split_metadata']={'source_manifest':str(split_source)}
    manifest=tmp_path/'train.json';manifest.write_text(json.dumps(payload),encoding='utf-8')
    return manifest,chart,audio


def test_source_contents_and_prepared_supervision_change_identity(tmp_path):
    manifest,chart,audio=_manifest(tmp_path)
    cache=tmp_path/'cache'
    base=dataset_content_identity(manifest,_dataset(),cache_dir=cache)
    assert base['schema']==SCHEMA and base['source_file_count']==2
    def identity(ds):return dataset_content_identity(manifest,ds,cache_dir=cache)['fingerprint']
    assert identity(_dataset(text='2'))!=base['fingerprint']
    assert identity(_dataset(weight=0.))!=base['fingerprint']
    assert identity(_dataset(condition=.5))!=base['fingerprint']
    assert identity(_dataset(chart_weight=.25))!=base['fingerprint']
    chart.write_text('chart B',encoding='utf-8')
    assert identity(_dataset())!=base['fingerprint']
    chart.write_text('chart A',encoding='utf-8')
    audio.write_bytes(b'audio B')
    assert identity(_dataset())!=base['fingerprint']


def test_split_quality_annotation_fallback_and_local_precedence(tmp_path):
    source=tmp_path/'origin';source.mkdir()
    source_manifest=source/'all.json';source_manifest.write_text('{}',encoding='utf-8')
    source_quality=source/'quality_annotations.json'
    source_quality.write_text('{"weight":1}',encoding='utf-8')
    manifest,_,_=_manifest(tmp_path,split_source=source_manifest)
    cache=tmp_path/'cache'
    inherited=dataset_content_identity(manifest,_dataset(),cache_dir=cache)
    assert inherited['quality_annotations_path']==str(source_quality.resolve())
    source_quality.write_text('{"weight":0}',encoding='utf-8')
    assert dataset_content_identity(manifest,_dataset(),cache_dir=cache)['fingerprint']!=inherited['fingerprint']
    local=manifest.with_name('quality_annotations.json')
    local.write_text('{"weight":2}',encoding='utf-8')
    preferred=dataset_content_identity(manifest,_dataset(),cache_dir=cache)
    assert preferred['quality_annotations_path']==str(local.resolve())
    source_quality.write_text('{"weight":3}',encoding='utf-8')
    assert dataset_content_identity(manifest,_dataset(),cache_dir=cache)['fingerprint']==preferred['fingerprint']


def test_reused_source_paths_are_digested_once_per_identity(tmp_path,monkeypatch):
    import maichart.composer.stream_data_identity as identity_module
    manifest,_,_=_manifest(tmp_path)
    seen=[]
    class Digests:
        def digest(self,path):
            seen.append(str(path))
            return str(path)
    monkeypatch.setattr(identity_module,'source_digests',lambda *_:Digests())
    dataset_content_identity(manifest,_dataset(),cache_dir=tmp_path/'cache')
    assert len(seen)==3  # manifest, one maidata, one audio


def test_old_full_checkpoint_is_inspectable_but_requires_finetune():
    from maichart.composer.stream_checkpoint import SCHEMA as CHECKPOINT_SCHEMA, FAMILY, inspect_payload
    from test_composer_stream_model import fixture
    model,_=fixture()
    payload=dict(schema=CHECKPOINT_SCHEMA,family=FAMILY,
        model_config=model.config.to_dict(),event_set_codec=model.codec.to_dict(),
        optimizer_state_dict={'state':{}},training_state={'epoch_complete':0})
    old=inspect_payload(payload)
    assert old['valid'] and old['can_init'] and not old['can_resume']
    assert 'finetune' in old['training_migration']
    assert old['model_config']['template_policy']=='family_preserving_v1'
    payload['model_config'].pop('template_policy')
    assert inspect_payload(payload)['model_config']['template_policy']=='cross_family_v1'
    payload['training_state']['dataset_content_identity']={'schema':SCHEMA,'fingerprint':'abc'}
    current=inspect_payload(payload)
    assert current['can_resume'] and current['training_migration'] is None
