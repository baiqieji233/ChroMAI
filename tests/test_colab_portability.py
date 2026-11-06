import hashlib
import json
from pathlib import Path

from maichart.portable_paths import material_path
from maichart.training.dataset_catalog import _source_asset
from maichart.composer.stream_data_identity import _annotation_path


def test_material_paths_are_opt_in_and_prefix_bounded(tmp_path, monkeypatch):
    monkeypatch.delenv('MAICHART_PATH_MAP', raising=False)
    assert material_path('relative/file') == Path('relative/file')
    mapping = tmp_path / 'map.json'
    mapping.write_text(json.dumps({'E:/corpus': str(tmp_path / 'assets')}))
    monkeypatch.setenv('MAICHART_PATH_MAP', str(mapping))
    assert material_path(r'e:\corpus\中文\track.mp3') == tmp_path / 'assets/中文/track.mp3'
    assert material_path('E:/corpus2/track.mp3') == Path('E:/corpus2/track.mp3')


def test_relocation_preserves_manifest_bytes_and_fallback_annotations(tmp_path, monkeypatch):
    data_root = tmp_path / 'relocated'
    data_root.mkdir()
    audio = data_root / 'track.mp3'
    audio.write_bytes(b'unchanged audio')
    annotation = data_root / 'quality_annotations.json'
    annotation.write_text('{}')
    mapping = tmp_path / 'map.json'
    mapping.write_text(json.dumps({'E:/dataset': str(data_root)}))
    monkeypatch.setenv('MAICHART_PATH_MAP', str(mapping))
    manifest = tmp_path / 'train.json'
    payload = dict(source_root='E:/dataset', split_metadata=dict(source_manifest=r'E:\dataset\training_manifest.json'),
                   songs=[dict(audio_path=r'E:\dataset\track.mp3')])
    manifest.write_text(json.dumps(payload))
    before = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert _source_asset(payload['songs'][0]['audio_path'], manifest, payload) == audio.resolve()
    assert _source_asset('track.mp3', manifest, payload) == audio.resolve()
    assert _annotation_path(manifest, payload) == annotation.resolve()
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == before


def test_resume_content_fingerprint_survives_relocation(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from maichart.composer.stream_data_identity import dataset_content_identity

    mapping = tmp_path / 'map.json'
    monkeypatch.setenv('MAICHART_PATH_MAP', str(mapping))
    manifest = tmp_path / 'train.json'
    manifest.write_text(json.dumps(dict(
        split_metadata=dict(source_manifest='E:/corpus/all.json'),
        songs=[dict(maidata_path='E:/corpus/maidata.txt', audio_path='E:/corpus/track.mp3')]
    )), encoding='utf-8')
    dataset = SimpleNamespace(streams=[[]], conditions=[{'level': 12.0}], chart_weights=[1.0])
    identities = []
    for name in ('original', 'cloud'):
        folder = tmp_path / name
        folder.mkdir()
        for filename, content in (('maidata.txt', b'chart'), ('track.mp3', b'audio'),
                                  ('quality_annotations.json', b'{"records":[]}')):
            (folder / filename).write_bytes(content)
        mapping.write_text(json.dumps({'E:/corpus': str(folder)}), encoding='utf-8')
        identities.append(dataset_content_identity(manifest, dataset, cache_dir=folder / 'cache'))
    assert identities[0]['fingerprint'] == identities[1]['fingerprint']
    assert identities[0]['quality_annotations_path'] != identities[1]['quality_annotations_path']
