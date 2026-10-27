import json

import pytest
import torch

from maichart.form import data
from test_form_pipeline import fixture_audio


@pytest.fixture
def cached_source(tmp_path, monkeypatch):
    # Each case measures its own cold build; cross-dataset reuse is covered
    # separately and must not import another test's identical fixture cache.
    from maichart.form.preparation_cache import PreparationCache
    monkeypatch.setattr(PreparationCache,'_legacy_roots',lambda self:[self.root])
    folder = tmp_path / 'song'; folder.mkdir()
    chart = folder / 'maidata.txt'
    chart.write_text('&title=fixture\n&wholebpm=120\n&lv_5=13\n'
                     '&inote_5=(120){12}1,2,3,1h[4:4],2,3,1-5[4:4],2,3,4,5,6,E\n',
                     encoding='utf-8')
    audio = folder / 'track.wav'; audio.write_bytes(b'fixture')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'songs': [{'song_id': 'fixture', 'maidata_path': 'song/maidata.txt',
        'audio_path': 'song/track.wav', 'difficulties': [{'difficulty_index': 5, 'level_raw': '13'}]}]}),
        encoding='utf-8')
    calls = []
    def load(path, **_kwargs):
        calls.append(path)
        return fixture_audio(7)
    monkeypatch.setattr(data, 'extract_event_audio', load)
    def build(**kwargs):
        return data.FormDataset(manifest, cache_dir=tmp_path / 'cache', audio_loader=load,
                                max_tokens=512, **kwargs)
    return build, chart, audio, calls


def test_warm_cache_skips_structure_preparation_and_preserves_all_sample_tensors(cached_source, monkeypatch):
    build, _, _, calls = cached_source
    cold = build()
    expected = [cold[i] for i in range(len(cold))]
    assert cold.preparation_cache_stats['rebuilt'] == 1
    def unexpected(*_args, **_kwargs):
        raise AssertionError('Warm cache must not parse or rebuild chart structure')
    monkeypatch.setattr(data, 'parse_event_chart', unexpected)
    monkeypatch.setattr(data, 'organize_song', unexpected)
    monkeypatch.setattr(data, 'SustainStateIndex', unexpected)
    calls.clear()
    warm = build()
    assert warm.preparation_cache_stats['hits'] == 1
    assert not calls  # Audio is loaded lazily when a training sample is fetched.
    assert cold.audit['charts'] == warm.audit['charts']
    assert cold.songs[0][5].to_dict() == warm.songs[0][5].to_dict()
    for index, original in enumerate(expected):
        actual = warm[index]
        assert original['meta'] == actual['meta']
        for key, value in original.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, actual[key], rtol=0, atol=0)
    # Cached preparation must not fix the reconstruction's epoch-dependent mask.
    warm.seed += 100003
    cold.seed = warm.seed
    for index, item in enumerate(warm.items):
        if item[3] == 1:
            assert torch.equal(warm[index]['draft'], cold[index]['draft'])


@pytest.mark.parametrize('change', ['chart', 'audio', 'structure', 'settings', 'quality'])
def test_preparation_cache_invalidates_changed_inputs(cached_source, change):
    build, chart, audio, _ = cached_source
    cold = build()
    extra = {}
    if change == 'chart':
        chart.write_text(chart.read_text(encoding='utf-8').replace('{12}1,2', '{12}7,2'), encoding='utf-8')
    elif change == 'audio':
        audio.write_bytes(b'changed-fixture')
    elif change == 'structure':
        chart.with_name('music_structure.json').write_text(json.dumps({'meter_map': [
            {'beat': '0', 'numerator': 3, 'denominator': 4}]}), encoding='utf-8')
    elif change == 'settings':
        extra['max_region_seconds'] = 3.
    else:
        from maichart.training.quality_annotations import SCHEMA, chart_sha256
        chart.parent.parent.joinpath('quality_annotations.json').write_text(json.dumps({
            'schema': SCHEMA, 'records': [{'difficulty_index': 5, 'label': 'excellent',
                                         'source_path': str(chart), 'audio_path': str(audio),
                                         'weight': 2., 'chart_sha256': chart_sha256(chart, 5)}]}), encoding='utf-8')
    rebuilt = build(**extra)
    assert rebuilt.preparation_cache_stats['hits'] == 0
    assert rebuilt.preparation_cache_stats['rebuilt'] == 1
    if change == 'quality':
        assert rebuilt.songs[0][3] == 2. and cold.songs[0][3] == 1.
        assert build(**extra).preparation_cache_stats['hits'] == 1


def test_damaged_array_cache_is_rebuilt_instead_of_blocking_training(cached_source):
    build, _, _, _ = cached_source
    cold = build()
    from pathlib import Path
    root = Path(cold.preparation_cache_stats['path'])
    next(root.glob('*.npz')).write_bytes(b'damaged')
    rebuilt = build()
    assert rebuilt.preparation_cache_stats['invalid'] == 1
    assert rebuilt.preparation_cache_stats['rebuilt'] == 1
    assert len(rebuilt) == len(cold)
    assert build().preparation_cache_stats['hits'] == 1


def test_recurrence_policy_has_a_distinct_semantic_cache_key(cached_source):
    from maichart.form.recurrence import RECURRENCE_DESCRIPTOR_POLICY
    build,_,_,_=cached_source
    legacy=build()
    assert legacy.preparation_cache_stats['rebuilt']==1
    assert legacy.songs[0][0].relations.audit['descriptor_policy']=='legacy_temporal_unit_v1'
    revised=build(recurrence_descriptor_policy=RECURRENCE_DESCRIPTOR_POLICY)
    assert revised.preparation_cache_stats['hits']==0
    assert revised.preparation_cache_stats['rebuilt']==1
    assert revised.songs[0][0].relations.audit['descriptor_policy']==RECURRENCE_DESCRIPTOR_POLICY
    assert build(recurrence_descriptor_policy=RECURRENCE_DESCRIPTOR_POLICY).preparation_cache_stats['hits']==1
    assert build().preparation_cache_stats['hits']==1


def test_unrelated_quality_edit_does_not_invalidate_other_chart_cache(cached_source):
    build, chart, _, _ = cached_source
    build()
    from maichart.training.quality_annotations import SCHEMA
    chart.parent.parent.joinpath('quality_annotations.json').write_text(json.dumps({
        'schema': SCHEMA, 'records': [{'difficulty_index': 5, 'label': 'reduced', 'weight': .25,
                                     'chart_sha256': 'unrelated-chart'}]}), encoding='utf-8')
    assert build().preparation_cache_stats['hits'] == 1
