import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor

import torch

from maichart.form.cache_support import SourceDigests, cache_roots
from maichart.form.blueprint_cache import BlueprintInputCache
from test_form_preparation_cache import cached_source


def test_source_hash_inventory_survives_restart_and_detects_replacement(tmp_path):
    path = tmp_path / 'track.bin'
    path.write_bytes(b'original-content')
    first = SourceDigests(tmp_path / 'cache')
    expected = first.digest(path)
    second = SourceDigests(tmp_path / 'cache')
    assert second.digest(path) == expected
    assert second.stats()['full_reads'] == 0
    stat = path.stat()
    path.write_bytes(b'changed--content')
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert second.digest(path) != expected
    assert second.stats()['full_reads'] == 1


def test_cache_roots_include_base_and_legacy_audio_without_scanning_each_query(tmp_path):
    base = tmp_path / 'current' / 'cache'
    old = tmp_path / 'previous' / 'cache'
    (base / 'audio').mkdir(parents=True)
    old.mkdir(parents=True)
    cache_roots.cache_clear()
    roots = cache_roots(str(base))
    assert roots[0] == base and base / 'audio' in roots and old in roots
    assert cache_roots(str(base)) is roots


def test_preparation_ignores_transport_code_revision_without_rescanning(cached_source, monkeypatch):
    from maichart.form.preparation_cache import PreparationCache
    build, _, _, _ = cached_source
    build()
    monkeypatch.setattr(PreparationCache,'_build_legacy_index',lambda _self:
                        (_ for _ in ()).throw(AssertionError('Stable cache hit must not scan legacy indexes')))
    assert build().preparation_cache_stats['hits'] == 1


def test_single_indexed_annotation_still_checks_chart_hash(cached_source):
    from maichart.training.quality_annotations import SCHEMA
    build, chart, audio, _ = cached_source
    chart.parent.parent.joinpath('quality_annotations.json').write_text(json.dumps({
        'schema':SCHEMA,'records':[{'source_path':str(chart),'audio_path':str(audio),
        'difficulty_index':5,'chart_sha256':'outdated','label':'excluded','weight':0}]}),encoding='utf-8')
    assert len(build().songs) == 1


def test_blueprint_legacy_promotion_is_thread_safe_and_stable(tmp_path, monkeypatch):
    cache = BlueprintInputCache(tmp_path / 'cache')
    key, identity = cache.identity({'chart_sha256':'chart','audio_sha256':'audio','difficulty':5})
    legacy = dict(identity,code_sha256='previous',source='old/song.txt')
    old_key = hashlib.sha256(json.dumps(legacy,sort_keys=True).encode()).hexdigest()
    assert cache.save(old_key,legacy,{'audio':torch.ones(1,2,120)},
                      {'load_curve_target':torch.zeros(1,2,4)},1.) is None
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _:cache.load(key,identity),range(4)))
    assert all(result['input_audio'].shape == (1,2,120) for result in results)
    following = BlueprintInputCache(tmp_path / 'cache')
    monkeypatch.setattr(following,'_build_legacy_index',lambda:
                        (_ for _ in ()).throw(AssertionError('Stable Blueprint hit must not scan')))
    changed = dict(identity,code_sha256='another-transport-edit',source='current/song.txt')
    assert following.load(key,changed) is not None
    assert not list(cache.root.glob('*.tmp'))
