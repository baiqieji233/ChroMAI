import json
from pathlib import Path

import pytest

from maichart.form.preparation_cache import PreparationCache, SCHEMA, _json_bytes, _read_index_identity


def _canonical(identity, *, result=None):
    return _json_bytes(dict(arrays_sha256='a'*64, identity=identity,
                           result={'state_markers': ['large-chart-data']*20000} if result is None else result,
                           schema=SCHEMA))


@pytest.mark.parametrize('payload', [
    {'settings': {'quality_profile': 'musical_evidence_v1'}, 'path': '全量官谱/谱面'},
    {'quote': 'escaped \\" braces } and nested {', 'array': [1, {'x': [False, None]}]},
    {'regions': [{'name': '段落'+str(i), 'weight': .5} for i in range(250)]},
])
def test_canonical_scan_only_parses_identity_prefix(tmp_path, monkeypatch, payload):
    path = tmp_path/'entry.index.json'
    path.write_bytes(_canonical(payload))
    def no_full_read(*args, **kwargs):
        raise AssertionError('Canonical index scan materialized the complete chart JSON')
    monkeypatch.setattr(Path, 'read_text', no_full_read)
    assert _read_index_identity(path) == payload


@pytest.mark.parametrize('record', [
    {'identity': {'version': 1}, 'arrays_sha256': 'a'*64, 'result': {'large': 'x'*20000}},
    {'result': {'large': 'x'*20000}, 'identity': {'version': 2}, 'schema': SCHEMA},
    {'schema': SCHEMA, 'arrays_sha256': 'a'*64, 'identity': {'version': 3}},
    {'result': {}},
])
def test_arbitrary_legacy_order_falls_back_to_full_parser(tmp_path, record):
    path = tmp_path/'legacy.index.json'
    path.write_text(json.dumps(record), encoding='utf-8')
    assert _read_index_identity(path) == record.get('identity')


def test_oversized_identity_retains_full_parser(tmp_path):
    path = tmp_path/'large.index.json'
    identity = {'many_regions': 'x'*(80*1024)}
    path.write_bytes(_canonical(identity, result={}))
    assert _read_index_identity(path) == identity


def test_corrupt_index_discovery_cannot_bypass_full_load_validation(tmp_path, monkeypatch):
    cache = PreparationCache(tmp_path)
    cache.root.mkdir(parents=True)
    identity = {'schema': SCHEMA, 'version': 4}
    path = cache.root/'bad.index.json'
    # Identity is intact but the full payload is corrupt. A discovery hint may
    # index it, while the unchanged acceptance path must reject the candidate.
    prefix = _canonical(identity, result={}).decode('utf-8')
    path.write_text(prefix[:-1]+' malformed tail', encoding='utf-8')
    monkeypatch.setattr(PreparationCache, '_legacy_roots', lambda self: [self.root])
    found = cache._build_legacy_index()
    assert found[cache._normalized_identity(identity)] == [path]
    value, status = cache.load('absent-target', identity)
    assert value is None and status == 'miss'


def test_valid_prefix_and_noncanonical_entries_share_same_identity_index(tmp_path, monkeypatch):
    cache = PreparationCache(tmp_path)
    cache.root.mkdir(parents=True)
    identity = {'schema': SCHEMA, 'version': 5}
    canonical = cache.root/'canonical.index.json'
    legacy = cache.root/'legacy.index.json'
    canonical.write_bytes(_canonical(identity))
    legacy.write_text(json.dumps({'result': {}, 'identity': identity}), encoding='utf-8')
    monkeypatch.setattr(PreparationCache, '_legacy_roots', lambda self: [self.root])
    assert set(cache._build_legacy_index()[cache._normalized_identity(identity)]) == {canonical, legacy}
