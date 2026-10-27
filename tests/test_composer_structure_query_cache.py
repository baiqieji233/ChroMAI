from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from maichart.composer import structure, structure_backend


def make_song():
    times = np.arange(0., 12., .02)
    features = np.zeros((len(times), 98), dtype=np.float32)
    features[:, 92] = np.exp(-((times % .5) / .05) ** 2)
    audio = SimpleNamespace(features=features, times=times, duration_sec=12.,
                            metadata={}, source='unavailable.wav', feature_policy='test')
    return SimpleNamespace(audio=audio, tempo=[SimpleNamespace(beat=Fraction(0), bpm=120.)], first=0.)


@pytest.mark.parametrize('policy', ['off', 'native_periodic_v1', 'metrical_context_v1'])
@pytest.mark.parametrize('previous', [None, (3.125, 1.5625)])
def test_combined_matches_separate_bitwise(tmp_path, policy, previous):
    structure.clear_structure_caches()
    song = make_song()
    beats = np.array([0., 2.25, 3.125, 7.9999, 8., 13.], dtype=np.float64)
    seconds = beats / 2
    kwargs = dict(policy=policy, cache_dir=tmp_path, backend='native')
    expected = structure.structure_features(song, beats, seconds, **kwargs)
    baseline = (structure.structure_features(song, [previous[0]], [previous[1]], **kwargs)
                if previous else np.zeros((1, structure.STRUCTURE_DIM), dtype=np.float32))
    actual, delta = structure.structure_features_and_transition(
        song, beats, seconds, *(previous or (None, None)), **kwargs)
    assert actual.dtype == delta.dtype == np.float32
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    np.testing.assert_array_equal(delta.view(np.uint32), (expected - baseline).view(np.uint32))


def test_warm_query_skips_identity_work(monkeypatch, tmp_path):
    structure.clear_structure_caches()
    song = make_song()
    kwargs = dict(cache_dir=tmp_path, backend='native')
    structure.structure_features(song, [1.], [.5], **kwargs)
    def forbidden(*args, **kwargs):
        raise AssertionError('identity recomputed on warm query')
    monkeypatch.setattr(structure, '_identity', forbidden)
    structure.structure_features(song, [2.], [1.], **kwargs)


def test_backend_identity_refresh_rechecks_availability(monkeypatch):
    # Mock the availability boundary itself; no checkpoint is created or changed.
    monkeypatch.setattr(structure_backend.importlib.util, 'find_spec', lambda _: None)
    first = structure_backend.backend_identity('auto', refresh=True)
    monkeypatch.setattr(structure_backend.importlib.util, 'find_spec', lambda _: object())
    class MissingCheckpoint:
        def is_file(self): return False
        def __str__(self): return 'mock-missing.ckpt'
    monkeypatch.setattr(structure_backend, 'checkpoint_path', lambda: MissingCheckpoint())
    assert structure_backend.backend_identity('auto') == first
    refreshed = structure_backend.backend_identity('auto', refresh=True)
    assert refreshed['name'] == first['name']
    assert refreshed['fallback'] != first['fallback']
    assert structure_backend.backend_identity('auto') == refreshed
    structure.clear_structure_caches()


def test_fast_bank_invalidates_changed_clock_audio_and_directory(monkeypatch,tmp_path):
    structure.clear_structure_caches();song=make_song()
    calls=[];original=structure._identity
    def counted(*args,**kwargs):
        calls.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(structure,'_identity',counted)
    def query(directory=tmp_path):
        return structure.structure_features(song,[2.],[1.],cache_dir=directory,backend='native')
    query();query();assert len(calls)==1
    song.first=.25;query();assert len(calls)==2
    song.tempo[0].bpm=135.;query();assert len(calls)==3
    song.audio=make_song().audio;query();assert len(calls)==4
    query(tmp_path/'other');assert len(calls)==5


def test_warm_combined_query_performs_no_discovery_or_path_resolution(monkeypatch,tmp_path):
    from pathlib import Path
    structure.clear_structure_caches();song=make_song()
    kwargs=dict(cache_dir=tmp_path,backend='native')
    structure.structure_features_and_transition(song,[2.],[1.],1.,.5,**kwargs)
    def forbidden(*args,**kwargs):raise AssertionError('filesystem/discovery on hot query')
    with monkeypatch.context() as m:
        m.setattr(Path,'resolve',forbidden)
        m.setattr(structure_backend.importlib.util,'find_spec',forbidden)
        m.setattr(structure,'_identity',forbidden)
        result,_=structure.structure_features_and_transition(song,[3.],[1.5],2.,1.,**kwargs)
        assert result.shape==(1,48)


def test_explicit_backend_refresh_invalidates_fast_banks(monkeypatch,tmp_path):
    structure.clear_structure_caches();song=make_song()
    calls=[];original=structure._identity
    def identity(*args,**kwargs):calls.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(structure,'_identity',identity)
    kwargs=dict(cache_dir=tmp_path,backend='native')
    structure.structure_features(song,[2.],[1.],**kwargs)
    structure.structure_features(song,[3.],[1.5],**kwargs)
    assert len(calls)==1
    monkeypatch.setattr(structure_backend,'_discover_backend_identity',
        lambda *args,**kwargs:{'name':'native_multicycle_v1','revision':'changed'})
    structure_backend.backend_identity('native',refresh=True)
    structure.structure_features(song,[3.],[1.5],**kwargs)
    assert len(calls)==2
    structure.clear_structure_caches()


def test_cached_identity_is_not_mutable_through_returned_dictionary():
    structure_backend.clear_backend_identity_cache()
    first=structure_backend.backend_identity('native')
    first['name']='accidental_caller_edit'
    assert structure_backend.backend_identity('native')['name']=='native_multicycle_v1'
