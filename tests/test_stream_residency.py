"""Resident inputs retain exact values and eliminate input-file reads."""
from fractions import Fraction
import numpy as np
import pytest,torch

from maichart.stream.input_cache import MappedInputCache,CompactBeatText,LazyBeats,tensor_bytes
from test_stream_pipeline import make_dataset,compare_items
from test_stream_grounding import source,cpu_threads


def test_compact_beat_text_keeps_every_exact_rational_and_slice():
    values=['0','1/997','1234567890123456789012345/617','7/3']
    compact=CompactBeatText(values);beats=LazyBeats(compact)
    assert len(beats)==4 and list(beats)==list(map(Fraction,values))
    assert beats[1:3]==tuple(map(Fraction,values[1:3]))


def test_resident_tensors_are_owned_and_get_does_not_read_files(tmp_path,monkeypatch):
    cache=MappedInputCache(tmp_path);identity={'source':'test'}
    cache.get(identity,lambda:({'values':torch.arange(20.)},{'beats':['0','1/17']}))
    report=cache.preload([identity],budget_bytes=1_000_000,policy='resident',
        memory=lambda:(16*1024**3,8*1024**3))
    assert report['mode']=='resident'
    monkeypatch.setattr(cache,'_read',lambda *_:pytest.fail('Resident access opened a file'))
    values,extra=cache.get(identity,lambda:pytest.fail('Rebuilt resident value'))
    torch.testing.assert_close(values['values'],torch.arange(20.))
    assert list(LazyBeats(extra['beats']))==[Fraction(0),Fraction(1,17)]
    assert cache.stats()['resident_hits']==1


def test_residency_budget_and_memory_reserve_are_enforced(tmp_path):
    cache=MappedInputCache(tmp_path);identities=[{'source':i} for i in range(2)]
    for identity in identities:cache.get(identity,lambda:({'x':torch.ones(10000)},{}))
    report=cache.preload(identities,budget_bytes=45000,memory=lambda:(16*1024**3,8*1024**3))
    assert report['loaded']==1 and report['capacity_skips']==1 and report['bytes']<=45000
    other=MappedInputCache(tmp_path)
    report=other.preload(identities,budget_bytes=100000,reserve_bytes=4*1024**3,
        memory=lambda:(16*1024**3,3*1024**3))
    assert report['loaded']==0 and report['memory_skips']==2
    assert other.preload(identities,budget_bytes=0)['mode']=='mapped'


def test_fully_resident_dataset_matches_mapped_and_never_opens_input_files(source,monkeypatch):
    original=make_dataset(source,'mapped');original.set_epoch(4)
    expected=[original[i] for i in range(len(original))]
    resident=make_dataset(source,'mapped');resident.set_epoch(4)
    status=resident.prepare_residency(policy='resident',budget_bytes=10*1024**2,reserve_bytes=1)
    assert status['mode']=='resident'
    monkeypatch.setattr(resident.input_cache,'_read',lambda *_:pytest.fail('Read file during resident sampling'))
    for i,item in enumerate(expected):compare_items(item,resident[i])
    assert resident.fingerprint==original.fingerprint
