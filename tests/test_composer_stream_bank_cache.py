import numpy as np
import torch
import pytest
from maichart.composer.stream_bank_cache import StreamBankDiskCache


def fixture():
    return torch.arange(714,dtype=torch.float32).reshape(3,238),torch.tensor([0.,.01,.02],dtype=torch.float64),torch.tensor([0.,.02,.04],dtype=torch.float64)


def test_persistent_bank_survives_process_cache_eviction_and_preserves_bits(tmp_path):
    calls=[];expected=fixture()
    def build():calls.append(1);return expected
    identity={'grid':.01,'source':'a','first':0.}
    first=StreamBankDiskCache(tmp_path);first.get(identity,build)
    second=StreamBankDiskCache(tmp_path);actual=second.get(identity,build)
    assert len(calls)==1 and second.hits==1
    for a,b in zip(expected,actual):assert torch.equal(a,b) and a.dtype==b.dtype
    actual[0][0,0]=999.
    # Mapping writes are private; neither disk nor the original array changes.
    third=StreamBankDiskCache(tmp_path).get(identity,build)
    assert third[0][0,0]==0 and expected[0][0,0]==0
    for changed in ({**identity,'grid':.02},{**identity,'source':'b'},{**identity,'first':.1}):
        second.get(changed,build)
    assert len(calls)==4


@pytest.mark.parametrize('broken',[b'',b'broken'])
def test_truncated_cache_rebuilds(tmp_path,broken):
    cache=StreamBankDiskCache(tmp_path);cache.get({'source':'a'},fixture)
    path=next(cache.root.glob('*.npy'));path.write_bytes(broken)
    restored=cache.get({'source':'a'},fixture)
    assert cache.invalid==1 and torch.equal(restored[0],fixture()[0])


def test_failed_builder_does_not_publish(tmp_path):
    cache=StreamBankDiskCache(tmp_path)
    def fail():raise RuntimeError('builder failed')
    with pytest.raises(RuntimeError,match='builder failed'):cache.get({'source':'b'},fail)
    assert not list(tmp_path.rglob('*.npy'))
