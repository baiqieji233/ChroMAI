from collections import OrderedDict

import numpy as np
import pytest
import torch

from maichart.composer.runtime_memory import GIB,cache_capacity,host_memory
from maichart.composer.train import RemainingBatches,_cuda_execution


def test_capacity_reserves_commit_and_physical_memory():
    assert cache_capacity(3*GIB,0,None)==3*GIB
    assert cache_capacity(3*GIB,0,{'free_commit':9*GIB,'free_physical':8*GIB})==3*GIB
    assert cache_capacity(3*GIB,2*GIB,{'free_commit':3*GIB,'free_physical':8*GIB})==0
    assert cache_capacity(3*GIB,2*GIB,{'free_commit':9*GIB,'free_physical':GIB})==0
    assert cache_capacity(3*GIB,GIB,{'free_commit':GIB,'free_physical':GIB})==0


def test_headroom_evicts_lru_without_changing_arrays(monkeypatch):
    from maichart.composer import data
    dataset=data.ComposerDataset.__new__(data.ComposerDataset)
    dataset._memory_checked_at=float('-inf')
    dataset._feature_cache_budget_bytes=3*GIB
    arrays=(np.arange(8,dtype=np.float32),np.arange(8,dtype=np.float64))
    dataset._feature_cache=OrderedDict([(1,arrays)])
    dataset._feature_cache_bytes=sum(a.nbytes for a in arrays)
    dataset._feature_lock=__import__('threading').RLock()
    dataset._mapped_bank_cache=OrderedDict()
    dataset._mapped_bank_cache_limit=768
    monkeypatch.setattr(data,'host_memory',lambda:{'free_commit':0,'free_physical':0})
    dataset._check_cache_headroom()
    assert dataset._feature_cache_bytes==0
    assert dataset._feature_cache_effective_budget_bytes==0
    assert not dataset._feature_cache
    np.testing.assert_array_equal(arrays[0],np.arange(8,dtype=np.float32))


def test_batch_limit_does_not_prefetch_past_training_stop():
    batches=list(range(10))
    assert list(RemainingBatches(batches,2,5))==[2,3,4]
    assert len(RemainingBatches(batches,2,5))==3
    assert list(RemainingBatches(batches,2))==list(range(2,10))
    assert len(RemainingBatches(batches,5,2))==0


def test_song_local_batch_sampler_preserves_all_samples():
    from types import SimpleNamespace
    from maichart.composer.train import SongLocalBatchSampler
    dataset=SimpleNamespace(samples=[SimpleNamespace(song_index=i//5) for i in range(23)])
    sampler=SongLocalBatchSampler(dataset,4,torch.Generator().manual_seed(3),locality=3)
    batches=list(sampler)
    flat=[index for batch in batches for index in batch]
    assert sorted(flat)==list(range(23))
    assert len(batches)==6


def test_cuda_failure_reports_host_memory_without_retry(monkeypatch,capsys):
    from maichart.composer import train
    monkeypatch.setattr(train,'host_memory',lambda:{'free_commit':GIB})
    monkeypatch.setattr(torch.cuda,'memory_allocated',lambda device:0)
    monkeypatch.setattr(torch.cuda,'memory_reserved',lambda device:0)
    with pytest.raises(RuntimeError,match='system commit'):
        with _cuda_execution(torch.device('cuda')):
            raise RuntimeError('CUBLAS_STATUS_EXECUTION_FAILED')
    assert 'free_commit_gib' in capsys.readouterr().out
    with pytest.raises(RuntimeError,match='ordinary error'):
        with _cuda_execution(torch.device('cpu')):
            raise RuntimeError('ordinary error')


def test_live_host_memory_contract():
    snapshot=host_memory()
    if snapshot is not None:
        assert 0<=snapshot['free_commit']<=snapshot['commit_limit']
        assert 0<=snapshot['free_physical']<=snapshot['physical']
