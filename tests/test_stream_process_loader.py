"""CPU spawn preserves real-row order, epoch seeds and resumed augmentation."""
from types import SimpleNamespace
import torch
from maichart.composer.stream_data import StreamDataset
from maichart.composer.stream_train import _loader,_prepared_batches,_shutdown_process_loader
from test_stream_phrase_execution import execution_fixture


class SpawnFixtureDataset(StreamDataset):
    def _bank(self,song):return self.spawn_fixture_bank


def args(workers):
    return SimpleNamespace(seed=947,batch_size=2,device='cpu',prefetch_batches=2,data_workers=workers,
        prefix_augmentation_rate=.65,prefix_augmentation_input_policy='causal_window_config_time_first_decision_v3')


def assert_same(left,right):
    if torch.is_tensor(left):torch.testing.assert_close(left,right,rtol=0,atol=0)
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for name in left:assert_same(left[name],right[name])
    elif isinstance(left,(tuple,list)):
        assert len(left)==len(right)
        for a,b in zip(left,right):assert_same(a,b)
    else:assert left==right


def test_spawn_real_batches_match_eager_epoch_and_resume(monkeypatch):
    torch.set_num_threads(1)
    ds,_,_,bank=execution_fixture(monkeypatch)
    ds.__class__=SpawnFixtureDataset;ds.spawn_fixture_bank=bank
    try:
        previous=None
        for epoch,skip,symbolic in ((1,0,False),(2,0,True),(2,1,False)):
            ds.set_symbolic_only(symbolic)
            eager=list(_loader(ds,args(0),epoch,skip))
            process=_loader(ds,args(1),epoch,skip)
            if previous is not None:assert process is previous
            previous=process
            with _prepared_batches(process,args(1)) as prepared:
                actual=list(prepared)
                stats=prepared.stats()
            assert len(actual)==len(eager)
            for a,b in zip(actual,eager):assert_same(a,b)
            assert stats['cache_scope']=='worker_process'
            assert stats['prepare_sec']>=0 and stats['consumer_wait_sec']>=0
    finally:_shutdown_process_loader(ds)

def test_process_sampler_keeps_exact_local_order_and_explicit_cursor(monkeypatch):
    from maichart.composer.stream_train import _LocalBatches,_ProcessBatches
    ds,_,_,_=execution_fixture(monkeypatch)
    ds.set_symbolic_only(True)
    sampler=_ProcessBatches(ds,args(1),7,1)
    expected=list(_LocalBatches(ds,2,947,7,1))
    actual=list(sampler)
    assert [[key[0] for key in batch] for batch in actual]==expected
    assert all(key[1:]==(7,number,True) for number,batch in enumerate(actual,2) for key in batch)
