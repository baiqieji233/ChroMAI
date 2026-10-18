"""Mapped inputs, causal structures, objective and ordering equivalence."""
from dataclasses import replace
from fractions import Fraction as F
from pathlib import Path
import json,time
import numpy as np
import pytest,torch
from torch.utils.data import DataLoader

from maichart.stream.input_cache import MappedInputCache
from maichart.tempo_clock import TempoClock
from maichart.stream.data import StreamDataset,collate_stream
from maichart.stream.prefetch import StreamBatchPrefetch
from maichart.event_timing import EventTempo,seconds_at_beat
from maichart.stream.checkpoint import save,load
from maichart.stream.model import ChartStreamTransformer
from maichart.stream.train import _loss
from maichart.stream.training_batch import transfer_batch
from test_stream_grounding import source,config,cpu_threads


def test_prepared_tempo_clock_matches_reference_at_changes_duplicates_and_exact_fractions():
    tempo=[EventTempo(F(0),123.45),EventTempo(F(3,7),177.7),EventTempo(F(3,7),99.8),EventTempo(F(11),92.55)]
    for first in (-1.82,0.,.45):
        clock=TempoClock(tempo,first)
        for beat in [F(0),F(1,997),F(3,7),F(4,7),F(11),F(23),F(2345,23)]:
            assert clock.at(beat)==seconds_at_beat(beat,tempo,first)


def test_mapped_cache_reuses_full_values_and_rebuilds_corruption(tmp_path):
    store=MappedInputCache(tmp_path);calls=[]
    def build():calls.append(1);return {'x':torch.arange(8.)},{'duration':1.25}
    a,_=store.get({'source':'a'},build)
    b,_=store.get({'source':'a'},build)
    torch.testing.assert_close(a['x'],b['x']);assert len(calls)==1
    store.get({'source':'b'},build);assert len(calls)==2
    # Replace an immutable artifact with an invalid file; a new cache reader
    # must reject it rather than silently serving stale/broken tensors.
    del b
    index=next(store.root.glob('*.json'));info=json.loads(index.read_text())
    (store.root/info['file']).write_bytes(b'corrupt')
    other=MappedInputCache(tmp_path)
    for identity in ({'source':'a'},{'source':'b'}):other.get(identity,build)
    assert other.stats()['invalid']==1 and len(calls)==3


def test_cache_write_failure_keeps_valid_training_inputs(tmp_path,monkeypatch):
    import maichart.stream.input_cache as cache_module
    store=MappedInputCache(tmp_path)
    def fail(**kwargs):raise OSError('disk full')
    monkeypatch.setattr(cache_module.tempfile,'mkstemp',fail)
    tensors,_=store.get({'source':'a'},lambda:({'x':torch.ones(2)},{}))
    torch.testing.assert_close(tensors['x'],torch.ones(2))
    assert store.stats()['write_errors']==1


def test_discovery_index_avoids_rereading_unchanged_metadata_and_invalidates_updates(tmp_path):
    from maichart.form.discovery_index import DiscoveryIndex
    path=tmp_path/'entry.json';path.write_text('{"version":1}')
    calls=[]
    def read(path):calls.append(1);return json.loads(path.read_text())
    index=DiscoveryIndex(tmp_path/'index.sqlite3')
    assert index.read(path,read)=={'version':1};index.close()
    index=DiscoveryIndex(tmp_path/'index.sqlite3')
    assert index.read(path,read)=={'version':1} and len(calls)==1
    path.write_text('{"version":200}')
    assert index.read(path,read)=={'version':200} and len(calls)==2
    index.close()


def make_dataset(source,mode):
    root,manifest=source
    return StreamDataset(manifest,cache_dir=root/'pipeline_cache',max_tokens=32,max_history_tokens=12,
        grounding_config=config(),preparation_workers=1,audio_cache_gb=.03,
        history_note_dropout=.45,prefix_note_dropout=.35,seed=83,input_cache_mode=mode)


def compare_items(a,b):
    assert a.keys()==b.keys()
    for key in a:
        if torch.is_tensor(a[key]):torch.testing.assert_close(a[key],b[key],atol=0,rtol=0,msg=key)
        else:assert a[key]==b[key],key


def test_mapped_inputs_and_reopened_cache_preserve_epoch_masks_and_targets(source):
    legacy=make_dataset(source,'legacy');mapped=make_dataset(source,'mapped')
    assert legacy.fingerprint==mapped.fingerprint
    for epoch in (1,2):
        legacy.set_epoch(epoch);mapped.set_epoch(epoch)
        for i in range(len(legacy)):compare_items(legacy[i],mapped[i])
    reopened=make_dataset(source,'mapped');reopened.set_epoch(2)
    # A warm transport hit must not fall back to base feature I/O just for
    # duration or structure. Preparation itself has already finished.
    reopened.base.audio_store.get=lambda *_:pytest.fail('Loaded base audio after mapped input cache hit')
    for i in range(len(legacy)):compare_items(legacy[i],reopened[i])
    assert reopened.input_cache.stats()['hits']>0


def test_cross_batch_prefetch_preserves_shuffle_and_does_not_wait_for_batch_barrier():
    import threading
    ready=threading.Event();events=[]
    class Dataset:
        def __len__(self):return 6
        def prepare_concurrency_key(self,i):return i
        def __getitem__(self,i):
            if i==0:assert ready.wait(2), 'Later batch never started while first waited'
            if i==2:ready.set()
            events.append(i);return {'value':torch.tensor([i])}
    loader=DataLoader(Dataset(),batch_size=2,collate_fn=lambda rows:{'value':torch.stack([r['value'] for r in rows])},
        generator=torch.Generator().manual_seed(1))
    with StreamBatchPrefetch(loader,depth=2,workers=3) as stream:
        result=torch.cat([b['value'] for b in stream]).flatten().tolist()
    assert result==list(range(6)) and events.index(2)<events.index(0)


def test_new_prefetch_matches_dataloader_shuffle_and_full_loss(source):
    ds=make_dataset(source,'mapped');ds.set_epoch(3)
    def loader():return DataLoader(ds,batch_size=2,shuffle=True,collate_fn=collate_stream,generator=torch.Generator().manual_seed(3))
    a=list(loader())
    with StreamBatchPrefetch(loader(),depth=2,workers=4) as stream:b=list(stream)
    for x,y in zip(a,b):compare_items(x,y)
    model=ChartStreamTransformer(config()).eval()
    baseline=model(a[0]);trimmed=transfer_batch(a[0],torch.device('cpu'));actual=model(trimmed)
    torch.testing.assert_close(baseline['logits'],actual['logits'],atol=0,rtol=0)
    la,_=_loss(baseline,a[0]);lb,_=_loss(actual,trimmed)
    torch.testing.assert_close(la,lb,atol=0,rtol=0)


def test_one_serialization_step_alias_survives_replacing_last(tmp_path,monkeypatch):
    model=ChartStreamTransformer(config());manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    last,step=tmp_path/'last.pt',tmp_path/'step.pt';calls=[];real_save=torch.save
    def writing(*args,**kwargs):calls.append(1);return real_save(*args,**kwargs)
    monkeypatch.setattr(torch,'save',writing)
    save(last,model,None,epoch=1,args={},manifest=manifest,aliases=(step,))
    assert len(calls)==1 and last.read_bytes()==step.read_bytes()
    save(last,model,None,epoch=2,args={},manifest=manifest)
    assert load(step)[1]['epoch']==1 and load(last)[1]['epoch']==2


def test_contiguous_cross_entropy_preserves_full_loss_and_gradients(source,monkeypatch):
    from torch.nn import functional as functional
    ds=make_dataset(source,'mapped');batch=collate_stream([ds[0],ds[-1]])
    model=ChartStreamTransformer(config()).eval()
    loss,_=_loss(model(batch),batch);loss.backward()
    expected={name:p.grad.clone() for name,p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    real=functional.cross_entropy;size=batch['target'].size(0)
    def strided(input,target,*args,**kwargs):
        if input.ndim==2 and target.ndim==1:
            values=input.reshape(size,-1,input.size(-1)).transpose(1,2)
            labels=target.reshape(size,-1)
            return real(values,labels,*args,**kwargs).reshape(-1)
        return real(input,target,*args,**kwargs)
    monkeypatch.setattr(functional,'cross_entropy',strided)
    actual,_=_loss(model(batch),batch);actual.backward()
    torch.testing.assert_close(actual,loss,rtol=1e-6,atol=1e-6)
    for name,p in model.named_parameters():
        if name in expected:torch.testing.assert_close(p.grad,expected[name],rtol=2e-5,atol=2e-6,msg=name)
