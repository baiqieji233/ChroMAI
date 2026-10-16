from fractions import Fraction as F

import pytest
import torch

from maichart.event_timing import EventPoint,EventTempo
from maichart.form.codec import RationalCodec
from maichart.form.rhythm_geometry import position_event_loss


def reference_loss(logits,target,prefix,codec):
    ce=torch.nn.functional.cross_entropy(logits.float().transpose(1,2),target,reduction='none')
    results=[]
    for row in range(len(target)):
        event=0
        spelling=False
        groups={}
        for i,(inp,tgt) in enumerate(zip(prefix[row].tolist(),target[row].tolist())):
            if inp in (codec.bos_id,codec.end_id) and tgt==codec.at_id:
                event+=1
            if inp==codec.at_id:
                spelling=True
            if inp in (codec.note_id,codec.end_id):
                spelling=False
            if spelling and tgt!=codec.pad_id and event>0:
                groups.setdefault(event,[]).append(i)
        losses=[ce[row,indices].mean() for indices in groups.values()]
        results.append(torch.stack(losses).mean() if losses else ce[row].sum()*0)
    return torch.stack(results)


@pytest.mark.parametrize('batch_size',[1,4,8])
def test_vector_position_loss_preserves_values_and_gradients(batch_size):
    codec=RationalCodec()
    examples=[]
    for row in range(batch_size):
        points=[EventPoint(F(0),'1'),EventPoint(F(1,6),'2h[4:1]'),
                EventPoint(F(4,7),'3/5'),EventPoint(F(123,128),'7-3[8:1]')]
        examples.append(codec.encode(points[:1+row%4]))
    width=max(map(len,examples))+5
    prefix=torch.tensor([ids[:-1]+[codec.pad_id]*(width-len(ids)+1) for ids in examples])
    target=torch.tensor([ids[1:]+[codec.pad_id]*(width-len(ids)+1) for ids in examples])
    logits=torch.randn(batch_size,width,len(codec),generator=torch.Generator().manual_seed(713),requires_grad=True)
    expected=reference_loss(logits,target,prefix,codec)
    actual=position_event_loss(logits,target,prefix,codec,reduction='none')
    torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-6)
    expected_grad=torch.autograd.grad(expected.sum(),logits,retain_graph=True)[0]
    actual_grad=torch.autograd.grad(actual.sum(),logits)[0]
    torch.testing.assert_close(actual_grad,expected_grad,rtol=1e-6,atol=1e-7)


def test_position_loss_can_reuse_main_token_ce(monkeypatch):
    codec=RationalCodec()
    ids=codec.encode([EventPoint(F(1,6),'1')])
    prefix,target=torch.tensor([ids[:-1]]),torch.tensor([ids[1:]])
    logits=torch.randn(1,len(ids)-1,len(codec),requires_grad=True)
    expected=position_event_loss(logits,target,prefix,codec)
    ce=torch.nn.functional.cross_entropy(logits.transpose(1,2),target,reduction='none')
    monkeypatch.setattr(torch.nn.functional,'cross_entropy',lambda *_a,**_k:
                        pytest.fail('Position auxiliary recomputed token CE'))
    actual=position_event_loss(logits,target,prefix,codec,token_ce=ce)
    torch.testing.assert_close(actual,expected)
    assert torch.autograd.grad(actual,logits)[0].isfinite().all()


def test_input_example_resolves_resident_audio_once_without_changing_tensors(tmp_path):
    from dataclasses import replace
    from maichart.form.data import _AudioStore,_LazyAudio,input_example,organize_song
    from test_form_pipeline import fixture_audio
    audio=fixture_audio(8)
    song=organize_song(audio,[EventTempo(F(0),120)],0)
    path=tmp_path/'track.wav'
    path.write_bytes(b'virtual-fixture')
    store=_AudioStore(lambda *_a,**_kw:audio,tmp_path/'cache')
    calls=[]
    original=store.get
    def get(source):
        calls.append(source)
        return original(source)
    store.get=get
    expected=input_example(song,0,[])
    lazy=replace(song,audio=_LazyAudio(store,path))
    actual=input_example(lazy,0,[])
    assert len(calls)==1
    assert isinstance(lazy.audio,_LazyAudio)
    assert actual['meta']==expected['meta']
    for name,value in expected.items():
        if torch.is_tensor(value):
            torch.testing.assert_close(actual[name],value,rtol=0,atol=0)
