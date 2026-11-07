"""Audio evidence must not become an event/rest gate or a second time grammar."""
from fractions import Fraction as F
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from maichart.composer_v2.onset_readout import (POLICY, audio_tiles, phase_values,
    apply_phase_evidence, effective_strength, resolve_policy)
from maichart.composer_v2.model import ComposerV2, ComposerV2Config
from maichart.composer_v2.data import ComposerV2Song
from maichart.composer_v2.stream_codec import StreamCodec
from maichart.composer_v2.stream_data import StreamDataset, collate_stream
from maichart.composer_v2.stream_generate import EventSampler
from maichart.composer_v2.audio import _interpolate
from maichart.event_timing import EventPoint, EventTempo
from maichart.tempo_clock import TempoClock


class Audio:
    duration_sec=15.
    identity={'fixture':'onset'}
    semantic_available=False
    mel_times_sec=np.arange(1501)/100
    mel=np.sin(mel_times_sec[:,None]*np.arange(1,99)[None,:]).astype(np.float32)
    def sample(self, seconds):
        return (_interpolate(self.mel,self.mel_times_sec,seconds),
                np.zeros((len(seconds),1024),np.float32),np.zeros(len(seconds),bool))


def fixture():
    data=StreamDataset.__new__(StreamDataset)
    data.codec=StreamCodec.fit(['1','2','3h[4:1]'],phases=[F(1,101)])
    data.max_tokens=256;data.history_beats=16;data.window_beats=16;data.future_beats=4
    data.onset_readout_policy=POLICY;data.onset_phases=phase_values(data.codec)
    clock=TempoClock([EventTempo(F(0),137.),EventTempo(F(5,2),193.)],.125)
    song=ComposerV2Song('onset',Path('chart'),Path('audio'),5,13.,[],clock,F(30),np.zeros(16,np.float32),audio=Audio())
    times=np.linspace(0,15,8);mel,sem,valid=song.audio.sample(times)
    song.summary=dict(mel=mel,semantic=sem,semantic_valid=valid,times_sec=times,beats=np.linspace(0,30,8))
    config=ComposerV2Config(d_model=32,layers=2,nhead=4,dim_feedforward=64,dropout=0.,
                           audio_layers=1,song_layers=1,max_tokens=256,audio_backend='native')
    return data,song,config


def pair():
    data,song,config=fixture();torch.manual_seed(7)
    base=ComposerV2(config,data.codec).eval()
    upgraded=ComposerV2(ComposerV2Config(**{**config.to_dict(),'onset_readout_policy':POLICY}),data.codec).eval()
    loaded=upgraded.load_state_dict(base.state_dict(),strict=False)
    assert all(k.startswith('onset_readout.') for k in loaded.missing_keys) and not loaded.unexpected_keys
    return data,song,base,upgraded


def test_old_checkpoint_policy_and_new_finetune_contract():
    assert resolve_policy('auto',None)==('off',False)
    assert resolve_policy(POLICY,{})==(POLICY,True)
    assert resolve_policy('auto',{'onset_readout_policy':POLICY},resume=True)==(POLICY,False)
    with pytest.raises(ValueError,match='finetune'):resolve_policy(POLICY,{},resume=True)
    assert effective_strength(None,'off')==0
    assert effective_strength(None,POLICY)==1
    with pytest.raises(ValueError,match='微调'):effective_strength(1,'off')
    for invalid in (float('nan'),float('inf'),-1,2.1):
        with pytest.raises(ValueError):effective_strength(invalid,POLICY)


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_mass_conservation_only_at_phase_query_and_gradients(dtype):
    torch.manual_seed(5)
    logits=torch.randn(2,9,30,dtype=dtype,requires_grad=True);scores=torch.randn(3,4,requires_grad=True)
    ids=torch.tensor([3,8,12,20]);queries=[(0,1,2),(1,7,0)]
    out=apply_phase_evidence(logits,scores,queries,ids)
    mask=torch.ones_like(logits,dtype=torch.bool)
    for row,pos,_ in queries:mask[row,pos,ids]=False
    assert torch.equal(logits[mask],out[mask])
    for row,pos,_ in queries:
        torch.testing.assert_close(logits[row,pos].float().logsumexp(0),out[row,pos].logsumexp(0))
    (-out[0,1,ids[1]]).backward()
    assert scores.grad[2].abs().sum()>0
    assert torch.equal(scores.grad[1],torch.zeros(4))
    assert apply_phase_evidence(logits,scores,queries,ids,strength=0) is logits


def test_native_rate_clock_interpolation_handles_midbeat_bpm_and_mixed_phases():
    data,song,_=fixture();anchors=[0,2,3]
    batch=audio_tiles(song,anchors,data.onset_phases)
    for row,a in enumerate(anchors):
        origin=np.floor(song.clock.at_floats([a])[0]*100)-15
        reconstructed=(batch['onset_positions'][row].numpy()+origin)/100
        np.testing.assert_allclose(reconstructed,song.clock.at_floats(a+data.onset_phases),atol=2e-7)
    assert float(F(1,6)) in data.onset_phases and float(F(1,4)) in data.onset_phases
    assert float(F(1,101)) in data.onset_phases


def test_zero_initialization_and_disabled_readout_preserve_original_model():
    data,song,base,model=pair()
    targets=[EventPoint(F(7,3),'1'),EventPoint(F(11,4),'2')]
    item=data.build_stream(song,[],F(0),target=targets,ended=True,training=True)
    batch=collate_stream([item])
    with torch.inference_mode():
        original=base(batch)['logits'];zero=model(batch)['logits']
        assert torch.equal(original,zero)
        model.onset_readout.score.weight.normal_(0,.2)
        off=model({**batch,'onset_strength':0})['logits']
        assert torch.equal(original,off)
        on=model(batch)['logits']
        queries={(r,p) for r,p,_ in batch['phase_queries_cpu']}
        for pos in range(original.size(1)):
            if (0,pos) not in queries:assert torch.equal(original[0,pos],on[0,pos])
    model.train();loss,_=model.loss(model(batch),batch);loss.backward()
    assert torch.isfinite(loss)
    assert model.onset_readout.score.weight.grad.abs().sum()>0
    assert model.onset_readout.encoder[0].weight.grad.abs().sum()>0


def test_future_chart_not_used_and_incremental_generation_matches_training():
    data,song,_,model=pair()
    with torch.no_grad():model.onset_readout.score.weight.normal_(0,.1)
    targets=[EventPoint(F(7,3),'1'),EventPoint(F(13,4),'2')]
    one=data.build_stream(song,[],F(0),target=targets,training=True)
    changed=data.build_stream(song,[],F(0),target=[EventPoint(F(9,4),'2'),targets[1]],training=True)
    before=one['phase_queries_cpu'][0][1]
    with torch.inference_mode():
        first=model(collate_stream([one]))['logits']
        other=model(collate_stream([changed]))['logits']
        torch.testing.assert_close(first[0,before],other[0,before],atol=2e-6,rtol=2e-6)
        sample=EventSampler(model,data,song,[],'cpu',torch.Generator(),SimpleNamespace(temperature=.9,top_p=.95,onset_strength=1.))
        prefix=[]
        for target in targets:
            time=data.codec._time(target.beat)
            for i in range(len(time)-1):sample._logits(prefix+time[:i+1])
            result=sample._logits(prefix+time[:-1])
            pos=len(sample.prefix)+len(prefix)+len(time)-2
            torch.testing.assert_close(result,first[0,pos],atol=4e-6,rtol=4e-6)
            prefix.extend(data.codec.encode_points([target]))


def test_batch_padding_does_not_change_shared_phase_audio():
    data,song,_,model=pair()
    one=data.build_stream(song,[],F(0),target=[EventPoint(F(1,6),'1')],training=True)
    two=data.build_stream(song,[],F(0),target=[EventPoint(F(10,3),'2')],training=True)
    with torch.no_grad():model.onset_readout.score.weight.normal_(0,.1)
    with torch.inference_mode():
        alone=model.encode_audio(collate_stream([one]))['onset_scores']
        together=model.encode_audio(collate_stream([one,two]))['onset_scores'][:len(one['anchors'])]
    torch.testing.assert_close(alone,together,atol=1e-6,rtol=1e-6)
