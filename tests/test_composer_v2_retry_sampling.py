from fractions import Fraction as F
from pathlib import Path
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from maichart.composer_v2.retry_sampling import RejectedEventTree,RetrySpaceExhausted
from maichart.composer_v2.diagnostics import write_failure_snapshot


def test_exclusion_removes_only_failed_complete_answer_and_keeps_other_same_time_answer():
    tree=RejectedEventTree()
    tree.remember([], (0,1), torch.tensor([.9,.1,1.,1.]).log())
    tree.remember([0], (2,3), torch.tensor([1.,1.,.8,.2]).log())
    assert tree.reject([0,2])
    logits,allowed,scaled=tree.condition([], (0,1),1.)
    probabilities=torch.softmax(logits[list(allowed)],0)
    torch.testing.assert_close(probabilities,torch.tensor([.18/.28,.1/.28]))
    assert 0 in allowed  # Another configuration at this same time survives.
    _,remaining,_=tree.condition([0],(2,3),1.)
    assert remaining==(3,)
    assert not tree.reject([0,2])


def test_near_certain_failed_sequence_does_not_cancel_remaining_probability_to_zero():
    tree=RejectedEventTree()
    tree.remember([], (0,1),torch.tensor([1000.,0.]))
    tree.remember([0], (0,1),torch.tensor([1000.,0.]))
    tree.reject([0,0])
    logits,allowed,_=tree.condition([], (0,1),.3)
    torch.testing.assert_close(torch.softmax(logits[list(allowed)],0),torch.tensor([.5,.5]))


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
def test_conditioning_does_not_requantize_half_precision_probability_mass(dtype):
    tree=RejectedEventTree()
    root=torch.tensor([.9,.1,1.,1.]).log().to(dtype)
    detail=torch.tensor([1.,1.,.8,.2]).log().to(dtype)
    tree.remember([], (0,1),root);tree.remember([0], (2,3),detail)
    tree.reject([0,2])
    logits,allowed,_=tree.condition([], (0,1),1.)
    expected=torch.softmax(root[:2].float(),0)
    expected[0]*=torch.softmax(detail[2:].float(),0)[1];expected/=expected.sum()
    torch.testing.assert_close(torch.softmax(logits[list(allowed)].float(),0),expected,atol=1e-6,rtol=1e-6)


def test_exhausted_single_subtree_is_reported_without_inventing_an_end_token():
    tree=RejectedEventTree();tree.remember([], (0,),torch.tensor([0.]))
    tree.reject([0])
    with pytest.raises(RetrySpaceExhausted):tree.condition([], (0,),1.)


def test_unobserved_alternative_is_not_blacklisted_without_its_model_path():
    tree=RejectedEventTree();tree.remember([], (0,1),torch.tensor([0.,0.]))
    assert not tree.reject([0,1])
    assert tree.rejections==0
    assert tree.condition([], (0,1),1.)[2] is False


def test_failure_snapshots_survive_reusing_the_same_user_output_path(tmp_path):
    output=tmp_path/'maidata.txt'
    first=write_failure_snapshot(output,dict(after='221',checkpoint_info={'optimizer_steps':10410}))
    before=first.read_bytes()
    second=write_failure_snapshot(output,dict(after='593/2',checkpoint_info={'optimizer_steps':10500}))
    assert first!=second and first.read_bytes()==before
    latest=json.loads(output.with_suffix('.failure.json').read_text(encoding='utf-8'))
    assert latest['failure_snapshot_path']==str(second)
    assert json.loads(first.read_text(encoding='utf-8'))['after']=='221'


def test_cached_decoder_replays_the_actual_new_branch_before_any_uncached_query(monkeypatch):
    from test_composer_v2_continuous import setup
    from maichart.composer_v2.model import ComposerV2,ComposerV2Config
    from maichart.composer_v2.stream_generate import EventSampler
    from maichart.event_timing import EventPoint
    data,song=setup()
    model=ComposerV2(ComposerV2Config(d_model=32,nhead=4,layers=1,dim_feedforward=64,
        dropout=0.,audio_layers=1,song_layers=1,max_tokens=256),data.codec).eval()
    calls=[];decode=model.decode
    def checked(batch,cache):
        got=decode(batch,cache)
        full=decode(batch,cache,incremental=False)
        torch.testing.assert_close(got[0,-1],full[0,-1],atol=4e-6,rtol=4e-5)
        calls.append(1);return got
    monkeypatch.setattr(model,'decode',checked)
    args=SimpleNamespace(temperature=.9,top_p=.95)
    with torch.inference_mode():
        sampler=EventSampler(model,data,song,[],'cpu',torch.Generator().manual_seed(0),args)
        sampler._retry_tree=RejectedEventTree();sampler._retry_base=[];sampler._retry_condition=False
        def feed(text):
            ids=data.codec.encode_points([EventPoint(F(1),text)])
            sampler._rewind([]);sampler._retry_cache_dirty=False
            for i,token in enumerate(ids):
                sampler._free_choices=(token,)
                sampler.next(ids[:i],(token,))
        feed('1');first=len(calls)
        feed('1');assert len(calls)==first
        feed('2');assert first<len(calls)<first*2
        assert sampler.retry_stats['decoder_cache_hits']>0


def test_physical_failure_cannot_finish_the_chart_by_retrying_eos(tmp_path,monkeypatch):
    import maichart.composer_v2.stream_generate as generation
    from maichart.composer_v2.stream_codec import StreamCodec
    from maichart.event_timing import EventPoint,EventTempo
    codec=StreamCodec.fit(['1'])
    class Audio:
        duration_sec=100.
        identity={'fixture':True}
        def summary(self,n):return {'times_sec':np.asarray([0.,100.])}
    class Sampler:
        calls=0
        def __init__(self,*args):self.cursor=F(0)
        def propose(self,*args,**kwargs):
            Sampler.calls+=1
            if Sampler.calls==1:return codec.encode_points([EventPoint(F(1),'1/1')])
            return [codec.eos_id]
        def _rewind(self,target):pass
    monkeypatch.setattr(generation,'EventSampler',Sampler)
    model=SimpleNamespace(codec=codec,config=SimpleNamespace(max_tokens=2048,history_beats=16,
        window_beats=16,future_beats=4,refresh_events=16,ticks_per_beat=48,sequence_policy='continuous_events_v1'))
    args=SimpleNamespace(temperature=.3,top_p=.95,window_retries=3,candidates=1,rollback_windows=2,
        level=14.5,seed=0,max_events=0,checkpoint='untouched.pt',output=str(tmp_path/'maidata.txt'))
    with pytest.raises(RuntimeError,match='recovery exhausted'):
        generation.generate_continuous(model,Audio(),tmp_path/'song.mp3',[EventTempo(F(0),190.)],0.,{},
            np.zeros(16,'float32'),np.zeros(16,bool),{},args,torch.Generator(),'cpu',F(200))
    assert not (tmp_path/'maidata.txt').exists()
    latest=json.loads((tmp_path/'maidata.failure.json').read_text(encoding='utf-8'))
    assert any(r['reason']=='eos_is_not_physical_recovery' for r in latest['repairs'])
    assert Path(latest['failure_snapshot_path']).exists()
