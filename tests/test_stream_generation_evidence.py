"""Authored candidate evidence, bounded cycles and unchanged training math."""
from copy import deepcopy
from fractions import Fraction as F
from types import SimpleNamespace
import math

import numpy as np
import pytest
import torch

from maichart.stream.decoding import eos_probability,validate_options
from maichart.stream.generate import parser
from maichart.stream.grounded_model import GroundedContext,relative_features,ALIGNMENT_BACK,ALIGNMENT_WINDOW
from maichart.stream.position_evidence import PositionEvidence
from maichart.stream.repetition import RepetitionGuard,RepeatBias
from maichart.stream.section_profile import SectionTypeProfile
from test_stream_grounding import config,cpu_threads,source


def test_eos_temperature_is_independent_and_records_actual_sampling_probability():
    args=parser().parse_args(['--audio','a','--checkpoint','c','--output','o','--position-temperature','0.15'])
    validate_options(args)
    assert args.eos_temperature==1.
    assert eos_probability(-1.2,args.eos_temperature)==pytest.approx(.2314752165)
    assert eos_probability(-1.2,args.position_temperature)<.0004
    assert eos_probability(-10000.,1.)==0 and eos_probability(10000.,1.)==1
    assert eos_probability(-1.,0.)==0 and eos_probability(1.,0.)==1


@pytest.mark.parametrize('name,value',[('eos_temperature',float('nan')),('position_audio_strength',-1),
    ('repetition_strength',float('inf')),('repetition_min_seconds',0),('repetition_min_beats',121)])
def test_generation_evidence_controls_reject_invalid_values(name,value):
    args=parser().parse_args(['--audio','a','--checkpoint','c','--output','o'])
    setattr(args,name,value)
    with pytest.raises(ValueError,match=name):validate_options(args)


def test_pointer_interpolates_without_quantizing_and_is_bounded_or_inactive_when_uncertain():
    pointer=PositionEvidence([0.,.2,1.],[-4.,0.,-4.],[.1,.8,.1],strength=.3)
    assert pointer.cost(.2)==0 and 0<pointer.cost(float(F(1,3)))<=.6
    assert pointer.cost(-1.)==pointer.cost(20.)==0  # no forced rest shortening
    assert pointer.cost(.25)!=pointer.cost(.3)  # not snapped to the nearest node
    flat=PositionEvidence([0.,.2,1.],[0.,0.,0.],[.1,.8,.1])
    assert flat.cost(.5)==0
    no_selection=PositionEvidence([0.,.2,1.],[-4.,0.,-4.],[.1,.1,.1])
    assert no_selection.cost(.5)==0
    invalid=PositionEvidence([-1.,0.,1.],[-1e4,-1e4,-1e4],[.5,.5,.5])
    assert invalid.cost(.5)==0


def frozen_alignment(module,hidden,batch):
    """Pre-refactor training law, independently kept for values and gradients."""
    indices=batch['alignment_event_indices'].clamp_min(0)
    query=module.alignment_query(hidden.gather(1,indices[...,None].expand(-1,-1,hidden.size(-1))))
    rows=batch['decoder_query_index'].clamp_min(0).gather(1,indices)
    seconds=batch['query_seconds'].gather(1,rows)
    bpm=(batch['query_clock'][...,19].gather(1,rows).float()*300.).clamp_min(1.)
    times=batch['detail_times_sec'];padding=batch['detail_audio_padding_mask'].bool()
    total=times.size(1);width=min(ALIGNMENT_WINDOW,total)
    start=(torch.searchsorted(times.contiguous(),seconds.contiguous())-ALIGNMENT_BACK).clamp(0,max(0,total-width))
    candidates=start[...,None]+torch.arange(width)
    batch_rows=torch.arange(times.size(0))[:,None,None]
    keys=module.alignment_key(batch['detail_audio'])[batch_rows,candidates]
    delta=times[batch_rows,candidates]-seconds[...,None]
    invalid=padding[batch_rows,candidates]
    relative=relative_features(delta.masked_fill(invalid,0.).float(),bpm[...,None],4.).to(keys.dtype)
    keys=keys+module.alignment_distance(relative)
    scores=(query[:,:,None].float()*keys.float()).sum(-1)/math.sqrt(keys.size(-1))
    scores=scores.masked_fill(invalid,-1e4)
    labels=batch['alignment_audio_index']-start
    labels=torch.where((labels>=0)&(labels<width),labels,torch.full_like(labels,-1))
    control=batch['condition'][:,None].expand(-1,batch['detail_audio'].size(1),-1).clone()
    control[...,1:]=torch.where(batch['condition_null'][:,None,None],.5,control[...,1:])
    evidence=module.local_value(batch['detail_audio'])
    selection=module.selection_head(torch.cat((evidence,control.to(evidence)),-1)).squeeze(-1)
    return {'alignment_event_logits':scores,'alignment_event_labels':labels,'position_alignment_logits':selection}


def test_shared_pointer_preserves_training_logits_and_parameter_gradients_exactly():
    torch.manual_seed(191)
    module=GroundedContext(config());reference=deepcopy(module)
    hidden=torch.randn(2,12,16)
    batch={'alignment_event_indices':torch.tensor([[0,5,10],[0,5,10]]),
        'decoder_query_index':torch.arange(12).repeat(2,1),'query_seconds':torch.linspace(0,4,12).repeat(2,1),
        'query_clock':torch.randn(2,12,20),'detail_times_sec':torch.linspace(-.2,5,150).repeat(2,1),
        'detail_audio_padding_mask':torch.zeros(2,150,dtype=torch.bool),
        'detail_audio':torch.randn(2,150,120),'condition':torch.randn(2,16),
        'condition_null':torch.tensor([False,True]),'alignment_audio_index':torch.tensor([[1,40,149],[1,40,149]])}
    batch['detail_audio_padding_mask'][:,:3]=True
    actual=module.alignment(hidden,batch);expected=frozen_alignment(reference,hidden,batch)
    for key in actual:torch.testing.assert_close(actual[key],expected[key],rtol=0,atol=0)
    def loss(output):
        return output['alignment_event_logits'].masked_fill(output['alignment_event_logits']<-1e3,0).square().mean()+output['position_alignment_logits'].square().mean()
    loss(actual).backward();loss(expected).backward()
    for (name,a),(_,b) in zip(module.named_parameters(),reference.named_parameters()):
        assert (a.grad is None)==(b.grad is None),name
        if a.grad is not None:torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0,msg=name)


def test_inference_pointer_keeps_committed_clock_and_reuses_static_keys(source,tmp_path,monkeypatch):
    from maichart.stream.data import StreamDataset,collate_stream
    from maichart.stream.model import ChartStreamTransformer
    from maichart.stream.grounded_inference import GroundedInference
    dataset=StreamDataset(source[1],cache_dir=tmp_path/'cache',max_tokens=32,max_history_tokens=12,
        grounding_config=config(),preparation_workers=1,audio_cache_gb=.05)
    batch=collate_stream([dataset[0]]);model=ChartStreamTransformer(config()).eval()
    with torch.inference_mode():
        encoded=model.encode_audio(batch['audio'],batch['condition'],audio_padding_mask=batch['audio_padding_mask'])
        song=dataset.base.songs[0][0]
        session=GroundedInference(model,dataset.codec,song,encoded,audio_times=batch['audio_times_sec'][0].numpy())
        session.begin_context([],[]);session.set_prefix([dataset.codec.bos_id])
        memory,mask=model.encode(batch['audio'],batch['condition'],encoded_audio=encoded)
        boundary,_=session.decode_step(torch.tensor([[dataset.codec.bos_id]]),0,memory,mask)
        original=model.grounding.alignment_key.forward;calls=[]
        def counted(*args):calls.append(1);return original(*args)
        monkeypatch.setattr(model.grounding.alignment_key,'forward',counted)
        first=session.position_evidence(boundary['hidden'],batch['condition'],strength=.25)
        second=session.position_evidence(boundary['hidden'],batch['condition'],strength=.25)
    np.testing.assert_array_equal(first.scores,second.scores)
    assert len(calls)==1 and session.clock.committed==0 and session.clock.beat==F(0)
    assert not session.points and session.clock_position==1


def test_section_time_integrals_ignore_nonuniform_node_counts_and_duplicate_clocks():
    times=np.array([0.,2.,10.]);curve=np.array([[.1,.8,.1,.1],[.5,.4,.3,.2],[.9,.2,.6,.4]])
    original=SectionTypeProfile(curve,times)
    fine=np.r_[np.linspace(0,2,101),[2.,10.]]
    values=np.stack([np.interp(fine,times,curve[:,i]) for i in range(4)],1)
    refined=SectionTypeProfile(values,fine)
    assert refined.mean_density==pytest.approx(original.mean_density)
    np.testing.assert_allclose(refined.reference,original.reference,rtol=1e-12,atol=1e-12)
    for second in (-2.,0.,1.,4.,10.,15.):
        np.testing.assert_allclose(refined(second),original(second),rtol=1e-12,atol=1e-12)
        assert refined.density_factor(second)==pytest.approx(original.density_factor(second))
    constant=SectionTypeProfile(np.tile([.2,.5,.2,.1],(3,1)),[0.,.1,10.])
    assert constant.head_rate(5.)==pytest.approx(2.)


class NeutralBias:
    def terminal_score(self,text):return 0.
    def biases(self,text,continuations):return [0.]*len(continuations)
    def token_bias(self,text,continuations):return [0.]*len(continuations)


def repeated(pattern,count,*,rate=2.):
    guard=RepetitionGuard(SimpleNamespace(head_rate=lambda _:rate))
    for i in range(count):guard.commit(pattern[i%len(pattern)],F(i,2),i*60/190/2)
    return guard,guard.event(NeutralBias(),F(count,2),count*60/190/2)


def test_repeat_guard_separates_short_trills_legitimate_load_and_long_overloaded_cycles():
    short,bias=repeated(['1','8'],12)
    assert not bias.cases and bias.terminal_score('1')==0
    intended,bias=repeated(['4'],72,rate=8.)
    assert not bias.cases
    long,bias=repeated(['4'],72)
    assert bias.terminal_score('4')<bias.terminal_score('3')==0
    assert -1.2<=bias.terminal_score('4')<0 and long.report()['hard_rejections']==0
    varied,bias=repeated(['3/5','4','2/5','3'],100,rate=2.)
    assert bias.terminal_score('1/7')<bias.terminal_score('1')  # structural chord/single, with lane variation
    assert len(varied.rows)==100  # candidate scoring never commits a proposed event
    assert all(not x for x in short.report()['diagnostics'])


def test_repetition_prefix_potential_and_terminal_preference_are_consistent():
    bias=RepeatBias(NeutralBias(),(('4',False,1.),))
    assert bias.biases('',list('34'))==[0.,-1.]
    total=bias.biases('',['4'])[0]+bias.biases('4',[None])[0]
    assert total==bias.terminal_score('4')==-1.
    total=bias.biases('',['4'])[0]+bias.biases('4',['h'])[0]+bias.biases('4h',[None])[0]
    assert total==bias.terminal_score('4h')==0.
