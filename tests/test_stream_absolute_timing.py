from fractions import Fraction as Q
from types import SimpleNamespace

import pytest
import torch

from maichart.composer.stream_timing import (
    AbsoluteTimingHead, absolute_target_indices, current_timing_deltas,
    encode_history_gap, grouped_logsumexp, phase_group, temper_grid_logits,
)
from maichart.composer.stream_data import DeltaVocabulary, StreamInferenceContext, StreamStep, NOTE
from maichart.composer.stream_model import StreamComposerConfig
from maichart.composer.event_set_codec import EventSetCodec
from maichart.event_timing import EventTempo


def test_absolute_support_returns_from_a_48th_offset_without_a_delta_class():
    head=AbsoluteTimingHead(8,('0','0','1/2','1','2'),('0','1/4','1/2','3/4'))
    cursor=torch.tensor([[float(Q(9985,48))]],dtype=torch.float64)
    beats,gaps=head.candidates(cursor)
    target=torch.tensor([[208.25]],dtype=torch.float64)
    index=absolute_target_indices(beats,target,torch.ones(1,1,dtype=torch.bool))
    assert beats.gather(-1,index[...,None]).item()==208.25
    # The next authored quarter grid is reachable even though its corrective
    # interval is absent from the original flat categorical vocabulary.
    corrective=Q(833,4)-Q(9985,48)
    assert corrective==Q(11,48)
    vocabulary=DeltaVocabulary([0,Q(1,2),1,2])
    assert encode_history_gap(vocabulary,corrective,'absolute_grid_v1')==0
    with pytest.raises(ValueError,match='out-of-vocabulary'):
        encode_history_gap(vocabulary,corrective,'relative_delta_v1')
    model=SimpleNamespace(config=SimpleNamespace(timing_output_policy='absolute_grid_v1'),timing_grid=head)
    exact=current_timing_deltas(model,{},Q(9985,48))
    for i,gap in enumerate(exact):
        assert gaps[0,0,i].item()==pytest.approx(float(gap),abs=1e-12)


def test_temperature_preserves_group_decisions_independently_of_size():
    ids=torch.tensor([0,1,1,2,2,2])
    probabilities=torch.tensor([.6,.15,.15,.1/3,.1/3,.1/3])
    scores=temper_grid_logits(probabilities.log(),.65,ids)
    grouped=grouped_logsumexp(scores,ids).exp()[:3]
    expected=(torch.tensor([.6,.3,.1]).log()/.65).softmax(-1)
    torch.testing.assert_close(grouped,expected)
    torch.testing.assert_close(temper_grid_logits(probabilities.log(),1.,ids).exp(),probabilities)
    assert torch.isfinite(scores).all()


def test_grid_family_mass_and_gradient_do_not_depend_on_rare_gap_logits():
    head=AbsoluteTimingHead(8,('0','0','1/2','1','2'),('0','1/4','1/2','3/4'))
    with torch.no_grad():head.grid_head.bias[phase_group(Q(1,4))]=4
    cursor=torch.tensor([[float(Q(1,48))]],dtype=torch.float64)
    hidden=torch.ones(1,1,8,requires_grad=True)
    relative=torch.tensor([[[-50.,-40.,2.,1.,0.]]],requires_grad=True)
    values=torch.tensor([0.,0.,.5,1.,2.],dtype=torch.float64)
    beats,gaps=head.candidates(cursor)
    allowed=gaps.gt(0);allowed[...,0]=False
    query=torch.ones(1,1,32)
    logits,_,_=head(hidden,cursor,relative,values,query,torch.zeros_like(beats),allowed)
    mass=grouped_logsumexp(logits,head.group_ids).exp()
    assert mass[0,0,phase_group(Q(1,4))]>.9
    target=absolute_target_indices(beats,torch.tensor([[.25]],dtype=torch.float64),torch.ones(1,1,dtype=torch.bool))
    loss=-logits.gather(-1,target[...,None]).mean()
    loss.backward()
    assert head.grid_head.weight.grad.abs().sum()>0
    assert head.feature_key.weight.grad.abs().sum()>0
    assert torch.isfinite(relative.grad).all()
    assert torch.isfinite(hidden.grad).all()


def test_missing_authored_phase_is_explicit_and_initial_zero_uses_real_class():
    head=AbsoluteTimingHead(8,('0','1','2'))
    beats,_=head.candidates(torch.zeros(1,1,dtype=torch.float64))
    assert absolute_target_indices(beats,torch.zeros(1,1,dtype=torch.float64),torch.ones(1,1,dtype=torch.bool)).item()==1
    with pytest.raises(RuntimeError,match='absent from the absolute phase catalog'):
        absolute_target_indices(beats,torch.tensor([[.25]],dtype=torch.float64),torch.ones(1,1,dtype=torch.bool))
    # Padding and WAIT/EOS have no absolute NOTE target.
    absolute_target_indices(beats,torch.tensor([[.25]],dtype=torch.float64),torch.zeros(1,1,dtype=torch.bool))


def test_inference_keeps_exact_corrective_gap_with_no_false_old_embedding():
    song=SimpleNamespace(audio=SimpleNamespace(duration_sec=4.),tempo=[EventTempo(Q(0),120.)],first=0.)
    bank=(torch.zeros(80,238),torch.arange(80,dtype=torch.float64)/20,torch.arange(80,dtype=torch.float64)/10)
    codec=EventSetCodec.fit(['1'])
    context=StreamInferenceContext(song,codec,DeltaVocabulary([0,Q(1,2),1,2]),torch.zeros(16),
        audio_bank=bank,timing_output_policy='absolute_grid_v1')
    context.commit(StreamStep(NOTE,Q(0),Q(1,48),0.,1/96,'1'))
    row=context.current_batch()
    assert row['history_delta'][0,-1]==0
    assert row['history_kind'][0,-1]==NOTE
    assert row['history_gap_beats'][0,-1].item()==float(Q(1,48))
    assert row['step_beat'][0,-1].item()==float(Q(1,48))


def test_old_checkpoint_policy_stays_relative_and_catalog_is_configuration():
    assert StreamComposerConfig.from_dict({}).timing_output_policy=='relative_delta_v1'
    assert StreamComposerConfig.from_dict({}).timing_grid_audio_policy=='off'
    config=StreamComposerConfig(timing_phase_catalog=('0','1/3','1/4'))
    restored=StreamComposerConfig.from_dict(config.to_dict())
    assert restored.timing_output_policy=='absolute_grid_v1'
    assert restored.timing_phase_catalog==config.timing_phase_catalog


def test_direct_grid_audio_changes_family_mass_at_fixed_hidden_and_has_gradient():
    values=torch.tensor([0.,0.,.5,1.,2.],dtype=torch.float64)
    head=AbsoluteTimingHead(8,tuple(map(str,values.tolist())),('1/3','1/4'),audio_policy='candidate_pool_v1')
    cursor=torch.tensor([[.1]],dtype=torch.float64)
    beats,gaps=head.candidates(cursor)
    allowed=gaps.gt(0);allowed[...,0]=False
    hidden=torch.zeros(1,1,8)
    relative=torch.zeros(1,1,5)
    query=torch.zeros(1,1,32)
    within=torch.zeros_like(beats)
    keys=torch.zeros(*beats.shape,32)
    target=phase_group(Q(1,3));members=head.group_ids.eq(target)
    keys[...,members,0]=2.
    # Zero initialization preserves the prior result; the path then learns
    # a direct reaction to candidate music, without changing hidden states.
    initial,_,_=head(hidden,cursor,relative,values,query,within,allowed,candidate_audio=keys)
    with torch.no_grad():head.audio_projection.weight[0,0]=1.
    music=keys.clone().requires_grad_()
    changed,_,_=head(hidden,cursor,relative,values,query,within,allowed,candidate_audio=music)
    before=grouped_logsumexp(initial,head.group_ids).exp()
    after=grouped_logsumexp(changed,head.group_ids).exp()
    assert after[0,0,target]>before[0,0,target]
    # Audio is a macro-family readout, separate from the conditional law.
    torch.testing.assert_close(changed[...,members].softmax(-1),initial[...,members].softmax(-1))
    loss=-grouped_logsumexp(changed,head.group_ids)[...,target].mean()
    loss.backward()
    assert head.audio_projection.weight.grad.abs().sum()>0
    assert music.grad.abs().sum()>0
    assert torch.isfinite(music.grad).all()
    polluted=keys.clone();polluted[~allowed]=10000.
    ignored,_,_=head(hidden,cursor,relative,values,query,within,allowed,candidate_audio=polluted)
    torch.testing.assert_close(ignored,changed.detach())


def test_grid_audio_pool_is_normalized_by_family_candidate_count():
    values=torch.tensor([0.,0.,.5,1.,2.],dtype=torch.float64)
    masses=[]
    for phases in (('1/3','1/4'),('1/3','2/3','1/4')):
        head=AbsoluteTimingHead(8,tuple(map(str,values.tolist())),phases,audio_policy='candidate_pool_v1')
        with torch.no_grad():head.audio_projection.weight[0,0]=1.
        cursor=torch.tensor([[.1]],dtype=torch.float64)
        beats,gaps=head.candidates(cursor)
        allowed=gaps.gt(0);allowed[...,0]=False
        keys=torch.zeros(*beats.shape,32)
        keys[...,head.group_ids.eq(phase_group(Q(1,3))),0]=2.
        logits,_,_=head(torch.zeros(1,1,8),cursor,torch.zeros(1,1,5),values,
            torch.zeros(1,1,32),torch.zeros_like(beats),allowed,candidate_audio=keys)
        masses.append(grouped_logsumexp(logits,head.group_ids).exp())
    torch.testing.assert_close(masses[0],masses[1])
