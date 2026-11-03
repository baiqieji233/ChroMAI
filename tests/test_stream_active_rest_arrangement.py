"""Active musical rests are learned; WAIT tokens never become rest evidence."""
import copy
import math
import random
from fractions import Fraction as F
from types import SimpleNamespace

import pytest
import torch

from maichart.composer.stream_arrangement import PhraseArrangementRate
from maichart.composer.stream_phrase import PHRASE_FEATURE_DIM
from maichart.composer.stream_train import upgrade_learning_stage
from maichart.composer.stream_data import NOTE, WAIT, EOS, StreamDataset, collate_stream
from maichart.composer.stream_generate import _Curve
from maichart.composer import stream_augmentation as aug
from maichart.event_timing import EventPoint,EventTempo


def inputs(batch=1,length=2):
    shape=(batch,length)
    data=dict(condition=torch.zeros(batch,3),phrase_start_beat=torch.full(shape,8.),
        phrase_end_beat=torch.full(shape,16.),step_beat=torch.full(shape,12.),
        audio_end_beat=torch.full((batch,),32.),phrase_progress=torch.zeros(*shape,PHRASE_FEATURE_DIM),
        phrase_memory_features=torch.zeros(*shape,4,PHRASE_FEATURE_DIM),
        phrase_memory_start_beat=torch.zeros(*shape,4),phrase_memory_end_beat=torch.zeros(*shape,4),
        phrase_memory_mask=torch.zeros(*shape,4,dtype=torch.bool))
    return {'phrase_music_anchor':torch.ones(*shape,16)},data


def make_nonzero(module):
    with torch.no_grad():
        module.envelope.weight.normal_(std=.1)
        module.closing[-1].weight.normal_(std=.1)


def test_no_wait_count_gap_or_reset_motion_can_change_the_arrangement():
    torch.manual_seed(3);module=PhraseArrangementRate(16,3);make_nonzero(module)
    phrase,batch=inputs()
    memory=batch['phrase_memory_features'].reshape(1,2,4,4,24)
    memory[:,:,0,0,23]=math.log(4.)
    memory[:,:,0,0,0]=math.log(4.)
    batch['phrase_memory_mask'][:,:,0]=True
    batch['phrase_memory_end_beat'][:,:,0]=8.
    batch['phrase_progress'].reshape(1,2,4,24)[:,:,0,23]=math.log(3.)
    before=module(phrase,batch)
    changed=copy.deepcopy(batch)
    for name in ('phrase_memory_features','phrase_progress'):
        shaped=changed[name].reshape(*changed[name].shape[:-1],4,24)
        shaped[...,13:22]=999.
    changed['history_kind']=torch.full((1,200),WAIT)
    changed['step_index']=torch.full((1,2),10000)
    after=module(phrase,changed)
    torch.testing.assert_close(before['gain'],after['gain'],atol=0,rtol=0)
    torch.testing.assert_close(before['intent'],after['intent'],atol=0,rtol=0)


def test_empty_and_stale_reference_histories_return_to_music_not_a_silent_copy_prior():
    torch.manual_seed(5);module=PhraseArrangementRate(16,3);make_nonzero(module)
    phrase,batch=inputs()
    base=module(phrase,batch)
    batch['phrase_memory_mask'][:,:,0]=True
    batch['phrase_memory_start_beat'][:,:,0]=-4.
    batch['phrase_memory_end_beat'][:,:,0]=0.
    # A long WAIT-only reference has no NOTE evidence.
    batch['phrase_memory_features'].reshape(1,2,4,4,24)[:,:,0,:,20]=10.
    torch.testing.assert_close(module(phrase,batch)['gain'],base['gain'],rtol=0,atol=0)
    batch['phrase_memory_features'].reshape(1,2,4,4,24)[:,:,0,0,23]=math.log(5.)
    fresh=module(phrase,batch)
    batch['phrase_memory_start_beat'][:,:,0]=-10004.
    batch['phrase_memory_end_beat'][:,:,0]=-10000.
    stale=module(phrase,batch)
    assert fresh['history_support'].min()>0 and stale['history_support'].max()==0
    torch.testing.assert_close(stale['gain'],base['gain'],rtol=0,atol=0)


def test_phrase_intent_is_sticky_but_actual_written_notes_can_change_closing():
    module=PhraseArrangementRate(16,3);make_nonzero(module)
    phrase,batch=inputs()
    before=module(phrase,batch)
    batch['step_beat']+=2
    batch['phrase_progress'].reshape(1,2,4,24)[:,:,2,23]=math.log(4.)
    after=module(phrase,batch)
    torch.testing.assert_close(before['intent'],after['intent'],rtol=0,atol=0)
    assert not torch.equal(before['gain'],after['gain'])


def test_same_music_can_learn_a_deliberate_rest_from_real_note_context():
    torch.manual_seed(7);module=PhraseArrangementRate(16,3)
    phrase,batch=inputs(batch=2,length=1)
    batch['phrase_progress'].reshape(2,1,4,24)[0,0,2,23]=math.log(5.)
    optimizer=torch.optim.Adam(module.parameters(),lr=.015)
    for _ in range(70):
        gain=module(phrase,batch)['gain'][:,0]
        expected=4*gain
        loss=expected[0]-(-torch.expm1(-expected[1])).clamp_min(1e-8).log()
        optimizer.zero_grad();loss.backward();optimizer.step()
    probability=-torch.expm1(-4*module(phrase,batch)['gain'][:,0])
    assert probability[0]<.2 and probability[1]>.8


def model_fixture():
    from test_composer_stream_model import fixture
    parent,batch=fixture(timing_output_policy='absolute_grid_v1')
    base=upgrade_learning_stage(parent,dict(plan_kind_policy='audio_hazard_v3',phrase_context_policy='causal_phrase_v1')).eval()
    shape=batch['step_beat'].shape
    batch.update(phrase_start_beat=torch.zeros(shape),phrase_end_beat=torch.full(shape,4.),
        phrase_progress=torch.zeros(*shape,PHRASE_FEATURE_DIM),
        phrase_memory_features=torch.zeros(*shape,4,PHRASE_FEATURE_DIM),
        phrase_memory_start_beat=torch.zeros(*shape,4),phrase_memory_end_beat=torch.zeros(*shape,4),
        phrase_memory_mask=torch.zeros(*shape,4,dtype=torch.bool),
        target_phrase_features=torch.zeros(*shape,PHRASE_FEATURE_DIM),
        target_phrase_mask=torch.zeros(shape,dtype=torch.bool),target_phrase_weight=torch.zeros(shape))
    arranged=upgrade_learning_stage(base,dict(plan_kind_policy='phrase_arranged_hazard_v4')).eval()
    return base,arranged,batch


def test_new_interface_starts_at_parent_probabilities_and_receives_real_rest_gradients():
    base,model,batch=model_fixture()
    a,b=base(batch),model(batch)
    for name in ('kind_logits','delta_logits','shape_logits','phase_logits'):
        torch.testing.assert_close(a[name],b[name],rtol=0,atol=0)
    loss,_=model.loss(b,batch);loss.backward()
    assert torch.isfinite(loss)
    assert model.occurrence_arrangement.envelope.weight.grad.abs().sum()>0
    altered=copy.deepcopy(batch)
    altered['target_phrase_features'].fill_(float('nan'))
    altered['target_second']+=100
    other=model(altered)
    torch.testing.assert_close(b['kind_logits'],other['kind_logits'],rtol=0,atol=0)
    torch.testing.assert_close(b['delta_logits'],other['delta_logits'],rtol=0,atol=0)


def test_closing_rest_changes_onset_mass_but_does_not_close_the_next_phrase_entry():
    base,model,batch=model_fixture()
    with torch.no_grad():
        for parent in (base,model):
            parent.plan_head.weight.zero_();parent.plan_head.bias[2]=math.log(math.expm1(.7))
        model.occurrence_arrangement.envelope.bias.fill_(-8.)
    before,after=base(batch),model(batch)
    gain=float(after['occurrence_arrangement_gain'][0,2])
    # Cursor 3 -> phrase boundary 4 -> horizon 5: only the first half closes.
    expected=.35*(1+gain)
    assert after['kind_logits'][0,2].softmax(-1)[NOTE].item()==pytest.approx(-math.expm1(-expected),abs=2e-6)
    candidates=after['candidate_beats'][0,2]
    inside=(candidates>3)&(candidates<4)
    assert after['delta_logits'][0,2].softmax(-1)[inside].sum()<before['delta_logits'][0,2].softmax(-1)[inside].sum()*.1
    outside=(candidates>=4)&(candidates<6)
    change=after['delta_logits'][0,2]-before['delta_logits'][0,2]
    torch.testing.assert_close(change[outside],torch.full_like(change[outside],(1-gain)*.35),rtol=0,atol=2e-6)


def test_curve_uses_the_arranged_interval_for_both_kind_and_onset_corrections():
    import numpy as np
    curve=object.__new__(_Curve);curve.active=True;curve.duration=2.
    curve.hazard_edges=np.array([0.,1.,2.]);curve.native_hazard=np.array([2.,3.])
    curve.adjusted_hazard=2*curve.native_hazard
    curve.native_prefix=np.array([0.,2.,5.]);curve.adjusted_prefix=2*curve.native_prefix
    options=dict(arrangement_gain=.1,arrangement_end_second=.5)
    note,tail=curve.kind_bias(0.,1.5,**options)
    assert note==pytest.approx(math.log(-math.expm1(-5.2))-math.log(-math.expm1(-2.6)))
    assert tail==pytest.approx(-2.6)
    assert curve.timing_bias(0.,np.array([1.5]),**options)[0]==pytest.approx(math.log(2)-2.6)


def test_authored_internal_rest_keeps_its_true_writing_and_music_and_budget(monkeypatch):
    from test_stream_augmentation import dataset
    original=dataset(monkeypatch)
    bank=original._bank(0);bank[0][:,198]=1.
    song=SimpleNamespace(audio=SimpleNamespace(duration_sec=12.),tempo=[EventTempo(F(0),120.)],first=0.)
    points=[EventPoint(F(i),str(i%8+1)) for i in (0,1,2,3,10,11,12,13,14,15,16)]
    base=SimpleNamespace(songs=[(song,points,12.,1.,None,torch.zeros(16),(),None)])
    ds=StreamDataset(base=base,lead_in=8,targets=4,audio_grid_seconds=.05,
        history_window_policy='block_aligned_v2',continuation_policy='pattern_runs_v2')
    before=ds[1]
    row,mode,_=aug.augment_stream_row(ds,1,rng=random.Random(5),mode='authored_rest',
        row=before,input_policy=aug.MUSICAL_ARRANGEMENT_INPUT_POLICY)
    assert mode=='authored_rest' and row['_authored_rest_internal']
    assert row['_authored_rest_rms']==pytest.approx(1.) and row['target_kind'][-1]==WAIT
    target=next(s for s in ds.streams[0] if float(s.start)==row['step_beat'][-1] and s.kind==WAIT)
    assert row['target_beat'][-1]==float(target.end)
    assert row['history_atoms'].ne(0).any()
    # The fast unchanged-prefix path must agree with a full causal replay.
    from maichart.composer.stream_data import recovery_sample
    at=next(i for i,s in enumerate(ds.streams[0]) if s is target)
    replay=recovery_sample(ds,0,list(ds.streams[0][:at]),target,configuration_trusted=False)
    torch.testing.assert_close(row['action_state'][-1],replay['action_state'][-1])
    torch.testing.assert_close(row['phrase_progress'][-1],replay['phrase_progress'][-1]) if 'phrase_progress' in row else None
    torch.testing.assert_close(row['sample_weight'].sum(),before['sample_weight'].sum(),rtol=0,atol=0)
    after=collate_stream([row]);old=collate_stream([before])
    assert after['authored_rest_row_mask'].tolist()==[True]
    torch.testing.assert_close(after['audio_plan_weight'],old['audio_plan_weight'],rtol=0,atol=0)
