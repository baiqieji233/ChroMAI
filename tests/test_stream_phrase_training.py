"""Whole-phrase target coverage and causal-label contracts (CPU only)."""
from fractions import Fraction
from types import SimpleNamespace
from collections import OrderedDict
import threading
from unittest.mock import patch

import numpy as np
import torch
import pytest

from maichart.composer.stream_phrase_training import (
    RELATION_DIM, collate_phrase_training, future_relation_rows, phrase_training_spans,normalize_phrase_plan_weights,
    phrase_training_blocks,CausalPhraseBlockIndex,
)


def step(kind, start, end, text=''):
    return SimpleNamespace(kind=kind,start=Fraction(start),end=Fraction(end),
                           start_second=float(start),end_second=float(end),text=text)


def test_phrase_partition_and_oversize_do_not_claim_exact_marginal():
    stream=[step(2,i,i+1,'1') for i in range(9)]
    spans=phrase_training_spans(stream,np.array([0.,4.,10.]),4)
    assert [i for lo,hi,*_ in spans for i in range(lo,hi)]==list(range(9))
    assert spans[0][3]
    assert all(not span[3] for span in spans[1:])
    assert {span[2] for span in spans[1:]}=={1}


def test_phrase_partition_uses_known_departure_not_target_arrival():
    stream=[step(2,0,1,'1'),step(2,1,3,'3'),step(2,3,4,'5')]
    spans=phrase_training_spans(stream,np.array([0.,2.,5.]),8)
    assert spans==[(0,2,0,True,0),(2,3,1,True,2)]


def test_relation_uses_physical_sets_and_preserves_symmetric_displacements():
    stream=[step(2,0,1,'1/5'),step(3,1,2),step(2,2,3,'3/7'),step(4,3,4)]
    target,mask=future_relation_rows(stream,[False]*4,0,4)
    assert target.shape==(4,2,RELATION_DIM)
    assert target[1,0,[2,6]].tolist()==[1.,1.]
    assert target[1,0,[10,14]].tolist()==[1.,1.]
    assert target[1,0,18]==1
    assert target[3,:,19].tolist()==[1.,1.]
    assert not mask[3,:,:19].any()


def test_unknown_future_masks_derived_labels():
    stream=[step(2,0,1,'1'),step(2,1,2,'2')]
    _,mask=future_relation_rows(stream,[False,True],0,2)
    assert not mask[0,1].any()
    assert not mask[1].any()


def test_collation_excludes_seam_from_mode_but_keeps_fractional_budget():
    item=dict(loss_mask=torch.tensor([False,True,True,True]),
              sample_weight=torch.tensor([0.,1.,.5,.5]),
              training_phrase_group=torch.tensor([0,0,0,-1]),
              training_phrase_position=torch.arange(4),
              training_phrase_complete=torch.tensor([False,True,True,False]),
              _training_target_budget=torch.tensor([0.,1.,.5,.5]))
    batch={}
    collate_phrase_training([item],batch)
    assert batch['training_phrase_group'].tolist()==[[-1,0,0,-1]]
    assert batch['training_phrase_group_weight'].tolist()==[1.5]
    assert batch['_loss_normalizer']==2.


def test_two_note_horizons_skip_wait_and_use_causal_cursor():
    stream=[step(2,0,1,'1'),step(3,1,2),step(2,2,3,'3'),step(2,3,4,'5')]
    target,mask=future_relation_rows(stream,[False]*4,1,2)
    assert target[0,0,2]==1 and target[0,1,4]==1
    assert target[0,:,18].tolist()==[1.,1.]
    assert torch.allclose(target[0,:,16],torch.log1p(torch.tensor([2.,3.])))
    assert mask[0,:,16].all()


def test_relation_song_cache_only_builds_once_and_returns_independent_slices():
    from maichart.composer import stream_data
    dataset=object.__new__(stream_data.StreamDataset)
    dataset.streams=[[step(2,0,1,'1'),step(2,1,2,'3')]]
    dataset.unknown_steps=[[False,False]]
    dataset._state_lock=threading.RLock()
    dataset._relation_cache=OrderedDict();dataset._relation_bytes=0;dataset._relation_budget=4096
    dataset._relation_hits=dataset._relation_misses=dataset._relation_evictions=0
    with patch.object(stream_data,'future_relation_rows',wraps=stream_data.future_relation_rows) as build:
        first,_=dataset._relation_rows(0,0,1)
        first.zero_()
        second,_=dataset._relation_rows(0,0,2)
        assert build.call_count==1
    assert second[0,0,0]==1
    assert dataset._relation_hits==1 and dataset._relation_misses==1
    dataset._relation_cache.clear();dataset._relation_bytes=0;dataset._relation_budget=128
    with patch.object(stream_data,'future_relation_index',wraps=stream_data.future_relation_index) as index_build:
        dataset._relation_rows(0,0,1)
        dataset._relation_rows(0,1,2)
        assert index_build.call_count==1
    assert dataset._relation_bytes<=128


def test_group_start_wait_remains_zero_and_available_for_mode_prior():
    item=dict(loss_mask=torch.tensor([False,True,True]),
              sample_weight=torch.tensor([0.,1.,1.]),
              configuration_mask=torch.tensor([True,True,True]),
              target_kind=torch.tensor([2,3,2]),
              training_phrase_group=torch.tensor([0,0,0]),
              training_phrase_position=torch.tensor([-1,0,1]),
              training_phrase_complete=torch.tensor([False,True,True]),
              _training_target_budget=torch.tensor([0.,1.,1.]))
    output={};collate_phrase_training([item],output)
    assert output['training_phrase_position'][0,1]==0
    assert output['training_phrase_complete'][0,1]
    assert output['training_phrase_group'][0,1]==0


@pytest.mark.parametrize('maximum',[4,64])
def test_phrase_primary_history_matches_incremental_and_replayed_prefix(monkeypatch,maximum):
    from maichart.event_timing import EventPoint,EventTempo
    from maichart.composer.stream_data import StreamDataset,StreamInferenceContext,EOS
    points=[EventPoint(Fraction(i),str(i%8+1)) for i in range(12)]
    points+=[EventPoint(Fraction(19),'3'),EventPoint(Fraction(22),'5')]
    song=SimpleNamespace(audio=SimpleNamespace(duration_sec=12.),tempo=[EventTempo(Fraction(0),120.)],first=0.)
    base=SimpleNamespace(songs=[(song,points,12.,1.,None,torch.zeros(16),(),None)])
    bank=(torch.zeros(48,238),torch.arange(48,dtype=torch.float64)/4,torch.arange(48,dtype=torch.float64)/2)
    monkeypatch.setattr(StreamDataset,'_bank',lambda self,song:bank)
    dataset=StreamDataset(base=base,lead_in=3,targets=5,history_window_policy='block_aligned_v2',
                          training_window_policy='phrase_complete_v1',max_phrase_events=maximum)
    def context():
        return StreamInferenceContext(song,dataset.codec,dataset.delta_vocab,dataset.conditions[0],
            lead_in=3,rollout_steps=5,history_window_policy='block_aligned_v2',audio_bank=bank,
            training_window_policy='phrase_complete_v1',max_phrase_events=maximum)
    live=context();primary={};observed_mass=torch.zeros(len(dataset.streams[0]));budget=0.
    for sample,(_,lo,hi,_) in enumerate(dataset.samples):
        row=dataset[sample]
        observed_mass.index_add_(0,row['step_index'],row['sample_weight'])
        budget+=float(row['_training_target_budget'].sum())
        for j in range(lo,hi):primary[j]=row
    expected_mass=torch.tensor([s.weight*dataset.target_window_weights[0][j]*dataset.chart_weights[0]
                                for j,s in enumerate(dataset.streams[0])])
    torch.testing.assert_close(observed_mass,expected_mass)
    assert budget==pytest.approx(len(dataset.streams[0]))
    assert sum(dataset.plan_weights)==pytest.approx(sum(dataset.target_window_weights[0])/dataset.targets)
    for j,target in enumerate(dataset.streams[0]):
        teacher=primary[j];first=int(teacher['step_index'][0]);length=j-first+1
        generated=live.current_batch()
        replay=context().batch(dataset.streams[0][:j],target.start)
        for key in ('history_kind','history_delta','history_shape','history_phase','history_phase_mask',
                    'step_beat','step_second','step_index','continuation_features'):
            torch.testing.assert_close(generated[key][0],teacher[key][:length])
            torch.testing.assert_close(replay[key],generated[key])
        if target.kind!=EOS:live.commit(target)


def test_plan_relative_budget_does_not_change_with_phrase_length():
    reference_targets=16
    for batch_count,target_budget in ((2,32.),(2,96.),(3,12.5)):
        items=[{'_training_reference_targets':reference_targets} for _ in range(batch_count)]
        weights=torch.arange(1,batch_count+1,dtype=torch.float32)[:,None]
        output=dict(_loss_normalizer=target_budget,audio_plan_weight=weights.clone(),
                    audio_plan_blank_weight=weights.clone(),target_phrase_weight=weights.clone())
        normalize_phrase_plan_weights(items,output)
        # The plan reducer divides by B. Its coefficient must instead equal
        # K/D against the same weighted numerator as the original plan rows.
        actual=output['audio_plan_weight'].sum()/batch_count
        expected=weights.sum()*reference_targets/target_budget
        torch.testing.assert_close(actual,expected)
        torch.testing.assert_close(output['audio_plan_weight'],output['audio_plan_blank_weight'])
        torch.testing.assert_close(output['audio_plan_weight'],output['target_phrase_weight'])
        assert output['_phrase_budget_scale']==batch_count*reference_targets/target_budget


def test_many_short_phrases_share_causal_blocks_without_changing_groups():
    stream=[step(2,i,i+1,str(i%8+1)) for i in range(120)]
    boundaries=np.arange(0.,125.,4.)
    spans,segments,positions,complete=phrase_training_blocks(stream,boundaries,16,64)
    assert len(spans)==8  # 30 phrases pack into seven K16 blocks and one tail.
    assert all(hi-lo<=64 for lo,hi in spans)
    assert complete.all()
    assert positions[:8].tolist()==[0,1,2,3,0,1,2,3]
    planner=CausalPhraseBlockIndex(boundaries,16,64)
    for i,target in enumerate(stream):
        anchor=planner.advance(i,target.start)
        assert anchor==next(lo for lo,hi in spans if lo<=i<hi)


def test_cap_split_marks_entire_audio_phrase_incomplete():
    stream=[step(2,i,i+1,'1') for i in range(20)]
    spans,segments,positions,complete=phrase_training_blocks(stream,np.array([0.,3.,20.]),8,10)
    assert spans==[(0,10),(10,20)]
    assert complete[:3].all()
    assert not complete[3:].any()


def test_collate_maps_multiple_phrase_groups_and_masks_only_untrusted_group():
    item=dict(loss_mask=torch.tensor([False,True,True,True,True,True]),
              sample_weight=torch.tensor([0.,1.,1.,1.,1.,.5]),
              configuration_mask=torch.tensor([True,True,False,True,True,True]),
              training_phrase_group=torch.tensor([2,2,2,3,3,-1]),
              training_phrase_position=torch.tensor([-1,0,1,0,1,2]),
              training_phrase_complete=torch.tensor([False,True,True,True,True,False]),
              _training_target_budget=torch.tensor([0.,1.,1.,1.,1.,.5]))
    output={};collate_phrase_training([item,item],output)
    assert output['training_phrase_group'].tolist()==[[-1,0,0,1,1,-1],[-1,2,2,3,3,-1]]
    assert output['training_phrase_complete'].tolist()==[[False,False,False,True,True,False]]*2
    assert output['training_phrase_group_weight'].tolist()==[2.,2.,2.,2.]
    assert output['_loss_normalizer']==9.
