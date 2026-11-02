from collections import Counter
from dataclasses import replace

import pytest
import torch

from maichart.composer import stream_augmentation as aug, stream_recovery as recovery
from maichart.composer.stream_data import NOTE,WAIT,EOS,collate_stream
from test_stream_musical_loop_recovery import LongestRotation
from test_stream_residual_loop_recovery import dense_dataset


class TimelineRandom(LongestRotation):
    def randrange(self,*args):return 10 if args==(2**31,) else 0


def capture_prefix(monkeypatch):
    captured=[]
    original=recovery.recovery_sample
    def sample(dataset,song,prefix,target,**kwargs):
        captured.append((song,list(prefix),target))
        return original(dataset,song,prefix,target,**kwargs)
    monkeypatch.setattr(recovery,'recovery_sample',sample)
    monkeypatch.setattr(recovery,'_guard',lambda *args:object())
    monkeypatch.setattr(aug,'_validate_window',lambda *args:None)
    return captured


def test_v7_is_nine_modes_and_keeps_v6_recipe_unchanged():
    old=aug._augmentation_modes(aug.MUSICAL_RESIDUAL_INPUT_POLICY)
    new=aug._augmentation_modes(aug.MUSICAL_BATCH_INPUT_POLICY)
    assert len(old)==8 and new==(*old,'timeline_loop')


@pytest.mark.parametrize('plan_scale',[None,.37])
def test_timeline_preserves_budgets_and_real_inference_layout(monkeypatch,plan_scale):
    ds=dense_dataset(monkeypatch);index=150;original=ds[index]
    original['sample_weight'][original['loss_mask']]=torch.tensor([.5,1.5,.75,2.25])
    if plan_scale is not None:original['_plan_scale']=plan_scale
    captured=capture_prefix(monkeypatch)
    # Weight already includes any chart scaling; the replacement must not
    # multiply it again through recovery_sample's ordinary target weighting.
    ds.chart_weights[0]=2.
    changed,mode,count=aug.augment_stream_row(ds,index,rng=TimelineRandom(0),mode='timeline_loop',
        row=original,input_policy=aug.MUSICAL_BATCH_INPUT_POLICY)
    assert mode=='timeline_loop' and count==128
    torch.testing.assert_close(changed['sample_weight'].sum(),original['sample_weight'].sum(),rtol=0,atol=0)
    assert changed['loss_mask'].sum()==1 and changed['sample_weight'].gt(0).sum()==1
    assert not changed.get('_augmentation_first_decision_rebalanced',False)
    for key in ('_sample_index','_window_weight','_song_bank','_song_plan','_literal_atom_types'):
        assert changed[key] is original[key]
    if plan_scale is None:assert '_plan_scale' not in changed
    else:assert changed['_plan_scale']==plan_scale
    old_batch=collate_stream([original]);new_batch=collate_stream([changed])
    for key in ('audio_plan_weight','audio_plan_blank_weight','audio_plan_target','audio_plan_exposure'):
        torch.testing.assert_close(old_batch[key],new_batch[key],rtol=0,atol=0)
    song,prefix,target=captured[0]
    raw=recovery._context(ds,song,original['_song_bank']).batch(prefix,target.start)
    for key in ('history_kind','history_atoms','history_shape','history_delta','history_gap_beats',
                'step_index','step_beat','step_second','continuation_features','action_state','action_state_mask'):
        torch.testing.assert_close(changed[key],raw[key][0])
    at=int((original['loss_mask']&original['sample_weight'].gt(0)).nonzero()[0])
    assert changed['step_index'][-1]!=original['step_index'][at]
    for key in ('target_kind','target_delta','target_shape','target_phase_mask','target_second','target_beat'):
        torch.testing.assert_close(changed[key][-1],original[key][at])
    torch.testing.assert_close(changed['condition'],original['condition'])


@pytest.mark.parametrize('kind',[WAIT,EOS])
def test_timeline_uses_first_trusted_rest_or_end_target_without_retiming_it(monkeypatch,kind):
    ds=dense_dataset(monkeypatch);capture_prefix(monkeypatch)
    target_index=next(i for i,step in enumerate(ds.streams[0]) if step.kind==kind)
    index=next(i for i,(_,start,end,_) in enumerate(ds.samples) if start<=target_index<end)
    original=ds[index];first=max(0,ds.samples[index][1]-ds.lead_in);at=target_index-first
    original['sample_weight'].zero_();original['sample_weight'][at]=2.6
    changed,_,_=aug.augment_stream_row(ds,index,rng=TimelineRandom(0),mode='timeline_loop',
        row=original,input_policy=aug.MUSICAL_BATCH_INPUT_POLICY)
    target=ds.streams[0][target_index]
    assert changed['target_kind'][-1]==kind
    assert changed['step_beat'][-1]==float(target.start)
    assert changed['target_second'][-1]==target.end_second
    assert changed['sample_weight'].sum()==original['sample_weight'].sum()
    assert changed['transition_exit_labels'][-1]


def test_target_atom_change_cannot_change_timeline_input(monkeypatch):
    ds=dense_dataset(monkeypatch);capture_prefix(monkeypatch)
    original=ds[150]
    before,_,_=aug.augment_stream_row(ds,150,rng=TimelineRandom(0),mode='timeline_loop',
        row=original,input_policy=aug.MUSICAL_BATCH_INPUT_POLICY)
    ds.streams[0][600]=replace(ds.streams[0][600],text='8')
    altered=dict(original);altered['target_phase_mask']=original['target_phase_mask'].clone()
    at=int((original['loss_mask']&original['sample_weight'].gt(0)).nonzero()[0])
    altered['target_phase_mask'][at]=torch.tensor(ds.codec.encode('8')['phase_mask'])
    after,_,_=aug.augment_stream_row(ds,150,rng=TimelineRandom(0),mode='timeline_loop',
        row=altered,input_policy=aug.MUSICAL_BATCH_INPUT_POLICY)
    for key in ('history_atoms','history_kind','history_shape','step_beat','step_second','continuation_features'):
        torch.testing.assert_close(before[key],after[key])
    assert not torch.equal(before['target_phase_mask'],after['target_phase_mask'])


def test_illegal_timeline_keeps_original_main_batch_row(monkeypatch):
    ds=dense_dataset(monkeypatch);original=ds[150]
    monkeypatch.setattr(aug,'_augmentation_modes',lambda _:('timeline_loop',))
    monkeypatch.setattr(recovery,'_synthetic_long_exit_row_at',lambda *args:None)
    changed,stats=aug.augment_stream_items(ds,[original],rate=1.,seed=0,input_policy=aug.MUSICAL_BATCH_INPUT_POLICY)
    assert changed[0] is original
    assert stats['skipped_unreachable_or_conflict']==1 and not stats.get('applied',0)


def _phrase_dataset(monkeypatch):
    from maichart.composer.stream_data import StreamDataset
    legacy=dense_dataset(monkeypatch)
    return StreamDataset(base=legacy.base,lead_in=128,targets=16,audio_grid_seconds=.05,
        history_window_policy='block_aligned_v2',continuation_policy='pattern_runs_v2',
        phrase_context_policy='aligned_phrase_v3',training_window_policy='phrase_complete_v1',
        relation_policy='future_sets_v1',max_phrase_events=64,phrase_seam_events=2)


def test_phrase_timeline_replacement_collates_with_real_phrase_and_keeps_budget(monkeypatch):
    ds=_phrase_dataset(monkeypatch);capture_prefix(monkeypatch)
    index=next(i for i,(_,lo,hi,_) in enumerate(ds.samples) if lo>=550 and hi<len(ds.streams[0])-3)
    original=ds[index];neighbor=ds[index+1]
    changed,_,_=aug.augment_stream_row(ds,index,rng=TimelineRandom(0),mode='timeline_loop',
        row=original,input_policy=aug.MUSICAL_BATCH_INPUT_POLICY)
    assert changed['training_phrase_group'].eq(-1).all()
    assert not changed['training_phrase_complete'].any()
    assert not changed['target_relation_mask'].any()
    torch.testing.assert_close(changed['_training_target_budget'].sum(),original['_training_target_budget'].sum())
    before=collate_stream([original,neighbor]);after=collate_stream([changed,neighbor])
    assert after['_loss_normalizer']==before['_loss_normalizer']
    assert after['_phrase_budget_scale']==before['_phrase_budget_scale']
    assert not after['training_phrase_complete'][0].any()
    for key in ('audio_plan_weight','audio_plan_blank_weight'):
        torch.testing.assert_close(after[key],before[key])
    solo=collate_stream([changed])
    assert solo['training_phrase_group_weight'].numel()==0


def test_phrase_local_augmentation_rebuilds_supervised_seam_rows(monkeypatch):
    ds=_phrase_dataset(monkeypatch);capture_prefix(monkeypatch)
    index=next(i for i,(_,lo,hi,_) in enumerate(ds.samples) if lo>=550 and hi<len(ds.streams[0])-3)
    original=ds[index]
    original_relation_mask=original['target_relation_mask'].clone()
    original_complete=original['training_phrase_complete'].clone()
    changed,_,_=aug.augment_stream_row(ds,index,rng=TimelineRandom(0),mode='abab',
        row=original,input_policy=aug.MUSICAL_BATCH_INPUT_POLICY)
    assert len(changed['history_kind'])==len(original['history_kind'])
    assert len(changed['phrase_start_beat'])==len(original['history_kind'])
    assert not changed['target_relation_mask'].any()
    assert not changed['training_phrase_complete'].any()
    torch.testing.assert_close(original['target_relation_mask'],original_relation_mask)
    torch.testing.assert_close(original['training_phrase_complete'],original_complete)
    collate_stream([changed,ds[index+1]])
