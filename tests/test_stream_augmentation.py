from fractions import Fraction as F
from types import SimpleNamespace
import random
import pytest

import torch

from maichart.event_timing import EventPoint,EventTempo
from maichart.composer import stream_data,stream_augmentation
from maichart.composer.stream_data import StreamDataset,collate_stream


def dataset(monkeypatch):
    class EmptyState:
        def __init__(self,*args):self.points=[]
        def append(self,point):self.points.append(point)
        def at(self,*args,**kwargs):return torch.zeros(1,8),torch.ones(1,dtype=torch.bool),{}
    monkeypatch.setattr(stream_data,'STATE_DIM',8)
    monkeypatch.setattr(stream_data,'SustainStateIndex',EmptyState)
    monkeypatch.setattr(stream_data,'IncrementalState',EmptyState)
    bank=(torch.zeros(240,238),torch.arange(240,dtype=torch.float64)/20,torch.arange(240,dtype=torch.float64)/10)
    monkeypatch.setattr(StreamDataset,'_bank',lambda self,song:bank)
    song=SimpleNamespace(audio=SimpleNamespace(duration_sec=12.),tempo=[EventTempo(F(0),120.)],first=0.)
    points=[EventPoint(F(i),str(i%8+1)) for i in range(20)]
    base=SimpleNamespace(songs=[(song,points,12.,1.,None,torch.zeros(16),(),None)])
    return StreamDataset(base=base,lead_in=8,targets=4,audio_grid_seconds=.05,
        history_window_policy='block_aligned_v2',continuation_policy='pattern_runs_v2')


def test_augmented_row_rebuilds_events_and_preserves_targets_and_clocks(monkeypatch):
    ds=dataset(monkeypatch);original=ds[3]
    changed,mode,count=stream_augmentation.augment_stream_row(ds,3,rng=random.Random(8),mode='abab')
    assert mode=='abab' and 4<=count<=12
    for key in ('history_delta','step_beat','step_second','step_index','target_kind','target_delta',
                'target_shape','target_phase_mask','target_details','target_second','loss_mask'):
        torch.testing.assert_close(changed[key],original[key])
    torch.testing.assert_close(changed['sample_weight'].sum(),original['sample_weight'].sum(),rtol=0,atol=0)
    assert changed['continuation_features'].shape[-1]==16
    assert not torch.equal(changed['history_atoms'],original['history_atoms'])
    assert not torch.equal(changed['continuation_features'],original['continuation_features'])
    assert changed['action_state'].size(0)==changed['history_kind'].size(0)
    for row in changed['history_atoms']:
        if row.any():assert all(int(atom)<ds.codec.num_atoms for atom in row)


def test_batch_augmentation_is_seeded_and_rate_zero_is_identity(monkeypatch):
    ds=dataset(monkeypatch);batch=collate_stream([ds[2],ds[3]])
    same,stats=stream_augmentation.augment_stream_batch(ds,batch,rate=0,seed=4)
    assert same is batch and stats=={}
    first,a=stream_augmentation.augment_stream_batch(ds,batch,rate=1,seed=4)
    second,b=stream_augmentation.augment_stream_batch(ds,batch,rate=1,seed=4)
    assert a==b and a['attempted']==2
    for key in first:
        if torch.is_tensor(first[key]):torch.testing.assert_close(first[key],second[key])


def test_conflicting_target_drops_augmentation_without_mutating_batch(monkeypatch):
    ds=dataset(monkeypatch);batch=collate_stream([ds[3]])
    def reject(*args):raise ValueError('native target conflict')
    monkeypatch.setattr(stream_augmentation,'_validate_window',reject)
    result,stats=stream_augmentation.augment_stream_batch(ds,batch,rate=1,seed=8)
    assert result is batch and stats['skipped_unreachable_or_conflict']==1
    assert not stats.get('applied',0)


def test_producer_items_reuse_unselected_rows_without_dataset_rereads(monkeypatch):
    ds=dataset(monkeypatch);items=[ds[2],ds[3]]
    original=[row['history_atoms'].clone() for row in items]
    result,diag=stream_augmentation.augment_stream_items(ds,items,rate=1,seed=4)
    assert len(result)==2 and diag['attempted']==2
    for row,atoms in zip(items,original):torch.testing.assert_close(row['history_atoms'],atoms)
    unchanged,stats=stream_augmentation.augment_stream_items(ds,items,rate=0,seed=1)
    assert unchanged is items and stats=={}


def test_local_rebuild_matches_full_replay_including_old_long_hold(monkeypatch):
    from maichart.generation.sustain_state import SustainStateIndex
    bank=(torch.zeros(320,238),torch.arange(320,dtype=torch.float64)/20,
          torch.arange(320,dtype=torch.float64)/10)
    monkeypatch.setattr(StreamDataset,'_bank',lambda self,song:bank)
    song=SimpleNamespace(audio=SimpleNamespace(duration_sec=16.),tempo=[EventTempo(F(0),120.)],first=0.)
    points=[EventPoint(F(0),'8h[1:5]')]+[EventPoint(F(i),str((i-1)%6+1)) for i in range(1,24)]
    native=SustainStateIndex(points,song.tempo,song.first)
    base=SimpleNamespace(songs=[(song,points,12.,1.,native,torch.zeros(16),(),None)])
    ds=StreamDataset(base=base,lead_in=16,targets=4,audio_grid_seconds=.05,
        history_window_policy='block_aligned_v2',continuation_policy='pattern_runs_v2')
    original=ds[4]
    # The old hold begins outside the 4--12 edited events and is still active
    # during the supervised block. No fixed history/time cutoff may lose it.
    fast,mode,count=stream_augmentation.augment_stream_row(ds,4,rng=random.Random(8),
        mode='same_shape',row=original)
    monkeypatch.setattr(stream_augmentation,'_native_prefix',lambda *args:None)
    full,mode2,count2=stream_augmentation.augment_stream_row(ds,4,rng=random.Random(8),
        mode='same_shape',row=original)
    assert (mode,count)==(mode2,count2)
    for key in ('history_kind','history_atoms','history_shape','history_phase','history_phase_mask',
                'continuation_features','transition_exit_mask','transition_exit_labels'):
        torch.testing.assert_close(fast[key],full[key])
    width=max(fast['action_state'].size(1),full['action_state'].size(1))
    def padded(row,key):
        value=row[key]
        return torch.nn.functional.pad(value,(0,0,0,width-value.size(1))) if key=='action_state' else torch.nn.functional.pad(value,(0,width-value.size(1)),value=True)
    torch.testing.assert_close(padded(fast,'action_state'),padded(full,'action_state'),atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(padded(fast,'action_state_mask'),padded(full,'action_state_mask'))


def _window_guard():
    from maichart.composer.guard import EventGuard
    return EventGuard([EventTempo(F(0),120.)],0.,30.,touch_layout_policy='open',intercept_tap_on_slide=True)


def _window_steps(texts,beats=None):
    from maichart.composer.stream_data import StreamStep,NOTE
    beats=beats or [F(i+4) for i in range(len(texts))]
    return [StreamStep(NOTE,beat-F(1),beat,float(beat-F(1))/2,float(beat)/2,text)
            for beat,text in zip(beats,texts)]


def test_window_native_validation_uses_at_most_two_calls_and_full_target_block(monkeypatch):
    guard=_window_guard();guard.commit(EventPoint(F(0),'8h[1:5]'))
    calls=[]
    def native(points,*args):
        calls.append(tuple(points))
        return SimpleNamespace(conflicts_detected=[],coverage={})
    monkeypatch.setattr(stream_augmentation,'detect_chart_conflicts',native)
    steps=_window_steps(['1','2','3','4','5'])
    source=list(steps)
    stream_augmentation._validate_window(guard,steps)
    assert len(calls)==2
    assert calls[0]==(EventPoint(F(0),'8h[1:5]'),)
    assert calls[1]==calls[0]+tuple(EventPoint(step.end,step.text) for step in steps)
    assert steps==source and len(guard.prefix)==1


def test_window_native_validation_preserves_old_long_hold_conflicts(monkeypatch):
    guard=_window_guard();guard.commit(EventPoint(F(0),'8h[1:5]'))
    observed=[]
    issue=SimpleNamespace(conflict_type='miacode_Overlap',time=2.,severity='hard',action='enforce',
        evidence='miacode_runtime:hold_overlap',notes=[SimpleNamespace(beat=F(4),raw='8')])
    def native(points,*args):
        observed.append(points)
        has_old=any(p.beat==0 and p.note_text=='8h[1:5]' for p in points)
        has_new=any(p.beat==4 and p.note_text=='8' for p in points)
        return SimpleNamespace(conflicts_detected=[issue] if has_old and has_new else [],coverage={})
    monkeypatch.setattr(stream_augmentation,'detect_chart_conflicts',native)
    with pytest.raises(ValueError,match='native hard conflict'):
        stream_augmentation._validate_window(guard,_window_steps(['8']))
    assert len(observed)==2 and observed[-1][0].beat==0


def test_window_preexisting_prefix_hard_conflict_is_not_rejected_again(monkeypatch):
    guard=_window_guard();guard.commit(EventPoint(F(0),'8h[1:5]'))
    issue=SimpleNamespace(conflict_type='miacode_Overlap',time=.5,severity='hard',action='enforce',
        evidence='miacode_runtime:old',notes=[SimpleNamespace(beat=F(0),raw='8h[1:5]')])
    calls=[]
    def native(points,*args):
        calls.append(1)
        return SimpleNamespace(conflicts_detected=[issue],coverage={})
    monkeypatch.setattr(stream_augmentation,'detect_chart_conflicts',native)
    stream_augmentation._validate_window(guard,_window_steps(['1','2']))
    assert len(calls)==2


@pytest.mark.parametrize('texts,beats',[
    (['1/1'],[F(4)]),
    (['1','1'],[F(4),F(401,100)]),
    (['1h[1:100]'],[F(4)]),
])
def test_window_cheap_guard_rules_reject_before_native(monkeypatch,texts,beats):
    calls=[]
    monkeypatch.setattr(stream_augmentation,'detect_chart_conflicts',lambda *args:calls.append(1))
    with pytest.raises(ValueError):
        stream_augmentation._validate_window(_window_guard(),_window_steps(texts,beats))
    assert not calls


def test_touch_candidates_are_cached_in_original_catalog_order(monkeypatch):
    ds=SimpleNamespace(codec=SimpleNamespace(atom_catalog=['2','A2','A1','C','A1h[4:1]']))
    original=stream_augmentation.typed_event;calls=[]
    def counted(text):calls.append(text);return original(text)
    monkeypatch.setattr(stream_augmentation,'typed_event',counted)
    first=stream_augmentation._loop_texts(ds,[],0,4,'touch',random.Random(4))
    second=stream_augmentation._loop_texts(ds,[],0,4,'touch',random.Random(4))
    assert first==second and len(calls)==len(ds.codec.atom_catalog)
    assert ds._augmentation_simple_touch[2]==('A2','A1','C')


class _LargestWindowRandom(random.Random):
    def randint(self,a,b):return b
    def randrange(self,*args):return 0


def timing_dataset(monkeypatch):
    # Keep the small deterministic action-state fixture, but use real tempo
    # conversion and a mixed rhythm window with an exactly feasible .5 loop.
    dataset(monkeypatch)
    bank=(torch.zeros(240,238),torch.arange(240,dtype=torch.float64)/20,
          torch.arange(240,dtype=torch.float64)/10)
    monkeypatch.setattr(StreamDataset,'_bank',lambda self,song:bank)
    song=SimpleNamespace(audio=SimpleNamespace(duration_sec=12.),
        tempo=[EventTempo(F(0),120.),EventTempo(F(4),180.)],first=0.)
    beats=list(map(F,['0','1','5/4','2','3','7/2','9/2','11/2','13/2','17/2','9','19/2']))
    points=[EventPoint(beat,str(i%8+1)) for i,beat in enumerate(beats)]
    base=SimpleNamespace(songs=[(song,points,12.,1.,None,torch.zeros(16),(),None)])
    return StreamDataset(base=base,lead_in=8,targets=4,audio_grid_seconds=.05,
        history_window_policy='block_aligned_v2',continuation_policy='pattern_runs_v2')


def test_same_gap_rebuilds_real_clocks_deltas_and_preserves_supervised_targets(monkeypatch):
    from maichart.composer.stream_data import second_at_beat
    ds=timing_dataset(monkeypatch);original=ds[2]
    source_before=tuple(ds.streams[0]);atoms_before=original['history_atoms'].clone()
    augmented,mode,count=stream_augmentation.augment_stream_row(ds,2,
        rng=_LargestWindowRandom(0),mode='same_gap',row=original)
    assert mode=='same_gap' and count==8
    mask=original['loss_mask']
    for key in ('target_kind','target_delta','target_shape','target_phase_mask','target_details',
                'target_second','step_beat','step_second'):
        torch.testing.assert_close(augmented[key][mask],original[key][mask])
    for key in ('loss_mask','step_index'):
        torch.testing.assert_close(augmented[key],original[key])
    torch.testing.assert_close(augmented['sample_weight'].sum(),original['sample_weight'].sum(),rtol=0,atol=0)
    torch.testing.assert_close(augmented['history_atoms'],original['history_atoms'])
    assert augmented['step_beat'][0]==original['step_beat'][0]==0
    assert augmented['step_beat'][8]==original['step_beat'][8]==5.5
    for i in range(1,len(augmented['step_beat'])):
        gap=augmented['step_beat'][i]-augmented['step_beat'][i-1]
        delta_id=int(augmented['history_delta'][i])
        assert float(ds.delta_vocab.values[delta_id-1])==float(gap)
        assert augmented['history_delta'][i]==augmented['target_delta'][i-1]
    for beat,second in zip(augmented['step_beat'],augmented['step_second']):
        assert abs(float(second)-max(0.,second_at_beat(ds.clocks[0],F(str(float(beat))))))<1e-9
    assert torch.unique(augmented['history_delta'][2:9]).numel()==1
    assert augmented['history_delta'][8]!=augmented['target_delta'][8]
    assert augmented['_augmentation_timing_replay_sec']>=0
    assert tuple(ds.streams[0])==source_before
    torch.testing.assert_close(original['history_atoms'],atoms_before)
    torch.testing.assert_close(ds[2]['step_beat'],original['step_beat'])


def test_same_gap_uses_full_causal_replay_and_ignores_later_target_contents(monkeypatch):
    ds=timing_dataset(monkeypatch);row=ds[2]
    first,_,_=stream_augmentation.augment_stream_row(ds,2,rng=_LargestWindowRandom(),mode='same_gap',row=row)
    source=ds.streams[0];song,start,end,_=ds.samples[2]
    modified=stream_augmentation._same_gap_steps(ds,song,source,0,start,end,_LargestWindowRandom())
    context=stream_augmentation._context(ds,song,row['_song_bank'])
    batch=context.batch(modified[:end-1],modified[end-2].end)
    for key in ('history_delta','history_atoms','step_beat','step_second','action_state',
                'action_state_mask','continuation_features'):
        torch.testing.assert_close(first[key],batch[key][0])
    # Selection/reconstruction of the pre-target loop does not inspect later
    # target contents. Those remain labels and causal history only after onset.
    from dataclasses import replace
    later=list(source);later[start+1]=replace(later[start+1],text='1/5')
    second=stream_augmentation._same_gap_steps(ds,song,later,0,start,end,_LargestWindowRandom())
    assert modified[:start]==second[:start]


def test_same_gap_unreachable_boundary_is_skipped_without_retiming_targets(monkeypatch):
    ds=dataset(monkeypatch);row=ds[3]
    original=row['step_beat'].clone()
    with pytest.raises(ValueError,match='No reachable constant-gap'):
        stream_augmentation.augment_stream_row(ds,3,rng=_LargestWindowRandom(),mode='same_gap',row=row)
    torch.testing.assert_close(row['step_beat'],original)


def test_retimed_earlier_onset_keeps_old_action_active_at_the_new_time():
    from dataclasses import replace
    from maichart.generation.sustain_state import SustainStateIndex
    from maichart.composer.stream_data import build_steps
    from maichart.tempo_clock import TempoClock
    song=SimpleNamespace(audio=SimpleNamespace(duration_sec=6.),
        tempo=[EventTempo(F(0),120.)],first=0.)
    points=[EventPoint(F(0),'8h[4:2]'),EventPoint(F(1),'1'),EventPoint(F(3),'2')]
    native=SustainStateIndex(points,song.tempo,song.first)
    ds=SimpleNamespace(base=SimpleNamespace(songs=[(song,points,12.,1.,native)]),
        views=[SimpleNamespace(context_points=points)])
    source=build_steps(points,TempoClock(song.tempo,0.),6.)
    earlier=replace(source[2],end=F(3,2),end_second=.75)
    incremental,guard_prefix=stream_augmentation._native_prefix(ds,0,source,2,first_step=earlier)
    assert any(marker['source_beat']==0 for marker in incremental.markers)
    assert any(step.end==0 and step.text=='8h[4:2]' for step in guard_prefix)
    assert source[2].end==3


def test_first_decision_rebalance_conserves_mass_and_keeps_every_target():
    row=dict(sample_weight=torch.tensor([0.,0.,.1,.2,.7,0.]),
             loss_mask=torch.tensor([False,False,True,True,True,True]))
    original=row['sample_weight'].clone()
    result=stream_augmentation._rebalance_first_decision(row)
    assert result is not row
    torch.testing.assert_close(result['sample_weight'].sum(),original.sum(),rtol=0,atol=0)
    torch.testing.assert_close(result['sample_weight'][:2],original[:2],rtol=0,atol=0)
    assert result['sample_weight'][2]>original[2]
    assert torch.all(result['sample_weight'][2:5]>0)
    torch.testing.assert_close(result['sample_weight'][3:5],original[3:5]*.5)
    torch.testing.assert_close(result['sample_weight'][2],original[2]*.5+original.sum()*.5)
    torch.testing.assert_close(row['sample_weight'],original,rtol=0,atol=0)
    twice=stream_augmentation._rebalance_first_decision(result)
    assert twice is result
    torch.testing.assert_close(twice['sample_weight'],result['sample_weight'],rtol=0,atol=0)


def test_first_decision_rebalance_is_per_row_and_skips_zero_targets():
    first=dict(sample_weight=torch.tensor([0.,0.,0.,2.,4.]),loss_mask=torch.tensor([False,False,True,True,True]))
    second=dict(sample_weight=torch.tensor([0.,.125,.25]),loss_mask=torch.tensor([False,True,True]))
    a=stream_augmentation._rebalance_first_decision(first)
    b=stream_augmentation._rebalance_first_decision(second)
    assert a['_augmentation_first_decision_index']==3
    assert b['_augmentation_first_decision_index']==1
    torch.testing.assert_close(a['sample_weight'].sum(),first['sample_weight'].sum(),rtol=0,atol=0)
    torch.testing.assert_close(b['sample_weight'].sum(),second['sample_weight'].sum(),rtol=0,atol=0)
    assert a['sample_weight'][3]==4. and a['sample_weight'][4]==2.
    assert b['sample_weight'][1]==.25 and b['sample_weight'][2]==.125
    empty=dict(sample_weight=torch.zeros(3),loss_mask=torch.ones(3,dtype=torch.bool))
    assert stream_augmentation._rebalance_first_decision(empty) is empty


def test_successful_augmentation_rebalances_once_without_changing_original_items(monkeypatch):
    ds=dataset(monkeypatch);original=ds[3];weights=original['sample_weight'].clone()
    changed,_,_=stream_augmentation.augment_stream_row(ds,3,rng=random.Random(8),mode='abab',row=original)
    first=int((original['loss_mask']&weights.gt(0)).nonzero()[0])
    assert changed['_augmentation_first_decision_rebalanced']
    assert changed['sample_weight'][first]>weights[first]
    torch.testing.assert_close(changed['sample_weight'].sum(),weights.sum(),rtol=0,atol=0)
    torch.testing.assert_close(original['sample_weight'],weights,rtol=0,atol=0)
    from collections import Counter
    stats=Counter();stream_augmentation._record_first_decision(stats,changed,original)
    assert stats['rebalanced_rows']==stats['first_decision_count']==1
    assert stats['first_decision_mass']==float(changed['sample_weight'][first])
