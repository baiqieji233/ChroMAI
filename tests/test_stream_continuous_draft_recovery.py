"""Continuous recovery and CPU packing; run in the final coordinated suite."""
from dataclasses import replace
from fractions import Fraction as F
from types import SimpleNamespace
import random

import torch

from maichart.composer import stream_augmentation as aug
from maichart.composer.stream_data import (NOTE,WAIT,StreamDataset,StreamStep,
    collate_stream,pack_draft_queries,pack_timing_queries,recovery_sequence_sample)
from maichart.composer.stream_target_correspondence import authored_seam_corresponds
from maichart.event_timing import EventPoint,EventTempo


def test_rest_recipe_retains_real_notes_on_both_sides_and_individual_weights(monkeypatch):
    from test_stream_augmentation import dataset
    prior=dataset(monkeypatch);bank=prior._bank(0);bank[0][:,198]=1.
    song=SimpleNamespace(audio=SimpleNamespace(duration_sec=12.),tempo=[EventTempo(F(0),120.)],first=0.)
    points=[EventPoint(F(i),str(i%8+1)) for i in (0,1,2,3,10,11,12,13,14,15,16)]
    ds=StreamDataset(base=SimpleNamespace(songs=[(song,points,12.,1.,None,torch.zeros(16),(),None)]),
        lead_in=8,targets=4,audio_grid_seconds=.05,history_window_policy='block_aligned_v2',
        continuation_policy='pattern_runs_v2')
    row,mode,_=aug.augment_stream_row(ds,1,rng=random.Random(5),mode='authored_rest',
        input_policy=aug.CONTINUOUS_DRAFT_INPUT_POLICY)
    active=row['loss_mask']&row['sample_weight'].gt(0)
    kinds=row['target_kind'][active].tolist()
    assert mode=='authored_rest' and WAIT in kinds
    first=kinds.index(WAIT)
    assert NOTE in kinds[:first] and NOTE in kinds[first+1:]
    assert row['_continuation_teacher_forced'] and not row['_counterfactual_history']
    for position in active.nonzero().flatten().tolist():
        absolute=int(row['step_index'][position]);step=ds.streams[0][absolute]
        expected=step.weight*ds.target_window_weights[0][absolute]*ds.chart_weights[0]
        assert float(row['sample_weight'][position])==float(torch.tensor(expected,dtype=torch.float32))
    assert row['configuration_mask'][active&row['target_kind'].eq(NOTE)].all()


def test_continuation_targets_are_committed_only_after_their_causal_query(monkeypatch):
    from test_stream_augmentation import dataset
    ds=dataset(monkeypatch);source=ds.streams[0];prefix=list(source[:3]);targets=list(source[3:7])
    ds.candidate_state_policy='selected_native_v1'
    row=recovery_sequence_sample(ds,0,prefix,targets,configuration_trusted=True)
    positions=row['loss_mask'].nonzero().flatten().tolist()
    assert len(positions)==4 and row['_continuation_teacher_forced']
    for offset,position in enumerate(positions):
        previous=prefix[-1] if offset==0 else targets[offset-1]
        encoded=ds.codec.encode(previous.text)
        assert row['history_atoms'][position,:len(encoded['atoms'])].tolist()==encoded['atoms']
        assert row['target_beat'][position]==float(targets[offset].end)
    changed=list(targets);changed[-1]=replace(changed[-1],text='8')
    other=recovery_sequence_sample(ds,0,prefix,changed,configuration_trusted=True)
    for key in ('history_atoms','history_shape','action_state','candidate_action_state'):
        torch.testing.assert_close(row[key],other[key],rtol=0,atol=0)


def test_continuation_native_landing_state_matches_actual_commits_with_a_live_hold(monkeypatch):
    from maichart.composer.stream_recovery import _context
    beats=torch.arange(0.,16.,.1,dtype=torch.float64)
    bank=(torch.zeros(len(beats),238),beats/2,beats)
    monkeypatch.setattr(StreamDataset,'_bank',lambda self,song:bank)
    song=SimpleNamespace(audio=SimpleNamespace(duration_sec=8.),tempo=[EventTempo(F(0),120.)],first=0.)
    points=[EventPoint(F(0),'1h[4:4]')]+[EventPoint(F(i),str(i+4)) for i in range(1,5)]
    ds=StreamDataset(base=SimpleNamespace(songs=[(song,points,8.,1.,None,torch.zeros(16),(),None)]),
        lead_in=8,targets=4,audio_grid_seconds=.05,history_window_policy='block_aligned_v2',
        continuation_policy='pattern_runs_v2',candidate_state_policy='selected_native_v1')
    source=ds.streams[0];prefix=list(source[:1]);targets=list(source[1:4])
    row=recovery_sequence_sample(ds,0,prefix,targets,configuration_trusted=True)
    context=_context(ds,0,bank);context.batch(prefix,prefix[-1].end)
    positions=row['loss_mask'].nonzero().flatten().tolist()
    assert (~row['candidate_action_state_mask'][positions[0]]).any(), 'Native hold must remain live at the first landing'
    for position,target in zip(positions,targets):
        expected=context.candidate_batch(target.end)
        actual_mask=row['candidate_action_state_mask'][position]
        expected_mask=expected['candidate_action_state_mask'][0,0]
        torch.testing.assert_close(row['candidate_action_state'][position][~actual_mask],
            expected['candidate_action_state'][0,0][~expected_mask],rtol=0,atol=0)
        context.commit(target)


def test_all_decision_arrival_pack_includes_wait_and_eos_without_padding():
    data=dict(target_kind=torch.tensor([[NOTE,WAIT,4,0]]),loss_mask=torch.tensor([[1,1,1,0]],dtype=torch.bool),
        sample_weight=torch.ones(1,4))
    pack_timing_queries(data,all_decisions=True)
    assert data['_timing_original_flat'].tolist()==[0,1,2]
    pack_timing_queries(data)
    assert data['_timing_original_flat'].tolist()==[0]


def test_draft_groups_are_row_local_even_for_identical_music_boundaries():
    data=dict(phrase_start_beat=torch.tensor([[0.,0.,4.],[0.,0.,4.]]),
        phrase_end_beat=torch.tensor([[4.,4.,8.],[4.,4.,8.]]))
    pack_draft_queries(data)
    assert data['draft_query_rows'].tolist()==[0,0,1,1]
    assert data['draft_query_positions'].tolist()==[0,2,0,2]
    assert data['draft_query_inverse'].tolist()==[[0,0,1],[2,2,3]]
    assert data['draft_query_keys'][0]!=(data['draft_query_keys'][2])


def test_prefix_signature_covers_old_content_and_replay_epoch(monkeypatch):
    from test_stream_augmentation import dataset
    from maichart.composer.stream_recovery import _context
    ds=dataset(monkeypatch);context=_context(ds,0,ds._bank(0));prefix=list(ds.streams[0][:5])
    context.batch(prefix,prefix[-1].end);first=context.prefix_signature();epoch=context._prefix_epoch
    changed=list(prefix);changed[0]=replace(changed[0],text='8')
    context.batch(changed,changed[-1].end)
    assert context._prefix_epoch>epoch and context.prefix_signature()!=first
    assert context.prefix_signature(0)!=context.prefix_signature(1)


def test_real_seam_requires_both_old_sustains_to_expire():
    def note(i,text):return StreamStep(NOTE,F(i),F(i+1),float(i),float(i+1),text)
    authored=[note(i,str(i+1)) for i in range(6)];actual=list(authored)
    actual[0]=replace(actual[0],text='8')
    guard=SimpleNamespace(interaction_start=lambda t:t-.3,
        _lifetime=lambda p:(float(p.beat),float(p.beat)+.05,False,()))
    assert authored_seam_corresponds(authored,actual,guard)
    old_hold=list(authored);old_hold[0]=replace(old_hold[0],text='1h[1:16]')
    guard._lifetime=lambda p:(float(p.beat),20. if 'h' in p.note_text else float(p.beat)+.05,False,())
    assert not authored_seam_corresponds(old_hold,actual,guard)


def test_new_archive_keeps_middle_events_without_changing_legacy_default():
    from test_stream_ordered_phrase_data import _bank,_step
    from maichart.composer.event_set_codec import EventSetCodec
    from maichart.composer.stream_phrase import PhraseIndex
    codec=EventSetCodec.fit(['1','2'])
    steps=[_step(F(i,10),F(i+1,10),'1' if i%2 else '2') for i in range(40)]
    old=PhraseIndex(_bank(),24.,steps,sequence_codec=codec)
    new=PhraseIndex(_bank(),24.,steps,sequence_codec=codec,sequence_length=64)
    assert int(old.sequence(0)['mask'].sum())==32
    assert int(new.sequence(0)['mask'].sum())==40


def test_post_construction_draft_policy_invalidates_only_the_stale_archive(monkeypatch):
    from test_stream_augmentation import dataset
    ds=dataset(monkeypatch)
    old=ds._phrase_index(0)
    assert old.sequence_length==32
    ds.draft_policy='beat_ordered_draft_v1'
    new=ds._phrase_index(0)
    assert new is not old and new.sequence_length==64
    assert ds._phrase_index(0) is new


def test_cross_phrase_time_and_geometry_own_different_groups_without_reweighting():
    from maichart.composer.stream_phrase_training import collate_draft_loss_groups
    expected={(0.,4.):(1,0,True),(4.,8.):(1,2,True)}
    def item(start,end,landing_start,landing_end,index,clock):
        return dict(loss_mask=torch.tensor([True]),sample_weight=torch.tensor([.5]),
            configuration_mask=torch.tensor([True]),target_kind=torch.tensor([NOTE]),
            phrase_start_beat=torch.tensor([start]),phrase_end_beat=torch.tensor([end]),
            candidate_phrase_start_beat=torch.tensor([landing_start]),candidate_phrase_end_beat=torch.tensor([landing_end]),
            step_beat=torch.tensor([clock]),step_index=torch.tensor([index]),
            _training_chart_id=7,_draft_loss_expected=expected)
    items=[item(0.,4.,4.,8.,10,3.),item(4.,8.,4.,8.,11,5.)]
    output=dict(loss_mask=torch.ones(2,1,dtype=torch.bool),phrase_start_beat=torch.tensor([[0.],[4.]]),
        unified_chart_inverse=torch.tensor([0,0]))
    before=[row['sample_weight'].clone() for row in items]
    collate_draft_loss_groups(items,output)
    departure=output['draft_departure_group'];landing=output['draft_landing_group']
    assert departure[0,0]!=landing[0,0]
    assert departure[1,0]==landing[0,0]==landing[1,0]
    group=int(landing[0,0])
    assert int(output['draft_loss_group_rows'][group])==0
    assert int(output['draft_loss_group_side'][group])==1
    assert output['draft_loss_group_complete'].all()
    for row,weight in zip(items,before):torch.testing.assert_close(row['sample_weight'],weight,rtol=0,atol=0)
    items[1]['_counterfactual_history']=True
    collate_draft_loss_groups(items,output)
    assert output['draft_departure_group'][1,0]!=output['draft_landing_group'][0,0]
    assert not output['draft_loss_group_complete'][int(output['draft_departure_group'][1,0])]


def test_recovery_mode_is_read_from_committed_prefix_not_next_gold_event(monkeypatch):
    from test_stream_phrase_execution import execution_fixture
    from maichart.composer.stream_data import recovery_sample
    from maichart.composer.stream_recovery import _context
    ds,_,_,bank=execution_fixture(monkeypatch)
    ds.draft_policy='beat_ordered_draft_v1';ds.unified_arrival=True
    prefix=list(ds.streams[0][:3]);context=_context(ds,0,bank)
    probe=context.batch(prefix,prefix[-1].end)
    start=float(probe['phrase_start_beat'][0,-1])
    prefix[-1]=replace(prefix[-1],phrase_mode=2,phrase_mode_start=start)
    target=replace(ds.streams[0][3],phrase_mode=3,phrase_mode_start=start)
    row=recovery_sample(ds,0,prefix,target,context=context)
    assert int(row['draft_committed_mode'][-1])==2
    expected=2 if float(row['candidate_phrase_start_beat'][-1])==start else -1
    assert int(row['candidate_draft_committed_mode'][-1])==expected
    batch=collate_stream([row]);group=int(batch['draft_departure_group'][0,-1])
    assert int(batch['draft_loss_group_mode'][group])==2
    # A target may announce a new intention only after it has actually landed.
    other=recovery_sample(ds,0,prefix,replace(target,phrase_mode=0),context=context)
    torch.testing.assert_close(other['draft_committed_mode'],row['draft_committed_mode'],rtol=0,atol=0)
    torch.testing.assert_close(other['candidate_draft_committed_mode'],row['candidate_draft_committed_mode'],rtol=0,atol=0)


def test_continuous_prefix_modes_appear_only_after_commit(monkeypatch):
    from test_stream_phrase_execution import execution_fixture
    from maichart.composer.stream_recovery import _context
    ds,_,_,bank=execution_fixture(monkeypatch);ds.draft_policy='beat_ordered_draft_v1'
    prefix=list(ds.streams[0][:2]);context=_context(ds,0,bank)
    start=float(context.batch(prefix,prefix[-1].end)['phrase_start_beat'][0,-1])
    targets=[replace(step,phrase_mode=3,phrase_mode_start=start) for step in ds.streams[0][2:5]]
    row=recovery_sequence_sample(ds,0,prefix,targets,configuration_trusted=True)
    positions=row['loss_mask'].nonzero().flatten()
    assert row['draft_committed_mode'][positions[0]]==-1
    for at in positions[1:].tolist():
        expected=3 if float(row['phrase_start_beat'][at])==start else -1
        assert int(row['draft_committed_mode'][at])==expected
    packed=collate_stream([row]);first=int(packed['draft_departure_group'][0,positions[0]])
    assert int(packed['draft_loss_group_mode'][first])==-1


def test_frozen_song_components_split_recovery_forward_without_overwriting_gold(monkeypatch):
    from test_stream_phrase_execution import execution_fixture
    ds,_,_,_=execution_fixture(monkeypatch)
    import copy
    source=ds[1];items=[copy.deepcopy(source) for _ in range(4)]
    for item,choice in zip(items[2:],(1,2)):
        controls=torch.full((16,),.2+choice*.1);known=torch.zeros(16,dtype=torch.bool);known[0]=True
        item.update(_committed_song_component=choice,
            _committed_execution_controls=controls,_committed_known_mask=known)
    packed=collate_stream(items)
    assert packed['unified_chart_inverse'].tolist()==[0,0,1,2]
    assert packed['committed_song_component'].tolist()==[-1,1,2]
    torch.testing.assert_close(packed['condition'],torch.stack([source['condition']]*4),rtol=0,atol=0)
    assert not packed['committed_known_mask'][0].any()
    assert packed['committed_known_mask'][1:,0].all()


def test_actual_new_draft_recovery_freezes_and_packs_the_executed_style(monkeypatch):
    from test_stream_draft_integration import build
    from maichart.composer.stream_data import recovery_sample
    from maichart.composer.stream_recovery import _context,_encode_song,_dataset_guard,_propose,_propose_fixed
    ds,model,_,bank=build(monkeypatch);model.eval()
    prefix=list(ds.streams[0][:2]);context=_context(ds,0,bank)
    context.batch(prefix,prefix[-1].end)
    cache=_encode_song(model,bank,torch.device('cpu'))
    generator=torch.Generator().manual_seed(8)
    guard=_dataset_guard(ds,0,prefix,len(prefix))
    _propose(ds,model,0,prefix,guard,generator,torch.device('cpu'),context=context,audio_cache=cache)
    frozen=context._recovery_frozen_style
    assert 'unified_style' not in cache, 'An episode must not mutate another episode\'s shared song cache'
    choice=frozen['choice'];controls=frozen['execution_controls'].clone()
    # The fixed-time short-recovery path must keep the same whole-song draw.
    _propose_fixed(ds,model,context,ds.streams[0][2],guard,generator,torch.device('cpu'),cache)
    row=recovery_sample(ds,0,prefix,ds.streams[0][2],context=context)
    packed=collate_stream([row])
    assert packed['committed_song_component'].tolist()==[choice]
    torch.testing.assert_close(packed['committed_execution_controls'][0],controls,rtol=0,atol=0)
    torch.testing.assert_close(context.condition,controls,rtol=0,atol=0)
    assert packed['committed_known_mask'][0,0] and packed['committed_known_mask'].sum()==1


def test_hot_native_row_cache_preserves_nonzero_entries_and_future_candidate_cutoffs(monkeypatch):
    from maichart.composer.stream_data import draft_entry_states
    from maichart.generation.sustain_state import SustainStateIndex
    beats=torch.arange(0.,24.,.1,dtype=torch.float64)
    bank=(torch.zeros(len(beats),238),beats/2,beats)
    monkeypatch.setattr(StreamDataset,'_bank',lambda self,song:bank)
    song=SimpleNamespace(audio=SimpleNamespace(duration_sec=12.),tempo=[EventTempo(F(0),120.)],first=0.)
    points=[EventPoint(F(0),'1h[1:3]')]+[EventPoint(F(i),str(i%7+2)) for i in (1,3,5,7,9,13,17)]
    ds=StreamDataset(base=SimpleNamespace(songs=[(song,points,13.,1.,None,torch.zeros(16),(),None)]),
        lead_in=8,targets=4,audio_grid_seconds=.05,history_window_policy='block_aligned_v2',
        phrase_context_policy='aligned_phrase_v3',candidate_state_policy='selected_native_v1',
        draft_policy='beat_ordered_draft_v1')
    row=ds._sample_row(0,0,0,len(ds.streams[0]),1.,include_seam=False)
    index,cache=ds._song_states(0)
    assert index is None and cache is not None, 'Regression must exercise the large-index release path'
    native=SustainStateIndex(ds.views[0].context_points,song.tempo,song.first)
    for prefix in ('','candidate_'):
        actual=row[prefix+'draft_entry_state']
        expected=draft_entry_states(native,row[prefix+'phrase_start_beat'],row['step_beat'])
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        assert actual.abs().sum()>0, 'The Hold crossing phrase boundaries must remain represented'
    # This query lands in the next phrase while the causal cursor is before
    # that boundary. A not-yet-written Hold in between must be invisible.
    at=(row['candidate_phrase_start_beat']>row['step_beat']).nonzero().flatten()[0]
    boundary=float(row['candidate_phrase_start_beat'][at]);cursor=float(row['step_beat'][at])
    starts=torch.tensor([boundary],dtype=torch.float64);cursors=torch.tensor([cursor],dtype=torch.float64)
    first=draft_entry_states(native,starts,cursors)
    future_beat=F(str((boundary+cursor)/2))
    future=SustainStateIndex(sorted(points+[EventPoint(future_beat,'6h[1:2]')],key=lambda point:point.beat),song.tempo,song.first)
    changed=draft_entry_states(future,starts,cursors)
    torch.testing.assert_close(first,changed,rtol=0,atol=0)
    torch.testing.assert_close(row['candidate_draft_entry_state'][at],first[0],rtol=0,atol=0)
    hot=ds._sample_row(0,0,0,len(ds.streams[0]),1.,include_seam=False)
    torch.testing.assert_close(hot['draft_entry_state'],row['draft_entry_state'],rtol=0,atol=0)
    assert ds._states[0][2]==ds._object_bytes(cache)<=ds._state_budget


def test_continuous_recovery_materializes_candidate_inputs_only_once(monkeypatch):
    from test_stream_phrase_execution import execution_fixture
    from maichart.composer import stream_data
    ds,_,_,_=execution_fixture(monkeypatch)
    calls=[];original=stream_data.prepare_candidate_phrase_inputs
    def prepare(*args,**kwargs):
        calls.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(stream_data,'prepare_candidate_phrase_inputs',prepare)
    def unused(*args,**kwargs):
        raise AssertionError('Intermediate single-target native candidates must not be constructed')
    monkeypatch.setattr(stream_data.StreamInferenceContext,'candidate_batch',unused)
    row=recovery_sequence_sample(ds,0,list(ds.streams[0][:2]),list(ds.streams[0][2:6]),configuration_trusted=True)
    assert len(calls)==1 and row['candidate_action_state'].shape[0]==len(row['history_kind'])


def test_direct_continuation_target_budget_matches_the_original_sample(monkeypatch):
    from test_stream_phrase_execution import execution_fixture
    ds,_,_,_=execution_fixture(monkeypatch)
    weights,budgets=aug._source_span_supervision(ds,0,3,11)
    reference=ds._sample_row(0,0,3,11,1.,include_seam=False)
    selected=reference['step_index'].ge(3)&reference['step_index'].lt(11)
    torch.testing.assert_close(weights,reference['sample_weight'][selected],rtol=0,atol=0)
    torch.testing.assert_close(budgets,reference['_training_target_budget'][selected],rtol=0,atol=0)
