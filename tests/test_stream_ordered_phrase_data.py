"""Ordered archive/data contracts. Run with the final coordinated test batch."""
from dataclasses import replace
from fractions import Fraction as F
from types import SimpleNamespace
import random

import numpy as np
import pytest
import torch

from maichart.composer.event_set_codec import EventSetCodec
from maichart.composer.stream_data import StreamDataset, StreamStep, StreamInferenceContext, collate_stream, recovery_sample
from maichart.composer.stream_augmentation import augment_stream_row
from maichart.composer.stream_phrase import PhraseIndex, PHRASE_SEQUENCE_FIELDS, PHRASE_SEQUENCE_LENGTH, collate_phrase_sequences
from maichart.event_timing import EventPoint, EventTempo


def _bank():
    beats = torch.arange(0.,24.,.1,dtype=torch.float64)
    return torch.zeros(len(beats),238),beats/2,beats


def _step(start,end,text):
    return StreamStep(2,F(start),F(end),float(start)/2,float(end)/2,text)


@pytest.fixture
def dataset_pair(monkeypatch):
    bank = _bank()
    monkeypatch.setattr(StreamDataset,'_bank',lambda self,song:bank)
    song = SimpleNamespace(audio=SimpleNamespace(duration_sec=12.),tempo=[EventTempo(F(0),120.)],first=0.)
    codec = EventSetCodec.fit([str(i) for i in range(1,9)]+['1h[4:1]','2-6[8:1]','A1'])
    def build(alternate):
        points = [EventPoint(F(i),str((i+(3 if alternate and i>=13 else 0))%8+1)) for i in range(20)]
        base = SimpleNamespace(songs=[(song,points,12.,1.,None,torch.zeros(16),(),None)])
        return StreamDataset(base=base,codec=codec,lead_in=8,targets=4,audio_grid_seconds=.05,
            history_window_policy='block_aligned_v2',continuation_policy='pattern_runs_v2',
            phrase_context_policy='ordered_phrase_v2')
    return build(False),build(True),song,bank


def test_archive_preserves_complete_atoms_and_original_clocks():
    codec = EventSetCodec.fit(['1h[4:1]/5','2-6[8:1]','A1'])
    steps = [_step(0,1,'1h[4:1]/5'),_step(1,2,'2-6[8:1]'),_step(2,3,'A1')]
    index = PhraseIndex(_bank(),24.,steps,sequence_codec=codec)
    sequence = index.sequence(0)
    assert sequence['mask'].sum()==3
    for i,step in enumerate(steps):
        encoded = codec.encode(step.text)
        assert sequence['atoms'][i,:len(encoded['atoms'])].tolist()==encoded['atoms']
        assert sequence['phase_mask'][i].tolist()==list(encoded['phase_mask'])
        assert sequence['shape'][i]==encoded['shape']
        assert sequence['beats'][i]==float(step.end)
        assert sequence['seconds'][i]==step.end_second
        assert sequence['gap_beats'][i]==float(step.end-step.start)
        assert sequence['gap_seconds'][i]==step.end_second-step.start_second


def test_archive_time_sampling_keeps_endpoints_and_full_unsampled_identity():
    codec = EventSetCodec.fit(['1','2'])
    steps = [_step(F(i,10),F(i+1,10),'1') for i in range(79)]
    index = PhraseIndex(_bank(),24.,steps,sequence_codec=codec)
    sequence = index.sequence(0)
    assert sequence['mask'].sum()==PHRASE_SEQUENCE_LENGTH
    selected = sequence['beats'][sequence['mask']]
    assert selected[0]==.1 and selected[-1]==7.9
    assert np.all(np.diff(selected)>0)
    omitted = next(i for i,step in enumerate(steps) if float(step.end) not in selected)
    changed = list(steps);changed[omitted] = replace(changed[omitted],text='2')
    changed_index = PhraseIndex(_bank(),24.,changed,sequence_codec=codec)
    other = changed_index.sequence(0)
    assert np.array_equal(sequence['atoms'],other['atoms'])
    assert not np.array_equal(sequence['_fingerprint'],other['_fingerprint'])
    assert np.array_equal(sequence['_visible_fingerprint'],other['_visible_fingerprint'])
    assert len(index.sequence_events[0])==79
    observed_change = list(steps);observed_change[0] = replace(observed_change[0],text='2')
    observed_index = PhraseIndex(_bank(),24.,observed_change,sequence_codec=codec)
    assert not np.array_equal(sequence['_visible_fingerprint'],observed_index.sequence(0)['_visible_fingerprint'])
    rows = []
    for owner in (index,changed_index,observed_index):
        row = owner.sequence_rows([0],[0.],target_mask=[True])
        row['history_kind'] = torch.zeros(1,dtype=torch.long)
        rows.append(row)
    packed = collate_phrase_sequences(rows)
    assert packed['target_phrase_sequence_positive_ids'].tolist()==[0,0,1]
    assert not torch.equal(packed['target_phrase_sequence_fingerprint'][0],packed['target_phrase_sequence_fingerprint'][1])
    assert torch.equal(packed['target_phrase_sequence_visible_fingerprint'][0],packed['target_phrase_sequence_visible_fingerprint'][1])


def test_reference_tables_cannot_see_future_and_match_committed_context(dataset_pair):
    first,alternate,song,bank = dataset_pair
    row,other = first[3],alternate[3]
    # Same completed reference segments despite different current/future labels.
    for field in PHRASE_SEQUENCE_FIELDS:
        torch.testing.assert_close(row['phrase_sequence_'+field],other['phrase_sequence_'+field],rtol=0,atol=0)
    assert not torch.equal(row['target_phrase_sequence_atoms'],other['target_phrase_sequence_atoms'])
    for local,query in enumerate(row['step_index'].tolist()):
        context = StreamInferenceContext(song,first.codec,first.delta_vocab,first.conditions[0],lead_in=8,audio_bank=bank,
            history_window_policy='block_aligned_v2',rollout_steps=4,continuation_policy='pattern_runs_v2',
            phrase_context_policy='ordered_phrase_v2')
        for step in first.streams[0][:query]:context.commit(step)
        inferred = context.current_batch()
        for slot,table in enumerate(row['phrase_sequence_inverse'][local].tolist()):
            observed = int(inferred['phrase_sequence_inverse'][0,-1,slot])
            assert (table<0)==(observed<0)
            if table>=0:
                for field in PHRASE_SEQUENCE_FIELDS:
                    torch.testing.assert_close(row['phrase_sequence_'+field][table],inferred['phrase_sequence_'+field][observed],rtol=0,atol=0)
                segment = row['_phrase_sequence_segments'][table]
                assert first._phrase_index(0).indices[segment][-1]<query
        assert (inferred['target_phrase_sequence_inverse']==-1).all()
        assert len(inferred['target_phrase_sequence_kind'])==0


def test_collate_deduplicates_archive_identity_and_never_merges_augmentation(dataset_pair):
    dataset,_,_,_ = dataset_pair
    original = dataset[3]
    duplicate = collate_stream([original,original])
    assert len(duplicate['phrase_sequence_kind'])==len(original['phrase_sequence_kind'])
    assert torch.equal(duplicate['phrase_sequence_inverse'][0],duplicate['phrase_sequence_inverse'][1])
    changed,_,_ = augment_stream_row(dataset,3,rng=random.Random(8),mode='abab')
    assert changed['_phrase_sequence_owner'] is not original['_phrase_sequence_owner']
    assert (changed['target_phrase_sequence_inverse']==-1).all()
    assert len(changed['target_phrase_sequence_kind'])==0
    batch = collate_stream([original,changed])
    ids0 = set(batch['phrase_sequence_inverse'][0].flatten().tolist())-{-1}
    ids1 = set(batch['phrase_sequence_inverse'][1].flatten().tolist())-{-1}
    assert not ids0.intersection(ids1)
    assert (batch['target_phrase_sequence_inverse'][1]==-1).all()


def test_recovery_prefix_rewrite_has_no_future_sequence_targets(dataset_pair):
    dataset,_,_,_ = dataset_pair
    prefix = list(dataset.streams[0][:13])
    prefix[6] = replace(prefix[6],text='8')
    row = recovery_sample(dataset,0,prefix,dataset.streams[0][13])
    assert (row['target_phrase_sequence_inverse']==-1).all()
    assert len(row['target_phrase_sequence_kind'])==0
    assert not row['target_phrase_mask'].any()
    batch = collate_stream([row])
    assert batch['phrase_sequence_inverse'].shape[:2]==batch['history_kind'].shape
    assert len(batch['target_phrase_sequence_positive_ids'])==0


def test_identical_translated_full_sequences_share_positive_id():
    codec = EventSetCodec.fit(['1'])
    first = [_step(F(i,10),F(i+1,10),'1') for i in range(79)]
    second = [_step(8+step.start,8+step.end,'1') for step in first]
    index = PhraseIndex(_bank(),24.,first+second,sequence_codec=codec)
    row = index.sequence_rows([0,len(first)],[0.,8.],target_mask=[True,True])
    assert len(row['target_phrase_sequence_fingerprint'])==2
    assert torch.equal(row['target_phrase_sequence_fingerprint'][0],row['target_phrase_sequence_fingerprint'][1])
    assert row['target_phrase_sequence_positive_ids'].tolist()==[0,0]
    assert torch.equal(row['target_phrase_sequence_visible_fingerprint'][0],row['target_phrase_sequence_visible_fingerprint'][1])


@pytest.mark.parametrize('policy',['ordered_phrase_v2','aligned_phrase_v3'])
@pytest.mark.parametrize('augmented',[False,True])
def test_lazy_training_archive_packing_is_exact_after_collation(dataset_pair,policy,augmented):
    dataset,_,_,_=dataset_pair
    dataset.phrase_context_policy=policy
    def rows():
        original=dataset[3]
        changed=(augment_stream_row(dataset,3,rng=random.Random(8),mode='abab')[0]
                 if augmented else dataset[2])
        return [original,changed]
    eager_rows=rows()
    eager=collate_stream(eager_rows)
    dataset.defer_phrase_sequence_packing=True
    lazy_rows=rows()
    assert 'phrase_sequence_atoms' not in lazy_rows[0]
    if policy=='aligned_phrase_v3':assert 'candidate_phrase_sequence_atoms' not in lazy_rows[0]
    lazy=collate_stream(lazy_rows)
    assert eager.keys()==lazy.keys()
    def exact(a,b):
        if isinstance(a,torch.Tensor):assert torch.equal(a,b)
        elif isinstance(a,(list,tuple)):
            assert len(a)==len(b)
            for left,right in zip(a,b):exact(left,right)
        else:assert a==b
    for key in eager:exact(eager[key],lazy[key])
    if policy=='aligned_phrase_v3':assert lazy['phrase_sequence_atoms'] is lazy['candidate_phrase_sequence_atoms']


def test_fused_aligned_collate_matches_original_reference_union_and_packs_once(dataset_pair,monkeypatch):
    from maichart.composer import stream_phrase
    dataset,_,_,_=dataset_pair
    dataset.phrase_context_policy='aligned_phrase_v3'
    items=[dataset[2],dataset[3]]
    original=stream_phrase.collate_phrase_sequences(items)
    original.update(stream_phrase.collate_candidate_phrase_sequences(items))
    stream_phrase.share_aligned_phrase_sequences(items,original)
    calls=[];pack=stream_phrase._pack_sequences
    def counted(sequences,prefix):
        calls.append(prefix)
        return pack(sequences,prefix)
    monkeypatch.setattr(stream_phrase,'_pack_sequences',counted)
    actual=collate_stream(items)
    assert calls.count('phrase_sequence_')==1
    for key,value in original.items():
        if isinstance(value,torch.Tensor):assert torch.equal(actual[key],value)
        else:assert actual[key]==value


def test_dataset_reuses_continuation_traversal_for_recovery_anchors(dataset_pair,monkeypatch):
    from maichart.composer.stream_recovery import _risk_exits
    from maichart.composer.stream_continuation import ContinuationState
    dataset,_,_,_=dataset_pair
    key,cached=dataset._mixed_recovery_exit_anchors
    assert key==(dataset.continuation_policy,len(dataset.streams))
    del dataset._mixed_recovery_exit_anchors
    assert _risk_exits(dataset)==cached
    dataset._mixed_recovery_exit_anchors=(key,cached)
    monkeypatch.setattr(ContinuationState,'append',lambda *_:pytest.fail('recovery rebuilt initialized anchors'))
    assert _risk_exits(dataset) is cached


def test_dataset_pickle_clears_only_hot_caches_and_preserves_batches(dataset_pair):
    import pickle
    dataset,_,_,_=dataset_pair
    dataset.defer_phrase_sequence_packing=True
    baseline=collate_stream([dataset[2],dataset[3]])
    assert dataset._states and dataset._symbols and dataset._phrases
    before=(len(dataset._states),len(dataset._symbols),len(dataset._phrases))
    restored=pickle.loads(pickle.dumps(dataset,protocol=pickle.HIGHEST_PROTOCOL))
    assert before==(len(dataset._states),len(dataset._symbols),len(dataset._phrases))
    for name in ('_banks','_states','_symbols','_phrases','_relation_cache'):assert not getattr(restored,name)
    assert restored._mixed_recovery_exit_anchors==dataset._mixed_recovery_exit_anchors
    assert restored.defer_phrase_sequence_packing
    with restored._bank_lock,restored._state_lock:pass
    actual=collate_stream([restored[2],restored[3]])
    for key,value in baseline.items():
        if isinstance(value,torch.Tensor):assert torch.equal(actual[key],value)
    assert restored.samples==dataset.samples
    assert restored.coverage==dataset.coverage


@pytest.mark.parametrize('include_audio',[False,True])
def test_release_hot_caches_preserves_labels_rows_and_other_process_state(dataset_pair,include_audio):
    import pickle
    import threading
    dataset,_,_,_=dataset_pair
    baseline=collate_stream([dataset[2],dataset[3]])
    worker=pickle.loads(pickle.dumps(dataset));worker[2]
    worker_sizes=(worker._state_bytes,worker._symbol_bytes,worker._phrase_bytes)
    anchors=dataset._mixed_recovery_exit_anchors
    plans,streams,continuations=dataset.plans,dataset.streams,dataset.continuations
    audio=SimpleNamespace(entries={'resident':object()},bytes=1234,_lock=threading.RLock())
    dataset.base.audio_store=audio
    released=dataset.release_hot_caches(include_audio=include_audio)
    assert released['dataset_bytes']>0
    assert released['audio_bytes']==(1234 if include_audio else 0)
    assert len(audio.entries)==(0 if include_audio else 1)
    assert audio.bytes==(0 if include_audio else 1234)
    for name in ('_banks','_states','_symbols','_phrases','_relation_cache'):assert not getattr(dataset,name)
    for name in ('_bank_bytes','_state_bytes','_symbol_bytes','_phrase_bytes','_relation_bytes'):assert getattr(dataset,name)==0
    assert dataset.plans is plans and dataset.streams is streams and dataset.continuations is continuations
    assert dataset._mixed_recovery_exit_anchors is anchors
    assert worker_sizes==(worker._state_bytes,worker._symbol_bytes,worker._phrase_bytes)
    actual=collate_stream([dataset[2],dataset[3]])
    for key,value in baseline.items():
        if isinstance(value,torch.Tensor):assert torch.equal(actual[key],value)


def test_profile_window_importance_gives_each_chart_equal_total_mass(dataset_pair):
    dataset,_,_,_=dataset_pair
    counts=dataset.profile_chart_window_counts
    assert counts.sum()==len(dataset.samples)
    for chart,count in enumerate(counts):
        if count:
            assert count*dataset.profile_chart_importance[chart]==pytest.approx(len(dataset.samples)/int(np.count_nonzero(counts)))
    row=dataset[2]
    assert row['_profile_chart_importance']==dataset.profile_chart_importance[row['song_index']]


def test_profile_importance_aggregates_once_per_chart_and_keeps_partial_batch_mass(dataset_pair):
    dataset,_,_,_=dataset_pair
    original=dataset[2]
    def item(chart,importance):
        row=dict(original);length=len(row['history_kind'])
        row.update(_training_chart_id=chart,_profile_chart_importance=importance,
            _profile_reference_batch_size=3,_training_reference_targets=4,
            _training_target_budget=row['loss_mask'].float(),
            training_phrase_group=torch.zeros(length,dtype=torch.long),
            training_phrase_position=torch.arange(length),
            training_phrase_complete=row['loss_mask'].clone())
        return row
    # Four total windows: chart 0 owns three; chart 1 owns one.
    # N_total / (N_charts*N_chart) gives 2/3 and 2 respectively.
    first=collate_stream([item(0,2/3) for _ in range(3)])
    last=collate_stream([item(1,2.)])
    assert first['profile_chart_rows'].tolist()==[0]
    assert last['profile_chart_rows'].tolist()==[0]
    assert first['profile_chart_weights'].tolist()==pytest.approx([2/3])
    assert last['profile_chart_weights'].tolist()==pytest.approx([2/3])
    mixed=collate_stream([item(0,2/3),item(1,2.),item(0,2/3)])
    assert mixed['profile_chart_rows'].tolist()==[0,1]
    assert mixed['profile_chart_weights'].tolist()==pytest.approx([4/9,2/3])
