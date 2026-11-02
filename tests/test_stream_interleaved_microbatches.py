"""Microbatch interleaving preserves each packed example and resume contract."""
from types import SimpleNamespace
from collections import Counter
from fractions import Fraction
import hashlib
import json
import pytest
import torch
from maichart.composer.stream_train import (_LocalBatches,_ProcessBatches,_resume_batch_mapping,
    resolve_batch_order_policy,batch_order_diagnostics,_fingerprint,upgrade_learning_stage,resolve_learning_policies)


class Rows:
    training_window_policy='phrase_complete_v1'
    def __init__(self,policy):
        self._sample_order_batch_size=8;self._batch_order_policy=policy
        self.samples=[(song,index,index+(index*7)%23+1,1.) for song in range(12) for index in range(37)]
    def __len__(self):return len(self.samples)


def test_interleaving_changes_only_full_microbatch_order_and_keeps_tail_and_resume():
    legacy=Rows('song_local_v1');new=Rows('interleaved_microbatches_v2')
    for epoch in (1,2,17):
        before=list(_LocalBatches(legacy,8,9,epoch));after=list(_LocalBatches(new,8,9,epoch))
        assert before!=after
        assert Counter(map(tuple,before))==Counter(map(tuple,after))
        assert before[-1]==after[-1] and len(after[-1])<8
        assert sorted(index for batch in after for index in batch)==list(range(len(new)))
        assert list(_LocalBatches(new,8,9,epoch,skip=7))==after[7:]
        assert list(_LocalBatches(new,8,9,epoch))==after
    assert list(_LocalBatches(new,8,9,1))!=list(_LocalBatches(new,8,9,2))


def test_interleaved_process_blueprints_use_same_microbatches_and_cursor():
    ds=Rows('interleaved_microbatches_v2');args=SimpleNamespace(batch_size=8,seed=6)
    expected=list(_LocalBatches(ds,8,6,4,skip=3))
    actual=list(_ProcessBatches(ds,args,4,3))
    assert [[entry[0] for entry in batch] for batch in actual]==expected
    assert all(entry[1:3]==(4,number) for number,batch in enumerate(actual,4) for entry in batch)


def test_batch_order_default_inheritance_and_new_rebatch_rejection():
    assert resolve_batch_order_policy()=='interleaved_microbatches_v2'
    assert resolve_batch_order_policy(old_args={},resume=True)=='song_local_v1'
    assert resolve_batch_order_policy(old_args={'batch_order_policy':'interleaved_microbatches_v2'},resume=True)=='interleaved_microbatches_v2'
    with pytest.raises(ValueError,match='Resume inherits batch_order_policy'):
        resolve_batch_order_policy('interleaved_microbatches_v2',{},resume=True)
    args=SimpleNamespace(batch_size=4,grad_accumulation=4,max_steps=0,rebatch_resume=True,batch_order_policy='interleaved_microbatches_v2')
    old=dict(batch_size=8,grad_accumulation=2,max_steps=0,batch_order_policy='interleaved_microbatches_v2')
    with pytest.raises(ValueError,match='cannot rebatch'):
        _resume_batch_mapping(old,dict(batch_in_epoch=4,epoch_complete=False),args,444)


def test_chart_diversity_diagnostic_uses_indices_without_getitem():
    args=SimpleNamespace(batch_size=8,seed=9,grad_accumulation=4)
    old=batch_order_diagnostics(Rows('song_local_v1'),args,1,max_updates=10)
    new=batch_order_diagnostics(Rows('interleaved_microbatches_v2'),args,1,max_updates=10)
    assert new['sampled_optimizer_steps']==10
    assert new['mean_unique_charts']>old['mean_unique_charts']


def test_batch_order_fingerprint_preserves_legacy_hash_and_tags_only_new_policy(tmp_path):
    from maichart.composer.event_set_codec import EventSetCodec
    from maichart.composer.stream_data import DeltaVocabulary
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    args=SimpleNamespace(manifest=manifest,history_events=64,rollout_steps=16,window_seconds=8.,audio_grid_seconds=.01)
    ds=SimpleNamespace(codec=EventSetCodec.fit(['1']),delta_vocab=DeltaVocabulary([Fraction(1)]))
    original=_fingerprint(args,ds)
    args.batch_order_policy='song_local_v1';assert _fingerprint(args,ds)==original
    args.batch_order_policy='interleaved_microbatches_v2';assert _fingerprint(args,ds)!=original


def test_typed_stage_preserves_regular_embedding_prefix_and_retires_only_old_heads(monkeypatch):
    from dataclasses import replace
    from maichart.composer.stream_model import StreamComposer
    from maichart.composer.event_set_codec import EventSetCodec
    from test_stream_phrase_execution import execution_fixture
    _,baseline,_,_=execution_fixture(monkeypatch)
    codec=EventSetCodec.fit(['1','2/6','3h[4:1]','A1','C','1>5[8:1]'])
    parent=StreamComposer(replace(baseline.config,song_profile_policy='masked_ordered_joint_v2',song_response_policy='music_response_v1'),codec)
    source=parent.state_dict();policies=resolve_learning_policies(SimpleNamespace(),parent.config)
    upgraded=upgrade_learning_stage(parent,policies);target=upgraded.state_dict()
    removed=set(source)-set(target)
    assert removed and all(key.startswith(('phrase_context.audio.','phrase_context.writing.',
        'phrase_context.shared.','song_response.song.')) for key in removed)
    grew=False
    for key,value in source.items():
        if key in removed:continue
        if key.startswith('shared_details.component_embeddings.'):
            assert torch.equal(target[key][:len(value)],value)
            grew|=len(target[key])>len(value)
        else:assert torch.equal(target[key],value),key
    assert grew
    for key in ('detail_mode_control.choices','detail_mode_control.stop','literal_component_projection.weight'):
        assert torch.count_nonzero(target[key])==0
