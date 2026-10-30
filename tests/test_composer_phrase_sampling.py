"""Phrase grouping changes sample order only, retaining every causal target."""
from fractions import Fraction
from itertools import chain
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

from maichart.composer.train import SongLocalBatchSampler, RebatchedSampler, parser
from maichart.composer.sampling import phrase_sample_groups, resolve_sample_sequence_policy
from test_composer_policy_webui import harness, value
from maichart.webui import composer_api


def dataset(bpm=120):
    beats = [Fraction(i,4) for i in range(41)] + [Fraction(40+i*9) for i in range(14)]
    return SimpleNamespace(samples=[SimpleNamespace(song_index=song,event_index=i,target=beat,
        second=float(beat)*60/bpm, history_prefix=tuple(range(i)))
        for song in range(4) for i,beat in enumerate(beats)])


def flattened(sampler):
    return list(chain.from_iterable(sampler))


def test_phrase_partition_is_chronological_and_bounded_in_beats_not_seconds():
    slow, fast = dataset(60), dataset(240)
    indices=[i for i,sample in enumerate(slow.samples) if sample.song_index==0]
    groups=phrase_sample_groups(slow.samples,indices)
    assert groups==phrase_sample_groups(fast.samples,indices)
    assert list(chain.from_iterable(groups))==indices
    for group in groups:
        assert 1 <= len(group) <= 8
        beats=[slow.samples[i].target for i in group]
        assert beats==sorted(beats)
        assert beats[-1]-beats[0] <= 16
    # Sparse targets force a boundary before the ninth target count is reached.
    assert any(len(group)<8 for group in groups[:-1])


@pytest.mark.parametrize('batch_size,locality',[(96,4),(7,3),(1,1)])
def test_phrase_sampler_exact_coverage_and_each_song_block_order(batch_size,locality):
    data=dataset()
    prefixes=[sample.history_prefix for sample in data.samples]
    sampler=SongLocalBatchSampler(data,batch_size,torch.Generator().manual_seed(13),locality,
        sequence_policy='phrase_groups_v1')
    batches=list(sampler)
    order=flattened(batches)
    assert sorted(order)==list(range(len(data.samples)))
    assert len(batches)==len(sampler)
    assert all(len(batch)==batch_size for batch in batches[:-1])
    for song_indices, groups in zip(sampler.groups,sampler.phrase_groups):
        song_set=set(song_indices)
        song_order=[i for i in order if i in song_set]
        positions={index:pos for pos,index in enumerate(song_order)}
        for group in groups:
            assert [positions[i] for i in group]==list(range(positions[group[0]],positions[group[0]]+len(group)))
    assert prefixes==[sample.history_prefix for sample in data.samples]


def test_phrase_sampler_rng_reproduction_and_rebatch_keep_exact_sequence():
    data=dataset()
    generator=torch.Generator().manual_seed(47)
    saved=generator.get_state().clone()
    source=SongLocalBatchSampler(data,96,generator,4,sequence_policy='phrase_groups_v1')
    order=flattened(source)
    replay_generator=torch.Generator(); replay_generator.set_state(saved)
    replay=SongLocalBatchSampler(data,96,replay_generator,4,sequence_policy='phrase_groups_v1')
    assert flattened(RebatchedSampler(replay,64,len(data.samples)))==order
    assert not torch.equal(saved,generator.get_state())


def test_sample_sequence_legacy_default_and_resume_contract():
    assert resolve_sample_sequence_policy(configuration_quality_policy='song_paired_v3')=='phrase_groups_v1'
    assert resolve_sample_sequence_policy()=='independent_events_v1'
    assert resolve_sample_sequence_policy(parent_args={},resume=True,
        configuration_quality_policy='song_paired_v3')=='independent_events_v1'
    assert resolve_sample_sequence_policy(parent_args={'sample_sequence_policy':'phrase_groups_v1'},resume=True)=='phrase_groups_v1'
    with pytest.raises(ValueError,match='Resume inherits'):
        resolve_sample_sequence_policy('phrase_groups_v1',{},resume=True)
    assert resolve_sample_sequence_policy('phrase_groups_v1',{})=='phrase_groups_v1'
    with pytest.raises(ValueError,match='Unknown'):
        resolve_sample_sequence_policy('invalid')
    data=dataset()
    legacy=SongLocalBatchSampler(data,96,torch.Generator().manual_seed(3),4)
    explicit=SongLocalBatchSampler(data,96,torch.Generator().manual_seed(3),4,'independent_events_v1')
    assert flattened(legacy)==flattened(explicit)
    assert parser().get_default('sample_sequence_policy')=='auto'


def test_sample_sequence_api_fresh_resume_and_explicit_migration(harness):
    payload,parent,calls,handler,runtime=harness
    composer_api.start_train(handler,dict(payload,mode='fresh'),runtime=runtime)
    assert value(calls[-1][1],'sample_sequence_policy')=='phrase_groups_v1'
    assert calls[-1][0].result['effective_settings']['sample_sequence_policy']=='phrase_groups_v1'
    composer_api.start_train(handler,dict(payload,mode='resume'),runtime=runtime)
    assert value(calls[-1][1],'sample_sequence_policy')=='independent_events_v1'
    with pytest.raises(ValueError,match='Resume inherits'):
        composer_api.start_train(handler,dict(payload,mode='resume',sample_sequence_policy='phrase_groups_v1'),runtime=runtime)
    composer_api.start_train(handler,dict(payload,mode='finetune',sample_sequence_policy='phrase_groups_v1'),runtime=runtime)
    assert value(calls[-1][1],'sample_sequence_policy')=='phrase_groups_v1'
    parent['args']['sample_sequence_policy']='phrase_groups_v1'
    composer_api.start_train(handler,dict(payload,mode='resume'),runtime=runtime)
    assert value(calls[-1][1],'sample_sequence_policy')=='phrase_groups_v1'


def test_both_training_pages_expose_sample_organization():
    static=Path(__file__).resolve().parents[1]/'src/maichart/webui/static'
    for name in ('index.html','composer.html'):
        text=(static/name).read_text(encoding='utf-8')
        assert '训练样本组织' in text
        assert 'value="phrase_groups_v1"' in text
        assert 'value="independent_events_v1"' in text
