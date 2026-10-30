from dataclasses import replace

import pytest
import torch

from maichart.composer.event_set_codec import EventSetCodec
from maichart.composer.stream_model import StreamComposer, StreamComposerConfig, BOS, NOTE


def fixture():
    torch.manual_seed(41)
    codec=EventSetCodec.fit(['1','2','A1','C','1/A1'])
    config=StreamComposerConfig(d_model=16,nhead=4,layers=1,dim_feedforward=32,audio_dim=6,
        condition_dim=3,dropout=0.,max_event_atoms=4,event_family_policy='audio_semantic_v2',
        timing_output_policy='relative_delta_v1')
    model=StreamComposer(config,codec).eval()
    atom=lambda text:codec._atoms[text]
    batch=dict(history_kind=torch.tensor([[BOS,NOTE,NOTE,NOTE]]),
        history_atoms=torch.tensor([[[0],[atom('1')],[atom('A1')],[atom('1')]]]),
        step_second=torch.tensor([[0.,.25,.5,1.]],dtype=torch.float64),
        step_beat=torch.tensor([[0.,.5,1.,2.]],dtype=torch.float64),
        step_mask=torch.ones(1,4,dtype=torch.bool),condition=torch.zeros(1,3))
    seconds=torch.tensor([[.25,.5,1.,1.25]],dtype=torch.float64)
    cache=model.encode_audio(torch.randn(1,12,6),torch.ones(1,12,dtype=torch.bool),
        torch.arange(12,dtype=torch.float64)[None]*.25,torch.arange(12,dtype=torch.float64)[None]*.5)
    return model,batch,cache,seconds


def test_family_context_does_not_observe_lane_or_touch_ring():
    model,batch,cache,seconds=fixture()
    changed=dict(batch,history_atoms=batch['history_atoms'].clone())
    changed['history_atoms'][0,1,0]=model.codec._atoms['2']
    changed['history_atoms'][0,2,0]=model.codec._atoms['C']
    torch.testing.assert_close(model._family_history_features(batch,seconds),
        model._family_history_features(changed,seconds),atol=0,rtol=0)
    a=model.family_context_at(cache,seconds,torch.tensor([0]),batch)
    b=model.family_context_at(cache,seconds,torch.tensor([0]),changed)
    torch.testing.assert_close(a['audio'],b['audio'],atol=0,rtol=0)
    torch.testing.assert_close(a['full'],b['full'],atol=0,rtol=0)


def test_type_window_statistics_never_read_later_teacher_forced_events():
    model,batch,cache,seconds=fixture()
    changed=dict(batch,history_atoms=batch['history_atoms'].clone())
    changed['history_atoms'][0,3,0]=model.codec._atoms['A1']
    a=model._family_history_features(batch,seconds)
    b=model._family_history_features(changed,seconds)
    torch.testing.assert_close(a[:,:3],b[:,:3],atol=0,rtol=0)
    assert not torch.equal(a[:,3],b[:,3])
    assert a.shape==(1,4,30)


def test_audio_context_changes_with_music_without_changing_symbolic_history():
    model,batch,cache,seconds=fixture()
    altered={**cache,'local':cache['local']+torch.randn_like(cache['local'])}
    a=model.family_context_at(cache,seconds,torch.tensor([0]),batch)
    b=model.family_context_at(altered,seconds,torch.tensor([0]),batch)
    assert not torch.equal(a['audio'],b['audio'])
    # The current-audio identity basis keeps gradients live despite zero new
    # neighborhood/history projections and a zero-initialized classifier.
    assert torch.count_nonzero(a['audio'])>0
    # A real classification target keeps the zero classifier's gradient live.
    loss=torch.nn.functional.cross_entropy(model.audio_event_family_head(a['audio']).flatten(0,1),torch.zeros(4,dtype=torch.long))
    loss.backward()
    assert model.audio_event_family_head.weight.grad.abs().sum()>0


def test_v2_decoding_requires_its_own_family_context():
    model,_,_,_=fixture()
    with pytest.raises(ValueError,match='requires candidate music family_context'):
        model.decode_event(torch.zeros(16),temperature=0.)


def test_family_override_survives_internal_configuration_retry():
    model,_,_,_=fixture()
    with torch.no_grad():
        model.audio_event_family_head.bias.fill_(-100.)
        model.audio_event_family_head.bias[0]=100.  # Main head favors Tap.
        model.literal_head.weight.zero_();model.literal_head.bias.fill_(-100.)
        model.literal_head.bias[model.codec._atoms['A1']]=100.
        model.literal_head.bias[0]=90.
    event=model.decode_event(torch.zeros(16),family_context=torch.ones(16),family_override=8,
        shape_override=0,temperature=0.)
    assert event['family']==8 and event['text']=='A1'
    assert model.sample_event_family(torch.ones(16),temperature=0.)==1


def test_audio_family_budget_replaces_part_of_existing_ce():
    from test_composer_stream_model import fixture as loss_fixture
    original,batch=loss_fixture()
    model=StreamComposer(replace(original.config,event_family_policy='audio_semantic_v2'),original.codec).eval()
    with torch.no_grad():
        model.family_history_projection.weight.normal_(0,.1)
        model.audio_event_family_head.weight.normal_(0,.1)
    out=model(batch)
    results=[]
    for alpha in (0.,1.,.25):
        model.config=replace(model.config,event_family_audio_weight=alpha)
        _,metrics=model.loss(out,batch)
        results.append(metrics)
        torch.testing.assert_close(metrics['shape_loss'],
            metrics['event_family_loss']+metrics['event_route_conditional_loss'])
    torch.testing.assert_close(results[2]['configuration_loss'],
        .75*results[0]['configuration_loss']+.25*results[1]['configuration_loss'])
    torch.testing.assert_close(results[2]['event_family_loss'],
        .75*results[2]['event_family_full_loss']+.25*results[2]['event_family_audio_loss'])
