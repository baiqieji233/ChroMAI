"""Candidate state and stable phrase intent reach v4 without decoder feedback."""
from dataclasses import replace
import math

import torch

from maichart.composer.stream_model import StreamComposer
from maichart.composer.stream_actions import StatePhraseFamilyResidual
from maichart.composer.stream_train import upgrade_learning_stage


def fixture():
    from test_stream_stationary_family import model_fixture
    old,batch,cache,seconds=model_fixture()
    cache={key:value.detach() if torch.is_tensor(value) else value for key,value in cache.items()}
    model=upgrade_learning_stage(old,{'event_family_policy':'audio_state_phrase_v4'}).eval()
    batch=dict(batch,action_state=torch.zeros(1,4,2,50),
               action_state_mask=torch.ones(1,4,2,dtype=torch.bool))
    phrase={'phrase_intent':torch.randn(1,4,16)}
    return old,model,batch,cache,seconds,phrase


def test_zero_residual_preserves_parent_distribution_and_small_parameter_budget():
    old,model,batch,cache,seconds,phrase=fixture()
    a=old.family_context_at(cache,seconds,torch.tensor([0]),batch)
    b=model.family_context_at(cache,seconds,torch.tensor([0]),batch,phrase_output=phrase)
    for key in ('full_logits','audio_logits','continuity_logits'):
        torch.testing.assert_close(a[key],b[key],atol=0,rtol=0)
    assert sum(p.numel() for p in model.family_state_phrase.parameters())<100000
    assert model.family_state_phrase.features[0].out_features<=128


def test_intent_and_occupancy_change_full_but_leave_pure_audio_unchanged():
    _,model,batch,cache,seconds,phrase=fixture()
    with torch.no_grad():model.family_state_phrase.output.weight.normal_(0,.2)
    read=lambda b,p:model.family_context_at(cache,seconds,torch.tensor([0]),b,phrase_output=p)
    a=read(batch,phrase)
    b=read(batch,{'phrase_intent':phrase['phrase_intent']+torch.randn_like(phrase['phrase_intent'])})
    occupied={**batch,'action_state':batch['action_state'].clone(),'action_state_mask':batch['action_state_mask'].clone()}
    occupied['action_state_mask'][...,0]=False
    occupied['action_state'][...,0,0]=1
    occupied['action_state'][...,0,46]=math.log1p(4)/math.log1p(60)
    c=read(occupied,phrase)
    for other in (b,c):
        torch.testing.assert_close(a['audio_logits'],other['audio_logits'],atol=0,rtol=0)
        assert not torch.equal(a['full_logits'],other['full_logits'])
        assert not torch.equal(a['continuity_logits'],other['continuity_logits'])


def test_new_correction_has_gradient_while_music_prior_keeps_its_own_budget():
    _,model,batch,cache,seconds,phrase=fixture()
    read=model.family_context_at(cache,seconds,torch.tensor([0]),batch,phrase_output=phrase)
    (-read['full_logits'][0,1,7]).backward()
    assert model.family_state_phrase.output.weight.grad.abs().sum()>0
    assert model.family_state_phrase.output.bias.grad[:31].abs().sum()>0
    assert model.family_state_phrase.output.bias.grad[31].abs()>0
    assert model.audio_event_family_head.weight.grad is None
    model.zero_grad(set_to_none=True)
    with torch.no_grad():model.family_state_phrase.output.weight.normal_(0,.1)
    intent=phrase['phrase_intent'].clone().requires_grad_()
    read=model.family_context_at(cache,seconds,torch.tensor([0]),batch,phrase_output={'phrase_intent':intent})
    (-read['full_logits'][0,1,7]).backward()
    assert intent.grad.abs().sum()>0
    assert model.family_state_phrase.features[0].weight.grad.abs().sum()>0


def test_candidate_occupancy_expires_at_actual_gap_and_masks_padding():
    _,_,batch,_,seconds,_=fixture()
    batch['action_state_mask'][...,0]=False
    batch['action_state'][...,0,0]=1
    batch['action_state'][...,0,46]=math.log1p(.4)/math.log1p(60)
    near=StatePhraseFamilyResidual.candidate_state(batch,batch['step_second']+.1)
    far=StatePhraseFamilyResidual.candidate_state(batch,batch['step_second']+.8)
    assert torch.all(near[...,0]==1)
    assert torch.equal(far,torch.zeros_like(far))
    batch['action_state'][...,1,:]=float('nan')
    torch.testing.assert_close(near,StatePhraseFamilyResidual.candidate_state(batch,batch['step_second']+.1))


def test_future_targets_and_decoder_hidden_do_not_enter_family_correction():
    _,model,batch,cache,seconds,phrase=fixture()
    with torch.no_grad():model.family_state_phrase.output.weight.normal_(0,.2)
    a=model.family_context_at(cache,seconds,torch.tensor([0]),batch,phrase_output=phrase)
    altered={**batch,'target_kind':torch.full((1,4),999),
        'target_phrase_features':torch.full((1,4,96),float('nan')),
        'target_phrase_sequence_atoms':torch.full((1,32,2),999)}
    b=model.family_context_at(cache,seconds,torch.tensor([0]),altered,
        phrase_output={**phrase,'hidden':torch.full((1,4,16),float('nan'))})
    torch.testing.assert_close(a['full_logits'],b['full_logits'],atol=0,rtol=0)
