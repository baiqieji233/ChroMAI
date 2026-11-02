"""Unified song latent contracts; run with the coordinated CPU test gate."""
import pytest
import torch
from maichart.composer.stream_song_profile import SongProfileCompleter
from maichart.composer.stream_song_response import SongResponse
from maichart.composer.stream_phrase_execution import unified_profile_response_loss


def test_unified_profile_prior_masks_full_supervision_and_uses_one_head_forward():
    torch.manual_seed(17);profile=SongProfileCompleter(8,3)
    audio=torch.randn(2,8);condition=torch.full((2,16),.3)
    mask=torch.zeros_like(condition,dtype=torch.bool);mask[:,0]=True;mask[:,8]=True
    changed=condition.clone();changed[:,1:8]=.8;changed[:,12]=.8
    calls=[];hook=profile.network.register_forward_hook(lambda *args:calls.append(1))
    a=profile.unified_evidence(audio,condition,mask)
    assert len(calls)==1
    b=profile.unified_evidence(audio,changed,mask);hook.remove()
    assert torch.equal(a['prior_logits'],b['prior_logits'])
    assert torch.equal(a['causal_condition'],b['causal_condition'])
    assert not torch.equal(a['component_profile_nll'],b['component_profile_nll'])
    assert a['component_profile_nll'].shape==(2,3)


def test_unified_training_mask_approximately_half_only_level():
    torch.manual_seed(9);profile=SongProfileCompleter(4,2)
    result=profile.unified_evidence(torch.zeros(1024,4),torch.full((1024,16),.3))
    known=result['known_mask']
    share=float((known.sum(-1)==1).float().mean())
    assert .45<share<.65 and bool(known[:,0].all())


def test_unified_response_reuses_supplied_prior_and_preserves_migration_weights():
    torch.manual_seed(5);old=SongResponse(8,4,16,3,2);module=SongResponse(8,4,16,3,2,unified=True)
    module.load_state_dict({key:value for key,value in old.state_dict().items() if not key.startswith('song.')},strict=True)
    assert not hasattr(module,'song')
    audio=torch.randn(2,3,8);mask=torch.ones(2,3,dtype=torch.bool);condition=torch.randn(2,16)
    causal=torch.zeros_like(condition);causal[:,0]=condition[:,0]
    logits=torch.randn(2,3,requires_grad=True);anchor=torch.randn(2,2,4)
    a=module(audio,mask,condition,anchor,prior_logits=logits,causal_condition=causal)
    b=module(audio,mask,condition+100,anchor,prior_logits=logits,causal_condition=causal)
    assert torch.equal(a['song_response_logits'],logits)
    assert torch.equal(a['song_response_phrase_logits'],b['song_response_phrase_logits'])
    with pytest.raises(ValueError,match='shared profile prior'):module.song_prior(audio,mask,condition)
    with pytest.raises(ValueError,match='masked causal controls'):module(audio,mask,condition,anchor,prior_logits=logits)


def test_unified_joint_likelihood_charges_one_prior_and_profile_per_chart():
    mode_nll=torch.tensor([[[.2,1.2],[.8,.3]]],requires_grad=True)
    phrase_logits=torch.tensor([[[[.2,.8],[.9,.1]],[[.7,.3],[.1,.9]]]],requires_grad=True)
    prior=torch.tensor([[.4,-.2]],requires_grad=True);profile=torch.tensor([[.3,1.1]],requires_grad=True)
    batch=dict(unified_chart_inverse=torch.tensor([0]),training_phrase_group=torch.tensor([[0,1]]),
        training_phrase_complete=torch.tensor([[True,True]]),training_phrase_position=torch.tensor([[0,0]]),
        training_phrase_group_weight=torch.ones(2))
    loss,metrics=unified_profile_response_loss(mode_nll,phrase_logits,batch,torch.ones(1,2),2.,
        prior_logits=prior,component_profile_nll=profile,profile_chart_weights=torch.ones(1),profile_weight=.25)
    conditional=-torch.logsumexp(phrase_logits.log_softmax(-1)-mode_nll[:,:,None,:],-1)
    expected=-torch.logsumexp(prior.log_softmax(-1)-conditional.sum(1)-.5*profile,-1).sum()/2
    assert torch.allclose(metrics['unified_exact_joint_nll'],expected,atol=1e-7)
    energy=conditional.sum(1)+.5*profile
    posterior=(prior.log_softmax(-1)-energy).log_softmax(-1).detach()
    objective=(posterior.exp()*energy).sum()/2+(posterior.exp()*(posterior-prior.log_softmax(-1))).sum()
    assert torch.allclose(loss,objective,atol=1e-7)
    loss.backward()
    assert all(t.grad is not None and bool(torch.isfinite(t.grad).all()) for t in (mode_nll,phrase_logits,prior,profile))
    split=dict(unified_chart_inverse=torch.tensor([0,0]),training_phrase_group=torch.tensor([[0],[1]]),
        training_phrase_complete=torch.ones(2,1,dtype=torch.bool),training_phrase_position=torch.zeros(2,1,dtype=torch.long),
        training_phrase_group_weight=torch.ones(2))
    other,_=unified_profile_response_loss(mode_nll.detach().transpose(0,1),phrase_logits.detach().transpose(0,1),split,torch.ones(2,1),2.,
        prior_logits=prior.detach(),component_profile_nll=profile.detach(),profile_chart_weights=torch.ones(1),profile_weight=.25)
    assert torch.allclose(loss,other,atol=1e-7)


def test_unified_component_override_controls_the_profile_without_resampling():
    profile=SongProfileCompleter(4,3);condition=torch.full((2,16),.3);known=torch.zeros_like(condition,dtype=torch.bool)
    values,diagnostics=profile.sample(torch.zeros(2,4),condition,known,component_choice=2,return_diagnostics=True)
    assert [row['selected_component'] for row in diagnostics['rows']]==[2,2]
    assert values.shape==(2,16)


def test_unified_prior_gradient_has_chart_budget_independent_of_target_normalizer():
    results=[]
    for normalizer in (1.,1000.):
        prior=torch.zeros(1,2,requires_grad=True)
        nll=torch.tensor([[[.1,2.]]],requires_grad=True)
        phrase=torch.tensor([[[[3.,-3.],[-3.,3.]]]])
        loss,metrics=unified_profile_response_loss(nll,phrase,dict(unified_chart_inverse=torch.tensor([0])),torch.ones(1,1),normalizer,
            prior_logits=prior,component_profile_nll=torch.zeros(1,2),profile_chart_weights=torch.ones(1),profile_weight=0.)
        loss.backward()
        results.append((prior.grad.clone(),nll.grad.clone()))
        assert metrics['unified_prior_weight_mass'].item()==1
    assert torch.allclose(results[0][0],results[1][0],atol=1e-7)
    assert results[0][0].norm()>.01
    assert torch.allclose(results[0][1],1000*results[1][1],atol=1e-6)


def test_deployed_auto_profile_freezes_same_choice_and_only_known_controls():
    from types import SimpleNamespace
    from maichart.generation.conditions import GenerationCondition
    from maichart.composer.stream_song_profile import resolve_song_profile
    model=SimpleNamespace(song_profile=SongProfileCompleter(4,3),config=SimpleNamespace(
        song_profile_policy='masked_deployment_joint_v3',song_response_policy='unified_profile_response_v2',song_response_modes=3))
    args=SimpleNamespace(song_style_source='auto',seed=123,song_response_mode=2)
    cache={'global':torch.zeros(1,2,4),'global_mask':torch.ones(1,2,dtype=torch.bool)}
    with torch.no_grad():
        condition,report=resolve_song_profile(model,cache,GenerationCondition(level=13.7),args,{})
    state=cache['unified_style']
    assert state['choice']==report['unified_choice']==2
    assert state['known_mask'].sum()==1 and state['known_mask'][0,0]
    assert torch.count_nonzero(state['causal_condition'][0,1:])==0
    assert report['prediction_diagnostics']['rows'][0]['selected_component']==2
    assert report['training_score']=='unified_configuration_profile_generalized_em_plus_crps'
    # Full explicit controls change the prior input, not the single-choice contract.
    args.song_style_source='manual'
    with torch.no_grad():other,manual=resolve_song_profile(model,cache,condition,args,{})
    assert other.to_dict()==condition.to_dict()
    assert cache['unified_style']['known_mask'].all()
    assert cache['unified_style']['choice']==manual['unified_choice']==2
    for invalid in (-2,3):
        args.song_response_mode=invalid
        with pytest.raises(ValueError,match='Invalid unified whole-song choice'):
            resolve_song_profile(model,cache,condition,args,{})
