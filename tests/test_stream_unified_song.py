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
    expected=-torch.logsumexp(prior.log_softmax(-1)-(conditional.sum(1)+.5*profile)/2,-1).sum()
    assert torch.allclose(metrics['unified_balanced_variational_objective'],expected,atol=1e-7)
    energy=conditional.sum(1)+.5*profile
    posterior=(prior.log_softmax(-1)-energy/2).log_softmax(-1).detach()
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


def test_unified_rescaling_data_mass_and_normalizer_keeps_posterior_and_gradients():
    results=[]
    for normalizer in (1.,1000.):
        prior=torch.zeros(1,2,requires_grad=True)
        nll=torch.tensor([[[.1,2.]]],requires_grad=True)
        phrase=torch.tensor([[[[3.,-3.],[-3.,3.]]]])
        loss,metrics=unified_profile_response_loss(nll,phrase,dict(unified_chart_inverse=torch.tensor([0])),torch.full((1,1),normalizer),normalizer,
            prior_logits=prior,component_profile_nll=torch.zeros(1,2),profile_chart_weights=torch.ones(1),profile_weight=0.)
        loss.backward()
        results.append((prior.grad.clone(),nll.grad.clone()))
        assert metrics['unified_prior_weight_mass'].item()==1
    assert torch.allclose(results[0][0],results[1][0],atol=1e-7)
    assert results[0][0].norm()>.01
    assert torch.allclose(results[0][1],results[1][1],atol=1e-6)


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


def _scalar_variational_minimum(energies,prior_probability,normalizer,chart_weight):
    """Independent 1-D minimization of the stated objective, not a softmax oracle."""
    import math
    def objective(r):
        q=prior_probability
        entropy=(r*math.log(r/q) if r else 0.)+((1-r)*math.log((1-r)/(1-q)) if r<1 else 0.)
        return (r*energies[0]+(1-r)*energies[1])/normalizer+chart_weight*entropy
    lo,hi=1e-12,1-1e-12
    # Golden-section search evaluates the original scalar loss directly.
    ratio=(math.sqrt(5)-1)/2
    for _ in range(110):
        left=hi-ratio*(hi-lo);right=lo+ratio*(hi-lo)
        if objective(left)<objective(right):hi=right
        else:lo=left
    r=(lo+hi)/2
    return r,objective(r),objective


def test_unified_e_step_minimizes_actual_chart_weighted_elbo_and_matches_independent_gradients():
    import math
    prior=torch.tensor([[math.log(.6),math.log(.4)]],requires_grad=True)
    numerical=torch.tensor([[1.,2.]],requires_grad=True)
    mode=torch.tensor([[[0.,math.log(4.)]]],requires_grad=True)
    phrase=torch.tensor([[[[math.log(.8),math.log(.2)],[math.log(.2),math.log(.8)]]]])
    loss,metrics=unified_profile_response_loss(mode,phrase,dict(unified_chart_inverse=torch.tensor([0])),torch.ones(1,1),5.,
        prior_logits=prior,component_profile_nll=numerical,profile_chart_weights=torch.tensor([.3]),profile_weight=.2)
    energies=[-math.log(.85)+.3,-math.log(.4)+.6]
    best,minimum,objective=_scalar_variational_minimum(energies,.6,5.,.3)
    assert loss.item()==pytest.approx(minimum,abs=2e-7)
    assert metrics['unified_balanced_variational_objective'].item()==pytest.approx(minimum,abs=2e-7)
    old=(.6*math.exp(-energies[0]))/(.6*math.exp(-energies[0])+.4*math.exp(-energies[1]))
    assert objective(old)>minimum+1e-4
    loss.backward()
    assert numerical.grad[0,0].item()/(.3*.2)==pytest.approx(best,abs=3e-6)
    # Differentiate the independently minimized scalar objective by finite
    # differences, including the posterior's re-optimization at both endpoints.
    def oracle(logit_shift=0.,numerical_shift=0.,mode_shift=0.):
        q=.6*math.exp(logit_shift)/(.6*math.exp(logit_shift)+.4)
        e=[-math.log(.8*math.exp(-mode_shift)+.2/4)+.3*(1+numerical_shift),
           -math.log(.2*math.exp(-mode_shift)+.8/4)+.6]
        return _scalar_variational_minimum(e,q,5.,.3)[1]
    h=1e-4
    expected_prior=(oracle(logit_shift=h)-oracle(logit_shift=-h))/(2*h)
    expected_numerical=(oracle(numerical_shift=h)-oracle(numerical_shift=-h))/(2*h)
    expected_mode=(oracle(mode_shift=h)-oracle(mode_shift=-h))/(2*h)
    assert prior.grad[0,0].item()==pytest.approx(expected_prior,abs=3e-6)
    assert numerical.grad[0,0].item()==pytest.approx(expected_numerical,abs=3e-6)
    assert mode.grad[0,0,0].item()==pytest.approx(expected_mode,abs=3e-6)


def test_zero_chart_weight_preserves_event_evidence_and_no_evidence_costs_zero():
    import math
    prior=torch.tensor([[math.log(.6),math.log(.4)]],requires_grad=True)
    numerical=torch.tensor([[5.,10.]],requires_grad=True)
    mode=torch.tensor([[[0.,math.log(4.)]]],requires_grad=True)
    phrase=torch.tensor([[[[math.log(.8),math.log(.2)],[math.log(.2),math.log(.8)]]]])
    kwargs=dict(prior_logits=prior,component_profile_nll=numerical,profile_chart_weights=torch.zeros(1),profile_weight=1.)
    loss,metrics=unified_profile_response_loss(mode,phrase,dict(unified_chart_inverse=torch.tensor([0])),torch.ones(1,1),5.,**kwargs)
    assert loss.item()==pytest.approx(-math.log(.85)/5,abs=1e-7)
    assert metrics['unified_zero_prior_mass_event_charts'].item()==1
    assert metrics['unified_prior_kl'].item()==0
    assert bool(torch.isfinite(metrics['unified_posterior_entropy']))
    loss.backward()
    assert prior.grad is not None and torch.count_nonzero(prior.grad)==0
    assert mode.grad is not None and mode.grad.abs().sum()>0
    empty,empty_metrics=unified_profile_response_loss(mode.detach(),phrase,dict(unified_chart_inverse=torch.tensor([0])),torch.zeros(1,1),5.,**kwargs)
    assert empty.item()==0
    assert empty_metrics['unified_zero_prior_mass_event_charts'].item()==0
    assert empty_metrics['unified_posterior_entropy'].item()==pytest.approx(-.6*math.log(.6)-.4*math.log(.4),abs=1e-7)
    for invalid in (-.1,float('nan'),float('inf')):
        with pytest.raises(ValueError,match='Invalid unified chart weight'):
            unified_profile_response_loss(mode.detach(),phrase,dict(unified_chart_inverse=torch.tensor([0])),torch.ones(1,1),5.,
                **{**kwargs,'profile_chart_weights':torch.tensor([invalid])})


def test_budget_correction_keeps_persistent_phrase_mode_joint_before_chart_scaling():
    import math
    nll=torch.tensor([[[0.,math.log(4.)],[math.log(4.),0.]]])
    phrase=torch.tensor([[[[math.log(.9),math.log(.1)],[math.log(.1),math.log(.9)]],
                          [[math.log(.9),math.log(.1)],[math.log(.1),math.log(.9)]]]])
    batch=dict(unified_chart_inverse=torch.tensor([0]),training_phrase_group=torch.zeros(1,2,dtype=torch.long),
        training_phrase_complete=torch.ones(1,2,dtype=torch.bool),training_phrase_position=torch.tensor([[0,1]]),
        training_phrase_group_weight=torch.ones(1))
    loss,_=unified_profile_response_loss(nll,phrase,batch,torch.ones(1,2),2.,
        prior_logits=torch.tensor([[.7,-.2]]),component_profile_nll=torch.zeros(1,2),
        profile_chart_weights=torch.tensor([.25]),profile_weight=0.)
    # Either single persistent m explains both events with likelihood 1/4.
    # Per-event mode hopping or dividing the phrase score before marginalizing
    # would yield a different value.
    assert loss.item()==pytest.approx(math.log(4.)/2,abs=2e-7)


def _balanced_phrase_call(nll,logits,*,normalizer=None,raw=False):
    length=nll.size(1)
    batch=dict(unified_chart_inverse=torch.tensor([0]),training_phrase_group=torch.zeros(1,length,dtype=torch.long),
        training_phrase_complete=torch.ones(1,length,dtype=torch.bool),training_phrase_position=torch.arange(length)[None],
        training_phrase_group_weight=torch.ones(1))
    return unified_profile_response_loss(nll,logits,batch,torch.ones(1,length),float(length) if normalizer is None else normalizer,
        prior_logits=torch.zeros(1,1),component_profile_nll=torch.zeros(1,1),profile_chart_weights=torch.ones(1),
        profile_weight=0.,raw_joint_diagnostics=raw)


def test_repeating_same_phrase_distribution_preserves_mode_posterior_and_prior_gradient_budget():
    import math
    results=[]
    for length in (1,64):
        mode=torch.tensor([math.log(.7),math.log(.3)],requires_grad=True)
        logits=mode[None,None,None,:].expand(1,length,1,2)
        nll=torch.tensor([.2,2.])[None,None,:].expand(1,length,2)
        loss,metrics=_balanced_phrase_call(nll,logits,raw=True)
        loss.backward()
        results.append((loss.detach(),mode.grad.clone(),metrics))
        original=-math.log(.7*math.exp(-length*.2)+.3*math.exp(-length*2.))/length
        assert metrics['unified_original_sequence_joint_nll'].item()==pytest.approx(original,abs=1e-6)
    assert torch.allclose(results[0][0],results[1][0],atol=1e-7)
    assert torch.allclose(results[0][1],results[1][1],atol=1e-7)
    assert results[1][1].norm()>.01
    for metric in ('unified_phrase_mode_posterior_entropy','unified_phrase_mode_prior_kl'):
        assert results[0][2][metric].item()==pytest.approx(results[1][2][metric].item(),abs=1e-7)


def test_balanced_phrase_objective_scales_linearly_with_phrase_mass():
    import math
    values=[]
    for length in (1,7):
        mode=torch.tensor([math.log(.7),math.log(.3)],requires_grad=True)
        logits=mode[None,None,None,:].expand(1,length,1,2)
        nll=torch.tensor([.2,2.])[None,None,:].expand(1,length,2)
        loss,_=_balanced_phrase_call(nll,logits,normalizer=1.)
        loss.backward();values.append((float(loss),mode.grad.clone()))
    assert values[1][0]==pytest.approx(7*values[0][0],abs=2e-6)
    assert torch.allclose(values[1][1],7*values[0][1],atol=1e-6)


def test_balanced_phrase_single_mode_and_identical_mode_errors_restore_weighted_event_loss():
    for modes in (1,4):
        nll=torch.tensor([.2,.7,1.3])[None,:,None].expand(1,3,modes)
        logits=torch.arange(modes,dtype=torch.float32)[None,None,None,:].expand(1,3,1,modes)
        loss,metrics=_balanced_phrase_call(nll,logits)
        assert loss.item()==pytest.approx((.2+.7+1.3)/3,abs=2e-7)
        assert metrics['unified_phrase_mode_prior_kl'].abs()<2e-7


def test_balanced_phrase_mode_cannot_hop_between_events_and_raw_diagnostic_is_opt_in():
    import math
    nll=torch.tensor([[[0.,math.log(4.)],[math.log(4.),0.]]])
    logits=torch.zeros(1,2,1,2)
    loss,metrics=_balanced_phrase_call(nll,logits)
    assert loss.item()==pytest.approx(math.log(4.)/2,abs=2e-7)
    hopping=-math.log(.5+.5/4)
    assert loss.item()>hopping+.1
    assert 'unified_original_sequence_joint_nll' not in metrics
    assert 'unified_exact_joint_nll' not in metrics
    assert torch.allclose(loss,metrics['unified_balanced_variational_objective'],atol=1e-7)
