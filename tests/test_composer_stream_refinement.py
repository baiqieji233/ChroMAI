import copy
import math

import torch
from torch.nn import functional as F

from maichart.composer.event_set_codec import EventSetCodec
from maichart.composer.metric_totals import MetricTotals
from maichart.composer.stream_model import StreamComposer,StreamComposerConfig,NOTE,WAIT
from maichart.composer.stream_train import _record_recovery_metrics,_averages
from test_composer_stream_model import fixture


def test_subbeat_delta_score_reads_projected_candidate_audio_and_its_gradient():
    model,_=fixture()
    with torch.no_grad():model.delta_query.weight.zero_();model.delta_query.bias.fill_(1)
    keys=torch.zeros(1,12,32);keys[0,2,0]=2;keys.requires_grad_()
    cache={'delta_keys':keys,'mask':torch.ones(1,12,dtype=torch.bool),
           'beats':torch.arange(12)[None].double()/4}
    scores=model.timing_audio_scores(torch.zeros(1,1,model.config.d_model),
                                    torch.zeros(1,1,dtype=torch.float64),cache,torch.tensor([0]))
    assert scores[0,0,2]>0  # +1/2 beat sees its own transient, not only an integer probe.
    assert scores[0,0,3]==0
    scores[0,0,2].backward()
    assert keys.grad[0,2].abs().sum()>0 and keys.grad[0,0].eq(0).all()
    assert keys.size(-1)==32
    assert model.near_offsets.tolist()==[0,.25,.5,.75,1,1.25,1.5,1.75,2,3,4,6,8]


def test_musical_clocks_have_exact_intended_periods_and_global_step_positions():
    model,_=fixture()
    periods=[.25,.5,1,2,3,4]
    for i,period in enumerate(periods):
        values=model._clock(torch.zeros(3),torch.tensor([0.,period/4,period]))
        torch.testing.assert_close(values[:,2+i],torch.tensor([0.,1.,0.]),atol=1e-6,rtol=0)
        torch.testing.assert_close(values[:,8+i],torch.tensor([1.,0.,1.]),atol=1e-6,rtol=0)
    assert not torch.equal(model._step_position(torch.tensor([100])),model._step_position(torch.tensor([101])))


def detail_fixture():
    base,_=fixture()
    codec=EventSetCodec.fit(['1h[4:1]/5h[4:1]','1h[8:1]/5h[8:1]','1h[4:1]/5h[8:1]'])
    model=StreamComposer(base.config,codec).eval()
    event=codec.encode('1h[4:1]/5h[4:1]');shape=event['shape']
    context=torch.randn(1,1,model.config.d_model,requires_grad=True)
    output={'kind_logits':torch.zeros(1,1,5,requires_grad=True),'delta_logits':torch.zeros(1,1,5,requires_grad=True),
        'shape_logits':torch.zeros(1,1,codec.num_shapes,requires_grad=True),
        'phase_logits':torch.randn(1,1,8,requires_grad=True),'action_context':context}
    batch={'loss_mask':torch.ones(1,1),'sample_weight':torch.ones(1,1),'target_kind':torch.tensor([[NOTE]]),
        'target_delta':torch.tensor([[2]]),'target_shape':torch.tensor([[shape]]),
        'target_phase_mask':torch.tensor([[event['phase_mask']]]),'target_details':torch.tensor([[event['details']]]),
        '_detail_groups':[(shape,torch.tensor([0]))],'_literal_indices':torch.empty(0,dtype=torch.long)}
    return model,event,output,batch


def test_phase_and_autoregressive_details_are_one_joint_marginal_likelihood():
    model,event,out,batch=detail_fixture()
    loss,metrics=model.loss(out,batch)
    phases=torch.arange(8)
    details=batch['target_details'][0].expand(8,-1)
    logdetail=model.regular_detail_log_probs(out['action_context'][0].expand(8,-1),event['shape'],phases,details)
    allowed=torch.tensor(event['phase_mask'])
    expected=-torch.logsumexp((out['phase_logits'][0,0].log_softmax(-1)+logdetail)[allowed],0)/model.config.rollout_steps
    torch.testing.assert_close(metrics['phase_loss']+metrics['detail_loss'],expected.detach())
    loss.backward()
    assert model.detail_cell.weight_hh.grad.abs().sum()>0
    assert model.detail_input.weight.grad.abs().sum()>0
    assert out['phase_logits'].grad.abs().sum()>0


def test_later_detail_depends_on_previous_detail_and_actual_phase():
    model,event,out,batch=detail_fixture()
    captured=[]
    hook=model.detail_output.register_forward_hook(lambda mod,args,value:captured.append(value.detach()))
    target=batch['target_details'][0].clone()
    model.regular_detail_log_probs(out['action_context'][0],event['shape'],torch.tensor([0]),target)
    first=captured[-1].clone()
    target[:,0]=1-target[:,0]
    model.regular_detail_log_probs(out['action_context'][0],event['shape'],torch.tensor([0]),target)
    changed=captured[-1].clone()
    assert not torch.equal(first,changed)
    model.regular_detail_log_probs(out['action_context'][0],event['shape'],torch.tensor([4]),target)
    assert not torch.equal(changed,captured[-1])
    hook.remove()


def test_rate_plan_integrates_real_seconds_and_detaches_kind_prior():
    model,_=fixture()
    cache={'seconds':torch.arange(6)[None].double()/2,'beats':torch.arange(6)[None].double(),
           'mask':torch.ones(1,6,dtype=torch.bool)}
    logits=torch.full((1,6,3),math.log(math.expm1(2.)),requires_grad=True)
    batch={'step_second':torch.tensor([[0.]],dtype=torch.float64),'step_beat':torch.tensor([[0.]],dtype=torch.float64),
           'audio_end_sec':torch.tensor([3.],dtype=torch.float64),'audio_end_beat':torch.tensor([6.],dtype=torch.float64)}
    prior,features,widths=model._plan_at_cursor(logits,cache,batch,torch.tensor([0]))
    torch.testing.assert_close(prior[0,0,WAIT],torch.tensor(-2.))
    expected_note=torch.tensor(math.log1p(-math.exp(-2.)))
    torch.testing.assert_close(prior[0,0,NOTE],expected_note)
    assert not prior.requires_grad and not features.requires_grad
    torch.testing.assert_close(widths,torch.full((1,6),.5))


def test_plan_rate_loss_respects_exposure_and_separate_blank_mask():
    model,batch=fixture();out=model(batch)
    out['plan_logits']=torch.zeros_like(out['plan_logits'],requires_grad=True)
    batch['audio_plan_blank_weight']=torch.zeros_like(batch['audio_plan_weight'])
    batch['audio_plan_target'].zero_()
    batch['audio_plan_exposure']=torch.full_like(batch['audio_plan_weight'],.01)
    _,first=model.loss(out,batch)
    batch['audio_plan_exposure']*=2
    _,second=model.loss(out,batch)
    # Zero-count Poisson contributes lambda per second: the exposure in its
    # expected count cancels the exposure in the temporal normalization.
    torch.testing.assert_close(second['plan_loss'],first['plan_loss'])
    assert first['plan_loss']>0  # Unknown centered blank does not erase known cell count learning.


def test_target_time_fast_load_decays_and_exposes_last_button_gap():
    model,batch=fixture()
    now=model._prefix_load(batch['history_atoms'],batch['history_kind'],batch['step_second'],batch['step_second'])
    later=model._prefix_load(batch['history_atoms'],batch['history_kind'],batch['step_second'],batch['step_second']+1.)
    assert later[0,1,0]<now[0,1,0]  # 0.15-second decay.
    assert later[0,1,24]>now[0,1,24]  # Last-gap channel.
    assert torch.isfinite(later).all()


def test_motion_orbit_pushforward_and_zero_initialized_continuation():
    model,batch=fixture()
    model.config.motion_continuation_policy='motion_distinct_v2'
    events=[model.codec.encode(x) for x in ('1/5','2/6')]
    kinds=torch.tensor([[NOTE,NOTE,WAIT]])
    shapes=torch.tensor([[events[0]['shape'],events[1]['shape'],0]])
    masks=torch.tensor([[events[0]['phase_mask'],events[1]['phase_mask'],[False]*8]])
    distribution=model.motion_phase_features(kinds,shapes,masks)
    assert distribution[0,0].eq(0).all()
    torch.testing.assert_close(distribution[0,1],distribution[0,2])
    for motion,indices in enumerate(((2,6),(0,4),(1,5))):
        expected=torch.zeros(8);expected[list(indices)]=.5
        torch.testing.assert_close(distribution[0,1,motion],expected)
    context=torch.randn(1,model.config.d_model);shape=torch.tensor([events[0]['shape']])
    refs=torch.tensor([1]);mask=masks[:,1]
    base=model.event_phase_logits(context,shape,refs,mask)
    fresh=model.event_phase_logits(context,shape,refs,mask,distribution[:,1])
    torch.testing.assert_close(base,fresh,atol=0,rtol=0)
    with torch.no_grad():model.motion_head.bias[0]=2
    changed=model.event_phase_logits(context,shape,refs,mask,distribution[:,1])
    torch.testing.assert_close(changed-base,2*distribution[:,1,0],atol=1e-6,rtol=1e-6)


def test_legacy_keep_gap_uses_previous_note_not_intervening_wait():
    model,batch=fixture()
    model.config.gap_continuation_policy='additive_v1'
    before=model(batch)['delta_logits']
    with torch.no_grad():model.keep_gap_head.bias.fill_(1)
    after=model(batch)['delta_logits']
    delta=after-before
    torch.testing.assert_close(delta[0,2,2],torch.tensor(1.))  # Previous NOTE's 1/2 beat ID.
    torch.testing.assert_close(delta[0,2,4],torch.tensor(0.))  # WAIT is not a rhythm target.


def test_real_recovery_backward_is_reported_once_on_effective_window_objective():
    model,batch=fixture();out=model(batch);loss,metrics=model.loss(out,batch)
    (loss/2).backward()
    window=MetricTotals();window.add(metrics);window.add(metrics)
    recovery,_=model.loss(model(batch),dict(batch,_loss_normalizer=1))
    weighted=.1*recovery;weighted.backward()
    _record_recovery_metrics(window,weighted,2)
    report=_averages(window,2)
    assert report['recovery_loss']>0
    assert abs(report['recovery_loss']-float(weighted.detach()))<1e-5
    assert abs(report['base_loss']-float(metrics['base_loss']))<1e-5
    assert abs(report['loss']-report['base_loss']-report['recovery_loss'])<1e-5


def test_prior_scratch_configuration_keeps_original_parameter_shapes():
    codec=EventSetCodec.fit(['1'])
    config=StreamComposerConfig.from_dict({'d_model':16,'nhead':4,'layers':1,'dim_feedforward':32})
    model=StreamComposer(config,codec)
    assert model.audio_clock.in_features==4 and model.cursor_clock.in_features==8
    assert model.near_audio.in_features==9*16
    assert hasattr(model,'detail_heads') and not hasattr(model,'detail_cell')
    assert not hasattr(model,'delta_audio_key') and not hasattr(model,'motion_head')
    assert not hasattr(model,'plan_kind_features')
    assert config.plan_target_units=='counts_per_audio_cell' and config.plan_blank_window_seconds==0
    assert config.plan_loss_normalization=='cell_mean_v1'
    restored=StreamComposer(StreamComposerConfig.from_dict(config.to_dict()),codec)
    restored.load_state_dict(model.state_dict(),strict=True)


def test_per_second_poisson_gradient_is_invariant_to_audio_grid_resolution():
    model,batch=fixture();original=model(batch)
    observed={}
    for normalization in ('cell_mean_v1','per_second_v2'):
        model.config.plan_loss_normalization=normalization
        for width in (.05,.01):
            cells=round(1./width)
            raw=torch.tensor(math.log(math.expm1(2.)),requires_grad=True)
            # One real second, one event with one head, plus an inert pad cell.
            logits=torch.stack((torch.zeros_like(raw),raw,raw)).reshape(1,1,3).expand(1,cells+1,3)
            target=torch.zeros(1,cells+1,3);target[0,0,1:]=1
            exposure=torch.full((1,cells+1),width);exposure[:,-1]=0
            labels=dict(batch,audio_plan_target=target,audio_plan_exposure=exposure,
                        audio_plan_weight=exposure,audio_plan_blank_weight=exposure)
            output={**original,'plan_logits':logits}
            loss,metrics=model.loss(output,labels)
            gradient=torch.autograd.grad(loss,raw)[0]/model.config.plan_weight
            observed[normalization,width]=(gradient.detach(),metrics)
            assert torch.isfinite(loss) and torch.isfinite(gradient)
    coarse,coarse_metrics=observed['per_second_v2',.05]
    fine,fine_metrics=observed['per_second_v2',.01]
    expected=torch.tensor(1.-math.exp(-2.))
    torch.testing.assert_close(coarse,expected,atol=2e-6,rtol=2e-6)
    torch.testing.assert_close(fine,coarse,atol=2e-6,rtol=2e-6)
    torch.testing.assert_close(fine_metrics['plan_blank_loss'],coarse_metrics['plan_blank_loss'])
    # Resolution changes the discrete count likelihood's additive log(dt)
    # constant, not its derivative with respect to the rate prediction.
    torch.testing.assert_close(fine_metrics['plan_rate_loss']-coarse_metrics['plan_rate_loss'],
                               torch.tensor(2*math.log(5.)),atol=3e-6,rtol=2e-6)
    old_coarse,_=observed['cell_mean_v1',.05]
    old_fine,_=observed['cell_mean_v1',.01]
    torch.testing.assert_close(old_fine,old_coarse*.2,atol=2e-6,rtol=2e-6)
    torch.testing.assert_close(fine_metrics['plan_loss'],fine_metrics['plan_rate_loss']+fine_metrics['plan_blank_loss'])


def test_plan_normalization_is_explicit_and_old_pilot_defaults_are_retained():
    fresh=StreamComposerConfig()
    assert fresh.plan_loss_normalization=='per_second_v2'
    pilot=fresh.to_dict();pilot.pop('plan_loss_normalization')
    loaded=StreamComposerConfig.from_dict(pilot)
    assert loaded.plan_target_units=='rates_per_second_v2'
    assert loaded.plan_loss_normalization=='cell_mean_v1'
    assert StreamComposerConfig.from_dict(fresh.to_dict()).plan_loss_normalization=='per_second_v2'
