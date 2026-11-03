"""Explicit curves transform arrival distributions and retain provenance."""
from types import SimpleNamespace
import copy
import numpy as np
import pytest
import torch
from maichart.composer.stream_generate import _Curve
from maichart.generation.conditions import GenerationCondition


def fixture(target=.25,reference=.5,strength=.65,native_heads=4.,condition=None,saved=None,rests=None,edges=None):
    condition=condition or GenerationCondition(level=13.7,radar={'note_density':.329})
    saved=saved or condition
    raw=dict(schema='maichart-intensity-curve-v1',duration_sec=4.,baseline_head_rate=4.,
        intensity_mapping_policy='legacy_linear_v1',points=[{'time_sec':0.,'intensity':target},{'time_sec':4.,'intensity':target}],
        reference_points=[{'time_sec':0.,'intensity':reference},{'time_sec':4.,'intensity':reference}],
        condition=saved.to_dict(),rest_regions=rests or [])
    if edges is not None:raw['interval_edges_sec']=edges
    head=torch.nn.Linear(1,3)
    with torch.no_grad():
        head.weight.zero_();head.bias.copy_(torch.tensor([0.,np.log(np.expm1(native_heads)),np.log(np.expm1(native_heads/2))]))
    model=SimpleNamespace(plan_head=head,condition=lambda x:x.new_zeros(x.size(0),1),
        config=SimpleNamespace(plan_target_units='rates_per_second_v2'))
    cache={'local':torch.zeros(1,4,1),'seconds':torch.arange(4,dtype=torch.float64)[None]}
    args=SimpleNamespace(intensity_curve_file=raw,intensity_curve_mode='follow',intensity_curve_strength=strength)
    return _Curve(args,SimpleNamespace(audio=SimpleNamespace(duration_sec=4.)),model,cache,condition)


def test_constant_rate_edit_changes_relative_near_far_arrival_odds():
    curve=fixture()
    bias=curve.timing_bias(0.,np.array([.25,1.5]))
    assert bias[1]>bias[0]
    expected=(curve.native_hazard[0]-curve.adjusted_hazard[0])*1.25
    assert bias[1]-bias[0]==pytest.approx(expected)
    note,tail=curve.kind_bias(0.,2.)
    assert note<0 and tail>0


def test_disabled_strength_is_exact_identity_even_with_condition_rebase():
    curve=fixture(strength=0.,condition=GenerationCondition(level=14.),saved=GenerationCondition(level=13.))
    assert not curve.active
    assert np.array_equal(curve.timing_bias(0.,np.array([0.,1.,4.])),np.zeros(3))
    assert curve.kind_bias(0.,2.)==(0.,0.)
    np.testing.assert_array_equal(curve.native_hazard,curve.adjusted_hazard)


def test_mismatched_conditions_rebase_to_saved_absolute_head_target():
    old=GenerationCondition(level=13.7,radar={'note_density':.329})
    new=GenerationCondition(level=13.7,radar={'note_density':.984375})
    curve=fixture(target=.5,reference=.5,native_heads=8.,condition=new,saved=old)
    assert curve.reference_rebased and curve.active
    assert curve.controller.reference_times.size==0
    # Saved curve requests 4heads/s; rebasing must reduce the new native8/s.
    assert curve.head_rate[0]==pytest.approx(8.+.65*.75*(4.-8.))
    assert curve.report()['adjusted_event_rate']['mean_per_second']<4.
    assert curve.report()['target_head_rate']['mean_per_second']==pytest.approx(4.)


def test_same_condition_unedited_curve_stays_exact_identity():
    curve=fixture(target=.5,reference=.5,native_heads=8.)
    assert not curve.reference_rebased and not curve.active
    assert curve.kind_bias(0.,2.)==(0.,0.)
    np.testing.assert_array_equal(curve.native_hazard,curve.adjusted_hazard)


def test_rest_boundaries_and_zero_rates_keep_finite_biases():
    curve=fixture(target=0.,reference=.5,rests=[{'start_sec':.5,'end_sec':1.5}])
    assert np.isfinite(curve.timing_bias(0.,np.array([0.,.5,1.,1.5,4.]))).all()
    assert np.isfinite(curve.kind_bias(0.,4.)).all()
    assert curve._integral(1.5,True)-curve._integral(.5,True)==pytest.approx(0.)
    assert curve._integral(4.,True)>=0


def test_curve_rebase_keeps_native_pulse_odds_and_applies_absolute_bin_target():
    condition=GenerationCondition(level=14.)
    saved=GenerationCondition(level=13.)
    raw=dict(schema='maichart-intensity-curve-v1',duration_sec=4.,baseline_head_rate=4.,curve_bin_seconds=2.,
        intensity_mapping_policy='legacy_linear_v1',points=[{'time_sec':0.,'intensity':.5},{'time_sec':4.,'intensity':.5}],
        reference_points=[{'time_sec':0.,'intensity':.5},{'time_sec':4.,'intensity':.5}],condition=saved.to_dict())
    # Two seconds of low background with one 10ms high pulse, repeated.
    native=np.full(400,.01,dtype=np.float32);native[[50,250]]=100.
    inverse=np.log(np.expm1(native.astype(np.float64))).astype(np.float32)
    local=torch.as_tensor(inverse)[None,:,None]
    head=torch.nn.Linear(1,3,bias=False)
    with torch.no_grad():head.weight.copy_(torch.tensor([[0.],[1.],[1.]]))
    model=SimpleNamespace(plan_head=head,condition=lambda x:x.new_zeros(x.size(0),1),config=SimpleNamespace(plan_target_units='rates_per_second_v2'))
    cache={'local':local,'seconds':torch.arange(400,dtype=torch.float64)[None]/100}
    args=SimpleNamespace(intensity_curve_file=raw,intensity_curve_mode='follow',intensity_curve_strength=.65)
    curve=_Curve(args,SimpleNamespace(audio=SimpleNamespace(duration_sec=4.)),model,cache,condition)
    assert curve.reference_rebased and curve.active
    # Every native cell in a bin receives the same multiplier; the pulse
    # retains its relative likelihood against nearby background.
    i=np.searchsorted(curve.hazard_edges,.5,side='right')-1
    j=np.searchsorted(curve.hazard_edges,.51,side='right')-1
    assert curve.adjusted_hazard[i]/curve.adjusted_hazard[j]==pytest.approx(curve.native_hazard[i]/curve.native_hazard[j],rel=1e-5)
    assert curve.adjusted_hazard[i]>curve.adjusted_hazard[j]
    assert curve.report()['curve_application']=='bin_mean_multiplicative_v1'
    assert curve.report()['curve_bin_seconds']==2.


def test_curve_envelope_honors_explicit_edges_and_rest_integrates_exactly():
    curve=fixture(rests=[{'start_sec':.5,'end_sec':1.5}],edges=[0.,1.25,4.])
    assert curve._integral(1.5,True)-curve._integral(.5,True)==pytest.approx(0.)
    assert 1.25 in curve.hazard_edges
    assert curve.report()['curve_scale_source']=='interval_edges_sec'
