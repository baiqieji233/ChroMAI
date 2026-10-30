import numpy as np
import pytest
import torch

from maichart.composer import curve, stream_curve


def test_plan_counts_use_real_cell_widths_and_do_not_create_rests():
    seconds=np.array([0., .05, .1])
    # Same head rate of 4/s and event rate of 2/s, including a short final cell.
    widths=np.array([.05,.05,.025])
    counts=np.column_stack((4*widths,2*widths))
    logits=np.column_stack((np.full(3,30.),np.log(np.expm1(counts))))
    result=stream_curve.profile_from_predictions(seconds,logits,.125,.05,
        metadata={'condition':{'level':13.},'bpm':120.,'first':.5})
    assert result['architecture_profile']=='event_set_stream_v1'
    assert result['baseline_head_rate']==pytest.approx(4.,rel=1e-6)
    assert result['reference_head_rates']==pytest.approx([4.]*len(result['points']),rel=1e-6)
    assert result['reference_event_rates']==pytest.approx([2.]*len(result['points']),rel=1e-6)
    assert result['rest_regions']==[]
    assert result['reference_blank_probabilities'][0]>.99
    assert result['reference_points']==result['points']
    assert result['condition']=={'level':13.}
    restored=curve.intensity_to_rate([point['intensity'] for point in result['points']],result['baseline_head_rate'])
    assert restored==pytest.approx(result['reference_head_rates'],rel=1e-6)


def test_legacy_curve_entry_dispatches_new_schema(monkeypatch):
    from maichart.composer import checkpoint
    monkeypatch.setattr(checkpoint,'read_payload',lambda path:{'schema':'maichart-composer-event-set-checkpoint-v1'})
    calls=[]
    monkeypatch.setattr(stream_curve,'generate_learned_profile',lambda *a,**kw:calls.append((a,kw)) or {'new':True})
    result=curve.generate_learned_profile('audio.wav','new.pt',condition={'level':14.},tempo_map=[{'beat':'0','bpm':123.}],first=.25)
    assert result=={'new':True}
    assert calls[0][1]['first']==.25
    assert calls[0][1]['condition']=={'level':14.}


def test_invalid_plan_cells_are_rejected():
    with pytest.raises(ValueError): stream_curve.profile_from_predictions([.1],torch.zeros(1,3),1.,.5)
    with pytest.raises(ValueError): stream_curve.profile_from_predictions([0.,0.],torch.zeros(2,3),1.,.5)
    with pytest.raises(ValueError): stream_curve.profile_from_predictions([0.],torch.full((1,3),float('nan')),1.,.5)


def test_rate_unit_head_does_not_divide_by_grid_width_again():
    rates=np.array([[4.,2.],[4.,2.],[4.,2.]])
    logits=np.column_stack((np.zeros(3),np.log(np.expm1(rates))))
    result=stream_curve.profile_from_predictions([0.,.01,.02],logits,.025,.01,plan_target_units='rates_per_second_v2')
    assert result['baseline_head_rate']==pytest.approx(4.,rel=1e-6)
    assert result['reference_head_rates']==pytest.approx([4.]*len(result['points']),rel=1e-6)
    assert result['reference_event_rates']==pytest.approx([2.]*len(result['points']),rel=1e-6)
    assert result['plan_target_units']=='rates_per_second_v2'


def test_interval_integration_removes_false_pulse_alias_without_losing_mass():
    seconds=np.arange(0.,8.,.01)
    heads=np.zeros(len(seconds))
    # Equal material per two-second interval; only alternating centers land
    # on the narrow onset pulse. Point sampling produces false on/off bins.
    for start in (0.,3.,4.,7.):
        index=int(round(start/.01));heads[index:index+5]=80.
    expected=np.column_stack((heads,heads/2.))
    # softplus has positive support; negligible floor models zero rate.
    logits=np.column_stack((np.zeros(len(seconds)),np.log(np.expm1(np.maximum(expected,1e-12)))))
    result=stream_curve.profile_from_predictions(seconds,logits,8.,2.,plan_target_units='rates_per_second_v2')
    assert result['curve_aggregation_policy']=='time_integrated_bins_v1'
    assert result['reference_head_rates']==pytest.approx([2.]*len(result['points']),abs=1e-5)
    assert result['reference_event_rates']==pytest.approx([1.]*len(result['points']),abs=1e-5)
    recovered=curve.intensity_to_rate([point['intensity'] for point in result['points']],result['baseline_head_rate'])
    assert recovered==pytest.approx(result['reference_head_rates'],abs=1e-6)
    total=np.dot(result['interval_head_rates'],np.diff(result['interval_edges_sec']))
    assert total==pytest.approx(float(np.sum(heads)*.01),abs=1e-5)


@pytest.mark.parametrize('units',['counts_per_audio_cell','rates_per_second_v2'])
def test_nonuniform_cells_partial_bin_and_real_section_change(units):
    seconds=np.array([0.,.1,.3,.8]);widths=np.diff(np.r_[seconds,1.1])
    rates=np.array([2.,4.,8.,16.]);target=np.column_stack((rates,rates/2.))
    values=target*widths[:,None] if units=='counts_per_audio_cell' else target
    logits=np.column_stack((np.zeros(4),np.log(np.expm1(values))))
    result=stream_curve.profile_from_predictions(seconds,logits,1.1,.5,plan_target_units=units)
    assert result['interval_head_rates']==pytest.approx([5.2,11.2,16.],rel=1e-6)
    assert result['reference_head_rates']==pytest.approx([5.2,5.2,11.2,16.,16.],rel=1e-6)
    assert np.dot(result['interval_head_rates'],np.diff(result['interval_edges_sec']))==pytest.approx(9.8,rel=1e-6)
    assert result['baseline_head_rate']==pytest.approx(9.8/1.1,rel=1e-6)
    assert result['rest_regions']==[]


def test_very_fine_resolution_stays_inside_editor_point_budget():
    result=stream_curve.profile_from_predictions([0.],torch.zeros(1,3),123.456789,1e-9,
        plan_target_units='rates_per_second_v2')
    assert len(result['points'])<=2048


def test_server_detects_new_composer_family():
    from maichart.webui.server import _checkpoint_model_family
    assert _checkpoint_model_family({'schema':'maichart-composer-event-set-checkpoint-v1',
                                      'family':'chart_composer_v1'})=='chart_composer_v1'


@pytest.mark.parametrize('saved_grid',[.01,.05])
def test_stream_curve_uses_saved_audio_grid(monkeypatch,saved_grid):
    from types import SimpleNamespace
    from maichart.event_timing import EventTempo
    from fractions import Fraction
    head=torch.nn.Linear(1,3)
    model=SimpleNamespace(config=SimpleNamespace(audio_grid_seconds=saved_grid),eval=lambda:None,
        plan_head=head,condition=lambda control:torch.zeros(1,1),
        encode_audio=lambda *args:{'local':torch.zeros(1,2,1)})
    audio=SimpleNamespace(duration_sec=.1,tempo_bpm=120.)
    song=SimpleNamespace(audio=audio,tempo=[EventTempo(Fraction(0),120.)],first=0.)
    observed=[]
    def bank(song,clock,grid):
        observed.append(grid)
        return torch.zeros(2,1),torch.tensor([0.,.05],dtype=torch.float64),torch.tensor([0.,.1],dtype=torch.float64)
    monkeypatch.setattr(stream_curve,'load',lambda *args:(model,{}))
    monkeypatch.setattr(stream_curve,'extract_rational_audio',lambda *args,**kw:audio)
    monkeypatch.setattr(stream_curve,'organize_song',lambda *args,**kw:song)
    monkeypatch.setattr(stream_curve,'build_audio_bank',bank)
    monkeypatch.setattr(stream_curve,'audio_sha256',lambda path:'test-audio-hash')
    result=stream_curve.generate_learned_profile('audio.wav','new.pt')
    assert observed==[saved_grid]
    assert result['audio_grid_seconds']==saved_grid
