"""Actual v5 curve construction must never calibrate with unused plan rows."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from maichart.composer.stream_generate import _Curve
from maichart.composer import stream_curve
from maichart.generation.conditions import GenerationCondition
from maichart.generation.music_intensity import SCHEMA


def profile():
    points=[dict(time_sec=0.,intensity=.5),dict(time_sec=2.,intensity=.5)]
    return dict(schema=SCHEMA,duration_sec=2.,method='learned_chart_composer_event_set_stream_v1',
        checkpoint='old-v48.pt',edited=True,condition={'level':14.6},
        intensity_mapping_policy='bounded_odds_v1',baseline_head_rate=5.287,
        points=points,reference_points=deepcopy(points),rest_regions=[])


def curve(raw,mode='follow'):
    def forbidden(*args,**kwargs):raise AssertionError('v5 must not read partial plan/head channels')
    model=SimpleNamespace(config=SimpleNamespace(plan_kind_policy='unified_first_arrival_v5'),plan_head=forbidden)
    return _Curve(SimpleNamespace(intensity_curve_file=raw,intensity_curve_mode=mode,intensity_curve_strength=.7),
        SimpleNamespace(audio=SimpleNamespace(duration_sec=2.)),model,
        {'seconds':torch.tensor([[0.,1.]],dtype=torch.float64)},GenerationCondition(level=14.7))


def test_saved_unedited_old_checkpoint_curve_remains_exact_identity():
    c=curve(profile())
    assert not c.active and c.preview_only and not c.reference_rebased
    np.testing.assert_array_equal(c.event_bin_ratios,[1.])
    assert c.allows(.5,33)


@pytest.mark.parametrize('flag',[False,True])
def test_real_edits_remain_relative_across_changed_conditions(flag):
    raw=profile();raw['edited']=flag
    for point in raw['points']:point['intensity']=.6
    c=curve(raw)
    assert c.active and not c.reference_rebased
    # Finite cap changes the odds conversion; independently evaluate its
    # relative ratio instead of treating the intensity as a linear rate.
    offset=(64.+5.287)/64.
    ratio=(.6/(offset-.6))/(.5/(offset-.5))
    assert c.event_bin_ratios==pytest.approx([1.+.525*(ratio-1.)])
    assert 'native_event_rate' not in c.report()  # no fictitious Hz calibration


def test_hard_mode_does_not_build_fake_head_quota_from_untrained_channels():
    raw=profile();raw['points'][0]['intensity']=.6
    c=curve(raw,mode='hard')
    for _ in range(20):c.commit(.5,8)
    assert c.allows(.5,8)


def setup(monkeypatch,*,trained=False):
    def forbidden(*args,**kwargs):raise AssertionError('undefined plan readout used')
    model=SimpleNamespace(config=SimpleNamespace(plan_kind_policy='unified_first_arrival_v5',
        curve_preview_policy='audio_counts_v1' if trained else 'off',curve_preview_weight=.05,
        audio_grid_seconds=.05,plan_blank_window_seconds=.5),eval=lambda:None,plan_head=forbidden,
        curve_preview_trained_updates=7 if trained else 0,
        encode_audio=lambda *a:{'local':torch.zeros(1,2,1)},
        curve_preview_logits=lambda *a:torch.log(torch.expm1(torch.tensor([[[4.,2.],[4.,2.]]]))))
    features=np.zeros((20,98));features[:,93]=np.linspace(0.,1.,20)
    audio=SimpleNamespace(duration_sec=1.,tempo_bpm=120.,times=np.arange(20)*.05,features=features)
    monkeypatch.setattr(stream_curve,'load',lambda *a:(model,{'epoch':13,'optimizer_steps':7592}))
    monkeypatch.setattr(stream_curve,'extract_rational_audio',lambda *a,**kw:audio)
    monkeypatch.setattr(stream_curve,'audio_sha256',lambda *a:'test-sha')
    monkeypatch.setattr(stream_curve,'organize_song',lambda *a,**kw:SimpleNamespace(audio=audio,tempo=a[1],first=0.))
    monkeypatch.setattr(stream_curve,'build_audio_bank',lambda *a:(torch.zeros(2,1),torch.tensor([0.,.5]),torch.tensor([0.,1.])))


def test_v50_automatic_preview_reports_audio_fallback_without_reading_old_head(monkeypatch):
    setup(monkeypatch)
    p=stream_curve.generate_learned_profile('audio.wav','old.pt')
    assert p['model_prediction_available'] is False and p['preview_only']
    assert p['method']=='audio_reference_untrained_composer_preview_v1'
    assert p['checkpoint_identity']=={'epoch':13,'optimizer_steps':7592}
    assert p['points']==p['reference_points'] and not p['edited']


def test_new_trained_head_uses_rate_units_and_retains_identity(monkeypatch):
    setup(monkeypatch,trained=True)
    p=stream_curve.generate_learned_profile('audio.wav','new.pt')
    assert p['model_prediction_available']
    assert p['readout_policy']=='trained_audio_count_preview_v1'
    assert p['baseline_head_rate']==pytest.approx(4.)
    assert p['reference_event_rates']==pytest.approx([2.]*len(p['points']))
