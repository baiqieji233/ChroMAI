import json
from fractions import Fraction as F
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from maichart.event_timing import EventPoint, EventTempo
from maichart.form import cli, generate, rational_generate
from maichart.form.codec import RationalCodec
from maichart.form.clock import AUDIO_CURVE_AXIS
from maichart.form.curve import decode_density, encode_density
from maichart.form.data import organize_song
from maichart.form.timing import MetricSpan
from maichart.generation.conditions import GenerationCondition
from test_form_pipeline import fixture_audio


QUALITY = 'musical_evidence_v1'


def _span(index, start, end):
    return MetricSpan(index, F(start), F(end), 4, 4, (), F(4), (), False, 1., 'test')


def _song(*, quality=QUALITY, bins=1., spans=None):
    return organize_song(fixture_audio(4), [EventTempo(F(0), 120)], 0.,
        spans=spans or [_span(0, 0, 8)], quality_profile=quality, curve_bin_seconds=bins)


def _profile(*, components=True, intensity=1.):
    profile = {'time_axis':AUDIO_CURVE_AXIS, 'curve_resolution_sec':2.,
        'quality_profile':QUALITY, 'density_profile':'musical_log_v1',
        'points':[{'time_sec':0.,'intensity':intensity},{'time_sec':4.,'intensity':intensity}]}
    if components:
        profile['components'] = {'event_density':[.5,.5], 'tap':[.8,.8],
            'sustain':[.1,.1], 'slide_touch':[.1,.1]}
    return profile


def test_curve_condition_uses_training_bin_and_scales_physical_density_before_encoding():
    song = _song()
    values, mask, clock = generate._curve_condition(song, song.spans[0], _profile())
    torch.testing.assert_close(clock, torch.tensor([.5,1.5,2.5,3.5]))
    assert mask.all()
    # Intensity 1 with follow span .5 requests 1.5 times the eight-head/s baseline.
    np.testing.assert_allclose(decode_density(values[:,0], 'musical_log_v1'), 12., rtol=1e-6)
    np.testing.assert_allclose(values[:,1:].numpy(), np.tile([.8,.1,.1], (4,1)), rtol=1e-6)
    assert values[0,0].item() == pytest.approx(float(encode_density(12., 'musical_log_v1')))


def test_scalar_only_curve_modulates_predicted_baseline_and_retains_type_information():
    song = _song()
    song.load_curve_times_sec = np.array([.5,1.5,2.5,3.5], dtype=np.float32)
    song.load_curve = np.tile([.5,.8,.1,.1], (4,1)).astype(np.float32)
    values, mask, _ = generate._curve_condition(song, song.spans[0], _profile(components=False, intensity=.5))
    np.testing.assert_allclose(values.numpy(), song.load_curve)
    assert mask.all()
    song.load_curve = None
    with pytest.raises(ValueError, match='predicted song workload'):
        generate._curve_condition(song, song.spans[0], _profile(components=False))


def test_legacy_curve_keeps_editor_clock_and_original_linear_scaling():
    song = _song(quality='legacy', bins=2.)
    values, _, clock = generate._curve_condition(song, song.spans[0], _profile())
    torch.testing.assert_close(clock, torch.tensor([1.,3.]))
    torch.testing.assert_close(values[:,0], torch.tensor([.75,.75]))


@pytest.mark.parametrize(('curve','expected_budget'), [(None,100),(_profile(),148)])
def test_rational_generation_passes_planner_clock_and_decodes_budget_density(monkeypatch, curve, expected_budget):
    song = _song()
    planner_calls, compose_calls = [], []
    monkeypatch.setattr('maichart.form.rational_audio.planner_audio_queries',
        lambda song: (torch.zeros(2,120), np.array([1.,3.]), [F(2),F(6)]))

    class Planner:
        def __call__(self, audio, form, condition, *, audio_times_sec):
            planner_calls.append(audio_times_sec)
            assert form.shape[-1] == 515
            return {'load_curve':torch.zeros(1,2,4)}

    def compose(model, codec, example, atoms, **kwargs):
        compose_calls.append(kwargs)
        torch.testing.assert_close(example['span_bounds'], torch.tensor([8/64,4/20]))
        assert example['meta']['quality_profile'] == QUALITY
        if curve is not None:
            np.testing.assert_allclose(decode_density(example['load_curve'][:,0], 'musical_log_v1'), 12., rtol=1e-6)
        return [EventPoint(F(1),'1')], {'closed':True,'reason':'eos','tokens':10}

    monkeypatch.setattr(rational_generate, 'compose_span', compose)
    model = SimpleNamespace(config=SimpleNamespace(max_memory_tokens=128))
    points, report = rational_generate.generate_rational_chart(model, Planner(), RationalCodec(), song,
        GenerationCondition(), ['1'], style_balance_strength=0.,local_chord_strength=0., rational_grid_snap_ms=0.,
        intensity_curve=curve)
    torch.testing.assert_close(planner_calls[0], torch.tensor([[1.,3.]], dtype=torch.float64))
    np.testing.assert_allclose(song.load_curve_times_sec, [.5,1.5,2.5,3.5])
    assert compose_calls[0]['max_events'] == expected_budget  # Physical heads/s * 4 seconds * 3 + 4.
    assert points == [EventPoint(F(1),'1')]
    assert report['quality']['duration_semantics_profile'] == 'miacode_head_bpm_v1'


def test_planner_inference_preserves_fraction_second_neighborhood_boundary(monkeypatch):
    from maichart.form.alignment import temporal_neighbors

    song = _song()
    times = np.array([1/3,4/3], dtype=np.float64)
    monkeypatch.setattr('maichart.form.rational_audio.planner_audio_queries',
        lambda song: (torch.zeros(2,120), times, [F(2,3),F(8,3)]))

    class Planner:
        def __call__(self, audio, form, condition, *, audio_times_sec):
            assert audio_times_sec.dtype == torch.float64
            hidden = torch.tensor([[[1.],[3.]]])
            neighbors = temporal_neighbors(hidden, audio_times_sec, seconds=1.)
            torch.testing.assert_close(neighbors, torch.tensor([[[0.,3.],[1.,0.]]]))
            return {'load_curve':torch.zeros(1,2,4)}

    monkeypatch.setattr(rational_generate,'compose_span',lambda *args,**kwargs:
        ([EventPoint(F(1),'1')],{'closed':True,'reason':'eos','tokens':10}))
    model = SimpleNamespace(config=SimpleNamespace(max_memory_tokens=128))
    rational_generate.generate_rational_chart(model,Planner(),RationalCodec(),song,GenerationCondition(),['1'],
        style_balance_strength=0.,local_chord_strength=0.,rational_grid_snap_ms=0.)


def test_cli_propagates_checkpoint_evidence_curve_meter_and_quality_contract(tmp_path, monkeypatch):
    from maichart.form import checkpoint
    received = {}
    model = SimpleNamespace(config=SimpleNamespace(max_tokens=512,max_memory_tokens=128,
        quality_profile=QUALITY,curve_bin_seconds=.5,rhythm_geometry_profile='none'))
    payload = {'family':'song_form_v2','source_checkpoint_identity':{'sha256':'0'*64},
        'model_config':{},'atom_catalog':['1']}
    monkeypatch.setattr(checkpoint, 'load', lambda *args, **kwargs: (model,None,RationalCodec(),payload))

    def extract(*args, evidence_profile='legacy', **kwargs):
        received['evidence_profile'] = evidence_profile
        return fixture_audio(4)

    def validate(profile, **kwargs):
        received['validation'] = kwargs
        return []

    def compose(model, planner, codec, song, *args, **kwargs):
        received['song'] = song
        return [EventPoint(F(1),'1')], {}

    monkeypatch.setattr('maichart.form.rational_audio.extract_rational_audio', extract)
    monkeypatch.setattr('maichart.form.curve_validation.validate_saved_curve', validate)
    monkeypatch.setattr('maichart.generation.music_intensity.load_intensity_profile',
        lambda path: {**_profile(),'method':'learned_song_form_v2'})
    monkeypatch.setattr(rational_generate,'generate_rational_chart',compose)
    meter_file = tmp_path/'meter.json'
    meter_file.write_text(json.dumps([{'beat':'0','numerator':3,'denominator':4}]))
    output = tmp_path/'maidata.txt'
    assert cli.main(['generate','--audio','fixture.wav','--checkpoint','fixture.pt','--output',str(output),
        '--bpm','120','--meter-map',str(meter_file),'--intensity-curve-file','curve.json']) == 0
    assert received['evidence_profile'] == 'balanced_evidence_v1'
    assert received['song'].quality_profile == QUALITY
    assert received['song'].curve_bin_seconds == .5
    assert received['song'].relations.layout_descriptors.shape[-1] == 512
    assert received['validation']['quality_profile'] == QUALITY
    assert received['validation']['meter_map'][0].numerator == 3
    report = json.loads(output.with_suffix('.form.json').read_text(encoding='utf-8'))
    assert report['effective_generation_settings']['curve_bin_seconds'] == .5
    assert report['quality']['duration_semantics_profile'] == 'miacode_head_bpm_v1'


def test_revision_preserves_quality_curve_clock_and_span_bounds_across_regions(monkeypatch):
    song = _song(spans=[_span(0,0,4),_span(1,4,8)])
    points = [EventPoint(F(1),'1'),EventPoint(F(3),'2'),EventPoint(F(5),'3'),EventPoint(F(7),'4')]
    observed = []

    def arrange(model, codec, example, desired, atoms, **kwargs):
        observed.append(example)
        return [point for point in points if F(2)<=point.beat<F(6)], 0

    monkeypatch.setattr(generate,'arrange_span',arrange)
    monkeypatch.setattr(generate,'apply_patch_candidate', lambda original,*args,**kwargs:
        SimpleNamespace(points=tuple(original),accepted=False,reason='no_change',checks=1,edits=0,before=None,after=None))
    model = SimpleNamespace(config=SimpleNamespace(max_memory_tokens=128))
    result, _ = generate.revise_chart(model,RationalCodec(),song,points,GenerationCondition(),['1','2','3','4'],
        start=F(2),end=F(6))
    assert result == points
    assert observed[0]['meta']['quality_profile'] == QUALITY
    assert observed[0]['meta']['curve_bin_seconds'] == 1.
    torch.testing.assert_close(observed[0]['span_bounds'], torch.tensor([4/64,2/20]))
    assert observed[0]['load_curve'].shape == (2,4)
