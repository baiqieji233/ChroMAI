"""Main WebUI curve controls survive the real Composer CLI boundary."""
import json
from pathlib import Path
from types import SimpleNamespace
import time

import pytest

from maichart.composer.generate import parser
from maichart.webui import composer_api, server as webui
from test_intensity_curve_api import api, profile


def test_status_advertises_the_generation_adapter_contract(api):
    code, status, _ = api('/api/status')
    assert code == 200
    assert status['composer_generation_contract'] == composer_api.GENERATION_CONTRACT


@pytest.mark.parametrize('span', ['full', 'trim'])
@pytest.mark.parametrize('strength', [0., .7])
def test_curve_controls_parse_real_generation_cli(tmp_path, span, strength):
    audio = tmp_path/'song.wav'; audio.write_bytes(b'audio')
    checkpoint = tmp_path/'model.pt'; checkpoint.write_bytes(b'weights')
    curve = profile(audio)
    curve.update(baseline_head_rate=4., baseline_p95_rate=8.,
        intensity_mapping_policy='bounded_odds_v1', intensity_rate_cap=64.,
        reference_points=curve['points'], reference_event_rates=[1., 1.],
        reference_head_rates=[4., 4.])
    task = SimpleNamespace(task_id='curve', result={})
    calls = []
    runtime = SimpleNamespace(_create_task=lambda _:task,
        _run_subprocess_task=lambda task,args,**kw:calls.append(args))
    composer_api.start_generate(SimpleNamespace(_json_response=lambda _:None), {
        'audio':str(audio), 'checkpoint':str(checkpoint), 'output':str(tmp_path/'chart.txt'),
        'curve_strength':1.3, 'intensity_curve_enabled':True, 'intensity_curve':curve,
        'intensity_curve_strength':strength, 'intensity_curve_span':span,
        'intensity_curve_chord_multiplier':.75,
        'meter_map':[{'beat':'0','numerator':3,'denominator':4,'grouping':[3]}],
    }, runtime=runtime)
    args = parser().parse_args(calls[0][2:])
    assert args.curve_strength == 1.3
    assert args.intensity_curve_strength == strength
    assert args.intensity_curve_span == span
    assert args.intensity_curve_chord_multiplier == .75
    saved = json.loads(Path(args.intensity_curve_file).read_text(encoding='utf-8'))
    assert saved['baseline_head_rate'] == 4.
    assert saved['reference_points'] == curve['reference_points']
    assert saved['reference_event_rates'] == [1., 1.]
    assert json.loads(Path(args.meter_map).read_text(encoding='utf-8'))[0]['numerator'] == 3
    assert task.result['effective_settings']['intensity_curve_strength'] == strength


def test_composer_analysis_resolves_generation_condition_and_timing(api, monkeypatch, tmp_path):
    from maichart.composer import curve
    audio = tmp_path/'song.wav'; audio.write_bytes(b'audio')
    checkpoint = tmp_path/'model.pt'; checkpoint.write_bytes(b'weights')
    condition_file = tmp_path/'condition.json'
    condition_file.write_text(json.dumps({'style':{'slide_bias':.73}}))
    monkeypatch.setattr(webui,'_load_checkpoint_payload',lambda _: {
        'family':'chart_composer_v1','schema':'maichart-chart-composer-checkpoint-v2',
        'model_config':{'architecture_profile':'audio_grounded_v3'}})
    captured = {}
    def predict(*args, **kwargs):
        captured.update(kwargs)
        return {**profile(audio), 'method':'learned_audio_grounded_v3'}
    monkeypatch.setattr(curve,'generate_learned_profile',predict)
    code, started, _ = api('/api/intensity/analyze', {
        'audio':str(audio),'checkpoint':str(checkpoint),'curve_source':'model',
        'model_family':'chart_composer_v1','level':14.7,
        'condition':{'style':{'tap_bias':.19}},'condition_file':str(condition_file),
        'first':.35,'bpm':180.,'tempo_map':[{'beat':'0','bpm':180.},{'beat':'12','bpm':150.}],
        'meter_map':[{'beat':'0','numerator':3,'denominator':4,'grouping':[3]}],
    })
    assert code == 200, started
    deadline = time.monotonic()+4
    while time.monotonic()<deadline:
        _,result,_ = api('/api/task?id='+started['task_id'])
        if result['status'] not in {'queued','running'}:
            break
        time.sleep(.02)
    assert result['status'] == 'completed', result
    assert captured['level'] == 14.7
    assert captured['condition'].level == 14.7
    assert captured['condition'].style['slide_bias'] == .73
    assert captured['condition'].style['tap_bias'] == .19
    assert captured['first'] == .35 and captured['bpm'] == 180.
    assert captured['tempo_map'][1] == {'beat':'12','bpm':150.}
    assert captured['meter_map'][0]['numerator'] == 3
    assert result['result']['condition_source'] == 'ui+file'
