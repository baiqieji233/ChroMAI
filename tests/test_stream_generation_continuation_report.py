from fractions import Fraction as F
from types import SimpleNamespace

from maichart.composer.stream_data import StreamStep,NOTE,WAIT
from maichart.composer.stream_generate import continuation_observation,evaluate_panel


def steps(texts,gap=F(1,4)):
    return [StreamStep(NOTE,i*gap,(i+1)*gap,float(i*gap)/2,float((i+1)*gap)/2,text)
            for i,text in enumerate(texts)]


def test_observation_separates_single_button_alternation_and_slow_repeat():
    r=continuation_observation(steps(['1/2','1/3','1/2','1/3','1/2','1/3']))
    assert r['button_fast_run']['count'] == 6
    assert r['button_fast_run']['button'] == 1
    assert r['exact_period2_run']['count'] == 6
    assert r['same_configuration_run']['count'] == 1
    slow=continuation_observation(steps(['1','1','1','1'],F(1)))
    assert slow['same_configuration_run']['count'] == 4
    assert slow['button_fast_run']['count'] == 1
    assert slow['same_configuration_run']['elapsed_seconds'] == 1.5
    assert 'bad_chart' not in r


def test_observation_resets_wait_and_tracks_touch_shape_and_motion():
    r=continuation_observation(steps(['1/3','2/4','3/5','4/6']))
    assert r['template_run']['count'] == 4
    assert r['motion_pattern_run']['count'] == 3
    touch=steps(['A1','B2','E3']);last=touch[-1]
    touch.append(StreamStep(WAIT,last.end,last.end+2,last.end_second,last.end_second+1))
    r=continuation_observation(touch)
    assert r['touch_only_run']['count'] == 3


def test_panel_passes_matched_stage_temperatures(monkeypatch,tmp_path):
    from maichart.composer import stream_generate as g
    monkeypatch.setattr(g,'load',lambda *args:(object(),{}))
    captured=[]
    monkeypatch.setattr(g,'generate',lambda args,**kwargs:captured.append(args) or {'status':'ok'})
    condition=SimpleNamespace(level=13.,to_dict=lambda:{'level':13.})
    ds=SimpleNamespace(base=SimpleNamespace(songs=[(object(),None,None,None,None,condition)]))
    evaluate_panel('unused',ds,tmp_path,count=1,seeds=(7,))
    assert (captured[0].temperature,captured[0].pointer_temperature,captured[0].action_temperature) == (.65,.65,.65)
    evaluate_panel('unused',ds,tmp_path,count=1,seeds=(7,),temperature=.4)
    assert (captured[1].temperature,captured[1].pointer_temperature,captured[1].action_temperature) == (.4,.4,.4)


def test_generation_disables_autograd_and_restores_the_callers_grad_mode():
    import pytest,torch
    from maichart.composer import stream_generate as g
    class ReachedInference(RuntimeError):pass
    class Model:
        def eval(self):
            assert not torch.is_grad_enabled()
            raise ReachedInference()
    args=g.parser().parse_args(['--audio','unused','--checkpoint','unused','--output','unused','--device','cpu'])
    with torch.enable_grad():
        with pytest.raises(ReachedInference):g.generate(args,model_override=Model(),payload_override={})
        assert torch.is_grad_enabled()
