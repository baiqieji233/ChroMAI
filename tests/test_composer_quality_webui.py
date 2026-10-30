import pytest
from test_composer_policy_webui import harness, value
from maichart.webui import composer_api as api


@pytest.mark.parametrize('policy',['cooperative_v1','cooperative_v2','song_paired_v3'])
def test_quality_training_is_explicit_and_cannot_silently_change_resume(harness,policy):
    payload, parent, calls, handler, runtime = harness
    api.start_train(handler, dict(payload, mode='fresh', configuration_quality_policy=policy), runtime=runtime)
    assert value(calls[0][1], 'configuration_quality_policy') == policy
    assert calls[0][0].result['effective_settings']['configuration_quality_policy'] == policy
    with pytest.raises(ValueError, match='[Rr]esume inherits'):
        api.start_train(handler, dict(payload, mode='resume', configuration_quality_policy=policy), runtime=runtime)


def test_quality_generation_controls_preserve_zero_and_reach_command(harness, tmp_path):
    payload, parent, calls, handler, runtime = harness
    audio=tmp_path/'audio.wav'; audio.write_bytes(b'mock')
    controls=dict(quality_context_strength=0.,quality_plan_strength=1.4,quality_lookahead_strength=.5)
    api.start_generate(handler,dict(audio=str(audio),checkpoint=payload['checkpoint'],**controls),runtime=runtime)
    task,args=calls[0]
    for key,wanted in controls.items():
        assert float(value(args,key))==wanted
        assert task.result['effective_settings'][key]==wanted
    with pytest.raises(ValueError,match='supported range'):
        api.start_generate(handler,dict(audio=str(audio),checkpoint=payload['checkpoint'],quality_lookahead_strength=2),runtime=runtime)
