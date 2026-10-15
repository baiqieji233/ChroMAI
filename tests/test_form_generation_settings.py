import json
import pytest
from fractions import Fraction as F
from types import SimpleNamespace

from maichart.form import cli, checkpoint, generate
from maichart.form.codec import RationalCodec
from maichart.event_timing import EventPoint, parse_event_chart
from maichart.generation.conditions import CONDITION_KEYS
from test_form_pipeline import fixture_audio


@pytest.mark.parametrize('family',['song_form_v1','song_form_v2'])
def test_generation_report_records_actual_vector_and_decoder_values(tmp_path,monkeypatch,family):
    from maichart.form import rational_audio,rational_generate
    monkeypatch.setattr(cli,'extract_event_audio',lambda *_a,**_kw:fixture_audio(3))
    monkeypatch.setattr(rational_audio,'extract_rational_audio',lambda *_a,**_kw:fixture_audio(3))
    payload = {'source_checkpoint_identity':{'sha256':'0'*64},'family':family,
               'epoch':1,'optimizer_steps':1,'blueprint_steps':1,'model_config':{'condition_dim':16},
               'atom_catalog':['1']}
    model = SimpleNamespace(config=SimpleNamespace(max_tokens=2048,max_memory_tokens=512))
    monkeypatch.setattr(checkpoint,'load',lambda *_a,**_kw:(model,None,RationalCodec(),payload))
    received = {}
    def fake_generate(*_args, **kwargs):
        received.update(kwargs)
        return [EventPoint(F(1),'1')], {}
    monkeypatch.setattr(generate,'generate_chart',fake_generate)
    monkeypatch.setattr(rational_generate,'generate_rational_chart',fake_generate)
    condition = tmp_path/'condition.json'
    condition.write_text(json.dumps({'level':14.5,'radar':{'note_density':.35},'style':{'slide_bias':.2}}))
    output = tmp_path/'maidata.txt'
    cli.main(['generate','--audio','fixture.wav','--output',str(output),'--checkpoint','fixture.pt',
              '--bpm','120','--level','13.3','--condition-file',str(condition),
              '--temperature','.2','--seed','42','--planner-type-strength','1.5','--style-balance-strength','0',
              '--local-chord-strength','0'])
    report = json.loads(output.with_suffix('.form.json').read_text())
    effective = report['effective_generation_settings']
    assert effective['temperature'] == .2 and effective['seed'] == 42
    assert effective['level'] == 14.5
    assert effective['condition_vector_keys'] == list(CONDITION_KEYS)
    assert len(effective['condition_vector']) == 16
    assert abs(effective['condition_vector'][0]-14.5/15) < 1e-6
    assert effective['condition_vector_dtype'] == 'float32'
    assert effective['condition_vector_level_divisor'] == 15.0
    assert effective['planner_type_strength'] == (0 if family=='song_form_v2' else 1.5)
    assert received['planner_type_strength'] == 1.5
    if family=='song_form_v2':
        assert effective['position_complexity']==received['position_complexity']==.2
        assert effective['early_eos_penalty']==received['early_eos_penalty']==0
        assert effective['position_beam_size']==received['position_beam_size']==1
    assert effective['style_balance_strength'] == received['style_balance_strength'] == 0
    assert effective['local_chord_strength'] == received['local_chord_strength'] == 0
    assert parse_event_chart(output,5).level == '14.5'


@pytest.mark.parametrize(('option','value','message'),[
    ('--position-complexity','nan','position_complexity'),
    ('--position-complexity','-1','position_complexity'),
    ('--early-eos-penalty','inf','early_eos_penalty'),
    ('--early-eos-penalty','9','early_eos_penalty'),
    ('--position-beam-size','9','position_beam_size'),
])
def test_rational_cli_rejects_invalid_controls_before_generation(tmp_path,monkeypatch,option,value,message):
    from maichart.form import rational_audio,rational_generate
    payload={'family':'song_form_v2','source_checkpoint_identity':{'sha256':'0'*64},
             'model_config':{},'atom_catalog':['1']}
    model=SimpleNamespace(config=SimpleNamespace(max_tokens=512,max_memory_tokens=64))
    monkeypatch.setattr(checkpoint,'load',lambda *_a,**_kw:(model,None,RationalCodec(),payload))
    monkeypatch.setattr(rational_audio,'extract_rational_audio',lambda *_a,**_kw:fixture_audio(3))
    monkeypatch.setattr(rational_generate,'generate_rational_chart',
        lambda *_a,**_kw:pytest.fail('Invalid controls must be rejected before decoding'))
    output=tmp_path/'maidata.txt'
    with pytest.raises(ValueError,match=message):
        cli.main(['generate','--audio','fixture.wav','--output',str(output),
                  '--checkpoint','fixture.pt','--bpm','120',option,value])
    assert not output.exists()


def test_running_webui_condition_transport_carries_rational_controls_with_cli_precedence():
    parser=cli.parser()
    common=['generate','--audio','a.wav','--output','out.txt']
    args=parser.parse_args(common)
    condition={'generation_options':{'position_complexity':.7,'position_beam_size':4,'early_eos_penalty':2}}
    cli._resolve_carried_generation_options(args,condition)
    assert (args.position_complexity,args.position_beam_size,args.early_eos_penalty)==(.7,4,2)
    args=parser.parse_args(common+['--position-complexity','0','--position-beam-size','1','--early-eos-penalty','0'])
    cli._resolve_carried_generation_options(args,condition)
    assert (args.position_complexity,args.position_beam_size,args.early_eos_penalty)==(0,1,0)


@pytest.mark.parametrize('explicit,expected', [(None,0.),('2',2.)])
def test_existing_condition_transport_carries_new_control_with_cli_precedence(tmp_path,monkeypatch,explicit,expected):
    # The existing server forwards condition JSON but does not know new flags.
    class ReachedAudio(Exception): pass
    received = {}
    from maichart.form import type_guidance
    real_validate = type_guidance.validate_strength
    def validated(value,name):
        result=real_validate(value,name)
        received[name]=result
        return result
    def stop_before_audio(*_args,**_kwargs): raise ReachedAudio()
    monkeypatch.setattr(type_guidance,'validate_strength',validated)
    monkeypatch.setattr(cli,'extract_event_audio',stop_before_audio)
    path=tmp_path/'condition.json'
    path.write_text(json.dumps({'level':14.7,'generation_options':{'local_chord_strength':0}}))
    argv=['generate','--audio','fixture.wav','--output',str(tmp_path/'out.txt'),
          '--checkpoint','fixture.pt','--condition-file',str(path)]
    if explicit is not None:argv+=['--local-chord-strength',explicit]
    with pytest.raises(ReachedAudio):cli.main(argv)
    assert received['local_chord_strength']==expected
