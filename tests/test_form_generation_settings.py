import json
from fractions import Fraction as F
from types import SimpleNamespace

from maichart.form import cli, checkpoint, generate
from maichart.form.codec import RationalCodec
from maichart.event_timing import EventPoint, parse_event_chart
from maichart.generation.conditions import CONDITION_KEYS
from test_form_pipeline import fixture_audio


def test_generation_report_records_actual_vector_and_decoder_values(tmp_path,monkeypatch):
    monkeypatch.setattr(cli,'extract_event_audio',lambda *_a,**_kw:fixture_audio(3))
    payload = {'source_checkpoint_identity':{'sha256':'0'*64},'family':'song_form_v1',
               'epoch':1,'optimizer_steps':1,'blueprint_steps':1,'model_config':{'condition_dim':16},
               'atom_catalog':['1']}
    model = SimpleNamespace(config=SimpleNamespace(max_tokens=2048,max_memory_tokens=512))
    monkeypatch.setattr(checkpoint,'load',lambda *_a,**_kw:(model,None,RationalCodec(),payload))
    monkeypatch.setattr(generate,'generate_chart',lambda *_a,**_kw:([EventPoint(F(1),'1')],{}))
    condition = tmp_path/'condition.json'
    condition.write_text(json.dumps({'level':14.5,'radar':{'note_density':.35},'style':{'slide_bias':.2}}))
    output = tmp_path/'maidata.txt'
    cli.main(['generate','--audio','fixture.wav','--output',str(output),'--checkpoint','fixture.pt',
              '--bpm','120','--level','13.3','--condition-file',str(condition),
              '--temperature','.2','--seed','42'])
    report = json.loads(output.with_suffix('.form.json').read_text())
    effective = report['effective_generation_settings']
    assert effective['temperature'] == .2 and effective['seed'] == 42
    assert effective['level'] == 14.5
    assert effective['condition_vector_keys'] == list(CONDITION_KEYS)
    assert len(effective['condition_vector']) == 16
    assert abs(effective['condition_vector'][0]-14.5/15) < 1e-6
    assert effective['condition_vector_dtype'] == 'float32'
    assert effective['condition_vector_level_divisor'] == 15.0
    assert parse_event_chart(output,5).level == '14.5'
