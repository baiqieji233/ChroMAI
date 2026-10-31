import json
import pytest
from maichart.composer import stream_generate
from maichart.composer.stream_search import PhraseSearchExhausted


def test_cli_keeps_search_failure_diagnostics_without_exporting_chart(tmp_path,monkeypatch,capsys):
    output=tmp_path/'result'/'maidata.txt'
    failure=PhraseSearchExhausted('no feasible candidate',dict(start_beat='342',deadline_exhausted=False,
        repairs_used=8,rollbacks_used=4,committed_steps=[dict(kind=2,start='0',end='1',text='1')]))
    def generate(args):raise failure
    monkeypatch.setattr(stream_generate,'generate',generate)
    with pytest.raises(PhraseSearchExhausted) as caught:
        stream_generate.main(['--audio','a','--checkpoint','c','--output',str(output),'--seed','1176305701'])
    assert caught.value is failure and not output.exists()
    report=json.loads(output.with_suffix('.failure.json').read_text(encoding='utf-8'))
    assert report['status']=='failed' and report['seed']==1176305701
    assert report['diagnostics']==failure.diagnostics
    assert 'composer_generation_failure' in capsys.readouterr().out


def test_report_write_error_preserves_original_search_exception(tmp_path,monkeypatch,capsys):
    output=tmp_path/'not_a_directory'/'maidata.txt'
    output.parent.write_text('file',encoding='utf-8')
    failure=PhraseSearchExhausted('original search failure',dict(start_beat='342'))
    def generate(args):raise failure
    monkeypatch.setattr(stream_generate,'generate',generate)
    with pytest.raises(PhraseSearchExhausted) as caught:
        stream_generate.main(['--audio','a','--checkpoint','c','--output',str(output)])
    assert caught.value is failure
    assert 'report_write_error' in capsys.readouterr().out
