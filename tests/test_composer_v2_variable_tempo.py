from fractions import Fraction as F
import numpy as np
import pytest
from maichart.composer_v2.tempo_map_fit import refine_acoustic,fit_variable_clock
from maichart.tempo_clock import TempoClock
from maichart.event_timing import EventTempo
from maichart.composer_v2 import tempo as timing


@pytest.mark.parametrize('truth,estimate',[(175.,174.93),(200.,200.14),(127.125,127.15)])
def test_full_song_audio_calibration_and_real_fractional_bpm(truth,estimate):
    t=np.arange(0,150,.01);period=60/truth;phase=.073
    r=(t-phase+period/2)%period-period/2
    flux=np.exp(-.5*(r/.008)**2)
    bpm,_,report=refine_acoustic(t,flux,estimate,phase)
    assert bpm==pytest.approx(truth,abs=.015)
    if truth==round(truth):assert bpm==truth
    else:assert not report['integer_selected']


def test_piecewise_map_is_continuous_and_not_a_fixed_bar_model():
    source=[EventTempo(F(0),170),EventTempo(F(184),180)]
    times=TempoClock(source,.013).at_floats(np.arange(400))
    times=np.round(times/.02)*.02
    result,first,report=fit_variable_clock(times,180,.013)
    assert report['automatic_tempo_map_applied'] and len(result)==2
    clock=TempoClock(result,first)
    assert result[0].bpm==pytest.approx(170,abs=.1)
    assert result[1].bpm==pytest.approx(180,abs=.1)
    assert abs(clock.at(result[1].beat)-TempoClock(source,.013).at(F(184)))<.25
    assert np.max(abs(clock.at_floats(np.arange(400))-TempoClock(source,.013).at_floats(np.arange(400))))<.03


def test_subdivision_density_change_is_not_automatically_a_tempo_change():
    base=.013+np.arange(250)*60/95
    times=np.unique(np.r_[base,.013+np.arange(50,110,.5)*60/95])
    result,_,report=fit_variable_clock(times,95,.013)
    assert result is None and not report['automatic_tempo_map_applied']


def test_auto_mode_preserves_explicit_bpm_and_offset(tmp_path,monkeypatch):
    p=tmp_path/'audio.mp3';p.write_bytes(b'fixture')
    monkeypatch.setattr(timing,'_detect',lambda *a:pytest.fail('explicit timing triggered detector'))
    result,first,report=timing.resolve_timing(p,tmp_path,bpm=175.1,first=0,tempo_detection='auto')
    assert result[0].bpm==175.1 and first==0 and not report['automatic_tempo_map_applied']


def test_auto_map_is_returned_to_generation_and_manual_map_wins(tmp_path,monkeypatch):
    p=tmp_path/'audio.mp3';p.write_bytes(b'fixture')
    original=[EventTempo(F(0),170),EventTempo(F(184),180)]
    times=TempoClock(original,.013).at_floats(np.arange(400))
    monkeypatch.setattr(timing,'_detect',lambda *a:(times,{}))
    def unavailable(*a):raise OSError('no waveform fixture')
    monkeypatch.setattr(timing,'_timing_envelope',unavailable)
    result,first,report=timing.resolve_timing(p,tmp_path,tempo_detection='auto')
    assert len(result)==2 and report['tempo_source']=='audio_piecewise_consensus'
    assert report['effective_tempo_map'][1]['beat']==str(result[1].beat)
    manual=[dict(beat=0,bpm=130),dict(beat='17/3',bpm=135)]
    result,first,report=timing.resolve_timing(p,tmp_path,tempo_map=manual,first=0,tempo_detection='auto')
    assert result[1].beat==F(17,3) and result[1].bpm==135 and first==0


def test_detected_map_survives_simai_export_without_clock_reset(tmp_path):
    from maichart.generation.event_compile import compile_event_points
    from maichart.event_timing import EventPoint,parse_event_chart
    original=[EventTempo(F(0),170),EventTempo(F(184),180)]
    peaks=TempoClock(original,.013).at_floats(np.arange(400))
    tempo,first,report=fit_variable_clock(peaks,180,.013)
    points=[EventPoint(F(b),'1') for b in (0,183,184,185,230)]
    text,_=compile_event_points(points,tempo_changes=tempo,bpm=tempo[0].bpm,first=first,
        end_beat=F(232),title='tempo fixture',level='13',designer='test',strict_notes=True)
    target=tmp_path/'maidata.txt';target.write_text(text,encoding='utf8')
    parsed=parse_event_chart(target,5,strict_notes=True)
    expected=TempoClock(tempo,first);actual=TempoClock(parsed.tempo_changes,parsed.audio_first)
    for p in points:assert abs(expected.at(p.beat)-actual.at(p.beat))<1e-5
