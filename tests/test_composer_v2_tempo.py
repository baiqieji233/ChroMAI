from fractions import Fraction
import json
from pathlib import Path
import numpy as np
import pytest
from maichart.composer_v2 import tempo as timing


def _audio(tmp_path):
    path = tmp_path / 'audio.mp3'
    path.write_bytes(b'not decoded by mocked detector')
    return path


def _beats():
    rng = np.random.default_rng(7)
    return .137 + np.arange(240) * 60 / 95 + rng.normal(0, .003, 240)


def test_real_detection_contract_global_clock_not_local_jitter(tmp_path, monkeypatch):
    monkeypatch.setattr(timing, '_detect', lambda *args: (_beats(), {'backend': {'name': 'fixture'}}))
    tempo, first, report = timing.resolve_timing(_audio(tmp_path), tmp_path)
    assert len(tempo) == 1 and tempo[0].beat == 0
    assert tempo[0].bpm == pytest.approx(95., abs=.02)
    assert first == pytest.approx(.137, abs=.005)
    assert report['bpm_auto'] and report['first_auto']
    assert report['global_fit_residual_p90_sec'] < .01


def test_manual_bpm_and_first_zero_bypass_detection(tmp_path, monkeypatch):
    def forbidden(*args):
        raise AssertionError('manual clock must not execute an analyzer')
    monkeypatch.setattr(timing, '_detect', forbidden)
    tempo, first, report = timing.resolve_timing(_audio(tmp_path), tmp_path, bpm=182., first=0.)
    assert tempo[0].bpm == 182. and first == 0.
    assert not report['bpm_auto'] and not report['first_auto']


def test_manual_first_preserved_when_bpm_auto(tmp_path, monkeypatch):
    monkeypatch.setattr(timing, '_detect', lambda *args: (_beats(), {}))
    tempo, first, report = timing.resolve_timing(_audio(tmp_path), tmp_path, first=-.25)
    assert tempo[0].bpm == pytest.approx(95., abs=.02) and first == -.25
    assert report['first_source'] == 'manual'


def test_explicit_tempo_map_has_precedence_and_does_not_infer_meter(tmp_path):
    tempo, first, report = timing.resolve_timing(_audio(tmp_path), tmp_path,
        bpm=120., first=0., tempo_map=[{'beat': 0, 'bpm': 95.}, {'beat': '17/3', 'bpm': 190.}])
    assert [point.bpm for point in tempo] == [95., 190.]
    assert tempo[1].beat == Fraction(17, 3) and first == 0.
    assert report['tempo_source'] == 'manual_tempo_map'


def test_half_pulse_passage_does_not_corrupt_global_regression():
    base = .01 + np.arange(250) * 60 / 95
    dense = .01 + np.arange(50, 90, .5) * 60 / 95
    beats = np.unique(np.r_[base, dense])
    bpm, report = timing._global_fit(beats)
    assert bpm == pytest.approx(95., abs=.001)
    assert report['off_pulse_peak_count'] >= 35
    assert report['global_fit_residual_p90_sec'] < .001


def test_cache_contains_detector_results_not_user_timing(tmp_path, monkeypatch):
    from maichart.composer import structure_backend
    identity = dict(name='beat_this_raw_logits_v1', fps=50, checkpoint_sha256='fixture')
    monkeypatch.setattr(structure_backend, 'backend_identity', lambda *args, **kwargs: identity)
    calls = []
    def detect(*args):
        calls.append(args)
        return _beats(), {}
    monkeypatch.setattr(timing, '_beat_this', detect)
    path = _audio(tmp_path)
    one = timing.resolve_timing(path, tmp_path, bpm=95., first=None)[2]
    two = timing.resolve_timing(path, tmp_path, bpm=190., first=0.)[2]
    three = timing.resolve_timing(path, tmp_path, bpm=190., first=None)[2]
    assert len(calls) == 1 and three['cache_hit']
    assert one['effective_bpm'] == 95. and two['effective_bpm'] == 190.
    payload = json.loads(next((tmp_path/'composer_v2_timing_v1').glob('*.json')).read_text())
    assert set(payload) == {'beat_times_sec', 'diagnostics'}
    assert 'effective_first' not in payload['diagnostics']


def test_detection_failure_never_returns_default_120(tmp_path, monkeypatch):
    def failed(*args):
        raise ValueError('No beats')
    monkeypatch.setattr(timing, '_detect', failed)
    with pytest.raises(ValueError, match='No beats'):
        timing.resolve_timing(_audio(tmp_path), tmp_path)


def test_fallback_backend_and_failure_reasons_are_explicit(tmp_path, monkeypatch):
    from maichart.composer import structure_backend
    monkeypatch.setattr(structure_backend, 'backend_identity', lambda *args, **kwargs:
        {'name': 'native_multicycle_v1', 'fallback': 'local weights unavailable'})
    monkeypatch.setattr(timing, '_acoustic', lambda *args: (_beats(), {'acoustic_confidence': .5}))
    report = timing.resolve_timing(_audio(tmp_path), tmp_path)[2]
    assert report['backend']['name'] == 'librosa_flux_global_bpm_v1'
    assert report['fallback_reasons'] == ['local weights unavailable']


@pytest.mark.parametrize('options', [{'bpm': float('nan')}, {'first': float('inf')},
    {'bpm': 0.}, {'tempo_map': []}, {'tempo_map': [{'beat': 1, 'bpm': 120.}]}])
def test_invalid_manual_timing_rejected(tmp_path, options):
    with pytest.raises(ValueError):
        timing.resolve_timing(_audio(tmp_path), tmp_path, **options)


def test_recorded_constant_205_uses_whole_song_consensus():
    from maichart.composer_v2.tempo_fit import inspect_tempo_regions
    fixture = json.loads((Path(__file__).parent/'fixtures'/'composer_v2_tempo_205_peaks.json').read_text(encoding='utf-8'))
    beats = np.array(fixture['beat_times_sec'])
    bpm, report = timing._global_fit(beats)
    assert bpm == 205.
    assert report['global_fit_coverage'] > .55
    assert report['global_fit_residual_p90_sec'] < .028
    assert report['clock_fit_policy'] == 'whole_song_pulse_consensus_v2'
    assert not inspect_tempo_regions(beats,bpm)['variable_tempo_suspected']


@pytest.mark.parametrize('bpm', [73.375, 95.375, 157.125, 205.125, 278.375])
def test_genuine_fractional_bpm_is_not_rounded(bpm):
    rng = np.random.default_rng(123)
    beats = .071+np.arange(400)*60/bpm+rng.normal(0, .002, 400)
    measured, report = timing._global_fit(beats)
    assert measured == pytest.approx(bpm, abs=.005)
    assert report['bpm_precision_policy'] == 'continuous_fit'


def test_phase_follows_dominant_pulse_not_subdivision_outliers():
    period = 60/205
    beats = .188+np.arange(360)*period
    subdivision = .188+(np.arange(50, 310)+.5)*period
    peaks = np.unique(np.round(np.r_[beats, subdivision]/.02)*.02)
    first, report = timing._phase(peaks, [timing.EventTempo(Fraction(0),205.)])
    assert first == pytest.approx(.188, abs=.003)
    assert report['phase_fit_scope'] == 'dominant_consistent_pulse'
    assert report['phase_fit_inliers'] >= 350
    assert report['all_peak_phase_residual_p90_sec'] > .10


def test_manual_bpm_auto_first_does_not_refit_bpm(tmp_path, monkeypatch):
    monkeypatch.setattr(timing, '_detect', lambda *args: (_beats(), {}))
    def forbidden(*args):
        raise AssertionError('manual BPM must not execute global fitting')
    monkeypatch.setattr(timing, '_global_fit', forbidden)
    tempo, first, report = timing.resolve_timing(_audio(tmp_path), tmp_path, bpm=95.)
    assert tempo[0].bpm == 95. and first == pytest.approx(.137, abs=.005)
    assert not report['bpm_auto'] and report['first_auto']


def test_manual_variable_tempo_auto_offset_keeps_changes_on_same_clock(tmp_path, monkeypatch):
    from maichart.tempo_clock import TempoClock
    tempo = [timing.EventTempo(Fraction(0),112.),timing.EventTempo(Fraction(36),180.)]
    clock = TempoClock(tempo,.143)
    beats = clock.at_floats(np.arange(220,dtype=float))
    beats = np.round(beats/.02)*.02
    monkeypatch.setattr(timing, '_detect', lambda *args: (beats, {}))
    result, first, report = timing.resolve_timing(_audio(tmp_path), tmp_path,tempo_map=tempo)
    assert result == tempo
    assert first == pytest.approx(.143,abs=.004)
    assert report['tempo_source'] == 'manual_tempo_map'
    assert report['phase_fit_coverage'] > .95


def test_local_hints_detect_real_change_without_applying_guessed_map(tmp_path, monkeypatch):
    from maichart.tempo_clock import TempoClock
    tempo = [timing.EventTempo(Fraction(0),170.),timing.EventTempo(Fraction(184),180.)]
    beats = TempoClock(tempo,.013).at_floats(np.arange(400,dtype=float))
    beats = np.round(beats/.02)*.02
    monkeypatch.setattr(timing, '_detect', lambda *args: (beats, {}))
    result, _, report = timing.resolve_timing(_audio(tmp_path),tmp_path)
    assert len(result) == 1
    assert report['variable_tempo_suspected']
    assert not report['automatic_tempo_map_applied']
    assert report['warnings']
    assert any(abs(region['bpm']-170.) < .5 for region in report['local_tempo_regions'])
    assert any(abs(region['bpm']-180.) < .5 for region in report['local_tempo_regions'])


def test_pulse_doubling_is_not_reported_as_a_tempo_change():
    from maichart.composer_v2.tempo_fit import inspect_tempo_regions
    period = 60/95
    base = .013+np.arange(250)*period
    dense = .013+np.arange(50,110,.5)*period
    report = inspect_tempo_regions(np.unique(np.r_[base,dense]),95.)
    assert not report['variable_tempo_suspected']


def test_invalid_peak_arrays_rejected():
    with pytest.raises(ValueError):
        timing._global_fit(np.full(20,np.nan))


def test_measure_only_cli_uses_same_manual_precedence(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(timing,'_detect',lambda *args: (_beats(),{}))
    report=tmp_path/'measurement.json'
    assert timing.main(['--audio',str(_audio(tmp_path)),'--bpm','95',
        '--output',str(report)]) == 0
    payload=json.loads(report.read_text(encoding='utf-8'))
    assert payload['effective_bpm'] == 95.
    assert payload['effective_first'] == pytest.approx(.137,abs=.005)
    assert json.loads(capsys.readouterr().out) == payload


def test_acceleration_warning_does_not_require_two_fixed_speed_plateaus():
    from maichart.composer_v2.tempo_fit import inspect_tempo_regions
    beats=[]
    elapsed=.017
    # A long constant beginning followed by a progressive accelerando.
    for bpm,count in [(168,240),(176,24),(184,24),(192,32)]:
        for _ in range(count):
            beats.append(elapsed)
            elapsed+=60/bpm
    report=inspect_tempo_regions(np.array(beats),168.)
    assert report['variable_tempo_suspected']
