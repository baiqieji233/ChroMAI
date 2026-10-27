from types import SimpleNamespace
from fractions import Fraction
import numpy as np
import pytest

from maichart.composer.structure import (PERIODS, STRUCTURE_DIM, structure_features,
                                         structure_fingerprint)


def song(silent=False, bpm=120., first=0.):
    t=np.arange(0., 48., .02)
    x=np.zeros((len(t),98), dtype=np.float32)
    if not silent:
        # Distinctive 7/8-aligned accent, plus quieter subdivisions.
        pulse=np.exp(-((t % .25)/.035)**2)
        accent=np.exp(-((t % 1.75)/.045)**2)
        x[:,92]=pulse*.2+accent
        x[:, :20]=(pulse*.1+accent)[:, None]
        x[:, 80]=(np.sin(t*2)+1)*.5
    audio=SimpleNamespace(features=x, times=t, duration_sec=48., metadata={},
                          source='unavailable.wav', feature_policy='test')
    return SimpleNamespace(audio=audio, tempo=[SimpleNamespace(beat=Fraction(0), bpm=bpm)], first=first)


def test_off_and_native_contract():
    s=song(); b=np.array([0.,1.,4.]); sec=b*.5
    off=structure_features(s,b,sec,policy='off')
    native=structure_features(s,b,sec,policy='native_periodic_v1')
    assert off.shape==(3, STRUCTURE_DIM) and off.dtype==np.float32
    assert not off.any() and not native[:,6:].any()
    np.testing.assert_allclose(native[0,:6], [0,1,0,1,0,1], atol=1e-6)
    np.testing.assert_allclose(native[2,:2], [0,1], atol=1e-6)


def test_rational_meter_coverage_without_grouping_claim():
    for length in (1.5,3.5,5.5,6.5):
        assert np.any(np.isclose(PERIODS,length))
    assert len(np.unique(PERIODS))==len(PERIODS)


def test_silence_has_no_claimed_hypotheses(tmp_path):
    result=structure_features(song(silent=True),[0., 8., 16.],[0., 4., 8.],
                              cache_dir=tmp_path,backend='native')
    assert np.isfinite(result).all()
    assert not result[:,6:].any()


def test_cache_identity_and_smooth_query(tmp_path):
    s=song(); b=np.array([15.9999,16.,16.0001]); sec=b*.5
    out=structure_features(s,b,sec,cache_dir=tmp_path,backend='native')
    assert np.isfinite(out).all()
    assert np.max(np.abs(np.diff(out,axis=0))) < .01
    assert list((tmp_path/'composer_structure_v4').glob('*.npy'))
    first=structure_fingerprint(s,tmp_path,'native')
    assert first==structure_fingerprint(s,tmp_path,'native')
    s.tempo[0].bpm=150.
    assert first!=structure_fingerprint(s,tmp_path,'native')


def test_fallback_is_named_and_unknown_backend_fails(monkeypatch):
    from maichart.composer.structure_backend import backend_identity
    monkeypatch.setenv('MAICHART_BEAT_THIS_CHECKPOINT','does-not-exist.ckpt')
    info=backend_identity('auto')
    assert info['name']=='native_multicycle_v1' and 'fallback' in info
    with pytest.raises(RuntimeError): backend_identity('beat_this')
    with pytest.raises(ValueError): backend_identity('other')


def synthetic_meter(period, accent=True, novelty=False):
    t=np.arange(0., 48., .02)
    beats=t*2.
    # Nearest attacks: a quarter-note kick plus half-quarter cymbal.
    def pulse(p):
        d=np.minimum(beats % p, p - beats % p)
        return np.exp(-(d/.035)**2)
    x=np.zeros((len(t),98),dtype=np.float32)
    x[:,92]=.5*pulse(1.)+.15*pulse(.5)
    if accent: x[:,92] += 1.2*pulse(period)
    if novelty: x[:,92] += 2.*np.exp(-((beats-9.25)/.035)**2)
    audio=SimpleNamespace(features=x,times=t,duration_sec=48.,metadata={},source='virtual',feature_policy='test')
    return SimpleNamespace(audio=audio,tempo=[SimpleNamespace(beat=Fraction(0),bpm=120.)],first=0.)


def test_uniform_pulse_is_not_a_certain_bar(tmp_path):
    s=synthetic_meter(4.,accent=False)
    out=structure_features(s,[20.],[10.],cache_dir=tmp_path,backend='native')
    assert out[0,[12,22,32,42]].sum() < .1


@pytest.mark.parametrize('period',[3.5,5.5,6.5])
def test_odd_meter_candidates_are_retained(period,tmp_path):
    from maichart.composer.structure import _bank
    s=synthetic_meter(period)
    bank, _, order, _ = _bank(s,tmp_path,'native')
    rec=np.asarray(bank[:,2:]).reshape(len(bank),len(PERIODS),8)
    idx=np.flatnonzero(np.isclose(PERIODS,period))[0]
    assert rec[:,idx,0].max() > 0.
    assert np.all(PERIODS[order[:,0]]>=3.)
    assert np.all(PERIODS[order[:,1]]>=6.)
    assert np.all(PERIODS[order[:,2]]>=3.)


def test_novelty_keeps_a_transient_between_anchors(tmp_path):
    changed=synthetic_meter(4.,novelty=True)
    steady=synthetic_meter(4.,novelty=False)
    changed_value=structure_features(changed,[9.25],[4.625],cache_dir=tmp_path,backend='native')[0,46]
    steady_value=structure_features(steady,[9.25],[4.625],cache_dir=tmp_path,backend='native')[0,46]
    assert changed_value > steady_value


def test_low_activity_remains_unknown(tmp_path):
    s=synthetic_meter(4.)
    s.audio.features *= 1e-9
    value=structure_features(s,[16.],[8.],cache_dir=tmp_path,backend='native')
    assert value[0,[12,22,32,42]].sum() < 1e-4
    assert value[0,46] < 1e-4


def test_inspection_reports_unknown_mass(tmp_path):
    from maichart.composer.structure import structure_segments
    segments=structure_segments(synthetic_meter(4.,accent=False),cache_dir=tmp_path,backend='native')
    assert segments and all(0 <= row['unknown_weight'] <= 1 for row in segments)
    assert all(c['grouping']=='unknown' for row in segments for c in row['candidates'])


def test_symmetric_accents_preserve_real_second_harmonic():
    from maichart.composer.structure import _harmonic_coefficients
    beats=np.array([0.,2.])
    phase=np.exp(-2j*np.pi*beats[:,None]/4.)
    z1,z2=_harmonic_coefficients(phase,np.ones((2,1)),np.array([2.]))
    assert abs(z1[0]) < 1e-12
    assert abs(z2[0]) > .999999


def test_query_keeps_second_harmonic_when_first_is_zero(monkeypatch):
    from maichart.composer import structure
    i=int(np.flatnonzero(np.isclose(PERIODS,4.))[0])
    bank=np.zeros((1,2+len(PERIODS)*8),np.float32)
    rec=bank[:,2:].reshape(1,len(PERIODS),8)
    rec[0,i]=[.5, .8, .9, 0., 0., 1., 1., 0.]
    order=np.full((1,4),i,dtype=int)
    monkeypatch.setattr(structure,'_bank',lambda *args: (bank,np.array([[0.,0.]]),order,None))
    out=structure_features(song(),[0.],[0.],backend='native')
    np.testing.assert_allclose(out[0,6:8],[0.,0.],atol=1e-7)
    np.testing.assert_allclose(out[0,8:10],[0.,.5],atol=1e-7)


def test_distinct_songs_build_concurrently(monkeypatch,tmp_path):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from maichart.composer import structure
    rendezvous=threading.Barrier(2,timeout=5)
    def fake_build(*args):
        rendezvous.wait()
        return object()
    monkeypatch.setattr(structure,'_build_bank',fake_build)
    a,b=song(),song(bpm=130.)
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(structure._bank,x,tmp_path,'native') for x in (a,b)]
        assert all(job.result(timeout=10) is not None for job in jobs)


def test_same_song_builds_once(monkeypatch,tmp_path):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from maichart.composer import structure
    count=[]
    barrier=threading.Barrier(2,timeout=5)
    def fake_build(*args):
        count.append(1)
        return object()
    def query(s):
        barrier.wait()
        return structure._bank(s,tmp_path,'native')
    monkeypatch.setattr(structure,'_build_bank',fake_build)
    s=song()
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(query,s) for _ in range(2)]
        first,second=[job.result(timeout=10) for job in jobs]
    assert first is second and len(count)==1
