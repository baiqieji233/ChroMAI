"""Waveform transport must preserve native evidence and spawned-worker sharing."""
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import json
import multiprocessing

import numpy as np
import pytest

from maichart.form import rational_audio as ra
from maichart.form import waveform_cache as wc


def _mapped_child(arguments):
    path, sha, cache = arguments
    wave, sr = wc.mapped_waveform(path, sha, cache)
    return isinstance(wave, np.memmap), sr, float(wave[100]), str(wave.filename)


@pytest.fixture
def stereo(tmp_path):
    import soundfile as sf
    sr = 22050
    t = np.arange(sr * 18) / sr
    wave = np.stack([.2 * np.sin(t * 1043), .3 * np.cos(t * 1731)], axis=1)
    path = tmp_path / 'stereo.wav'
    sf.write(path, wave, sr, subtype='PCM_16')
    return path


def test_streamed_mono_matches_old_decode_and_shares_read_only_pages(stereo, tmp_path):
    import librosa
    expected, sr = librosa.load(stereo, sr=None, mono=True)
    current, actual_sr = wc.mapped_waveform(stereo, cache_dir=tmp_path / 'cache')
    assert actual_sr == sr and isinstance(current, np.memmap)
    assert not current.flags.writeable
    np.testing.assert_array_equal(current, expected)


@pytest.mark.parametrize('policy', ra.FEATURE_POLICIES)
def test_window_queries_equal_full_source_at_boundaries_and_mid_sample_ties(stereo, tmp_path, monkeypatch, policy):
    import librosa
    expected_wave, sr = librosa.load(stereo, sr=None, mono=True)
    audio = ra.RationalAudio(np.ones((2, 98), dtype=np.float32), np.array([0., 18.]), 18., 120.,
        dict(audio_sha256=ra._digest(stereo), sample_rate=sr, normalization_scale=[1.] * 95),
        str(stereo), str(tmp_path/'cache'), feature_policy=policy)
    times = np.array([-.05, -.01, 0., .5/sr, 1.5/sr, 7.999, 8., 8.001, 10.1, 17.999, 18., 18.01, 19.])
    with monkeypatch.context() as context:
        context.setattr(ra, '_query_waveform_spectra', lambda audio, seconds, offsets:
            ra.query_spectra(expected_wave, sr, (seconds[:, None] + offsets).reshape(-1))
              .reshape(len(seconds), len(offsets), 98))
        expected = ra.acoustic_at_times(audio, times)
    actual = ra.acoustic_at_times(audio, times)
    np.testing.assert_array_equal(actual, expected)
    assert actual[8].any(), 'Late-song evidence must not be masked as local-window EOF'


def test_parallel_build_publishes_one_complete_pcm_cache(stereo, tmp_path):
    sha = ra._digest(stereo)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(_mapped_child, [(str(stereo), sha, str(tmp_path/'cache'))] * 4))
    assert len({result[3] for result in results}) == 1
    assert len(list((tmp_path/'cache').rglob('*.npy'))) == 1


def test_spawned_workers_map_same_immutable_pcm_file(stereo, tmp_path):
    arguments = (str(stereo), ra._digest(stereo), str(tmp_path/'cache'))
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn')) as pool:
        results = list(pool.map(_mapped_child, [arguments] * 2))
    assert all(value[0] for value in results)
    assert results[0] == results[1]


def test_warm_cache_never_decodes_source_again(stereo, tmp_path, monkeypatch):
    sha = ra._digest(stereo)
    first, sr = wc.mapped_waveform(stereo, sha, tmp_path/'cache')
    wc._mapped.cache_clear()
    import soundfile as sf
    monkeypatch.setattr(sf, 'SoundFile', lambda *args, **kwargs: pytest.fail('Warm mmap cache decoded source'))
    second, rate = wc.mapped_waveform(stereo, sha, tmp_path/'cache')
    assert sr == rate
    np.testing.assert_array_equal(first, second)
