"""CPU transport optimizations preserve acoustic and supervision semantics."""
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from pathlib import Path
import json,threading,time

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from maichart.form import rational_audio as ra
from maichart.form.layout_prefetch import LayoutBatchPrefetch
from maichart.stream.data import StreamDataset,collate_stream
from maichart.stream.runtime import training_cpu_budget
from test_stream_grounding import config,source,cpu_threads


def test_native_selective_measurements_equal_all_five_full_spectra(source,tmp_path,monkeypatch):
    root,_=source
    audio=ra.extract_rational_audio(root/'track.wav',cache_dir=tmp_path/'audio',evidence_profile='balanced_evidence_v1')
    audio.feature_policy='native_measured_v2'
    times=np.r_[-.05,0.,np.linspace(0,6,333),6.01]
    optimized=ra.acoustic_at_times(audio,times)
    from maichart.form.waveform_cache import mapped_waveform
    signal,sr=mapped_waveform(audio.source,audio.metadata['audio_sha256'],audio.cache_dir)
    def full(audio,seconds,offsets):
        return ra.query_spectra(signal,sr,(seconds[:,None]+offsets).reshape(-1)).reshape(len(seconds),len(offsets),98)
    monkeypatch.setattr(ra,'_query_waveform_spectra',full)
    reference=ra.acoustic_at_times(audio,times)
    np.testing.assert_allclose(optimized,reference,rtol=2e-6,atol=2e-6)


def test_concurrent_cache_builds_identical_key_once_and_retains_budget():
    dataset=object.__new__(StreamDataset)
    dataset._audio_queries=OrderedDict();dataset.query_hits=dataset.query_misses=dataset.query_bytes=0
    dataset.query_budget=600;dataset._query_lock=threading.RLock()
    dataset._query_build_locks=[threading.Lock() for _ in range(8)]
    calls=[]
    def build():
        calls.append(1);time.sleep(.01)
        return torch.ones(1,120),np.array([0.]),(0,)
    with ThreadPoolExecutor(4) as executor:
        results=list(executor.map(lambda _:dataset._cached_query('same',build),range(4)))
    assert len(calls)==1 and all(value is results[0] for value in results)
    assert dataset.query_bytes<=dataset.query_budget and dataset.query_hits==3


def test_threaded_stream_samples_equal_serial_including_random_masks(source,tmp_path):
    import soundfile as sf
    root,manifest=source;records=[]
    for i in range(4):
        directory=tmp_path/str(i);directory.mkdir()
        signal,sr=sf.read(root/'track.wav',dtype='float32')
        sf.write(directory/'track.wav',signal*(1-i*.1),sr)
        (directory/'maidata.txt').write_bytes((root/'maidata.txt').read_bytes())
        records.append({'song_id':str(i),'audio_path':str(directory/'track.wav'),
            'maidata_path':str(directory/'maidata.txt'),'difficulties':[{'difficulty_index':5,'level_raw':'13'}]})
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'songs':records}),encoding='utf8')
    dataset=StreamDataset(manifest,cache_dir=tmp_path/'cache',grounding_config=config(),
        max_tokens=32,max_history_tokens=12,preparation_workers=1,audio_cache_gb=.03,
        history_note_dropout=.5,prefix_note_dropout=.5,seed=57)
    dataset.set_epoch(3)
    def loader():return DataLoader(dataset,batch_size=4,shuffle=True,collate_fn=collate_stream,
        generator=torch.Generator().manual_seed(73))
    serial=list(loader())
    with LayoutBatchPrefetch(loader(),depth=2,sample_workers=4) as stream:parallel=list(stream)
    assert len(serial)==len(parallel)
    for a,b in zip(serial,parallel):
        assert a.keys()==b.keys()
        for key in a:
            torch.testing.assert_close(a[key],b[key],atol=0,rtol=0,msg=key)
    assert dataset.query_bytes<=dataset.query_budget


def test_cpu_budget_restores_callers_thread_setting():
    previous=torch.get_num_threads()
    with training_cpu_budget(1,torch.device('cpu')) as settings:
        assert torch.get_num_threads()==1 and settings['torch_threads']==1
    assert torch.get_num_threads()==previous
