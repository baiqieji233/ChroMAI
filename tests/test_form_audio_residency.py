import json
from pathlib import Path

import numpy as np
import pytest

from maichart.form import audio_residency
from maichart.form.data import _AudioStore
from maichart.generation.event_audio import EventAudio, ACOUSTIC_DIM


def store_with_features(tmp_path, monkeypatch, budget):
    features = np.zeros((30, ACOUSTIC_DIM), dtype=np.float32)
    times = np.arange(30, dtype=np.float32)*.02
    audio = EventAudio(features, times, .6, 120.)
    size = features.nbytes+times.nbytes
    sources = [tmp_path/'one.wav', tmp_path/'two.wav']
    for source in sources:
        source.write_bytes(b'fixture')
        source.with_suffix('.json').write_text(json.dumps({'times':times.tolist(),
            'metadata':{'acoustic_dim':ACOUSTIC_DIM}}))
    calls = []
    def load(path, **_kwargs):
        calls.append(path)
        return audio
    store = _AudioStore(load, tmp_path, budget=budget*size)
    monkeypatch.setattr(store, 'disk_cache_path', lambda path: Path(path).with_suffix('.json'))
    monkeypatch.setattr(audio_residency, 'physical_memory', lambda: (32*1024**3, 16*1024**3))
    return store, sources, calls, size


def test_resident_preload_retains_unique_sources_and_no_further_loads(tmp_path, monkeypatch):
    store, sources, calls, size = store_with_features(tmp_path, monkeypatch, 2)
    reports = []
    result = audio_residency.preload_audio(store, sources+sources[:1], report=lambda **v: reports.append(v))
    assert result['unique_audio'] == 2 and result['bytes'] == 2*size
    assert reports[-1]['phase'] == 'ready'
    assert len(calls) == 2
    for path in sources*10:
        store.get(path)
    assert len(calls) == 2 and store.evictions == 0


def test_resident_preload_rejects_budget_before_loading_audio(tmp_path, monkeypatch):
    store, sources, calls, _ = store_with_features(tmp_path, monkeypatch, 1)
    with pytest.raises(ValueError, match='Raise the budget'):
        audio_residency.preload_audio(store, sources, report=lambda **_: None)
    assert not calls


def test_resident_preload_preserves_physical_memory_reserve(tmp_path, monkeypatch):
    store, sources, calls, _ = store_with_features(tmp_path, monkeypatch, 2)
    monkeypatch.setattr(audio_residency, 'physical_memory', lambda: (8*1024**3, 1024**3))
    with pytest.raises(ValueError, match='Insufficient available system RAM'):
        audio_residency.preload_audio(store, sources, report=lambda **_: None)
    assert not calls


def test_resident_mode_rejects_worker_copies_before_dataset_preparation():
    from maichart.form.train import parser, _validate
    args = parser().parse_args(['--manifest','fixture.json','--checkpoint-dir','unused',
                               '--audio-cache-mode','resident','--loader-workers','1'])
    with pytest.raises(ValueError, match='loader-workers=0'):
        _validate(args)
