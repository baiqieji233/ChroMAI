from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import torch


def test_preparation_annotation_matches_source_path_and_audio_path(monkeypatch, tmp_path):
    from maichart.form.data import _preparation_annotation
    from maichart.training import quality_annotations

    source = tmp_path / 'selected' / 'maidata.txt'
    audio = tmp_path / 'selected' / 'track.wav'
    unrelated = tmp_path / 'other' / 'maidata.txt'
    monkeypatch.setattr(quality_annotations, 'chart_sha256', lambda *_: 'same-chart-hash')
    monkeypatch.setattr(quality_annotations, 'audio_sha256', lambda *_: 'same-audio-hash')
    records = [
        {'difficulty_index': 5, 'chart_sha256': 'same-chart-hash',
         'source_path': str(unrelated), 'label': 'excluded'},
        {'difficulty_index': 5, 'chart_sha256': 'same-chart-hash',
         'source_path': str(source), 'audio_path': str(audio), 'label': 'excellent'},
    ]

    result = _preparation_annotation({'records': records}, source, 5, audio, None)

    assert result['label'] == 'excellent'


def test_blueprint_input_uses_song_start_descriptor_not_a_retained_item(monkeypatch):
    from maichart.form import train
    from maichart.form import blueprint, curve
    from maichart.event_timing import EventTempo

    tempo = [EventTempo(Fraction(0), 120.)]
    span = SimpleNamespace(start=Fraction(0), end=Fraction(4))
    song = SimpleNamespace(
        spans=[span], tempo=tempo, first=0., load_curve_times_sec=np.asarray([0., 2.]),
        load_curve=np.zeros((2, 4), dtype=np.float32),
    )
    condition = SimpleNamespace(to_vector=lambda: torch.ones(4))
    dataset = SimpleNamespace(
        songs=[(song, [], 13., 1., None, condition, [], None)],
        codec=object(),
        __getitem__=lambda _self, _index: (_ for _ in ()).throw(AssertionError('dataset item read')),
    )
    descriptor_indexes = []
    monkeypatch.setattr(train, 'form_audio', lambda *_: torch.zeros((2, 3)))
    monkeypatch.setattr(blueprint, 'build_blueprint_targets', lambda *_: {})
    monkeypatch.setattr(curve, 'resample_load_curve', lambda _times, _values, frames: np.zeros((len(frames), 4), dtype=np.float32))
    monkeypatch.setattr('maichart.generation.event_audio.segment_frame_times', lambda *_: np.asarray([.5, 1.5]))
    monkeypatch.setattr(train, 'validate_cpu_batch', lambda _batch: None)

    def descriptor(_song, index):
        descriptor_indexes.append(index)
        return np.full((1, 2), index + 7, dtype=np.float32)

    monkeypatch.setattr('maichart.form.data._form_descriptor', descriptor)
    inputs, _targets, weight = train._blueprint_input(dataset, 0)

    assert descriptor_indexes == [0]
    assert inputs['form'].tolist() == [[[7., 7.]]]
    assert weight == 1.
