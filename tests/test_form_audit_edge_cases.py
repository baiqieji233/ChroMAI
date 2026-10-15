"""Regressions for audit findings outside the original rhythm-geometry scope."""
from fractions import Fraction as F
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from maichart.event_timing import EventPoint, EventTempo, seconds_at_beat
from maichart.form.codec import RationalCodec
from maichart.form.data import organize_song
from maichart.form.rational_audio import audio_queries, extract_rational_audio
from maichart.form.rational_generate import _spacing_beat_gap, compose_span, generate_rational_chart
from maichart.generation.conditions import GenerationCondition
from test_form_rational_events import source


@pytest.mark.parametrize('previous,seconds,expected', [
    (F(0), '.05', F(1, 40)),
    (F(399, 100), '.05', F(13, 100)),
    (F(4), '.05', F(1, 5)),
])
def test_minimum_time_gap_integrates_bpm_change(previous, seconds, expected):
    tempo = [EventTempo(F(0), 30), EventTempo(F(4), 240)]
    delta = _spacing_beat_gap(previous, tempo, seconds)
    assert delta == expected
    assert seconds_at_beat(previous + delta, tempo, -.7) - seconds_at_beat(previous, tempo, -.7) == pytest.approx(.05)


def test_compose_minimum_time_gap_does_not_apply_fast_section_bpm_to_slow_section():
    codec = RationalCodec()
    events = [EventPoint(F(1, 32), '1'), EventPoint(F(1, 16), '2')]
    sequence = codec.encode(events)
    class Model:
        config = SimpleNamespace(max_tokens=512)
        def eval(self): return self
        def encode(self, _): return torch.zeros(1, 1, 1), None
        def decode_step(self, token, position, *args, **kwargs):
            assert int(token[0, 0]) == sequence[position]
            logits = torch.full((1, 1, len(codec)), -100.)
            logits[0, 0, sequence[position + 1]] = 100.
            return logits, None
    example = {'meta': {}, 'action_state_padding_mask': torch.ones(1, dtype=torch.bool),
               '_tempo': [EventTempo(F(0), 30), EventTempo(F(4), 240)], '_first': 0.}
    points, status = compose_span(Model(), codec, example, ['1', '2'], origin=F(0), end=F(8),
        history_last=F(0), min_spacing_seconds=.05, temperature=0, position_temperature=0)
    assert points == events and status['closed']


@pytest.mark.parametrize('start,end', [(F(0), F(0)), (F(2), F(1))])
def test_audio_query_rejects_invalid_range_before_touching_audio(start, end):
    with pytest.raises(ValueError, match='end > start'):
        audio_queries(SimpleNamespace(), start, end)


def test_long_song_native_queries_fit_budget_without_restricting_event_clock(monkeypatch):
    from maichart.form import rational_audio as audio_module
    features = np.zeros((64, 98), dtype=np.float32)
    features[:, 92] = np.arange(64)
    audio = audio_module.RationalAudio(features, np.linspace(0, 599, 64), 600, 120,
        {'audio_sha256': 'audit-long-song-anchors', 'sample_rate': 22050}, 'fixture.wav',
        feature_policy=audio_module.NATIVE_FEATURE_POLICY)
    tempo = [EventTempo(F(0), 120), EventTempo(F(7, 3), 120)]
    song = SimpleNamespace(audio=audio, first=0., tempo=tempo, meters=[],
        spans=[SimpleNamespace(start=F(0), end=F(1200))])
    monkeypatch.setattr(audio_module, 'acoustic_at_times',
        lambda audio, times, **kwargs: np.zeros((len(times), 98), dtype=np.float32))
    values, times, beats = audio_queries(song, F(0), F(1200), max_nodes=32)
    assert 0 < len(beats) <= 32 and values.shape == (len(beats), 120)
    assert F(0) in beats and F(7, 3) in beats
    assert max(beats) > 1100
    assert all(a < b for a, b in zip(beats, beats[1:]))


@pytest.mark.parametrize('explicit_cap', [0, 8])
def test_event_budget_continues_automatic_budget_but_marks_explicit_cap_incomplete(source, tmp_path, monkeypatch, capsys, explicit_cap):
    import maichart.form.rational_generate as generate
    audio = extract_rational_audio(source[0], tmp_path / 'cache')
    song = organize_song(audio, [EventTempo(F(0), 120)], 0, min_seconds=3, max_seconds=3)
    assert len(song.spans) == 1
    song.load_curve_times_sec = np.array([1., 2.5], dtype=np.float32)
    song.load_curve = np.zeros((2, 4), dtype=np.float32)
    calls = []
    def compose(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return [EventPoint(F(i, 8), '1') for i in range(1, 9)], {
                'closed': False, 'reason': 'event_budget', 'tokens': 60, 'max_events': 8}
        assert kwargs['history_last'] == F(1)
        assert args[2]['meta']['history_events_kept'] > 0
        assert args[2]['meta']['history_coverage_seconds'] > 0
        return [EventPoint(F(2), '2')], {'closed': True, 'reason': 'eos', 'tokens': 10}
    monkeypatch.setattr(generate, 'compose_span', compose)
    class Planner:
        def __call__(self, audio, *args):
            return {'load_curve': torch.zeros((*audio.shape[:2], 4))}
    model = SimpleNamespace(config=SimpleNamespace(max_memory_tokens=64))
    points, report = generate_rational_chart(model, Planner(), RationalCodec(), song,
        GenerationCondition(), ['1', '2'], max_events_per_span=explicit_cap,
        style_balance_strength=0, local_chord_strength=0)
    if explicit_cap:
        assert len(calls) == 1 and len(points) == 8
        assert report['blueprint']['selection']['incomplete_spans'] == [0]
        assert 'form_warning' in capsys.readouterr().out
    else:
        assert len(calls) == 2 and len(points) == 9
        assert calls[1]['max_events'] > 0
        assert not report['blueprint']['selection']['incomplete_spans']
        assert 'form_event_budget' in capsys.readouterr().out
