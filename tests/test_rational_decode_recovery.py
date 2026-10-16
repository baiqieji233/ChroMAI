from fractions import Fraction as F
from types import SimpleNamespace

import pytest
import torch

from maichart.event_timing import EventPoint
from maichart.form.codec import RationalCodec
from maichart.form import rational_generate as generation


def example():
    return {'meta': {}, 'action_state_padding_mask': torch.ones(1, dtype=torch.bool)}


def test_prefix_oracle_checks_delta_denominator_and_callback():
    codec = RationalCodec()
    grammar = generation.PositionGrammar(F(1, 4093), 1, characters=8, max_delta_denominator=16384)
    assert not grammar.prefix('1/100')
    assert grammar.allowed('1/100', codec) == []
    limited = generation.PositionGrammar(0, 1, characters=4,
        valid_position=lambda beat: beat < F(1, 2))
    assert not limited.prefix('7/12')
    assert limited.complete('1/3')


def test_reachability_keeps_inclusive_minimum_spacing_boundary():
    grammar = generation.PositionGrammar(0, 1, characters=8,
        min_spacing=F(1, 8), allowed_ranges=[(F(1, 8), F(126, 1000))])
    assert grammar.complete('1/8')
    assert all(grammar.prefix('1/8'[:i]) for i in range(1, 4))


def test_optional_absolute_and_interval_denominator_caps_preserve_small_unusual_values():
    grammar = generation.PositionGrammar(0, 4, origin=F(0),
        max_beat_denominator=96, max_delta_denominator=96)
    assert grammar.complete('1/7')
    assert grammar.complete('5/12')
    assert not grammar.complete('1/97')
    shifted = generation.PositionGrammar(0, 4, origin=F(1, 97), max_beat_denominator=96)
    assert not shifted.complete('1/7')


def test_partial_position_recovery_preserves_notes_and_causal_cache(monkeypatch):
    codec = RationalCodec()
    original_prefix = generation.PositionGrammar.prefix
    original_init = generation.PositionGrammar.__init__
    bad = '1/100'

    def loose_prefix(self, text):
        if self.lower == F(1, 4093) and text and bad.startswith(text):
            return True
        return original_prefix(self, text)

    monkeypatch.setattr(generation.PositionGrammar, 'prefix', loose_prefix)
    # Inject an explicitly restricted policy to exercise recovery. Production
    # no longer rejects a legal interval merely for its LCM denominator.
    def restricted_init(self,*args,**kwargs):
        kwargs['max_delta_denominator']=16384
        original_init(self,*args,**kwargs)
    monkeypatch.setattr(generation.PositionGrammar,'__init__',restricted_init)

    class Model:
        config = SimpleNamespace(max_tokens=512)
        rewinds = 0

        def eval(self): return self
        def encode(self, _): return torch.zeros(1, 1, 1), None
        def rewind_decode_cache(self, cache, length):
            self.rewinds += 1
            return cache[:length]

        def decode_step(self, token, position, memory, mask, cache=None):
            history = list(cache or [])
            assert position == len(history)
            history.append(int(token.item()))
            logits = torch.full((1, 1, len(codec)), -100.)
            finished = history.count(codec.end_id)
            if history[-1] in (codec.bos_id, codec.end_id):
                wanted = codec.eos_id if finished >= 2 else codec.at_id
            elif history[-1] == codec.note_id:
                wanted = codec.ids['N:1']
            elif codec.note_id in history[history.index(codec.at_id) + 1:] and codec.at_id not in history[history.index(codec.note_id) + 1:]:
                wanted = codec.end_id
            else:
                start = max(i for i, value in enumerate(history) if value == codec.at_id)
                tail = history[start+1:]
                if codec.note_id in tail:
                    wanted = codec.end_id
                else:
                    prefix = ''.join(codec.tokens[value][2:] for value in tail)
                    planned = '1/4093' if not finished else bad
                    wanted = (codec.note_id if prefix == planned else
                              codec.ids['P:'+planned[len(prefix)]] if planned.startswith(prefix) else codec.note_id)
                    # If the bad branch is rejected, a legal alternative stays
                    # available at the denominator choice.
                    logits[..., codec.ids['P:4']] = 0.
            logits[..., wanted] = 100.
            return logits, history

    model = Model()
    points, status = generation.compose_span(model, codec, example(), ['1'],
        origin=F(0), end=F(1), temperature=0, position_temperature=0, position_complexity=0)
    assert points == [EventPoint(F(1, 4093), '1'), EventPoint(F(1, 4), '1')]
    assert model.rewinds > 0
    assert status['reason'] == 'eos' and status['position_recoveries'] > 0


def test_optional_eos_penalty_can_continue_without_forcing_explicit_rests():
    codec = RationalCodec()

    class Model:
        config = SimpleNamespace(max_tokens=512)
        def eval(self): return self
        def encode(self, _): return torch.zeros(1, 1, 1), None
        def decode_step(self, token, position, memory, mask, cache=None):
            history = list(cache or []) + [int(token.item())]
            logits = torch.full((1, 1, len(codec)), -100.)
            if history[-1] == codec.bos_id:
                logits[..., codec.at_id] = 0.
                wanted = codec.eos_id
                logits[..., wanted] = 1.
                return logits, history
            if history[-1] == codec.at_id: wanted = codec.ids['P:0']
            elif history[-1] == codec.ids['P:0']: wanted = codec.note_id
            elif history[-1] == codec.note_id: wanted = codec.ids['N:1']
            elif history[-1] == codec.ids['N:1']: wanted = codec.end_id
            else: wanted = codec.eos_id
            logits[..., wanted] = 100.
            return logits, history

    common = dict(origin=F(0), end=F(1), temperature=0, position_temperature=0)
    points, _ = generation.compose_span(Model(), codec, example(), ['1'], **common)
    assert points == []
    points, _ = generation.compose_span(Model(), codec, example(), ['1'], early_eos_penalty=2, **common)
    assert len(points) == 1
    rest = {**example(), '_position_ranges': []}
    points, _ = generation.compose_span(Model(), codec, rest, ['1'], early_eos_penalty=8, **common)
    assert points == []


@pytest.mark.parametrize('settings', [{'position_complexity': -1}, {'early_eos_penalty': 9}])
def test_generation_corrections_validate_settings(settings):
    with pytest.raises(ValueError):
        generation.compose_span(object(), RationalCodec(), example(), ['1'], origin=F(0), end=F(1), **settings)


def _generation_song(bounds, *, bpm=185):
    import numpy as np
    from maichart.event_timing import EventTempo
    from maichart.form.data import FormSong
    from maichart.form.recurrence import DESCRIPTOR_DIM, SongRelations
    from maichart.form.timing import MetricSpan
    from maichart.generation.event_audio import ACOUSTIC_DIM, EventAudio

    spans = [MetricSpan(index, F(start), F(end), 4, 4, (), F(4), (), False, 1., 'test')
             for index, (start, end) in enumerate(bounds)]
    duration = float(spans[-1].end) * 60 / bpm
    times = np.linspace(0, duration, 100, dtype=np.float32)
    audio = EventAudio(np.zeros((len(times), ACOUSTIC_DIM), dtype=np.float32), times, duration, bpm)
    relations = SongRelations(tuple(spans), np.zeros((len(spans), DESCRIPTOR_DIM)), [])
    return FormSong(audio, [EventTempo(F(0), bpm)], 0., [], spans, relations, {},
        load_curve_times_sec=np.array([duration / 2], dtype=np.float32),
        load_curve=np.zeros((1, 4), dtype=np.float32))


def test_generation_cleanup_keeps_history_before_next_span(monkeypatch):
    import numpy as np
    from maichart.generation.conditions import GenerationCondition

    song = _generation_song([(0, 3), (3, 6)])
    monkeypatch.setattr('maichart.form.rational_audio.planner_audio_queries',
        lambda song: (torch.zeros(1, 120), np.zeros(1), []))
    calls = []

    def compose(*args, **kwargs):
        calls.append(kwargs)
        if kwargs['origin'] == 0:
            return [EventPoint(F(2), '1'), EventPoint(F(287, 96), '2')], {
                'closed': True, 'reason': 'eos', 'tokens': 20}
        assert kwargs['history_last'] == F(287, 96)
        assert args[2]['meta']['history_events_available'] == 2
        return [EventPoint(F(4), '1')], {'closed': True, 'reason': 'eos', 'tokens': 10}

    monkeypatch.setattr(generation, 'compose_span', compose)
    model = SimpleNamespace(config=SimpleNamespace(max_memory_tokens=256))
    planner = lambda audio, *args: {'load_curve': torch.zeros((*audio.shape[:2], 4))}
    points, report = generation.generate_rational_chart(model, planner, RationalCodec(), song,
        GenerationCondition(), ['1', '2'], style_balance_strength=0, local_chord_strength=0)

    assert len(calls) == 2
    assert points[1].beat == F(287, 96)
    assert report['rational_diagnostics']['near_grid_simplification']['changed'] == 0


def test_generation_replays_guidance_after_actual_time_change(monkeypatch):
    import numpy as np
    from maichart.generation.conditions import GenerationCondition

    song = _generation_song([(0, 36), (36, 40)], bpm=130)
    monkeypatch.setattr('maichart.form.rational_audio.planner_audio_queries',
        lambda song: (torch.zeros(1, 120), np.zeros(1), []))
    replays = []

    class Guidance:
        def __init__(self, *args, **kwargs):
            pass
        def replay(self, points, tempo, first, curve_factor=None):
            replays.append(tuple(points))
            assert curve_factor(0.) == 1.
        def report(self):
            return {}

    monkeypatch.setattr('maichart.form.type_guidance.TypeGuidance', Guidance)

    def compose(*args, **kwargs):
        if kwargs['origin'] == 0:
            return [EventPoint(F(35), '1'), EventPoint(F(3313, 94), '2')], {
                'closed': True, 'reason': 'eos', 'tokens': 20}
        assert replays == [(EventPoint(F(35), '1'), EventPoint(F(141, 4), '2'))]
        assert kwargs['history_last'] == F(141, 4)
        return [EventPoint(F(37), '1')], {'closed': True, 'reason': 'eos', 'tokens': 10}

    monkeypatch.setattr(generation, 'compose_span', compose)
    model = SimpleNamespace(config=SimpleNamespace(max_memory_tokens=256))
    planner = lambda audio, *args: {'load_curve': torch.zeros((*audio.shape[:2], 4))}
    points, report = generation.generate_rational_chart(model, planner, RationalCodec(), song,
        GenerationCondition(), ['1', '2'], style_balance_strength=1., local_chord_strength=0)

    assert len(replays) == 1
    assert points[1].beat == F(141, 4)
    assert report['rational_diagnostics']['near_grid_simplification']['changed'] == 1
