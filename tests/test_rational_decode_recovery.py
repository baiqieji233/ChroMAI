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
