from fractions import Fraction as F
from types import SimpleNamespace
import re

import pytest
import torch

from maichart.event_timing import EventPoint, EventTempo, parse_event_chart_text
from maichart.form.codec import RationalCodec
from maichart.form.rational_generate import PositionGrammar
from maichart.form.rational_position_search import search_position
from maichart.generation.event_compile import compile_event_points
from maichart.stream.data import _target_ranges
from maichart.stream.generate import parser
from maichart.stream.timing import export_end_timing, timing_diagnostics
from maichart.stream.timing import CanonicalPositionGrammar


class ScriptedModel:
    config = SimpleNamespace(max_tokens=128)

    def __init__(self, codec, paths):
        self.codec, self.paths = codec, paths
        self.batch_sizes = []

    def logits(self, prefixes):
        logits = torch.full((len(prefixes), 1, len(self.codec)), -30.)
        for row, prefix in enumerate(prefixes):
            for path, preference in self.paths:
                if path.startswith(prefix):
                    token = (self.codec.note_id if prefix == path else
                             self.codec.ids['P:'+path[len(prefix)]])
                    logits[row, 0, token] = max(float(logits[row, 0, token]), preference)
        return logits

    def decode_step(self, tokens, position, memory, mask, caches=None):
        self.batch_sizes.append(len(tokens))
        previous = (caches[0][0] if caches is not None else
                    torch.zeros((len(tokens), 0, 1), dtype=torch.long))
        assert previous.shape[1] == position
        present = torch.cat((previous, tokens[..., None]), 1)
        prefixes = [''.join(self.codec.tokens[int(token)][2:] for token in row[:, 0])
                    for row in present]
        return {'logits': self.logits(prefixes)}, [(present, present, None, None)]


def run_search(paths, lower=F(-1), upper=F(400), complexity=.35):
    codec = RationalCodec()
    model = ScriptedModel(codec, paths)
    grammar = PositionGrammar(lower, upper, max_denominator=4096)
    text, status = search_position(model, codec, grammar, model.logits([''])[0, 0], None,
        torch.zeros(1, 1, 1), None, consumed=0, width=4, complexity=complexity,
        canonical_only=True)
    return text, status


def test_whole_position_probability_recovers_twenty_after_nineteen():
    codec = RationalCodec()

    class DecimalTrap(ScriptedModel):
        def logits(self, prefixes):
            logits = torch.full((len(prefixes), 1, len(codec)), -30.)
            for row, prefix in enumerate(prefixes):
                choices = {'': {'1': 3., '2': 2.5}, '1': {'0': 6.},
                           '10': {'<note>': 8., '0': 0.}, '100': {'<note>': 6.},
                           '2': {'0': 6.}, '20': {'<note>': 6.}}
                for token, value in choices.get(prefix, {}).items():
                    logits[row, 0, codec.note_id if token == '<note>' else codec.ids['P:'+token]] = value
            return logits

    model = DecimalTrap(codec, [])
    grammar = PositionGrammar(F(19), F(400), max_denominator=384)
    text, status = search_position(model, codec, grammar, model.logits([''])[0, 0], None,
        torch.zeros(1, 1, 1), None, consumed=0, width=4, canonical_only=True)
    assert text == '20'
    assert any(row['beat'] == '100' for row in status['candidates']) or status['complete_score'] > -2
    assert any(size > 1 for size in model.batch_sizes)


def test_joint_search_retains_intentional_long_rest_and_unusual_exact_fraction():
    text, _ = run_search([('100', 10.)], lower=F(19))
    assert text == '100'  # No maximum-gap or forced density rule.
    text, _ = run_search([('37/997', 10.)], upper=F(4))
    assert F(text) == F(37, 997)


def test_noncanonical_spelling_cannot_win_over_a_canonical_fraction():
    text, _ = run_search([('2/4', 6.), ('1/2', 5.)], upper=F(2))
    assert text == '1/2'


def test_canonical_grammar_reachability_uses_the_same_completion_contract():
    grammar = CanonicalPositionGrammar(F(0), F(2), characters=4, max_denominator=4)
    assert not grammar.complete('2/4') and not grammar.prefix('2/4')
    assert grammar.complete('1/2') and grammar.prefix('1/2')


def test_stream_target_partition_has_no_overlap_no_holes_and_one_final_eos():
    codec = RationalCodec()
    points = [EventPoint(F(i, 4), '1h[4:1]' if i % 3 == 0 else '2') for i in range(24)]
    ids = codec.encode(points)
    for budget in [16, 23, 32, 1024]:
        ranges = _target_ranges(ids, budget, codec.at_id, codec.eos_id)
        actual = [token for start, end in ranges for token in ids[start:end]]
        assert actual == ids[1:]
        assert all(end-start <= budget for start, end in ranges)
        assert all(ids[start] in {codec.at_id, codec.eos_id} for start, _ in ranges)
        assert sum(codec.eos_id in ids[start:end] for start, end in ranges) == 1
        assert all(codec.eos_id not in ids[start:end] for start, end in ranges[:-1])


def test_eos_only_tail_is_supervised_when_the_last_event_exactly_fills_context():
    codec = RationalCodec()
    ids = codec.encode([EventPoint(F(0), '1')])
    ranges = _target_ranges(ids, len(ids)-2, codec.at_id, codec.eos_id)
    assert ranges[-1] == (len(ids)-1, len(ids))
    with pytest.raises(ValueError, match='单个配置'):
        _target_ranges(ids, 1, codec.at_id, codec.eos_id)


def test_eof_does_not_multiply_the_last_note_denominator():
    points = [EventPoint(F(1195, 3), '5'), EventPoint(F(31075, 78), '4/6h[4:1]')]
    tempo = [EventTempo(F(0), 185.05)]
    audio_end = F('398.3988027210884')
    end, status = export_end_timing(audio_end, tempo, 0.)
    text, _ = compile_event_points(points, tempo_changes=tempo, end_beat=end)
    assert parse_event_chart_text(text, 5).points == points
    assert end == 399 and status['notes_unchanged']
    assert max(map(int, re.findall(r'\{(\d+)\}', text))) <= 336
    report = timing_diagnostics(points, tempo, 0., 129.1755102, text)
    assert report['max_export_division'] <= 336
    assert report['max_gap_sec'] > 120  # Diagnostics do not delete real rests.


def test_default_stream_search_and_eos_policy_are_explicit():
    args = parser().parse_args(['--audio', 'a', '--checkpoint', 'b', '--output', 'c'])
    assert args.position_beam_size == 4 and args.early_eos_penalty == 0
    assert args.position_interval_strength == 1


def test_bounded_learned_interval_cost_changes_ambiguous_choice_but_keeps_a_long_rest_legal():
    codec = RationalCodec()
    model = ScriptedModel(codec, [('100', 6.), ('20', 5.8)])
    grammar = CanonicalPositionGrammar(F(19), F(400), max_denominator=384)
    costs = lambda beat: .5*(min(float(beat-19), 8.)/8.-1/8.)**2
    value, _ = search_position(model, codec, grammar, model.logits([''])[0, 0], None,
        torch.zeros(1, 1, 1), None, consumed=0, width=4, canonical_only=True,
        adaptive=True, candidate_cost=costs)
    assert value == '20'
    model = ScriptedModel(codec, [('100', 10.)])
    value, _ = search_position(model, codec, grammar, model.logits([''])[0, 0], None,
        torch.zeros(1, 1, 1), None, consumed=0, width=4, canonical_only=True,
        adaptive=True, candidate_cost=costs)
    assert value == '100'
