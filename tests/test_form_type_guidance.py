from fractions import Fraction as F
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from maichart.form.codec import RationalCodec
from maichart.form.generate import arrange_span
from maichart.form.type_guidance import TypeGuidance, exclusive_preferences, validate_strength
from maichart.generation.conditions import GenerationCondition, GENERATION_DEFAULTS


ATOMS = ['1', '2', '1h[4:1]', '1-5[8:1]', 'C', 'Ch[4:1]']
EXAMPLE = {'action_state_padding_mask': torch.ones(1, dtype=torch.bool), 'meta': {}}


class TapPreferringModel:
    def __init__(self, codec, max_tokens=4096):
        self.codec = codec
        self.config = SimpleNamespace(max_tokens=max_tokens)
    def eval(self): pass
    def encode(self, _): return torch.zeros(1, 1, 1), torch.zeros(1, 1, dtype=torch.bool)
    def decode_step(self, *_):
        logits = torch.zeros(1, 1, len(self.codec))
        logits[..., self.codec.tokens.index('N:1')] = 1.0
        logits[..., self.codec.tokens.index('N:/')] = -4.0
        logits[..., self.codec.end_id] = .8
        return logits, None


def condition(slide=.125):
    return GenerationCondition(radar=GENERATION_DEFAULTS, style={
        'tap_bias': 1-slide, 'slide_bias': slide, 'hold_bias': 0, 'touch_bias': 0})


def decode(guide=None, *, temperature=0., beats=None, atoms=ATOMS):
    codec = RationalCodec()
    return arrange_span(TapPreferringModel(codec), codec, EXAMPLE,
        beats or [F(0)], atoms, origin=F(0), temperature=temperature, seed=23, type_guidance=guide)


def test_planner_slide_cue_can_change_actual_decoder_output_without_retraining():
    cue = {'beat': '0', 'type_probabilities': [.01, .01, .97, .01]}
    guide = TypeGuidance(ATOMS, condition(), [cue], planner_strength=1, style_strength=0)
    before, _ = decode()
    after, _ = decode(guide)
    assert before[0].note_text == '1'
    assert after[0].note_text == '1-5[8:1]'
    assert guide.grammar.is_complete(after[0].note_text)
    assert guide.report()['planner_argmax_matches'] == 1


def test_zero_strength_preserves_random_sampling_tokens_and_does_not_require_cues():
    beats = [F(i, 2) for i in range(8)]
    guide = TypeGuidance(ATOMS, condition(), [], planner_strength=0, style_strength=0)
    assert decode(guide, temperature=.7, beats=beats) == decode(temperature=.7, beats=beats)


def test_style_head_feedback_changes_generation_and_tracks_across_calls():
    atoms = ['1', '1-5[8:1]']
    outcomes = []
    for target in (.05, .45):
        guide = TypeGuidance(atoms, condition(target), [], planner_strength=0, style_strength=1)
        first, _ = decode(guide, beats=[F(i) for i in range(40)], atoms=atoms)
        second, _ = decode(guide, beats=[F(i) for i in range(40, 80)], atoms=atoms)
        report = guide.report()
        assert report['heads'] == 80
        assert all(guide.grammar.is_complete(p.note_text) for p in first+second)
        assert report['hard_quota'] is False
        assert report['max_abs_token_bias'] <= report['token_bias_bound']+1e-10
        outcomes.append(report['actual_style']['slide'])
    assert 0 < outcomes[0] < outcomes[1]


def test_touch_hold_overlap_and_blueprint_event_priority_are_preserved():
    cond = GenerationCondition(style={'tap_bias': .5, 'slide_bias': .1,
                                     'hold_bias': .3, 'touch_bias': .3})
    target = exclusive_preferences(cond.style)
    np.testing.assert_allclose(target, [.5, .1, .1, .1, .2])
    cues = [{'beat': '0', 'type_probabilities': [0, 0, 1, 0]}]
    guide = TypeGuidance(ATOMS, cond, cues)
    guide.commit('Ch[4:1]/1-5[8:1]', F(0))
    report = guide.report()
    assert report['actual_style']['hold'] == .5
    assert report['actual_style']['touch'] == .5
    assert report['actual_style']['slide'] == .5
    assert report['planner_argmax_matches'] == 1


def test_bias_never_relaxes_grammar_and_does_not_directly_reward_an_extra_chord_head():
    guide = TypeGuidance(ATOMS, condition(), [], planner_strength=0, style_strength=1)
    assert not guide.grammar.can_continue('1/1')
    assert not guide.grammar.is_complete('C/Ch[4:1]')
    assert guide.grammar.is_complete('C/2')
    bias = guide.event(F(0))
    eos, slash = bias.biases('1', [None, '/'])
    assert eos == pytest.approx(slash)


def test_unreachable_catalog_type_is_reported_without_inventing_notes():
    guide = TypeGuidance(['1'], condition(.5), [{'beat': '0', 'type_probabilities': [0, 0, 1, 0]}])
    points, fallbacks = decode(guide, atoms=['1'])
    assert points[0].note_text == '1' and not fallbacks
    assert 'slide' in guide.report()['unreachable_target_types']


def test_touch_hold_catalog_can_supply_both_hold_and_touch_style_shares():
    cond = GenerationCondition(style={'tap_bias': 0, 'slide_bias': 0, 'hold_bias': 1, 'touch_bias': 1})
    guide = TypeGuidance(['Ch[4:1]'], cond, [])
    report = guide.report()
    assert report['catalog_types'] == ['touch_hold']
    assert report['catalog_style_types'] == ['hold', 'touch']
    assert report['unreachable_target_types'] == []


def test_prefix_cache_is_bounded_even_when_many_head_combinations_are_visited(monkeypatch):
    from maichart.form import type_guidance
    monkeypatch.setattr(type_guidance, 'CACHE_BYTES', 32)
    monkeypatch.setattr(type_guidance, 'CACHE_ENTRIES', 2)
    guide = TypeGuidance(ATOMS, condition(), [])
    for prefix in ('', '1', 'C', 'Ch', '2'):
        for used in (frozenset(), frozenset({'1'}), frozenset({'C'})):
            indices = guide.compatible(prefix, used)
            assert all(not (guide.grammar.heads[guide.atoms[i]] & used) for i in indices)
            report = guide.report()['prefix_cache']
            assert report['entries'] <= 2 and report['bytes'] <= 32


@pytest.mark.parametrize('bad', [-1, 3.1, float('nan'), float('inf')])
def test_strength_validation_rejects_invalid_inputs(bad):
    with pytest.raises(ValueError, match='planner_type_strength'):
        validate_strength(bad, 'planner_type_strength')


def test_invalid_planner_probabilities_are_rejected_instead_of_changing_sampling():
    with pytest.raises(ValueError, match='type_probabilities'):
        TypeGuidance(ATOMS, condition(), [{'beat': '0', 'type_probabilities': [1, 0]}])


def test_uniform_planner_cue_does_not_add_a_type_preference():
    guide = TypeGuidance(ATOMS, condition(), [{'beat': '0', 'type_probabilities': [.25]*4}],
                         planner_strength=1, style_strength=0)
    assert guide.event(F(0)).biases('1', [None, '-', 'h', '/']) == pytest.approx([0]*4)
    huge = TypeGuidance(ATOMS, condition(), [{'beat': '0', 'type_probabilities': [1e308]*4}],
                        planner_strength=1, style_strength=0)
    assert huge.event(F(0)).biases('1', [None, '-', 'h', '/']) == pytest.approx([0]*4)
