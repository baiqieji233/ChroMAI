import math

import torch

from maichart.composer.generate import _sample_actions
from maichart.generation.event_cli import _NoteGrammar


ATOMS = ['1', '2', '3']


def sample(rows, *, seed=19, **kwargs):
    if kwargs.get('diagnostics') is not None:
        kwargs['diagnostics'].setdefault('action_top_probability_sum', torch.zeros(()))
        kwargs['diagnostics'].setdefault('action_decisions', 0)
    return _sample_actions(torch.tensor(rows), ATOMS, _NoteGrammar.from_atoms(ATOMS),
                           1., torch.Generator().manual_seed(seed), **kwargs)


def test_tiny_legal_residual_rejects_instead_of_becoming_certain():
    diagnostics = {}
    rows = [[0., math.log(.999999), math.log(.000001), float('-inf')]]
    assert sample(rows, blocked={'1'}, min_legal_mass=.01, diagnostics=diagnostics) == []
    assert diagnostics['low_action_support_rejections'] == 1
    assert 0. < diagnostics['min_retained_mass'] < .000002


def test_reasonably_supported_diverse_choices_remain_available():
    rows = [[0., math.log(.5), math.log(.3), math.log(.2)]]
    outputs = {tuple(sample(rows, seed=seed, blocked={'1'}, min_legal_mass=.01))
               for seed in range(40)}
    assert outputs == {('2',), ('3',)}


def test_first_slot_stop_does_not_dilute_onset_support():
    diagnostics = {}
    rows = [[1000., 0., float('-inf'), float('-inf')]]
    assert sample(rows, min_legal_mass=.01, diagnostics=diagnostics) == ['1']
    assert diagnostics['min_retained_mass'] == 1.


def test_zero_threshold_preserves_default_draws_and_rng_state():
    grammar = _NoteGrammar.from_atoms(ATOMS)
    rows = torch.tensor([[0., 2., 1., 0.], [2., 0., 1., 0.]])
    default_rng = torch.Generator().manual_seed(29)
    disabled_rng = torch.Generator().manual_seed(29)
    default = [_sample_actions(rows, ATOMS, grammar, 1., default_rng) for _ in range(12)]
    disabled = [_sample_actions(rows, ATOMS, grammar, 1., disabled_rng, min_legal_mass=0.)
                for _ in range(12)]
    assert disabled == default
    assert torch.equal(disabled_rng.get_state(), default_rng.get_state())


def test_later_slot_keeps_stop_after_first_slot_support_passes():
    rows = [[-100., 10., 0., 0.], [100., -100., -100., -100.]]
    assert sample(rows, min_legal_mass=.01) == ['1']
