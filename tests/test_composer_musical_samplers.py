"""Rhythm-grid mass conservation, official-tail repetition guard and structure metrics."""
from fractions import Fraction as F

import numpy as np
import pytest
import torch

from maichart.composer.candidates import (candidate_grid_levels, GRID_LEVEL_COUNT, family_switch_levels,
                                          rhythm_family_state)
from maichart.composer.generate import (_anti_repetition_logit_bias, _grounded_pointer_probabilities,
                                        parser, OFFICIAL_RUN_TAILS, pointer_switch_levels)
from maichart.composer.structure_metrics import structure_metrics

CANDIDATES = [F(1, 4), F(1, 8), F(1, 3), F(1, 6), F(3, 8), F(1, 2), F(2, 3), F(1)]
BINS = torch.tensor([0, 0, 0, 0, 0, 1, 1, 2])
JOINT = torch.tensor([.55, .15, .12, .08, .10, .5, .2, .9])


def _level_mass(probabilities, levels):
    return torch.zeros(GRID_LEVEL_COUNT).scatter_add_(0, levels, probabilities)


def test_grid_levels_cover_canonical_denominators_and_padding():
    levels = candidate_grid_levels([F(2), F(5, 2), F(9, 4), F(17, 8), F(7, 3), F(13, 6), F(1, 12)], width=9)
    assert levels.tolist() == [0, 1, 2, 3, 4, 5, 6, 6, 6]


@pytest.mark.parametrize('temperature', [.3, .5, .8, 1., 1.5])
def test_grid_level_mass_is_temperature_invariant(temperature):
    levels = torch.from_numpy(candidate_grid_levels(CANDIDATES))
    reference = _grounded_pointer_probabilities(JOINT.log(), BINS, 1., levels=levels, top_p=1., min_p=0., max_k=0)
    result = _grounded_pointer_probabilities(JOINT.log(), BINS, temperature, levels=levels)
    torch.testing.assert_close(_level_mass(result, levels), _level_mass(reference, levels))
    torch.testing.assert_close(result.sum(), torch.tensor(1.))
    # Exact T=1 ternary share of this distribution: bin0 1/2.6*.2 + bin1 .7/2.6*(.2/.7).
    assert abs(float(result[[2, 3, 6]].sum()) - (.2 / 2.6 + .2 / 2.6)) < 1e-5


def test_legacy_within_bin_temperature_erases_minority_grid_levels():
    levels = torch.from_numpy(candidate_grid_levels(CANDIDATES))
    legacy = _grounded_pointer_probabilities(JOINT.log(), BINS, .5)
    musical = _grounded_pointer_probabilities(JOINT.log(), BINS, .5, levels=levels)
    assert float(legacy[[2, 3, 6]].sum()) < .5 * float(musical[[2, 3, 6]].sum())


def test_temperature_still_acts_inside_one_level_and_noise_levels_are_pruned():
    candidates = [F(1, 8), F(3, 8), F(1, 4), F(1, 3)]
    levels = torch.from_numpy(candidate_grid_levels(candidates))
    joint = torch.tensor([.30, .10, .595, .005])
    bins = torch.zeros(4, dtype=torch.long)
    cold = _grounded_pointer_probabilities(joint.log(), bins, .3, levels=levels, top_p=1., min_p=0.)
    hot = _grounded_pointer_probabilities(joint.log(), bins, 1., levels=levels, top_p=1., min_p=0.)
    assert float(cold[0] / cold[1]) > float(hot[0] / hot[1])
    torch.testing.assert_close(cold[:2].sum(), hot[:2].sum())
    assert float(cold[3]) == 0.  # 0.5% third-beat level is below 5% of the strongest level


def test_grid_levels_validate_shape_and_range():
    with pytest.raises(ValueError, match='grid levels'):
        _grounded_pointer_probabilities(JOINT.log(), BINS, .5, levels=torch.zeros(3, dtype=torch.long))
    with pytest.raises(ValueError, match='grid levels'):
        _grounded_pointer_probabilities(JOINT.log(), BINS, .5, levels=torch.full((8,), 9, dtype=torch.long))


BITS = np.asarray([0] + [1 << i for i in range(8)], dtype=np.uint64)
BUTTONS = np.uint64(0xFF)


def _history(*lanes):
    return [((lane, 0, 0), F(index, 2)) for index, lane in enumerate(lanes)]


def _bias(history, beat, policy='official_tail_v2'):
    return _anti_repetition_logit_bias(history, beat, BITS, BUTTONS, device=torch.device('cpu'), policy=policy)


def test_official_tail_guard_allows_common_human_patterns():
    assert _bias(_history(4, 4, 5), F(3, 2)) is None          # chained double tap 4,4,5 -> 5
    assert _bias(_history(1, 2, 1), F(3, 2)) is None          # ABA -> B
    assert _bias(_history(3, 3), F(1)) is None                # AA -> A (run 3 is the official p95)
    assert _bias(_history(4, 4, 5), F(3, 2), 'legacy_v1') is not None


def test_official_tail_guard_breaks_runs_beyond_official_tails():
    p95, p99 = OFFICIAL_RUN_TAILS['repeat']
    moderate = _bias(_history(*([3] * p95)), F(p95, 2))
    strong = _bias(_history(*([3] * p99)), F(p99, 2))
    assert float(moderate[3]) == pytest.approx(-1.65)
    assert float(strong[3]) <= -2.35
    pingpong = _bias(_history(1, 2, 1, 2, 1, 2, 1, 2, 1), F(9, 2))
    assert float(pingpong[2]) == pytest.approx(-2.45) and float(pingpong[1]) == 0.
    assert _bias(_history(3, 3, 3), F(3, 2), 'off') is None


def test_sampler_policies_are_generation_flags():
    args = parser().parse_args(['--audio', 'a', '--checkpoint', 'c', '--output', 'o'])
    assert (args.repetition_guard, args.pointer_grid_policy) == ('official_tail_v2', 'grid_family_coherent_v3')


def test_structure_metrics_detects_mirror_echo_and_ternary():
    bar = [(F(0), '1'), (F(1), '2'), (F(2), '3'), (F(3), '4')]
    mirror = [(beat + 4, str(9 - int(text))) for beat, text in bar]
    triplets = [(F(8) + F(k, 3), '5') for k in range(3)]
    result = structure_metrics(bar + mirror + triplets)
    assert result['bars'] == 2
    assert result['layout_echo_by_transform']['mirror_lr'] == pytest.approx(.5)
    assert result['rhythm_echo_rate'] == pytest.approx(.5)
    assert result['ternary_rate'] == pytest.approx(2 / 11)
    assert structure_metrics([])['events'] == 0


def test_rhythm_family_state_reads_recent_non_integer_onset():
    assert rhythm_family_state([]) == 'neutral'
    assert rhythm_family_state([F(0), F(1), F(2)]) == 'neutral'
    assert rhythm_family_state([F(0), F(1, 3), F(2, 3), F(1)]) == 'ternary'
    assert rhythm_family_state([F(0), F(1, 3), F(1, 2), F(1)]) == 'binary'
    # A triplet more than two beats before the last onset no longer sets the state.
    assert rhythm_family_state([F(1, 3), F(3), F(4)]) == 'neutral'
    assert family_switch_levels('neutral').tolist() == family_switch_levels('binary').tolist()
    assert family_switch_levels('ternary')[[1, 2, 3]].all() and not family_switch_levels('ternary')[[0, 4, 5]].any()
    assert not family_switch_levels('binary')[0] and family_switch_levels('binary')[[4, 5, 6]].all()


def _family_case():
    # One bin: integer 1 (.50), 3/4 quarter (.30), 2/3 third (.12), 5/6 sixth (.08).
    candidates = [F(1), F(3, 4), F(2, 3), F(5, 6)]
    levels = torch.from_numpy(candidate_grid_levels(candidates))
    joint = torch.tensor([.50, .30, .12, .08])
    return levels, joint, torch.zeros(4, dtype=torch.long)


def test_family_coherent_sampler_tempers_switches_only():
    levels, joint, bins = _family_case()
    binary = pointer_switch_levels('grid_family_coherent_v2', [F(0), F(1, 2)], torch.device('cpu'))
    ternary = pointer_switch_levels('grid_family_coherent_v2', [F(0), F(1, 3)], torch.device('cpu'))
    plain = _grounded_pointer_probabilities(joint.log(), bins, .5, levels=levels)
    straight = _grounded_pointer_probabilities(joint.log(), bins, .5, levels=levels, switch_levels=binary)
    triplets = _grounded_pointer_probabilities(joint.log(), bins, .5, levels=levels, switch_levels=ternary)
    # Straight passage: ternary entry .20 -> .2^2/(.2^2+.8^2) = 1/17.
    assert float(straight[[2, 3]].sum()) == pytest.approx(1 / 17, abs=1e-5)
    assert float(plain[[2, 3]].sum()) == pytest.approx(.20, abs=1e-5)
    # Inside a triplet passage the quarter is the switch; ternary continuation keeps more than T=1 mass.
    assert float(triplets[1]) == pytest.approx(.3 ** 2 / (.3 ** 2 + .7 ** 2), abs=1e-5)
    assert float(triplets[[2, 3]].sum()) > .20
    for result in (straight, triplets):
        torch.testing.assert_close(result.sum(), torch.tensor(1.))


def test_family_coherent_sampler_is_identity_at_unit_temperature():
    levels, joint, bins = _family_case()
    switch = pointer_switch_levels('grid_family_coherent_v2', [F(1, 2)], torch.device('cpu'))
    torch.testing.assert_close(
        _grounded_pointer_probabilities(joint.log(), bins, 1., levels=levels, switch_levels=switch),
        _grounded_pointer_probabilities(joint.log(), bins, 1., levels=levels))
    assert pointer_switch_levels('grid_level_mass_v1', [F(1, 2)], torch.device('cpu')) is None
    with pytest.raises(ValueError, match='switch_levels'):
        _grounded_pointer_probabilities(joint.log(), bins, .5, switch_levels=switch)


def test_structure_metrics_isolated_ternary_and_reference_comparison():
    from maichart.composer.evaluation import compare_chart_reports
    run = [(F(k, 3), '1') for k in range(6)]          # coherent triplet passage
    lone = [(F(8), '2'), (F(26, 3), '3'), (F(9), '4')]  # one isolated triplet onset
    result = structure_metrics(run + lone)
    assert result['ternary_onsets'] == 5 and result['ternary_isolated_rate'] == pytest.approx(1 / 5)
    assert structure_metrics([(F(0), '1'), (F(1), '2')])['ternary_isolated_rate'] is None
    comparison = compare_chart_reports({'structure': result}, {'structure': structure_metrics(run)})
    assert comparison['structure']['ternary_isolated_rate'] == {'reference': pytest.approx(.2), 'generated': 0.}
