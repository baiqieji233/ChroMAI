"""Causal physical feasibility: occupancy, hand travel, and bias plumbing."""
from fractions import Fraction as F

import numpy as np
import pytest

from maichart.event_timing import EventTempo
from maichart.form.physical_guard import (
    MAX_PHYSICAL_PENALTY, OCCUPANCY_PENALTY, PhysicalFeasibilityGuard, assign_hands,
    lane_mask, validate_physical_strength,
)
from maichart.form.type_guidance import TypeGuidance
from maichart.generation.conditions import GENERATION_DEFAULTS, GenerationCondition

TEMPO = [EventTempo(F(0), 120.0)]
ATOMS = ['1', '15', '5', '1h[4:4]', '1-5[8:1]']


def condition():
    return GenerationCondition(level=13.3, radar=GENERATION_DEFAULTS, style=GENERATION_DEFAULTS)


def mask(*lanes):
    values = np.zeros((1, 8), dtype=bool)
    for lane in lanes:
        values[0, lane - 1] = True
    return values


def test_lane_mask_ignores_touch_positions():
    # Touch notes report "A4"/"C"; they are not a hand on a physical button.
    assert lane_mask(['1', '5', 'A4', 'C']).tolist() == [True, False, False, False, True, False, False, False]
    assert not lane_mask(['A4', 'C']).any()


def test_spread_chord_is_split_across_two_hands():
    # 1 and 5 are 2.0 cabinet units apart; one hand cannot cover both.
    hands, cost = assign_hands((None, None), {1, 5})
    assert set(hands) == {1, 5}
    assert cost == 0.0


def test_hold_occupies_its_lane_until_it_ends():
    guard = PhysicalFeasibilityGuard(1.0)
    guard.register(F(0), 0.0, TEMPO, 0.0)
    guard.commit('1h[4:4]', F(0), 0.0)          # 4 beats at 120 BPM = 2.0 s
    guard.register(F(1), 0.5, TEMPO, 0.0)
    event = guard.event(F(1))
    assert event.vector_penalty(mask(1))[0] == pytest.approx(OCCUPANCY_PENALTY)
    assert event.vector_penalty(mask(4))[0] == 0.0
    # The hold is released at 2.0 s, so the lane is free again.
    guard.register(F(4), 2.0, TEMPO, 0.0)
    assert guard.event(F(4)).vector_penalty(mask(1))[0] == 0.0


def test_travel_penalty_scales_with_elapsed_time():
    guard = PhysicalFeasibilityGuard(1.0)
    guard.register(F(0), 0.0, TEMPO, 0.0)
    guard.commit('15', F(0), 0.0)
    guard.register(F(1), 0.05, TEMPO, 0.0)
    fast = guard.event(F(1)).vector_penalty(mask(3))[0]
    guard.register(F(2), 1.0, TEMPO, 0.0)
    slow = guard.event(F(2)).vector_penalty(mask(3))[0]
    assert fast > 0.0
    assert slow == 0.0


def test_penalty_is_bounded_and_never_removes_candidates():
    guard = PhysicalFeasibilityGuard(3.0)
    guard.register(F(0), 0.0, TEMPO, 0.0)
    guard.commit('1h[4:4]', F(0), 0.0)
    guard.register(F(1), 0.001, TEMPO, 0.0)
    value = guard.event(F(1)).vector_penalty(mask(1, 2, 3, 4))[0]
    assert 0.0 < value <= MAX_PHYSICAL_PENALTY * 3.0
    report = guard.report()
    assert report['removes_candidates'] is False
    assert report['feedback_scope'] == 'causal_committed_prefix_only'


def test_zero_strength_disables_every_penalty():
    guard = PhysicalFeasibilityGuard(0.0)
    guard.register(F(0), 0.0, TEMPO, 0.0)
    guard.commit('1h[4:4]', F(0), 0.0)
    guard.register(F(1), 0.01, TEMPO, 0.0)
    assert guard.event(F(1)).vector_penalty(mask(1))[0] == 0.0
    assert guard.report()['enabled'] is False


@pytest.mark.parametrize('value', [-0.1, 3.1, float('nan'), float('inf')])
def test_strength_bounds(value):
    with pytest.raises(ValueError, match='physical_feasibility_strength'):
        validate_physical_strength(value)


def test_committed_collision_is_recorded_not_rejected():
    guard = PhysicalFeasibilityGuard(1.0)
    guard.register(F(0), 0.0, TEMPO, 0.0)
    guard.commit('1h[4:4]', F(0), 0.0)
    # The decoder may still choose the impossible note; the guard records it.
    guard.commit('5', F(0), 0.5)
    assert guard.report()['committed_collisions'] == 0
    guard.register(F(1), 0.5, TEMPO, 0.0)
    guard.commit('1', F(1), 0.5)
    report = guard.report()
    assert report['committed_collisions'] == 1
    assert report['diagnostics'][0]['colliding_lanes'] == [1]


def test_guidance_reports_and_applies_physical_term():
    guided = TypeGuidance(ATOMS, condition(), [], planner_strength=0, style_strength=0,
                          physical_strength=1.0)
    plain = TypeGuidance(ATOMS, condition(), [], planner_strength=0, style_strength=0)
    assert guided.physical is not None and plain.physical is None
    assert guided.report()['applied'] is True
    assert guided.report()['physical_feasibility']['enabled'] is True
    assert plain.report()['applied'] is False
    assert guided.report()['token_bias_bound'] > plain.report()['token_bias_bound']


def test_guidance_penalises_a_cell_that_reuses_a_held_lane():
    guide = TypeGuidance(ATOMS, condition(), [], planner_strength=0, style_strength=0,
                         physical_strength=1.0)
    guide.physical.register(F(0), 0.0, TEMPO, 0.0)
    guide.physical.commit('1h[4:4]', F(0), 0.0)
    guide.physical.register(F(1), 0.5, TEMPO, 0.0)
    bias = guide.event(F(1))
    # Starting a cell on the held lane must score below starting on a free one.
    # The bias telescopes against the parent prefix, so it steers the choice
    # while alternatives still differ and cancels once every continuation
    # shares the same penalty -- exactly like the local-chord term.
    values = bias.biases('', ['1', '5'])
    assert values[0] < values[1]


def test_lane_sets_cache_is_bounded_by_distinct_cells():
    guard = PhysicalFeasibilityGuard(1.0)
    guard.register(F(0), 0.0, TEMPO, 0.0)
    guard._last_second = 0.0
    rows = np.zeros((64, 8), dtype=bool)
    rows[:, 0] = True
    guard.matrix_penalty(rows, 0.1)
    # 64 identical rows collapse to a single cached lane set.
    assert len(guard._lane_set_cache) == 1
