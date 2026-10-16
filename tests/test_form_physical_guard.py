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


@pytest.mark.parametrize('bpm', [60, 120, 240])
def test_explicit_seconds_hold_does_not_scale_with_song_bpm(bpm):
    guard = PhysicalFeasibilityGuard(1.0)
    guard.register(F(0), 0.0, [EventTempo(F(0), bpm)], 0.0)
    guard.commit('1h[#1]', F(0), 0.0)
    assert guard.occupied_mask(0.99)[0]
    assert not guard.occupied_mask(1.0)[0]


def test_hold_grid_duration_stays_at_head_bpm_across_tempo_change():
    tempo = [EventTempo(F(0), 120), EventTempo(F(1), 240)]
    guard = PhysicalFeasibilityGuard(1.0)
    guard.register(F(0), 0, tempo, 0)
    guard.commit('1h[2:1]', F(0), 0)
    assert guard.occupied_mask(0.9)[0]
    assert not guard.occupied_mask(1.0)[0]


def test_hold_temporary_bpm_uses_its_own_duration():
    guard = PhysicalFeasibilityGuard(1.0)
    guard.register(F(0), 0, [EventTempo(F(0), 60)], 0)
    guard.commit('1h[240#4:1]', F(0), 0)
    assert guard.occupied_mask(0.249)[0]
    assert not guard.occupied_mask(0.25)[0]


def test_slide_wait_does_not_reserve_head_or_tail_and_path_moves():
    guard = PhysicalFeasibilityGuard(1.0)
    guard.register(F(0), 0.0, TEMPO, 0.0)
    guard.commit('1-5[4:1]', F(0), 0.0)
    assert len(guard._slide_actions) == 1
    action = guard._slide_actions[0]
    assert action.begin == pytest.approx(0.5)
    assert action.begin + action.duration == pytest.approx(1.0)
    assert not guard.occupied_mask(0.6).any()
    assert not guard.slide_proximity_mask(0.1).any()
    assert guard.slide_proximity_mask(0.5)[0]
    assert not guard.slide_proximity_mask(0.75)[[0, 4]].any()
    assert guard.slide_proximity_mask(0.99)[4]
    assert not guard.slide_proximity_mask(0.99)[0]


@pytest.mark.parametrize('text, begin, end', [
    ('1-5[240#4:1]', 0.25, 0.5),
    ('1-5[0.2##0.7]', 0.2, 0.9),
    ('1-5[0.2##240#4:1]', 0.2, 0.45),
])
def test_slide_duration_overrides_match_reference(text, begin, end):
    guard = PhysicalFeasibilityGuard(1.0)
    guard.register(F(0), 0, TEMPO, 0)
    guard.commit(text, F(0), 0)
    action = guard._slide_actions[0]
    assert action.begin == pytest.approx(begin)
    assert action.begin + action.duration == pytest.approx(end)


def test_alternating_hands_keep_independent_movement_clocks():
    guard = PhysicalFeasibilityGuard(1.0)
    guard.register(F(0), 0, TEMPO, 0)
    guard.commit('15', F(0), 0)
    guard.register(F(4, 5), 0.4, TEMPO, 0)
    guard.commit('1', F(4, 5), 0.4)
    # One hand can stay on 1; the other has had 450 ms to move from 5 to 3.
    assert guard.lane_penalty([3], 0.45) == 0
    assert any(0.0 in times and 0.4 in times for _hands, times, _cost in guard._hand_states)


def test_hand_alternatives_stay_bounded_and_touch_does_not_move_them():
    guard = PhysicalFeasibilityGuard(1.0)
    guard.register(F(0), 0, TEMPO, 0)
    guard.commit('15', F(0), 0)
    for index in range(1, 25):
        guard.commit(str(index % 8 + 1), F(index), index * 0.5)
    states = list(guard._hand_states)
    assert 1 <= len(states) <= 16
    guard.commit('A1/A2/B1/B2/C/D1/E1', F(25), 12.5)
    assert guard._hand_states == states


def test_out_of_order_hold_ends_are_all_pruned():
    guard = PhysicalFeasibilityGuard(1.0)
    guard.register(F(0), 0, TEMPO, 0)
    guard.commit('1h[#10]', F(0), 0)
    guard.commit('5h[#1]', F(1), 0.5)
    guard._prune(2.0)
    assert len(guard._sustains) == 1
    assert guard.occupied_mask(2.0).tolist() == [True, False, False, False, False, False, False, False]


def test_reference_parse_failure_is_reported_without_fabricating_state():
    guard = PhysicalFeasibilityGuard(1.0)
    guard.register(F(0), 0, TEMPO, 0)
    guard.commit('1h[not-a-duration]', F(0), 0)
    assert guard.report()['reference_parse_failures'] == 1
    assert not guard.occupied_mask(0).any()
