"""Sustain release kinematics, attribute plan labels and plan-intent gating."""
from fractions import Fraction as F

import numpy as np
import pytest

from maichart.event_timing import EventPoint, EventTempo
from maichart.form.type_guidance import TypeGuidance, _ex_count
from maichart.generation.conditions import GENERATION_DEFAULTS, GenerationCondition
from maichart.generation.sustain_state import SustainStateIndex
from maichart.stream.data import PLAN_DIM, _plan_for_event
from maichart.stream.symmetry import (KINEMATIC_DIM, event_lanes, kinematic_rows,
                                      sustain_trajectories, transform_trajectories)

TEMPO = [EventTempo(F(0), 120.0)]


def _chart(*cells):
    return [EventPoint(F(beat), text) for beat, text in cells]


def test_slide_release_is_visible_only_after_it_ends():
    points = _chart((0, '1-5[4:1]'), (1, '3'))
    state = SustainStateIndex(points, TEMPO, 0.0)
    seconds = [0.0, 0.5]
    trajectories = sustain_trajectories(state.markers, seconds)
    assert [(k, h, e) for k, h, e, _, _ in trajectories[0]] == [(1, 1, 5)] and trajectories[1] == ()
    assert trajectories[0][0][3] == 0.5 and trajectories[0][0][4] == 1.0   # waits one beat, ends at 1.0 s
    lanes = [event_lanes(p.note_text) for p in points]
    before, after = kinematic_rows(lanes, seconds, [2, 2], [0.9, 1.1], trajectories=trajectories).numpy()
    assert KINEMATIC_DIM == 65
    assert not before[36:44].any()             # the slide is still moving
    assert after[36 + 4] > 0.8 and after[36:44].sum() == after[36 + 4]
    assert not after[44:60].any() and after[60] == 0     # nothing left in flight
    # Mirrors move the head and the end button like every other lane.
    mirrored = transform_trajectories(trajectories, 'mirror_lr')[0]
    assert [(k, h, e) for k, h, e, _, _ in mirrored] == [(1, 8, 4)]


def test_slide_in_flight_exposes_destination_before_it_ends():
    points = _chart((0, '1-5[4:1]'), (1, '3'))
    state = SustainStateIndex(points, TEMPO, 0.0)
    seconds = [0.0, 0.5]
    trajectories = sustain_trajectories(state.markers, seconds)
    lanes = [event_lanes(p.note_text) for p in points]
    waiting, tracing = kinematic_rows(lanes, seconds, [1, 1], [0.3, 0.75], trajectories=trajectories).numpy()
    # The destination (lane 5) is known while the star still waits and while it moves.
    assert waiting[44 + 4] > .5 and tracing[44 + 4] > waiting[44 + 4]
    # Origin phase: 1.0 while waiting on the head, 0.5 once tracing.
    assert waiting[52] == 1.0 and tracing[52] == 0.5
    assert waiting[60] == .25 and tracing[60] == .25 and tracing[61] > 0
    assert not waiting[36:44].any() and not tracing[36:44].any()
    # Effective step needs two events; with one committed event it stays unset.
    assert waiting[64] == 0.
    both = kinematic_rows(lanes, seconds, [2], [0.75], trajectories=trajectories).numpy()[0]
    assert both[64] == 1.0                      # step measured from the slide's end lane 5 to lane 3
    # Without trajectories the new channels are exactly zero (older callers).
    plain = kinematic_rows(lanes, seconds, [1], [0.75]).numpy()[0]
    assert not plain[36:65].any()


def test_hold_release_uses_its_own_button_and_no_trajectories_means_zero():
    points = _chart((0, '2h[4:1]'))
    state = SustainStateIndex(points, TEMPO, 0.0)
    trajectories = sustain_trajectories(state.markers, [0.0])
    assert [(k, h, e, end) for k, h, e, _, end in trajectories[0]] == [(0, 2, 2, 0.5)]
    row = kinematic_rows([(2,)], [0.0], [1], [0.6]).numpy()[0]
    assert not row[36:44].any()
    released = kinematic_rows([(2,)], [0.0], [1], [0.6], trajectories=trajectories).numpy()[0]
    assert released[36 + 1] > .8 and not released[44:60].any()      # a Hold is not a Slide in flight


def test_plan_labels_treat_break_and_ex_as_plain_attributes():
    point = EventPoint(F(1), '1b/8x')
    labels = _plan_for_event(point, F(0), point.note_text, .3)
    assert len(labels) == PLAN_DIM == 10
    assert labels[3] == 1.0 and labels[8] == 1.0 and labels[9] == 1.0
    plain = _plan_for_event(EventPoint(F(1), '1'), F(0), '1', 0.)
    assert plain[8] == plain[9] == 0.0
    assert _ex_count('1-5[8:1]') == 0 and _ex_count('1x') == 1


def test_plan_geometry_uses_relative_movement_and_is_mirror_invariant():
    previous = EventPoint(F(0), '1')
    clockwise = _plan_for_event(EventPoint(F(1), '2'), previous.beat, '2', previous_text=previous.note_text)
    mirrored = _plan_for_event(EventPoint(F(1), '7'), previous.beat, '7', previous_text='8')
    assert clockwise[1] == pytest.approx(mirrored[1], abs=1e-6)
    assert clockwise[2] == pytest.approx(mirrored[2], abs=1e-6)
    assert clockwise[1] > 0 and clockwise[2] > 0


def _guide(strength):
    condition = GenerationCondition(level=13.3, radar=GENERATION_DEFAULTS, style=GENERATION_DEFAULTS)
    return TypeGuidance(['1', '1b', '15', '1-5[8:1]', '1h[4:1]'], condition, [], planner_strength=0,
                        style_strength=1.0, plan_intent_strength=strength)


def test_intent_gate_only_attenuates_style_feedback():
    plain, gated = _guide(0.0), _guide(1.0)
    # Build a reference where the model usually expects slides ~50%.
    for guide in (plain, gated):
        for beat in range(4):
            guide.event(F(beat), intent=np.asarray([.2, .5, .2, 0., .5, .1, 0.]))
            guide.commit('1', F(beat))
    low = np.asarray([.2, .05, .2, 0., .5, .1, 0.])     # model: no slide here
    a = plain.event(F(10), intent=low).style_log
    b = gated.event(F(10), intent=low).style_log
    assert np.all(np.abs(b) <= np.abs(a) + 1e-12)
    assert np.all(np.sign(b[np.abs(b) > 1e-12]) == np.sign(a[np.abs(b) > 1e-12]))
    if a[1] > 0:                                        # slide deficit push
        assert b[1] < a[1]


def test_break_suppression_is_gated_like_chords_and_legacy_plans_are_neutral():
    guide = _guide(1.0)
    for beat in range(12):
        guide.event(F(beat), intent=np.asarray([.2, .1, .1, 0., .5, .05, 0.]))
        guide.commit('1b', F(beat))
    usual = guide.event(F(20), intent=np.asarray([.2, .1, .1, 0., .5, .05, 0.]))
    accent = guide.event(F(21), intent=np.asarray([.2, .1, .1, 0., .5, .9, 0.]))
    assert accent.break_gate < usual.break_gate <= 1.0
    assert accent.token_bias('1', ['b'])[0] >= usual.token_bias('1', ['b'])[0]
    legacy = guide.event(F(22), intent=np.asarray([.2, .1, .1, 0., .5]))   # 8-dim plan
    assert legacy.break_gate == 1.0
    assert guide.report()['plan_intent_gating']['attenuation_only'] is True
