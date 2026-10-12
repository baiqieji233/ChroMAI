from __future__ import annotations

from fractions import Fraction
import math

import pytest
import torch

from maichart.event_timing import EventPoint, EventTempo
from maichart.generation.muri.model import PADS, RELEASE
from maichart.generation.sustain_state import (
    HAND_RADIUS,
    MAX_STATES,
    PROFILE,
    STATE_DIM,
    SustainStateIndex,
)


def _index(points, *, tempos=None, first=0.0):
    return SustainStateIndex(points, tempos or [EventTempo(Fraction(0), 120.0)], first)


def test_short_sustain_ages_out_while_long_sustain_is_carried():
    index = _index([EventPoint(Fraction(0), "1h[4:1]/2h[4:20]")])

    states, mask, metadata = index.at(Fraction(2))

    assert states.shape == (1, STATE_DIM)
    assert states.dtype == torch.float32
    assert mask.tolist() == [False]
    assert metadata["profile"] == PROFILE
    # Type one-hot and current A2 pad occupancy.
    assert states[0, 0].item() == 1.0
    assert states[0, 4 + PADS.index("A2")].item() == 1.0
    assert states[0, 4 + PADS.index("A1")].item() == 0.0
    assert states[0, 46].item() > 0.0
    assert metadata["states"][0]["source_second"] == pytest.approx(0.0)


def test_prior_sustain_survives_when_source_is_outside_context_window():
    index = _index([EventPoint(Fraction(0), "3h[4:40]")])

    states, _mask, metadata = index.at(Fraction(20))

    assert states.shape == (1, STATE_DIM)
    assert metadata["states"][0]["source_second"] == pytest.approx(0.0)
    assert metadata["boundary_second"] == pytest.approx(10.0)
    assert states[0, 4 + PADS.index("A3")].item() == 1.0


def test_waiting_slide_and_wifi_are_present_before_trace_then_progress():
    index = _index([EventPoint(Fraction(0), "1-5[4:4]/2w6[4:4]")])

    waiting, _mask, waiting_meta = index.at(Fraction(1, 2))
    waiting_by_type = {item["type"]: i for i, item in enumerate(waiting_meta["states"])}
    slide_row = waiting[waiting_by_type["slide"]]
    wifi_row = waiting[waiting_by_type["wifi"]]
    assert slide_row[45].item() == wifi_row[45].item() == 1.0
    assert slide_row[4 + PADS.index("A1")].item() == 1.0
    assert wifi_row[4 + PADS.index("A2")].item() == 1.0
    assert wifi_row[37:45].sum().item() == 3.0
    assert slide_row[47].item() > 0.0

    active, _active_mask, active_meta = index.at(Fraction(2))
    active_by_type = {item["type"]: i for i, item in enumerate(active_meta["states"])}
    assert active[active_by_type["slide"], 45].item() == 0.0
    assert active[active_by_type["slide"], 48].item() == pytest.approx(0.25)


def test_variable_tempo_first_and_release_boundary_are_exact():
    tempos = [EventTempo(Fraction(0), 120.0), EventTempo(Fraction(4), 60.0)]
    index = _index([EventPoint(Fraction(0), "1h[4:20]")], tempos=tempos, first=0.25)
    state, _mask, metadata = index.at(Fraction(8))
    assert metadata["boundary_second"] == pytest.approx(6.25)
    assert metadata["states"][0]["end_second"] == pytest.approx(10.25)
    assert state[0, 46].item() == pytest.approx(
        math.log1p(4.0 + RELEASE) / math.log1p(60.0)
    )

    short = _index([EventPoint(Fraction(0), "1h[4:2]")], first=0.25)
    at_end = short.at(Fraction(2))
    after_release = short.at(Fraction(17, 8))
    assert at_end[2]["state_count"] == 1
    assert at_end[2]["states"][0]["remaining_seconds"] == pytest.approx(RELEASE)
    assert after_release[0].shape == (1, STATE_DIM)
    assert torch.count_nonzero(after_release[0]).item() == 0
    assert after_release[1].tolist() == [True]


def test_empty_future_and_truncation_are_explicit():
    empty = _index([])
    states, mask, metadata = empty.at(0)
    assert states.shape == (1, STATE_DIM)
    assert torch.count_nonzero(states).item() == 0
    assert mask.tolist() == [True]
    assert metadata["state_count"] == 0

    future_only = _index([EventPoint(Fraction(4), "1h[4:20]")])
    future, _future_mask, future_meta = future_only.at(4)
    assert torch.count_nonzero(future).item() == 0
    assert future_meta["state_count"] == 0

    points = [
        EventPoint(Fraction(beat), f"{beat % 8 + 1}h[4:100]")
        for beat in range(MAX_STATES + 8)
    ]
    truncated, _truncated_mask, truncated_meta = _index(points).at(MAX_STATES + 8)
    assert truncated.shape == (MAX_STATES, STATE_DIM)
    assert truncated_meta["selected_state_count"] == MAX_STATES + 8
    assert truncated_meta["truncated_count"] == 8


def test_future_slide_head_cannot_change_current_hold_release_state():
    prefix = [EventPoint(Fraction(0), "1h[4:2]")]
    # Immediate slide head in the target would change the parser's tail flag,
    # but it must not leak into model inputs for this boundary.
    full = prefix + [EventPoint(Fraction(2), "1-5[0##0.5]")]
    before, mask, meta = _index(prefix).at(2)
    after, after_mask, after_meta = _index(full).at(2)
    torch.testing.assert_close(before, after)
    assert mask.tolist() == after_mask.tolist() == [False]
    assert meta["state_count"] == after_meta["state_count"] == 1


def test_negative_audio_clock_still_carries_active_slide():
    states, mask, metadata = _index([EventPoint(0, "1-5[4:4]")], first=-2).at(2)
    assert metadata["boundary_second"] == -1
    assert metadata["state_count"] == 1 and mask.tolist() == [False]
