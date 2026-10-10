"""Circular cabinet coordinates and a hand-side preference, never a hand lock."""
from __future__ import annotations

import math
from maichart.generation.judgment_system import pad_center


class ButtonLayout:
    @staticmethod
    def button_angle(button: int) -> float:
        if not isinstance(button, int) or isinstance(button, bool) or not 1 <= button <= 8:
            raise ValueError("button must be an integer in 1..8")
        return (67.5 - (button - 1) * 45) % 360

    @staticmethod
    def angular_distance(a: int, b: int) -> float:
        gap = abs(ButtonLayout.button_angle(a) - ButtonLayout.button_angle(b))
        return min(gap, 360 - gap)

    @staticmethod
    def physical_distance(a: int, b: int, radius: float = 1.0) -> float:
        return math.dist(pad_center(a), pad_center(b)) * radius

    @staticmethod
    def is_adjacent(a: int, b: int) -> bool:
        return ButtonLayout.angular_distance(a, b) == 45


class HandDispatch:
    # From the player's viewpoint; either hand can cross the centre line.
    LEFT_PRIMARY = frozenset({5, 6, 7, 8})
    RIGHT_PRIMARY = frozenset({1, 2, 3, 4})

    @staticmethod
    def primary_hand(button: int) -> str:
        ButtonLayout.button_angle(button)
        return "left" if button in HandDispatch.LEFT_PRIMARY else "right"

    @staticmethod
    def can_reach_simultaneous(buttons: list[int], allow_cross: bool = False) -> bool:
        """Two isolated presses always admit a two-hand assignment.

        More presses require geometry and ongoing-action search; this helper
        deliberately makes no claim that a single hand spans 135 degrees.
        """
        return len(set(buttons)) <= 2
