"""Sensor reference geometry, not a cabinet judgement emulator.

Pad constants and adjacency are ported from MiaCode (MIT), commit
6573efb611056bccd046dd17bc2f4cceda78bf88, MuriConfig.h / MuriAnalyzer.cpp.
See THIRD_PARTY_NOTICES.md.
"""
from __future__ import annotations

import math
import re

REFERENCE_VERSION = "miacode-6573efb-static-v1"
PADS = tuple(f"{ring}{i}" for ring in "ABDE" for i in range(1, 9)) + ("C",)
OVERLAP_SECONDS = 6 / 180
TAP_AVAILABLE_SECONDS = 27 / 180
TOUCH_AVAILABLE_SECONDS = 27 / 180
RELEASE_SECONDS = 3 / 180
PAD_RADII = {"A": 100 / 410, "B": 75 / 410, "C": 105 / 410,
             "D": 65 / 410, "E": 60 / 410}
PAD_DISTANCES = {"A": 1.0, "B": 2.2 / 4, "C": 0.0, "D": 4.1 / 4, "E": 3.1 / 4}


def pad_name(position: int | str) -> str:
    """Keep the lane: A1 and A8 are different sensors; C is the centre."""
    value = f"A{position}" if isinstance(position, int) else str(position).upper()
    if value != "C" and not re.fullmatch(r"[ABDE][1-8]", value):
        raise ValueError(f"Invalid judgement pad: {position!r}")
    return value


def pad_center(pad: int | str) -> tuple[float, float]:
    name = pad_name(pad)
    if name == "C":
        return 0.0, 0.0
    # Cartesian y points up. 1 and 8 form the horizontal top edge.
    angle = (90 if name[0] in "DE" else 67.5) - (int(name[1]) - 1) * 45
    radius = PAD_DISTANCES[name[0]]
    return radius * math.cos(math.radians(angle)), radius * math.sin(math.radians(angle))


def pads_at(x: float, y: float) -> tuple[str, ...]:
    """Reference circular footprints can overlap; do not return only nearest."""
    return tuple(p for p in PADS if math.dist((x, y), pad_center(p)) <= PAD_RADII[p[0]] + 1e-9)


def touch_pads_adjacent(a: str, b: str) -> bool:
    """MiaCode's Touch-group adjacency; groups are formed only at one onset."""
    a, b = pad_name(a), pad_name(b)
    if a == b:
        return False
    order = {"A": 0, "B": 1, "D": 2, "E": 3, "C": 4}
    g1, g2 = order[a[0]], order[b[0]]
    i1 = int(a[1]) - 1 if a != "C" else -1
    i2 = int(b[1]) - 1 if b != "C" else -1
    if g1 > g2:
        g1, g2, i1, i2 = g2, g1, i2, i1
    if g2 == 4:
        return g1 == 1
    if (g1, g2) in {(0, 0), (2, 2), (3, 3), (1, 2)}:
        return False
    if (g1, g2) == (1, 1):
        return (i2 - i1) % 8 in {1, 7}
    if (g1, g2) in {(0, 1), (2, 3)}:
        return i1 == i2
    return i2 == i1 or i2 == (i1 + 1) % 8


def touch_groups(pads: list[str]) -> list[list[int]]:
    remaining = set(range(len(pads)))
    result = []
    while remaining:
        group = {min(remaining)}
        frontier = list(group)
        remaining -= group
        while frontier:
            i = frontier.pop()
            neighbours = {j for j in remaining if touch_pads_adjacent(pads[i], pads[j])}
            group |= neighbours
            remaining -= neighbours
            frontier.extend(neighbours)
        result.append(sorted(group))
    return result


def touch_group_threshold(size: int) -> int:
    """MiaCode clears remaining children after at least 51% were judged."""
    return math.ceil(size * 0.51)
