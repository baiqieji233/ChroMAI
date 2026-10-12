"""Training-only, auditable chart filters applied before strict validation."""
from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Any

from maichart.event_timing import EventChart, EventPoint
from maichart.notes import _split_top_level_slashes, parse_basic_slide_token


LEGACY_STRICT_POLICY = "legacy_strict_v1"
DROP_HEADLESS_STARS_POLICY = "drop_headless_stars_v1"
DEFAULT_CHART_FILTER_POLICY = DROP_HEADLESS_STARS_POLICY
CHART_FILTER_POLICIES = (LEGACY_STRICT_POLICY, DROP_HEADLESS_STARS_POLICY)

_HEADLESS_HEAD = re.compile(r"^(?P<lane>[1-8])(?P<mods>[bx]*\?[bx]*)(?P<body>.*)$")


@dataclass(frozen=True, slots=True)
class ChartSanitization:
    chart: EventChart
    removed: tuple[dict[str, Any], ...]
    policy: str

    def to_dict(self, *, preview_count: int = 3) -> dict[str, Any]:
        records = [dict(item) for item in self.removed]
        return {
            "policy": self.policy,
            "removed_count": len(records),
            "removed_preview": records[:max(0, int(preview_count))],
            "removed": records,
        }


def validate_chart_filter_policy(policy: str) -> str:
    value = str(policy)
    if value not in CHART_FILTER_POLICIES:
        raise ValueError(
            f"Unknown chart_filter_policy {value!r}; expected one of {CHART_FILTER_POLICIES}."
        )
    return value


def _is_headless_slide_atom(atom: str) -> tuple[bool, int]:
    match = _HEADLESS_HEAD.fullmatch(atom)
    if match is None:
        return False, 0
    mods = match.group("mods")
    marker_index = mods.find("?")
    if marker_index < 0:
        return False, 0
    canonical = match.group("lane") + mods[:marker_index] + mods[marker_index + 1:] + match.group("body")
    parsed = parse_basic_slide_token(canonical)
    if parsed is None or parsed.note_type != "slide":
        return False, 0
    return True, len(parsed.slide_segments)


def sanitize_chart(chart: EventChart, policy: str = DEFAULT_CHART_FILTER_POLICY) -> ChartSanitization:
    """Drop only recognized headless Slide atoms, preserving each cell's time.

    A same-head ``*`` chain is one atom to this filter. It is either removed in
    full or left untouched for strict validation; branches are never split into
    separately learned notes.
    """
    policy = validate_chart_filter_policy(policy)
    if policy == LEGACY_STRICT_POLICY:
        return ChartSanitization(chart=chart, removed=(), policy=policy)

    filtered_points: list[EventPoint] = []
    removed: list[dict[str, Any]] = []
    for point in chart.points:
        atoms = _split_top_level_slashes(point.note_text)
        # Empty slash operands are unrelated malformed syntax. Leave the full
        # cell intact so strict validation reports them after filtering.
        if any(not atom for atom in atoms):
            filtered_points.append(EventPoint(point.beat, point.note_text))
            continue
        kept_atoms = []
        for atom in atoms:
            is_headless, branch_count = _is_headless_slide_atom(atom)
            if is_headless:
                removed.append({
                    "beat": str(point.beat),
                    "atom": atom,
                    "same_head_branches": branch_count,
                })
            else:
                kept_atoms.append(atom)
        note_text = "/".join(kept_atoms)
        if note_text:
            filtered_points.append(EventPoint(point.beat, note_text))

    filtered_chart = replace(chart, points=filtered_points)
    return ChartSanitization(chart=filtered_chart, removed=tuple(removed), policy=policy)
