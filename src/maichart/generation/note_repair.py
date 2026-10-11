"""Bounded single-note rewrite candidates.

This module only proposes raw-note alternatives.  A caller must apply one
candidate atomically and re-run parsing plus playability checks before accepting
it.  The candidates are intentionally small and local; they are not an AI
infill or a claim of verified cabinet playability.
"""
from __future__ import annotations

import math
import re
from fractions import Fraction
from typing import Iterable

from maichart.event_timing import seconds_at_beat


_TAP_RE = re.compile(r"^(?P<position>[1-8])(?P<mods>[bx]*)$")
_TOUCH_RE = re.compile(r"^(?P<area>[ABDE])(?P<position>[1-8])(?P<mods>[bf]*)$", re.IGNORECASE)
_SLIDE_RE = re.compile(
    r"^(?P<start>[1-8])(?P<head>[bx]*)(?P<pattern>[-<>^])"
    r"(?P<end>[1-8])(?P<duration>\[[^\]]+\])(?P<tail>[bx]*)$"
)
_DURATION_RE = re.compile(r"\[[^\]]+\]")


def _value(item, key, default=None):
    return item.get(key, default) if isinstance(item, dict) else getattr(item, key, default)


def _protected(event) -> bool:
    if bool(_value(event, "protected", False)):
        return True
    modifiers = _value(event, "modifiers", {}) or {}
    return bool(modifiers.get("ex")) if isinstance(modifiers, dict) else False


def _beat_and_bpm_entries(tempo):
    entries = []
    for item in tempo:
        if isinstance(item, (tuple, list)) and len(item) == 2:
            beat, bpm = item
        else:
            beat, bpm = _value(item, "beat"), _value(item, "bpm")
        entries.append((beat if isinstance(beat, Fraction) else Fraction(str(beat)), float(bpm)))
    return sorted(entries)


def _latest_grid_beat(deadline: float, start_beat: Fraction, tempo, first: float) -> Fraction | None:
    """Return the latest positive 1/16 beat before ``deadline``."""
    entries = _beat_and_bpm_entries(tempo)
    if not entries or not math.isfinite(deadline):
        return None
    # Find an upper bound by extending from the source beat; this avoids a
    # float-to-beat inverse and keeps the result on an exact chart grid.
    candidate = start_beat
    step = Fraction(1, 16)
    for _ in range(4096):
        probe = candidate + step
        if float(seconds_at_beat(probe, tempo, first)) > deadline + 1e-8:
            break
        candidate = probe
    if candidate <= start_beat:
        return None
    return candidate


def _duration_token(event, end_beat: Fraction) -> str | None:
    beat = _value(event, "beat")
    if beat is None:
        return None
    beat = beat if isinstance(beat, Fraction) else Fraction(str(beat))
    duration = end_beat - beat
    if duration <= 0:
        return None
    # Simai hold duration is [grid:numerator].  This representation preserves
    # the exact Fraction and works across tempo changes because it is beat based.
    return f"[{duration.denominator * 4}:{duration.numerator}]"


def _candidate(raw: str, operation: str, cost: float) -> dict:
    return {"raw": raw, "operation": operation, "cost": float(cost)}


def _tap_candidates(event) -> list[dict]:
    raw = str(_value(event, "raw", ""))
    match = _TAP_RE.fullmatch(raw)
    if not match or not isinstance(_value(event, "button"), int):
        return []
    position = int(match["position"])
    mods = match["mods"]
    return [
        _candidate(f"{((position - 2) % 8) + 1}{mods}", "move_tap", 1.0),
        _candidate(f"{position % 8 + 1}{mods}", "move_tap", 1.0),
    ]


def _touch_candidates(event) -> list[dict]:
    raw = str(_value(event, "raw", ""))
    match = _TOUCH_RE.fullmatch(raw)
    if not match or str(_value(event, "button", "")).upper() == "C":
        return []
    area, position, mods = match["area"].upper(), int(match["position"]), match["mods"]
    return [
        _candidate(f"{area}{((position - 2) % 8) + 1}{mods}", "move_touch", 1.0),
        _candidate(f"{area}{position % 8 + 1}{mods}", "move_touch", 1.0),
    ]


def _hold_candidates(event, all_events: Iterable, tempo, first: float) -> list[dict]:
    if _protected(event) or _value(event, "note_type") not in {"hold", "touch_hold"}:
        return []
    start = float(_value(event, "time", 0.0))
    pad = _value(event, "pad", None)
    later = [other for other in all_events
             if other is not event and _value(other, "pad", None) == pad
             and float(_value(other, "time", 0.0)) > start + 1e-9]
    if not later:
        return []
    target = min(float(_value(other, "time", 0.0)) for other in later)
    # Release delay is the native judge's 3/180 second margin.
    end = _latest_grid_beat(target - 3.0 / 180.0, _value(event, "beat"), tempo, first)
    if end is None:
        return []
    duration = _duration_token(event, end)
    raw = str(_value(event, "raw", ""))
    if duration is None or not _DURATION_RE.search(raw):
        return []
    return [_candidate(_DURATION_RE.sub(duration, raw, count=1), "shorten_hold", 1.5)]


def _slide_candidates(event) -> list[dict]:
    if _protected(event) or _value(event, "note_type") != "slide":
        return []
    raw = str(_value(event, "raw", ""))
    match = _SLIDE_RE.fullmatch(raw)
    if not match:
        return []
    start, end = int(match["start"]), int(match["end"])
    result = []
    for delta in (-1, 1):
        new_end = ((end - 1 + delta) % 8) + 1
        if new_end == start:
            continue
        result.append(_candidate(
            f"{match['start']}{match['head']}{match['pattern']}{new_end}"
            f"{match['duration']}{match['tail']}",
            "move_slide_endpoint", 2.0,
        ))
    return result


def candidate_note_edits(event, issue, all_events, tempo, first) -> Iterable[dict]:
    """Yield at most eight bounded, atomic raw-note edits for one event."""
    del issue  # The caller uses the issue during counterfactual verification.
    if event is None or _protected(event):
        return ()
    note_type = str(_value(event, "note_type", ""))
    if note_type == "tap":
        candidates = _tap_candidates(event)
    elif note_type == "touch":
        candidates = _touch_candidates(event)
    elif note_type in {"hold", "touch_hold"}:
        candidates = _hold_candidates(event, all_events, tempo, first)
    elif note_type == "slide":
        candidates = _slide_candidates(event)
    else:
        candidates = []
    # Keep order stable and avoid returning a no-op or duplicate raw text.
    seen = {str(_value(event, "raw", ""))}
    result = []
    for item in candidates:
        if item["raw"] in seen:
            continue
        seen.add(item["raw"])
        result.append(item)
        if len(result) >= 8:
            break
    return tuple(result)

