"""Static MiaCode Muri references, ported from the upstream MIT checker.

Source: native/miacode/upstream/src/tools/muri/MuriStaticChecker.cpp.
The report fields retain the generation API's Python naming and note shape.
"""

from __future__ import annotations

import math
from typing import Any

from .model import (
    STATIC_EPS,
    head_alert,
    head_emit_key,
    head_target,
    has_head,
    marker_key,
    pad as marker_pad,
    reference_note,
    slide_with_tap_keys,
    tap_alert,
    usable_slide,
)

JUDGE_TPS = 180.0
TAP_AVAILABLE = 27.0 / JUDGE_TPS
OVERLAY_SECONDS = 6.0 / JUDGE_TPS
TAP_ON_SLIDE_SECONDS = 1.0 / JUDGE_TPS
SLIDE_HEAD_NO_TAP_WARNING = 0.050
SLIDE_HEAD_LATE_WARNING = 0.150


def _lane_pad(lane: Any) -> str:
    try:
        lane = int(lane)
    except (TypeError, ValueError):
        return ""
    return f"A{lane}" if 1 <= lane <= 8 else ""


def _pad(marker: dict[str, Any]) -> str:
    if "marker_key" in marker and "key" not in marker:
        return str(marker.get("pad", "")).strip().upper()
    if "key" in marker:
        return str(marker_pad(marker)).strip().upper()
    kind = marker.get("type")
    if kind in {"touch", "touch_hold"}:
        return str(marker.get("touch_pad", "")).strip().upper()
    if kind in {"tap", "hold"}:
        return _lane_pad(marker.get("lane"))
    return ""


def _moment_in_range(moment: float, start: float, end: float) -> bool:
    return moment + STATIC_EPS >= start and moment <= end + STATIC_EPS


def _critical_second(marker: dict[str, Any]) -> float:
    kind = marker.get("type")
    if kind == "slide":
        shoots = marker.get("slide_segment_shoot_seconds", [])
        durations = marker.get("slide_segment_durations", [])
        proportions = marker.get("slide_segment_critical_proportions", [])
        count = min(len(shoots), len(durations), len(proportions))
        if count:
            index = count - 1
            return float(shoots[index]) + float(durations[index]) * float(proportions[index])
    if kind == "wifi" and usable_slide(marker):
        trace = float(marker["slide_trace_second"])
        end = float(marker.get("end_second", trace))
        return trace + (end - trace) * float(marker.get("wifi_critical_proportion", 0.0))
    return -1.0


def _alert_slide_head(has_tap: bool, gap: float) -> str:
    return head_alert(has_tap, gap)


def _record(kind: str, alert: str, affected: dict[str, Any], cause: dict[str, Any],
            *, delta: float | None = None, has_tap: bool | None = None) -> dict[str, Any]:
    note = reference_note(affected)
    cause_note = reference_note(cause) if cause else {}
    record = {
        "kind": kind,
        "alert_level": alert,
        "line": int(note.get("line", 1)),
        "col": int(note.get("col", 1)),
        "second": float(note.get("second", 0.0)),
        "occurrence_second": float(note.get("second", 0.0)),
        "marker_key": note.get("marker_key", note.get("key")),
        "detail": f"MiaCode {kind}",
        "affected": note,
        "cause": cause_note,
    }
    if delta is not None:
        record["delta_second"] = delta
    if has_tap is not None:
        record["slide_head_has_tap_on_slide"] = has_tap
    return record


def _head_targets(markers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    emitted: set[str] = set()
    for marker in markers:
        if not has_head(marker):
            continue
        emit_key = head_emit_key(marker)
        if emit_key in emitted:
            continue
        emitted.add(emit_key)
        targets.append(head_target(marker))
    return targets


def _head_note(head: dict[str, Any]) -> dict[str, Any]:
    # head_target supplies a synthetic tap-like reference with native source identity.
    return head


def static_references(markers: list[dict[str, Any]], threshold_seconds: float) -> list[dict[str, Any]]:
    """Build the native checker's static Slide, Wifi, and Overlap references."""
    markers = [m for m in markers if not m.get("is_mine") and not m.get("track_mine")]
    threshold = min(0.250, max(0.0, float(threshold_seconds)))
    if not math.isfinite(threshold):
        threshold = 0.0
    # Zero is an explicit policy switch in the Python API.
    extra_delta = threshold - TAP_AVAILABLE
    targets = _head_targets(markers)
    slides_with_tap = slide_with_tap_keys(markers)
    slides = [m for m in markers if m.get("type") == "slide"]
    wifis = [m for m in markers if m.get("type") == "wifi"]
    simple = [m for m in markers if m.get("type") in {"tap", "hold", "touch", "touch_hold"}]
    records: list[dict[str, Any]] = []

    def add_head(slide: dict[str, Any], affected: dict[str, Any], second: float) -> None:
        has_tap = marker_key(slide) in slides_with_tap
        gap = abs(float(slide["slide_trace_second"]) - second)
        alert = _alert_slide_head(has_tap, gap)
        if affected.get("has_protection", affected.get("is_ex", False)):
            alert = "Warning"
        records.append(_record("SlideHeadTap", alert, affected, slide,
                               delta=float(slide["slide_trace_second"]) - second,
                               has_tap=has_tap))

    def add_tap_slide(slide: dict[str, Any], affected: dict[str, Any], reference_second: float) -> None:
        gap = abs(reference_second - float(affected.get("second", 0.0)))
        alert = tap_alert(gap, threshold)
        if affected.get("has_protection", affected.get("is_ex", False)):
            alert = "Warning"
        records.append(_record("TapOnSlide", alert, affected, slide,
                               delta=reference_second - float(affected.get("second", 0.0))))

    if threshold > 0:
        for slide in [*slides, *wifis]:
            if not usable_slide(slide):
                continue
            trace = float(slide["slide_trace_second"])
            end = float(slide["end_second"])
            critical = _critical_second(slide)
            if not math.isfinite(critical) or critical < 0:
                continue
            lane = int(slide.get("lane") or 0)
            head_pad, end_pad = _lane_pad(lane), _lane_pad(slide.get("end_lane"))
            is_wifi = slide.get("type") == "wifi"
            if is_wifi:
                end_lane = int(slide.get("end_lane") or 0)
                end_lanes = {end_lane, ((end_lane + 6) % 8) + 1, (end_lane % 8) + 1}
                start = max(critical - extra_delta, trace + TAP_ON_SLIDE_SECONDS)
                finish = max(critical + threshold, end + extra_delta)
                collides = []
            else:
                end_lanes = set()
                collides = []
                shoots = slide.get("slide_segment_shoot_seconds", [])
                durations = slide.get("slide_segment_durations", [])
                pad_entries = slide.get("slide_segment_pad_enter_times", [])
                for i in range(min(len(shoots), len(durations), len(pad_entries))):
                    for entry in pad_entries[i]:
                        pad = str(entry.get("pad", "")).upper()
                        if not pad.startswith("A"):
                            continue
                        enter = float(shoots[i]) + float(entry.get("t", 0.0)) * float(durations[i])
                        collides.append((pad, enter,
                                         max(enter - extra_delta, trace + TAP_ON_SLIDE_SECONDS),
                                         enter + threshold))

            for note in simple:
                if note.get("type") not in {"tap", "hold"}:
                    continue
                second = float(note.get("second", 0.0))
                if second + STATIC_EPS < trace or second > end + threshold + STATIC_EPS:
                    continue
                pad = _pad(note)
                head_delta = second - trace
                if ((int(note.get("lane") or 0) == lane if is_wifi else pad == head_pad)
                        and head_delta + STATIC_EPS >= TAP_ON_SLIDE_SECONDS
                        and head_delta <= threshold + STATIC_EPS):
                    add_head(slide, note, second)
                if is_wifi:
                    if int(note.get("lane") or 0) in end_lanes and _moment_in_range(second, start, finish):
                        add_tap_slide(slide, note, critical)
                else:
                    for collide_pad, enter, collide_start, collide_end in collides:
                        if pad == collide_pad and _moment_in_range(second, collide_start, collide_end):
                            add_tap_slide(slide, note, enter)
                    if (pad == end_pad and second > critical + threshold + STATIC_EPS
                            and second <= end + extra_delta + STATIC_EPS):
                        add_tap_slide(slide, note, critical)

            for target in targets:
                second = float(target.get("second", 0.0))
                target_lane = int(target.get("lane") or 0)
                if second + STATIC_EPS < trace or second > end + threshold + STATIC_EPS:
                    continue
                head_delta = second - trace
                if ((target_lane == lane if is_wifi else _pad(target) == head_pad)
                        and head_delta + STATIC_EPS >= TAP_ON_SLIDE_SECONDS
                        and head_delta <= threshold + STATIC_EPS):
                    add_head(slide, _head_note(target), second)
                if is_wifi:
                    if target_lane in end_lanes and _moment_in_range(second, start, finish):
                        add_tap_slide(slide, target, critical)
                else:
                    for collide_pad, enter, collide_start, collide_end in collides:
                        if _pad(target) == collide_pad and _moment_in_range(second, collide_start, collide_end):
                            add_tap_slide(slide, target, enter)
                    if (_pad(target) == end_pad and second > critical + threshold + STATIC_EPS
                            and second <= end + extra_delta + STATIC_EPS):
                        add_tap_slide(slide, target, critical)

    candidates: list[tuple[dict[str, Any], str, bool, bool]] = []
    for marker in simple:
        pad = _pad(marker)
        if pad:
            kind = marker.get("type")
            candidates.append((reference_note(marker), pad, kind in {"tap", "touch"},
                               kind in {"hold", "touch_hold"}))
    emitted_heads: set[tuple[int, int, int, float]] = set()
    for marker in markers:
        if not has_head(marker):
            continue
        pad = _lane_pad(marker.get("lane"))
        key = (int(marker.get("line", 1)), int(marker.get("col", 1)),
               int(marker.get("lane") or 0), round(float(marker.get("second", 0.0)), 6))
        if pad and key not in emitted_heads:
            emitted_heads.add(key)
            head = head_target(marker)
            candidates.append((head, pad, True, False))

    for i, (note, pad, tap_like, hold_like) in enumerate(candidates):
        for j, (other, other_pad, other_tap, other_hold) in enumerate(candidates):
            if i == j or pad != other_pad:
                continue
            second = float(note.get("second", 0.0))
            other_second = float(other.get("second", 0.0))
            end_value = note.get("end_second", -1.0)
            other_end_value = other.get("end_second", -1.0)
            end = -1.0 if end_value is None else float(end_value)
            other_end = -1.0 if other_end_value is None else float(other_end_value)
            overlap = False
            if tap_like and other_tap:
                overlap = i < j and abs(second - other_second) <= OVERLAY_SECONDS + STATIC_EPS
            elif tap_like and other_hold and other_end >= 0:
                overlap = _moment_in_range(second, other_second - OVERLAY_SECONDS,
                                           other_end + OVERLAY_SECONDS)
            elif hold_like and end >= 0 and other_hold and other_end >= 0 and i < j:
                overlap = (_moment_in_range(second, other_second - OVERLAY_SECONDS,
                                            other_end + OVERLAY_SECONDS)
                           or _moment_in_range(other_second, second - OVERLAY_SECONDS,
                                               end + OVERLAY_SECONDS))
            if overlap:
                records.append(_record("Overlap", "Muri", note, other))

    records.sort(key=lambda r: (float(r["second"]), int(r["line"]), int(r["col"]),
                                {"SlideHeadTap": 1, "TapOnSlide": 2, "Overlap": 3}.get(r["kind"], 0),
                                int(r["cause"].get("line", 1)), int(r["cause"].get("col", 1))))
    kept: list[dict[str, Any]] = []
    overlap_seconds: set[int] = set()
    for record in records:
        if record["kind"] == "Overlap":
            second_key = round(float(record["second"]) * 1_000_000)
            if second_key in overlap_seconds:
                continue
            overlap_seconds.add(second_key)
        kept.append(record)
    return kept
