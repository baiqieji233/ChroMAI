"""Shared identities, timebase and sensor geometry from MiaCode 6573efb (MIT)."""
from __future__ import annotations

import math

TPS = 180.0
EPS = 1e-6
STATIC_EPS = 1e-9
RELEASE = 3 / TPS
AVAILABLE = 27 / TPS
PADS = tuple(f"{ring}{i}" for ring in "ABDE" for i in range(1, 9)) + ("C",)
RADII = {"A": 50., "B": 37.5, "C": 52.5, "D": 32.5, "E": 30.}
DISTANCES = {"A": 205., "B": 112.75, "C": 0., "D": 210.125, "E": 158.875}


def center(pad):
    if pad == "C":
        return (0., 0.)
    angle = math.radians((-90 if pad[0] in "DE" else -67.5) + (int(pad[1]) - 1) * 45)
    radius = DISTANCES[pad[0]]
    return radius * math.cos(angle), radius * math.sin(angle)


CENTERS = {p: center(p) for p in PADS}


def start_tick(second):
    return math.ceil(second * TPS - EPS)


def end_tick(second):
    return math.ceil(second * TPS - EPS) - 1


def marker_key(m):
    return m["key"]


def head_key(m):
    return f"slide_head_star|{m['second']:.6f}|{m['lane']}|{m['line']}|{m['col']}|{m.get('each_group_id', -1)}"


def has_head(m):
    return (m["type"] in {"slide", "wifi"} and m.get("has_head_star", True)
            and not m.get("head_mine", False) and 1 <= m["lane"] <= 8)


def head_emit_key(m):
    if m.get("same_head_slide"):
        return (m["second"], m["line"], m["lane"], bool(m.get("head_break")),
                bool(m.get("head_ex")), bool(m.get("head_each")))
    return m["second"], m["line"], m["col"], m["lane"], m.get("parse_order", -1)


def usable_slide(m):
    return (m["type"] in {"slide", "wifi"}
            and m.get("slide_trace_second", -1) + EPS >= m["second"]
            and m.get("end_second", -1) + EPS >= m.get("slide_trace_second", -1))


def pad(m):
    if m["type"] in {"tap", "hold"}:
        return f"A{m['lane']}"
    return m.get("touch_pad", "")


def reference_note(m):
    if "marker_key" in m and "key" not in m:
        return dict(m)
    result = {"marker_key": m["key"], "marker_type": m["type"],
              "line": m["line"], "col": m["col"], "second": m["second"],
              "lane": m.get("lane", 1), "end_lane": m.get("end_lane", 1),
              "head_star_tap_like": False, "has_protection": bool(m.get("is_ex"))}
    if pad(m):
        result["pad"] = pad(m)
    for key in ("end_second", "slide_trace_second"):
        if m.get(key, -1) >= 0:
            result[key] = m[key]
    if m.get("slide_track_key"):
        result["slide_track_key"] = m["slide_track_key"]
    return result


def head_target(m):
    result = reference_note(m)
    result.update(marker_key=head_key(m), marker_type="tap", pad=f"A{m['lane']}",
                  col=m["col"] + 1, end_lane=m["lane"], head_star_tap_like=True,
                  has_protection=bool(m.get("head_ex")))
    for key in ("end_second", "slide_trace_second", "slide_track_key"):
        result.pop(key, None)
    return result


def tap_alert(gap, threshold):
    return "Warning" if gap > .150 + STATIC_EPS and gap <= threshold + STATIC_EPS else "Muri"


def head_alert(has_tap, gap):
    if not has_tap and STATIC_EPS < gap <= .050 + STATIC_EPS:
        return "Warning"
    return "Warning" if gap + STATIC_EPS >= .150 else "Muri"


def slide_with_tap_keys(markers):
    starts = {}
    for m in markers:
        if usable_slide(m):
            starts.setdefault((m["lane"], round(m["slide_trace_second"] * 1e6)), []).append(m["key"])
    found = set()
    for m in markers:
        if m.get("slide_head") and (m["type"] == "tap" or has_head(m)):
            found.update(starts.get((m["lane"], round(m["second"] * 1e6)), []))
    return found


def diagnostic(kind, alert, marker, occurrence, cause=None, detail=""):
    result = {"kind": kind, "alert_level": alert, "second": occurrence,
              "occurrence_second": occurrence, "marker_key": marker.get("key", marker.get("marker_key", "")),
              "line": marker.get("line", 1), "col": marker.get("col", 1),
              "title": kind, "detail_kind": kind, "detail_args": {},
              "detail": detail or f"MiaCode {kind}"}
    if cause:
        result["cause"] = reference_note(cause) if "key" in cause else dict(cause)
    return result
