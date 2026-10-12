"""Muri marker parser ported from MiaCode 6573efb (MIT).

Preserves source columns and the reference engine's wait/duration semantics.
Unsupported syntax is reported explicitly rather than silently discarded.
"""
from __future__ import annotations

import math
import re
from typing import Any


_LANE = "12345678"
_SHAPES = "-^v<>VpqszwW"
_SHAPE_TOKEN = re.compile(r"([1-8])([bxm@?!]*)(.*)")
_DURATION = re.compile(r"\[([^\]]*)\]")
_EPS = 1e-6


def _reference_data() -> dict[str, Any]:
    from .data import reference_data
    value = reference_data()
    if not isinstance(value, dict):
        raise TypeError("MiaCode reference_data() must return a mapping")
    return value


def _duration(signature: str, bpm: float, *, slide: bool = False) -> tuple[float, float] | None:
    try:
        result = _duration_value(signature, bpm, slide=slide)
        return result if result is not None and all(math.isfinite(x) for x in result) else None
    except (OverflowError, ZeroDivisionError):
        return None


def _duration_value(signature: str, bpm: float, *, slide: bool = False) -> tuple[float, float] | None:
    """Return (wait, duration) in seconds for native MiaCode signatures."""
    if not signature:
        return None

    def fraction(text: str, use_bpm: float) -> float | None:
        match = re.fullmatch(r"\s*(\d+)\s*:\s*(-?\d+)\s*", text)
        if not match or int(match.group(1)) <= 0 or use_bpm <= 0:
            return None
        value = 240.0 * int(match.group(2)) / (use_bpm * int(match.group(1)))
        return value if not slide or value > 0 else None

    if not slide:
        parts = signature.split("#")
        if len(parts) > 2:
            return None
        if len(parts) == 2:
            if not parts[0]:
                try:
                    return 0.0, max(0.0, float(parts[1]))
                except ValueError:
                    return None
            try:
                temp_bpm = float(parts[0])
            except ValueError:
                return None
            result = fraction(parts[1], temp_bpm)
            if result is None:
                try:
                    result = max(0.0, float(parts[1]))
                except ValueError:
                    return None
            return 0.0, result
        result = fraction(signature, bpm)
        return (0.0, max(0.0, result)) if result is not None else None

    if "###" in signature or signature.count("#") > 3:
        return None

    def positive(text: str, use_bpm: float) -> float | None:
        result = fraction(text, use_bpm)
        return result if result is not None and result > 0 else None

    if "##" in signature:
        wait_text, tail = signature.split("##", 1)
        try:
            wait = max(0.0, float(wait_text))
        except ValueError:
            return None
        if "#" in tail:
            temp, duration_text = tail.split("#", 1)
            try:
                use_bpm = float(temp)
            except ValueError:
                return None
            duration = positive(duration_text, use_bpm)
        else:
            duration = positive(tail, bpm)
            if duration is None:
                try:
                    duration = float(tail)
                except ValueError:
                    return None
        return (wait, duration) if duration is not None and duration > 0 else None
    if "#" in signature:
        temp, tail = signature.split("#", 1)
        try:
            use_bpm = float(temp)
        except ValueError:
            return None
        if use_bpm <= 0:
            return None
        duration = positive(tail, use_bpm)
        if duration is None:
            try:
                duration = float(tail)
            except ValueError:
                return None
        return (60.0 / use_bpm, duration) if duration > 0 else None
    wait = 60.0 / max(1.0, bpm)
    duration = positive(signature, bpm)
    return (wait, duration) if duration is not None else None


def _shape_key(core: str) -> str:
    # Native normalizedSlideLookupKey strips timing and modifiers, retaining
    # lane digits and supported path symbols in their original order.
    core = core.split("[", 1)[0]
    match = re.match(r"[1-8]", core)
    if not match:
        return ""
    return match.group(0) + "".join(ch for ch in core[1:] if ch in _LANE + _SHAPES)


def _track_key(key: str, refs: dict[str, Any]) -> str:
    slides = refs.get("slides", {})
    if len(key) == 3 and "^" in key and key in slides:
        start, end = key[0], key[-1]
        for shape in ("<", ">"):
            candidate = start + shape + end
            if slides.get(candidate) == slides.get(key):
                return candidate
    return key


def _marker(token: str, line: int, col: int, second: float, order: int) -> dict[str, Any]:
    lane = int(token[0]) if token and token[0] in _LANE else None
    return {
        "key": f"note|{line}|{col}|{order}", "type": "tap", "second": second,
        "line": line, "col": col, "lane": lane, "end_lane": lane,
        "end_second": -1.0, "slide_trace_second": -1.0,
        "slide_track_key": "", "slide_segment_keys": [],
        "slide_segment_durations": [], "slide_segment_shoot_seconds": [],
        "slide_segment_critical_proportions": [], "slide_segment_pad_enter_times": [],
        "wifi_pad_enter_times": [], "wifi_critical_proportion": 1.0,
        "has_head_star": False, "head_ex": False, "is_ex": False,
        "is_mine": False, "track_mine": False, "head_mine": False,
        "same_head_slide": False, "head_break": False, "head_each": False,
        "slide_head": False, "slide_each": False, "is_each": False,
        "is_break": False, "track_break": False, "is_firework": False, "each_group_id": -1, "parse_order": order,
        "touch_pad": "", "on_slide": False, "tail_on_slide_head": False,
        "before_slide": False, "after_slide": False, "raw": token,
    }


def _parse_chain(core: str, bpm: float, refs: dict[str, Any]) -> tuple[list[str], float, list[float]] | None:
    """Port of parseStandardSlideChain, including same-head chain timing."""
    if not core or core[0] not in _LANE:
        return None
    shapes: list[str] = []
    waits_durations: list[tuple[float, float]] = []
    state = 0
    found = False
    target = core[0]
    i = 1
    while i < len(core):
        ch = core[i]
        if ch == "[":
            if not found:
                return None
            found = False
            if state == 0:
                state = 2
            elif state == 1:
                state = 3
            elif state == 3:
                return None
            close = core.find("]", i + 1)
            if close < 0:
                return None
            timed = _duration(core[i + 1:close], bpm, slide=True)
            if timed is None:
                return None
            waits_durations.append(timed)
            i = close + 1
            continue
        if ch not in "-^v<>Vpqsz":
            i += 1
            continue
        if state == 3:
            return None
        if found:
            if state == 0:
                state = 1
            elif state == 2:
                return None
        found = True
        shape = target + ch
        if ch == "V":
            if i + 2 >= len(core):
                return None
            shape += core[i + 1:i + 3]
            target = core[i + 2]
            i += 3
        elif ch in "pq":
            if i + 1 >= len(core):
                return None
            next_ch = core[i + 1]
            shape += next_ch
            if next_ch == ch:
                if i + 2 >= len(core):
                    return None
                shape += core[i + 2]
                target = core[i + 2]
                i += 3
            else:
                target = next_ch
                i += 2
        else:
            if i + 1 >= len(core):
                return None
            shape += core[i + 1]
            target = core[i + 1]
            i += 2
        shapes.append(shape)
    if not shapes or not waits_durations:
        return None
    wait = waits_durations[0][0]
    if state == 2:
        if len(waits_durations) != len(shapes):
            return None
        durations = [v for _, v in waits_durations]
        if len(shapes) > 1:
            entries = refs.get("slides", {})
            lengths = []
            for key in shapes:
                path = entries.get(_track_key(key, refs), {}).get("samples", [])
                length = sum(math.hypot(float(b.get("x", 0))-float(a.get("x", 0)),
                                        float(b.get("y", 0))-float(a.get("y", 0)))
                             for a, b in zip(path, path[1:]))
                lengths.append(max(1.0, length))
            total = sum(durations)
            durations = [total * length / sum(lengths) for length in lengths]
    elif state == 3:
        total = waits_durations[0][1]
        entries = refs.get("slides", {})
        lengths = []
        for key in shapes:
            path = entries.get(_track_key(key, refs), {}).get("samples", [])
            length = sum(math.hypot(float(b.get("x", 0))-float(a.get("x", 0)),
                                    float(b.get("y", 0))-float(a.get("y", 0)))
                         for a, b in zip(path, path[1:]))
            lengths.append(max(1.0, length))
        durations = [total * length / sum(lengths) for length in lengths]
    else:
        durations = [waits_durations[0][1]]
    return shapes, wait, durations


def _parse_one(token: str, line: int, col: int, second: float, order: int,
               bpm: float, refs: dict[str, Any]) -> list[dict[str, Any]]:
    if not token:
        return []
    marker = _marker(token, line, col, second, order)
    touch = re.match(r"^([ABCDE])([1-8]?)(.*)$", token)
    if touch:
        area, lane_text, tail = touch.groups()
        if area != "C" and not lane_text:
            return []
        base = area + lane_text
        duration_match = _DURATION.search(tail)
        modifier_text = tail[:duration_match.start()] + tail[duration_match.end():] if duration_match else tail
        if any(ch not in "bfhm" for ch in modifier_text):
            return []
        if ("[" in tail or "]" in tail) and (duration_match is None or not re.search("h", modifier_text, re.I)):
            return []
        marker["lane"] = marker["end_lane"] = 9
        marker["touch_pad"] = area if area == "C" else base
        marker["type"] = "touch"
        marker["is_break"] = "b" in tail.lower()
        marker["is_firework"] = "f" in tail.lower()
        marker["is_mine"] = "m" in tail
        if "h" in tail.lower():
            marker["type"] = "touch_hold"
            dur = _duration(duration_match.group(1), bpm) if duration_match else (0.0, 0.0)
            if dur is None:
                return []
            marker["end_second"] = second + dur[1]
        return [marker]

    match = _SHAPE_TOKEN.fullmatch(token)
    if not match:
        return []
    lane_s, modifiers, body = match.groups()
    head_break = "b" in modifiers
    head_ex = "x" in modifiers
    head_mine = "m" in modifiers
    headless = "?" in modifiers or "!" in modifiers
    mine = "m" in body
    core = lane_s + "".join(ch for ch in body if ch not in "bm")
    if any(ch in body for ch in _SHAPES):
        # Native branches share one physical head and source location.
        branches = core.split("*")
        out = []
        for branch in branches:
            if not branch:
                continue
            branch = branch if branch[0] in _LANE else lane_s + branch
            bm = _marker(token, line, col, second, order + len(out))
            bm.update(type="wifi" if "w" in branch.lower() else "slide",
                      lane=int(lane_s), end_lane=int(lane_s), has_head_star=not headless,
                      head_break=head_break, track_break="b" in body, is_break=head_break or ("b" in body),
                      head_ex=head_ex, is_ex=False, head_mine=head_mine,
                      track_mine=mine, slide_head=False)
            clean = "".join(ch for ch in branch if ch not in "bm")
            parsed = _parse_chain(clean, bpm, refs) if bm["type"] == "slide" else None
            if bm["type"] == "slide":
                if not parsed:
                    raise ValueError("Invalid slide duration or shape chain")
                keys, wait, durations = parsed
                keys = [_track_key(k, refs) for k in keys]
                entries = refs.get("slides", {})
                trace = second + wait
                bm["slide_trace_second"] = trace
                bm["slide_track_key"] = keys[0] if keys else ""
                bm["slide_segment_keys"] = keys
                bm["slide_segment_durations"] = durations
                bm["slide_segment_shoot_seconds"] = []
                bm["slide_segment_critical_proportions"] = []
                bm["slide_segment_pad_enter_times"] = []
                shoot = trace
                for key, duration in zip(keys, durations):
                    entry = entries.get(key, {})
                    if not entry:
                        raise ValueError(f"unknown slide shape {key}")
                    bm["slide_segment_shoot_seconds"].append(shoot)
                    bm["slide_segment_critical_proportions"].append(float(entry.get("critical_proportion", 1.0)))
                    bm["slide_segment_pad_enter_times"].append(list(entry.get("pad_enter_times", [])))
                    bm["end_lane"] = int(entry.get("end", bm["end_lane"]))
                    shoot += max(0.001, duration)
                bm["end_second"] = shoot
            else:
                # Single Wifi references use one wait/duration signature.
                found = list(_DURATION.finditer(clean))
                timed = _duration(found[-1].group(1), bpm, slide=True) if found else None
                if timed is None:
                    raise ValueError("Invalid slide duration")
                wait, duration = timed
                trace = second + wait
                bm["slide_trace_second"] = trace
                bm["end_second"] = trace + duration
                key = _shape_key(clean)
                entry = refs.get("wifi", {}).get(key, {})
                if not entry:
                    raise ValueError(f"unknown wifi shape {key}")
                bm["slide_track_key"] = key
                bm["slide_segment_keys"] = [key]
                bm["slide_segment_shoot_seconds"] = [trace]
                bm["slide_segment_durations"] = [max(0.001, duration)]
                bm["wifi_pad_enter_times"] = list(entry.get("pad_enter_times", []))
                bm["wifi_critical_proportion"] = float(entry.get("critical_proportion", 1.0))
                bm["end_lane"] = int(entry.get("end", bm["end_lane"]))
            out.append(bm)
        return out

    # Native tap/hold modifiers may occur on either side of [duration].
    suffix = token[1:]
    hm = re.search(r"h", suffix, re.I)
    bracket = _DURATION.search(suffix)
    modifier_text = suffix
    if bracket:
        modifier_text = suffix[:bracket.start()] + suffix[bracket.end():]
    if "[" in suffix or "]" in suffix:
        if bracket is None or suffix.count("[") != 1 or suffix.count("]") != 1 or not hm:
            return []
    if any(ch in "BX" or (ch.lower() not in "bxh" and ch not in "m$") for ch in modifier_text):
        return []
    if modifier_text.lower().count("b") > 1 or modifier_text.lower().count("x") > 1 \
            or modifier_text.lower().count("h") > 1 or modifier_text.count("m") > 1:
        return []
    if hm and "$" in modifier_text:
        return []
    marker.update(is_break="b" in suffix.lower(), is_ex="x" in suffix.lower(),
                  is_mine="m" in suffix, lane=int(lane_s), end_lane=int(lane_s))
    if hm:
        marker["type"] = "hold"
        dur = _duration(bracket.group(1), bpm) if bracket else (0.0, 0.0)
        if dur is None:
            return []
        marker["end_second"] = second + dur[1]
    return [marker]


def _interval_match(starts: list[tuple[float, int]], queries: list[tuple[float, int]],
                    before: float, after: float, *, exclude_self: bool = False) -> set[int]:
    result: set[int] = set()
    for start, _ in starts:
        for query, index in queries:
            delta = query - start
            if -before < delta < after and not (exclude_self and index == _):
                result.add(index)
    return result


def _postprocess(markers: list[dict[str, Any]], each_groups: list[list[int]]) -> None:
    for group_id, indices in enumerate(each_groups):
        eligible = [i for i in indices if not markers[i]["is_mine"] and not markers[i]["track_mine"]]
        if not eligible:
            continue
        for i in eligible:
            markers[i]["each_group_id"] = group_id
        touches = [i for i in eligible if markers[i]["type"] == "touch"]
        taps = [i for i in eligible if markers[i]["type"] == "tap"]
        holds = [i for i in eligible if markers[i]["type"] in {"hold", "touch_hold"}]
        slides = [i for i in eligible if markers[i]["type"] in {"slide", "wifi"}]
        head_lanes = {markers[i]["lane"] for i in slides if markers[i]["has_head_star"]}
        headless = sum(not markers[i]["has_head_star"] for i in slides)
        group_count = len(touches) + len(taps) + len(holds) + len(head_lanes) + headless
        if group_count >= 2:
            for i in touches + taps + holds:
                markers[i]["is_each"] = True
            for i in slides:
                markers[i]["head_each"] = True
        if len(slides) >= 2:
            for i in slides:
                markers[i]["slide_each"] = True
        for i in slides:
            markers[i]["same_head_slide"] = any(
                j != i and markers[j]["type"] in {"slide", "wifi"}
                and not markers[j]["track_mine"] and markers[j]["lane"] == markers[i]["lane"]
                for j in indices
            ) and not markers[i]["track_mine"]

    trace: dict[int, list[tuple[float, int]]] = {}
    ends: dict[int, list[tuple[float, int]]] = {}
    taps: dict[int, list[tuple[float, int]]] = {}
    tails: dict[int, list[tuple[float, int]]] = {}
    touches: dict[str, list[tuple[float, int]]] = {}
    slide_heads: dict[str, list[tuple[float, int]]] = {}
    wifi_heads: dict[str, list[tuple[float, int]]] = {}
    pad_enters: dict[str, list[tuple[float, int]]] = {}
    for i, m in enumerate(markers):
        typ = m["type"]
        lane = m["lane"]
        if typ == "tap":
            taps.setdefault(lane, []).append((m["second"], i))
        elif typ == "hold" and m["end_second"] >= 0:
            tails.setdefault(lane, []).append((m["end_second"], i))
        elif typ == "touch":
            if m["touch_pad"]:
                touches.setdefault(m["touch_pad"].upper(), []).append((m["second"], i))
        if typ not in {"slide", "wifi"} or m["slide_trace_second"] < 0:
            continue
        trace.setdefault(lane, []).append((m["slide_trace_second"], i))
        if m["has_head_star"]:
            taps.setdefault(lane, []).append((m["second"], i))
        headpad = f"A{lane}"
        (slide_heads if typ == "slide" else wifi_heads).setdefault(headpad, []).append((m["slide_trace_second"], i))
        if m["end_second"] >= m["slide_trace_second"]:
            ends.setdefault(m["end_lane"], []).append((m["end_second"], i))
        if typ == "slide":
            for keys, shoots, durations, pad_sets in zip(
                m["slide_segment_keys"], m["slide_segment_shoot_seconds"],
                m["slide_segment_durations"], m["slide_segment_pad_enter_times"]
            ):
                for item in pad_sets:
                    pad_enters.setdefault(str(item.get("pad", "")).upper(), []).append(
                        (shoots + float(item.get("t", 0)) * durations, i))
        elif m["end_second"] >= m["slide_trace_second"]:
            duration = m["end_second"] - m["slide_trace_second"]
            for item in m["wifi_pad_enter_times"]:
                pad_enters.setdefault(str(item.get("pad", "")).upper(), []).append(
                    (m["slide_trace_second"] + float(item.get("t", 0)) * duration, i))

    tap_threshold = 1.0 / 180.0
    touch_threshold = 24.0 / 180.0
    for lane, starts in trace.items():
        for i in _interval_match(starts, taps.get(lane, []), tap_threshold, tap_threshold):
            markers[i]["slide_head"] = True
        for i in _interval_match(starts, tails.get(lane, []), tap_threshold, tap_threshold):
            markers[i]["tail_on_slide_head"] = True
    for pad, queries in touches.items():
        for starts in (slide_heads.get(pad, []), wifi_heads.get(pad, []), pad_enters.get(pad, [])):
            for i in _interval_match(starts, queries, touch_threshold, tap_threshold):
                markers[i]["on_slide"] = True
    for lane, queries in trace.items():
        for i in _interval_match(ends.get(lane, []), queries, tap_threshold, tap_threshold, exclude_self=True):
            markers[i]["after_slide"] = True
    for lane, queries in ends.items():
        for i in _interval_match(trace.get(lane, []), queries, tap_threshold, tap_threshold, exclude_self=True):
            markers[i]["before_slide"] = True


def parse_markers(inote: str, first: float = 0.0) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Parse a simai inote string into MiaCode-shaped markers and diagnostics."""
    refs = _reference_data()
    markers: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    second = float(first or 0.0)
    bpm = 120.0
    beats = 4
    groups: list[list[int]] = []
    group: list[int] = []

    def fail(line: int, col: int, raw: str, message: str) -> None:
        errors.append({"line": line, "col": col, "raw": raw, "message": message})

    for line_no, source_line in enumerate(str(inote).splitlines(), 1):
        line = source_line.rstrip("\r")
        i = 0
        token = ""
        token_col = 1

        def flush() -> None:
            nonlocal token
            if not token:
                return
            if token.upper() != "E":
                pieces = token.split("*") if "*" not in token else [token]
                # Numeric chord shorthand is expanded into native individual taps.
                if len(token) > 1 and all(ch in _LANE for ch in token):
                    pieces = list(token)
                    columns = [token_col + n for n in range(len(token))]
                else:
                    columns = [token_col] * len(pieces)
                for piece, col in zip(pieces, columns):
                    try:
                        parsed = _parse_one(piece, line_no, col, second, len(markers), bpm, refs)
                        if not parsed:
                            fail(line_no, col, piece, f"Invalid note: {piece}")
                            continue
                        for m in parsed:
                            m["parse_order"] = len(markers)
                            markers.append(m)
                            group.append(len(markers) - 1)
                    except (ValueError, TypeError, KeyError) as exc:
                        fail(line_no, col, piece, str(exc))
            token = ""

        while i < len(line):
            ch = line[i]
            if line.startswith("||", i):
                flush()
                break
            if ch.isspace():
                flush()
                i += 1
                continue
            if ch == "(":
                flush()
                close = line.find(")", i + 1)
                if close < 0:
                    fail(line_no, i + 1, line[i:], "Unterminated BPM block")
                    break
                try:
                    new_bpm = float(line[i + 1:close].strip())
                    if not math.isfinite(new_bpm) or new_bpm <= 0:
                        raise ValueError
                    bpm = new_bpm
                except ValueError:
                    fail(line_no, i + 1, line[i:close + 1], "Invalid BPM value")
                i = close + 1
                continue
            if ch == "{":
                flush()
                close = line.find("}", i + 1)
                if close < 0:
                    fail(line_no, i + 1, line[i:], "Unterminated beat block")
                    break
                try:
                    beats = int(line[i + 1:close].strip())
                    if beats <= 0:
                        raise ValueError
                except ValueError:
                    fail(line_no, i + 1, line[i:close + 1], "Invalid beat value")
                i = close + 1
                continue
            if ch in ",/`":
                flush()
                if ch == "`":
                    if group:
                        groups.append(group)
                        group = []
                elif ch == ",":
                    if group:
                        groups.append(group)
                        group = []
                    second += 240.0 / (bpm * max(1, beats))
                i += 1
                continue
            if not token:
                token_col = i + 1
            token += ch
            i += 1
        flush()
        if group:
            groups.append(group)
            group = []
    _postprocess(markers, groups)
    return markers, errors, warnings
