"""180 Hz action and simple-note simulation ported from MiaCode 6573efb (MIT).

The simple-note pad-down stream and the geometric Slide judge are different
upstream stages. Keep both: replacing either with static overlap changes Muri.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math

from maichart.generation.judgment_system import touch_groups
from .data import reference_data
from .model import (TPS, EPS, RELEASE, AVAILABLE, PADS, RADII, CENTERS,
                    start_tick, end_tick, has_head, head_key, head_emit_key,
                    reference_note, head_target, diagnostic, pad, usable_slide,
                    tap_alert, head_alert, slide_with_tap_keys)


def simple_notes(markers):
    notes = []
    for m in markers:
        if m.get("is_mine") or m.get("track_mine") or m["type"] in {"slide", "wifi"}:
            continue
        notes.append(dict(m, pad=pad(m), protected=bool(m.get("is_ex")), head=False))
    emitted = set()
    for m in markers:
        if not has_head(m) or head_emit_key(m) in emitted:
            continue
        emitted.add(head_emit_key(m))
        notes.append(dict(m, type="tap", key=head_key(m), col=m["col"] + 1,
                          pad=f"A{m['lane']}", end_second=-1., head=True,
                          protected=bool(m.get("head_ex")), owner_key=m["key"]))
    for n in notes:
        hold = n["type"] in {"hold", "touch_hold"}
        n["press_end"] = (max(n["second"], n.get("end_second", -1)) if hold else n["second"]) + (
            0 if hold and n.get("tail_on_slide_head") else RELEASE)
        n["critical"] = AVAILABLE if n["type"] in {"touch", "touch_hold"} else RELEASE
        n["expiry"] = math.floor((n["second"] + AVAILABLE) * TPS + EPS) + 1
        n.update(judged=False, judge_bad=False, judge_second=-1., cause=None)
    # C++ top-level order: each cell's button targets first, Touch components
    # afterwards, each ordered by original parse position.
    cells = defaultdict(list)
    for n in notes:
        cells[n.get("each_group_id", -1)].append(n)
    ordered, groups = [], []
    for _, cell in sorted(cells.items(), key=lambda item: min(n["parse_order"] for n in item[1])):
        cell.sort(key=lambda n: n["parse_order"])
        ordered.extend(n for n in cell if n["type"] != "touch")
        touches = [n for n in cell if n["type"] == "touch"]
        for indexes in touch_groups([n["pad"] for n in touches]):
            children = [touches[i] for i in indexes]
            if len(children) > 1:
                group = {"children": children, "all_on_slide": all(n.get("on_slide") for n in children)}
                groups.append(group)
                for n in children:
                    n["touch_group"] = group
            ordered.extend(children)
    return ordered, groups


def _cause(n, second, *, extra=False):
    return {"marker_key": n["key"], "source_type": n["type"],
            "source_order": n.get("parse_order", -1), "line": n["line"],
            "col": n["col"], "second": second, "extra_pad_down": extra}


def simple_pad_events(markers, notes):
    events = defaultdict(dict)

    def put(tick, p, cause, overwrite=False):
        if overwrite or p not in events[tick] or events[tick][p]["source_order"] <= cause["source_order"]:
            events[tick][p] = dict(cause, pad=p, second=tick / TPS)

    for m in markers:
        if m.get("is_mine") or m.get("track_mine") or not usable_slide(m):
            continue
        if not m.get("after_slide"):
            times = (m.get("slide_segment_pad_enter_times") or [[]])[0] if m["type"] == "slide" else m["wifi_pad_enter_times"]
            duration = (m.get("slide_segment_durations") or [0])[0] if m["type"] == "slide" else m["end_second"] - m["slide_trace_second"]
            extra = min(max(0., times[0]["t"] * duration), .05) if times else .05
            tick = math.floor((m["slide_trace_second"] + extra) * TPS + EPS) + 1
            put(tick, f"A{m['lane']}", _cause(m, tick / TPS, extra=True), True)
    for m in markers:
        if m.get("is_mine") or m.get("track_mine") or not usable_slide(m):
            continue
        put(start_tick(m["slide_trace_second"]), f"A{m['lane']}", _cause(m, m["slide_trace_second"]))
    for m in markers:
        if m.get("is_mine") or m.get("track_mine"):
            continue
        if m["type"] == "slide":
            sections = zip(m["slide_segment_pad_enter_times"], m["slide_segment_shoot_seconds"], m["slide_segment_durations"])
        elif m["type"] == "wifi":
            sections = [(m["wifi_pad_enter_times"], m["slide_trace_second"], m["end_second"]-m["slide_trace_second"])]
        else:
            continue
        for times, shoot, duration in sections:
            for item in times:
                second = shoot + item["t"] * duration
                put(start_tick(second), item["pad"], _cause(m, second))

    # Presses share electrical pad occupancy; an additional hold/tap on a pad
    # already held down does not manufacture a second rising edge.
    intervals = defaultdict(list)
    for n in notes:
        group = n.get("touch_group")
        if n["type"] == "tap" and n.get("slide_head") and not n["head"]:
            continue
        if n["type"] == "touch" and (group["all_on_slide"] if group else n.get("on_slide")):
            continue
        begin = start_tick(n["second"])
        end = max(begin, end_tick(n["press_end"]))
        intervals[n["pad"]].append((begin, end + 1, n))
    for p, entries in intervals.items():
        changes = defaultdict(lambda: ([], []))
        for i, (begin, end, n) in enumerate(entries):
            changes[begin][0].append(i)
            changes[end][1].append(i)
        active = set()
        for tick, (add, remove) in sorted(changes.items()):
            had_active = bool(active)
            active.difference_update(remove)
            active.update(add)
            if not had_active and active:
                chosen = max(active, key=lambda i: entries[i][2]["parse_order"])
                put(tick, p, _cause(entries[chosen][2], tick/TPS), True)
    return events


def judge_simple(markers, notes, groups, threshold):
    events = simple_pad_events(markers, notes)
    ticks = sorted(set(events) | {n["expiry"] for n in notes})
    by_pad = defaultdict(list)
    for n in notes:
        by_pad[n["pad"]].append(n)

    def judge(n, second, cause):
        n.update(judged=True, judge_second=second,
                 judge_bad=abs(second - n["second"]) > n["critical"] + EPS, cause=cause)

    for tick in ticks:
        second = tick / TPS
        for p, cause in sorted(events.get(tick, {}).items()):
            for n in by_pad[p]:
                if not n["judged"] and second - n["second"] >= -AVAILABLE - EPS:
                    judge(n, second, cause)
                    break
        for n in notes:
            if not n["judged"] and second - n["second"] > AVAILABLE + EPS:
                judge(n, second, None)
        for group in groups:
            children = group["children"]
            if sum(n["judged"] for n in children) + EPS >= .51 * len(children):
                for n in children:
                    if not n["judged"]:
                        judge(n, second, _cause(n, second))

    lookup = {m["key"]: m for m in markers}
    owners = {n["key"]: n["owner_key"] for n in notes if n["head"]}
    tap_heads = slide_with_tap_keys(markers)
    diagnostics = []
    for n in notes:
        if not n["judged"] or not n["judge_bad"] or n.get("touch_group"):
            continue
        cause = n["cause"]
        occurrence = n["judge_second"]
        if occurrence + EPS < n["second"] and cause and cause["marker_key"]:
            owner = owners.get(cause["marker_key"], cause["marker_key"])
            source = lookup.get(owner)
            at_head = (cause.get("extra_pad_down") or cause["marker_key"].startswith("slide_head_star|")
                       or source is not None and source["type"] in {"slide", "wifi"} and n["pad"] == f"A{source['lane']}")
            gap = max(0., n["second"] - (source["slide_trace_second"] if at_head and source and usable_slide(source) else occurrence))
            alert = head_alert(owner in tap_heads, gap) if at_head else tap_alert(gap, threshold)
            if n["protected"]:
                alert = "Warning"
            kind = "SlideHeadTap" if at_head else "TapOnSlide"
            diagnostics.append(diagnostic(kind, alert, n, occurrence, cause,
                                          f"{n['pad']} 提前 {gap*1000:.1f} ms 触发；来源 {cause['marker_key']}"))
        elif not n.get("slide_head"):
            diagnostics.append(diagnostic("Overlap", "Muri", n, occurrence, cause,
                                          f"{n['pad']} 在有效窗口内未得到独立按下，存在动作重叠。"))
    return diagnostics


@dataclass
class Action:
    marker: dict
    begin: float
    end: float
    duration: float
    points: tuple
    radius: float = 15.
    merge: str = ""

    def sample(self, second):
        if second + EPS < self.begin or second >= self.end - EPS:
            return None
        if not self.merge or len(self.points) == 1:
            return self.points[0], None
        progress = min(1., max(0., (second - self.begin) / self.duration)) if self.duration > EPS else 1.
        value = progress * (len(self.points) - 1)
        index = min(int(value), len(self.points) - 1)
        a, b = self.points[index], self.points[min(index + 1, len(self.points)-1)]
        t = value - index
        dx, dy = b[0]-a[0], b[1]-a[1]
        length = math.hypot(dx, dy)
        tangent = (dx/length, dy/length) if length > EPS else None
        return (a[0]+dx*t, a[1]+dy*t), tangent


def build_actions(markers, notes):
    actions = []
    for n in notes:
        group = n.get("touch_group")
        if n["type"] == "tap" and n.get("slide_head"):
            continue
        if n["type"] == "touch" and (group["all_on_slide"] if group else n.get("on_slide")):
            continue
        actions.append(Action(n, n["second"], n["press_end"], 0, (CENTERS[n["pad"]],)))
    data = reference_data()
    for m in markers:
        if m.get("is_mine") or m.get("track_mine"):
            continue
        if m["type"] == "slide":
            segments = list(zip(m["slide_segment_keys"], m["slide_segment_shoot_seconds"], m["slide_segment_durations"]))
            for i, (key, begin, duration) in enumerate(segments):
                entry = data["slides"][key]
                points = tuple((p["x"], p["y"]) for p in entry.get("real_path_samples", entry["samples"]))
                end = begin + duration + (RELEASE if i == len(segments)-1 and not m.get("before_slide") else 0.)
                actions.append(Action(m, begin, end, duration, points, merge="normal"))
        elif m["type"] == "wifi":
            entry = data["wifi"][m["slide_track_key"]]
            paths = entry.get("di_real_path_samples", entry["lane_samples"])
            for path in (paths[0], paths[-1]):
                actions.append(Action(m, m["slide_trace_second"], m["end_second"],
                                      m["end_second"]-m["slide_trace_second"],
                                      tuple((p["x"],p["y"]) for p in path), 50., "wifi"))
    return sorted(actions, key=lambda a: a.begin)


def _merged_points(actions, active, second, previous=None):
    points, pairs = [], set()
    for index in active:
        action = actions[index]
        sample = action.sample(second)
        if sample is None:
            continue
        position, tangent = sample
        merged = False
        if action.merge:
            for ei, ep, et in points:
                other = actions[ei]
                if other.merge != action.merge:
                    continue
                pair = tuple(sorted((ei, index)))
                regular = (tangent is not None and et is not None
                           and math.dist(ep, position) < 10. and math.dist(et, tangent) < .05235389661574631)
                tail = (previous is not None and pair in previous
                        and (second + EPS >= action.begin + action.duration
                             or second + EPS >= other.begin + other.duration))
                if regular or tail:
                    pairs.add(pair)
                    merged = True
                    break
        if not merged:
            points.append((index, position, tangent))
    return points, pairs


def simulate_actions(markers, notes, states):
    from .slide import step_slides

    actions = build_actions(markers, notes)
    if not actions:
        return []
    previous_pads, active, previous_pairs = {}, [], set()
    shared_start = set()
    diagnostics, signatures = [], {}
    maximum = max(0, max(end_tick(a.end) for a in actions))
    pointer, tick = 0, 0
    # Geometry remains in MiaCode's logical canvas units; no resampling or
    # replacement by the simplified legacy detector's circle footprints.
    sensor_geometry = [(p, CENTERS[p], RADII[p[0]]) for p in PADS]
    while tick <= maximum:
        second = tick / TPS
        while pointer < len(actions) and second + EPS >= actions[pointer].begin:
            active.append(pointer)
            pointer += 1
        regular, _ = _merged_points(actions, active, second)
        pad_states = {}
        for ai, position, _ in regular:
            action = actions[ai]
            cause = _cause(action.marker, second)
            for p, pc, radius in sensor_geometry:
                if math.dist(position, pc) <= action.radius + radius + EPS:
                    pad_states[p] = dict(cause, pad=p)
        pad_up = {p: c for p, c in previous_pads.items() if p not in pad_states}
        step_slides(states, second, pad_states, pad_up)
        previous_pads = pad_states

        # Native MuriRenderOptions defaults excludeTouchFromMultiTouch=true.
        # Touch still occupies pads for Slide judgement and simple-note checks.
        hand_active = [i for i in active if actions[i].marker["type"] not in {"touch", "touch_hold"}]
        points, pairs = _merged_points(actions, hand_active, second, previous_pairs)
        for left, right in pairs:
            a, b = actions[left], actions[right]
            am, bm = a.marker, b.marker
            if (am["type"] == bm["type"] == "slide" and am["lane"] == bm["lane"]
                    and abs(am["second"]-bm["second"]) <= EPS
                    and abs(a.begin-b.begin) <= EPS and abs(a.begin-am["slide_trace_second"]) <= EPS
                    and abs(b.begin-bm["slide_trace_second"]) <= EPS and tick == start_tick(a.begin)):
                shared_start.add((am["key"],bm["key"]))
        clusters = []
        for ai, position, _ in points:
            for indexes, cp, radius in clusters:
                if (not actions[ai].merge and not actions[indexes[0]].merge
                        and math.dist(cp, position) <= .001 and abs(radius-actions[ai].radius) <= .001):
                    indexes.append(ai)
                    break
            else:
                clusters.append(([ai], position, actions[ai].radius))
        count = len(clusters)
        indexes = [c[0][0] for c in clusters]
        merged_start = False
        if count == 2 and any(not actions[i].merge for i in indexes):
            for a, b in pairs:
                if (actions[a].marker["key"], actions[b].marker["key"]) in shared_start:
                    merged_start = True
                    indexes.extend(i for i in (a,b) if i not in indexes)
        if count > 2 or merged_start:
            non_touch = sum(any(actions[i].marker["type"] not in {"touch", "touch_hold"} for i in c[0]) for c in clusters)
            involves_touch = any(actions[i].marker["type"] in {"touch", "touch_hold"} for c in clusters for i in c[0])
            alert = "Warning" if merged_start or involves_touch and non_touch <= 2 else "Muri"
            indexes.sort(key=lambda i: (actions[i].marker["parse_order"], actions[i].marker["line"], actions[i].marker["col"]))
            unique = {}
            for i in indexes:
                unique.setdefault(actions[i].marker["key"], i)
            signature = tuple(unique)
            existing = signatures.get(signature)
            if existing is None or diagnostics[existing]["alert_level"] == "Warning" and alert == "Muri":
                anchor = min(unique.values(), key=lambda i: (actions[i].begin, actions[i].marker["parse_order"], actions[i].marker["line"], actions[i].marker["col"]))
                item = diagnostic("MultiTouch", alert, actions[anchor].marker, second,
                                  detail=f"同时动作 {count} 个：" + ", ".join(signature))
                item["sources"] = [reference_note(actions[i].marker) for i in unique.values()]
                if existing is None:
                    signatures[signature] = len(diagnostics)
                    diagnostics.append(item)
                else:
                    diagnostics[existing] = item
        previous_pairs = pairs
        active = [i for i in active if second + EPS < actions[i].end]
        tick += 1
        if not active and not previous_pads and pointer < len(actions):
            tick = max(tick, start_tick(actions[pointer].begin))
    return diagnostics
