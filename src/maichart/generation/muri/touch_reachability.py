"""Symmetrical 2-hand Touch & TouchHold reachability checker for maimai DX.

Upstream MiaCode sets ``excludeTouchFromMultiTouch = true``, which avoids false
positives on multi-sensor Touch fireworks and Slide pass-throughs, but leaves
genuine 3-point locks unflagged (for example ``1h + 5h + C``, ``Ch + 1h + 5``,
or ``1h + 4>5 + E7``).

This module checks whether active button/hold/slide demands together with
simultaneous ``Touch`` / ``TouchHold`` demands can be physically covered by two
hands without assigning permanent left/right ownership:

* Simultaneous ``Touch`` sensors connected in ``touch_groups`` merge according
  to maimai DX's >=51% group auto-clear rule.
* ``Touch`` notes inside a generous spatio-temporal corridor around any active
  ``Slide`` / ``Wifi`` trajectory (or already flagged ``on_slide``) are absorbed
  by the sweeping hand at zero extra hand cost.
* Sustained ``Hold`` / ``TouchHold`` / ``Slide`` tails include a release grace
  window so normal release-into-Touch transitions within Touch's +-150 ms
  judgment window are never treated as 3-hand locks.
* Slide wait intervals ``[second, slide_trace_second)`` do not lock a hand.
"""
from __future__ import annotations

from collections import defaultdict, deque
import math
from typing import Any

from maichart.generation.judgment_system import touch_groups
from .data import reference_data
from .model import (
    CENTERS, EPS, TPS, diagnostic, end_tick, has_head, head_emit_key, head_key,
    pad as marker_pad, reference_note, start_tick, usable_slide,
)

TOUCH_REACHABILITY_POLICIES = ("off", "palm_span_v1", "strict_on_slide_only")
DEFAULT_COHOLD_RADIUS = 180.0
STRICT_COHOLD_RADIUS = 45.0
SLIDE_SWEEP_RADIUS = 175.0
WIFI_SWEEP_RADIUS = 195.0
SLIDE_SWEEP_EARLY_SEC = 0.22
SLIDE_SWEEP_LATE_SEC = 0.28
TAIL_RELEASE_GRACE_SEC = 0.075
MIN_SUSTAIN_LOCK_SEC = 0.10
SIMULTANEOUS_PULSE_SEC = 0.025
PARALLEL_SLIDE_MERGE_DIST = 120.0


def validate_touch_reachability(
    policy: str = "off",
    cohold_radius: float = DEFAULT_COHOLD_RADIUS,
) -> tuple[str, float]:
    normalized = str(policy or "off").strip()
    if normalized not in TOUCH_REACHABILITY_POLICIES:
        raise ValueError(
            f"touch_reachability_policy must be one of {', '.join(TOUCH_REACHABILITY_POLICIES)}"
        )
    try:
        radius = float(cohold_radius)
    except (TypeError, ValueError) as exc:
        raise ValueError("touch_cohold_radius must be a finite number") from exc
    if not math.isfinite(radius) or not 60.0 <= radius <= 260.0:
        raise ValueError("touch_cohold_radius must be between 60 and 260 canvas units")
    return normalized, radius


def effective_cohold_radius(policy: str, radius: float) -> float:
    return min(float(radius), STRICT_COHOLD_RADIUS) if policy == "strict_on_slide_only" else float(radius)


def _bipartite_partitions(
    points: list[tuple[float, float]],
    radius: float,
) -> list[tuple[list[tuple[float, float]], list[tuple[float, float]]]] | None:
    """Partition ``points`` into two cliques of diameter <= ``radius``.

    Two subsets each have pairwise distance <= ``radius`` iff the conflict graph
    with edges ``dist(u, v) > radius`` is bipartite. Returns up to 16 valid
    2-colorings, or ``None`` if an odd conflict cycle makes 2-hand cover impossible.
    """
    n = len(points)
    if n == 0:
        return [([], [])]
    if n == 1:
        return [([points[0]], [])]
    adj: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            if math.dist(pi, points[j]) > radius + EPS:
                adj[i].append(j)
                adj[j].append(i)
    color = [-1] * n
    components: list[tuple[list[int], list[int]]] = []
    for start in range(n):
        if color[start] != -1:
            continue
        color[start] = 0
        queue = deque([start])
        side0: list[int] = []
        side1: list[int] = []
        while queue:
            u = queue.popleft()
            (side0 if color[u] == 0 else side1).append(u)
            for v in adj[u]:
                if color[v] == -1:
                    color[v] = 1 - color[u]
                    queue.append(v)
                elif color[v] == color[u]:
                    return None
        components.append((side0, side1))
    # Combine connected conflict components (cap combinations to 16; nontrivial
    # components with edges are at most 2..4 in any chart timestamp).
    nontrivial = [comp for comp in components if comp[1]]
    singletons = [comp[0][0] for comp in components if not comp[1]]
    partitions: list[tuple[list[int], list[int]]] = [([], [])]
    for idx, (s0, s1) in enumerate(nontrivial):
        next_parts: list[tuple[list[int], list[int]]] = []
        options = ((s0, s1),) if idx == 0 else ((s0, s1), (s1, s0))
        for left, right in partitions:
            for a, b in options:
                cand_l = left + a
                cand_r = right + b
                if _clique_ok(points, cand_l, radius) and _clique_ok(points, cand_r, radius):
                    next_parts.append((cand_l, cand_r))
                    if len(next_parts) >= 16:
                        break
            if len(next_parts) >= 16:
                break
        if not next_parts:
            return None
        partitions = next_parts
    for u in singletons:
        next_parts = []
        for left, right in partitions:
            if _can_join_clique(points, left, u, radius):
                next_parts.append((left + [u], right))
            if _can_join_clique(points, right, u, radius):
                next_parts.append((left, right + [u]))
            if len(next_parts) >= 16:
                break
        if not next_parts:
            return None
        partitions = next_parts
    return [([points[i] for i in left], [points[j] for j in right]) for left, right in partitions]


def _clique_ok(points: list[tuple[float, float]], idxs: list[int], radius: float) -> bool:
    for i in range(len(idxs)):
        pi = points[idxs[i]]
        for j in range(i + 1, len(idxs)):
            if math.dist(pi, points[idxs[j]]) > radius + EPS:
                return False
    return True


def _can_join_clique(points: list[tuple[float, float]], idxs: list[int], candidate: int, radius: float) -> bool:
    pc = points[candidate]
    return all(math.dist(pc, points[i]) <= radius + EPS for i in idxs)


def can_two_hands_cover(
    point_demands: list[tuple[float, float]],
    group_demands: list[tuple[list[tuple[float, float]], int]] | None = None,
    radius: float = DEFAULT_COHOLD_RADIUS,
) -> bool:
    """Return True iff 2 hands of span ``radius`` can satisfy all demands."""
    partitions = _bipartite_partitions(point_demands, radius)
    if partitions is None:
        return False
    if not group_demands:
        return True
    for hand0, hand1 in partitions:
        if _groups_coverable(hand0, hand1, group_demands, radius):
            return True
    return False


def _groups_coverable(
    hand0: list[tuple[float, float]],
    hand1: list[tuple[float, float]],
    group_demands: list[tuple[list[tuple[float, float]], int]],
    radius: float,
) -> bool:
    free_hands = int(not hand0) + int(not hand1)
    unmet = 0
    for pads, need in group_demands:
        covered: set[int] = set()
        if hand0:
            for idx, q in enumerate(pads):
                if all(math.dist(p, q) <= radius + EPS for p in hand0):
                    covered.add(idx)
        if hand1:
            for idx, q in enumerate(pads):
                if all(math.dist(p, q) <= radius + EPS for p in hand1):
                    covered.add(idx)
        if len(covered) >= need:
            continue
        # A completely free hand can slap/wipe a connected Touch group within +-150ms.
        unmet += 1
        if unmet > free_hands:
            return False
    return True


class SlideSweepTrack:
    """Precomputed native Slide / Wifi trajectory for fast sweep & position queries."""

    __slots__ = ("marker", "trace_start", "end_sec", "is_wifi", "segments")

    def __init__(
        self,
        marker: dict[str, Any],
        trace_start: float,
        end_sec: float,
        is_wifi: bool,
        segments: list[tuple[float, float, tuple[tuple[float, float], ...]]],
    ):
        self.marker = marker
        self.trace_start = trace_start
        self.end_sec = end_sec
        self.is_wifi = is_wifi
        self.segments = segments

    def sample_at(self, second: float) -> tuple[float, float] | None:
        if second < self.trace_start - EPS or second > self.end_sec - TAIL_RELEASE_GRACE_SEC:
            return None
        for begin, finish, pts in self.segments:
            if begin - EPS <= second <= finish + EPS and pts:
                if len(pts) == 1 or finish <= begin + EPS:
                    return pts[0]
                progress = min(1.0, max(0.0, (second - begin) / (finish - begin)))
                val = progress * (len(pts) - 1)
                idx = min(int(val), len(pts) - 1)
                nxt = min(idx + 1, len(pts) - 1)
                frac = val - idx
                return (
                    pts[idx][0] + (pts[nxt][0] - pts[idx][0]) * frac,
                    pts[idx][1] + (pts[nxt][1] - pts[idx][1]) * frac,
                )
        return None

    def sweeps_pad(self, pad_xy: tuple[float, float], touch_sec: float) -> bool:
        win_lo = touch_sec - SLIDE_SWEEP_LATE_SEC
        win_hi = touch_sec + SLIDE_SWEEP_EARLY_SEC
        if self.end_sec < win_lo or self.trace_start > win_hi:
            return False
        limit = WIFI_SWEEP_RADIUS if self.is_wifi else SLIDE_SWEEP_RADIUS
        for begin, finish, pts in self.segments:
            if finish < win_lo or begin > win_hi or not pts:
                continue
            # Sample densely along the overlapping sub-interval of this segment.
            if finish <= begin + EPS:
                sub_pts = pts
            else:
                p_lo = max(0.0, (win_lo - begin) / (finish - begin))
                p_hi = min(1.0, (win_hi - begin) / (finish - begin))
                i_lo = max(0, int(math.floor(p_lo * (len(pts) - 1))) - 1)
                i_hi = min(len(pts) - 1, int(math.ceil(p_hi * (len(pts) - 1))) + 1)
                sub_pts = pts[i_lo : i_hi + 1]
            for pt in sub_pts:
                if math.dist(pad_xy, pt) <= limit + EPS:
                    return True
        return False


def build_slide_tracks(markers: list[dict[str, Any]]) -> list[SlideSweepTrack]:
    data = reference_data()
    slides_ref = data.get("slides", {})
    wifi_ref = data.get("wifi", {})
    tracks: list[SlideSweepTrack] = []
    for m in markers:
        if m.get("is_mine") or m.get("track_mine") or not usable_slide(m):
            continue
        kind = m.get("type")
        trace = float(m.get("slide_trace_second", 0.0))
        end = float(m.get("end_second", 0.0))
        if end <= trace + EPS:
            continue
        if kind == "slide":
            segments: list[tuple[float, float, tuple[tuple[float, float], ...]]] = []
            for key, begin, dur in zip(
                m.get("slide_segment_keys", ()),
                m.get("slide_segment_shoot_seconds", ()),
                m.get("slide_segment_durations", ()),
            ):
                entry = slides_ref.get(key)
                if not entry:
                    continue
                raw_pts = entry.get("real_path_samples") or entry.get("samples") or ()
                pts = tuple((float(p["x"]), float(p["y"])) for p in raw_pts)
                if pts:
                    segments.append((float(begin), float(begin) + float(dur), pts))
            if segments:
                tracks.append(SlideSweepTrack(m, trace, end, False, segments))
        elif kind == "wifi":
            entry = wifi_ref.get(m.get("slide_track_key"))
            if not entry:
                continue
            lanes = entry.get("di_real_path_samples") or entry.get("lane_samples") or ()
            # Center lane represents the hand center; outer lanes are covered by WIFI_SWEEP_RADIUS.
            mid_idx = len(lanes) // 2 if lanes else -1
            if mid_idx >= 0 and lanes[mid_idx]:
                pts = tuple((float(p["x"]), float(p["y"])) for p in lanes[mid_idx])
                tracks.append(SlideSweepTrack(m, trace, end, True, [(trace, end, pts)]))
    return tracks


def _moving_touch_check_times(sustains, tracks):
    """Check native ticks where moving tracks could make three demands diverge."""
    changes = defaultdict(lambda: [0, 0, 0])

    def add_interval(start, end, kind):
        first, stop = start_tick(start), end_tick(end) + 1
        if first < stop:
            changes[first][kind] += 1
            changes[stop][kind] -= 1

    for sustain in sustains:
        kind = 0 if sustain["kind"] == "touch_hold" else 1
        add_interval(sustain["start"], sustain["lock_end"], kind)
    for track in tracks:
        add_interval(track.trace_start, track.end_sec - TAIL_RELEASE_GRACE_SEC, 2)

    touch_holds = holds = slides = 0
    previous = None
    for tick, (touch_delta, hold_delta, slide_delta) in sorted(changes.items()):
        if touch_holds and slides and touch_holds + holds + slides >= 3:
            for sample in range(previous, tick):
                yield round(sample / TPS * 1e6)
        touch_holds += touch_delta
        holds += hold_delta
        slides += slide_delta
        previous = tick


def _active_at(items, times, start, end):
    ordered = sorted(items, key=start)
    active = []
    index = 0
    for second in times:
        while index < len(ordered) and start(ordered[index]) <= second:
            active.append(ordered[index])
            index += 1
        active = [item for item in active if end(item) >= second]
        yield active


def detect_touch_reachability_conflicts(
    markers: list[dict[str, Any]],
    *,
    policy: str = "palm_span_v1",
    cohold_radius: float = DEFAULT_COHOLD_RADIUS,
) -> list[dict[str, Any]]:
    """Return MiaCode-shaped ``TouchMultiAction`` Muri diagnostics on ``markers``."""
    policy, user_radius = validate_touch_reachability(policy, cohold_radius)
    if policy == "off" or not any(m.get("type") in {"touch", "touch_hold"} for m in markers):
        return []
    radius = effective_cohold_radius(policy, user_radius)

    slide_tracks = build_slide_tracks(markers)
    sustains: list[dict[str, Any]] = []
    pulses: list[dict[str, Any]] = []
    touches_by_cell: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    emitted_heads: set[Any] = set()

    for m in markers:
        if m.get("is_mine") or m.get("track_mine"):
            continue
        kind = m.get("type")
        sec = float(m.get("second", 0.0))
        if kind in {"hold", "touch_hold"}:
            p = marker_pad(m)
            if p not in CENTERS:
                continue
            end_sec = max(sec, float(m.get("end_second", sec)))
            dur = end_sec - sec
            if dur > MIN_SUSTAIN_LOCK_SEC:
                lock_end = max(sec + SIMULTANEOUS_PULSE_SEC, end_sec - TAIL_RELEASE_GRACE_SEC)
                sustains.append(
                    {"marker": m, "kind": kind, "pad": p, "xy": CENTERS[p], "start": sec, "lock_end": lock_end}
                )
            elif kind == "touch_hold":
                touches_by_cell[(int(m.get("each_group_id", -1)), round(sec * 1e6))].append(
                    {"marker": m, "pad": p, "xy": CENTERS[p], "second": sec, "on_slide": bool(m.get("on_slide"))}
                )
            else:
                pulses.append({"marker": m, "kind": "tap", "pad": p, "xy": CENTERS[p], "second": sec})
        elif kind == "tap":
            if m.get("slide_head"):
                continue
            p = marker_pad(m)
            if p in CENTERS:
                pulses.append({"marker": m, "kind": "tap", "pad": p, "xy": CENTERS[p], "second": sec})
        elif kind in {"slide", "wifi"}:
            if has_head(m):
                ek = head_emit_key(m)
                if ek not in emitted_heads:
                    emitted_heads.add(ek)
                    p = f"A{m['lane']}"
                    if p in CENTERS and not m.get("slide_head"):
                        pulses.append(
                            {
                                "marker": m,
                                "kind": "slide_head",
                                "pad": p,
                                "xy": CENTERS[p],
                                "second": sec,
                                "head_key": head_key(m),
                            }
                        )
        elif kind == "touch":
            p = marker_pad(m)
            if p in CENTERS:
                touches_by_cell[(int(m.get("each_group_id", -1)), round(sec * 1e6))].append(
                    {"marker": m, "pad": p, "xy": CENTERS[p], "second": sec, "on_slide": bool(m.get("on_slide"))}
                )

    # Group simultaneous Touch sensors using maimai DX's connected touch_groups.
    touch_clusters: list[dict[str, Any]] = []
    for (_, _), cell_touches in sorted(touches_by_cell.items(), key=lambda kv: kv[0][1]):
        pads = [item["pad"] for item in cell_touches]
        for idxs in touch_groups(pads):
            members = [cell_touches[i] for i in idxs]
            touch_clusters.append(
                {
                    "second": members[0]["second"],
                    "members": members,
                    "all_on_slide": all(item["on_slide"] for item in members),
                }
            )

    # Candidate check instants: every Touch cluster onset, plus every non-Touch
    # onset while at least one TouchHold is actively held.
    touch_holds = [s for s in sustains if s["kind"] == "touch_hold"]
    check_times: set[int] = {round(tc["second"] * 1e6) for tc in touch_clusters}
    if touch_holds:
        for s in sustains:
            if any(th["start"] - EPS <= s["start"] <= th["lock_end"] + EPS for th in touch_holds):
                check_times.add(round(s["start"] * 1e6))
        for p in pulses:
            if any(th["start"] - EPS <= p["second"] <= th["lock_end"] + EPS for th in touch_holds):
                check_times.add(round(p["second"] * 1e6))
        for tr in slide_tracks:
            if any(th["start"] - EPS <= tr.trace_start <= th["lock_end"] + EPS for th in touch_holds):
                check_times.add(round(tr.trace_start * 1e6))
        if slide_tracks:
            check_times.update(_moving_touch_check_times(sustains, slide_tracks))

    if not check_times:
        return []

    diagnostics: list[dict[str, Any]] = []
    seen_signatures: set[tuple[str, ...]] = set()

    times = [stamp / 1e6 for stamp in sorted(check_times)]
    sustain_states = _active_at(sustains, times,
        lambda s: s["start"] - SIMULTANEOUS_PULSE_SEC, lambda s: s["lock_end"] + EPS)
    pulse_states = _active_at(pulses, times,
        lambda p: p["second"] - SIMULTANEOUS_PULSE_SEC, lambda p: p["second"] + SIMULTANEOUS_PULSE_SEC)
    touch_states = _active_at(touch_clusters, times,
        lambda tc: tc["second"] - SIMULTANEOUS_PULSE_SEC, lambda tc: tc["second"] + SIMULTANEOUS_PULSE_SEC)
    sweep_states = _active_at(slide_tracks, times,
        lambda tr: tr.trace_start - SLIDE_SWEEP_EARLY_SEC, lambda tr: tr.end_sec + SLIDE_SWEEP_LATE_SEC)
    for t, active_sustains, active_pulses, active_touch_clusters, nearby_tracks in zip(
        times, sustain_states, pulse_states, touch_states, sweep_states
    ):
        active_slides: list[tuple[SlideSweepTrack, tuple[float, float]]] = []
        for tr in nearby_tracks:
            pos = tr.sample_at(t)
            if pos is not None:
                # Merge twin/parallel slides that a single hand sweeps together.
                if any(math.dist(pos, prev_pos) <= PARALLEL_SLIDE_MERGE_DIST for _, prev_pos in active_slides):
                    continue
                active_slides.append((tr, pos))

        has_non_touch = bool(active_sustains or active_pulses or active_slides)
        if not has_non_touch:
            # Open-hand pure Touch fireworks can be wiped across the screen within +-150 ms.
            continue

        has_touch_involvement = bool(active_touch_clusters) or any(
            s["kind"] == "touch_hold" for s in active_sustains
        )
        if not has_touch_involvement:
            continue

        point_demands: list[tuple[float, float]] = []
        contributors: list[dict[str, Any]] = []

        for s in active_sustains:
            point_demands.append(s["xy"])
            contributors.append(s["marker"])
        for p in active_pulses:
            # Deduplicate exact same pad at exact same instant (e.g. hold start + pulse).
            if any(math.dist(p["xy"], xy) <= 1.0 for xy in point_demands):
                continue
            point_demands.append(p["xy"])
            contributors.append(p["marker"])
        for tr, pos in active_slides:
            point_demands.append(pos)
            contributors.append(tr.marker)

        group_demands: list[tuple[list[tuple[float, float]], int]] = []
        unexempt_touch_markers: list[dict[str, Any]] = []
        for tc in active_touch_clusters:
            if tc["all_on_slide"]:
                continue
            members = tc["members"]
            need = max(1, math.ceil(0.51 * len(members) - 1e-9))
            unswept_xys = [
                item["xy"] for item in members
                if not (item["on_slide"] or any(tr.sweeps_pad(item["xy"], t) for tr in nearby_tracks))
            ]
            swept = len(members) - len(unswept_xys)
            if swept >= need:
                continue
            rem_needed = need - swept
            if len(unswept_xys) == 1 and rem_needed == 1:
                point_demands.append(unswept_xys[0])
            else:
                group_demands.append((unswept_xys, rem_needed))
            unexempt_touch_markers.extend(item["marker"] for item in members)

        if not unexempt_touch_markers and not any(s["kind"] == "touch_hold" for s in active_sustains):
            continue

        if can_two_hands_cover(point_demands, group_demands, radius):
            continue

        # Prefer attributing the repair target to the instantaneous Touch or latest onset.
        victim_candidates = (
            unexempt_touch_markers
            + [p["marker"] for p in active_pulses]
            + [s["marker"] for s in active_sustains if abs(s["start"] - t) <= SIMULTANEOUS_PULSE_SEC]
        )
        all_involved = contributors + unexempt_touch_markers
        if not victim_candidates:
            victim_candidates = all_involved
        victim = max(
            victim_candidates,
            key=lambda m: (
                m.get("type") == "touch",
                not bool(m.get("is_ex")),
                float(m.get("second", 0.0)),
                int(m.get("parse_order", -1)),
            ),
        )
        unique_keys: dict[str, dict[str, Any]] = {}
        for m in all_involved:
            unique_keys.setdefault(str(m.get("key", "")), m)
        signature = tuple(sorted(unique_keys))
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        cause_marker = next((m for k, m in unique_keys.items() if k != victim.get("key")), None)
        item = diagnostic(
            "TouchMultiAction",
            "Muri",
            victim,
            t,
            cause=cause_marker,
            detail=f"双手已被占用且超出同手可达半径（{radius:.0f}），无法同时触达：" + ", ".join(signature),
        )
        item["sources"] = [reference_note(m) for m in unique_keys.values()]
        diagnostics.append(item)

    return diagnostics
